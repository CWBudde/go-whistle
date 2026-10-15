package needle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// DefaultMaxSteps is Run's number of tool rounds when RunOptions leaves it
// at 0: upstream's max_steps.
const DefaultMaxSteps = 8

// ToolFunc runs one call of Run: args are the call's arguments, a JSON
// object under the caller's names. The value it returns goes back to the
// model as the call's result, serialised by encoding/json and written as
// Python's json.dumps writes it. Where Go's encoding differs from a Python
// value's, the model reads Go's: a float64 of 1 is 1, not 1.0, a map's keys
// come sorted, not in insertion order, and NaN and the infinities, which
// encoding/json refuses, come as their text; a struct, a json.Number or a
// json.RawMessage keeps the order and the literals. An error goes back as
// {"error": err.Error()}; a panic is not recovered.
type ToolFunc func(args json.RawMessage) (any, error)

// RunOptions configure Run.
type RunOptions struct {
	// MaxSteps bounds the rounds of tool calls; 0 is DefaultMaxSteps.
	MaxSteps int
	// MaxNewTokens is each answer's token limit; 0 is
	// DefaultMaxNewTokens.
	MaxNewTokens int
	// Lenient runs a call even when a field of it is ungrounded
	// (upstream's strict=False). By default such a call does not run and
	// its result is {"error": "ungrounded field, …"}.
	Lenient bool
}

// Run answers query and runs the calls of each answer with tools, feeding
// their results back as one tool-result turn, until an answer makes no
// call or MaxSteps rounds have run: upstream's Needle.run
// (docs/FINDINGS.md F-064). A call to a tool that tools does not have gets
// {"error": "unknown tool: name"}. Unless Lenient, a call with a field the
// engine's validation lists as ungrounded, or with a date whose year the
// query does not license (the Python client's year check, on the first
// answer only), gets {"error": "ungrounded field, …"} instead of running,
// but for fields that are numbers the query or the system facts write. The
// returned Response is the last answer's, with Results holding every
// result in call order.
func (a *Agent) Run(query string, tools map[string]ToolFunc, o RunOptions) (*Response, error) {
	if a.stateless {
		a.Reset()
	}

	steps := o.MaxSteps
	if steps <= 0 {
		steps = DefaultMaxSteps
	}

	maxNew := maxNewTokens(o.MaxNewTokens)

	r, err := a.complete(query, maxNew)
	if err != nil {
		return nil, err
	}

	years := a.yearFailures(r, query)
	results := []any{}

	for range steps {
		if r.Type != "call" || len(r.FunctionCalls) == 0 {
			break
		}

		ungrounded := a.ungroundedPaths(r)
		for tool, paths := range years {
			ungrounded[tool] = append(ungrounded[tool], paths...)
		}

		years = nil

		step := make([]any, len(r.FunctionCalls))
		for i, c := range r.FunctionCalls {
			step[i] = a.call(c, ungrounded[c.Name], query, tools, !o.Lenient)
		}

		results = append(results, step...)

		in, err := dumps(step)
		if err != nil {
			return nil, err
		}

		r, err = a.complete(in, maxNew)
		if err != nil {
			return nil, err
		}
	}

	r.Results = results

	return r, nil
}

// call runs one call with tools: the error of the strict check for the
// fields ungrounded lists (but its numbers that query or the system facts
// write), of an unknown tool or of the tool, else what the tool returned.
func (a *Agent) call(c Call, ungrounded []string, query string, tools map[string]ToolFunc, strict bool) any {
	fabricated := slices.Compact(slices.Sorted(slices.Values(ungrounded)))

	if strict && len(fabricated) > 0 {
		grounded := groundedNumberPaths(decodeJSON(c.Arguments), query, a.system)
		fabricated = slices.DeleteFunc(fabricated, func(p string) bool { return grounded[p] })
	}

	if strict && len(fabricated) > 0 {
		return errorResult("ungrounded " + strings.Join(fabricated, ", "))
	}

	fn := tools[c.Name]
	if fn == nil {
		return errorResult("unknown tool: " + c.Name)
	}

	args := c.Arguments
	if len(args) == 0 || string(args) == jsonNull {
		args = json.RawMessage("{}")
	}

	v, err := fn(args)
	if err != nil {
		return errorResult(err.Error())
	}

	return v
}

// errorResult is a call's result for an error.
func errorResult(msg string) map[string]any { return map[string]any{"error": msg} }

// ungroundedPaths is _ungrounded_paths: the fields the engine's validation
// lists, by tool. The engine names them under the model's snake_case names
// and upstream looks them up under the caller's, which misses every tool
// and field the engine renamed; the port maps them to the caller's names
// (docs/FINDINGS.md F-064).
func (a *Agent) ungroundedPaths(r *Response) map[string][]string {
	out := map[string][]string{}

	if r.Validation == nil || a.n == nil {
		return out
	}

	for _, name := range r.Validation.Ungrounded {
		tool, field, _ := strings.Cut(name, ".")
		caller := a.n.CallerName(tool)

		path := caller
		if field != "" {
			path = a.n.CallerField(tool, field)
		}

		out[caller] = append(out[caller], path)
	}

	return out
}

// yearFailures is the Python client's _annotate_ungrounded on the answer r
// to text: the date fields of r's calls whose year the years written since
// Reset do not license (licensedYears), by tool. Upstream adds them to the
// validation of Run's first answer, which its strict check reads; the port
// gives them to that check alone and leaves the envelope the engine's.
func (a *Agent) yearFailures(r *Response, text string) map[string][]string {
	years := licensedYears(a.years, a.system, relativeCue(text))
	if len(r.FunctionCalls) == 0 || len(years) == 0 {
		return nil
	}

	out := map[string][]string{}

	for _, c := range r.FunctionCalls {
		schema, ok := a.schemas[c.Name]
		if !ok {
			continue
		}

		_, failures := walkGrounding(schema, decodeJSON(c.Arguments), years)
		out[c.Name] = append(out[c.Name], slices.Sorted(maps.Keys(failures))...)
	}

	return out
}

// dumps is a step's results as upstream feeds them back,
// json.dumps(results, default=_jsonable).
func dumps(results []any) (string, error) {
	parts := make([][]byte, len(results))
	for i, v := range results {
		parts[i] = resultJSON(v)
	}

	s, err := toolcall.Dumps(slices.Concat([]byte("["), bytes.Join(parts, []byte(",")), []byte("]")))
	if err != nil {
		return "", wrap(err)
	}

	return s, nil
}

// resultJSON is a tool's result as JSON: what encoding/json writes, or for
// a value it cannot write the value's text, as _jsonable's str(value).
func resultJSON(v any) []byte {
	js, err := json.Marshal(v)
	if err == nil {
		return js
	}

	js, err = json.Marshal(fmt.Sprint(v))
	if err != nil {
		return []byte(jsonNull) // a string always marshals
	}

	return js
}
