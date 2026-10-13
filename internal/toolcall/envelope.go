package toolcall

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Envelope is the engine's answer to a turn (docs/FINDINGS.md F-054); it
// marshals with the engine's field order and its confidence to four
// decimals.
type Envelope struct {
	// Type is "call" (an empty call is a refusal) or "respond".
	Type      string
	Success   bool
	Error     *string
	ErrorCode *string
	Reason    *string
	// FunctionCalls and SuppressedCalls carry the caller's names.
	FunctionCalls   []Call
	SuppressedCalls []Call
	Reasoning       *string
	Confidence      float64
	// Validation is set when the answer has calls, suppressed ones included.
	Validation *Validation

	noTools bool // the engine's answer without tools leaves out error_code and reason
}

// Call is one function call.
type Call struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Validation is the engine's check of the calls against the request.
type Validation struct {
	// Ungrounded are "tool.field" (the model's names) whose value the
	// request does not contain.
	Ungrounded []string `json:"ungrounded"`
	// Negation reports a negated request.
	Negation bool `json:"negation"`
}

// SuppressBelow is the confidence below which the engine withholds the
// calls in suppressed_calls (llms.txt).
const SuppressBelow = 0.1

// The engine's texts as its answers write them (testdata/text/engine:
// responses.json, triggers.json).
const (
	truncatedError  = "tool call truncated: token budget exhausted"
	noToolsResponse = "no tools declared, returning the empty call"
)

// NoTools is the engine's answer to any request over an empty tool list; it
// decodes nothing.
func NoTools() *Envelope {
	return &Envelope{Type: "call", Success: true, Reasoning: new(noToolsResponse), Confidence: 1, noTools: true}
}

// Envelope is the engine's answer built from a decoded answer a to the
// request input over the tools n, with the system facts system (whose
// date: fact the repair resolves dates against) and the confidence conf
// (Decoder.Confidence) (docs/FINDINGS.md F-054, F-056):
//
//   - a truncated answer fails with error_code "truncated" and no reasoning;
//   - the reasoning is the text between "<think>\n" and "\n</think>";
//   - an answer with a <tool_call>[…]</tool_call> block is a "call" (an
//     empty list is a refusal), any other a "respond" without calls;
//   - the calls are repaired as the engine does after decoding (repair):
//     values grounded in the request, dates, polarity, capitalisation, and
//     dropped arguments and calls; an answer whose calls are all dropped is
//     a refusal;
//   - the calls are suppressed below SuppressBelow, unless a trigger of the
//     tools matches input and the request is not negated;
//   - validation, when there are calls: the fields whose value input does
//     not contain (strings case-insensitively, numbers as written; booleans,
//     enum and const values and dates exempt) and whether input negates.
func (n *Normalised) Envelope(a *Decoded, input, system string, conf float64) (*Envelope, error) {
	e := &Envelope{Type: "call", Success: true, Confidence: conf}

	if a.Truncated {
		e.Success = false
		e.Error, e.ErrorCode, e.Reason = new(truncatedError), new("truncated"), new("runtime_failure")

		return e, nil
	}

	think, rest := splitThink(a.Text)
	e.Reasoning = &think

	block, ok := callBlock(rest)
	if !ok {
		e.Type = "respond"

		return e, nil
	}

	calls, err := parseBlock(block)
	if err != nil {
		return nil, err
	}

	// The engine lists the ungrounded fields of the repaired calls
	// (docs/FINDINGS.md F-056: pizza_to_lights).
	calls = n.repair(calls, input, system)
	if len(calls) == 0 {
		return e, nil
	}

	out := make([]Call, len(calls))

	for i, c := range calls {
		args, err := n.CallerArgs(c.name, c.args)
		if err != nil {
			return nil, err
		}

		out[i] = Call{Name: n.CallerName(c.name), Arguments: args}
	}

	negated := negation.MatchString(input)
	if conf >= SuppressBelow || len(n.Triggered(input)) > 0 && !negated {
		e.FunctionCalls = out
	} else {
		e.SuppressedCalls = out
	}

	e.Validation = &Validation{Ungrounded: n.ungrounded(calls, input), Negation: negated}

	return e, nil
}

// negation finds a negated request (docs/FINDINGS.md F-054: "don't" in the
// corpus; "do not" and "never" are the port's).
var negation = regexp.MustCompile(`(?i)\b(?:don['’]?t|do not|never)\b`)

// splitThink is the reasoning between "<think>\n" and "\n</think>" and what
// follows it.
func splitThink(text string) (think, rest string) {
	text = strings.TrimPrefix(text, thinkOpen)

	think, rest, ok := strings.Cut(text, "\n</think>")
	if !ok {
		return text, ""
	}

	return think, strings.TrimLeft(rest, "\n")
}

// callBlock is the JSON inside a leading <tool_call>…</tool_call>.
func callBlock(s string) (string, bool) {
	body, ok := strings.CutPrefix(s, "<tool_call>")
	if !ok {
		return "", false
	}

	body, _, ok = strings.Cut(body, toolCallEnd)

	return body, ok
}

// modelCall is a call as the model wrote it.
type modelCall struct {
	name string
	args []byte
}

// parseBlock reads a call block, a JSON list of {"name", "arguments"}.
func parseBlock(block string) ([]modelCall, error) {
	var raw []struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}

	err := json.Unmarshal([]byte(block), &raw)
	if err != nil {
		return nil, fmt.Errorf("toolcall: call block: %w", err)
	}

	calls := make([]modelCall, len(raw))
	for i, r := range raw {
		if len(r.Arguments) == 0 {
			r.Arguments = json.RawMessage("{}")
		}

		calls[i] = modelCall{name: r.Name, args: r.Arguments}
	}

	return calls, nil
}

// ungrounded are the "tool.field" paths of calls whose value input does not
// contain, in the calls' and the arguments' order, each once.
func (n *Normalised) ungrounded(calls []modelCall, input string) []string {
	out := []string{}
	seen := map[string]bool{}
	lower := strings.ToLower(input)

	var walk func(path string, v any, schemas []*object)

	walk = func(path string, v any, schemas []*object) {
		switch t := v.(type) {
		case *object:
			for _, k := range t.keys {
				walk(path+"."+k, t.vals[k], under(schemas, "properties", k))
			}
		case []any:
			items := under(schemas, "items", "")
			for _, x := range t {
				walk(path, x, items)
			}
		default:
			if !grounded(t, schemas, lower) && !seen[path] {
				seen[path] = true
				out = append(out, path)
			}
		}
	}

	for _, c := range calls {
		p := &pyParser{s: c.args}

		v, err := p.value()
		if err != nil {
			continue
		}

		var params []*object
		if ps := n.params[c.name]; ps != nil {
			params = composed(ps)
		}

		walk(c.name, v, params)
	}

	return out
}

// member is o's key as an object, nil when o is nil or the key is not one.
func member(o *object, key string) *object {
	if o == nil {
		return nil
	}

	m, _ := o.vals[key].(*object)

	return m
}

// composed is s and, recursively, the branches of its allOf, anyOf and
// oneOf lists: every schema that may describe a value of s (a port-side
// rule, docs/FINDINGS.md F-054).
func composed(s *object) []*object {
	out := []*object{s}

	for _, key := range compositions {
		list, _ := s.vals[key].([]any)
		for _, b := range list {
			if o, ok := b.(*object); ok {
				out = append(out, composed(o)...)
			}
		}
	}

	return out
}

// under are the composed schemas of key under schemas: of property k of
// their "properties", or of their "items" when k is "".
func under(schemas []*object, key, k string) []*object {
	var out []*object

	for _, s := range schemas {
		sub := member(s, key)
		if k != "" {
			sub = member(sub, k)
		}

		if sub != nil {
			out = append(out, composed(sub)...)
		}
	}

	return out
}

// isoDate is a calendar date as the model writes one.
var isoDate = regexp.MustCompile(`\d{4}-\d{2}-\d{2}`)

// grounded reports whether the request lowered contains the leaf value v of
// a field with the schemas schemas; an enum, const or date format in any of
// them exempts it.
func grounded(v any, schemas []*object, lowered string) bool {
	date := false

	for _, s := range schemas {
		_, enum := s.vals["enum"]
		_, constant := s.vals["const"]

		if enum || constant {
			return true
		}

		if f, _ := s.vals["format"].(string); f == "date" || f == "date-time" {
			date = true
		}
	}

	switch t := v.(type) {
	case string:
		if date || isoDate.MatchString(t) {
			return true
		}

		return strings.Contains(lowered, strings.ToLower(t))
	case json.Number:
		return containsNumber(lowered, string(t))
	default:
		return true
	}
}

// containsNumber reports whether s contains the number literal lit, or its
// shortest decimal form, not inside a longer run of digits.
func containsNumber(s, lit string) bool {
	forms := []string{lit}

	f, err := strconv.ParseFloat(lit, 64)
	if err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) {
		forms = append(forms, strconv.FormatFloat(f, 'f', -1, 64))
	}

	for _, form := range forms {
		for i := 0; ; {
			j := strings.Index(s[i:], form)
			if j < 0 {
				break
			}

			j += i
			before := j == 0 || !isDigit(s[j-1])
			after := j+len(form) == len(s) || !isDigit(s[j+len(form)])

			if before && after {
				return true
			}

			i = j + 1
		}
	}

	return false
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// MarshalJSON writes the envelope as the engine does.
func (e *Envelope) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer

	b.WriteByte('{')

	field := func(name string, v any) error {
		js, err := json.Marshal(v)
		if err != nil {
			return err
		}

		if b.Len() > 1 {
			b.WriteByte(',')
		}

		b.WriteString(`"` + name + `":`)
		b.Write(js)

		return nil
	}

	calls := func(c []Call) []Call {
		if c == nil {
			return []Call{}
		}

		return c
	}

	fields := []struct {
		name string
		v    any
	}{
		{"type", e.Type},
		{"success", e.Success},
		{"error", e.Error},
		{"error_code", e.ErrorCode},
		{"reason", e.Reason},
		{"function_calls", calls(e.FunctionCalls)},
		{"suppressed_calls", calls(e.SuppressedCalls)},
		{"reasoning", e.Reasoning},
	}

	for _, f := range fields {
		if e.noTools && (f.name == "error_code" || f.name == "reason") {
			continue
		}

		err := field(f.name, f.v)
		if err != nil {
			return nil, err
		}
	}

	b.WriteString(`,"confidence":` + strconv.FormatFloat(e.Confidence, 'f', 4, 64))

	if e.Validation != nil {
		err := field("validation", e.Validation)
		if err != nil {
			return nil, err
		}
	}

	b.WriteByte('}')

	return b.Bytes(), nil
}
