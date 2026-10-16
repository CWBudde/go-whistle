package needle

import (
	"encoding/json"
	"errors"
	"maps"
	"slices"
	"strings"
	"time"
)

// ExtractOptions configure Extract.
type ExtractOptions struct {
	// System are the system facts. The date fact of Now (nil is
	// time.Now) always goes in front of them unless they hold a date or
	// are a JSON object, as upstream's extract builds its agent
	// (needle/__init__.py 639, auto_date=True).
	System string
	Now    func() time.Time
	// MaxNewTokens is the answer's token limit; 0 is DefaultMaxNewTokens.
	MaxNewTokens int
	// Lenient returns the arguments without the strict check (upstream's
	// strict=False).
	Lenient bool

	// noYears is the NoYearCheck of the agent whose Extract this is.
	noYears bool
}

// ExtractionValidationError is Extract's strict check failing
// (upstream's ExtractionValidationError): the fields whose values the
// input does not ground, and "negated request" for a negated one.
type ExtractionValidationError struct {
	Fields []string
}

func (e *ExtractionValidationError) Error() string {
	return "needle: extraction returned values not grounded in the input: " + strings.Join(e.Fields, ", ")
}

// Extract fills the tool schema (one tool, a JSON object as NewAgent takes
// it) from text, as upstream's needle.extract does: a new agent over that
// tool alone answers text, and Extract returns the arguments of its first
// call, a JSON object under the caller's names, taken from the suppressed
// calls when none was made (docs/FINDINGS.md F-064); nil when the answer
// has no call. Unless Lenient, the arguments must be grounded in text, else
// the error is an *ExtractionValidationError, returned with the arguments:
// a date whose year text does not license, a field the engine reports as
// ungrounded (but a number text or the system facts write, or a date whose
// year text licenses), or a negated request.
func (m *Model) Extract(text string, schema []byte, o ExtractOptions) (json.RawMessage, error) {
	tool, err := extractTool(schema)
	if err != nil {
		return nil, err
	}

	a, err := m.NewAgent(slices.Concat([]byte("["), schema, []byte("]")), o.agent())
	if err != nil {
		return nil, err
	}

	return a.extract(text, tool, o)
}

// agent are the options of Extract's agent: the system facts with the date
// fact, which upstream's extract always adds, and the NoYearCheck of the
// agent whose Extract this is.
func (o ExtractOptions) agent() AgentOptions {
	return AgentOptions{System: o.System, Now: o.Now, NoYearCheck: o.noYears}
}

// ExtractInto is Extract decoding the arguments into v with
// encoding/json. It reports whether there were any.
func (m *Model) ExtractInto(text string, schema []byte, v any, o ExtractOptions) (bool, error) {
	args, err := m.Extract(text, schema, o)
	if err != nil || args == nil {
		return false, err
	}

	err = json.Unmarshal(args, v)
	if err != nil {
		return false, wrap(err)
	}

	return true, nil
}

// Extract is Model.Extract with the agent's system facts, as upstream's
// Needle.extract: o.System is not used. The agent's NoYearCheck carries
// over. The conversation is not touched.
func (a *Agent) Extract(text string, schema []byte, o ExtractOptions) (json.RawMessage, error) {
	if a.m == nil {
		return nil, errors.New("needle: the agent has no model")
	}

	o.System, o.noYears = a.system, a.noYears

	return a.m.Extract(text, schema, o)
}

// extractTool is the schema Extract fills, as JSON values.
func extractTool(schema []byte) (any, error) {
	tool, ok := decodeJSON(schema).(map[string]any)
	if !ok {
		return nil, errors.New("needle: the extraction schema is not a JSON object")
	}

	return unwrapTool(tool), nil
}

// extract is upstream's extract on the one-tool agent a.
func (a *Agent) extract(text string, tool any, o ExtractOptions) (json.RawMessage, error) {
	r, err := a.complete(text, maxNewTokens(o.MaxNewTokens), true)
	if err != nil {
		return nil, err
	}

	calls := r.FunctionCalls
	if len(calls) == 0 {
		calls = r.SuppressedCalls
	}

	if len(calls) == 0 {
		return nil, nil
	}

	args := calls[0].Arguments
	if len(args) == 0 || string(args) == jsonNull {
		args = json.RawMessage("{}")
	}

	if !o.Lenient {
		err := a.validateExtraction(text, tool, args, r)
		if err != nil {
			return args, err
		}
	}

	return args, nil
}

// validateExtraction is _validate_extraction with the agent's system
// facts: the date fields whose year text does not license; the fields the
// engine reports as ungrounded, and the paths annotated adds, that are
// neither numbers text or the system facts write nor dates whose year text
// licenses; and a negated request. The validation holds the year failures
// that upstream's _complete adds for every dispatched call (annotate), so
// a later call's date counts too. The engine names its fields under the
// model's names, upstream compares them with the caller's; the port maps
// the engine's to the caller's (docs/FINDINGS.md F-064).
func (a *Agent) validateExtraction(text string, tool any, args json.RawMessage, r *Response) error {
	argv := decodeJSON(args)
	checked, failures := walkGrounding(tool, argv, licensedYears(sourceYears(text), a.system, relativeCue(text)))

	var flagged []string

	negation := false

	if r.Validation != nil {
		flagged, negation = r.Validation.Ungrounded[:r.engine], r.Validation.Negation
	}

	grounded := map[string]bool{}

	paths := make([]string, 0, len(r.annotated)+len(flagged))
	for _, g := range r.annotated {
		paths = append(paths, g.path)
	}

	for _, name := range flagged {
		path := name
		if tool, field, ok := strings.Cut(name, "."); ok && a.n != nil {
			path = a.n.CallerField(tool, field)
		}

		paths = append(paths, path)
	}

	if len(paths) > 0 {
		grounded = groundedNumberPaths(argv, text, a.system)
	}

	for _, path := range paths {
		if !grounded[path] && (!checked[path] || failures[path]) {
			failures[path] = true
		}
	}

	if negation {
		failures["negated request"] = true
	}

	if len(failures) > 0 {
		return &ExtractionValidationError{Fields: slices.Sorted(maps.Keys(failures))}
	}

	return nil
}
