package needle

import (
	"errors"
	"testing"

	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// extractSchema is the tool of the extraction scenarios.
const extractSchema = `{"name":"event","parameters":{"type":"object","properties":{"title":{"type":"string"},` +
	`"day":{"type":"string","format":"date"},"seats":{"type":"integer"}}}}`

// scriptedExtract runs upstream's extract on a one-tool agent over schema
// whose answer is e.
func scriptedExtract(t *testing.T, schema, system, text string, e *toolcall.Envelope, o ExtractOptions) (string, error) {
	t.Helper()

	a, _ := scripted(t, "["+schema+"]", AgentOptions{System: system}, e)

	tool, err := extractTool([]byte(schema))
	if err != nil {
		t.Fatal(err)
	}

	args, err := a.extract(text, tool, o)

	return string(args), err
}

// TestExtractMatchesPython pins _validate_extraction: the message is
// upstream's ExtractionValidationError's for the same text, arguments,
// validation and system facts ("" where it raises none).
func TestExtractMatchesPython(t *testing.T) {
	date := "date: 2026-10-15 Thu 12:00"

	for _, c := range []struct {
		text, args, system string
		ungrounded         []string
		negation           bool
		want               string
	}{
		{text: "dinner on 5 june 2024 for 4", args: `{"title":"dinner","day":"2024-06-05","seats":4}`},
		{
			text: "dinner on 5 june 2024 for 4", args: `{"title":"dinner","day":"2025-06-05","seats":4}`,
			want: "day",
		},
		{
			text: "dinner tomorrow for four", args: `{"title":"Dinner","day":"2026-10-16","seats":4}`, system: date,
			ungrounded: []string{"event.seats", "event.day"}, want: "day, seats",
		},
		{
			text: "dinner for 4", args: `{"title":"dinner","seats":4}`,
			ungrounded: []string{"event.seats", "event.title"}, negation: true, want: "negated request, title",
		},
		{
			text: "dinner on 5 june 2024", args: `{"title":"dinner","day":"2024-06-05"}`,
			ungrounded: []string{"event.day"},
		},
		{text: "dinner tomorrow in 2024", args: `{"title":"dinner","day":"2026-06-05"}`, system: date},
	} {
		e := answer(one("event", c.args), c.ungrounded...)
		e.Validation.Negation = c.negation

		got, err := scriptedExtract(t, extractSchema, c.system, c.text, e, ExtractOptions{})
		if got != c.args {
			t.Errorf("%q: arguments %s, want %s", c.text, got, c.args)
		}

		var ve *ExtractionValidationError

		switch {
		case c.want == "" && err != nil:
			t.Errorf("%q: %v", c.text, err)
		case c.want != "" && !errors.As(err, &ve):
			t.Errorf("%q: %v, want an ExtractionValidationError", c.text, err)
		case c.want != "" && err.Error() != "needle: extraction returned values not grounded in the input: "+c.want:
			t.Errorf("%q: %v, want the fields %s", c.text, err, c.want)
		}

		got, err = scriptedExtract(t, extractSchema, c.system, c.text, e, ExtractOptions{Lenient: true})
		if got != c.args || err != nil {
			t.Errorf("%q, lenient: %s, %v", c.text, got, err)
		}
	}
}

// TestExtractCalls: the first call's arguments, taken from the suppressed
// calls when none was dispatched; nil without any call; an empty object for
// a call without arguments.
func TestExtractCalls(t *testing.T) {
	suppressed := &toolcall.Envelope{
		Type: "call", Success: true, Reasoning: new("r"), Confidence: 0.05,
		SuppressedCalls: []toolcall.Call{call("event", `{"title":"a"}`), call("event", `{"title":"b"}`)},
		Validation:      &toolcall.Validation{},
	}

	for _, c := range []struct {
		e    *toolcall.Envelope
		want string
	}{
		{answer([]toolcall.Call{call("event", `{"title":"x"}`), call("event", `{"title":"y"}`)}), `{"title":"x"}`},
		{suppressed, `{"title":"a"}`},
		{respond(), ""},
		{answer(nil), ""},
		{answer(one("event", `null`)), `{}`},
	} {
		got, err := scriptedExtract(t, extractSchema, "", "x", c.e, ExtractOptions{})
		if got != c.want || err != nil {
			t.Errorf("%+v: %s, %v; want %s", c.e, got, err, c.want)
		}
	}

	_, err := extractTool([]byte(`[` + extractSchema + `]`))
	if err == nil {
		t.Error("extractTool accepted a list")
	}
}

// TestExtractLaterCalls: upstream's _complete adds the year failures of
// every dispatched call to the validation, so the date of a later call
// fails the extraction of the first; suppressed calls are not annotated.
// The results are upstream's for the same answers.
func TestExtractLaterCalls(t *testing.T) {
	calls := []toolcall.Call{call("event", `{"title":"dinner"}`), call("event", `{"day":"2023-06-05"}`)}
	text := "dinner on 5 june 2024"

	_, err := scriptedExtract(t, extractSchema, "", text, answer(calls), ExtractOptions{})

	var ve *ExtractionValidationError
	if !errors.As(err, &ve) || len(ve.Fields) != 1 || ve.Fields[0] != "day" {
		t.Errorf("dispatched: %v, want the field day", err)
	}

	suppressed := &toolcall.Envelope{
		Type: "call", Success: true, Reasoning: new("r"), Confidence: 0.05, SuppressedCalls: calls,
		Validation: &toolcall.Validation{},
	}

	args, err := scriptedExtract(t, extractSchema, "", text, suppressed, ExtractOptions{})
	if err != nil || args != `{"title":"dinner"}` {
		t.Errorf("suppressed: %s, %v", args, err)
	}
}

// TestExtractCallerNames: the fields the engine flags under the model's
// names are checked under the caller's (docs/FINDINGS.md F-064): a number
// the text writes passes, the other flagged field fails under its own name.
func TestExtractCallerNames(t *testing.T) {
	schema := `{"name":"addEvent","parameters":{"type":"object","properties":{"eventTitle":{"type":"string"},` +
		`"seatCount":{"type":"integer"}}}}`
	e := answer(one("addEvent", `{"eventTitle":"Dinner","seatCount":4}`), "add_event.seat_count",
		"add_event.event_title")

	_, err := scriptedExtract(t, schema, "", "dinner for 4", e, ExtractOptions{})

	var ve *ExtractionValidationError
	if !errors.As(err, &ve) || len(ve.Fields) != 1 || ve.Fields[0] != "eventTitle" {
		t.Errorf("%v, want the field eventTitle alone", err)
	}
}
