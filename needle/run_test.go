package needle

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// runTools are the tools of the Run scenarios.
const runTools = `[{"name":"set_timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer"},` +
	`"ms":{"type":"integer"},"label":{"type":"string"}}}},{"name":"nope","parameters":{}},` +
	`{"name":"lights","parameters":{"type":"object","properties":{"room":{"type":"string"}}}},` +
	`{"name":"set_alarm","parameters":{"type":"object","properties":{"time":{"type":"string"},` +
	`"label":{"type":"string"}}}},{"name":"t","parameters":{}},` +
	`{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}},{"name":"b","parameters":{}},` +
	`{"name":"book","parameters":{"type":"object","properties":{"day":{"type":"string","format":"date"},` +
	`"seats":{"type":"integer"}}}}]`

// TestRunMatchesPython runs upstream's Needle.run scenarios against a
// scripted engine. fed is what the loop feeds the engine, turn by turn,
// and results the json.dumps of the response's "results", both as
// upstream's run gives them for the same engine answers and tools.
func TestRunMatchesPython(t *testing.T) {
	timer := func(args json.RawMessage) (any, error) {
		var in struct {
			Minutes int    `json:"minutes"`
			Label   string `json:"label"`
		}

		err := json.Unmarshal(args, &in)

		return struct {
			OK      bool   `json:"ok"`
			Minutes int    `json:"minutes"`
			Label   string `json:"label"`
			Note    string `json:"note"`
		}{true, in.Minutes, in.Label, "Zeit läuft ✓"}, err
	}
	value := func(v any) ToolFunc { return func(json.RawMessage) (any, error) { return v, nil } }
	booked := value("booked")

	for _, c := range []struct {
		name, query, system string
		answers             []*toolcall.Envelope
		tools               map[string]ToolFunc
		o                   RunOptions
		fed                 []string
		results             string
	}{
		{name: "respond", query: "hi", answers: []*toolcall.Envelope{respond()}, fed: []string{"hi"}, results: `[]`},
		{
			name: "one_call", query: "set a timer for 10 minutes",
			answers: []*toolcall.Envelope{answer(one("set_timer", `{"minutes":10}`)), respond()},
			tools:   map[string]ToolFunc{"set_timer": timer},
			fed: []string{
				"set a timer for 10 minutes",
				`[{"ok": true, "minutes": 10, "label": "", "note": "Zeit l\u00e4uft \u2713"}]`,
			},
			results: `[{"ok": true, "minutes": 10, "label": "", "note": "Zeit l\u00e4uft \u2713"}]`,
		},
		{
			name: "unknown", query: "x", answers: []*toolcall.Envelope{answer(one("nope", `{}`)), respond()},
			fed: []string{"x", `[{"error": "unknown tool: nope"}]`}, results: `[{"error": "unknown tool: nope"}]`,
		},
		{
			name: "raises", query: "lights in hall",
			answers: []*toolcall.Envelope{answer(one("lights", `{"room":"hall"}`)), respond()},
			tools: map[string]ToolFunc{"lights": func(args json.RawMessage) (any, error) {
				var in struct{ Room string }

				_ = json.Unmarshal(args, &in)

				return nil, errors.New("no such room: " + in.Room)
			}},
			fed:     []string{"lights in hall", `[{"error": "no such room: hall"}]`},
			results: `[{"error": "no such room: hall"}]`,
		},
		{
			name: "ungrounded", query: "wake me up",
			answers: []*toolcall.Envelope{
				answer(one("set_alarm", `{"time":"07:00","label":"work"}`), "set_alarm.time", "set_alarm.label"),
				respond(),
			},
			tools:   map[string]ToolFunc{"set_alarm": value("set")},
			fed:     []string{"wake me up", `[{"error": "ungrounded label, time"}]`},
			results: `[{"error": "ungrounded label, time"}]`,
		},
		{
			name: "grounded_number", query: "a timer in 10 minutes, 1,500 ms",
			answers: []*toolcall.Envelope{
				answer(one("set_timer", `{"minutes":10,"ms":1500,"label":"x"}`), "set_timer.minutes", "set_timer.ms"),
				respond(),
			},
			tools: map[string]ToolFunc{"set_timer": value(json.Number("1.0"))},
			fed:   []string{"a timer in 10 minutes, 1,500 ms", `[1.0]`}, results: `[1.0]`,
		},
		{
			name: "lenient", query: "wake me up",
			answers: []*toolcall.Envelope{answer(one("set_alarm", `{"time":"07:00"}`), "set_alarm.time"), respond()},
			tools:   map[string]ToolFunc{"set_alarm": value(nil)}, o: RunOptions{Lenient: true},
			fed: []string{"wake me up", `[null]`}, results: `[null]`,
		},
		{
			name: "max_steps", query: "loop",
			answers: []*toolcall.Envelope{
				answer(one("t", `{}`)), answer(one("t", `{}`)), answer(one("t", `{}`)), answer(one("t", `{}`)),
			},
			tools: map[string]ToolFunc{"t": value([]any{1, "two", nil})}, o: RunOptions{MaxSteps: 3},
			fed:     []string{"loop", `[[1, "two", null]]`, `[[1, "two", null]]`, `[[1, "two", null]]`},
			results: `[[1, "two", null], [1, "two", null], [1, "two", null]]`,
		},
		{
			name: "two_calls", query: "both",
			answers: []*toolcall.Envelope{
				answer([]toolcall.Call{call("a", `{"x":1}`), call("b", `{}`)}), answer(one("a", `{"x":2}`)), respond(),
			},
			tools: map[string]ToolFunc{
				"a": func(args json.RawMessage) (any, error) { return args, nil }, "b": value("b"),
			},
			fed: []string{"both", `[{"x": 1}, "b"]`, `[{"x": 2}]`}, results: `[{"x": 1}, "b", {"x": 2}]`,
		},
		{
			name: "empty_call", query: "nothing", answers: []*toolcall.Envelope{answer(nil)},
			fed: []string{"nothing"}, results: `[]`,
		},
		{
			name: "year", query: "book 2 seats on 5 june 2024",
			answers: []*toolcall.Envelope{answer(one("book", `{"day":"2025-06-05","seats":2}`)), respond()},
			tools:   map[string]ToolFunc{"book": booked},
			fed:     []string{"book 2 seats on 5 june 2024", `[{"error": "ungrounded day"}]`},
			results: `[{"error": "ungrounded day"}]`,
		},
		{
			name: "year_relative", query: "book 2 seats tomorrow, the 5 june 2024", system: "date: 2026-10-15 Thu 12:00",
			answers: []*toolcall.Envelope{answer(one("book", `{"day":"2026-10-16","seats":2}`)), respond()},
			tools:   map[string]ToolFunc{"book": booked},
			fed:     []string{"book 2 seats tomorrow, the 5 june 2024", `["booked"]`}, results: `["booked"]`,
		},
		{
			name: "year_later_step", query: "book on 5 june 2024",
			answers: []*toolcall.Envelope{answer(one("t", `{}`)), answer(one("book", `{"day":"2025-06-05"}`)), respond()},
			tools:   map[string]ToolFunc{"t": value("ok"), "book": booked},
			fed:     []string{"book on 5 june 2024", `["ok"]`, `["booked"]`}, results: `["ok", "booked"]`,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			a, s := scripted(t, runTools, AgentOptions{System: c.system}, c.answers...)

			r, err := a.Run(c.query, c.tools, c.o)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(s.fed, c.fed) {
				t.Errorf("fed %q\nwant %q", s.fed, c.fed)
			}

			got, err := dumps(r.Results)
			if err != nil || got != c.results {
				t.Errorf("results %s (%v), want %s", got, err, c.results)
			}

			if len(s.answers) != 0 {
				t.Errorf("%d answers left", len(s.answers))
			}
		})
	}
}

// TestRunYearsAcrossTurns: the years a turn licenses are those written
// since Reset, earlier turns included, as upstream's _seen_years.
func TestRunYearsAcrossTurns(t *testing.T) {
	book := map[string]ToolFunc{"book": func(json.RawMessage) (any, error) { return "booked", nil }}
	booking := one("book", `{"day":"2025-06-05"}`)

	a, s := scripted(t, runTools, AgentOptions{}, respond(), answer(booking), respond(), answer(booking), respond())

	_, _ = a.Complete("we fly on june 5, 2025", CompleteOptions{})

	_, err := a.Run("book a table on 5 june 2024", book, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	a.Reset()

	_, err = a.Run("book a table on 5 june 2024", book, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	want := []string{
		"we fly on june 5, 2025", "book a table on 5 june 2024", `["booked"]`,
		"book a table on 5 june 2024", `[{"error": "ungrounded day"}]`,
	}
	if !slices.Equal(s.fed, want) {
		t.Errorf("fed %q\nwant %q", s.fed, want)
	}
}

// TestRunCallerNames: the engine's validation names a tool and its fields
// under the model's snake_case names; Run checks them under the caller's,
// where upstream's lookup by the caller's name misses them and would run
// the call (docs/FINDINGS.md F-064).
func TestRunCallerNames(t *testing.T) {
	tools := `[{"name":"setAlarm","parameters":{"type":"object","properties":{"wakeTime":{"type":"string"},` +
		`"minutesLeft":{"type":"integer"}}}}]`
	a, s := scripted(t, tools, AgentOptions{},
		answer(one("setAlarm", `{"wakeTime":"07:00","minutesLeft":5}`), "set_alarm.wake_time", "set_alarm.minutes_left"),
		respond())

	ran := false

	r, err := a.Run("wake me in 5 minutes", map[string]ToolFunc{"setAlarm": func(json.RawMessage) (any, error) {
		ran = true

		return "set", nil
	}}, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if ran || len(s.fed) != 2 || s.fed[1] != `[{"error": "ungrounded wakeTime"}]` {
		t.Errorf("ran %v, fed %q", ran, s.fed)
	}

	if r.Type != "respond" || len(r.Results) != 1 {
		t.Errorf("response %+v", r)
	}
}

// TestRunError: an engine error ends Run with it.
func TestRunError(t *testing.T) {
	a, _ := scripted(t, runTools, AgentOptions{}, answer(one("t", `{}`)))

	_, err := a.Run("loop", map[string]ToolFunc{"t": func(json.RawMessage) (any, error) { return 1, nil }}, RunOptions{})
	if err == nil {
		t.Error("Run without the engine's second answer succeeded")
	}
}

// TestCompleteAnnotates: Complete adds the Python client's year check to
// the envelope as _annotate_ungrounded does (needle/__init__.py 585): a
// date whose year the turns do not license goes into
// validation.ungrounded as "tool.path" after the engine's fields and not
// twice, a validation is made for it when the engine wrote none, and a
// licensed date adds nothing.
func TestCompleteAnnotates(t *testing.T) {
	const bad, good = `{"title":"dinner","day":"2019-03-05"}`, `{"title":"dinner","day":"2025-03-05"}`

	for _, c := range []struct {
		args         string
		engine       []string
		noValidation bool
		want         []string
	}{
		{bad, nil, false, []string{"event.day"}},
		{bad, nil, true, []string{"event.day"}},
		{bad, []string{"event.title"}, false, []string{"event.title", "event.day"}},
		{bad, []string{"event.day"}, false, []string{"event.day"}},
		{good, []string{"event.title"}, false, []string{"event.title"}},
	} {
		e := answer(one("event", c.args), c.engine...)
		if c.noValidation {
			e.Validation = nil
		}

		a, _ := scripted(t, "["+extractSchema+"]", AgentOptions{}, e)

		r, err := a.Complete("dinner on 5 March 2025", CompleteOptions{})
		if err != nil {
			t.Fatal(err)
		}

		if r.Validation == nil || !slices.Equal(r.Validation.Ungrounded, c.want) {
			t.Errorf("%s %v: validation %+v, want ungrounded %v", c.args, c.engine, r.Validation, c.want)
		}

		js, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}

		want, err := json.Marshal(c.want)
		if err != nil {
			t.Fatal(err)
		}

		if !strings.Contains(string(js), `"ungrounded":`+string(want)) {
			t.Errorf("%s %v: JSON %s, want ungrounded %s", c.args, c.engine, js, want)
		}
	}
}

// TestRunAnnotatesFirstAnswer: Run's first answer is annotated and its
// date blocks the call, a later answer is not (upstream's ground=False)
// and its call runs.
func TestRunAnnotatesFirstAnswer(t *testing.T) {
	const bad = `{"title":"dinner","day":"2019-03-05"}`

	a, _ := scripted(t, "["+extractSchema+"]", AgentOptions{},
		answer(one("event", bad)), answer(one("event", bad)), respond())

	ran := 0

	r, err := a.Run("dinner on 5 March 2025", map[string]ToolFunc{
		"event": func(json.RawMessage) (any, error) {
			ran++

			return map[string]any{"ok": true}, nil
		},
	}, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	js, err := json.Marshal(r.Results)
	if err != nil {
		t.Fatal(err)
	}

	if want := `[{"error":"ungrounded day"},{"ok":true}]`; string(js) != want || ran != 1 {
		t.Errorf("results %s after %d runs, want %s after 1", js, ran, want)
	}
}

// TestOpenAIFormChecked: a tool in OpenAI's form ({"type": "function",
// "function": {…}}) gets the year check as its plain form does, in
// Complete's validation and in Extract's, where upstream's checks read the
// wrapper and miss it (docs/FINDINGS.md F-064).
func TestOpenAIFormChecked(t *testing.T) {
	const bad = `{"title":"dinner","day":"2019-03-05"}`

	wrapped := `{"type":"function","function":` + extractSchema + `}`

	a, _ := scripted(t, "["+wrapped+"]", AgentOptions{}, answer(one("event", bad)))

	r, err := a.Complete("dinner on 5 March 2025", CompleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if r.Validation == nil || !slices.Equal(r.Validation.Ungrounded, []string{"event.day"}) {
		t.Errorf("Complete: validation %+v, want ungrounded [event.day]", r.Validation)
	}

	_, err = scriptedExtract(t, wrapped, "", "dinner on 5 March 2025", answer(one("event", bad)), ExtractOptions{})

	ve, ok := errors.AsType[*ExtractionValidationError](err)
	if !ok || !slices.Equal(ve.Fields, []string{"day"}) {
		t.Errorf("Extract: error %v, want an ExtractionValidationError for [day]", err)
	}
}
