package needle

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// script is a completer that answers each turn with the next envelope and
// records what it was fed: the engine without the model, for the tests of
// the Python client's logic.
type script struct {
	answers []*toolcall.Envelope
	fed     []string
	resets  int
}

func (s *script) complete(input string, _ int) (*toolcall.Envelope, error) {
	s.fed = append(s.fed, input)

	if len(s.answers) == 0 {
		return nil, errors.New("script: no answer left")
	}

	e := s.answers[0]
	s.answers = s.answers[1:]

	return e, nil
}

func (s *script) reset() { s.resets++ }

// scripted is an agent over tools whose turns script answers. Without a
// clock it renders no date fact: the scenarios' expectations are what
// upstream's functions return for the system facts as given.
func scripted(t *testing.T, tools string, o AgentOptions, answers ...*toolcall.Envelope) (*Agent, *script) {
	t.Helper()

	if o.Now == nil {
		o.NoAutoDate = true
	}

	a, err := newAgent([]byte(tools), o)
	if err != nil {
		t.Fatal(err)
	}

	s := &script{answers: answers}
	a.c = s

	return a, s
}

// answer is an engine answer with calls (none: a refusal) and the fields
// its validation lists.
func answer(calls []toolcall.Call, ungrounded ...string) *toolcall.Envelope {
	e := &toolcall.Envelope{Type: "call", Success: true, Reasoning: new("r"), Confidence: 0.9, FunctionCalls: calls}
	if len(calls) > 0 {
		e.Validation = &toolcall.Validation{Ungrounded: ungrounded}
	}

	return e
}

// respond is an engine answer without calls.
func respond() *toolcall.Envelope {
	return &toolcall.Envelope{Type: "respond", Success: true, Reasoning: new("r"), Confidence: 0.9}
}

func call(name, args string) toolcall.Call {
	return toolcall.Call{Name: name, Arguments: json.RawMessage(args)}
}

func one(name, args string) []toolcall.Call { return []toolcall.Call{call(name, args)} }

// TestResponseJSON: a Response marshals to the engine's envelope byte for
// byte (toolcall.Envelope's MarshalJSON), the answer without tools
// included, and Run's results follow it.
func TestResponseJSON(t *testing.T) {
	trunc, code, reason := "tool call truncated: token budget exhausted", "truncated", "runtime_failure"

	for _, e := range []*toolcall.Envelope{
		answer(one("setAlarm", `{"time":"06:45"}`), "set_alarm.time"),
		respond(),
		answer(nil),
		{Type: "call", Error: &trunc, ErrorCode: &code, Reason: &reason, Confidence: 0.25},
		{
			Type: "call", Success: true, Reasoning: new("r"), Confidence: 0.05,
			SuppressedCalls: one("x", `{}`), Validation: &toolcall.Validation{Ungrounded: []string{}, Negation: true},
		},
	} {
		want, err := e.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}

		got, err := json.Marshal(newResponse(e, false))
		if err != nil {
			t.Fatal(err)
		}

		if string(got) != string(want) {
			t.Errorf("got  %s\nwant %s", got, want)
		}
	}

	none, err := newAgent(nil, AgentOptions{})
	if err != nil {
		t.Fatal(err)
	}

	r, err := none.Complete("set an alarm", CompleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	want := `{"type":"call","success":true,"error":null,"function_calls":[],"suppressed_calls":[],` +
		`"reasoning":"no tools declared, returning the empty call","confidence":1.0000}`

	got, err := json.Marshal(r)
	if err != nil || string(got) != want {
		t.Errorf("no tools: got %s (%v), want %s", got, err, want)
	}

	r.Results = []any{map[string]any{"error": "unknown tool: x"}, "<ok>", nil, func() {}}
	want = want[:len(want)-1] + `,"results":[{"error":"unknown tool: x"},"\u003cok\u003e",null,"0x`

	got, err = json.Marshal(r)
	if err != nil || len(got) < len(want) || string(got[:len(want)]) != want {
		t.Errorf("results: got %s, want the prefix %s", got, want)
	}
}

// TestAgentNoTools: an empty tool list, given or not, answers every turn
// with the engine's empty call and decodes nothing.
func TestAgentNoTools(t *testing.T) {
	for _, tools := range []string{"", " ", "[]", " [ ] "} {
		a, err := newAgent([]byte(tools), AgentOptions{})
		if err != nil {
			t.Fatal(err)
		}

		if a.n != nil {
			t.Errorf("%q: tools %v", tools, a.n.JSON)
		}

		r, err := a.Run("hi", nil, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		if r.Type != "call" || len(r.FunctionCalls) != 0 || r.Results == nil || len(r.Results) != 0 {
			t.Errorf("%q: %+v", tools, r)
		}
	}

	_, err := newAgent([]byte(`{"name":"x"}`), AgentOptions{})
	if err == nil {
		t.Error("newAgent accepted tools that are not a list")
	}
}

// TestAgentSystem: by default the date fact of Now goes in front of the
// system facts as upstream's _with_date_fact does, unless they hold a date
// or are a JSON object; with NoAutoDate the facts stay as given.
func TestAgentSystem(t *testing.T) {
	now := func() time.Time { return time.Date(2026, 10, 15, 12, 3, 0, 0, time.UTC) }

	for _, c := range []struct {
		o    AgentOptions
		want string
	}{
		{AgentOptions{System: "home: Berlin", NoAutoDate: true}, "home: Berlin"},
		{AgentOptions{System: "home: Berlin", Now: now}, "date: 2026-10-15 Thu 12:03; home: Berlin"},
		{AgentOptions{Now: now}, "date: 2026-10-15 Thu 12:03"},
		{AgentOptions{System: "update: 2024-01-02", Now: now}, "update: 2024-01-02"},
		{AgentOptions{System: `{"home":"Berlin"}`, Now: now}, `{"home":"Berlin"}`},
	} {
		a, err := newAgent(nil, c.o)
		if err != nil {
			t.Fatal(err)
		}

		if a.System() != c.want {
			t.Errorf("%+v: system %q, want %q", c.o, a.System(), c.want)
		}
	}
}

// TestAgentReset: Reset starts the conversation and the years seen anew;
// Stateless does it before every Complete and Run, as upstream's
// stateless=True.
func TestAgentReset(t *testing.T) {
	a, s := scripted(t, runTools, AgentOptions{}, respond(), respond())

	_, _ = a.Complete("on 5 june 2024", CompleteOptions{})
	if !a.years[2024] {
		t.Errorf("years %v after a turn that writes 2024", a.years)
	}

	a.Reset()

	if s.resets != 1 || len(a.years) != 0 {
		t.Errorf("Reset: %d resets, years %v", s.resets, a.years)
	}

	a, s = scripted(t, runTools, AgentOptions{Stateless: true}, respond(), respond())

	_, _ = a.Complete("a", CompleteOptions{})
	_, _ = a.Run("b", nil, RunOptions{})

	if s.resets != 2 {
		t.Errorf("Stateless: %d resets, want 2", s.resets)
	}

	// An engine error resets the agent, years included.
	a, s = scripted(t, runTools, AgentOptions{}, respond())

	_, _ = a.Complete("on 5 june 2024", CompleteOptions{})

	_, err := a.Complete("again", CompleteOptions{})
	if err == nil || s.resets != 1 || len(a.years) != 0 {
		t.Errorf("after an error: %v, %d resets, years %v", err, s.resets, a.years)
	}
}
