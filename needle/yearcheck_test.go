package needle

import (
	"encoding/json"
	"slices"
	"testing"
)

// TestNoYearCheck: with NoYearCheck, Complete's validation is the engine's
// as the engine's CLI writes it, without the Python client's year check:
// no field added after the engine's, no validation made where the engine
// wrote none, and the JSON unchanged. Run then runs a call whose date the
// query does not license, since only the engine's fields block it.
func TestNoYearCheck(t *testing.T) {
	const bad = `{"title":"dinner","day":"2019-03-05"}`

	for _, c := range []struct {
		engine       []string
		noValidation bool
	}{
		{nil, false},
		{nil, true},
		{[]string{"event.title"}, false},
	} {
		e := answer(one("event", bad), c.engine...)
		if c.noValidation {
			e.Validation = nil
		}

		want, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}

		a, _ := scripted(t, "["+extractSchema+"]", AgentOptions{NoYearCheck: true}, e)

		r, err := a.Complete("dinner on 5 March 2025", CompleteOptions{})
		if err != nil {
			t.Fatal(err)
		}

		switch {
		case c.noValidation && r.Validation != nil:
			t.Errorf("%v: validation %+v, want none", c.engine, r.Validation)
		case !c.noValidation && (r.Validation == nil || !slices.Equal(r.Validation.Ungrounded, c.engine)):
			t.Errorf("%v: validation %+v, want ungrounded %v", c.engine, r.Validation, c.engine)
		}

		js, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}

		if string(js) != string(want) {
			t.Errorf("%v: JSON %s, want the engine's %s", c.engine, js, want)
		}
	}

	a, _ := scripted(t, "["+extractSchema+"]", AgentOptions{NoYearCheck: true}, answer(one("event", bad)), respond())

	r, err := a.Run("dinner on 5 March 2025", map[string]ToolFunc{
		"event": func(json.RawMessage) (any, error) { return map[string]any{"ok": true}, nil },
	}, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	js, err := json.Marshal(r.Results)
	if err != nil {
		t.Fatal(err)
	}

	if want := `[{"ok":true}]`; string(js) != want {
		t.Errorf("Run results %s, want %s", js, want)
	}
}
