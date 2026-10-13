package toolcall

import (
	"encoding/json"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const engineValidation = "../../testdata/text/engine/validation.json"

// Validation probe turns that differ from the engine's, and why
// (docs/FINDINGS.md F-057). Keys are "probe/turn"; a margin is the smallest
// top-1 margin of the reasoning or, where the reasoning is the engine's, of
// the call block.
var (
	// validationCallsDiffer are the turns whose calls (dispatched or
	// suppressed) differ from the engine's.
	validationCallsDiffer = map[string]string{
		// The decode differs at a near-tie (float activations against the
		// engine's int8 ones, 11.5).
		"neg_german_nicht/0":      "reasoning at a near-tie (0.096)",
		"neg_french_ne_pas/0":     "reasoning at a near-tie (0.023)",
		"trigger_will_not/0":      "reasoning at a near-tie (0.036)",
		"trigger_cant/0":          "reasoning at a near-tie (0.034)",
		"trigger_cannot/0":        "reasoning at a near-tie (0.023)",
		"trigger_doesnt/0":        "reasoning at a near-tie (0.025)",
		"trigger_must_not/0":      "reasoning at a near-tie (0.102)",
		"drop_wont/0":             "reasoning at a near-tie (0.010); the engine decodes one call",
		"ground_nested_and_top/0": "reasoning (0.098) and block at a near-tie (0.018)",
		// Routed to play_music, the call block writes free-text queries
		// the engine writes otherwise, as F-055's routed probes.
		"trigger_dont_curly/0": "routed block (0.384)",
		"trigger_not/0":        "routed block (0.445)",
		"trigger_no/0":         "routed block (0.157)",
		"trigger_stop/0":       "routed block (0.184)",
		// The paired tool reads a negation in the 13 bytes before the verb,
		// which repair.go does not port (outside this item).
		"paired_dont_curly/0":  "paired tool: the curly don’t does not keep lock_door",
		"paired_not/0":         "paired tool: not keeps lock_door",
		"paired_nothing/0":     "paired tool: not inside a word keeps lock_door",
		"paired_dont_wait/0":   "paired tool: don't 12 bytes before the verb keeps lock_door",
		"paired_dont_delay/0":  "paired tool: don't 13 bytes before the verb keeps lock_door",
		"paired_never_mind/0":  "paired tool: never mind keeps lock_door",
		"paired_dont_forget/0": "paired tool: don't forget does not keep lock_door",
	}

	// validationSplitDiffers are the turns with the engine's calls whose
	// split into dispatched and suppressed calls differs: the confidence is
	// on the other side of 0.1.
	validationSplitDiffers = map[string]string{
		"ground_nested_time/0": "the engine's confidence 0.0000 on a call with a nested object (F-055); reasoning at 0.085",
	}
)

// TestValidationMatchesEngine decodes every validation probe of the engine
// (testdata/text/engine/validation.json, `just oracle-text validation`)
// with the dequantised needle3.cact under the grammar, turn by turn, and
// builds its envelope (docs/FINDINGS.md F-057): validation.negation equals
// the engine's on every answer with a validation block; the calls equal the
// engine's but on the turns listed; where they do, the ungrounded list and
// the split into dispatched and suppressed calls equal the engine's too, but
// on the turns listed. Every listed turn must still differ.
func TestValidationMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the validation probes")
	}

	p := loadTokenizer(t)
	m := loadModel(t, needle3Cact)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var fx struct {
		System string `json:"system"`
		Probes []struct {
			ID      string           `json:"id"`
			Kind    string           `json:"kind"`
			Tools   string           `json:"tools"`
			System  string           `json:"system"`
			Turns   []string         `json:"turns"`
			Answers []engineEnvelope `json:"answers"`
		} `json:"probes"`
	}

	readJSON(t, engineValidation, &fx)

	var negations, calls, ungrounded, split, total int

	for _, pr := range fx.Probes {
		system := fx.System
		if pr.System != "" {
			system = pr.System
		}

		turns := decodeTurns(t, d, m, p, system, []byte(pr.Tools), pr.Turns, 512)

		for i, turn := range turns {
			key := pr.ID + "/" + strconv.Itoa(i)
			want := pr.Answers[i]
			total++

			js, err := json.Marshal(turn.env)
			if err != nil {
				t.Fatal(err)
			}

			var got engineEnvelope

			err = json.Unmarshal(js, &got)
			if err != nil {
				t.Fatal(err)
			}

			// negation is the request's, whatever the decode: the port's
			// reading of the input against every engine answer that has
			// a validation block.
			if want.Validation != nil {
				negations++

				if negated(pr.Turns[i]) != want.Validation.Negation {
					t.Errorf("%s: negation %v, engine %v", key, !want.Validation.Negation, want.Validation.Negation)
				}

				if got.Validation != nil && got.Validation.Negation != want.Validation.Negation {
					t.Errorf("%s: envelope negation %v, engine %v", key, got.Validation.Negation,
						want.Validation.Negation)
				}
			}

			diff := diffCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...))
			think := reflect.DeepEqual(got.Reasoning, want.Reasoning)

			block := slices.Index(turn.a.Tokens, d.open)
			if block < 0 {
				block = len(turn.a.Tokens)
			}

			t.Logf("%-36s %-9s reasoning equal %-5v min margin %.3f, in the block %.3f confidence %.4f engine %.4f "+
				"calls %d/%d engine %d/%d  %v engine %v %q", key, pr.Kind, think, minMargin(turn.a, 0, block),
				minMargin(turn.a, block, len(turn.a.Tokens)), got.Confidence, want.Confidence, len(got.Calls),
				len(got.Suppressed), len(want.Calls), len(want.Suppressed), got.Validation, want.Validation, diff)

			if why, listed := validationCallsDiffer[key]; listed == (diff == "") {
				t.Errorf("%s: calls %q (listed: %q)", key, diff, why)
			}

			if diff != "" {
				continue
			}

			calls++

			if (got.Validation == nil) != (want.Validation == nil) {
				t.Errorf("%s: validation %v, engine %v", key, got.Validation, want.Validation)
			} else if got.Validation != nil {
				if !reflect.DeepEqual(got.Validation.Ungrounded, want.Validation.Ungrounded) {
					t.Errorf("%s: ungrounded %v, engine %v", key, got.Validation.Ungrounded,
						want.Validation.Ungrounded)
				} else {
					ungrounded++
				}
			}

			same := len(got.Calls) == len(want.Calls) && len(got.Suppressed) == len(want.Suppressed)
			if same {
				split++
			}

			if why, listed := validationSplitDiffers[key]; listed == same {
				t.Errorf("%s: %d calls, %d suppressed; engine %d, %d (listed: %q)", key, len(got.Calls),
					len(got.Suppressed), len(want.Calls), len(want.Suppressed), why)
			}
		}
	}

	t.Logf("%d turns: negation checked on %d, calls equal on %d, of those ungrounded equal on %d and the split on %d",
		total, negations, calls, ungrounded, split)
}

// TestNegated covers the negation cues without the model: the spellings
// the engine reads as a negation, those it does not, and the words after a
// cue that cancel it (docs/FINDINGS.md F-057).
func TestNegated(t *testing.T) {
	for _, c := range []struct {
		s    string
		want bool
	}{
		{"don't turn off the bedroom lights", true},
		{"DON'T turn off the bedroom lights", true},
		{"Don't turn off the bedroom lights", true},
		{"do not turn off the bedroom lights", true},
		{"DO NOT turn off the bedroom lights", true},
		{"dont turn off the bedroom lights", true},
		{"never turn off the bedroom lights", true},
		{"you shouldn't turn off the bedroom lights", true},
		{"you should not turn off the bedroom lights", true},
		{"you mustn't turn off the bedroom lights", true},
		{"you must not turn off the bedroom lights", true},
		{"turn off the bedroom lights, don't wait", true},
		{`create a note titled "Don't panic"`, true},
		{"nevertheless turn off the bedroom lights", true},
		{"whenever you can, turn on the bedroom lights", true},
		{"the xdon't room", true},
		{"the don'tx room", true},
		{"never-mind, turn on the bedroom lights", true},
		{"turn off the bedroom lights and don't worries", true},
		{"turn off the bedroom lights and don't hesitating", true},
		{"turn off the bedroom lights and don't, forget the kitchen", true},
		{"don't forget the kitchen and don't touch the hallway", true},

		// Not a negation.
		{"turn off the bedroom lights", false},
		{"don’t turn off the bedroom lights", false},
		{"don´t turn off the bedroom lights", false},
		{"you shouldn’t turn off the bedroom lights", false},
		{"you shouldnt turn off the bedroom lights", false},
		{"dont, turn off the bedroom lights", false},
		{"turn off the bedroom lights dont", false},
		{"the dontx room", false},
		{"dont't turn on the bedroom lights", false},
		{"do  not turn off the bedroom lights", false},
		{"do-not turn off the bedroom lights", false},
		{"donot turn off the bedroom lights", false},
		{"turn on the bedroom lights, not the kitchen lights", false},
		{"no, turn off the bedroom lights", false},
		{"I won't need the bedroom lights", false},
		{"I can't see", false},
		{"you cannot turn off the bedroom lights", false},
		{"it doesn't matter", false},
		{"stop the music", false},
		{"cancel the music", false},
		{"turn on the bedroom lights without the kitchen", false},
		{"turn on the bedroom lights instead of the kitchen lights", false},
		{"turn on the lights except the bedroom", false},
		{"Schalte das Licht nicht aus", false},
		{"No apagues las luces", false},
		{"N'éteins pas les lumières", false},

		// A cue cancelled by the word after it.
		{"never mind, turn on the bedroom lights", false},
		{"nevermind, turn on the bedroom lights", false},
		{"turn off the bedroom lights and don't forget the kitchen", false},
		{"turn off the bedroom lights and DON'T FORGET the kitchen", false},
		{"turn off the bedroom lights and don't  forget the kitchen", false},
		{"turn off the bedroom lights and dont forget the kitchen", false},
		{"turn off the bedroom lights and don'tforget the kitchen", false},
		{"turn off the bedroom lights and don't forgetting", false},
		{"turn off the bedroom lights and do not worry about the kitchen", false},
		{"turn off the bedroom lights and never hesitate", false},
		{"turn off the bedroom lights and don't mind the kitchen", false},
		{"turn off the bedroom lights and must not disturb me", false},
		{`create a note titled "Do not disturb"`, false},
	} {
		if got := negated(c.s); got != c.want {
			t.Errorf("negated(%q) = %v, want %v", c.s, got, c.want)
		}
	}
}

// TestNegatesVerb covers the verbs the repair's drop finds negated: a tool
// verb among the three whitespace tokens after a drop cue, which match
// whole (docs/FINDINGS.md F-057).
func TestNegatesVerb(t *testing.T) {
	for _, c := range []struct {
		s, verb string
		want    bool
	}{
		{"turn on the kitchen lights and don't lock the front door", "lock", true},
		{"turn on the kitchen lights and DON'T Lock the front door", "lock", true},
		{"play jazz and don't play anything", "play", true},
		{"turn on the kitchen lights and dont lock the front door", "lock", true},
		{"turn on the kitchen lights and do not lock the front door", "lock", true},
		{"you should not unlock it", "unlock", true},
		{"you must not open it", "open", true},
		{"you shouldn't unlock it", "unlock", true},
		{"you mustn't unlock it", "unlock", true},
		{"never ever lock the front door", "lock", true},
		{"don't wait, guys, lock the front door", "lock", true},
		{"don't wait-up-now lock the front door", "lock", true},
		{"don't  lock the front door", "lock", true},

		// Not negated.
		{"turn on the kitchen lights and don't lock the front door", "set", false},
		{"turn on the kitchen lights but never unlock the garage", "lock", false},
		{"don't be so slow lock the front door", "lock", false},
		{"don't wait 5 min, lock the front door", "lock", false},
		{"lock the front door and don't wait", "lock", false},
		{"turn on the kitchen lights and don't forget to lock the front door", "lock", false},
		{"turn on the kitchen lights and don’t lock the front door", "lock", false},
		{"turn on the kitchen lights and won't lock the front door", "lock", false},
		{"turn on the kitchen lights and not lock the front door", "lock", false},
		{"don't, wait, lock the front door", "lock", false},
		{"don't! lock the front door", "lock", false},
		{"do not, lock the front door", "lock", false},
		{"please,don't lock the front door", "lock", false},
		{"xdon't lock the front door", "lock", false},
		{"nevertheless lock the front door", "lock", false},
		{"don't wait,lock the front door", "lock", false},
	} {
		if got := negatesVerb(c.s, c.verb); got != c.want {
			t.Errorf("negatesVerb(%q, %q) = %v, want %v", c.s, c.verb, got, c.want)
		}
	}
}

// TestUngroundedRules covers the grounding rules the probes pin without the
// model: top-level fields only, once per call, against the request and the
// conversation's earlier inputs but not the system facts
// (docs/FINDINGS.md F-057).
func TestUngroundedRules(t *testing.T) {
	tools := `[{"name":"set_alarm","parameters":{"type":"object","properties":{"time":{"type":"string"},` +
		`"snooze":{"type":"object","properties":{"until":{"type":"string"},"minutes":{"type":"integer"}}},` +
		`"times":{"type":"array","items":{"type":"string"}},"prices":{"type":"array","items":{"type":"number"}},` +
		`"items":{"type":"array","items":{"type":"object","properties":{"product":{"type":"string"}}}},` +
		`"minutes":{"type":"integer"}},"required":["time","minutes"]}}]`

	n, err := Normalise([]byte(tools))
	if err != nil {
		t.Fatal(err)
	}

	answer := func(calls ...string) *Decoded {
		return &Decoded{Text: thinkOpen + "r\n</think>\n<tool_call>[" + strings.Join(calls, ",") + "]</tool_call>"}
	}
	alarm := func(args string) string { return `{"name":"set_alarm","arguments":{` + args + `}}` }

	for _, c := range []struct {
		name, input, system string
		earlier             []string
		a                   *Decoded
		want                []string
	}{
		{
			"a top-level time", "wake me at 6:45", "", nil, answer(alarm(`"time":"06:45"`)),
			[]string{"set_alarm.time"},
		},
		{
			"a nested time: not listed", "snooze until 7:10", "", nil,
			answer(alarm(`"snooze":{"until":"07:10","minutes":5}`)),
			[]string{},
		},
		{
			"array items: not listed", "set alarms at 6:45 and 7:15 for $1,200.00", "", nil,
			answer(alarm(`"times":["06:45","07:15"],"prices":[1200.0],"items":[{"product":"bread"}]`)),
			[]string{},
		},
		{
			"a top-level field beside nested ones", "wake me at 6:45 and snooze until 7:10", "", nil,
			answer(alarm(`"time":"06:45","snooze":{"until":"07:10"}`)),
			[]string{"set_alarm.time"},
		},
		{
			"once per call", "wake me at 6:45 and at 7:15", "", nil,
			answer(alarm(`"time":"06:45"`), alarm(`"time":"07:15"`)),
			[]string{"set_alarm.time", "set_alarm.time"},
		},
		{
			"the system facts do not ground", "set the default timer", "date: 2026-10-11 Sun 09:30\ndefault: 25", nil,
			answer(alarm(`"minutes":25,"time":"Hamburg"`)),
			[]string{"set_alarm.minutes", "set_alarm.time"},
		},
		{
			"an earlier user turn grounds", "set another one of the same length", "",
			[]string{"set an alarm called Hamburg for 25 minutes"},
			answer(alarm(`"minutes":25,"time":"hamburg"`)),
			[]string{},
		},
		{
			"an earlier tool result grounds", "take me there", "",
			[]string{"look it up", `[{"address": "14 Herrengasse", "minutes": 25}]`},
			answer(alarm(`"minutes":25,"time":"14 Herrengasse"`)),
			[]string{},
		},
		{
			"an earlier turn grounds whole numbers only", "same again", "",
			[]string{"for 125 minutes"},
			answer(alarm(`"minutes":25`)),
			[]string{"set_alarm.minutes"},
		},
	} {
		e, err := n.Envelope(c.a, c.input, c.system, c.earlier, 0.5)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if e.Validation == nil || !reflect.DeepEqual(e.Validation.Ungrounded, c.want) {
			t.Errorf("%s: validation %+v, want ungrounded %v", c.name, e.Validation, c.want)
		}
	}
}

// TestEnvelopeWithheld covers the calls the repair would drop to none: a
// request that negates or excludes its only call keeps it, suppressed
// whatever the confidence and the triggers (docs/FINDINGS.md F-057).
func TestEnvelopeWithheld(t *testing.T) {
	home := `[{"name":"set_lights","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":` +
		`{"type":"boolean"}}},"triggers":["lights"]},{"name":"lock_door","parameters":{"type":"object","properties":` +
		`{"door":{"type":"string"}}}}]`

	n, err := Normalise([]byte(home))
	if err != nil {
		t.Fatal(err)
	}

	answer := func(calls ...string) *Decoded {
		return &Decoded{Text: thinkOpen + "r\n</think>\n<tool_call>[" + strings.Join(calls, ",") + "]</tool_call>"}
	}
	lights := func(room string) string { return `{"name":"set_lights","arguments":{"room":"` + room + `","on":true}}` }
	lock := func(door string) string { return `{"name":"lock_door","arguments":{"door":"` + door + `"}}` }

	for _, c := range []struct {
		name, input    string
		a              *Decoded
		ship, withheld int
	}{
		{"the only call negated", "do not lock the front door", answer(lock("front door")), 0, 1},
		{"dont before the only call", "dont lock the front door", answer(lock("front door")), 0, 1},
		{
			"every call negated", "don't lock the front door and don't lock the back door",
			answer(lock("front door"), lock("back door")), 0, 2,
		},
		{"the only call excluded, triggered", "turn on the lights except the bedroom", answer(lights("bedroom")), 0, 1},
		{
			"one of two negated: dropped", "turn on the bedroom lights and don't lock the door",
			answer(lights("bedroom"), lock("door")), 1, 0,
		},
		{"another verb: shipped", "never lock the front door", answer(lights("front door")), 1, 0},
		{"won't: shipped", "won't lock the front door", answer(lock("front door")), 1, 0},
	} {
		e, err := n.Envelope(c.a, c.input, "", nil, 0.9)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if len(e.FunctionCalls) != c.ship || len(e.SuppressedCalls) != c.withheld {
			t.Errorf("%s: %d calls, %d suppressed; want %d, %d", c.name, len(e.FunctionCalls),
				len(e.SuppressedCalls), c.ship, c.withheld)
		}
	}
}
