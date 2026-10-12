package toolcall

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

const engineTriggers = "../../testdata/text/engine/triggers.json"

// engineEnvelope is an engine answer as recorded, parsed.
type engineEnvelope struct {
	Type       string      `json:"type"`
	Success    bool        `json:"success"`
	Error      *string     `json:"error"`
	ErrorCode  *string     `json:"error_code"`
	Reason     *string     `json:"reason"`
	Calls      []call      `json:"function_calls"`
	Suppressed []call      `json:"suppressed_calls"`
	Reasoning  *string     `json:"reasoning"`
	Confidence float64     `json:"confidence"`
	Validation *Validation `json:"validation"`
}

// Envelope turns that differ from the engine's, and why (docs/FINDINGS.md
// F-053, F-054). Keys are "case/turn".
var (
	// The confidence is off by more than 0.02: the decode differs at a
	// near-tie or under the grammar (decode_test.go), the head's score of a
	// respond turn is not fitted, nor event_tomorrow's (0.026).
	confidenceOff = map[string]string{
		"alarm_morning/0":        "near-tie",
		"alarm_untriggered/0":    "near-tie",
		"alarm_triggered/0":      "near-tie",
		"alarm_invalid/0":        "near-tie",
		"alarm_no_match/0":       "near-tie",
		"catalogue_reminder/0":   "near-tie",
		"lights_and_lock/0":      "grammar",
		"catalogue_lock/0":       "grammar",
		"spanish_lights_music/0": "grammar",
		"timer_then_result/1":    "respond turn",
		"event_tomorrow/0":       "0.026",
	}
)

// The calls differ from the engine's, by its repair after decoding (the
// capitalisation, an empty optional dropped, a date re-resolved, a polarity
// flipped) or its schema grammar (keys outside the schema); PLAN.md 11.4.
var callsDiffer = map[string]string{
	"timer_pasta/0":            "capitalisation",
	"timer_then_result/0":      "capitalisation",
	"timer_triggered/0":        "capitalisation",
	"message_and_timer/0":      "capitalisation",
	"event_tomorrow/0":         "capitalisation",
	"event_explicit/0":         "capitalisation",
	"french_message/0":         "capitalisation",
	"catalogue_email/0":        "capitalisation",
	"alarm_morning/0":          "empty optional",
	"alarm_untriggered/0":      "empty optional",
	"alarm_triggered/0":        "empty optional",
	"alarm_invalid/0":          "empty optional",
	"alarm_no_match/0":         "empty optional",
	"weather_friday/0":         "date",
	"event_no_year/0":          "date, capitalisation",
	"negation_door/0":          "polarity",
	"negation_triggered/0":     "polarity",
	"negation_two_triggered/0": "polarity",
	"lights_and_lock/0":        "grammar",
	"spanish_lights_music/0":   "grammar",
	"catalogue_lock/0":         "grammar",
}

// reasoningDiffers are the turns decoded differently from the engine at a
// near-tie (decode_test.go).
var reasoningDiffers = map[string]bool{
	"alarm_morning/0": true, "alarm_untriggered/0": true, "alarm_triggered/0": true, "alarm_invalid/0": true,
	"alarm_no_match/0": true, "event_no_year/0": true, "catalogue_reminder/0": true,
}

// envelopeTurn checks one turn's envelope against the engine's: every field,
// the calls and the reasoning but on the turns listed.
func envelopeTurn(t *testing.T, key string, got turnOut, want engineEnvelope) {
	t.Helper()

	e := got.env

	js, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}

	var g engineEnvelope

	err = json.Unmarshal(js, &g)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("%-28s confidence %.4f engine %.4f  calls %d/%d engine %d/%d  validation %v engine %v",
		key, g.Confidence, want.Confidence, len(g.Calls), len(g.Suppressed), len(want.Calls), len(want.Suppressed),
		g.Validation, want.Validation)

	if g.Type != want.Type || g.Success != want.Success || !reflect.DeepEqual(g.Error, want.Error) ||
		!reflect.DeepEqual(g.ErrorCode, want.ErrorCode) || !reflect.DeepEqual(g.Reason, want.Reason) {
		t.Errorf("%s: %s %v %v %v %v, engine %s %v %v %v %v", key, g.Type, g.Success, g.Error, g.ErrorCode, g.Reason,
			want.Type, want.Success, want.Error, want.ErrorCode, want.Reason)
	}

	if len(g.Calls) != len(want.Calls) || len(g.Suppressed) != len(want.Suppressed) {
		t.Errorf("%s: %d calls, %d suppressed; engine %d, %d", key, len(g.Calls), len(g.Suppressed),
			len(want.Calls), len(want.Suppressed))
	}

	d := diffCalls(append(g.Calls, g.Suppressed...), append(want.Calls, want.Suppressed...))
	if why, listed := callsDiffer[key]; listed == (d == "") {
		t.Errorf("%s: calls %q (listed: %q)", key, d, why)
	}

	same := reflect.DeepEqual(g.Reasoning, want.Reasoning)
	if near := reasoningDiffers[key]; near == same {
		t.Errorf("%s: reasoning %v, engine %v (listed as a near-tie: %v)", key, g.Reasoning, want.Reasoning, near)
	}

	if (g.Validation == nil) != (want.Validation == nil) {
		t.Errorf("%s: validation %v, engine %v", key, g.Validation, want.Validation)
	} else if g.Validation != nil {
		if g.Validation.Negation != want.Validation.Negation {
			t.Errorf("%s: negation %v, engine %v", key, g.Validation.Negation, want.Validation.Negation)
		}

		if !reflect.DeepEqual(g.Validation.Ungrounded, want.Validation.Ungrounded) {
			t.Errorf("%s: ungrounded %v, engine %v", key, g.Validation.Ungrounded, want.Validation.Ungrounded)
		}
	}

	off := math.Abs(g.Confidence - want.Confidence)
	if why, listed := confidenceOff[key]; listed == (off <= 0.02) {
		t.Errorf("%s: confidence %.4f, engine %.4f (listed: %q)", key, g.Confidence, want.Confidence, why)
	}
}

// TestEnvelopeMatchesEngine builds the envelope of every turn of the engine
// corpus and of the trigger probes the model answers without routing, from
// the dequantised needle3.cact decode (docs/FINDINGS.md F-054).
func TestEnvelopeMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the engine corpus")
	}

	p := loadTokenizer(t)
	m := loadModel(t, needle3Cact)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var (
		corpus    engineCorpus
		responses map[string][]engineEnvelope
		triggers  struct {
			System string `json:"system"`
			Probes []struct {
				ID     string         `json:"id"`
				Tools  string         `json:"tools"`
				Query  string         `json:"query"`
				Answer engineEnvelope `json:"answer"`
				Stderr string         `json:"stderr"`
			} `json:"probes"`
		}
	)

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineResponse, &responses)
	readJSON(t, engineTriggers, &triggers)

	for _, c := range corpus.Cases {
		for i, turn := range decodeCase(t, d, m, p, &corpus, c) {
			envelopeTurn(t, c.ID+"/"+string(rune('0'+i)), turn, responses[c.ID][i])
		}
	}

	for _, pr := range triggers.Probes {
		if routed[pr.ID] {
			continue
		}

		turn := decodeTurns(t, d, m, p, triggers.System, []byte(pr.Tools), []string{pr.Query}, 512)[0]
		envelopeTurn(t, pr.ID+"/0", turn, pr.Answer)
	}
}

// routed are the trigger probes the engine routes to a tool the model does
// not call; that takes its grammar (PLAN.md 11.4).
var routed = map[string]bool{"weather_to_timer": true, "pizza_to_lights": true, "capital_upper": true}

// TestEnvelope covers the envelope's rules without the model.
func TestEnvelope(t *testing.T) {
	tools := `[{"name":"setAlarm","description":"Set an alarm","parameters":{"type":"object","properties":{` +
		`"time":{"type":"string"},"mode":{"type":"string","enum":["loud","soft"]},"day":{"type":"string",` +
		`"format":"date"},"minutes":{"type":"integer"},"on":{"type":"boolean"}}},"triggers":["WAKE me","("]}]`

	n, err := Normalise([]byte(tools))
	if err != nil {
		t.Fatal(err)
	}

	if want := []string{"set_alarm: ("}; !reflect.DeepEqual(n.Ignored, want) {
		t.Errorf("Ignored = %v, want %v", n.Ignored, want)
	}

	answer := func(body string) *Decoded { return &Decoded{Text: thinkOpen + "r\n</think>\n" + body} }
	alarm := `<tool_call>[{"name":"set_alarm","arguments":{"time":"06:45","mode":"soft","day":"2026-10-12",` +
		`"minutes":10,"on":true}}]</tool_call>`

	for name, c := range map[string]struct {
		a     *Decoded
		input string
		conf  float64
		want  string
	}{
		"truncated": {
			&Decoded{Text: thinkOpen + "r", Truncated: true}, "x", 0.5,
			`{"type":"call","success":false,"error":"tool call truncated: token budget exhausted",` +
				`"error_code":"truncated","reason":"runtime_failure","function_calls":[],"suppressed_calls":[],` +
				`"reasoning":null,"confidence":0.5000}`,
		},
		"respond": {
			answer("Done."), "x", 0.25,
			`{"type":"respond","success":true,"error":null,"error_code":null,"reason":null,"function_calls":[],` +
				`"suppressed_calls":[],"reasoning":"r","confidence":0.2500}`,
		},
		"refusal": {
			answer("<tool_call>[]</tool_call>"), "x", 0.9,
			`{"type":"call","success":true,"error":null,"error_code":null,"reason":null,"function_calls":[],` +
				`"suppressed_calls":[],"reasoning":"r","confidence":0.9000}`,
		},
		"dispatched, grounding": {
			answer(alarm), "set an alarm at 6:45 for 10 minutes", 0.1,
			`{"type":"call","success":true,"error":null,"error_code":null,"reason":null,"function_calls":[` +
				`{"name":"setAlarm","arguments":{"day":"2026-10-12","minutes":10,"mode":"soft","on":true,` +
				`"time":"06:45"}}],"suppressed_calls":[],"reasoning":"r","confidence":0.1000,` +
				`"validation":{"ungrounded":["set_alarm.time"],"negation":false}}`,
		},
		"suppressed": {
			answer(alarm), "set an alarm at 06:45 in 100 minutes", 0.0999,
			`{"type":"call","success":true,"error":null,"error_code":null,"reason":null,"function_calls":[],` +
				`"suppressed_calls":[{"name":"setAlarm","arguments":{"day":"2026-10-12","minutes":10,"mode":"soft",` +
				`"on":true,"time":"06:45"}}],"reasoning":"r","confidence":0.0999,` +
				`"validation":{"ungrounded":["set_alarm.minutes"],"negation":false}}`,
		},
		"triggered below the floor": {
			answer(alarm), "wake me at 06:45 in 10 minutes", 0.01,
			`{"type":"call","success":true,"error":null,"error_code":null,"reason":null,"function_calls":[` +
				`{"name":"setAlarm","arguments":{"day":"2026-10-12","minutes":10,"mode":"soft","on":true,` +
				`"time":"06:45"}}],"suppressed_calls":[],"reasoning":"r","confidence":0.0100,` +
				`"validation":{"ungrounded":[],"negation":false}}`,
		},
		"triggered but negated": {
			answer(alarm), "don't wake me at 06:45 in 10 minutes", 0.01,
			`{"type":"call","success":true,"error":null,"error_code":null,"reason":null,"function_calls":[],` +
				`"suppressed_calls":[{"name":"setAlarm","arguments":{"day":"2026-10-12","minutes":10,"mode":"soft",` +
				`"on":true,"time":"06:45"}}],"reasoning":"r","confidence":0.0100,` +
				`"validation":{"ungrounded":[],"negation":true}}`,
		},
	} {
		e, err := n.Envelope(c.a, c.input, c.conf)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		js, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}

		if string(js) != c.want {
			t.Errorf("%s:\n got %s\nwant %s", name, js, c.want)
		}
	}

	_, err = n.Envelope(answer("<tool_call>[{]</tool_call>"), "x", 1)
	if err == nil {
		t.Error("a broken call block parsed")
	}
}

// TestNoTools is the engine's answer over no tools, byte for byte as
// libneedle3 3.2.0 writes it without the timings (triggers.json).
func TestNoTools(t *testing.T) {
	js, err := json.Marshal(NoTools())
	if err != nil {
		t.Fatal(err)
	}

	want := `{"type":"call","success":true,"error":null,"function_calls":[],"suppressed_calls":[],` +
		`"reasoning":"no tools declared, returning the empty call","confidence":1.0000}`
	if string(js) != want {
		t.Errorf("NoTools = %s, want %s", js, want)
	}

	var fx struct {
		NoTools json.RawMessage `json:"no_tools"`
	}

	readJSON(t, engineTriggers, &fx)

	var got, rec any

	_ = json.Unmarshal(js, &got)
	_ = json.Unmarshal(fx.NoTools, &rec)

	if !reflect.DeepEqual(got, rec) {
		t.Errorf("NoTools = %s, recorded %s", js, fx.NoTools)
	}
}

func TestContainsNumber(t *testing.T) {
	for _, c := range []struct {
		s, lit string
		want   bool
	}{
		{"10 minutes", "10", true},
		{"100 minutes", "10", false},
		{"$1,200.00", "1200.0", false},
		{"21.5 degrees", "21.50", true},
		{"at 21.5", "1.5", false},
		{"x", "1e999", false},
	} {
		if got := containsNumber(c.s, c.lit); got != c.want {
			t.Errorf("containsNumber(%q, %q) = %v", c.s, c.lit, got)
		}
	}
}

// TestUngroundedComposed: a field's schemas include the allOf, anyOf and
// oneOf branches that declare it, so their enums exempt it (PR #64 review).
func TestUngroundedComposed(t *testing.T) {
	tools := `[{"name":"t","parameters":{"type":"object","properties":{"o":{"allOf":[{"properties":{` +
		`"mode":{"enum":["fast","slow"]},"name":{"type":"string"}}}]},"list":{"type":"array","items":{"oneOf":[` +
		`{"properties":{"day":{"type":"string","format":"date"}}}]}}},"anyOf":[{"properties":{"color":{` +
		`"enum":["red"]}}}]}}]`

	n, err := Normalise([]byte(tools))
	if err != nil {
		t.Fatal(err)
	}

	calls := []modelCall{{"t", []byte(`{"o":{"mode":"fast","name":"zed"},"color":"red",` +
		`"list":[{"day":"next week"}],"other":"elsewhere"}`)}}

	got := n.ungrounded(calls, "x")
	if want := []string{"t.o.name", "t.other"}; !reflect.DeepEqual(got, want) {
		t.Errorf("ungrounded = %v, want %v", got, want)
	}
}

func TestTop2(t *testing.T) {
	for _, c := range []struct {
		row          []float32
		best, second int
	}{
		{[]float32{3}, 0, 0},
		{[]float32{1, 2}, 1, 0},
		{[]float32{5, 1, 7, 6}, 2, 3},
		{[]float32{2, 2, 1}, 0, 1},
	} {
		if b, s := top2(c.row); b != c.best || s != c.second {
			t.Errorf("top2(%v) = %d, %d, want %d, %d", c.row, b, s, c.best, c.second)
		}
	}
}
