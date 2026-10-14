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
// F-053, F-054, F-055, F-062). Keys are "case/turn".
var (
	// The confidence is off by more than 0.02: the decode differs at a
	// near-tie (decode_test.go), or the ConfidenceHead's score of a turn
	// decoded as the engine does is off in float32. Under the engine's int8
	// numerics lights_and_lock comes within 0.02, and event_tomorrow
	// (-0.043) and catalogue_lock (+0.067) stay off: on the dequantised
	// weights with the same int8 numerics both come within 0.02 and two
	// other turns fall out, the weight path's implementation noise of
	// F-060 (docs/FINDINGS.md F-062). The respond turn takes the decode
	// probability, the geometric mean of its tokens, and is within 0.02.
	confidenceOff = map[string]string{
		"alarm_morning/0":      "near-tie",
		"alarm_untriggered/0":  "near-tie",
		"alarm_triggered/0":    "near-tie",
		"alarm_invalid/0":      "near-tie",
		"alarm_no_match/0":     "near-tie",
		"catalogue_reminder/0": "near-tie",
		"event_tomorrow/0":     "the head -0.026 (int8 -0.043)",
		"lights_and_lock/0":    "the head -0.031 (int8 within 0.02)",
		"catalogue_lock/0":     "the head +0.027 (int8 +0.067)",
	}
)

// callsDiffer are the turns whose calls differ from the engine's, and why.
// None does: the repair after decoding (repair.go) closed the 20 turns the
// engine's capitalisation, dropped optionals, dates and polarity changed
// (docs/FINDINGS.md F-056).
var callsDiffer = map[string]string{}

// ungroundedDiffers are the turns whose ungrounded list differs from the
// engine's. None does: the engine lists the repaired calls, and so does
// Envelope (pizza_to_lights' brightness, docs/FINDINGS.md F-056).
var ungroundedDiffers = map[string]bool{}

// reasoningDiffers are the turns decoded differently from the engine at a
// near-tie (decode_test.go).
var reasoningDiffers = map[string]bool{
	"alarm_morning/0": true, "alarm_untriggered/0": true, "alarm_triggered/0": true, "alarm_invalid/0": true,
	"alarm_no_match/0": true, "event_no_year/0": true, "catalogue_reminder/0": true,
}

// envelopeTurn checks one turn's envelope against the engine's: every field,
// the calls and the reasoning but on the turns listed. It returns whether
// the calls (dispatched and suppressed), the function_calls alone, the
// reasoning and the confidence (within 0.02) agree.
func envelopeTurn(t *testing.T, key string, got turnOut, want engineEnvelope) (calls, fc, reasoning, conf bool) {
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
		listedf(t, "%s: %s %v %v %v %v, engine %s %v %v %v %v", key, g.Type, g.Success, g.Error, g.ErrorCode, g.Reason,
			want.Type, want.Success, want.Error, want.ErrorCode, want.Reason)
	}

	if len(g.Calls) != len(want.Calls) || len(g.Suppressed) != len(want.Suppressed) {
		listedf(t, "%s: %d calls, %d suppressed; engine %d, %d", key, len(g.Calls), len(g.Suppressed),
			len(want.Calls), len(want.Suppressed))
	}

	d := diffCalls(append(g.Calls, g.Suppressed...), append(want.Calls, want.Suppressed...))
	if why, isListed := callsDiffer[key]; isListed == (d == "") {
		listedf(t, "%s: calls %q (listed: %q)", key, d, why)
	}

	same := reflect.DeepEqual(g.Reasoning, want.Reasoning)
	if near := reasoningDiffers[key]; near == same {
		listedf(t, "%s: reasoning %v, engine %v (listed as a near-tie: %v)", key, g.Reasoning, want.Reasoning, near)
	}

	if (g.Validation == nil) != (want.Validation == nil) {
		listedf(t, "%s: validation %v, engine %v", key, g.Validation, want.Validation)
	} else if g.Validation != nil {
		if g.Validation.Negation != want.Validation.Negation {
			listedf(t, "%s: negation %v, engine %v", key, g.Validation.Negation, want.Validation.Negation)
		}

		same := reflect.DeepEqual(g.Validation.Ungrounded, want.Validation.Ungrounded)
		if isListed := ungroundedDiffers[key]; isListed == same {
			listedf(t, "%s: ungrounded %v, engine %v (listed: %v)", key, g.Validation.Ungrounded,
				want.Validation.Ungrounded, isListed)
		}
	}

	off := math.Abs(g.Confidence - want.Confidence)
	if why, isListed := confidenceOff[key]; isListed == (off <= 0.02) {
		listedf(t, "%s: confidence %.4f, engine %.4f (listed: %q)", key, g.Confidence, want.Confidence, why)
	}

	return d == "", functionCallsEqual(g.Calls, want.Calls), same, off <= 0.02
}

// TestEnvelopeMatchesEngine builds the envelope of every turn of the engine
// corpus and of every trigger probe, from the dequantised needle3.cact
// decode under the grammar; the routed probes (weather_to_timer,
// pizza_to_lights, capital_upper) decode their call under the triggered
// tool's grammar (docs/FINDINGS.md F-054, F-055).
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

	var calls, fc, reasoning, conf, total int

	count := func(c, d, r, f bool) {
		total++

		for _, x := range []struct {
			ok bool
			n  *int
		}{{c, &calls}, {d, &fc}, {r, &reasoning}, {f, &conf}} {
			if x.ok {
				*x.n++
			}
		}
	}

	// 11.4's Accept, on the 25 corpus turns alone: type and function_calls
	// equal, reasoning equal, and the confidence within 0.02 among the turns
	// decoded as the engine's (type, reasoning and every call).
	var acceptCalls, acceptReasoning, acceptDecoded, acceptConf, corpusTurns int

	for _, c := range corpus.Cases {
		for i, turn := range decodeCase(t, d, m, p, &corpus, c) {
			want := responses[c.ID][i]
			callsEq, fcEq, reasoningEq, confEq := envelopeTurn(t, c.ID+"/"+string(rune('0'+i)), turn, want)
			count(callsEq, fcEq, reasoningEq, confEq)

			corpusTurns++

			if fcEq && turn.env.Type == want.Type {
				acceptCalls++
			}

			if reasoningEq {
				acceptReasoning++
			}

			if callsEq && reasoningEq && turn.env.Type == want.Type {
				acceptDecoded++

				if confEq {
					acceptConf++
				}
			}
		}
	}

	logCount(t, "corpus type and function_calls equal", acceptCalls, corpusTurns)
	logCount(t, "corpus reasoning equal", acceptReasoning, corpusTurns)
	logCount(t, "corpus turns decoded as the engine's", acceptDecoded, corpusTurns)
	t.Logf("%s numerics: corpus confidence within 0.02 on %d of %d turns decoded as the engine's",
		testNumerics(t), acceptConf, acceptDecoded)

	for _, pr := range triggers.Probes {
		turn := decodeTurns(t, d, m, p, triggers.System, []byte(pr.Tools), []string{pr.Query}, 512)[0]
		count(envelopeTurn(t, pr.ID+"/0", turn, pr.Answer))
	}

	int8Floor(t, "calls equal", calls, total, int8EnvelopeCalls)
	int8Floor(t, "reasoning equal", reasoning, total, int8EnvelopeReasoning)
	int8Floor(t, "confidence within 0.02", conf, total, int8EnvelopeConfidence)
	logCount(t, "function_calls equal", fc, total)
}

// TestEnvelope covers the envelope's rules without the model.
func TestEnvelope(t *testing.T) {
	tools := `[{"name":"setAlarm","description":"Set an alarm","parameters":{"type":"object","properties":{` +
		`"time":{"type":"string"},"mode":{"type":"string","enum":["loud","soft"]},"day":{"type":"string",` +
		`"format":"date"},"minutes":{"type":"integer"},"on":{"type":"boolean"}},"required":["minutes"]},` +
		`"triggers":["WAKE me","("]}]`

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
		e, err := n.Envelope(c.a, c.input, "", nil, c.conf)
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

	_, err = n.Envelope(answer("<tool_call>[{]</tool_call>"), "x", "", nil, 1)
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
// oneOf branches that declare it, so their enums exempt it (PR #64 review);
// fields inside objects and arrays are not checked (docs/FINDINGS.md F-057).
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
	if want := []string{"t.other"}; !reflect.DeepEqual(got, want) {
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
