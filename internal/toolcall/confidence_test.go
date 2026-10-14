package toolcall

import (
	"encoding/json"
	"math"
	"reflect"
	"strconv"
	"testing"
)

// TestConfidenceDecodeProbability covers the decode probability the
// envelope takes the min of with the head's score, without the model
// (docs/FINDINGS.md F-062).
func TestConfidenceDecodeProbability(t *testing.T) {
	const eps = 1e-12

	for name, c := range map[string]struct {
		a    *Decoded
		want float64
	}{
		"call: the least likely block token among the allowed ones": {
			// <think>\n r </think> \n <tool_call> [ … ] </tool_call>
			&Decoded{
				Text:    thinkOpen + "r\n</think>\n<tool_call>[]</tool_call>",
				Tokens:  []int{10, 11, 12, 13, 14, 15, 16, 17},
				Probs:   []float64{1, 1, 0.01, 0.9, 0.8, 0.5, 0.2, 0.9},
				Allowed: []float64{1, 1, 0.01, 0.9, 0.8, 0.6, 0.3, 1},
				block:   5,
			},
			0.3,
		},
		"call: the reasoning does not count": {
			&Decoded{
				Text:    thinkOpen + "r\n</think>\n<tool_call>[]</tool_call>",
				Tokens:  []int{10, 11, 12, 13, 14},
				Probs:   []float64{1, 0.001, 0.9, 0.9, 0.9},
				Allowed: []float64{1, 0.001, 0.95, 0.97, 0.99},
				block:   2,
			},
			0.95,
		},
		"respond: the geometric mean over every token, the forced ones as 1": {
			&Decoded{
				Text:    thinkOpen + "r\n</think>\nDone.",
				Tokens:  []int{10, 11, 12, 13},
				Probs:   []float64{1, 1, 0.25, 0.64},
				Allowed: []float64{1, 1, 0.5, 0.64},
				block:   -1,
			},
			math.Sqrt(math.Sqrt(0.25 * 0.64)),
		},
		"respond: the stop token counts": {
			&Decoded{
				Text:    thinkOpen + "r\n</think>\nDone.",
				Tokens:  []int{10, 11, 12},
				Probs:   []float64{1, 1, 0.25},
				Allowed: []float64{1, 1, 0.25},
				stop:    0.64,
				block:   -1,
			},
			math.Sqrt(math.Sqrt(0.25 * 0.64)),
		},
		"truncated: none": {
			&Decoded{
				Text: thinkOpen + "r", Tokens: []int{10, 11}, Probs: []float64{1, 0.1}, Allowed: []float64{1, 0.1},
				Truncated: true, block: -1,
			},
			1,
		},
		"no probabilities recorded": {&Decoded{Text: thinkOpen + "r\n</think>\nDone."}, 1},
	} {
		if got := c.a.Probability(); math.Abs(got-c.want) > eps {
			t.Errorf("%s: Probability = %v, want %v", name, got, c.want)
		}
	}
}

// TestConfidenceEnvelopeMin: the envelope reports the min of the head's
// score and the decode probability, and suppresses below 0.1 on it
// (docs/FINDINGS.md F-062).
func TestConfidenceEnvelopeMin(t *testing.T) {
	n, err := Normalise([]byte(`[{"name":"lock_door","description":"Lock a door","parameters":{"type":"object",` +
		`"properties":{"door":{"type":"string"}},"required":["door"]}}]`))
	if err != nil {
		t.Fatal(err)
	}

	call := &Decoded{
		Text: thinkOpen + "r\n</think>\n" +
			`<tool_call>[{"name":"lock_door","arguments":{"door":"front"}}]</tool_call>`,
		Tokens:  []int{10, 11, 12, 13, 14, 15, 16},
		Probs:   []float64{1, 1, 0.9, 0.9, 0.9, 0.05, 0.9},
		Allowed: []float64{1, 1, 0.9, 0.9, 0.9, 0.08, 0.9},
		block:   4,
	}
	respond := &Decoded{
		Text: thinkOpen + "r\n</think>\nDone.", Tokens: []int{10, 11, 12, 13},
		Probs: []float64{1, 1, 0.25, 0.25}, Allowed: []float64{1, 1, 0.25, 0.25}, block: -1,
	}

	for name, c := range map[string]struct {
		a          *Decoded
		head       float64
		want       float64
		suppressed int
	}{
		"call, the decode probability binds and suppresses": {call, 0.9, 0.08, 1},
		"call, the head binds":                              {call, 0.05, 0.05, 1},
		"respond, the decode probability binds":             {respond, 0.9, 0.5, 0},
		"respond, the head binds":                           {respond, 0.3, 0.3, 0},
	} {
		e, err := n.Envelope(c.a, "lock the front door", "", nil, c.head)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		if math.Abs(e.Confidence-c.want) > 1e-12 || len(e.SuppressedCalls) != c.suppressed {
			t.Errorf("%s: confidence %v, %d suppressed; want %v and %d", name, e.Confidence, len(e.SuppressedCalls),
				c.want, c.suppressed)
		}
	}
}

func TestConfidenceSoftmaxAmong(t *testing.T) {
	row := []float32{0, float32(math.Log(3)), 5, float32(math.Log(6))}

	if got, want := softmaxAmong(row, 1, []int{0, 1, 3}), 0.3; math.Abs(got-want) > 1e-6 {
		t.Errorf("softmaxAmong among three = %v, want %v", got, want)
	}

	if got, want := softmaxAmong(row, 2, nil), softmaxAt(row, 2); got != want {
		t.Errorf("softmaxAmong over all = %v, want softmaxAt's %v", got, want)
	}
}

const engineConfidence = "../../testdata/text/engine/confidence.json"

// confidenceProbesOff are the confidence probe turns decoded as the engine
// does (type, reasoning and calls) whose confidence is still more than 0.02
// from the engine's in float32, and why (docs/FINDINGS.md F-062). Keys are
// "probe/turn"; "the head" is the ConfidenceHead's score off by the amount
// given where it, not the decode probability, sets the confidence, and the
// amount under the engine's int8 numerics follows. The engine's 0.0000 and
// the decimal for an integer are F-062's Open.
var confidenceProbesOff = map[string]string{
	"chat_home_03/0":               "the head +0.026 (int8 +0.028)",
	"result_timer_5/1":             "respond: the geometric mean +0.036 (int8 +0.035)",
	"result_weather_0/0":           "the head +0.026 (int8 within 0.02)",
	"result_weather_1/0":           "the head +0.026 (int8 within 0.02)",
	"result_weather_2/0":           "the head +0.026 (int8 within 0.02)",
	"result_weather_3/0":           "the head +0.026 (int8 within 0.02)",
	"result_weather_4/0":           "the head +0.026 (int8 within 0.02)",
	"result_event_0/0":             "the head -0.026 (int8 -0.043)",
	"result_event_1/0":             "the head -0.026 (int8 -0.043)",
	"result_more_alarm_7/1":        "respond: the geometric mean -0.029 (int8 -0.039)",
	"result_more_event_lunch/0":    "the head -0.038 (int8 within 0.02)",
	"result_more_lights_on/0":      "the head -0.023 (int8 within 0.02)",
	"result_more_lights_bedroom/0": "the head +0.032 (int8 within 0.02)",
	"result_more_navigate/1":       "respond: the geometric mean +0.029 (int8 +0.027)",
	"followup_weather_wear/0":      "the head +0.026 (int8 within 0.02)",
	"object_table_3/0":             "the engine's 0.0000 on an object argument, open (int8 +0.412)",
	"flat_parcel_1/0":              "the head +0.020 (int8 decodes otherwise)",
	"object_vacuum_1/0":            "the head -0.035 (int8 within 0.02)",
	"flat_vacuum_2/0":              "the head +0.043 (int8 +0.044)",
	"flat_vacuum_opt_0/0":          "the head +0.041 (int8 decodes otherwise)",
	"object_cart_3/0":              "the head -0.081 (int8 within 0.02)",
	"decimal_timer_0/0":            "a decimal for an integer: the closing token binds at 0.694, the engine's grammar lets the fraction through (int8 -0.222)",
	"decimal_timer_2/0":            "a decimal for an integer: the closing token binds at 0.592, the engine's grammar lets the fraction through (int8 -0.235)",
	"enum_fan_2/0":                 "the head +0.081 (int8 +0.076)",
	"guess_assistant_05/0":         "the head +0.096 (int8 -0.111)",
	"event_00/0":                   "the head -0.026 (int8 -0.043)",
	"event_09/0":                   "the head -0.038 (int8 within 0.02)",
	"catalogue_00/0":               "the head +0.027 (int8 +0.067)",
	"catalogue_14/0":               "the head -0.030 (int8 within 0.02)",
}

// TestConfidenceMatchesEngine decodes every confidence probe of the engine
// (testdata/text/engine/confidence.json, `just oracle-text confidence`)
// with the dequantised needle3.cact under the grammar, turn by turn, and
// compares the envelope's confidence, the min of the head's score and the
// decode probability (docs/FINDINGS.md F-062): within 0.02 of the engine's
// on every turn decoded as the engine does (type, reasoning and calls,
// dispatched and suppressed) but the turns listed, each of which must
// still be off. Turns decoded otherwise are logged, not compared.
func TestConfidenceMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the confidence probes")
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
			ID       string           `json:"id"`
			Question string           `json:"question"`
			Tools    string           `json:"tools"`
			Turns    []string         `json:"turns"`
			Answers  []engineEnvelope `json:"answers"`
		} `json:"probes"`
	}

	readJSON(t, engineConfidence, &fx)

	var within, decoded, total int

	for _, pr := range fx.Probes {
		turns := decodeTurns(t, d, m, p, fx.System, []byte(pr.Tools), pr.Turns, 512)

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

			same := got.Type == want.Type && reflect.DeepEqual(got.Reasoning, want.Reasoning) &&
				diffCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...)) == ""
			off := math.Abs(got.Confidence - want.Confidence)

			t.Logf("%-28s %-9s decoded as the engine %-5v head %.4f decode %.4f confidence %.4f engine %.4f",
				key, pr.Question, same, turn.conf, turn.a.Probability(), got.Confidence, want.Confidence)

			if !same {
				continue
			}

			decoded++

			if off <= 0.02 {
				within++
			}

			if why, isListed := confidenceProbesOff[key]; isListed == (off <= 0.02) {
				listedf(t, "%s: confidence %.4f, engine %.4f (listed: %q)", key, got.Confidence, want.Confidence, why)
			}
		}
	}

	logCount(t, "decoded as the engine's", decoded, total)
	int8Floor(t, "confidence within 0.02 of the turns decoded as the engine's", within, decoded, int8ConfidenceWithin)
}
