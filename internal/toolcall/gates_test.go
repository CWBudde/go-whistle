package toolcall

import (
	"encoding/json"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const engineGates = "../../testdata/text/engine/gates.json"

// gatesDiffer are the gate probe turns whose calls (names and arguments) or
// whose split into dispatched and suppressed calls differ from the
// engine's, and why (docs/FINDINGS.md F-058). Keys are "probe/turn"; a
// margin is the smallest top-1 margin of the reasoning or, where the
// reasoning is the engine's, of the call block.
var gatesDiffer = map[string]string{
	// The decode differs from the engine's at a near-tie (float
	// activations against the engine's int8 ones, 11.5).
	"enum_key_setting/0":            "no call block after the engine's reasoning (min margin 0.129)",
	"control_there/0":               "reasoning at a near-tie (0.013); confidence 0.097 against 0.184",
	"control_longer_value/0":        "reasoning at a near-tie (0.041); confidence 0.029 against 0.102",
	"place_two_calls/0":             "reasoning at a near-tie (0.059); the engine's model invents Kenya",
	"place_door/0":                  "reasoning at a near-tie (0.012); the port's model writes door \"up\"",
	"quantity_hours/0":              "reasoning at a near-tie (0.035)",
	"quantity_hourly/0":             "reasoning at a near-tie (0.049)",
	"quantity_earlier_turn_other/1": "reasoning differs (min margin 0.292); the engine's model writes 20 minutes",
	"origin_optional_destination/0": "reasoning at a near-tie (0.054)",
	"polarity_never_13/0":           "reasoning at a near-tie (0.084)",
	// A repair outside the gates: the engine writes the request's
	// "hamburg" where the model writes the system facts' "Hamburg" at the
	// engine's reasoning (block margin 8.420); not identified.
	"place_named/0": "the request's spelling hamburg for the model's Hamburg, a repair not identified",
}

// TestGatesMatchesEngine decodes every gate probe of the engine
// (testdata/text/engine/gates.json, `just oracle-text gates`) with the
// dequantised needle3.cact under the grammar, turn by turn, and builds its
// envelope (docs/FINDINGS.md F-058): the calls, as parsed JSON, and their
// split into function_calls and suppressed_calls equal the engine's but on
// the turns listed, each of which must still differ.
func TestGatesMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the gate probes")
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
			Gate    string           `json:"gate"`
			Tools   string           `json:"tools"`
			System  string           `json:"system"`
			Turns   []string         `json:"turns"`
			Answers []engineEnvelope `json:"answers"`
		} `json:"probes"`
	}

	readJSON(t, engineGates, &fx)

	var equal, total int

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

			diff := diffCalls(got.Calls, want.Calls)
			if s := diffCalls(got.Suppressed, want.Suppressed); diff == "" && s != "" {
				diff = "suppressed: " + s
			}

			think := reflect.DeepEqual(got.Reasoning, want.Reasoning)

			block := slices.Index(turn.a.Tokens, d.open)
			if block < 0 {
				block = len(turn.a.Tokens)
			}

			t.Logf("%-34s %-12s reasoning equal %-5v min margin %.3f, in the block %.3f confidence %.4f engine %.4f "+
				"calls %d/%d engine %d/%d %q", key, pr.Gate, think, minMargin(turn.a, 0, block),
				minMargin(turn.a, block, len(turn.a.Tokens)), got.Confidence, want.Confidence, len(got.Calls),
				len(got.Suppressed), len(want.Calls), len(want.Suppressed), diff)

			if diff == "" {
				equal++
			}

			if why, listed := gatesDiffer[key]; listed == (diff == "") {
				t.Errorf("%s: calls %q (listed: %q)", key, diff, why)
			}
		}
	}

	t.Logf("%d turns: calls and split equal on %d", total, equal)
}

// gateCase is one model-free case of a grounding gate: the model's calls
// for a request over tools, with the conversation's earlier inputs, and
// how many calls the envelope dispatches and suppresses at confidence 0.9.
type gateCase struct {
	name, tools, input string
	earlier            []string
	calls              []string
	ship, held         int
}

// runGates builds each case's envelope and compares its split.
func runGates(t *testing.T, cases []gateCase) {
	t.Helper()

	for _, c := range cases {
		n, err := Normalise([]byte(c.tools))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		a := &Decoded{Text: thinkOpen + "r\n</think>\n<tool_call>[" + strings.Join(c.calls, ",") + "]</tool_call>"}

		e, err := n.Envelope(a, c.input, dateFact, c.earlier, 0.9)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if len(e.FunctionCalls) != c.ship || len(e.SuppressedCalls) != c.held {
			t.Errorf("%s: %d calls, %d suppressed; want %d, %d", c.name, len(e.FunctionCalls), len(e.SuppressedCalls),
				c.ship, c.held)
		}
	}
}

// withTrigger is the tools JSON of one tool with a trigger regex added.
func withTrigger(tools, re string) string {
	return strings.TrimSuffix(tools, "}]") + `,"triggers":["` + re + `"]}]`
}

// TestGateReportedQuote covers the contradiction gate of a command quoted
// as someone's report (docs/FINDINGS.md F-058).
func TestGateReportedQuote(t *testing.T) {
	lights := oneTool("set_lights", `"room":{"type":"string"},"on":{"type":"boolean"}`, `"room","on"`)
	off := `{"name":"set_lights","arguments":{"room":"kitchen","on":false}}`
	on := `{"name":"set_lights","arguments":{"room":"hallway","on":true}}`

	cases := []gateCase{
		{"said", lights, `my wife said "turn off the kitchen lights"`, nil, []string{off}, 0, 1},
		{"single quotes", lights, "my wife said 'turn off the kitchen lights'", nil, []string{off}, 0, 1},
		{"typographic quotes", lights, "my wife said “turn off the kitchen lights”", nil, []string{off}, 0, 1},
		{"told me", lights, `he told me "turn off the kitchen lights"`, nil, []string{off}, 0, 1},
		{"an apostrophe after it", lights, "my wife told me it's late, turn off the kitchen lights", nil, []string{off}, 0, 1},
		{"the speaker's own", lights, `I said "turn off the kitchen lights"`, nil, []string{off}, 0, 1},
		{"inside a word, in upper case", lights, `leave it UNSAID "turn off the kitchen lights"`, nil, []string{off}, 0, 1},
		{"a quote of something else", lights, `my wife said "hi", now turn off the kitchen lights`, nil, []string{off}, 0, 1},
		{
			"the whole answer", lights, `turn on the hallway lights, and my wife said "turn off the kitchen lights"`, nil,
			[]string{on, off},
			0, 2,
		},
		{
			"a trigger does not ship it", withTrigger(lights, "lights"), `my wife said "turn off the kitchen lights"`, nil,
			[]string{off},
			0, 1,
		},

		// Not withheld.
		{"says", lights, `my wife says "turn off the kitchen lights"`, nil, []string{off}, 1, 0},
		{"told us", lights, `my wife told us "turn off the kitchen lights"`, nil, []string{off}, 1, 0},
		{"no quote mark", lights, "my wife said to turn off the kitchen lights", nil, []string{off}, 1, 0},
		{"the quote before said", lights, `"turn off the kitchen lights", my wife said`, nil, []string{off}, 1, 0},
		{"backticks", lights, "my wife said `turn off the kitchen lights`", nil, []string{off}, 1, 0},
		{"a quote, no report", lights, `"turn off the kitchen lights"`, nil, []string{off}, 1, 0},
	}

	runGates(t, cases)
}

// TestGateUnnamedEnum covers the guess gate of a required enum filled with
// an option the conversation never names (docs/FINDINGS.md F-058).
func TestGateUnnamedEnum(t *testing.T) {
	fan := func(key string) string {
		return oneTool("set_fan", `"`+key+`":{"type":"string","enum":["low","medium","high"]}`, `"`+key+`"`)
	}
	call := func(key, v string) string { return `{"name":"set_fan","arguments":{"` + key + `":"` + v + `"}}` }
	optional := oneTool("set_fan", `"speed":{"type":"string","enum":["low","medium","high"]}`, "")
	defaulted := oneTool("set_fan", `"speed":{"type":"string","enum":["low","medium","high"],"default":"medium"}`,
		`"speed"`)
	lang := oneTool("compile", `"language":{"type":"string","enum":["C++","C#",".NET","Go"]}`, `"language"`)
	compile := func(v string) string { return `{"name":"compile","arguments":{"language":"` + v + `"}}` }

	cases := []gateCase{
		{"never named", fan("speed"), "set the fan", nil, []string{call("speed", "medium")}, 0, 1},
		{"a synonym", fan("speed"), "fan to max please", nil, []string{call("speed", "high")}, 0, 1},
		{"a short option at a word's start", fan("speed"), "fan lowest", nil, []string{call("speed", "low")}, 0, 1},
		{"inside a word", fan("speed"), "fan thigh", nil, []string{call("speed", "high")}, 0, 1},
		{"no polar word", fan("speed"), "set the downstairs fan", nil, []string{call("speed", "low")}, 0, 1},
		{"a default", defaulted, "set the fan", nil, []string{call("speed", "medium")}, 0, 1},
		{"a key without an exempt word", fan("modal"), "set the fan", nil, []string{call("modal", "medium")}, 0, 1},

		// Not withheld.
		{"named", fan("speed"), "set the fan to high", nil, []string{call("speed", "high")}, 1, 0},
		{"named in upper case", fan("speed"), "fan LOW", nil, []string{call("speed", "low")}, 1, 0},
		{"a word that starts with it", fan("speed"), "fan highway", nil, []string{call("speed", "high")}, 1, 0},
		{"a raising word", fan("speed"), "turn the fan up to max", nil, []string{call("speed", "high")}, 1, 0},
		{"a lowering word", fan("speed"), "make the fan quieter", nil, []string{call("speed", "low")}, 1, 0},
		{"an earlier turn names it", fan("speed"), "now the other one", []string{"fan on high"}, []string{call("speed", "high")}, 1, 0},
		{"optional", optional, "set the fan", nil, []string{call("speed", "medium")}, 1, 0},
		{"key mode", fan("mode"), "set the fan", nil, []string{call("mode", "medium")}, 1, 0},
		{"a key holding level", fan("speed_level"), "set the fan", nil, []string{call("speed_level", "medium")}, 1, 0},
		{"key units", fan("units"), "set the fan", nil, []string{call("units", "medium")}, 1, 0},
		// Options that start or end in punctuation are named as words too
		// (PR #69 review); "c" alone does not name "C++".
		{"named with punctuation", lang, "compile it as c++ please", nil, []string{compile("C++")}, 1, 0},
		{"named with a trailing #", lang, "build the C# project", nil, []string{compile("C#")}, 1, 0},
		{"named with a leading dot", lang, "target .NET", nil, []string{compile(".NET")}, 1, 0},
		{"a prefix of a punctuated option", lang, "compile it as c", nil, []string{compile("C++")}, 0, 1},
		{
			"a trigger ships it", withTrigger(fan("speed"), "fan"), "set the fan", nil,
			[]string{call("speed", "medium")},
			1, 0,
		},
	}

	runGates(t, cases)
}

// TestGateControlWord covers the guess gate of a required slot filled with
// a control word (docs/FINDINGS.md F-058).
func TestGateControlWord(t *testing.T) {
	music := oneTool("play_music", `"query":{"type":"string"},"room":{"type":"string"}`, `"query"`)
	call := func(args string) string { return `{"name":"play_music","arguments":{` + args + `}}` }
	power := oneTool("set_power", `"state":{"type":"string","enum":["on","off"]}`, `"state"`)

	cases := []gateCase{
		{"up", music, "music up", nil, []string{call(`"query":"up"`)}, 0, 1},
		{"in any case", music, "music Off", nil, []string{call(`"query":"Off"`)}, 0, 1},
		{"lock", music, "music lock", nil, []string{call(`"query":"lock"`)}, 0, 1},
		{"pause", music, "music pause", nil, []string{call(`"query":"pause"`)}, 0, 1},

		// Not withheld.
		{"it", music, "play it", nil, []string{call(`"query":"it"`)}, 1, 0},
		{"true", music, "music true", nil, []string{call(`"query":"true"`)}, 1, 0},
		{"inside the value", music, "music on", nil, []string{call(`"query":"music on"`)}, 1, 0},
		{"an optional slot", music, "jazz, room up", nil, []string{call(`"query":"jazz","room":"up"`)}, 1, 0},
		{
			"an enum option", power, "turn on the power", nil,
			[]string{`{"name":"set_power","arguments":{"state":"on"}}`},
			1, 0,
		},
		{"a trigger ships it", withTrigger(music, "music"), "music up", nil, []string{call(`"query":"up"`)}, 1, 0},
	}

	runGates(t, cases)
}

// TestGateUnmentionedPlace covers the guess gate of a required place slot
// filled with a place the conversation never mentions (docs/FINDINGS.md
// F-058).
func TestGateUnmentionedPlace(t *testing.T) {
	weather := func(key string) string { return oneTool("get_weather", `"`+key+`":{"type":"string"}`, `"`+key+`"`) }
	call := func(key, v string) string { return `{"name":"get_weather","arguments":{"` + key + `":"` + v + `"}}` }
	defaulted := oneTool("get_weather", `"location":{"type":"string","default":"Hamburg"}`, `"location"`)
	optional := oneTool("get_weather", `"location":{"type":"string"}`, "")

	cases := []gateCase{
		{"never mentioned", weather("location"), "what's the weather there", nil, []string{call("location", "Hamburg")}, 0, 1},
		{"a key holding city", weather("city_name"), "what's the weather there", nil, []string{call("city_name", "Hamburg")}, 0, 1},
		{"a room", weather("room"), "lights up", nil, []string{call("room", "living room")}, 0, 1},

		// Not withheld.
		{"named, in another case", weather("location"), "the weather in hamburg", nil, []string{call("location", "Hamburg")}, 1, 0},
		{
			"an earlier tool result names it", weather("location"), "what's the weather there",
			[]string{"find pizza", `[{"address": "14 Herrengasse"}]`},
			[]string{call("location", "14 Herrengasse")},
			1, 0,
		},
		{"key where", weather("where"), "what's the weather there", nil, []string{call("where", "Hamburg")}, 1, 0},
		{"key town", weather("town"), "what's the weather there", nil, []string{call("town", "Hamburg")}, 1, 0},
		{"the default", defaulted, "what's the weather there", nil, []string{call("location", "Hamburg")}, 1, 0},
		{"optional", optional, "what's the weather there", nil, []string{call("location", "Hamburg")}, 1, 0},
		{
			"a trigger ships it", withTrigger(weather("location"), "weather"), "what's the weather there", nil,
			[]string{call("location", "Hamburg")},
			1, 0,
		},
	}

	runGates(t, cases)
}

// TestGateMissingQuantity covers the guess gate of a required number with
// no default when the conversation names no quantity (docs/FINDINGS.md
// F-058).
func TestGateMissingQuantity(t *testing.T) {
	timer := oneTool("set_timer", `"minutes":{"type":"integer"},"label":{"type":"string"}`, `"minutes"`)
	call := `{"name":"set_timer","arguments":{"minutes":15,"label":"Pasta"}}`
	defaulted := oneTool("set_timer", `"minutes":{"type":"integer","default":15}`, `"minutes"`)
	optional := oneTool("set_timer", `"minutes":{"type":"integer"},"label":{"type":"string"}`, `"label"`)

	cases := []gateCase{
		{"no quantity", timer, "set a timer for the pasta", nil, []string{call}, 0, 1},
		{"a plural unit", timer, "set a timer for hours", nil, []string{call}, 0, 1},
		{"a pair", timer, "set a timer for a pair of minutes", nil, []string{call}, 0, 1},
		{"a number inside a word", timer, "set a timer for tennis", nil, []string{call}, 0, 1},
		{"twentyish", timer, "set a timer for twentyish minutes", nil, []string{call}, 0, 1},

		// Not withheld.
		{"digits", timer, "set a timer for 15 minutes", nil, []string{call}, 1, 0},
		{"a number word", timer, "set a timer for fifteen minutes", nil, []string{call}, 1, 0},
		{"a couple", timer, "set a timer for a couple of minutes", nil, []string{call}, 1, 0},
		{"an hour", timer, "set a timer for an hour", nil, []string{call}, 1, 0},
		{"a third", timer, "set a timer for a third", nil, []string{call}, 1, 0},
		{"an earlier turn", timer, "set the same timer for the rice", []string{"a timer for 15 minutes"}, []string{call}, 1, 0},
		{"a default", defaulted, "set a timer", nil, []string{`{"name":"set_timer","arguments":{"minutes":15}}`}, 1, 0},
		{"optional", optional, "set a timer for the pasta", nil, []string{call}, 1, 0},
		{"a trigger ships it", withTrigger(timer, "timer"), "set a timer for the pasta", nil, []string{call}, 1, 0},
	}

	runGates(t, cases)
}

// TestGateOriginIsDestination covers the guess gate of the destination
// copied into the origin (docs/FINDINGS.md F-058).
func TestGateOriginIsDestination(t *testing.T) {
	ride := func(a, b string) string {
		return oneTool("book_ride", `"`+a+`":{"type":"string"},"`+b+`":{"type":"string"}`, `"`+a+`","`+b+`"`)
	}
	call := func(a, av, b, bv string) string {
		return `{"name":"book_ride","arguments":{"` + a + `":"` + av + `","` + b + `":"` + bv + `"}}`
	}
	lights := `{"name":"set_lights","arguments":{"room":"kitchen","on":false}}`
	both := `[{"name":"book_ride","parameters":{"type":"object","properties":{"origin":{"type":"string"},` +
		`"destination":{"type":"string"}}}},{"name":"set_lights","parameters":{"type":"object","properties":` +
		`{"room":{"type":"string"},"on":{"type":"boolean"}}}}]`

	cases := []gateCase{
		{"origin, destination", ride("origin", "destination"), "a taxi to the airport", nil, []string{call("origin", "airport", "destination", "airport")}, 0, 1},
		{"from, to", ride("from", "to"), "a taxi to the airport", nil, []string{call("from", "airport", "to", "airport")}, 0, 1},
		{"pickup, dropoff", ride("pickup", "dropoff"), "a taxi to the airport", nil, []string{call("pickup", "airport", "dropoff", "airport")}, 0, 1},
		{"stated twice", ride("origin", "destination"), "from the airport to the airport", nil, []string{call("origin", "airport", "destination", "airport")}, 0, 1},
		{"the whole answer", both, "lights off and a taxi to the airport", nil, []string{lights, call("origin", "airport", "destination", "airport")}, 0, 2},

		// Not withheld.
		{"different", ride("origin", "destination"), "from the zoo to the beach", nil, []string{call("origin", "the zoo", "destination", "the beach")}, 1, 0},
		{"an article apart", ride("origin", "destination"), "to the station", nil, []string{call("origin", "station", "destination", "the station")}, 1, 0},
		{"pickup_location", ride("pickup_location", "dropoff_location"), "to the airport", nil, []string{call("pickup_location", "airport", "dropoff_location", "airport")}, 1, 0},
		{"location", ride("location", "destination"), "to the airport", nil, []string{call("location", "airport", "destination", "airport")}, 1, 0},
		{
			"a trigger ships it", withTrigger(ride("origin", "destination"), "taxi"), "a taxi to the airport", nil,
			[]string{call("origin", "airport", "destination", "airport")},
			1, 0,
		},
	}

	runGates(t, cases)
}

// TestNegatedBefore covers the window the paired tool and polarity read a
// negation in (docs/FINDINGS.md F-057, F-058).
func TestNegatedBefore(t *testing.T) {
	for _, c := range []struct {
		s, verb string
		want    bool
	}{
		{"don't unlock the front door", "unlock", true},
		{"don't delay, unlock the front door", "unlock", true},
		{"don't wait, unlock the front door", "unlock", true},
		{"not unlock the front door", "unlock", true},
		{"nothing to unlock the front door", "unlock", true},
		{"never mind, unlock the front door", "unlock", true},
		{"dontx unlock the front door", "unlock", true},
		{"DON'T really turn off the bedroom lights", "turn off", true},
		{"you must not unlock it", "unlock", true},

		// Not negated.
		{"don't linger, unlock the front door", "unlock", false},
		{"not tomorrow, unlock the front door", "unlock", false},
		{"don't forget to unlock the front door", "unlock", false},
		{"don't already turn on the bedroom lights", "turn on", false},
		{"nothing really turn on the bedroom lights", "turn on", false},
		{"you shouldn't unlock the front door", "unlock", false},
		{"you mustn't unlock the front door", "unlock", false},
		{"don’t unlock the front door", "unlock", false},
		{"kindly unlock the front door", "unlock", false},
		{"unlock the front door, don't wait", "unlock", false},
	} {
		if got := negatedBefore(c.s, strings.Index(c.s, c.verb)); got != c.want {
			t.Errorf("negatedBefore(%q, %q) = %v, want %v", c.s, c.verb, got, c.want)
		}
	}
}
