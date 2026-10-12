package toolcall

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

const engineRepair = "../../testdata/text/engine/repair.json"

// repairCase is one model-free case of a repair rule: the model's call block
// for a request over tools with the system facts, and the calls after the
// repair.
type repairCase struct {
	name, tools, input, system, block, want string
}

// dateFact is the corpus's date: fact, a Sunday.
const dateFact = "date: 2026-10-11 Sun 09:30"

// runRepair repairs each case's calls and compares them with want.
func runRepair(t *testing.T, cases []repairCase) {
	t.Helper()

	for _, c := range cases {
		n, err := Normalise([]byte(c.tools))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		calls, err := parseBlock(c.block)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		var b strings.Builder

		b.WriteByte('[')

		for i, rc := range n.repair(calls, c.input, c.system) {
			if i > 0 {
				b.WriteByte(',')
			}

			b.WriteString(`{"name":"` + rc.name + `","arguments":` + string(rc.args) + `}`)
		}

		b.WriteByte(']')

		if b.String() != c.want {
			t.Errorf("%s:\n got %s\nwant %s", c.name, b.String(), c.want)
		}
	}
}

// oneTool is a tool JSON array of one tool with the given properties and
// required keys (JSON fragments).
func oneTool(name, properties, required string) string {
	return `[{"name":"` + name + `","parameters":{"type":"object","properties":{` + properties +
		`},"required":[` + required + `]}}]`
}

func TestRepairCapitalise(t *testing.T) {
	note := func(key string) string { return oneTool("save_note", `"`+key+`":{"type":"string"}`, `"`+key+`"`) }
	call := func(key, v string) string { return `[{"name":"save_note","arguments":{"` + key + `":"` + v + `"}}]` }

	cases := make([]repairCase, 0, 19)

	for _, key := range []string{"label", "title", "message", "subject", "note", "event_title", "event_name"} {
		cases = append(cases, repairCase{key, note(key), "x", "", call(key, "buy milk"), call(key, "Buy milk")})
	}

	for _, key := range []string{"body", "text", "name", "description", "title_text", "task_title", "email_subject"} {
		cases = append(cases, repairCase{key + " not", note(key), "x", "", call(key, "buy milk"), call(key, "buy milk")})
	}

	cases = append(cases,
		repairCase{"camelCase key", note("eventTitle"), "x", "", call("event_title", "buy milk"), call("event_title", "Buy milk")},
		repairCase{"non-ASCII first", note("title"), "x", "", call("title", "über alles"), call("title", "über alles")},
		repairCase{"digit first", note("title"), "x", "", call("title", "3 eggs"), call("title", "3 eggs")},
		repairCase{"first byte only", note("title"), "x", "", call("title", "iPhone case"), call("title", "IPhone case")},
		repairCase{
			"nested", oneTool("save_notes", `"meta":{"type":"object","properties":{"title":{"type":"string"}}}`, ""),
			"x", "", `[{"name":"save_notes","arguments":{"meta":{"title":"errands"}}}]`,
			`[{"name":"save_notes","arguments":{"meta":{"title":"errands"}}}]`,
		},
	)

	runRepair(t, cases)
}

func TestRepairDate(t *testing.T) {
	weather := oneTool("get_weather", `"location":{"type":"string"},"date":{"type":"string","format":"date"},`+
		`"day":{"type":"string"},"when":{"type":"string"}`, `"location"`)
	call := func(key, v string) string {
		return `[{"name":"get_weather","arguments":{"location":"Berlin","` + key + `":"` + v + `"}}]`
	}

	cases := []repairCase{
		{"weekday", weather, "weather on Friday", dateFact, call("date", "2026-10-13"), call("date", "2026-10-16")},
		{"the fact's weekday", weather, "weather on Sunday", dateFact, call("date", "2026-10-13"), call("date", "2026-10-18")},
		{"the next day", weather, "weather on monday", dateFact, call("date", "2026-10-13"), call("date", "2026-10-12")},
		{"next Friday", weather, "weather next Friday", dateFact, call("date", "2026-10-14"), call("date", "2026-10-16")},
		{"fri", weather, "weather on fri", dateFact, call("date", "2026-10-12"), call("date", "2026-10-12")},
		{"month day past", weather, "weather on March 4th", dateFact, call("date", "2026-03-04"), call("date", "2027-03-04")},
		{
			"month day ahead", weather, "weather on March 4th", "date: 2026-02-01 Sun 09:30", call("date", "2026-03-04"),
			call("date", "2026-03-04"),
		},
		{"the fact's day", weather, "weather on October 11th", dateFact, call("date", "2026-10-11"), call("date", "2026-10-11")},
		{"the day before", weather, "weather on October 10th", dateFact, call("date", "2026-10-10"), call("date", "2027-10-10")},
		{"day month", weather, "weather on 4 March", dateFact, call("date", "2026-03-04"), call("date", "2027-03-04")},
		{"day of month", weather, "on the 4th of March", dateFact, call("date", "2026-03-04"), call("date", "2027-03-04")},
		{"explicit year", weather, "weather on March 4th 2026", dateFact, call("date", "2027-03-04"), call("date", "2026-03-04")},
		{"no date format needed", weather, "weather on Friday", dateFact, call("day", "2026-10-13"), call("day", "2026-10-16")},
		{"no date value", weather, "weather on Friday", dateFact, call("day", "Friday"), call("day", "Friday")},
		{
			"a date and time", weather, "on Friday at 9:00", dateFact, call("when", "2026-10-03 09:00"),
			call("when", "2026-10-03 09:00"),
		},
		{"no date fact", weather, "weather on Friday", "locale: en-US", call("date", "2026-10-13"), call("date", "2026-10-13")},
		{"two dates", weather, "Friday or March 4th", dateFact, call("date", "2026-10-13"), call("date", "2026-10-13")},
		{"no such day", weather, "on February 30th", dateFact, call("date", "2027-03-02"), call("date", "2027-03-02")},
	}

	runRepair(t, cases)
}

func TestRepairNameSplit(t *testing.T) {
	pair := func(first, last string) string {
		return oneTool("add_contact", `"`+first+`":{"type":"string"},"`+last+`":{"type":"string"}`, `"`+first+`"`)
	}
	call := func(args string) string { return `[{"name":"add_contact","arguments":{` + args + `}}]` }

	cases := []repairCase{
		{"given_name", pair("given_name", "family_name"), "x", "", call(`"given_name":"Ravi Kumar"`), call(`"given_name":"Ravi"`)},
		{
			"family_name given", pair("given_name", "family_name"), "x", "",
			call(`"given_name":"Ravi Kumar","family_name":"Kumar"`), call(`"given_name":"Ravi","family_name":"Kumar"`),
		},
		{"first", pair("first", "last"), "x", "", call(`"first":"Ravi Kumar"`), call(`"first":"Ravi"`)},
		{"first_name", pair("first_name", "last_name"), "x", "", call(`"first_name":"Ravi Kumar"`), call(`"first_name":"Ravi"`)},
		{"one name", pair("first_name", "last_name"), "x", "", call(`"first_name":"Ravi"`), call(`"first_name":"Ravi"`)},
		{
			"no pair", oneTool("add_contact", `"first_name":{"type":"string"}`, `"first_name"`), "x", "",
			call(`"first_name":"Ravi Kumar"`), call(`"first_name":"Ravi Kumar"`),
		},
	}

	runRepair(t, cases)
}

func TestRepairPhone(t *testing.T) {
	tool := oneTool("call_number", `"phone":{"type":"string"},"code":{"type":"string"}`, `"phone"`)
	call := func(v string) string { return `{"name":"call_number","arguments":{"phone":"` + v + `"}}` }
	both := `[{"name":"call_number","arguments":{"phone":"0171 2345678","code":"0171"}}]`

	cases := []repairCase{
		{
			"truncated", tool, "call 001 415 555 0162 9876", "", "[" + call("001 415 555 0162") + "]",
			"[" + call("001 415 555 0162 9876") + "]",
		},
		{
			"split, then a duplicate", tool, "call +44 (0) 20 7946 0958 1234", "",
			"[" + call("+44") + "," + call("20 7946 0958 1234") + "]", "[" + call("+44 (0) 20 7946 0958 1234") + "]",
		},
		{"whole", tool, "call 0171 2345678", "", "[" + call("0171 2345678") + "]", "[" + call("0171 2345678") + "]"},
		{"no phone key", tool, "call 0171 2345678", "", both, both},
		{"too short for a number", tool, "call 12 34", "", "[" + call("12") + "]", "[" + call("12") + "]"},
	}

	runRepair(t, cases)
}

func TestRepairQuoted(t *testing.T) {
	tool := oneTool("create_note", `"title":{"type":"string"}`, `"title"`)
	call := func(v string) string { return `[{"name":"create_note","arguments":{"title":"` + v + `"}}]` }

	cases := []repairCase{
		{
			"cut short, then capitalised", tool, `new note "the quick brown fox jumps again and again"`, "",
			call("the quick brown fox jumps again"), call("The quick brown fox jumps again and again"),
		},
		{"single quotes", tool, `a note titled 'Ode to a Nightingale'`, "", call("Ode to a"), call("Ode to a Nightingale")},
		{"in full", tool, `a note titled "R&D sync"`, "", call("R&D sync"), call("R&D sync")},
		{"not a word prefix", tool, `a note titled "Anna's plan"`, "", call("Anna"), call("Anna")},
		{"no quotes", tool, `a note titled the quick brown fox`, "", call("the quick"), call("The quick")},
		{"an apostrophe is no quote", tool, `Gino's note and Ann's`, "", call("note"), call("Note")},
	}

	runRepair(t, cases)
}

func TestRepairPlace(t *testing.T) {
	tool := oneTool("search_places", `"query":{"type":"string"}`, `"query"`)
	call := func(v string) string { return `[{"name":"search_places","arguments":{"query":"` + v + `"}}]` }

	cases := []repairCase{
		{
			"located at", tool, "look up Cafe Central located at 14 Herrengasse", "", call("Cafe Central"),
			call("Cafe Central 14 Herrengasse"),
		},
		{
			"which is at", tool, "find Mama's kitchen which is at 22 Oak Avenue, please", "", call("Mama's kitchen"),
			call("Mama's kitchen 22 Oak Avenue"),
		},
		{"which is located at", tool, "find the bakery which is located at 12 Elm Street", "", call("bakery"), call("bakery")},
		{"no street number", tool, "look up Cafe Central located at the corner", "", call("Cafe Central"), call("Cafe Central")},
		{"not after the value", tool, "look up the cafe located at 14 Herrengasse", "", call("Cafe Central"), call("Cafe Central")},
	}

	runRepair(t, cases)
}

func TestRepairPolarity(t *testing.T) {
	home := oneTool("set_lights", `"room":{"type":"string"},"on":{"type":"boolean"}`, `"room","on"`)
	lights := func(room, on string) string {
		return `[{"name":"set_lights","arguments":{"room":"` + room + `","on":` + on + `}}]`
	}
	volume := oneTool("change_volume", `"direction":{"type":"string","enum":["up","down"]},"amount":{"type":"integer"}`,
		`"direction","amount"`)
	vol := func(dir string) string {
		return `[{"name":"change_volume","arguments":{"direction":"` + dir + `","amount":4}}]`
	}
	power := oneTool("set_power", `"device":{"type":"string"},"state":{"type":"string","enum":["on","off"]}`, `"device"`)

	cases := []repairCase{
		{
			"turn on in the call's clause", home, "turn on the living room lights but don't unlock the front door", "",
			lights("living room", "false"), lights("living room", "true"),
		},
		{"turn off", home, "turn off the hall lights", "", lights("hall", "true"), lights("hall", "false")},
		{"negated: kept", home, "don't turn off the bedroom lights", "", lights("bedroom", "false"), lights("bedroom", "false")},
		{
			"another clause: kept", home, "turn off the hall lights and dim the kitchen lights", "",
			lights("kitchen", "true"), lights("kitchen", "true"),
		},
		{"no verb: kept", home, "the kitchen lights to 20", "", lights("kitchen", "false"), lights("kitchen", "false")},
		{"louder", volume, "make it louder by 4", "", vol("down"), vol("up")},
		{"down", volume, "turn it down", "", vol("up"), vol("down")},
		{"neither", volume, "change the volume by 4", "", vol("down"), vol("down")},
		{
			"an on/off enum", power, "switch on the fan", "", `[{"name":"set_power","arguments":{"device":"fan","state":"off"}}]`,
			`[{"name":"set_power","arguments":{"device":"fan","state":"on"}}]`,
		},
	}

	runRepair(t, cases)
}

func TestRepairPairedTool(t *testing.T) {
	tools := `[{"name":"lock_door","parameters":{"type":"object","properties":{"door":{"type":"string"}}}},` +
		`{"name":"unlock_door","parameters":{"type":"object","properties":{"door":{"type":"string"}}}},` +
		`{"name":"set_lights","parameters":{"type":"object","properties":{"room":{"type":"string"}}}}]`
	call := func(name, door string) string { return `{"name":"` + name + `","arguments":{"door":"` + door + `"}}` }
	porch := `{"name":"set_lights","arguments":{"room":"porch"}}`
	safe := `[{"name":"lock_safe","arguments":{"safe":"safe"}}]`

	cases := []repairCase{
		{
			"unlock", tools, "unlock the front door", "", "[" + call("lock_door", "front door") + "]",
			"[" + call("unlock_door", "front door") + "]",
		},
		{
			"the call's clause", tools, "turn on the porch lights and unlock the back door", "",
			"[" + porch + "," + call("lock_door", "back door") + "]", "[" + porch + "," + call("unlock_door", "back door") + "]",
		},
		{
			"lock", tools, "lock the front door", "", "[" + call("unlock_door", "front door") + "]",
			"[" + call("lock_door", "front door") + "]",
		},
		{
			"negated: kept", tools, "turn on the lights but don't unlock the front door", "",
			"[" + call("lock_door", "front door") + "]", "[" + call("lock_door", "front door") + "]",
		},
		{"no pair", oneTool("lock_safe", `"safe":{"type":"string"}`, ""), "unlock the safe", "", safe, safe},
	}

	runRepair(t, cases)
}

func TestRepairEnumNamed(t *testing.T) {
	fan := oneTool("set_fan", `"speed":{"type":"string","enum":["low","medium","high"]}`, `"speed"`)
	call := func(v string) string { return `[{"name":"set_fan","arguments":{"speed":"` + v + `"}}]` }

	cases := []repairCase{
		{"one named", fan, "fan on high and oscillating", "", call("low"), call("high")},
		{"case-insensitive", fan, "fan LOW", "", call("medium"), call("low")},
		{"two named", fan, "set the fan to medium or high", "", call("high"), call("high")},
		{"none named", fan, "turn the fan up to max", "", call("high"), call("high")},
		{"a longer word", fan, "the fan is too highly set", "", call("low"), call("low")},
	}

	runRepair(t, cases)
}

func TestRepairFromTo(t *testing.T) {
	nav := oneTool("navigate", `"destination":{"type":"string"},"origin":{"type":"string"}`, `"destination"`)
	ride := oneTool("book_ride", `"pickup":{"type":"string"},"dropoff":{"type":"string"}`, "")
	office := `[{"name":"navigate","arguments":{"destination":"the office"}}]`
	zoo := `[{"name":"book_ride","arguments":{"pickup":"the zoo","dropoff":"the beach"}}]`

	cases := []repairCase{
		{
			"a missing origin", nav, "navigate from home to the office", "", office,
			`[{"name":"navigate","arguments":{"destination":"the office","origin":"home"}}]`,
		},
		{
			"without the article", nav, "directions from the hotel to the museum, please", "",
			`[{"name":"navigate","arguments":{"destination":"the museum"}}]`,
			`[{"name":"navigate","arguments":{"destination":"the museum","origin":"hotel"}}]`,
		},
		{
			"both missing", ride, "a taxi from the zoo to the beach", "", `[{"name":"book_ride","arguments":{}}]`,
			`[{"name":"book_ride","arguments":{"pickup":"zoo","dropoff":"beach"}}]`,
		},
		{"given: kept", ride, "a taxi from the zoo to the beach", "", zoo, zoo},
		{"no from", nav, "navigate to the office", "", office, office},
	}

	runRepair(t, cases)
}

func TestRepairByN(t *testing.T) {
	temp := oneTool("adjust_temperature", `"degrees":{"type":"number"}`, `"degrees"`)
	call := func(v string) string { return `[{"name":"adjust_temperature","arguments":{"degrees":` + v + `}}]` }
	two := oneTool("set_volume", `"from":{"type":"integer"},"by":{"type":"integer"}`, "")
	twoCall := `[{"name":"set_volume","arguments":{"from":20,"by":20}}]`
	steps := oneTool("set_volume", `"steps":{"type":"integer"}`, "")
	stepsCall := `[{"name":"set_volume","arguments":{"steps":2}}]`

	cases := []repairCase{
		{"by 2 to 23", temp, "raise the temperature by 2 to 23 degrees", "", call("23"), call("2")},
		{"equal: kept", temp, "warm it up by 3 degrees, it's 19 now", "", call("3.0"), call("3.0")},
		{"no by", temp, "set it to 23 degrees", "", call("23"), call("23")},
		{"two number arguments", two, "turn it from 20 up by 5", "", twoCall, twoCall},
		{"a fraction for an integer", steps, "turn it up by 2.5", "", stepsCall, stepsCall},
	}

	runRepair(t, cases)
}

func TestRepairDefaultNumber(t *testing.T) {
	timer := oneTool("set_timer", `"minutes":{"type":"integer","default":7},"label":{"type":"string"}`, `"minutes"`)
	call := func(v string) string { return `[{"name":"set_timer","arguments":{"minutes":` + v + `}}]` }
	optional := oneTool("set_timer", `"minutes":{"type":"integer","default":7}`, "")

	cases := []repairCase{
		{"no quantity", timer, "set a timer for the pasta", "", call("15"), call("7")},
		{"digits", timer, "set a timer for 3 minutes", "", call("3"), call("3")},
		{"a number word", timer, "set a timer for five minutes", "", call("5"), call("5")},
		{"optional: no default, dropped unstated", optional, "set a timer", "", call("15"), `[{"name":"set_timer","arguments":{}}]`},
		{
			"a boolean property schema", oneTool("set_timer", `"off":false,"minutes":{"type":"integer","default":7}`,
				`"minutes"`), "set a timer for the pasta", "", call("15"), call("7"),
		},
	}

	runRepair(t, cases)
}

func TestRepairDropArguments(t *testing.T) {
	lights := oneTool("set_lights", `"room":{"type":"string"},"on":{"type":"boolean"},"brightness":{"type":"integer"},`+
		`"mode":{"type":"integer","enum":[1,2]},"scene":{"type":"string"},"level":{"type":"number"}`, `"room","on"`)
	call := func(args string) string {
		return `[{"name":"set_lights","arguments":{"room":"kitchen","on":false` + args + `}}]`
	}

	cases := []repairCase{
		{"an unstated optional number", lights, "set the kitchen lights to 150 percent", "", call(`,"brightness":100`), call("")},
		{"0 unstated", lights, "the kitchen lights", "", call(`,"brightness":0`), call("")},
		{"stated", lights, "set the kitchen lights to 40 percent", "", call(`,"brightness":40`), call(`,"brightness":40`)},
		{"a number word", lights, "dim the kitchen lights to fifty percent", "", call(`,"brightness":50`), call(`,"brightness":50`)},
		{"compound", lights, "the kitchen lights to twenty-five", "", call(`,"brightness":25`), call(`,"brightness":25`)},
		{"half", lights, "the kitchen lights to half brightness", "", call(`,"brightness":50`), call(`,"brightness":50`)},
		{"half a unit", lights, "the kitchen lights to level half", "", call(`,"level":0.5`), call(`,"level":0.5`)},
		{"an enum", lights, "the kitchen lights", "", call(`,"mode":2`), call(`,"mode":2`)},
		{"an empty optional string", lights, "the kitchen lights", "", call(`,"scene":""`), call("")},
		{"an unstated optional string: kept", lights, "the kitchen lights", "", call(`,"scene":"cosy"`), call(`,"scene":"cosy"`)},
		{
			"required: kept", lights, "the lights", "", `[{"name":"set_lights","arguments":{"room":"","on":false,"brightness":7}}]`,
			`[{"name":"set_lights","arguments":{"room":"","on":false}}]`,
		},
		{
			"required in an allOf branch: kept",
			`[{"name":"set_lights","parameters":{"type":"object","properties":{"room":{"type":"string"},` +
				`"brightness":{"type":"integer"}},"allOf":[{"required":["room","brightness"]}]}}]`,
			"the lights", "", `[{"name":"set_lights","arguments":{"room":"","brightness":7}}]`,
			`[{"name":"set_lights","arguments":{"room":"","brightness":7}}]`,
		},
	}

	runRepair(t, cases)
}

func TestRepairDropCalls(t *testing.T) {
	home := `[{"name":"set_lights","parameters":{"type":"object","properties":{"room":{"type":"string"},"on":` +
		`{"type":"boolean"}}}},{"name":"lock_door","parameters":{"type":"object","properties":{"door":{"type":"string"}}}},` +
		`{"name":"play_music","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}]`
	lights := func(room string) string {
		return `{"name":"set_lights","arguments":{"room":"` + room + `","on":false}}`
	}
	lock := func(door string) string { return `{"name":"lock_door","arguments":{"door":"` + door + `"}}` }
	list := func(calls ...string) string { return "[" + strings.Join(calls, ",") + "]" }
	jazz := `{"name":"play_music","arguments":{"query":"jazz"}}`

	cases := []repairCase{
		{
			"a duplicate", home, "turn off the kitchen lights and turn off the kitchen lights", "",
			list(lights("kitchen"), lights("kitchen")), list(lights("kitchen")),
		},
		{
			"different: kept", home, "the kitchen and the bedroom lights", "", list(lights("kitchen"), lights("bedroom")),
			list(lights("kitchen"), lights("bedroom")),
		},
		{
			"except", home, "turn off the lights in the kitchen, the bedroom and the hallway except the hallway", "",
			list(lights("kitchen"), lights("bedroom"), lights("hallway")), list(lights("kitchen"), lights("bedroom")),
		},
		{
			"but not", home, "turn off the lights in the kitchen and the hallway but not the hallway", "",
			list(lights("kitchen"), lights("hallway")), list(lights("kitchen")),
		},
		{
			"apart from", home, "lock the front door, the back door and the garage, apart from the garage", "",
			list(lock("front door"), lock("back door"), lock("garage")), list(lock("front door"), lock("back door")),
		},
		{
			"the only call: kept", home, "the lights everywhere except the bedroom", "", list(lights("bedroom")),
			list(lights("bedroom")),
		},
		{
			"inside a value: kept", home, "lock all the doors but not the garage", "",
			list(lock("front door"), lock("garage door")), list(lock("front door"), lock("garage door")),
		},
		{
			"a negated verb", home, "play jazz in the kitchen and don't play anything in the bedroom", "",
			list(jazz, lights("bedroom")), list(lights("bedroom")),
		},
		{
			"don't lock", home, "turn off the kitchen lights and don't lock the front door", "",
			list(lights("kitchen"), lock("front door")), list(lights("kitchen")),
		},
		{
			"another verb: kept", home, "turn on the kitchen lights but never unlock the garage", "", list(lock("garage")),
			list(lock("garage")),
		},
	}

	runRepair(t, cases)
}

// TestWordNumber covers the number words that state an optional number.
func TestWordNumber(t *testing.T) {
	for _, c := range []struct {
		s    string
		v    float64
		want bool
	}{
		{"fifty percent", 50, true},
		{"twenty-five", 25, true},
		{"twenty five", 25, true},
		{"one hundred", 100, true},
		{"a hundred", 100, true},
		{"half", 50, true},
		{"half", 0.5, true},
		{"a dozen eggs", 12, true},
		{"fifty", 15, false},
		{"fifteen", 50, false},
		{"none", 0, false},
	} {
		if got := wordNumber(c.s, c.v); got != c.want {
			t.Errorf("wordNumber(%q, %v) = %v", c.s, c.v, got)
		}
	}
}

// TestClauses covers the clause split the repair reads verbs in.
func TestClauses(t *testing.T) {
	got := clauses("turn on the lights, lock it and then play jazz but don't stop; Dr. Who at 21.5. Done")
	want := []string{"turn on the lights", "lock it", "play jazz", "don't stop", "Dr", "Who at 21.5", "Done"}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("clauses = %q, want %q", got, want)
	}
}

// repairTurnDiffers are the repair probes whose calls differ from the
// engine's after the repair, and why (docs/FINDINGS.md F-056): rules of the
// engine the probes show but do not identify.
var repairTurnDiffers = map[string]string{
	"bool_other_clause": "the engine turns the kitchen lights off after an unlock in another clause, " +
		"not identified (bool_other_clause_after keeps them on)",
	"enum_named_high":  "the engine adds oscillate: true to the second call, not identified; ours are then duplicates",
	"enum_named_low":   "the engine adds oscillate: true to the second call, not identified; ours are then duplicates",
	"required_default": "the engine replaces the shed the request names by the door's default, not identified",
	"required_default_modifier": "the engine replaces the kitchen of \"the kitchen lights\" by the room's default " +
		"(\"in the kitchen\" keeps it), not identified",
}

// TestRepairMatchesEngine decodes every repair probe of the engine
// (testdata/text/engine/repair.json, `just oracle-text repair`) with the
// dequantised needle3.cact under the grammar and builds its envelope with
// the repair: the calls (dispatched or suppressed) are the engine's but on
// the probes listed, which the test checks still differ
// (docs/FINDINGS.md F-056).
func TestRepairMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the repair probes")
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
			ID     string         `json:"id"`
			Rule   string         `json:"rule"`
			Tools  string         `json:"tools"`
			System string         `json:"system"`
			Query  string         `json:"query"`
			Answer engineEnvelope `json:"answer"`
		} `json:"probes"`
	}

	readJSON(t, engineRepair, &fx)

	equal, reasoning := 0, 0

	for _, pr := range fx.Probes {
		system := fx.System
		if pr.System != "" {
			system = pr.System
		}

		turn := decodeTurns(t, d, m, p, system, []byte(pr.Tools), []string{pr.Query}, 512)[0]

		js, err := json.Marshal(turn.env)
		if err != nil {
			t.Fatal(err)
		}

		var got engineEnvelope

		err = json.Unmarshal(js, &got)
		if err != nil {
			t.Fatal(err)
		}

		want := pr.Answer
		diff := diffCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...))
		think := reflect.DeepEqual(got.Reasoning, want.Reasoning)

		if diff == "" {
			equal++
		}

		if think {
			reasoning++
		}

		t.Logf("%-26s %-17s reasoning equal %-5v %q", pr.ID, pr.Rule, think, diff)

		if why, listed := repairTurnDiffers[pr.ID]; listed == (diff == "") {
			t.Errorf("%s: calls %q (listed: %q)", pr.ID, diff, why)
		}
	}

	t.Logf("%d probes: calls equal on %d, reasoning equal on %d", len(fx.Probes), equal, reasoning)
}
