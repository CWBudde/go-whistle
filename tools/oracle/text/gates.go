//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// gateProbe is the engine's answer to one conversation that exercises a grounding gate, which withholds the calls in
// suppressed_calls whatever their confidence (upstream's confidence guide), or the polarity rule's reading of a
// negation: the gate, what the probe covers, the tools JSON as sent (with any triggers), the system facts when they
// are not the corpus's, the turns and the answer to each, without its timings.
type gateProbe struct {
	ID       string              `json:"id"`
	Gate     string              `json:"gate"`
	Covers   string              `json:"covers"`
	Tools    string              `json:"tools"`
	Triggers map[string][]string `json:"triggers,omitempty"`
	System   string              `json:"system,omitempty"`
	Turns    []string            `json:"turns"`
	Answers  []json.RawMessage   `json:"answers"`
}

// gatesFixture is testdata/text/engine/gates.json.
type gatesFixture struct {
	header

	System string      `json:"system"`
	Probes []gateProbe `json:"probes"`
}

// The gates of upstream's confidence guide the probes exercise ("the request … quotes a command reported by someone
// else, fills a required enum with an option it never names, fills a required slot with a control word or a place
// the conversation never mentions, names no quantity for a required number with no default, copies the destination
// into an origin"), and the polarity rule's negation window (F-057).
const (
	gateQuote    = "quote"
	gateEnum     = "enum"
	gateControl  = "control word"
	gatePlace    = "place"
	gateQuantity = "quantity"
	gateOrigin   = "origin"
	gatePolarity = "polarity"
)

// Tools of the gate probes.
var (
	fanSpeedTool = fanEnumTool("speed")
	fanOptTool   = tool("set_fan", "Set the fan speed",
		`"speed":{"type":"string","enum":["low","medium","high"]}`, "")
	fanDefaultTool = tool("set_fan", "Set the fan speed",
		`"speed":{"type":"string","enum":["low","medium","high"],"default":"medium"}`, `"speed"`)
	washerTool = tool("start_washer", "Start the washing machine",
		`"program":{"type":"string","enum":["cotton","synthetics","wool","quick"]},"temperature":{"type":"integer"}`,
		`"program"`)
	lightColourTool = tool("set_light_color", "Set the light colour",
		`"color":{"type":"string","enum":["red","green","blue","white"]}`, `"color"`)
	powerTool = tool("set_power", "Switch a device on or off",
		`"device":{"type":"string"},"state":{"type":"string","enum":["on","off"]}`, `"device","state"`)
	queryMusicTool = tool("play_music", "Play music",
		`"query":{"type":"string","description":"Artist, album, song or genre"}`, `"query"`)
	roomLightsTool = tool("set_lights", "Turn a room's lights on or off",
		`"room":{"type":"string"},"on":{"type":"boolean"}`, `"room","on"`)
	doorTool        = tool("lock_door", "Lock a door", `"door":{"type":"string"}`, `"door"`)
	weatherCityTool = placeTool("location")
	weatherDefault  = tool("get_weather", "Get the weather forecast for a place",
		`"location":{"type":"string","default":"Hamburg"}`, `"location"`)
	minutesTool = tool("set_timer", "Start a countdown timer",
		`"minutes":{"type":"integer","description":"Duration in minutes"},"label":{"type":"string",`+
			`"description":"What the timer is for"}`, `"minutes"`)
	minutesNumberTool = tool("set_timer", "Start a countdown timer",
		`"minutes":{"type":"number"},"label":{"type":"string"}`, `"minutes"`)
	minutesDefaultTool = tool("set_timer", "Start a countdown timer", `"minutes":{"type":"integer","default":5}`,
		`"minutes"`)
	taxiTool   = rideKeysTool("origin", "destination")
	renameTool = tool("rename_file", "Rename a file", `"origin":{"type":"string"},"destination":{"type":"string"}`,
		`"origin","destination"`)
	copyTool = tool("copy_file", "Copy a file", `"source":{"type":"string"},"target":{"type":"string"}`,
		`"source","target"`)
	taxiOptDest = tool("book_ride", "Book a taxi", `"origin":{"type":"string"},"destination":{"type":"string"}`,
		`"origin"`)
	navBothTool = tool("navigate", "Start navigation", `"origin":{"type":"string"},"destination":{"type":"string"}`,
		`"origin","destination"`)
)

// fanEnumTool is set_fan with one required enum of speeds under key.
func fanEnumTool(key string) string {
	return tool("set_fan", "Set the fan speed", `"`+key+`":{"type":"string","enum":["low","medium","high"]}`,
		`"`+key+`"`)
}

// placeTool is get_weather with one required string under key.
func placeTool(key string) string {
	return tool("get_weather", "Get the weather forecast for a place", `"`+key+`":{"type":"string"}`, `"`+key+`"`)
}

// rideKeysTool is book_ride with two required strings, an origin and a destination under the keys given.
func rideKeysTool(origin, destination string) string {
	return tool("book_ride", "Book a taxi", `"`+origin+`":{"type":"string"},"`+destination+`":{"type":"string"}`,
		`"`+origin+`","`+destination+`"`)
}

// joinTools is the tool JSON arrays ts as one array.
func joinTools(ts ...string) string {
	parts := make([]string, len(ts))
	for i, t := range ts {
		parts[i] = strings.TrimSuffix(strings.TrimPrefix(t, "["), "]")
	}

	return "[" + strings.Join(parts, ",") + "]"
}

// inHamburg are system facts that name a place the requests do not.
const inHamburg = "date: 2026-10-11 Sun 09:30\nThe user is in Hamburg."

// gateSpec is one probe to record; a probe with more than one turn runs over `needle --serve`.
type gateSpec struct {
	id, gate, covers, tools string
	triggers                map[string][]string
	system                  string
	turns                   []string
}

// offKitchen is the command the quote probes report.
const offKitchen = "turn off the kitchen lights"

// The request of the probes on the keys of a place slot.
const weatherThere = "what's the weather there"

// The request of the probes on the keys of an origin and a destination.
const taxiAirport = "get me a taxi to the airport"

// gateProbes are the recorded probes: for each gate requests where it fires and where it is held back, with a
// trigger that matches, over two calls, and the polarity rule's negation cues before its verb.
func gateProbes() []gateSpec {
	one := func(id, gate, covers, tools, query string) gateSpec {
		return gateSpec{id: id, gate: gate, covers: covers, tools: tools, turns: []string{query}}
	}
	trig := func(s gateSpec, tr map[string][]string) gateSpec {
		s.triggers = tr

		return s
	}
	sys := func(s gateSpec) gateSpec {
		s.system = inHamburg

		return s
	}

	q := func(id, covers, query string) gateSpec { return one("quote_"+id, gateQuote, covers, homeSet, query) }
	e := func(id, covers, tools, query string) gateSpec { return one("enum_"+id, gateEnum, covers, tools, query) }
	c := func(id, covers, tools, query string) gateSpec {
		return one("control_"+id, gateControl, covers, tools, query)
	}
	pl := func(id, covers, tools, query string) gateSpec {
		return one("place_"+id, gatePlace, covers, tools, query)
	}
	n := func(id, covers, tools, query string) gateSpec {
		return one("quantity_"+id, gateQuantity, covers, tools, query)
	}
	o := func(id, covers, tools, query string) gateSpec {
		return one("origin_"+id, gateOrigin, covers, tools, query)
	}

	ps := []gateSpec{
		// A command reported by someone else, in quotes.
		q("said", "said and a quoted command", `my wife said "`+offKitchen+`"`),
		q("said_single", "said and single quotes", "my wife said '"+offKitchen+"'"),
		q("said_curly", "said and typographic double quotes", "my wife said “"+offKitchen+"”"),
		q("said_curly_single", "said and typographic single quotes", "my wife said ‘"+offKitchen+"’"),
		q("said_comma", "said, then the quote", `my wife said, "`+offKitchen+`"`),
		q("said_far", "words between said and the quote", `my wife said yesterday evening "`+offKitchen+`"`),
		q("i_said", "the speaker's own report", `I said "`+offKitchen+`"`),
		q("she_said", "she said", `she said "`+offKitchen+`"`),
		q("said_upper", "SAID", `My wife SAID "`+offKitchen+`"`),
		q("said_inside", "said inside a word", `my wife saidx "`+offKitchen+`"`),
		q("unsaid", "said at the end of a word", `leave it unsaid "`+offKitchen+`"`),
		q("said_other_quote", "a quote of something else", `my wife said "hi", now `+offKitchen),
		q("said_in_title", "said inside a quoted title", `play the song "Said and Done" in the kitchen`),
		q("told_me", "told me", `my wife told me "`+offKitchen+`"`),
		q("told_me_apostrophe", "told me and an apostrophe after it", "my wife told me it's late, "+offKitchen),
		q("wrote", "wrote", `my wife wrote "`+offKitchen+`"`),
		q("texted", "texted", `my wife texted "`+offKitchen+`"`),
		q("mentioned", "mentioned", `my wife mentioned "`+offKitchen+`"`),
		q("messaged", "messaged", `my wife messaged "`+offKitchen+`"`),
		q("emailed", "emailed", `my wife emailed "`+offKitchen+`"`),
		q("posted", "posted", `my wife posted "`+offKitchen+`"`),
		q("says", "says: not", `my wife says "`+offKitchen+`"`),
		q("tells", "tells me: not", `my wife tells me "`+offKitchen+`"`),
		q("asked", "asked: not", `my wife asked "`+offKitchen+`"`),
		q("told", "told without me: not", `my wife told "`+offKitchen+`"`),
		q("told_us", "told us: not", `my wife told us "`+offKitchen+`"`),
		q("sent", "sent: not", `my wife sent "`+offKitchen+`"`),
		q("shouted", "shouted: not", `my wife shouted "`+offKitchen+`"`),
		q("colon", "a colon, no verb: not", `my wife: "`+offKitchen+`"`),
		q("said_after", "said after the quote: not", `"`+offKitchen+`", my wife said`),
		q("said_no_quote", "said and no quote: not", "my wife said to "+offKitchen),
		q("said_backtick", "backticks: not", "my wife said `"+offKitchen+"`"),
		q("said_no_apostrophe", "said and no quote mark after it: not", "my wife said its time, "+offKitchen),
		q("own_quote", "the speaker's quote, no verb: not", `"`+offKitchen+`"`),
		q("title", "a quoted title: not", `play "Turn Off the Light" in the kitchen`),
		q("two_calls", "one call outside the quote", `turn on the hallway lights, and my wife said "lock the front door"`),
		trig(q("trigger", "said, with a trigger on the quoted tool", `my wife said "`+offKitchen+`"`),
			map[string][]string{"set_lights": {"lights"}}),
		trig(q("trigger_lock", "told me, with a trigger on the quoted tool", `he told me "lock the front door"`),
			map[string][]string{"lock_door": {"door"}}),

		// A required enum filled with an option the request never names.
		e("unnamed", "set the fan: medium", fanSpeedTool, "set the fan"),
		e("unnamed_bedroom", "the fan in the bedroom: medium", fanSpeedTool, "turn on the fan in the bedroom"),
		e("synonym_fast", "fast for high", fanSpeedTool, "make the fan go fast"),
		e("synonym_full", "full speed for high", fanSpeedTool, "fan at full speed"),
		e("synonym_max", "max for high", fanSpeedTool, "fan to max please"),
		e("lowest", "lowest: low is too short to name by a prefix", fanSpeedTool, "fan lowest"),
		e("inside_word", "thigh: high inside a word", fanSpeedTool, "fan thigh"),
		e("downstairs", "downstairs is no lowering word", fanSpeedTool, "set the downstairs fan"),
		e("default", "a default option does not exempt", fanDefaultTool, "set the fan"),
		e("washer", "jeans for wool", washerTool, "wash my jeans"),
		e("colour", "the colour of the sea: blue", lightColourTool, "make the lights the colour of the sea"),
		e("key_setting", "key setting", fanEnumTool("setting"), "set the fan"),
		e("key_state", "key state", fanEnumTool("state"), "set the fan"),
		e("key_modal", "key modal holds no exempt word", fanEnumTool("modal"), "set the fan"),
		e("named", "the option named: not", fanSpeedTool, "set the fan to high"),
		e("named_upper", "the option in upper case: not", fanSpeedTool, "fan LOW"),
		e("prefix", "a word that starts with the option: not", fanSpeedTool, "fan highway"),
		e("prefix_quick", "quickly for quick: not", washerTool, "wash the shirts quickly"),
		e("prefix_medium", "mediumish for medium: not", fanSpeedTool, "fan mediumish"),
		e("raising_up", "up for high: not", fanSpeedTool, "turn the fan up"),
		e("raising_max", "up to max: not", fanSpeedTool, "turn the fan up to max"),
		e("raising_louder", "louder for high: not", fanSpeedTool, "make the fan louder"),
		e("lowering_down", "down for low: not", fanSpeedTool, "turn the fan down"),
		e("lowering_quieter", "quieter for low: not", fanSpeedTool, "make the fan quieter"),
		e("lowering_lower", "lower for low: not", fanSpeedTool, "fan lower"),
		e("optional", "an optional enum: not", fanOptTool, "set the fan"),
		e("key_mode", "key mode: not", fanEnumTool("mode"), "set the fan"),
		e("key_fan_mode", "a key holding mode: not", fanEnumTool("fan_mode"), "set the fan"),
		e("key_speed_level", "a key holding level: not", fanEnumTool("speed_level"), "set the fan"),
		e("key_type", "key type: not", fanEnumTool("type"), "set the fan"),
		e("key_status", "key status: not", fanEnumTool("status"), "set the fan"),
		e("key_category", "key category: not", fanEnumTool("category"), "set the fan"),
		e("key_format", "key format: not", fanEnumTool("format"), "set the fan"),
		e("key_units", "key units: not", fanEnumTool("units"), "set the fan"),
		e("two_calls", "with a call to another tool", joinTools(fanSpeedTool, roomLightsTool),
			"set the fan and turn off the kitchen lights"),
		trig(e("trigger", "a trigger on the tool", fanSpeedTool, "set the fan"),
			map[string][]string{"set_fan": {"fan"}}),

		// A required slot filled with a control word.
		c("up", "up", queryMusicTool, "music up"),
		c("off_capital", "Off", queryMusicTool, "music Off"),
		c("on_upper", "ON", queryMusicTool, "music ON"),
		c("lock", "lock", queryMusicTool, "music lock"),
		c("unlock", "unlock", queryMusicTool, "music unlock"),
		c("open", "open", queryMusicTool, "music open"),
		c("close", "close", queryMusicTool, "music close"),
		c("stop", "stop", queryMusicTool, "music stop"),
		c("pause", "pause", queryMusicTool, "music pause"),
		c("raise", "raise", queryMusicTool, "music raise"),
		c("room_on", "a room on", roomLightsTool, "on"),
		c("true", "true: not", queryMusicTool, "music true"),
		c("yes", "yes: not", queryMusicTool, "music yes"),
		c("high", "high: not", queryMusicTool, "music high"),
		c("enable", "enable: not", queryMusicTool, "music enable"),
		c("halt", "halt: not", queryMusicTool, "music halt"),
		c("everything", "everything: not", queryMusicTool, "music everything"),
		c("it", "it: not", queryMusicTool, "play it"),
		c("there", "there: not", queryMusicTool, "play music there"),
		c("longer_value", "a control word inside the value: not", queryMusicTool, "music on"),
		c("enum_state", "an enum option on", powerTool, "turn on the fan"),
		c("two_calls", "with a call to another tool", joinTools(queryMusicTool, roomLightsTool),
			"music up and turn on the kitchen lights"),
		trig(c("trigger", "a trigger on the tool", queryMusicTool, "music up"),
			map[string][]string{"play_music": {"music"}}),

		// A required slot filled with a place the conversation never mentions.
		sys(pl("system", "a place only the system facts name", weatherCityTool, weatherThere)),
		sys(pl("key_city", "key city", placeTool("city"), weatherThere)),
		sys(pl("key_area", "key area", placeTool("area"), weatherThere)),
		sys(pl("key_venue", "key venue", placeTool("venue"), weatherThere)),
		sys(pl("key_zone", "key zone", placeTool("zone"), weatherThere)),
		sys(pl("key_site", "key site", placeTool("site"), weatherThere)),
		sys(pl("key_region", "key region", placeTool("region"), weatherThere)),
		sys(pl("key_city_name", "a key holding city", placeTool("city_name"), weatherThere)),
		pl("current_location", "current location for around here", weatherCityTool, "the weather around here"),
		pl("room", "a room the request never names", roomLightsTool, "lights up"),
		pl("two_calls", "with a call to another tool", joinTools(weatherCityTool, roomLightsTool),
			"what's the weather there and turn on the kitchen lights"),
		sys(pl("named", "the place named: not", weatherCityTool, "what's the weather in hamburg")),
		sys(pl("key_where", "key where: not", placeTool("where"), weatherThere)),
		sys(pl("key_address", "key address: not", placeTool("address"), weatherThere)),
		sys(pl("key_town", "key town: not", placeTool("town"), weatherThere)),
		sys(pl("key_country", "key country: not", placeTool("country"), weatherThere)),
		sys(pl("key_destination", "key destination: not", placeTool("destination"), weatherThere)),
		sys(pl("default", "the slot's default: not", weatherDefault, weatherThere)),
		pl("door", "a door: not", doorTool, "lock it up"),
		{
			id: "place_tool_result", gate: gatePlace, covers: "a place an earlier tool result holds: not",
			tools: catalogueSet, turns: []string{"search the web for pizza places nearby", pizzaPlace, weatherThere},
		},
		trig(sys(pl("trigger", "a trigger on the tool", weatherCityTool, weatherThere)),
			map[string][]string{"get_weather": {"weather"}}),

		// A required number with no default, and no quantity named.
		n("pasta", "no quantity", minutesTool, "set a timer for the pasta"),
		n("hours", "a plural unit", minutesTool, "set a timer for hours"),
		n("minutes", "minutes", minutesTool, "set a timer in minutes"),
		n("pair", "a pair", minutesTool, "set a timer for a pair of minutes"),
		n("some", "some", minutesTool, "set a timer for some minutes"),
		n("moment", "a moment", minutesTool, "set a timer for a moment"),
		n("fifth", "a fifth", minutesTool, "set a timer for a fifth"),
		n("tenth", "tenth", minutesTool, "set a timer for the tenth time"),
		n("twentyish", "twentyish", minutesTool, "set a timer for twentyish minutes"),
		n("tennis", "ten inside a word", minutesTool, "set a timer for tennis"),
		n("phone", "one inside a word", minutesTool, "set a timer for my phone"),
		n("number_type", "type number", minutesNumberTool, "set a timer for the pasta"),
		n("thermostat", "a temperature", homeSet, "turn up the heat"),
		n("two_calls", "with a call to another tool", joinTools(minutesTool, roomLightsTool),
			"set a timer for the pasta and turn on the kitchen lights"),
		n("digits", "digits: not", minutesTool, "set a timer for 10 minutes for the pasta"),
		n("ten", "ten: not", minutesTool, "set a timer for ten minutes"),
		n("couple", "a couple: not", minutesTool, "set a timer for a couple of minutes"),
		n("few", "a few: not", minutesTool, "set a timer for a few minutes"),
		n("several", "several: not", minutesTool, "set a timer for several minutes"),
		n("dozen", "a dozen: not", minutesTool, "set a timer for a dozen minutes"),
		n("half", "half: not", minutesTool, "set a timer for half an hour"),
		n("quarter", "a quarter: not", minutesTool, "set a timer for a quarter"),
		n("third", "a third: not", minutesTool, "set a timer for a third"),
		n("an_hour", "an hour: not", minutesTool, "set a timer for an hour"),
		n("a_week", "a week: not", minutesTool, "set a timer for a week"),
		n("a_second", "a second: not", minutesTool, "set a timer for a second"),
		n("hourly", "hourly: not", minutesTool, "set an hourly timer"),
		n("weekly", "weekly: not", minutesTool, "set a weekly timer"),
		n("halfway", "halfway: not", minutesTool, "set a timer halfway"),
		n("none", "none: not", minutesTool, "set a timer for none"),
		n("default", "a default: not", minutesDefaultTool, "set a timer"),
		{
			id: "quantity_earlier_turn", gate: gateQuantity, covers: "a quantity an earlier user turn names: not",
			tools: minutesTool, turns: []string{"set a timer for 10 minutes for the pasta", "set the same timer for the rice"},
		},
		{
			id: "quantity_earlier_turn_other", gate: gateQuantity,
			covers: "a quantity an earlier user turn names, a new timer", tools: minutesTool,
			turns: []string{"set a timer for 10 minutes for the pasta", "set a timer for the rice"},
		},
		trig(n("trigger", "a trigger on the tool", minutesTool, "set a timer for the pasta"),
			map[string][]string{"set_timer": {"timer"}}),

		// The destination copied into the origin.
		o("taxi", "origin and destination", taxiTool, taxiAirport),
		o("pickup", "pickup and destination", rideKeysTool("pickup", "destination"), taxiAirport),
		o("from_to", "from and to", rideKeysTool("from", "to"), taxiAirport),
		o("start_end", "start and end", rideKeysTool("start", "end"), taxiAirport),
		o("source_target", "source and target", rideKeysTool("source", "target"), taxiAirport),
		o("departure_arrival", "departure and arrival", rideKeysTool("departure", "arrival"), taxiAirport),
		o("pickup_dropoff", "pickup and dropoff", rideKeysTool("pickup", "dropoff"), taxiAirport),
		o("from_location", "from_location and to_location", rideKeysTool("from_location", "to_location"), taxiAirport),
		o("stated", "the request names the same place twice", taxiTool, "get me a taxi from the airport to the airport"),
		o("from_only", "the origin copied into the destination", taxiTool, "get me a taxi from the airport"),
		o("optional_destination", "an optional destination", taxiOptDest, "get me a taxi from the airport"),
		o("rename", "a tool that is no ride", renameTool, "rename report to report"),
		o("copy", "source and target of a copy", copyTool, "copy notes.txt to notes.txt"),
		o("two_calls", "with a call to another tool", joinTools(taxiTool, roomLightsTool),
			"turn off the kitchen lights and get me a taxi to the airport"),
		o("different", "different places: not", taxiTool, "get me a taxi from the zoo to the beach"),
		o("article", "the train station against train station: not", taxiTool, "book a taxi to the train station"),
		o("pickup_location", "pickup_location and dropoff_location: not",
			rideKeysTool("pickup_location", "dropoff_location"), taxiAirport),
		o("start_point", "start_point and end_point: not", rideKeysTool("start_point", "end_point"), taxiAirport),
		o("location", "location and destination: not", rideKeysTool("location", "destination"), taxiAirport),
		o("navigate", "office against the office: not", navBothTool, "navigate to the office"),
		trig(o("trigger", "a trigger on the tool", taxiTool, taxiAirport), map[string][]string{"book_ride": {"taxi"}}),
	}

	return append(ps, polarityProbes()...)
}

// polarityProbes put each negation cue 12, 13 and 14 bytes before "turn off" (the polarity rule's verb), and record
// requests where the model writes the polarity the verb does not, which show whether the rule keeps it.
func polarityProbes() []gateSpec {
	p := func(id, covers, query string) gateSpec {
		return gateSpec{id: "polarity_" + id, gate: gatePolarity, covers: covers, tools: homeSet, turns: []string{query}}
	}

	// Fillers of each length, so that the cue starts 12, 13 or 14 bytes before the verb.
	fill := map[int]string{
		1: "a", 2: "so", 3: "now", 4: "ever", 5: "today", 6: "really", 7: "tonight", 8: "honestly", 9: "basically",
	}

	var ps []gateSpec

	for _, c := range []struct{ id, cue string }{
		{"dont", "don't"},
		{"dont_bare", "dont"},
		{"not", "not"},
		{"never", "never"},
		{"shouldnt", "shouldn't"},
		{"mustnt", "mustn't"},
		{"dont_curly", "don’t"},
	} {
		for _, d := range []int{12, 13, 14} {
			ps = append(ps, p(fmt.Sprintf("%s_%d", c.id, d), fmt.Sprintf("%s %d bytes before turn off", c.cue, d),
				c.cue+" "+fill[d-len(c.cue)-2]+" turn off the bedroom lights"))
		}
	}

	return append(ps,
		p("dont_13_kept", "don't 13 bytes before turn off; the model writes on: true",
			"don't really turn off the bedroom lights"),
		p("dont_14_flipped", "don't 14 bytes before turn on; the model writes on: false",
			"don't already turn on the bedroom lights"),
		p("dont_bare_12_kept", "dont 12 bytes before turn on; the model writes on: false",
			"dont really turn on the bedroom lights"),
		p("never_6_kept", "never right before turn off; the model writes on: true", "never turn off the bedroom lights"),
		p("nothing_8_kept", "not inside a word 8 bytes before turn on", "nothing turn on the bedroom lights"),
		p("nothing_15_flipped", "not inside a word 15 bytes before turn on", "nothing really turn on the bedroom lights"),
	)
}

// cmdGates records the CLI's answers to requests that exercise the grounding gates of upstream's confidence guide
// and the polarity rule's negation window, to set the port's gates against.
func cmdGates(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := gatesFixture{
		header: o.header("needle --model needle3.cact --tools <tools> --system <system file> --threads 4 " +
			"--prompt <query>; with more than one turn --serve, POST /complete per turn"),
		System: o.corpus.System,
	}

	for _, p := range gateProbes() {
		tools := p.tools
		if tools == homeSet || tools == assistantSet || tools == catalogueSet {
			tools = o.corpus.tools(tools)
		}

		tools, err = withTriggers(tools, p.triggers)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		system := p.system
		if system == "" {
			system = o.corpus.System
		}

		answers, err := o.validationAnswers(ctx, "gates-"+p.id, tools, system, p.turns)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		rec := gateProbe{
			ID: p.id, Gate: p.gate, Covers: p.covers, Tools: tools, Triggers: p.triggers, System: p.system,
			Turns: p.turns,
		}

		for _, a := range answers {
			rec.Answers = append(rec.Answers, compact(a))
		}

		fx.Probes = append(fx.Probes, rec)
		_, _ = fmt.Fprintf(log, "%-32s %s\n", p.id, summary(answers[len(answers)-1]))
	}

	return writeJSON(cfg.outDir, "gates.json", fx)
}
