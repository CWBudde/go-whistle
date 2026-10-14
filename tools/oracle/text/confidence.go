//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

// confidenceProbe is the engine's answer to one conversation that exercises its confidence: the question it serves
// (docs/FINDINGS.md F-062), what it covers, the tools JSON as sent, the turns and the answer to each, without its timings.
type confidenceProbe struct {
	ID       string            `json:"id"`
	Question string            `json:"question"`
	Covers   string            `json:"covers"`
	Tools    string            `json:"tools"`
	Turns    []string          `json:"turns"`
	Answers  []json.RawMessage `json:"answers"`
}

// confidenceFixture is testdata/text/engine/confidence.json.
type confidenceFixture struct {
	header

	System string            `json:"system"`
	Probes []confidenceProbe `json:"probes"`
}

// The questions of the confidence probes (docs/FINDINGS.md F-062).
const (
	questionRespond   = "respond"   // turns the model answers with text: chit-chat, questions, after a tool result
	questionObject    = "object"    // calls with object-valued arguments, and the same flattened to scalars
	questionForced    = "forced"    // values the grammar cuts or forces, against ones it lets through
	questionDecode    = "decode"    // calls whose decode probability should be low: ambiguous, unstated values
	questionDate      = "date"      // relative dates around event_tomorrow
	questionCatalogue = "catalogue" // requests over the 17 catalogue tools around catalogue_lock
)

// Tools of the confidence probes: objects at the top level (required and optional), nested, empty and in arrays,
// each with a flat twin of scalars; a bounded timer.
var (
	callFlatTool = tool("schedule_call", "Schedule a phone call",
		`"contact":{"type":"string"},"time":{"type":"string","description":"Start time as HH:MM, 24-hour"},`+
			`"minutes":{"type":"integer","description":"Length in minutes"}`, `"contact","time"`)
	snoozeFlatTool = tool("set_alarm", "Set an alarm clock",
		`"time":{"type":"string","description":"Alarm time as HH:MM, 24-hour"},"snooze_until":{"type":"string",`+
			`"description":"Last snooze as HH:MM, 24-hour"},"snooze_minutes":{"type":"integer"}`, `"time"`)
	tableTool = tool("book_table", "Book a restaurant table",
		`"restaurant":{"type":"string"},"party":{"type":"object","properties":{"size":{"type":"integer"},`+
			`"name":{"type":"string"}},"required":["size"]}`, `"restaurant","party"`)
	tableFlatTool = tool("book_table", "Book a restaurant table",
		`"restaurant":{"type":"string"},"size":{"type":"integer"},"name":{"type":"string"}`, `"restaurant","size"`)
	parcelTool = tool("ship_parcel", "Ship a parcel",
		`"to":{"type":"object","properties":{"name":{"type":"string"},"address":{"type":"object","properties":{`+
			`"street":{"type":"string"},"city":{"type":"string"}},"required":["city"]}},"required":["name","address"]}`,
		`"to"`)
	parcelFlatTool = tool("ship_parcel", "Ship a parcel",
		`"name":{"type":"string"},"street":{"type":"string"},"city":{"type":"string"}`, `"name","city"`)
	vacuumTool = tool("start_vacuum", "Start the robot vacuum",
		`"room":{"type":"string"},"options":{"type":"object"}`, `"room","options"`)
	vacuumOptTool = tool("start_vacuum", "Start the robot vacuum",
		`"room":{"type":"string"},"options":{"type":"object","properties":{"mode":{"type":"string",`+
			`"enum":["quiet","normal","turbo"]}}}`, `"room"`)
	vacuumFlatTool = tool("start_vacuum", "Start the robot vacuum",
		`"room":{"type":"string"},"mode":{"type":"string","enum":["quiet","normal","turbo"]}`, `"room"`)
	cartTool = tool("add_to_cart", "Add products to the shopping cart",
		`"items":{"type":"array","items":{"type":"object","properties":{"product":{"type":"string"},`+
			`"quantity":{"type":"integer"}},"required":["product"]}}`, `"items"`)
	cartFlatTool = tool("add_to_cart", "Add products to the shopping cart",
		`"products":{"type":"array","items":{"type":"string"}}`, `"products"`)
	cartOneTool = tool("add_to_cart", "Add a product to the shopping cart",
		`"product":{"type":"string"},"quantity":{"type":"integer"}`, `"product"`)
	boundedTimerTool = tool("set_timer", "Start a countdown timer",
		`"minutes":{"type":"integer","minimum":1,"maximum":60}`, `"minutes"`)
)

// confidenceSpec is one probe to record; a probe with more than one turn runs over `needle --serve`.
type confidenceSpec struct {
	id, question, covers, tools string
	turns                       []string
}

// addProbe records one probe: its id, question, what it covers, the tools and the turns.
type addProbe func(id, question, covers, tools string, turns ...string)

// confidenceProbes are the recorded probes: dozens per question, so that the hypotheses of F-062 can be told apart.
func confidenceProbes() []confidenceSpec {
	var ps []confidenceSpec

	add := func(id, question, covers, tools string, turns ...string) {
		ps = append(ps, confidenceSpec{id: id, question: question, covers: covers, tools: tools, turns: turns})
	}

	respondProbes(add)
	objectProbes(add)
	forcedProbes(add)
	decodeProbes(add)
	dateProbes(add)
	catalogueProbes(add)

	return ps
}

// respondProbes are chit-chat and questions with tools declared, and later turns after tool results of several
// shapes (objects, errors, strings, booleans, arrays, nested objects, null) and after a respond turn.
func respondProbes(add addProbe) {
	for i, q := range []string{
		"hello", "thanks!", "how are you today?", "what can you do?", "tell me a joke", "good morning",
		"who are you?", "what time is it?", "can you help me?", "thank you, that's all",
	} {
		add(fmt.Sprintf("chat_assistant_%02d", i), questionRespond, "a first turn without a request, assistant tools",
			assistantSet, q)
	}

	for i, q := range []string{
		"hi there", "what's the temperature inside?", "is the front door locked?", "are the kitchen lights on?",
		"what music is playing?",
	} {
		add(fmt.Sprintf("chat_home_%02d", i), questionRespond, "a first turn asking about state, home tools",
			homeSet, q)
	}

	const (
		timerQuery  = "set a timer for 10 minutes for the pasta"
		timerResult = `{"status": "ok", "timer_id": "t1", "ends_at": "09:40"}`
	)

	for _, f := range []struct {
		id, tools, query string
		results          []string
	}{
		{"timer", assistantSet, timerQuery, []string{
			timerResult, `{"status": "started"}`, `{"error": "timer service unavailable"}`, `"ok"`, `true`,
			`[{"id": "t1"}]`, `{"timer": {"id": "t1", "ends_at": "09:40"}}`, `null`,
		}},
		{"weather", assistantSet, "what's the weather in Berlin today?", []string{
			`{"forecast": "sunny", "high_c": 24}`, `{"forecast": "rain", "high_c": 12, "low_c": 7}`,
			`{"error": "location not found"}`, `[{"day": "2026-10-11", "forecast": "cloudy"}]`, `"sunny, 18 degrees"`,
		}},
		{"message", assistantSet, "text Anna that I'm running late", []string{
			`{"status": "sent"}`, `{"delivered": true, "message_id": "m42"}`, `{"error": "unknown contact"}`, `"sent"`,
		}},
		{"lights", homeSet, "turn off the kitchen lights", []string{
			`{"status": "ok"}`, `{"room": "kitchen", "on": false}`, `{"error": "device offline"}`, `[]`,
		}},
		{"event", assistantSet, "add the dentist tomorrow at 3pm", []string{
			`{"event_id": "e7", "status": "created"}`, `{"error": "calendar conflict"}`,
		}},
		{"thermostat", homeSet, "set the heating to 21 degrees", []string{
			`{"temperature": 21, "unit": "celsius"}`, `{"status": "ok"}`,
		}},
	} {
		for i, r := range f.results {
			add(fmt.Sprintf("result_%s_%d", f.id, i), questionRespond, "a call, then a tool result", f.tools,
				f.query, r)
		}
	}

	for _, f := range []struct{ id, tools, query, result string }{
		{"timer_5", assistantSet, "set a timer for 5 minutes", `{"status": "ok", "ends_at": "09:35"}`},
		{"timer_laundry", assistantSet, "set a timer for 20 minutes for the laundry", `{"status": "ok"}`},
		{"timer_eggs", assistantSet, "set a timer for 15 minutes for the eggs", `{"timer_id": "t2", "ends_at": "09:45"}`},
		{"alarm_7", assistantSet, "wake me up at 7", `{"status": "ok", "alarm_id": "a1"}`},
		{"alarm_630", assistantSet, "set an alarm for 6:30", `{"status": "set", "time": "06:30"}`},
		{"message_tom", assistantSet, "text Tom that I'll be home soon", `{"status": "sent", "to": "Tom"}`},
		{"message_julie", assistantSet, "text Julie that the meeting moved to 3pm", `{"status": "delivered"}`},
		{"weather_paris", assistantSet, "what's the weather in Paris?", `{"forecast": "cloudy", "high_c": 17}`},
		{"weather_rome", assistantSet, "what's the weather in Rome tomorrow?", `{"forecast": "sunny", "high_c": 26}`},
		{"event_lunch", assistantSet, "add lunch with Marco tomorrow at noon", `{"status": "created", "event_id": "e9"}`},
		{"event_review", assistantSet, "put the quarterly review on November 3rd 2026", `{"status": "ok"}`},
		{"lights_on", homeSet, "turn on the living room lights", `{"status": "ok"}`},
		{"lights_bedroom", homeSet, "turn off the bedroom lights", `{"room": "bedroom", "on": false}`},
		{"lock_front", homeSet, "lock the front door", `{"status": "locked"}`},
		{"unlock_back", homeSet, "unlock the back door", `{"status": "unlocked"}`},
		{"music_jazz", homeSet, "play jazz in the kitchen", `{"status": "playing", "track": "So What"}`},
		{"heating_22", homeSet, "set the heating to 22 degrees", `{"status": "ok", "target": 22}`},
		{"lights_dim", homeSet, "dim the kitchen lights to 30 percent", `{"status": "ok", "brightness": 30}`},
		{"list_milk", catalogueSet, "add milk to the groceries list", `{"status": "added", "count": 1}`},
		{"call_anna", catalogueSet, "call Anna", `{"status": "calling"}`},
		{"translate", catalogueSet, "translate good morning into Italian", `{"translation": "buongiorno"}`},
		{"search", catalogueSet, "search the web for pizza places", `[{"title": "Luigi's", "rating": 4.5}]`},
		{"navigate", catalogueSet, "take me to the central station", `{"status": "route started", "minutes": 12}`},
		{"reminder", catalogueSet, "remind me to call the bank on October 13th 2026 at 9:00", `{"status": "ok"}`},
	} {
		add("result_more_"+f.id, questionRespond, "a call, then a tool result", f.tools, f.query, f.result)
	}

	add("followup_timer_thanks", questionRespond, "a tool result, then thanks", assistantSet,
		timerQuery, timerResult, "thanks!")
	add("followup_weather_wear", questionRespond, "a tool result, then a follow-up question", assistantSet,
		"what's the weather in Berlin today?", `{"forecast": "sunny", "high_c": 24}`, "and what should I wear?")
	add("followup_message_thanks", questionRespond, "a tool result, then thanks", assistantSet,
		"text Anna that I'm running late", `{"status": "sent"}`, "great, thank you")
	add("followup_lights_what", questionRespond, "a tool result, then a question about it", homeSet,
		"turn off the kitchen lights", `{"status": "ok"}`, "what did you just do?")
	add("followup_timer_call", questionRespond, "a tool result, then a new request", assistantSet,
		timerQuery, timerResult, "and text Anna that dinner is at eight")
}

// objectProbes are calls with object-valued arguments (required and optional, nested, empty, in arrays) and the
// same requests over tools with the objects flattened to scalars.
func objectProbes(add addProbe) {
	for _, p := range []struct {
		id, object, flat string
		queries          []string
	}{
		{"call", scheduleCallTool, callFlatTool, []string{
			"schedule a call with Anna at 3pm", "schedule a call with Tom at 10:30 for 20 minutes",
			"set up a 15 minute call with Marco at 16:00", "schedule a call with my mother at 9am",
		}},
		{"snooze", snoozeTool, snoozeFlatTool, []string{
			"wake me at 6:45 and let me snooze until 7:10", "set an alarm for 6:45 with 5 minute snoozes",
			"set an alarm at 7:00", "wake me up at 5:30, snooze every 10 minutes until 6",
		}},
		{"table", tableTool, tableFlatTool, []string{
			"book a table for 4 at Luigi's under the name Miller", "book a table for two at the Ritz",
			"reserve a table at Nobu for 6 people", "book Luigi's for 3, name Anna",
		}},
		{"parcel", parcelTool, parcelFlatTool, []string{
			"send the parcel to Anna at 14 Herrengasse, Vienna", "ship this to Tom in Berlin",
			"ship the box to Marco, 3 Via Roma, Milan",
		}},
		{"vacuum", vacuumTool, vacuumFlatTool, []string{
			"vacuum the kitchen", "start the vacuum in the living room in quiet mode", "vacuum the hallway on turbo",
		}},
		{"vacuum_opt", vacuumOptTool, vacuumFlatTool, []string{
			"vacuum the bedroom", "vacuum the bedroom in quiet mode", "run the vacuum in the office on turbo",
		}},
		{"cart", cartTool, cartFlatTool, []string{
			"add 2 apples and 3 pears to the cart", "put milk and bread in my cart", "add a dozen eggs to the cart",
			"add 4 bananas, 1 melon and 2 lemons to my cart",
		}},
	} {
		for i, q := range p.queries {
			add(fmt.Sprintf("object_%s_%d", p.id, i), questionObject, "an object-valued argument", p.object, q)
			add(fmt.Sprintf("flat_%s_%d", p.id, i), questionObject, "the same request, the object flattened", p.flat, q)
		}
	}

	for i, q := range []string{"add 2 apples to the cart", "put milk in my cart", "add 6 eggs to the cart"} {
		add(fmt.Sprintf("flat_cart_one_%d", i), questionObject, "one product, scalars only", cartOneTool, q)
	}
}

// forcedProbes are values the grammar cuts (above a maximum, a decimal for an integer, an option the enum lacks)
// and the same tools with values it lets through.
func forcedProbes(add addProbe) {
	for _, m := range []int{90, 75, 120, 61, 45, 60, 5} {
		add(fmt.Sprintf("bounded_timer_%d", m), questionForced, "a timer of 1 to 60 minutes", boundedTimerTool,
			fmt.Sprintf("set a timer for %d minutes", m))
	}

	for _, b := range []int{150, 250, 80} {
		add(fmt.Sprintf("bounded_lights_%d", b), questionForced, "a brightness of 0 to 100", homeSet,
			fmt.Sprintf("set the kitchen lights to %d percent", b))
	}

	for i, q := range []string{
		"set a timer for 2.5 minutes", "set a timer for 7.5 minutes for the eggs", "set a timer for 1.5 minutes",
	} {
		add(fmt.Sprintf("decimal_timer_%d", i), questionForced, "a decimal for an integer", assistantSet, q)
	}

	for i, q := range []string{"set the fan to turbo", "set the fan to medium", "put the fan on full blast"} {
		add(fmt.Sprintf("enum_fan_%d", i), questionForced, "an enum option the request may not name", fanTool, q)
	}
}

// decodeProbes are requests whose call the model has to guess: values not stated, alternatives, pronouns.
func decodeProbes(add addProbe) {
	for i, q := range []string{
		"set a timer", "text mom", "remind me", "set the alarm", "what's the weather",
		"set a timer for ten or fifteen minutes", "text Anna or Tom that I'm late", "set an alarm for 6 or 7",
		"add something to my calendar", "send a message",
	} {
		add(fmt.Sprintf("guess_assistant_%02d", i), questionDecode, "an ambiguous or incomplete request",
			assistantSet, q)
	}

	for i, q := range []string{
		"turn it off", "lock it", "make it warmer", "play something", "turn the lights on or off",
		"set the lights to half", "lock the door", "play music",
	} {
		add(fmt.Sprintf("guess_home_%02d", i), questionDecode, "an ambiguous or incomplete request", homeSet, q)
	}

	for i, q := range []string{"call him", "add milk", "translate this", "search for it", "email the team"} {
		add(fmt.Sprintf("guess_catalogue_%02d", i), questionDecode, "an ambiguous or incomplete request",
			catalogueSet, q)
	}
}

// dateProbes are relative dates against the corpus's date fact, around event_tomorrow (its request included).
func dateProbes(add addProbe) {
	for i, q := range []string{
		"add the dentist tomorrow at 3pm", "add the dentist tomorrow", "add the dentist the day after tomorrow at 3pm",
		"add the dentist on Monday at 3pm", "add the dentist next Friday at 10am", "add a haircut tomorrow at 11",
		"add yoga tomorrow at 7pm", "add the dentist today at 5pm", "add the dentist in two days",
		"add lunch with Marco tomorrow at noon", "add the dentist tomorrow at 15:00",
		"add a meeting tomorrow at 3pm for 30 minutes", "add the dentist at 3pm tomorrow",
		"put the dentist in my calendar for tomorrow at 3pm", "add the dentist on October 12th at 3pm",
		"add the dentist on 2026-10-12 at 15:00",
	} {
		add(fmt.Sprintf("event_%02d", i), questionDate, "a relative or explicit day for create_event", assistantSet, q)
	}

	for i, q := range []string{
		"what's the weather in Paris tomorrow", "what's the weather in Rome the day after tomorrow",
		"weather in Madrid on Saturday", "what's the weather in Oslo today",
	} {
		add(fmt.Sprintf("weather_%02d", i), questionDate, "a relative day for get_weather", assistantSet, q)
	}
}

// catalogueProbes are requests over the 17 catalogue tools, catalogue_lock's among them.
func catalogueProbes(add addProbe) {
	for i, q := range []string{
		"lock the garage door", "lock the front door", "lock the back door", "unlock the garage door",
		"lock the garage", "lock the shed door", "lock all the doors", "turn off the garage lights",
		"set a timer for 10 minutes", "play jazz in the kitchen", "add milk to the groceries list", "call Anna",
		"what's the weather in Berlin", "turn on the kitchen lights", "translate thank you into German",
		"set the heating to 21 degrees", "search the web for pizza places", "set an alarm for 6:45",
	} {
		add(fmt.Sprintf("catalogue_%02d", i), questionCatalogue, "a request over the catalogue tools", catalogueSet, q)
	}
}

// cmdConfidence records the CLI's answers to requests that exercise the envelope's confidence: respond turns, object
// arguments, forced and guessed values, relative dates and the catalogue tools (docs/FINDINGS.md F-062).
func cmdConfidence(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := confidenceFixture{
		header: o.header("needle --model needle3.cact --tools <tools> --system <system file> --threads 4 " +
			"--prompt <query>; with more than one turn --serve, POST /complete per turn"),
		System: o.corpus.System,
	}

	for _, p := range confidenceProbes() {
		tools := p.tools
		if tools == homeSet || tools == assistantSet || tools == catalogueSet {
			tools = o.corpus.tools(tools)
		}

		answers, err := o.validationAnswers(ctx, "confidence-"+p.id, tools, o.corpus.System, p.turns)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		rec := confidenceProbe{ID: p.id, Question: p.question, Covers: p.covers, Tools: tools, Turns: p.turns}

		for _, a := range answers {
			rec.Answers = append(rec.Answers, compact(a))
		}

		fx.Probes = append(fx.Probes, rec)
		_, _ = fmt.Fprintf(log, "%-28s %s\n", p.id, summary(answers[len(answers)-1]))
	}

	return writeJSON(cfg.outDir, "confidence.json", fx)
}
