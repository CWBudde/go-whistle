//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
)

// repairProbe is the CLI's answer to one query that exercises a rule of the engine's repair of the decoded calls
// (llms.txt: "Arguments arrive after the engine's deterministic repair step"): the rule, what the probe covers, the
// tools JSON as sent, the system facts when they are not the corpus's, and the answer without its timings.
type repairProbe struct {
	ID     string          `json:"id"`
	Rule   string          `json:"rule"`
	Covers string          `json:"covers"`
	Tools  string          `json:"tools"`
	System string          `json:"system,omitempty"`
	Query  string          `json:"query"`
	Answer json.RawMessage `json:"answer"`
}

// repairFixture is testdata/text/engine/repair.json.
type repairFixture struct {
	header

	System string        `json:"system"`
	Probes []repairProbe `json:"probes"`
}

// noteTool is a tool with one required string argument named key, to pin which arguments the engine capitalises.
func noteTool(key string) string {
	return tool("save_note", "Save a note", `"`+key+`":{"type":"string"}`, `"`+key+`"`)
}

// tool is a tool JSON array of one tool with the given parameters' properties and required keys (JSON fragments).
func tool(name, description, properties, required string) string {
	return `[{"name":"` + name + `","description":"` + description + `","parameters":{"type":"object","properties":{` +
		properties + `},"required":[` + required + `]}}]`
}

// Tools of the repair probes; "home", "assistant" and "catalogue" stand for the corpus tool sets.
var (
	weatherTool = tool("get_weather", "Get the weather forecast for a place and day",
		`"location":{"type":"string","description":"City name"},"date":{"type":"string","format":"date",`+
			`"description":"Day of the forecast"}`, `"location"`)
	weatherDayTool = tool("get_weather", "Get the weather forecast for a place and day",
		`"location":{"type":"string"},"day":{"type":"string"}`, `"location","day"`)
	eventTool = tool("create_event", "Add an event to the calendar",
		`"title":{"type":"string"},"date":{"type":"string","format":"date"},"time":{"type":"string",`+
			`"description":"Start time as HH:MM, 24-hour"},"location":{"type":"string"}`, `"title","date"`)
	nestedNoteTool = tool("save_notes", "Save notes",
		`"notes":{"type":"array","items":{"type":"object","properties":{"title":{"type":"string"}}}},`+
			`"meta":{"type":"object","properties":{"title":{"type":"string"}}}`, `"notes"`)
	givenFamilyTool = tool("add_contact", "Add a contact",
		`"given_name":{"type":"string"},"family_name":{"type":"string"}`, `"given_name"`)
	firstLastTool = tool("add_contact", "Add a contact", `"first":{"type":"string"},"last":{"type":"string"}`, `"first"`)
	firstNameTool = tool("add_contact", "Add a contact",
		`"first_name":{"type":"string","description":"Given name"},"last_name":{"type":"string",`+
			`"description":"Family name"}`, `"first_name"`)
	contactTool = tool("add_contact", "Add a contact to the address book",
		`"first_name":{"type":"string"},"last_name":{"type":"string"},"phone":{"type":"string"}`,
		`"first_name","last_name"`)
	callTool   = tool("call_number", "Start a phone call", `"phone":{"type":"string"}`, `"phone"`)
	noteTitle  = tool("create_note", "Create a note", `"title":{"type":"string"}`, `"title"`)
	placesTool = tool("search_places", "Search the map for a place", `"query":{"type":"string"}`, `"query"`)
	volumeTool = tool("change_volume", "Change the speaker volume",
		`"direction":{"type":"string","enum":["up","down"]},"amount":{"type":"integer"}`, `"direction","amount"`)
	fanTool = tool("set_fan", "Set the fan",
		`"speed":{"type":"string","enum":["low","medium","high"]},"oscillate":{"type":"boolean"}`, `"speed"`)
	navTool = tool("navigate", "Start navigation", `"destination":{"type":"string"},"origin":{"type":"string"}`,
		`"destination"`)
	rideTool = tool("book_ride", "Book a taxi", `"origin":{"type":"string"},"destination":{"type":"string"}`,
		`"origin","destination"`)
	temperatureTool = tool("adjust_temperature", "Raise or lower the heating", `"degrees":{"type":"number"}`,
		`"degrees"`)
	defaultTimerTool = tool("set_timer", "Start a countdown timer",
		`"minutes":{"type":"integer","default":7},"label":{"type":"string"}`, `"minutes"`)
	defaultDoorTool = tool("lock_door", "Lock a door", `"door":{"type":"string","default":"front door"}`, `"door"`)
	defaultRoomTool = tool("set_lights", "Turn the lights on or off",
		`"room":{"type":"string","default":"living room"},"on":{"type":"boolean"}`, `"room","on"`)
)

// The corpus tool sets, named.
const (
	homeSet      = "home"
	assistantSet = "assistant"
	catalogueSet = "catalogue"
)

// The requests of the capitalisation probes over a one-key tool.
const buyMilk = "save a note saying buy milk"

// repairProbes exercise every rule llms.txt lists for the repair step, each with cases where it must not fire, and
// pin which strings the engine capitalises and how it resolves dates against the date: fact. A probe's system is
// the corpus's date: fact (Sunday 2026-10-11) unless given.
// The rules of llms.txt's repair paragraph the probes exercise, and the capitalisation it does not name.
const (
	ruleCapitalise    = "capitalise"
	ruleDate          = "date"
	ruleNameSplit     = "name split"
	rulePhone         = "phone"
	ruleQuoted        = "quoted title"
	rulePlace         = "place"
	rulePolarity      = "polarity"
	rulePaired        = "paired tool"
	ruleEnum          = "enum"
	ruleFromTo        = "from to"
	ruleByN           = "by N"
	ruleDefaultNumber = "default number"
	ruleRequiredSlot  = "required slot"
	ruleUnstated      = "unstated optional"
	ruleEmpty         = "empty optional"
	ruleDuplicate     = "duplicate"
	ruleExclude       = "exclude"
	ruleNegated       = "negated clause"
)

var repairProbes = []struct {
	id, rule, covers, tools, system, query string
}{
	// Capitalisation: which keys, which values.
	{"cap_label", ruleCapitalise, "key label", noteTool("label"), "", buyMilk},
	{"cap_title", ruleCapitalise, "key title", noteTool("title"), "", buyMilk},
	{"cap_message", ruleCapitalise, "key message", noteTool("message"), "", buyMilk},
	{"cap_subject", ruleCapitalise, "key subject", noteTool("subject"), "", buyMilk},
	{"cap_note", ruleCapitalise, "key note", noteTool("note"), "", buyMilk},
	{"cap_event_title", ruleCapitalise, "key event_title", noteTool("event_title"), "", buyMilk},
	{"cap_event_name", ruleCapitalise, "key event_name", noteTool("event_name"), "", buyMilk},
	{"cap_camel", ruleCapitalise, "key eventTitle, snake event_title", noteTool("eventTitle"), "", buyMilk},
	{"cap_body", ruleCapitalise, "key body: not", noteTool("body"), "", buyMilk},
	{"cap_text", ruleCapitalise, "key text: not", noteTool("text"), "", buyMilk},
	{"cap_name", ruleCapitalise, "key name: not", noteTool("name"), "", buyMilk},
	{"cap_description", ruleCapitalise, "key description: not", noteTool("description"), "", buyMilk},
	{"cap_title_text", ruleCapitalise, "key title_text: not", noteTool("title_text"), "", buyMilk},
	{"cap_task_title", ruleCapitalise, "key task_title: not", noteTool("task_title"), "", buyMilk},
	{"cap_email_subject", ruleCapitalise, "key email_subject: not", noteTool("email_subject"), "", buyMilk},
	{"cap_umlaut", ruleCapitalise, "a non-ASCII first letter: not", noteTool("title"), "", "save a note titled über alles"},
	{"cap_digit", ruleCapitalise, "a digit first: not", noteTool("title"), "", "save a note titled 3 eggs"},
	{"cap_mixed", ruleCapitalise, "iPhone: the first byte only", noteTool("title"), "", "save a note titled iPhone case"},
	{
		"cap_nested", ruleCapitalise, "titles inside an array and an object: not", nestedNoteTool, "",
		"save two notes titled buy milk and call mom, with the meta title errands",
	},

	// Dates against the date: fact.
	{"date_weekday", ruleDate, "a weekday", weatherTool, "", "what's the weather in Berlin on Friday"},
	{"date_weekday_today", ruleDate, "the date fact's weekday: a week on", weatherTool, "", "what's the weather in Berlin on Sunday"},
	{"date_weekday_tomorrow", ruleDate, "the next day's weekday", weatherTool, "", "what's the weather in Berlin on Monday"},
	{"date_next_weekday", ruleDate, "next Friday: the coming one", weatherTool, "", "what's the weather in Berlin next Friday"},
	{"date_weekday_abbrev", ruleDate, "fri: not a weekday", weatherTool, "", "what's the weather in Berlin on fri"},
	{"date_month_day", ruleDate, "March 4th, past this year", weatherTool, "", "what's the weather in Berlin on March 4th"},
	{
		"date_month_day_before", ruleDate, "March 4th, still ahead this year", weatherTool, "date: 2026-02-01 Sun 09:30",
		"what's the weather in Berlin on March 4th",
	},
	{"date_month_day_today", ruleDate, "the date fact's own day", weatherTool, "", "what's the weather in Berlin on October 11th"},
	{"date_month_day_past", ruleDate, "the day before the date fact", weatherTool, "", "what's the weather in Berlin on October 10th"},
	{"date_day_month", ruleDate, "4 March", weatherTool, "", "what's the weather in Berlin on 4 March"},
	{"date_explicit_year", ruleDate, "an explicit year", weatherTool, "", "what's the weather in Berlin on March 4th 2026"},
	{"date_free_string", ruleDate, "a weekday in a string that is no date: kept", weatherDayTool, "", "what's the weather in Berlin on Friday"},
	{"date_event_weekday", ruleDate, "a weekday in an event", eventTool, "", "add yoga on Thursday"},
	{"date_time_weekday", ruleDate, "a date and time string: kept", catalogueSet, "", "remind me to call the bank on Friday at 9:00"},
	{"date_time_month_day", ruleDate, "a date alone where a date and time is asked", catalogueSet, "", "remind me to pay rent on March 4th at 8:00"},

	// Name split.
	{"name_given", ruleNameSplit, "a full name in given_name", givenFamilyTool, "", "add a contact for Ravi Kumar"},
	{"name_given_family", ruleNameSplit, "a full name in given_name, family_name given", givenFamilyTool, "", "add Ravi Kumar"},
	{"name_first", ruleNameSplit, "a full name in first", firstLastTool, "", "create a contact named Ravi Kumar"},
	{"name_first_name", ruleNameSplit, "a full name in first_name", firstNameTool, "", "create a contact named Ravi Kumar"},
	{"name_split_kept", ruleNameSplit, "first and last apart: kept", contactTool, "", "add Ravi Kumar, 415-555-0162"},
	{"name_single", ruleNameSplit, "one name: kept", givenFamilyTool, "", "add a contact for Ravi"},

	// Phone completion.
	{"phone_truncated", rulePhone, "a truncated number", callTool, "", "call 001 415 555 0162 9876"},
	{"phone_split", rulePhone, "a number split into two calls", callTool, "", "call +44 (0) 20 7946 0958 1234"},
	{"phone_whole", rulePhone, "a whole number: kept", callTool, "", "call 0171 2345678"},

	// Quoted titles.
	{"quoted_truncated", ruleQuoted, "a quoted title cut short", noteTitle, "", `new note "the quick brown fox jumps over the lazy dog again and again"`},
	{"quoted_whole", ruleQuoted, "a quoted title in full: kept", noteTitle, "", `create a note titled "R&D sync -- part 2 (final!!)"`},

	// Place queries.
	{"place_located_at", rulePlace, "located at an address", placesTool, "", "look up Cafe Central located at 14 Herrengasse"},
	{"place_which_is_at", rulePlace, "which is at an address", placesTool, "", "find Mama's kitchen which is at 22 Oak Avenue"},
	{"place_which_is_located_at", rulePlace, "which is located at: not", placesTool, "", "find the Blue Door bakery which is located at 12 Elm Street"},
	{"place_at", rulePlace, "at an address, written by the model", placesTool, "", "look up Cafe Central at 14 Herrengasse"},

	// Polarity: booleans and enums follow the verb, a paired tool the request.
	{"bool_own_clause", rulePolarity, "turn on in the call's clause", homeSet, "", "turn on the living room lights but don't unlock the front door"},
	{"bool_negated", rulePolarity, "don't turn off: kept", homeSet, "", "don't turn off the bedroom lights"},
	{"bool_other_clause", rulePolarity, "unlock before a clause without a verb", homeSet, "", "unlock the garage and dim the kitchen lights to 20"},
	{"bool_other_clause_after", rulePolarity, "unlock after a clause without a verb", homeSet, "", "dim the kitchen lights to 20 and unlock the garage"},
	{"enum_louder", rulePolarity, "louder to up", volumeTool, "", "make it louder by 4"},
	{"enum_down", rulePolarity, "down: kept", volumeTool, "", "turn it down by 2"},
	{"paired_unlock", rulePaired, "unlock to unlock_door", homeSet, "", "unlock the front door"},
	{"paired_second", rulePaired, "unlock in the second clause", homeSet, "", "turn on the porch lights and unlock the back door"},
	{"paired_lock", rulePaired, "lock: kept", homeSet, "", "lock the front door"},

	// An enum moves to the one option the request names.
	{"enum_named_high", ruleEnum, "the one option named", fanTool, "", "fan on high and oscillating"},
	{"enum_named_low", ruleEnum, "the one option named", fanTool, "", "fan low, oscillate"},
	{"enum_two_named", ruleEnum, "two options named: kept", fanTool, "", "set the fan to medium or high"},
	{"enum_none_named", ruleEnum, "no option named: kept", fanTool, "", "turn the fan up to max"},

	// Pickup and destination from "from X to Y".
	{"from_to_missing", ruleFromTo, "a missing origin", navTool, "", "navigate from home to the office"},
	{"from_to_article", ruleFromTo, "a missing origin after the", navTool, "", "directions from the hotel to the museum"},
	{"from_to_given", ruleFromTo, "an origin given: kept", rideTool, "", "get me a taxi from the zoo to the beach"},

	// by N fills the only number argument.
	{"by_n", ruleByN, "by 2 to 23", temperatureTool, "", "raise the temperature by 2 to 23 degrees"},
	{"by_n_kept", ruleByN, "by 3, another number after", temperatureTool, "", "warm it up by 3 degrees, it's 19 now"},

	// Required slots and their defaults.
	{"default_number", ruleDefaultNumber, "a required number with a default, none named", defaultTimerTool, "", "set a timer for the pasta"},
	{"default_number_named", ruleDefaultNumber, "a quantity named: kept", defaultTimerTool, "", "set a timer for 3 minutes"},
	{"required_default", ruleRequiredSlot, "a required string with a default", defaultDoorTool, "", "lock the shed"},
	{"required_default_named", ruleRequiredSlot, "a required string with a default, the default named", defaultDoorTool, "", "lock the door"},
	{"required_default_in", ruleRequiredSlot, "a required string with a default, after in", defaultRoomTool, "", "turn on the lights in the kitchen"},
	{"required_default_modifier", ruleRequiredSlot, "a required string with a default, as a modifier", defaultRoomTool, "", "turn on the kitchen lights"},

	// Dropped arguments: unstated optional numbers and strings, empty optional strings.
	{"optional_number", ruleUnstated, "an optional number not stated", homeSet, "", "set the kitchen lights to 150 percent"},
	{"optional_number_stated", ruleUnstated, "an optional number stated: kept", homeSet, "", "set the kitchen lights to 40 percent"},
	{"optional_number_word", ruleUnstated, "an optional number as a word: kept", homeSet, "", "dim the kitchen lights to fifty percent"},
	{"optional_number_half", ruleUnstated, "half for 50: kept", homeSet, "", "set the kitchen lights to half brightness"},
	{"optional_zero", ruleUnstated, "an optional 0 not stated", homeSet, "", "turn on the lights everywhere except the bedroom"},
	{"optional_time", ruleUnstated, "an optional time from 3pm: kept", eventTool, "", "add the dentist tomorrow at 3pm"},
	{"empty_optional", ruleEmpty, "an empty optional string", assistantSet, "", "wake me up at 6:45 tomorrow"},
	{"empty_optional_time", ruleEmpty, "an empty optional string", assistantSet, "", "set an alarm for 6:30 tomorrow"},

	// Dropped calls: duplicates, exclusions, negations.
	{"duplicate", ruleDuplicate, "the same call twice", homeSet, "", "turn on the kitchen lights and turn on the kitchen lights"},
	{"duplicate_differs", ruleDuplicate, "two different calls: kept", homeSet, "", "switch the kitchen lights on and the bedroom lights off"},
	{"exclude_except", ruleExclude, "except", homeSet, "", "turn off the lights in the kitchen, the bedroom and the hallway except the hallway"},
	{"exclude_but_not", ruleExclude, "but not", homeSet, "", "turn off the lights in the kitchen and the hallway but not the hallway"},
	{"exclude_apart_from", ruleExclude, "apart from", homeSet, "", "lock the front door, the back door and the garage, apart from the garage"},
	{"exclude_other_value", ruleExclude, "an excluded word inside a value: kept", homeSet, "", "lock all the doors but not the garage"},
	{"negated_verb", ruleNegated, "a call whose verb a clause negates", homeSet, "", "play jazz in the kitchen and don't play anything in the bedroom"},
	{"negated_lock", ruleNegated, "don't lock", homeSet, "", "turn on the kitchen lights and don't lock the front door"},
	{"negated_other_verb", ruleNegated, "don't unlock against lock_door: kept", homeSet, "", "turn on the kitchen lights but never unlock the garage"},
}

// cmdRepair records the CLI's answers to requests that exercise the engine's repair of the decoded calls, to set the
// port's repair against.
func cmdRepair(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := repairFixture{
		header: o.header("needle --model needle3.cact --tools <tools> --system <system file> --threads 4 " +
			"--prompt <query>"),
		System: o.corpus.System,
	}

	for _, p := range repairProbes {
		tools := p.tools
		if tools == homeSet || tools == assistantSet || tools == catalogueSet {
			tools = o.corpus.tools(tools)
		}

		system := p.system
		if system == "" {
			system = o.corpus.System
		}

		a, err := o.repairAnswer(ctx, p.id, tools, system, p.query)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		fx.Probes = append(fx.Probes, repairProbe{
			ID: p.id, Rule: p.rule, Covers: p.covers, Tools: tools, System: p.system, Query: p.query, Answer: compact(a),
		})
		_, _ = fmt.Fprintf(log, "%-24s %s\n", p.id, summary(a))
	}

	return writeJSON(cfg.outDir, "repair.json", fx)
}

// repairAnswer answers query with the CLI over tools and the system facts system.
func (o *oracle) repairAnswer(ctx context.Context, id, tools, system, query string) (node, error) {
	p := filepath.Join(o.scratch, "tools-repair-"+id+".json")

	err := writeFile(p, []byte(tools))
	if err != nil {
		return node{}, err
	}

	sys, err := o.systemFile(system)
	if err != nil {
		return node{}, err
	}

	res, err := runCLI(ctx, o.needle(), []string{
		flagModel, o.needle3Path(), flagTools, p, flagSystem, sys, "--threads", cliThreads, "--prompt", query,
	})
	if err != nil {
		return node{}, err
	}

	if res.Exit != 0 {
		return node{}, fmt.Errorf("exit %d: %s", res.Exit, res.Stderr)
	}

	return normaliseResponse(res.stdout)
}
