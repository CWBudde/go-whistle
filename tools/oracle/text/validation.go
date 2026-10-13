//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
)

// validationProbe is the engine's answer to one conversation that exercises the envelope's validation block: the
// kind (a negation cue, a cue with a matching trigger, a cue before a tool's verb, the paired tool's reading of a
// negation, or grounding), what it covers, the
// tools JSON as sent (with any triggers), the system facts when they are not the corpus's, the turns and the answer
// to each, without its timings.
type validationProbe struct {
	ID       string              `json:"id"`
	Kind     string              `json:"kind"`
	Covers   string              `json:"covers"`
	Tools    string              `json:"tools"`
	Triggers map[string][]string `json:"triggers,omitempty"`
	System   string              `json:"system,omitempty"`
	Turns    []string            `json:"turns"`
	Answers  []json.RawMessage   `json:"answers"`
}

// validationFixture is testdata/text/engine/validation.json.
type validationFixture struct {
	header

	System string            `json:"system"`
	Probes []validationProbe `json:"probes"`
}

// The kinds of validation probes.
const (
	kindNegation  = "negation"
	kindTrigger   = "trigger"
	kindDrop      = "drop"
	kindPaired    = "paired"
	kindGrounding = "grounding"
)

// Tools of the validation probes: one with a nested object property, ones with array properties (of strings and
// numbers), and one with a top-level and a nested time.
var (
	scheduleCallTool = tool("schedule_call", "Schedule a phone call",
		`"contact":{"type":"string"},"slot":{"type":"object","properties":{"time":{"type":"string",`+
			`"description":"Start time as HH:MM, 24-hour"},"minutes":{"type":"integer","description":"Length in minutes"}},`+
			`"required":["time"]}`, `"contact","slot"`)
	alarmsTool = tool("set_alarms", "Set several alarms",
		`"times":{"type":"array","items":{"type":"string","description":"Alarm time as HH:MM, 24-hour"}}`, `"times"`)
	pricesTool = tool("log_prices", "Record prices", `"prices":{"type":"array","items":{"type":"number"}}`, `"prices"`)
	snoozeTool = tool("set_alarm", "Set an alarm clock",
		`"time":{"type":"string","description":"Alarm time as HH:MM, 24-hour"},"snooze":{"type":"object",`+
			`"properties":{"until":{"type":"string","description":"Last snooze as HH:MM, 24-hour"},`+
			`"minutes":{"type":"integer"}}}`, `"time"`)
)

// System facts that hold values the requests do not.
const (
	livesInHamburg = "date: 2026-10-11 Sun 09:30\nThe user lives in Hamburg."
	defaultTimer   = "date: 2026-10-11 Sun 09:30\ndefault timer: 25 minutes"
)

// pizzaPlace is a tool result naming a place and its address.
const pizzaPlace = `[{"title": "Luigi's Pizzeria", "address": "14 Herrengasse"}]`

// validationSpec is one probe to record; a probe with more than one turn runs over `needle --serve`.
type validationSpec struct {
	id, kind, covers, tools string
	triggers                map[string][]string
	system                  string
	turns                   []string
}

// negationProbe is a one-turn request over tools whose validation.negation the probe records.
func negationProbe(id, covers, tools, query string) validationSpec {
	return validationSpec{id: id, kind: kindNegation, covers: covers, tools: tools, turns: []string{query}}
}

// The cues the trigger and drop probes put before a verb, with the id suffix of each.
var verbCues = []struct{ id, cue string }{
	{"dont", "don't"},
	{"do_not", "do not"},
	{"dont_bare", "dont"},
	{"dont_upper", "DON'T"},
	{"dont_curly", "don’t"},
	{"never", "never"},
	{"shouldnt", "shouldn't"},
	{"should_not", "should not"},
	{"mustnt", "mustn't"},
	{"must_not", "must not"},
	{"not", "not"},
	{"no", "no"},
	{"wont", "won't"},
	{"will_not", "will not"},
	{"cant", "can't"},
	{"cannot", "cannot"},
	{"doesnt", "doesn't"},
	{"stop", "stop"},
	{"cancel", "cancel"},
	{"without", "without"},
	{"instead_of", "instead of"},
	{"except", "except"},
}

// validationProbes are the recorded probes: the negation cues (spellings, words that contain them, the words after
// them that cancel them, other languages), each cue with a trigger that matches the request, each cue before a
// tool's verb, the requests whose only call a cue negates or excludes, the paired tool's reading of a negation, and
// the grounding of nested fields, array items, the system facts and earlier turns.
func validationProbes() []validationSpec {
	n := func(id, covers, query string) validationSpec {
		return negationProbe("neg_"+id, covers, homeSet, query)
	}

	ps := []validationSpec{
		// Spellings of the cues.
		n("dont", "don't", "don't turn off the bedroom lights"),
		n("do_not", "do not", "do not turn off the bedroom lights"),
		n("dont_bare", "dont without the apostrophe", "dont turn off the bedroom lights"),
		n("dont_curly", "the curly don’t", "don’t turn off the bedroom lights"),
		n("dont_acute", "don´t with an acute accent", "don´t turn off the bedroom lights"),
		n("dont_upper", "DON'T", "DON'T turn off the bedroom lights"),
		n("do_not_upper", "DO NOT", "DO NOT turn off the bedroom lights"),
		n("never", "never", "never turn off the bedroom lights"),
		n("never_upper", "NEVER", "NEVER turn off the bedroom lights"),
		n("shouldnt", "shouldn't", "you shouldn't turn off the bedroom lights"),
		n("shouldnt_curly", "the curly shouldn’t", "you shouldn’t turn off the bedroom lights"),
		n("shouldnt_bare", "shouldnt without the apostrophe", "you shouldnt turn off the bedroom lights"),
		n("should_not", "should not", "you should not turn off the bedroom lights"),
		n("mustnt", "mustn't", "you mustn't turn off the bedroom lights"),
		n("must_not", "must not", "you must not turn off the bedroom lights"),
		n("not", "not", "turn on the bedroom lights, not the kitchen lights"),
		n("no", "no", "no, turn off the bedroom lights"),
		n("stop", "stop", "stop the music in the kitchen"),
		n("cancel", "cancel", "cancel the music in the kitchen"),
		n("wont", "won't", "I won't need the bedroom lights, turn them off"),
		n("cant", "can't", "I can't see, turn on the bedroom lights"),
		n("cannot", "cannot", "you cannot turn off the bedroom lights"),
		n("doesnt", "doesn't", "it doesn't matter, turn on the bedroom lights"),
		n("wouldnt", "wouldn't", "you wouldn't turn off the bedroom lights"),
		n("didnt", "didn't", "you didn't turn off the bedroom lights"),
		n("isnt", "isn't", "isn't it dark, turn on the bedroom lights"),
		n("without", "without", "turn on the bedroom lights without the kitchen"),
		n("instead_of", "instead of", "turn on the bedroom lights instead of the kitchen lights"),
		n("except", "except", "turn on the lights except the bedroom"),
		n("other_clause", "don't in a clause other than the call's", "turn off the bedroom lights, don't wait"),

		// Where a cue matches: inside words, and what must follow dont.
		n("never_inside", "never inside a word", "nevertheless turn off the bedroom lights"),
		n("never_inside_end", "never at the end of a word", "whenever you can, turn on the bedroom lights"),
		n("dont_inside", "don't after letters", "turn on the bedroom lights, the xdon't room"),
		n("dont_before_letters", "don't before letters", "turn on the bedroom lights, the don'tx room"),
		n("dont_bare_before_letters", "dont before letters", "turn on the bedroom lights, the dontx room"),
		n("dont_bare_comma", "dont before a comma", "dont, turn off the bedroom lights"),
		n("dont_bare_end", "dont at the end", "turn off the bedroom lights dont"),
		n("dont_bare_apostrophe", "dont before an apostrophe", "dont't turn on the bedroom lights"),
		n("do_not_two_spaces", "do  not with two spaces", "do  not turn off the bedroom lights"),
		n("do_not_hyphen", "do-not", "do-not turn off the bedroom lights"),
		n("donot", "donot", "donot turn off the bedroom lights"),

		// Words after a cue that cancel it.
		n("never_mind", "never mind", "never mind, turn on the bedroom lights"),
		n("nevermind", "nevermind", "nevermind, turn on the bedroom lights"),
		n("never_hyphen_mind", "never-mind", "never-mind, turn on the bedroom lights"),
		n("dont_forget", "don't forget", "turn off the bedroom lights and don't forget the kitchen"),
		n("do_not_worry", "do not worry", "turn off the bedroom lights and do not worry about the kitchen"),
		n("never_hesitate", "never hesitate", "turn off the bedroom lights and never hesitate"),
		n("dont_mind", "don't mind", "turn off the bedroom lights and don't mind the kitchen"),
		n("dont_disturb", "don't disturb", "turn off the bedroom lights and don't disturb me"),
		n("shouldnt_forget", "shouldn't forget", "turn off the bedroom lights and shouldn't forget the kitchen"),
		n("dont_forgetting", "a word that starts with forget", "turn off the bedroom lights and don't forgetting"),
		n("dont_worries", "worries, which does not start with worry", "turn off the bedroom lights and don't worries"),
		n("dont_hesitating", "hesitating, which does not start with hesitate", "turn off the bedroom lights and don't hesitating"),
		n("dont_touch", "don't touch: a verb that does not cancel", "turn off the bedroom lights and don't touch the kitchen"),
		n("dont_wait", "don't wait", "turn off the bedroom lights and don't wait"),
		n("dont_forget_upper", "DON'T FORGET", "turn off the bedroom lights and DON'T FORGET the kitchen"),
		n("dont_forget_two_spaces", "don't and forget two spaces apart", "turn off the bedroom lights and don't  forget the kitchen"),
		n("dont_comma_forget", "don't, forget", "turn off the bedroom lights and don't, forget the kitchen"),
		n(
			"dont_forget_and_touch", "one cue cancelled, another not",
			"turn off the bedroom lights, don't forget the kitchen and don't touch the hallway",
		),

		// Quoted titles.
		negationProbe("neg_quoted_dont", "don't only inside a quoted title", noteTitle, `create a note titled "Don't panic"`),
		negationProbe("neg_quoted_never", "never only inside a quoted title", noteTitle, `create a note titled "Never again"`),
		negationProbe("neg_quoted_do_not", "do not disturb inside a quoted title", noteTitle, `create a note titled "Do not disturb"`),
		negationProbe("neg_quoted_none", "a quoted title without a cue", noteTitle, `create a note titled "Keep calm"`),

		// Other languages.
		n("german_nicht", "German nicht", "Schalte das Licht im Schlafzimmer nicht aus"),
		n("german_niemals", "German niemals", "Schalte niemals das Licht im Schlafzimmer aus"),
		n("spanish_no", "Spanish no", "No apagues las luces del dormitorio"),
		n("spanish_nunca", "Spanish nunca", "Nunca apagues las luces del dormitorio"),
		n("french_ne_pas", "French ne … pas", "N'éteins pas les lumières de la chambre"),
		n("french_jamais", "French ne jamais", "Ne jamais éteindre les lumières de la chambre"),
	}

	// Each cue with a trigger that matches the request: a trigger on play_music routes it there unless the request
	// is negated (F-054, F-055); where the calls are below 0.1 without the trigger, one on set_lights shows whether
	// they ship.
	for _, c := range verbCues {
		ps = append(ps, validationSpec{
			id: "trigger_" + c.id, kind: kindTrigger, covers: c.cue + " with a trigger that routes to play_music",
			tools: homeSet, triggers: map[string][]string{"play_music": {"front door"}},
			turns: []string{"turn on the living room lights but " + c.cue + " unlock the front door"},
		})
	}

	ps = append(ps,
		validationSpec{
			id: "trigger_dont_forget", kind: kindTrigger, covers: "don't forget with a trigger that routes to play_music",
			tools: homeSet, triggers: map[string][]string{"play_music": {"front door"}},
			turns: []string{"turn on the living room lights but don't forget to unlock the front door"},
		},
	)

	for _, c := range []struct{ id, cue string }{{"dont_curly", "don’t"}, {"cannot", "cannot"}, {"do_not", "do not"}} {
		ps = append(ps, validationSpec{
			id: "trigger_lights_" + c.id, kind: kindTrigger, covers: c.cue + " with a trigger on set_lights",
			tools: homeSet, triggers: map[string][]string{"set_lights": {"lights"}},
			turns: []string{"turn on the living room lights but " + c.cue + " unlock the front door"},
		})
	}

	// Each cue before a tool's verb: does it drop that tool's call (F-056's negated clause)?
	for _, c := range verbCues {
		ps = append(ps, validationSpec{
			id: "drop_" + c.id, kind: kindDrop, covers: c.cue + " before lock",
			tools: homeSet, turns: []string{"turn on the kitchen lights and " + c.cue + " lock the front door"},
		})
	}

	drop := func(id, covers, query string) validationSpec {
		return validationSpec{id: "drop_" + id, kind: kindDrop, covers: covers, tools: homeSet, turns: []string{query}}
	}

	ps = append(ps,
		drop("wouldnt", "wouldn't before lock", "turn on the kitchen lights and wouldn't lock the front door"),
		drop("dont_forget", "don't forget to lock", "turn on the kitchen lights and don't forget to lock the front door"),
		drop("wont_play", "won't before play", "turn on the kitchen lights and won't play music"),
		drop("wont_only", "won't before the only call's verb", "won't lock the front door"),
		drop("never_other_verb", "never lock against unlock_door", "never lock the front door"),

		// The only call negated or excluded.
		drop("only_do_not", "do not before the only call's verb", "do not lock the front door"),
		drop("only_dont_bare", "dont before the only call's verb", "dont lock the front door"),
		drop("only_dont_two", "don't before the verb of both calls", "don't lock the front door and don't lock the back door"),
		drop("only_except", "except the only call's room", "turn on the lights except the bedroom"),
		drop("only_except_value", "except a word inside the only call's value", "play jazz everywhere except the kitchen"),
		drop("except_phrase", "except a phrase that holds a call's value",
			"turn off the kitchen and hallway lights except the hallway lights"),

		// How far after a cue the verb may be, and what counts as a cue token.
		drop("reach_two", "two words between don't and lock", "don't wait, guys, lock the front door"),
		drop("reach_three", "three words between don't and lock", "don't be so slow lock the front door"),
		drop("reach_hyphen", "a hyphenated word between don't and lock", "don't wait-up-now lock the front door"),
		drop("reach_digit", "a number among three words between don't and lock", "don't wait 5 min, lock the front door"),
		drop("cue_comma", "don't with a comma before lock", "don't, wait, lock the front door"),
		drop("cue_glued", "don't glued to the word before", "please,don't lock the front door"),
		drop("cue_inside", "don't inside a word before lock", "xdon't lock the front door"),
		validationSpec{
			id: "drop_only_except_trigger", kind: kindDrop, covers: "except the only call's room, with a trigger",
			tools: homeSet, triggers: map[string][]string{"set_lights": {"lights"}},
			turns: []string{"turn on the lights except the bedroom"},
		},
		validationSpec{
			id: "drop_only_do_not_trigger", kind: kindDrop, covers: "do not before the only call's verb, with a trigger",
			tools: homeSet, triggers: map[string][]string{"lock_door": {"door"}},
			turns: []string{"do not lock the front door"},
		},
	)

	// The paired tool (lock_door to unlock_door) and its own reading of a negation (F-056): the model writes
	// lock_door for each, and whether the engine keeps it shows which cues, at which distance before the verb,
	// stop the flip.
	paired := func(id, covers, query string) validationSpec {
		return validationSpec{id: "paired_" + id, kind: kindPaired, covers: covers, tools: homeSet, turns: []string{query}}
	}

	ps = append(ps,
		paired("dont_curly", "the curly don’t before unlock", "turn on the living room lights but don’t unlock the front door"),
		paired("not", "not before unlock", "not unlock the front door"),
		paired("nothing", "not inside a word before unlock", "nothing to unlock the front door"),
		paired("mustnt", "mustn't before unlock", "you mustn't unlock the front door"),
		paired("shouldnt", "shouldn't before unlock", "you shouldn't unlock the front door"),
		paired("dont_wait", "don't 12 bytes before unlock", "don't wait, unlock the front door"),
		paired("dont_delay", "don't 13 bytes before unlock", "don't delay, unlock the front door"),
		paired("dont_linger", "don't 14 bytes before unlock", "don't linger, unlock the front door"),
		paired("not_tomorrow", "not 14 bytes before unlock", "not tomorrow, unlock the front door"),
		paired("never_mind", "never mind before unlock", "never mind, unlock the front door"),
		paired("dont_forget", "don't forget to unlock", "don't forget to unlock the front door"),
		paired("kindly", "no cue before unlock", "kindly unlock the front door"),
	)

	g := func(id, covers, tools, system string, turns ...string) validationSpec {
		return validationSpec{id: "ground_" + id, kind: kindGrounding, covers: covers, tools: tools, system: system, turns: turns}
	}

	return append(ps,
		// Nested objects and arrays.
		g("nested_time", "a time in a nested object", scheduleCallTool, "", "schedule a call with Anna at 3pm"),
		g("nested_and_top", "a time at the top level of two calls, an empty nested object", snoozeTool, "",
			"set an alarm at 6:45 and snooze until 7:10"),
		g("array_strings", "times in an array of strings", alarmsTool, "", "alarms at 6:45 and 7:15"),
		g("array_numbers", "numbers in an array", pricesTool, "", "log prices of $1,200.00 and $2,300.00"),
		g("two_calls", "the same field ungrounded in two calls", assistantSet, "", "wake me up at 6:45 and at 7:15"),

		// The system facts.
		g("system_string", "a place only the system facts name", assistantSet, livesInHamburg, "what's the weather where I live"),
		g("system_string_stated", "the same place in the request", assistantSet, livesInHamburg, "what's the weather in Hamburg"),
		g("system_number", "a number only the system facts state", assistantSet, defaultTimer, "set the default timer"),

		// Earlier turns.
		g("turn_user_string", "a name only an earlier user turn states", assistantSet, "",
			"text Anna that I'm running late", "and text her that I'll be there at 6"),
		g("turn_user_number", "a number only an earlier user turn states", assistantSet, "",
			"set a timer for 10 minutes for the pasta", "set the same timer for the rice"),
		g("turn_answer_only", "a time only the model's earlier answer writes", assistantSet, "",
			"wake me up at 6:45 tomorrow", "set another alarm at the same time for Monday"),
		g("turn_tool_result", "values the tool result being answered and an earlier one hold", catalogueSet, "",
			"search the web for pizza places nearby", pizzaPlace, "call them"),
		g("turn_tool_result_address", "an address only an earlier tool result holds", catalogueSet, "",
			"search the web for pizza places nearby", pizzaPlace, "navigate there"),
		g("turn_negation_earlier", "don't only in an earlier turn", homeSet, "",
			"don't turn off the bedroom lights", "turn on the kitchen lights"),
	)
}

// cmdValidation records the CLI's answers to requests that exercise the envelope's validation block (negation,
// ungrounded), how a negation cue meets triggers and the repair's drop of negated calls, to set the port against.
func cmdValidation(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := validationFixture{
		header: o.header("needle --model needle3.cact --tools <tools> --system <system file> --threads 4 " +
			"--prompt <query>; with more than one turn --serve, POST /complete per turn"),
		System: o.corpus.System,
	}

	for _, p := range validationProbes() {
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

		answers, err := o.validationAnswers(ctx, p.id, tools, system, p.turns)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		rec := validationProbe{
			ID: p.id, Kind: p.kind, Covers: p.covers, Tools: tools, Triggers: p.triggers, System: p.system,
			Turns: p.turns,
		}

		for _, a := range answers {
			rec.Answers = append(rec.Answers, compact(a))
		}

		fx.Probes = append(fx.Probes, rec)
		_, _ = fmt.Fprintf(log, "%-32s %s\n", p.id, summary(answers[len(answers)-1]))
	}

	return writeJSON(cfg.outDir, "validation.json", fx)
}

// validationAnswers answers the turns over tools and the system facts system: one turn with the CLI's --prompt, more
// on one `needle --serve` session.
func (o *oracle) validationAnswers(ctx context.Context, id, tools, system string, turns []string) ([]node, error) {
	if len(turns) == 1 {
		a, err := o.repairAnswer(ctx, "validation-"+id, tools, system, turns[0])
		if err != nil {
			return nil, err
		}

		return []node{a}, nil
	}

	p := filepath.Join(o.scratch, "tools-validation-"+id+".json")

	err := writeFile(p, []byte(tools))
	if err != nil {
		return nil, err
	}

	sys, err := o.systemFile(system)
	if err != nil {
		return nil, err
	}

	s, err := startServe(ctx, o.needle(), []string{
		flagModel, o.needle3Path(), flagTools, p, flagSystem, sys, "--threads", cliThreads,
	})
	if err != nil {
		return nil, err
	}
	defer s.close()

	answers := make([]node, 0, len(turns))

	for i, t := range turns {
		a, err := s.complete(ctx, t)
		if err != nil {
			return nil, fmt.Errorf("turn %d: %w", i+1, err)
		}

		answers = append(answers, a)
	}

	return answers, nil
}
