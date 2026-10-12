//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

// normaliseToolset is the corpus tool set the normalisation probes rewrite.
const normaliseToolset = "assistant"

// normaliseProbe is needle_init's result for one tool set that needs normalising: the tools JSON as sent, byte for
// byte, and the static prefix's token count or the error.
type normaliseProbe struct {
	ID    string `json:"id"`
	Tools string `json:"tools"`
	Init  int    `json:"init"`
	Error string `json:"error,omitempty"`
}

// normaliseAnswer is the CLI's answer to a query over one of the probes' tool sets.
type normaliseAnswer struct {
	Probe  string          `json:"probe"`
	Query  string          `json:"query"`
	Answer json.RawMessage `json:"answer"`
}

// normaliseFixture is testdata/text/engine/normalise.json.
type normaliseFixture struct {
	header

	System  string            `json:"system"`
	Probes  []normaliseProbe  `json:"probes"`
	Answers []normaliseAnswer `json:"answers"`
}

// normaliseQueries are asked of the tool sets in normaliseAnswered: a call to the first tool and one to the last,
// which has a two-word property.
var normaliseQueries = []string{
	"set a timer for 10 minutes for the pasta",
	"add a meeting called standup on 2026-10-13 at 09:00 for 15 minutes",
}

// normaliseAnswered are the probes the CLI answers normaliseQueries over.
var normaliseAnswered = []string{"assistant_camel", "assistant_openai", "collide_tools"}

// cmdNormalise records how the engine prepares tools before it renders them: needle_init's count on tool sets with
// OpenAI-style entries, camelCase names, triggers, extra keys, nested schemas and colliding names, and the CLI's
// answers over three of them, which show the names the calls carry.
func cmdNormalise(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	lib, err := openLibrary(cfg.engineDir, o.needle3Path())
	if err != nil {
		return err
	}

	sets, err := normaliseSets(o.corpus.tools(normaliseToolset))
	if err != nil {
		return err
	}

	fx := normaliseFixture{
		header: o.header("libneedle3: needle_init(system, tools, NULL); " +
			"needle --model needle3.cact --tools <probe tools> --system <system file> --threads 4 --prompt <query>"),
		System: o.corpus.System,
	}

	for _, s := range sets {
		rc, err := lib.session(o.corpus.System, s.tools)

		p := normaliseProbe{ID: s.id, Tools: s.tools, Init: int(rc)}
		if err != nil {
			if rc >= 0 {
				return err
			}

			p.Error = lib.lastError()
		}

		fx.Probes = append(fx.Probes, p)
		_, _ = fmt.Fprintf(log, "%-28s init %4d %s\n", s.id, p.Init, p.Error)
	}

	for _, id := range normaliseAnswered {
		tools := ""

		for _, s := range sets {
			if s.id == id {
				tools = s.tools
			}
		}

		for _, q := range normaliseQueries {
			a, err := o.normaliseAnswer(ctx, id, tools, q)
			if err != nil {
				return err
			}

			fx.Answers = append(fx.Answers, normaliseAnswer{Probe: id, Query: q, Answer: compact(a)})
			_, _ = fmt.Fprintf(log, "%-18s %-70q %s\n", id, q, summary(a))
		}
	}

	return writeJSON(cfg.outDir, "normalise.json", fx)
}

// normaliseAnswer answers query with the CLI over the tool set tools.
func (o *oracle) normaliseAnswer(ctx context.Context, id, tools, query string) (node, error) {
	p := filepath.Join(o.scratch, "tools-normalise-"+id+".json")

	err := writeFile(p, []byte(tools))
	if err != nil {
		return node{}, err
	}

	sys, err := o.systemFile(o.corpus.System)
	if err != nil {
		return node{}, err
	}

	return answer(ctx, o.needle(), []string{
		flagModel, o.needle3Path(), flagTools, p, flagSystem, sys, "--threads", cliThreads, "--prompt", query,
	})
}

// normaliseSet is one probe's tool set.
type normaliseSet struct{ id, tools string }

// camelNames are the corpus assistant tools' snake names and the camelCase spellings the probes give them.
var camelNames = [][2]string{
	{"set_timer", "setTimer"},
	{"set_alarm", "setAlarm"},
	{"send_message", "sendMessage"},
	{"get_weather", "getWeather"},
	{"create_event", "createEvent"},
	{"duration_minutes", "durationMinutes"},
}

// snakeProbeNames are single-tool names whose snake_case the counts pin: camel humps, acronyms, digits,
// separators, edges and non-ASCII.
var snakeProbeNames = []string{
	"setTimer", "SetTimer", "HTTPServer", "getURL2", "getHTTPResponseCode", "XMLHttpRequest", "IOError",
	"turnOnTV", "get2FACode", "v2Api", "a1B2c3", "lightsOn42Now", "ABC", "x", "already_snake", "Mixed_Case",
	"set timer", "_private", "trailing_", "__zoom_in__", "kebab-case-name", "send.message", "set-Timer",
	"ÄpfelZählen", "naïveBayes", "getÜbersicht",
}

// snakeProbeProps are property names probed the same way.
var snakeProbeProps = []string{"minutesLeft", "HTTPCode", "maxRetryCount", "userID", "max count", "größeWert"}

// normaliseLiterals are hand-written probes: nested schemas, colliding names, OpenAI-style variants and entries the
// engine leaves alone.
var normaliseLiterals = []normaliseSet{
	{"nested_object_items", `[{"name":"setThing","description":"Do it","parameters":{"type":"object","properties":{` +
		`"outerKey":{"type":"object","properties":{"innerKey":{"type":"string"}},"required":["innerKey"]},` +
		`"listKey":{"type":"array","items":{"type":"object","properties":{"itemKey":{"type":"string"}}}},` +
		`"enumKey":{"type":"string","enum":["fooBar","bazQux"]}},"required":["outerKey"]}}]`},
	{"nested_anyof", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{"a":{"anyOf":[` +
		`{"type":"object","properties":{"innerKey":{"type":"string"}}},{"type":"string"}]}}}}]`},
	{"nested_oneof", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{"a":{"oneOf":[` +
		`{"type":"object","properties":{"innerKey":{"type":"string"}}},{"type":"string"}]}}}}]`},
	{"nested_allof", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{"a":{"allOf":[` +
		`{"type":"object","properties":{"innerKey":{"type":"string"}}}]}}}}]`},
	{"nested_additional", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{"a":{` +
		`"type":"object","additionalProperties":{"type":"object","properties":{"innerKey":{"type":"string"}}}}}}}]`},
	{"nested_property_named_properties", `[{"name":"t","description":"Do it","parameters":{"type":"object",` +
		`"properties":{"properties":{"type":"object","properties":{"innerKey":{"type":"string"}}}}}}]`},
	{"untouched_defs", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{"a":{` +
		`"$ref":"#/$defs/MyThing"}},"$defs":{"MyThing":{"type":"object","properties":{"innerKey":{"type":"string"}}}}}}]`},
	{"untouched_items_list", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{"a":{` +
		`"type":"array","items":[{"type":"object","properties":{"innerKey":{"type":"string"}}}]}}}}]`},
	{"untouched_required_unknown", `[{"name":"t","description":"Do it","parameters":{"type":"object",` +
		`"properties":{"maxCount":{"type":"string"}},"required":["otherKey"]}}]`},
	{"untouched_nested_triggers", `[{"name":"t","description":"Do it","parameters":{"type":"object",` +
		`"properties":{"a":{"type":"string"}},"triggers":["x"]}}]`},
	{"untouched_function_string", `[{"type":"function","function":"x"}]`},
	{"untouched_name_number", `[{"name":5,"description":"x"}]`},
	{"no_parameters", `[{"name":"getTime","description":"x"}]`},
	{"collide_tools", `[{"name":"setTimer","description":"Start a countdown timer","parameters":{"type":"object",` +
		`"properties":{"minutes":{"type":"integer","description":"Duration in minutes"},"label":{"type":"string",` +
		`"description":"What the timer is for"}},"required":["minutes"]}},{"name":"set_timer","description":` +
		`"Start a kitchen countdown timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer",` +
		`"description":"Duration in minutes"},"label":{"type":"string","description":"What the timer is for"}},` +
		`"required":["minutes"]}}]`},
	{"collide_tools_three", `[{"name":"setTimer","description":"Do it","parameters":{"type":"object",` +
		`"properties":{"x":{"type":"string"}},"required":["x"]}},{"name":"set_timer","description":"Do it",` +
		`"parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}},{"name":"SetTimer",` +
		`"description":"Do it","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}]`},
	{"collide_tools_identical", `[{"name":"set_timer","description":"Do it","parameters":{"type":"object",` +
		`"properties":{"x":{"type":"string"}},"required":["x"]}},{"name":"set_timer","description":"Do it",` +
		`"parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}]`},
	{"collide_props", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{` +
		`"maxCount":{"type":"string"},"max_count":{"type":"integer"}},"required":["maxCount","max_count"]}}]`},
	{"collide_props_three", `[{"name":"t","description":"Do it","parameters":{"type":"object","properties":{` +
		`"maxCount":{"type":"string"},"max_count":{"type":"integer"},"MaxCount":{"type":"boolean"}}}}]`},
}

// normaliseSets are the probes: the corpus assistant tools rewritten (OpenAI-style, camelCase, triggers, extra keys
// that stay), single tools and properties for snake_case, and normaliseLiterals.
func normaliseSets(assistant string) ([]normaliseSet, error) {
	var tools []json.RawMessage

	err := json.Unmarshal([]byte(assistant), &tools)
	if err != nil {
		return nil, fmt.Errorf("toolset %s: %w", normaliseToolset, err)
	}

	each := func(f func(t string) string) string {
		parts := make([]string, len(tools))
		for i, t := range tools {
			parts[i] = f(string(t))
		}

		return "[" + strings.Join(parts, ",") + "]"
	}

	camel := func(t string) string {
		for _, n := range camelNames {
			t = strings.ReplaceAll(t, `"`+n[0]+`"`, `"`+n[1]+`"`)
		}

		return t
	}
	openai := func(t string) string { return `{"type":"function","function":` + t + `}` }
	add := func(keys string) func(string) string {
		return func(t string) string { return strings.TrimSuffix(t, "}") + "," + keys + "}" }
	}
	triggers := add(`"triggers":["\\b(timer|alarm)\\b"]`)

	sets := []normaliseSet{
		{"assistant", each(func(t string) string { return t })},
		{"assistant_openai", each(openai)},
		{"assistant_camel", each(camel)},
		{"assistant_triggers", each(triggers)},
		{"assistant_openai_camel_triggers", each(func(t string) string { return openai(triggers(camel(t))) })},
		{"assistant_openai_extra_keys", each(func(t string) string { return add(`"strict":true,"x-foo":1`)(openai(t)) })},
		{"assistant_openai_type_tool", each(func(t string) string { return `{"type":"tool","function":` + t + `}` })},
		{"assistant_kept_keys", each(add(`"examples":["x"],"x-foo":1,"strict":true,"title":"T","returns":{"type":"string"}`))},
	}

	single := func(name, prop string) (string, error) {
		n, err := json.Marshal(name)
		if err != nil {
			return "", fmt.Errorf("probe name: %w", err)
		}

		p, err := json.Marshal(prop)
		if err != nil {
			return "", fmt.Errorf("probe property: %w", err)
		}

		return fmt.Sprintf(`[{"name":%s,"description":"Do it","parameters":{"type":"object","properties":{%s:`+
			`{"type":"string"}},"required":[%s]}}]`, n, p, p), nil
	}

	for _, name := range snakeProbeNames {
		js, err := single(name, "x")
		if err != nil {
			return nil, err
		}

		sets = append(sets, normaliseSet{"name " + name, js})
	}

	for _, prop := range snakeProbeProps {
		js, err := single("set_timer", prop)
		if err != nil {
			return nil, err
		}

		sets = append(sets, normaliseSet{"property " + prop, js})
	}

	return append(sets, normaliseLiterals...), nil
}
