//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

// grammarProbe is the CLI's answer to one query over a tool whose schema the decode grammar has to enforce against
// what the model would write freely: the tools JSON as sent and the answer without its timings.
type grammarProbe struct {
	ID     string          `json:"id"`
	Covers string          `json:"covers"`
	Tools  string          `json:"tools"`
	Query  string          `json:"query"`
	Answer json.RawMessage `json:"answer"`
}

// grammarFixture is testdata/text/engine/grammar.json.
type grammarFixture struct {
	header

	System string         `json:"system"`
	Probes []grammarProbe `json:"probes"`
}

// Tools of the grammar probes.
const (
	lightsTool = `{"name":"set_lights","description":"Turn a room's lights on or off and set their brightness",` +
		`"parameters":{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},` +
		`"brightness":{"type":"integer","minimum":0,"maximum":100}},"required":["room","on"]}}`
	timerTool = `{"name":"set_timer","description":"Start a countdown timer","parameters":{"type":"object",` +
		`"properties":{"minutes":{"type":"integer","description":"Duration in minutes"},"label":{"type":"string",` +
		`"description":"What the timer is for"}},"required":["minutes"]}}`
	lockTool = `{"name":"lock_door","description":"Lock a door","parameters":{"type":"object","properties":` +
		`{"door":{"type":"string"}},"required":["door"]%s}}`
	lockQuery = "lock the front door"
)

// grammarProbes each break one schema rule in the free decode: bounds, an integer, an enum, a const, a date, anyOf,
// $ref, keys outside the properties under additionalProperties, item and length bounds, a missing required key.
var grammarProbes = []struct {
	id, covers, tools, query string
}{
	{"bounds_max", "an integer above its maximum", "[" + lightsTool + "]", "set the kitchen lights to 150 percent"},
	{
		"bounds_min_max", "an integer within minimum and maximum",
		`[{"name":"set_timer","description":"Start a countdown timer","parameters":{"type":"object","properties":` +
			`{"minutes":{"type":"integer","minimum":1,"maximum":60}},"required":["minutes"]}}]`,
		"set a timer for 90 minutes",
	},
	{"integer_decimal", "a decimal for an integer", "[" + timerTool + "]", "set a timer for 2.5 minutes"},
	{
		"number_bounds", "a number with exclusive bounds",
		`[{"name":"set_thermostat","description":"Set the heating target temperature","parameters":{"type":"object",` +
			`"properties":{"temperature":{"type":"number","exclusiveMinimum":5,"exclusiveMaximum":30}},` +
			`"required":["temperature"]}}]`,
		"set the heating to 35.5 degrees",
	},
	{
		"enum_unnamed", "an enum value the request does not name",
		`[{"name":"set_fan","description":"Set the fan speed","parameters":{"type":"object","properties":` +
			`{"speed":{"type":"string","enum":["low","medium","high"]}},"required":["speed"]}}]`,
		"turn the fan up to max",
	},
	{
		"const", "a const value",
		`[{"name":"set_timer","description":"Start a countdown timer","parameters":{"type":"object","properties":` +
			`{"minutes":{"type":"integer"},"kind":{"const":"countdown"}},"required":["minutes","kind"]}}]`,
		"set a timer for 5 minutes",
	},
	{
		"date_free", "format: date for a vague date",
		`[{"name":"Invoice","description":"The fields of an invoice","parameters":{"type":"object","properties":` +
			`{"vendor":{"type":"string"},"total":{"type":"number"},"due_date":{"type":"string","format":"date"}},` +
			`"required":["vendor","total","due_date"]}}]`,
		"Invoice from Bolt Ltd, total 40 EUR, due at the end of next month.",
	},
	{
		"any_of", "anyOf of a date and an integer",
		`[{"name":"set_reminder","description":"Remind the user of something","parameters":{"type":"object",` +
			`"properties":{"text":{"type":"string"},"when":{"anyOf":[{"type":"string","format":"date"},` +
			`{"type":"integer","minimum":1,"description":"days from now"}]}},"required":["text","when"]}}]`,
		"remind me in 3 days to water the plants",
	},
	{
		"ref", "a $ref to an object in $defs",
		`[{"name":"send_message","description":"Send a text message to a contact","parameters":{"type":"object",` +
			`"$defs":{"contact":{"type":"object","properties":{"name":{"type":"string"},"phone":{"type":"string"}},` +
			`"required":["name"]}},"properties":{"recipient":{"$ref":"#/$defs/contact"},"message":{"type":"string"}},` +
			`"required":["recipient","message"]}}]`,
		"text Anna that I'm running late",
	},
	{"closed", "no additionalProperties", "[" + fmt.Sprintf(lockTool, "") + "]", lockQuery},
	{
		"additional_true", "additionalProperties: true",
		"[" + fmt.Sprintf(lockTool, `,"additionalProperties":true`) + "]", lockQuery,
	},
	{
		"additional_integer", "additionalProperties of a type",
		"[" + fmt.Sprintf(lockTool, `,"additionalProperties":{"type":"integer"}`) + "]", lockQuery,
	},
	{
		"max_items", "an array above maxItems",
		`[{"name":"add_to_list","description":"Add items to a named list","parameters":{"type":"object",` +
			`"properties":{"list":{"type":"string"},"items":{"type":"array","items":{"type":"string"},"maxItems":2}},` +
			`"required":["list","items"]}}]`,
		"add milk, eggs, bread and butter to the groceries list",
	},
	{
		"max_length", "a string above maxLength",
		`[{"name":"send_message","description":"Send a text message to a contact","parameters":{"type":"object",` +
			`"properties":{"recipient":{"type":"string"},"message":{"type":"string","maxLength":12}},` +
			`"required":["recipient","message"]}}]`,
		"text Anna that I'm running late and will be there in twenty minutes",
	},
	{"required_missing", "a required key the request does not give", "[" + timerTool + "]", "set a timer for the pasta"},
}

// cmdGrammar records the CLI's answers to requests whose free decode would break the tool's schema, to set the decode
// grammar (llms.txt: "a byte-level grammar compiled from your schemas constrains every token") against.
func cmdGrammar(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := grammarFixture{
		header: o.header("needle --model needle3.cact --tools <tools> --system <system file> --threads 4 " +
			"--prompt <query>"),
		System: o.corpus.System,
	}

	for _, p := range grammarProbes {
		a, _, err := o.triggerAnswer(ctx, "grammar-"+p.id, p.tools, p.query)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		fx.Probes = append(fx.Probes, grammarProbe{
			ID: p.id, Covers: p.covers, Tools: p.tools, Query: p.query, Answer: compact(a),
		})
		_, _ = fmt.Fprintf(log, "%-20s %s\n", p.id, summary(a))
	}

	return writeJSON(cfg.outDir, "grammar.json", fx)
}
