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

// triggerProbe is the CLI's answer to one query over a corpus tool set whose tools carry the given triggers: the
// tools JSON as sent, the answer without its timings, and the CLI's stderr (where it warns of invalid triggers).
type triggerProbe struct {
	ID       string              `json:"id"`
	Toolset  string              `json:"toolset"`
	Triggers map[string][]string `json:"triggers"`
	Tools    string              `json:"tools"`
	Query    string              `json:"query"`
	Answer   json.RawMessage     `json:"answer"`
	Stderr   string              `json:"stderr"`
}

// triggersFixture is testdata/text/engine/triggers.json.
type triggersFixture struct {
	header

	System string         `json:"system"`
	Probes []triggerProbe `json:"probes"`
	// NoTools is the answer to a query over an empty tool list.
	NoTools json.RawMessage `json:"no_tools"`
}

// triggerProbes pair corpus queries with triggers: one that matches a call the model suppresses below confidence
// 0.1, ones that route a refusal or another tool's request to the triggered tool, matches on a negated request,
// case-insensitive matching, an invalid regex and a trigger that matches nothing.
var triggerProbes = []struct {
	id, toolset, query string
	triggers           map[string][]string
}{
	{"alarm_untriggered", "assistant", "wake me up at 6:45 tomorrow", nil},
	{"alarm_triggered", "assistant", "wake me up at 6:45 tomorrow", map[string][]string{"set_alarm": {"wake me"}}},
	{"alarm_invalid", "assistant", "wake me up at 6:45 tomorrow", map[string][]string{"set_alarm": {"("}}},
	{"alarm_no_match", "assistant", "wake me up at 6:45 tomorrow", map[string][]string{"set_alarm": {`\bsnooze\b`}}},
	{
		"timer_triggered", "assistant", "set a timer for 10 minutes for the pasta",
		map[string][]string{"set_timer": {`\btimer\b`}},
	},
	{
		"weather_to_timer", "assistant", "what's the weather in Berlin on Friday",
		map[string][]string{"set_timer": {"weather"}},
	},
	{"pizza_to_lights", "home", "order a large pepperoni pizza", map[string][]string{"set_lights": {`\bpizza\b`}}},
	{"capital_upper", "home", "what's the capital of France", map[string][]string{"set_lights": {"CAPITAL"}}},
	{
		"negation_triggered", "home", "turn on the living room lights but don't unlock the front door",
		map[string][]string{"set_lights": {`\b(turn|switch)\b.*\b(on|off)\b`}},
	},
	{
		"negation_two_triggered", "home", "turn on the living room lights but don't unlock the front door",
		map[string][]string{"set_lights": {"lights"}, "lock_door": {"door"}},
	},
	{
		"negation_only_triggered", "home", "don't turn off the bedroom lights",
		map[string][]string{"set_lights": {"lights"}},
	},
}

// noToolsQuery is asked over an empty tool list.
const noToolsQuery = "set a timer for 10 minutes for the pasta"

// cmdTriggers records what a tool's "triggers" do to the CLI's answer (llms.txt: case-insensitive regexes that route
// a matching request to the tool, require a call and ship it below the confidence floor) and the answer without
// tools.
func cmdTriggers(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := triggersFixture{
		header: o.header("needle --model needle3.cact --tools <tools with triggers> --system <system file> " +
			"--threads 4 --prompt <query>"),
		System: o.corpus.System,
	}

	for _, p := range triggerProbes {
		tools, err := withTriggers(o.corpus.tools(p.toolset), p.triggers)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		a, stderr, err := o.triggerAnswer(ctx, p.id, tools, p.query)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		fx.Probes = append(fx.Probes, triggerProbe{
			ID: p.id, Toolset: p.toolset, Triggers: p.triggers, Tools: tools, Query: p.query,
			Answer: compact(a), Stderr: stderr,
		})
		_, _ = fmt.Fprintf(log, "%-24s %s\n", p.id, summary(a))
	}

	a, _, err := o.triggerAnswer(ctx, "no_tools", "[]", noToolsQuery)
	if err != nil {
		return fmt.Errorf("no tools: %w", err)
	}

	fx.NoTools = compact(a)
	_, _ = fmt.Fprintf(log, "%-24s %s\n", "no_tools", summary(a))

	return writeJSON(cfg.outDir, "triggers.json", fx)
}

// triggerAnswer answers query with the CLI over tools and returns the answer and the CLI's stderr.
func (o *oracle) triggerAnswer(ctx context.Context, id, tools, query string) (node, string, error) {
	p := filepath.Join(o.scratch, "tools-triggers-"+id+".json")

	err := writeFile(p, []byte(tools))
	if err != nil {
		return node{}, "", err
	}

	sys, err := o.systemFile(o.corpus.System)
	if err != nil {
		return node{}, "", err
	}

	res, err := runCLI(ctx, o.needle(), []string{
		flagModel, o.needle3Path(), flagTools, p, flagSystem, sys, "--threads", cliThreads, "--prompt", query,
	})
	if err != nil {
		return node{}, "", err
	}

	if res.Exit != 0 {
		return node{}, "", fmt.Errorf("exit %d: %s", res.Exit, res.Stderr)
	}

	a, err := normaliseResponse(res.stdout)

	return a, res.Stderr, err
}

// withTriggers is the tool set toolset (a JSON array) with a "triggers" key appended to the tools that triggers
// names.
func withTriggers(toolset string, triggers map[string][]string) (string, error) {
	var tools []json.RawMessage

	err := json.Unmarshal([]byte(toolset), &tools)
	if err != nil {
		return "", err
	}

	parts := make([]string, len(tools))

	for i, t := range tools {
		var named struct {
			Name string `json:"name"`
		}

		err = json.Unmarshal(t, &named)
		if err != nil {
			return "", err
		}

		parts[i] = string(t)

		if re, ok := triggers[named.Name]; ok {
			js, err := json.Marshal(re)
			if err != nil {
				return "", err
			}

			parts[i] = strings.TrimSuffix(parts[i], "}") + `,"triggers":` + string(js) + "}"
		}
	}

	return "[" + strings.Join(parts, ",") + "]", nil
}
