//go:build oracle

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// turnsToolset is the corpus tool set the conversations run over.
const turnsToolset = "assistant"

// turnInput is one turn's input as Head + Unit×Repeat + Tail, so that the probes, about a thousand tokens each, stay
// small in the fixture. Reset marks a POST /reset instead of a turn.
type turnInput struct {
	Head   string `json:"head,omitempty"`
	Unit   string `json:"unit,omitempty"`
	Repeat int    `json:"repeat,omitempty"`
	Tail   string `json:"tail,omitempty"`
	Reset  bool   `json:"reset,omitempty"`
}

func (in turnInput) text() string { return in.Head + strings.Repeat(in.Unit, in.Repeat) + in.Tail }

// turnStep is one step of a recorded conversation: its input (with the SHA-256 of the text) and the answer.
type turnStep struct {
	Input  turnInput       `json:"input"`
	SHA256 string          `json:"sha256,omitempty"`
	Answer json.RawMessage `json:"answer,omitempty"`
}

// turnScenario is one conversation on a fresh `needle --serve` session: --max Max, --fail-input-overflow when
// Refuse, the steps from a reset.
type turnScenario struct {
	ID     string     `json:"id"`
	Covers string     `json:"covers"`
	Max    int        `json:"max"`
	Refuse bool       `json:"fail_input_overflow"`
	Steps  []turnStep `json:"steps"`
}

// turnsFixture is testdata/text/engine/turns.json.
type turnsFixture struct {
	header

	System    string         `json:"system"`
	Toolset   string         `json:"toolset"`
	Context   int            `json:"context"`
	Scenarios []turnScenario `json:"scenarios"`
}

// Real turns of the conversations.
var (
	turnCall     = turnInput{Tail: "set a timer for 10 minutes for the pasta"}
	turnTwoCalls = turnInput{Tail: "text Anna that I'm running late and set a timer for 20 minutes"}
	turnRefusal  = turnInput{Tail: "order a large pepperoni pizza"}
	turnSuppress = turnInput{Tail: "wake me up at 6:45 tomorrow"}
	turnResult   = turnInput{Tail: `[{"status": "ok", "timer_id": 7}]`}
	turnReset    = turnInput{Reset: true}
)

// fillSentence and spacedObject are the units the plain and JSON probes repeat.
const (
	fillSentence = "the weather was nice and we walked along the river. "
	spacedObject = `{"status": "ok", "note": "fine"}, `
)

// Probes: inputs of about a thousand tokens, more than a turn may take at --max 7000 (8192 − 314 − 7000 = 878), so
// the engine refuses them and reports their length. The JSON ones differ in layout, which changes their token count.
var (
	probePlain   = turnInput{Unit: fillSentence, Repeat: 100}
	probeCompact = turnInput{Head: "[", Unit: `{"status":"ok","note":"fine"},`, Repeat: 120, Tail: `{"status":"ok"}]`}
	probeDefault = turnInput{Head: "[", Unit: spacedObject, Repeat: 100, Tail: `{"status": "ok"}]`}
	probePretty  = turnInput{
		Head: "[\n", Unit: "  {\n    \"status\": \"ok\",\n    \"note\": \"fine\"\n  },\n", Repeat: 150, Tail: "  {\n    \"status\": \"ok\"\n  }\n]",
	}
	probeObject  = turnInput{Head: `{"results": [`, Unit: spacedObject, Repeat: 100, Tail: `{"status": "ok"}]}`}
	probeString  = turnInput{Head: `"`, Unit: "the weather was nice ", Repeat: 200, Tail: `"`}
	probeNumber  = turnInput{Head: "1", Unit: "0123456789", Repeat: 300}
	probePadded  = turnInput{Head: "  \n[", Unit: spacedObject, Repeat: 100, Tail: "{\"status\": \"ok\"}]\n  "}
	probeInvalid = turnInput{Head: "[", Unit: spacedObject, Repeat: 100, Tail: `{"status": "ok"}`}
	probeTrue    = turnInput{Head: "true", Unit: "\n", Repeat: 1000}

	// How the engine writes a tool result back: floats, exponents, escapes, non-ASCII, large integers, repeated keys.
	probeFloat   = turnInput{Head: "[", Unit: `{"v": 1.50, "w": 2.0, "x": 0.1, "y": -0.0}, `, Repeat: 50, Tail: `{"v": 1.0}]`}
	probeExp     = turnInput{Head: "[", Unit: `{"v": 1e2, "w": 1E-7, "x": 2.5e+3, "y": 1.0e20}, `, Repeat: 40, Tail: `{"v": 1e0}]`}
	probeUnicode = turnInput{Head: "[", Unit: `{"v": "Grüße été naïve"}, `, Repeat: 60, Tail: `{"v": "é"}]`}
	probeRawUTF8 = turnInput{Head: "[", Unit: `{"v": "Grüße été naïve"}, `, Repeat: 60, Tail: `{"v": "é"}]`}
	probeEscape  = turnInput{Head: "[", Unit: `{"v": "a\/b A \t \" \\ <&>"}, `, Repeat: 60, Tail: `{"v": "\/"}]`}
	probeBigInt  = turnInput{Head: "[", Unit: `{"v": 123456789012345678901234567890, "w": -0}, `, Repeat: 90, Tail: `{"v": 1}]`}
	probeDupKey  = turnInput{Head: "[", Unit: `{"a": 1, "b": 2, "a": "three"}, `, Repeat: 80, Tail: `{"a": 1}]`}
)

// bsU is a JSON \u escape's start, spelled out so that no tool rewrites it.
const bsU = `\` + "u"

// escapeProbe is a tool result whose strings hold one escape or special character, esc as written in the JSON.
func escapeProbe(esc string) turnInput {
	return turnInput{Head: "[", Unit: `{"v": "a` + esc + `b"}, `, Repeat: 150, Tail: `{"v": "a"}]`}
}

// escapeProbes are the characters whose spelling in a tool result the engine's counts pin, one scenario each.
var escapeProbes = []struct{ id, esc string }{
	{"slash", "/"},
	{"slash_escaped", `\/`},
	{"tab", `\t`},
	{"newline", `\n`},
	{"quote", `\"`},
	{"backslash", `\\`},
	{"lt", "<"},
	{"amp", "&"},
	{"gt", ">"},
	{"u_nonascii", bsU + "00fc"},
	{"u_ascii", bsU + "0041"},
	{"u_control", bsU + "0001"},
	{"u_upper", bsU + "00FC"},
	{"lt_amp_gt", "<&>"},
	{"space_lt", " <"},
	{"lt_space", "< "},
	{"space_amp_space", " & "},
	{"backslash_space_lt", `\\ <`},
	{"quote_space_lt", `\" <`},
	{"tab_space_lt", `\t <`},
	{"lt_slash", "</"},
}

// turnSpec is one scenario to record: its id, what it covers, --max, --fail-input-overflow and the steps.
type turnSpec struct {
	id, covers string
	max        int
	refuse     bool
	steps      []turnInput
}

// turnScenarios are the recorded conversations; the last step of each is the probe.
var turnScenarios = []turnSpec{
	{"first_plain", "the first turn continues the open user message", 7000, true, []turnInput{probePlain}},
	{"first_json", "JSON as the first turn is a user turn", 7000, true, []turnInput{probeCompact}},
	{"later_plain", "a later turn opens its own user message", 7000, true, []turnInput{turnCall, probePlain}},
	{"after_reset", "reset starts over: the first turn again", 7000, true, []turnInput{turnCall, turnReset, probePlain}},
	{"after_refused_first", "a refused first turn still counts as the first", 7000, true, []turnInput{probePlain, probePlain}},
	{"tool_compact", "JSON after a call is a tool result", 7000, true, []turnInput{turnCall, probeCompact}},
	{"tool_default", "a tool result keeps its layout (or not)", 7000, true, []turnInput{turnCall, probeDefault}},
	{"tool_pretty", "a pretty-printed tool result", 7000, true, []turnInput{turnCall, probePretty}},
	{"tool_padded", "a tool result with white space around it", 7000, true, []turnInput{turnCall, probePadded}},
	{"tool_object", "an object as the tool result", 7000, true, []turnInput{turnCall, probeObject}},
	{"tool_string", "a JSON string as the tool result", 7000, true, []turnInput{turnCall, probeString}},
	{"tool_number", "a JSON number as the tool result", 7000, true, []turnInput{turnCall, probeNumber}},
	{"tool_true", "a JSON literal as the tool result", 7000, true, []turnInput{turnCall, probeTrue}},
	{"tool_invalid", "invalid JSON after a call is a user turn", 7000, true, []turnInput{turnCall, probeInvalid}},
	{"tool_float", "float spelling in a tool result", 7000, true, []turnInput{turnCall, probeFloat}},
	{"tool_exp", "exponent spelling in a tool result", 7000, true, []turnInput{turnCall, probeExp}},
	{"tool_unicode", "escaped non-ASCII in a tool result", 7000, true, []turnInput{turnCall, probeUnicode}},
	{"tool_raw_utf8", "raw non-ASCII in a tool result", 7000, true, []turnInput{turnCall, probeRawUTF8}},
	{"tool_escape", "escapes in a tool result", 7000, true, []turnInput{turnCall, probeEscape}},
	{"tool_bigint", "large integers and -0 in a tool result", 7000, true, []turnInput{turnCall, probeBigInt}},
	{"tool_dupkey", "repeated keys in a tool result", 7000, true, []turnInput{turnCall, probeDupKey}},
	{"after_two_calls", "JSON after two calls", 7000, true, []turnInput{turnTwoCalls, probeCompact}},
	{"after_refusal", "JSON after a refusal (a call with no calls)", 7000, true, []turnInput{turnRefusal, probeCompact}},
	{"after_suppressed", "JSON after a suppressed call", 7000, true, []turnInput{turnSuppress, probeCompact}},
	{"after_tool_result", "JSON after the answer to a tool result", 7000, true, []turnInput{turnCall, turnResult, probeCompact}},
	{"after_refused_tool", "JSON after a refused tool result", 7000, true, []turnInput{turnCall, probeCompact, probeCompact}},
	{"after_refused_user", "JSON after a refused user turn that followed a call", 7000, true, []turnInput{turnCall, probePlain, probeCompact}},
	{"three_turns", "a third user turn", 7000, true, []turnInput{turnCall, turnRefusal, probePlain}},
	{"budget_512", "the budget at the default --max", 512, true, []turnInput{{Unit: fillSentence, Repeat: 700}}},
	{"budget_100", "the budget at --max 100", 100, true, []turnInput{{Unit: fillSentence, Repeat: 750}}},
	{
		"trim_request_last", "without the flag an oversized turn is trimmed: the request at its end", 7000, false,
		[]turnInput{{Unit: fillSentence, Repeat: 150, Tail: "Set a timer for 25 minutes for the eggs."}},
	},
	{
		"trim_request_first", "without the flag an oversized turn is trimmed: the request at its start", 7000, false,
		[]turnInput{{Head: "Set a timer for 25 minutes for the eggs. ", Unit: fillSentence, Repeat: 150}},
	},
}

// cmdTurns records conversations with `needle --serve`: each scenario on a fresh session, its turns answered and a
// last oversized probe that --fail-input-overflow refuses with its length and the room a turn has, which pins the
// rendering of a first and a later user turn, a tool result, reset, and what makes a turn a tool result.
func cmdTurns(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	_, budget, err := cactHeads(o.needle3Path())
	if err != nil {
		return err
	}

	fx := turnsFixture{
		header: o.header("needle --model needle3.cact --tools <assistant> --system <system file> --threads 4 " +
			"--max <max> [--fail-input-overflow] --serve; POST /complete per turn, POST /reset"),
		System: o.corpus.System, Toolset: turnsToolset, Context: budget,
	}

	specs := append([]turnSpec{}, turnScenarios...)
	for _, e := range escapeProbes {
		specs = append(specs, turnSpec{
			"escape_" + e.id, "the spelling of " + e.esc + " in a tool result", 7000, true,
			[]turnInput{turnCall, escapeProbe(e.esc)},
		})
	}

	for _, sc := range specs {
		rec, err := o.turnScenario(ctx, sc.id, sc.covers, sc.max, sc.refuse, sc.steps)
		if err != nil {
			return fmt.Errorf("%s: %w", sc.id, err)
		}

		fx.Scenarios = append(fx.Scenarios, rec)

		last := rec.Steps[len(rec.Steps)-1]
		_, _ = fmt.Fprintf(log, "%-20s %s\n", sc.id, last.Answer)
	}

	return writeJSON(cfg.outDir, "turns.json", fx)
}

// turnScenario runs one conversation on its own server.
func (o *oracle) turnScenario(ctx context.Context, id, covers string, maxNew int, refuse bool, steps []turnInput) (turnScenario, error) {
	args, err := o.baseArgs(turnsToolset, o.corpus.System)
	if err != nil {
		return turnScenario{}, err
	}

	args = append(args, "--max", strconv.Itoa(maxNew))
	if refuse {
		args = append(args, "--fail-input-overflow")
	}

	s, err := startServe(ctx, o.needle(), args)
	if err != nil {
		return turnScenario{}, err
	}
	defer s.close()

	rec := turnScenario{ID: id, Covers: covers, Max: maxNew, Refuse: refuse}

	for _, in := range steps {
		if in.Reset {
			err := s.reset(ctx)
			if err != nil {
				return turnScenario{}, err
			}

			rec.Steps = append(rec.Steps, turnStep{Input: in})

			continue
		}

		text := in.text()
		sum := sha256.Sum256([]byte(text))

		a, err := s.complete(ctx, text)
		if err != nil {
			return turnScenario{}, err
		}

		rec.Steps = append(rec.Steps, turnStep{Input: in, SHA256: hex.EncodeToString(sum[:]), Answer: compact(a)})
	}

	return rec, nil
}
