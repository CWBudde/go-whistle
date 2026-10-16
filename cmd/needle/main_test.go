package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/needle"
)

// The engine fixtures, relative to the package.
const (
	engineDir      = "../../testdata/text/engine/"
	engineCases    = engineDir + "cases.json"
	engineAnswers  = engineDir + "responses.json"
	engineDepth    = engineDir + "depth.json"
	engineTriggers = engineDir + "triggers.json"
	engineTurns    = engineDir + "turns.json"
	engineIndex    = engineDir + "retrieval.json"
)

func TestHelp(t *testing.T) {
	for _, arg := range []string{"--help", "-h"} {
		h := newHarness()

		if code := h.run(t, arg); code != 0 {
			t.Fatalf("%s: exit %d, stderr %q", arg, code, h.stderr)
		}

		for _, flag := range []string{
			"--model", "--tools", "--system", "--prompt", "--max", "--depth", "--threads",
			"--fail-input-overflow", "--serve", "--port", "--tool-index", "--audio",
		} {
			if !strings.Contains(h.stdout.String(), flag+" ") {
				t.Errorf("%s: the help does not list %s:\n%s", arg, flag, h.stdout)
			}
		}

		if h.opened != 0 {
			t.Errorf("%s opened the model", arg)
		}
	}
}

// TestCommandLineErrors: a wrong command line exits 1 with a message on
// stderr prefixed "needle: ", before the model is opened.
func TestCommandLineErrors(t *testing.T) {
	q := []string{"--prompt", "hi"}

	for _, c := range []struct {
		args []string
		want string
	}{
		{[]string{"--frobnicate"}, "needle: flag provided but not defined: -frobnicate"},
		{[]string{"--prompt", "hi", "extra"}, `needle: unexpected argument "extra"`},
		{nil, "needle: give --prompt <query> or --serve"},
		{[]string{"--prompt", "hi", "--serve"}, "needle: --prompt and --serve exclude each other"},
		{append([]string{"--model", "a/needle3.cact", "--model", "a/whistle.cact"}, q...), "PLAN.md 11.8"},
		{append([]string{"--audio", "clip.wav"}, q...), "PLAN.md 11.8"},
		{append([]string{"--depth", "2"}, q...), "PLAN.md 11.6"},
		{append([]string{"--depth", "19"}, q...), "PLAN.md 11.6"},
		{append([]string{"--depth", "21"}, q...), "needle: cannot load models/needle3/needle3.cact: --depth 21"},
		{append([]string{"--depth", "-3"}, q...), `needle: --depth takes a layer count of 2 or more, not "-3"`},
		{append([]string{"--depth", "x"}, q...), `needle: --depth takes a layer count of 2 or more, not "x"`},
		{append([]string{"--max", "0"}, q...), `needle: --max takes a token count of 1 or more, not "0"`},
		{append([]string{"--max", "lots"}, q...), `needle: --max takes a token count of 1 or more, not "lots"`},
		{append([]string{"--threads", "0"}, q...), `needle: --threads takes a thread count of 1 or more, not "0"`},
		{[]string{"--serve", "--port", "65536"}, `needle: --port takes a port number of 0 to 65535, not "65536"`},
		{append([]string{"--tools", filepath.Join(t.TempDir(), "none.json")}, q...), "needle: cannot read --tools"},
	} {
		h := newHarness()

		code := h.run(t, c.args...)
		if code != 1 || !strings.HasPrefix(h.stderr.String(), "needle: ") || !strings.Contains(h.stderr.String(), c.want) {
			t.Errorf("%q: exit %d, stderr %q; want exit 1 and %q", c.args, code, h.stderr, c.want)
		}

		if h.stdout.Len() != 0 || h.opened != 0 {
			t.Errorf("%q: stdout %q, model opened %d times", c.args, h.stdout, h.opened)
		}
	}
}

// TestDepthMatchesEngine: --depth refuses 0 and 1 with the engine's exit
// code and message byte for byte (depth.json) and 21 with its exit code,
// and runs the full depth, 20, as no --depth does. 2 to 19 wait for PLAN.md
// 11.6 (TestCommandLineErrors).
func TestDepthMatchesEngine(t *testing.T) {
	var fx struct {
		FullDepth int `json:"full_depth"`
		Refused   map[string]struct {
			Exit   int    `json:"exit"`
			Stderr string `json:"stderr"`
		} `json:"refused"`
	}

	readJSON(t, engineDepth, &fx)

	if fx.FullDepth != fullDepth || len(fx.Refused) != 3 {
		t.Fatalf("depth.json: full depth %d, %d refusals; want %d and 3", fx.FullDepth, len(fx.Refused), fullDepth)
	}

	for depth, want := range fx.Refused {
		h := newHarness()

		code := h.run(t, "--depth", depth, "--prompt", "set a timer for 10 minutes for the pasta")
		if code != want.Exit {
			t.Errorf("--depth %s: exit %d, the engine's %d", depth, code, want.Exit)
		}

		if got := strings.TrimSpace(h.stderr.String()); depth != "21" && got != want.Stderr {
			t.Errorf("--depth %s: stderr %q, the engine's %q", depth, got, want.Stderr)
		}
	}

	h := newHarness()
	if code := h.run(t, "--depth", strconv.Itoa(fullDepth), "--prompt", "hi"); code != 0 || h.stdout.String() != timerJSON {
		t.Errorf("--depth %d: exit %d, stdout %q, stderr %q", fullDepth, code, h.stdout, h.stderr)
	}
}

// TestPrompt: --prompt opens the model at --model, makes one agent over
// the --tools file with the --system file as given (no date fact, no year
// check: what the engine renders and writes), answers the query at --max
// and prints the envelope as one line of JSON.
func TestPrompt(t *testing.T) {
	const tools = `[{"name":"set_timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer"}}}}]`

	h := newHarness()
	toolsFile := writeFile(t, "tools.json", tools)
	systemFile := writeFile(t, "system.txt", "date: 2026-10-11 Sun 09:30\n")

	code := h.run(t, "--model", "m/needle3.cact", "--tools", toolsFile, "--system", systemFile, "--threads", "4",
		"--max", "64", "--fail-input-overflow", "--prompt", "set a timer for 10 minutes for the pasta")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, h.stderr)
	}

	if h.stdout.String() != timerJSON || h.stderr.Len() != 0 {
		t.Errorf("stdout %q, stderr %q; want %q and nothing", h.stdout, h.stderr, timerJSON)
	}

	m := h.model

	want := needle.AgentOptions{
		System: "date: 2026-10-11 Sun 09:30\n", NoAutoDate: true, NoYearCheck: true, FailInputOverflow: true,
	}
	if m.path != "m/needle3.cact" || string(m.tools) != tools || !reflect.DeepEqual(m.opts, want) || !m.closed {
		t.Errorf("model %q tools %s options %+v closed %v", m.path, m.tools, m.opts, m.closed)
	}

	inputs, _ := m.agent.asked()
	if !reflect.DeepEqual(inputs, []string{"set a timer for 10 minutes for the pasta"}) ||
		!reflect.DeepEqual(m.agent.maxNew, []int{64}) || !reflect.DeepEqual(h.threads, []int{4}) {
		t.Errorf("asked %q at --max %v with threads %v", inputs, m.agent.maxNew, h.threads)
	}
}

// TestPromptDefaults: without flags but --prompt the CLI opens the
// default model path, sends no tools and no system facts, answers at
// --max 512 without refusing an oversized turn, and leaves the threads.
func TestPromptDefaults(t *testing.T) {
	h := newHarness()

	if code := h.run(t, "--prompt", "hi"); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, h.stderr)
	}

	m := h.model

	want := needle.AgentOptions{NoAutoDate: true, NoYearCheck: true}
	if m.path != defaultModel || m.tools != nil || !reflect.DeepEqual(m.opts, want) {
		t.Errorf("model %q tools %q options %+v", m.path, m.tools, m.opts)
	}

	if !reflect.DeepEqual(m.agent.maxNew, []int{needle.DefaultMaxNewTokens}) || h.threads != nil {
		t.Errorf("--max %v, threads %v", m.agent.maxNew, h.threads)
	}
}

// TestSystemNotAFile: --system with a text that names no readable file
// renders no system facts, as the engine ignores it (date.json's
// literal_text answers as no_system; docs/FINDINGS.md F-047), and the CLI
// says so on stderr.
func TestSystemNotAFile(t *testing.T) {
	h := newHarness()

	code := h.run(t, "--system", "date: 2026-10-11 Sun 09:30", "--prompt", "hi")
	if code != 0 || h.model.opts.System != "" || h.stdout.String() != timerJSON {
		t.Fatalf("exit %d, system %q, stdout %q", code, h.model.opts.System, h.stdout)
	}

	if want := `needle: ignoring --system "date: 2026-10-11 Sun 09:30": not a readable file` + "\n"; h.stderr.String() != want {
		t.Errorf("stderr %q, want %q", h.stderr, want)
	}
}

// TestPromptOverflow: a turn that --fail-input-overflow refuses prints the
// engine's refusal envelope and exits 0, as --serve answers it; the
// envelope equals a recorded one (turns.json, first_plain).
func TestPromptOverflow(t *testing.T) {
	var fx struct {
		Scenarios []struct {
			ID    string `json:"id"`
			Steps []struct {
				Answer json.RawMessage `json:"answer"`
			} `json:"steps"`
		} `json:"scenarios"`
	}

	readJSON(t, engineTurns, &fx)

	if fx.Scenarios[0].ID != "first_plain" {
		t.Fatalf("turns.json starts with %s, want first_plain", fx.Scenarios[0].ID)
	}

	h := newHarness()
	h.model.agent.reply = func(string) (*needle.Response, error) {
		return nil, &needle.OverflowError{Tokens: 1407, Available: 878}
	}

	if code := h.run(t, "--max", "7000", "--fail-input-overflow", "--prompt", "long"); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, h.stderr)
	}

	if got, want := parsed(t, h.stdout.Bytes()), parsed(t, fx.Scenarios[0].Steps[0].Answer); !reflect.DeepEqual(got, want) {
		t.Errorf("refusal %s\nthe engine's %s", h.stdout, fx.Scenarios[0].Steps[0].Answer)
	}
}

// TestPromptErrors: a model that fails to open, to make the agent or to
// answer exits 1 with its error on stderr, the package's "needle: " once.
func TestPromptErrors(t *testing.T) {
	h := newHarness()
	h.model.agent.reply = func(string) (*needle.Response, error) { return nil, errBoom }

	if code := h.run(t, "--prompt", "hi"); code != 1 || h.stderr.String() != "needle: boom\n" || h.stdout.Len() != 0 {
		t.Errorf("answer: exit %d, stdout %q, stderr %q", code, h.stdout, h.stderr)
	}

	h = newHarness()
	h.model.err = errBoom

	if code := h.run(t, "--prompt", "hi"); code != 1 || h.stderr.String() != "needle: boom\n" {
		t.Errorf("agent: exit %d, stderr %q", code, h.stderr)
	}

	h = newHarness()
	c := h.cli()
	c.open = func(string) (model, error) { return nil, errors.New("needle: open x: no such file") }

	if code := c.run(t.Context(), []string{"--model", "x/needle3.cact", "--prompt", "hi"}); code != 1 ||
		h.stderr.String() != "needle: cannot load x/needle3.cact: open x: no such file\n" {
		t.Errorf("open: exit %d, stderr %q", code, h.stderr)
	}
}

// TestTriggerWarningsMatchEngine: the CLI warns of each tool trigger it
// cannot compile on stderr as the engine does, byte for byte, and of none
// for the valid ones (triggers.json).
func TestTriggerWarningsMatchEngine(t *testing.T) {
	var fx struct {
		Probes []struct {
			ID     string `json:"id"`
			Tools  string `json:"tools"`
			Query  string `json:"query"`
			Stderr string `json:"stderr"`
		} `json:"probes"`
	}

	readJSON(t, engineTriggers, &fx)

	warned := 0

	for _, p := range fx.Probes {
		h := newHarness()

		if code := h.run(t, "--tools", writeFile(t, "tools.json", p.Tools), "--prompt", p.Query); code != 0 {
			t.Fatalf("%s: exit %d, stderr %q", p.ID, code, h.stderr)
		}

		if got := strings.TrimSpace(h.stderr.String()); got != p.Stderr {
			t.Errorf("%s: stderr %q, the engine's %q", p.ID, got, p.Stderr)
		}

		if p.Stderr != "" {
			warned++
		}
	}

	if warned == 0 {
		t.Error("triggers.json has no warning to compare")
	}
}

// TestToolIndexIgnored: --tool-index is accepted and writes nothing, as
// the engine does on needle3.cact, which carries no embedding head
// (retrieval.json; docs/FINDINGS.md F-049).
func TestToolIndexIgnored(t *testing.T) {
	var fx struct {
		CLI struct {
			Exit         int  `json:"exit"`
			IndexWritten bool `json:"index_written"`
		} `json:"cli"`
	}

	readJSON(t, engineIndex, &fx)

	h := newHarness()
	index := filepath.Join(t.TempDir(), "tools.index")

	code := h.run(t, "--tool-index", index, "--prompt", "hi")

	_, err := os.Stat(index)
	if written := err == nil; code != fx.CLI.Exit || written != fx.CLI.IndexWritten || h.stdout.String() != timerJSON {
		t.Errorf("exit %d, index written %v, stdout %q; the engine: exit %d, written %v", code, written, h.stdout,
			fx.CLI.Exit, fx.CLI.IndexWritten)
	}
}

// TestModelDir: --model names needle3.cact, whose directory Load opens,
// or the directory itself; another file is refused.
func TestModelDir(t *testing.T) {
	dir := t.TempDir()

	for _, c := range []struct {
		path, want string
		ok         bool
	}{
		{"models/needle3/needle3.cact", "models/needle3", true},
		{"needle3.cact", ".", true},
		{dir, dir, true},
		{"models/whistle/whistle.cact", "", false},
		{"models/needle3/needle3.safetensors", "", false},
	} {
		got, err := modelDir(c.path)
		if (err == nil) != c.ok || got != c.want {
			t.Errorf("modelDir(%q) = %q, %v; want %q, ok %v", c.path, got, err, c.want, c.ok)
		}
	}
}

// TestEncode: the envelope is one line of JSON with '<', '>' and '&'
// written as themselves, not as the \u escapes encoding/json writes, and a
// literal backslash before "u003c" left alone.
func TestEncode(t *testing.T) {
	reasoning := `'a' -> b & <c> \u003c`

	r := timerResponse()
	r.Reasoning = &reasoning
	r.FunctionCalls[0].Arguments = json.RawMessage(`{"label":"<&>"}`)

	js, err := encode(r)
	if err != nil {
		t.Fatal(err)
	}

	want := `"reasoning":"'a' -> b & <c> \\u003c"`
	if !strings.Contains(string(js), want) || !strings.Contains(string(js), `{"label":"<&>"}`) ||
		strings.Count(string(js), "\n") != 1 || !strings.HasSuffix(string(js), "\n") {
		t.Errorf("encode = %s, want %s and the label as written on one line", js, want)
	}

	direct, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(parsed(t, js), parsed(t, direct)) {
		t.Errorf("encode = %s, not the value of %s", js, direct)
	}
}
