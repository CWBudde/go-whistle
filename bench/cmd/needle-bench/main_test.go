package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/bench"
)

const (
	testCases     = "../../../testdata/text/engine/cases.json"
	testResponses = "../../../testdata/text/engine/responses.json"
	testCact      = "../../../models/needle3/needle3.cact"
	testTokenizer = "../../../models/needle3/tokenizer.model"
)

// TestLoadCases reads the committed corpus: 24 first turns, each with its
// tools, the system text, its budget and the engine's recorded calls.
func TestLoadCases(t *testing.T) {
	cases, err := loadCases(testCases, testResponses)
	if err != nil {
		t.Fatal(err)
	}

	if len(cases) != 24 {
		t.Fatalf("%d cases, want 24", len(cases))
	}

	budgets := map[int]int{}

	for _, c := range cases {
		if c.Query == "" || !strings.Contains(c.System, "date:") || len(c.Tools) == 0 || len(c.EngineCalls) == 0 {
			t.Errorf("%s: query %q, system %q, %d tool bytes, calls %s", c.ID, c.Query, c.System, len(c.Tools),
				c.EngineCalls)
		}

		budgets[c.Max]++
	}

	// timer_budget is the corpus's one case with a --max of its own.
	if budgets[defaultMax] != 23 || budgets[8] != 1 {
		t.Errorf("budgets %v", budgets)
	}
}

func TestLoadCasesErrors(t *testing.T) {
	dir := t.TempDir()

	write := func(name, body string) string {
		p := filepath.Join(dir, name)

		err := os.WriteFile(p, []byte(body), 0o600)
		if err != nil {
			t.Fatal(err)
		}

		return p
	}

	none := write("none.json", `{"system":"date: x","toolsets":{},"cases":[]}`)
	toolset := write("toolset.json", `{"system":"date: x","toolsets":{},"cases":[{"id":"a","toolset":"t","query":"q"}]}`)
	ok := write("ok.json", `{"system":"date: x","toolsets":{"t":[]},"cases":[{"id":"a","toolset":"t","query":"q"}]}`)
	responses := write("responses.json", `{"b":[{"function_calls":[]}]}`)

	for name, c := range map[string][2]string{
		"no cases":    {none, responses},
		"no toolset":  {toolset, responses},
		"no answer":   {ok, responses},
		"no corpus":   {filepath.Join(dir, "missing.json"), responses},
		"bad answers": {ok, write("bad.json", `[`)},
	} {
		_, err := loadCases(c[0], c[1])
		if err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestParseMemReport(t *testing.T) {
	r, err := parseMemReport(strings.NewReader("loading\n"+memLine+
		`{"mode":"int8","max_rss_load":1048576,"heap_load":2097152,"max_rss_peak":3145728,"answers":24}`+"\n"), "int8")
	if err != nil {
		t.Fatal(err)
	}

	if r.Mode != "int8" || mebi(r.MaxRSSLoad) != 1 || mebi(r.HeapLoad) != 2 || mebi(r.MaxRSSPeak) != 3 || r.Answers != 24 {
		t.Errorf("%+v", r)
	}

	for _, out := range []string{"nothing\n", memLine + "{\n"} {
		_, err := parseMemReport(strings.NewReader(out), "int8")
		if err == nil {
			t.Errorf("%q: no error", out)
		}
	}
}

func TestArgmax(t *testing.T) {
	for _, c := range []struct {
		row  []float32
		want int
	}{{[]float32{1}, 0}, {[]float32{1, 3, 2}, 1}, {[]float32{-1, -1, -0.5}, 2}, {[]float32{2, 2}, 0}} {
		if got := argmax(c.row); got != c.want {
			t.Errorf("argmax(%v) = %d, want %d", c.row, got, c.want)
		}
	}
}

func TestDispatchErrors(t *testing.T) {
	var log bytes.Buffer

	for _, args := range [][]string{nil, {"nope"}} {
		err := dispatch(t.Context(), args, &log, &log)
		if err == nil || !strings.Contains(err.Error(), "usage") {
			t.Errorf("%q: %v", args, err)
		}
	}

	for _, flags := range [][]string{
		{"-warmup=0", "-passes=0"}, {"-passes=-1"}, {"-warmup=-1"}, {"-loads=0"}, {"-procs=-2"}, {"-modes=packed,fp16"},
	} {
		err := dispatch(t.Context(), append([]string{"run"}, flags...), &log, &log)
		if err == nil || !strings.Contains(err.Error(), "usage") {
			t.Errorf("run %q: %v", flags, err)
		}
	}

	err := dispatch(t.Context(), []string{"run", "-cact", filepath.Join(t.TempDir(), "none.cact")}, &log, &log)
	if err == nil || !strings.Contains(err.Error(), "just fetch-needle3") {
		t.Errorf("run without a model: %v", err)
	}

	err = dispatch(t.Context(), []string{"child", "-mode", "fp16", "-tokenizer", testTokenizer}, &log, &log)
	if err == nil {
		t.Errorf("child of an unknown mode: no error")
	}
}

// TestReport renders a report without any results.
func TestReport(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "RESULTS-text.md")

	var log bytes.Buffer

	err := dispatch(t.Context(), []string{
		"report", "-go", filepath.Join(dir, "go.json"), "-engine", filepath.Join(dir, "e.json"), "-report", out,
	}, &log, &log)
	if err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.HasPrefix(string(raw), "# Benchmarks: the Needle 3 text model") {
		t.Errorf("report:\n%s", raw)
	}
}

// TestAnswer answers one case per mode on the real model, timing every
// part: the model files are needed, so it skips without them.
func TestAnswer(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the text model")
	}

	for _, p := range []string{testCact, testTokenizer} {
		if !exists(p) {
			t.Skipf("%s missing; run `just fetch-needle3`", p)
		}
	}

	cases, err := loadCases(testCases, testResponses)
	if err != nil {
		t.Fatal(err)
	}

	cfg := config{cact: testCact, tokenizer: testTokenizer}

	for _, mode := range []string{bench.TextModeInt8LUT, bench.TextModePacked} {
		l, err := load(cfg, mode)
		if err != nil {
			t.Fatal(err)
		}

		a, err := l.answer(cases[0]) // timer_pasta
		if err != nil {
			t.Fatal(err)
		}

		if a.PrefillTokens < 300 || a.Generated < 10 || a.TTFTMS <= 0 || a.PrefillTPS <= 0 || a.DecodeTPS <= 0 ||
			a.WallMS < a.TTFTMS || !bench.FunctionCallsEqual(a.calls, cases[0].EngineCalls) {
			t.Errorf("%s: %+v, calls %s, engine %s", mode, a.TextAnswer, a.calls, cases[0].EngineCalls)
		}
	}
}
