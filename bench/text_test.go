package bench

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTextSamples(t *testing.T) {
	var s TextSamples

	// Two passes over two cases; the second case has no decode rate (one
	// generated token) in either pass.
	s.Add(0, "a", TextAnswer{PrefillTokens: 300, Generated: 20, TTFTMS: 10, PrefillTPS: 30000, DecodeTPS: 100, WallMS: 50})
	s.Add(0, "b", TextAnswer{PrefillTokens: 100, Generated: 1, TTFTMS: 4, PrefillTPS: 25000, WallMS: 8})
	s.Add(1, "a", TextAnswer{PrefillTokens: 300, Generated: 20, TTFTMS: 12, PrefillTPS: 25000, DecodeTPS: 300, WallMS: 70})
	s.Add(1, "b", TextAnswer{PrefillTokens: 100, Generated: 1, TTFTMS: 6, PrefillTPS: 16667, WallMS: 10})

	got := s.Set()
	if got.Cases != 2 || got.Passes != 2 {
		t.Fatalf("cases %d, passes %d", got.Cases, got.Passes)
	}

	if got.TTFT.N != 4 || got.TTFT.Median != 8 || got.DecodeTPS.N != 2 || got.DecodeTPS.Median != 200 {
		t.Errorf("TTFT %+v, decode %+v", got.TTFT, got.DecodeTPS)
	}

	if got.PrefillTokens.Median != 200 || got.Generated.Max != 20 || got.Wall.Median != 30 || got.PrefillTPS.N != 4 {
		t.Errorf("prefill tokens %+v, generated %+v, wall %+v", got.PrefillTokens, got.Generated, got.Wall)
	}

	cases := s.Cases()
	want := []TextCase{
		{ID: "a", PrefillTokens: 300, Generated: 20, TTFTMS: 11, DecodeTPS: 200, WallMS: 60},
		{ID: "b", PrefillTokens: 100, Generated: 1, TTFTMS: 5, WallMS: 9},
	}

	if len(cases) != len(want) {
		t.Fatalf("%d cases", len(cases))
	}

	for i := range want {
		if cases[i] != want[i] {
			t.Errorf("case %d: %+v, want %+v", i, cases[i], want[i])
		}
	}
}

func TestFunctionCallsEqual(t *testing.T) {
	for _, c := range []struct {
		a, b string
		want bool
	}{
		{
			`[{"name":"set_timer","arguments":{"minutes":10,"label":"Pasta"}}]`,
			`[ {"name": "set_timer", "arguments": {"label": "Pasta", "minutes": 10.0}} ]`, true,
		},
		{`[{"name":"set_timer","arguments":{"minutes":10}}]`, `[{"name":"set_timer","arguments":{"minutes":11}}]`, false},
		{
			`[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]`,
			`[{"name":"b","arguments":{}},{"name":"a","arguments":{}}]`, false,
		},
		{`[]`, ``, true},
		{`null`, `[]`, true},
		{`[]`, `[{"name":"a","arguments":{}}]`, false},
		{`[`, `[`, false},
	} {
		if got := FunctionCallsEqual(json.RawMessage(c.a), json.RawMessage(c.b)); got != c.want {
			t.Errorf("FunctionCallsEqual(%s, %s) = %v", c.a, c.b, got)
		}
	}
}

var sampleTextInputs = Inputs{Model: "c9d915eca282ed42", Clips: "0123456789abcdef"}

func sampleTextSet(ttft, decode, wall float64) TextSet {
	s := func(v float64) Summary { return Summary{N: 120, Median: v, P90: v * 1.5, Min: v / 2, Max: v * 2} }

	return TextSet{
		Cases: 24, Passes: 5, PrefillTokens: s(330), Generated: s(30), TTFT: s(ttft), PrefillTPS: s(330 / ttft * 1000),
		DecodeTPS: s(decode), Wall: s(wall),
	}
}

// sampleTextCases are two cases with their prefill token counts, the same
// on both sides.
func sampleTextCases() []TextCase {
	return []TextCase{
		{ID: "timer_pasta", PrefillTokens: 330, Generated: 30, TTFTMS: 100, DecodeTPS: 1400, WallMS: 50},
		{ID: "catalogue_email", PrefillTokens: 1049, Generated: 60, TTFTMS: 280, DecodeTPS: 1300, WallMS: 300},
	}
}

func sampleText() (*TextGoResults, *TextEngineResults) {
	machine := Machine{CPU: "CPU X", Cores: 8, OS: "os", Arch: "arm64"}
	g := &TextGoResults{
		Run:         Run{Date: "2026-10-14T10:00:00+02:00", Commit: "abc", Machine: machine, Inputs: sampleTextInputs},
		Safetensors: "5a1e7e750a0c3d2b", Tokenizer: "70e1c3a2b9d84f06", Responses: "4e5f6a7b8c9d0e1f", GOMAXPROCS: 8,
		WarmUp: 1,
		Modes: []TextGoMode{
			{
				Mode: TextModeSafetensors, Answers: sampleTextSet(200, 100, 1000), FunctionCallsEqual: 20, MaxRSSLoadMiB: 700,
				Cases: sampleTextCases(),
			},
			{
				Mode: TextModePacked, Answers: sampleTextSet(400, 110, 2000), FunctionCallsEqual: 24, HeapLoadMiB: 41.4,
				Cases: sampleTextCases(),
			},
			{
				Mode: TextModeInt8, Answers: sampleTextSet(420, 130, 2100), FunctionCallsEqual: 23, MaxRSSPeakMiB: 90,
				Cases: sampleTextCases(),
			},
			{
				Mode: TextModeInt8LUT, Answers: sampleTextSet(140, 300, 700), FunctionCallsEqual: 23,
				Cases: sampleTextCases(),
			},
		},
	}
	e := &TextEngineResults{
		Run: Run{Machine: machine, Inputs: sampleTextInputs}, Version: "3.2.0", Revision: "2ae11323dc000f5e", Threads: "4",
		Responses: "4e5f6a7b8c9d0e1f",
		WarmUp:    1,
		CLI: TextEngineCLI{
			Answers: sampleTextSet(100, 1500, 120), PeakRAM: Summary{N: 120, Median: 94.3, Max: 97.1},
			MaxRSS: Summary{N: 120, Median: 94.0, Max: 96.8}, FunctionCallsEqual: 24,
		},
		Library: TextEngineLibrary{
			Load: Summary{N: 5, Median: 2.5, P90: 3}, Answers: sampleTextSet(100, 1400, 50), FunctionCallsEqual: 24,
			Cached: sampleTextSet(10, 1450, 40), CachedFunctionCallsEqual: 22, Cases: sampleTextCases(),
		},
	}

	return g, e
}

func TestRenderText(t *testing.T) {
	g, e := sampleText()

	var b bytes.Buffer

	err := RenderText(&b, g, e, DefaultTextPaths)
	if err != nil {
		t.Fatal(err)
	}

	out := b.String()
	for _, want := range []string{
		"| TTFT, median (ms)                         |                  200.0 |",
		"| decode, median (tok/s)                    |                    100 |",
		"| `function_calls` equal to the recording   |                  20/24 |           24/24 |",
		"|          24/24 |                         22/24 |                24/24 |",
		"| Go packed exact               | 4.00× |   0.25× |  0.08× | 40.00× |",
		"| peak RSS after the 24 answers, load included |",
		"**Packed weights:** 38.252 MiB (40,109,936 bytes)",
		"on exact kernels (F-059): under the Accept's 40 MiB",
		"| needle3.safetensors SHA-256 | 5a1e7e750a",
		"| model load (ms) |",
		"3.2.0 (2ae11323dc)",
		"just bench-text          # the Go side",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q:\n%s", want, out)
		}
	}

	if strings.Contains(out, "not comparable") {
		t.Errorf("the same machine and inputs are reported as not comparable:\n%s", out)
	}
}

// TestRenderTextNotComparable leaves out the ratios when the sides ran on
// other machines or inputs.
func TestRenderTextNotComparable(t *testing.T) {
	for _, change := range []func(*TextEngineResults){
		func(e *TextEngineResults) { e.Run.Machine.CPU = "CPU Y" },
		func(e *TextEngineResults) { e.Run.Inputs.Clips = "fedcba9876543210" },
		func(e *TextEngineResults) { e.Responses = "f1e0d9c8b7a6f5e4" },
		func(e *TextEngineResults) { e.Responses = "" },
		func(e *TextEngineResults) { e.Library.Cases[1].PrefillTokens = 1050 }, // another tokenizer
		func(e *TextEngineResults) { e.Library.Cases = e.Library.Cases[:1] },
		func(e *TextEngineResults) { e.Library.Cases = nil },
	} {
		g, e := sampleText()
		change(e)

		var b bytes.Buffer

		err := RenderText(&b, g, e, DefaultTextPaths)
		if err != nil {
			t.Fatal(err)
		}

		out := b.String()
		if !strings.Contains(out, notComparableText) || strings.Contains(out, "Go / engine library") {
			t.Errorf("report across machines or inputs:\n%s", out)
		}
	}
}

func TestRenderTextWithoutResults(t *testing.T) {
	var b bytes.Buffer

	err := RenderText(&b, nil, nil, DefaultTextPaths)
	if err != nil {
		t.Fatal(err)
	}

	out := b.String()
	if !strings.Contains(out, "| TTFT, median (ms)") || !strings.Contains(out, "## Speed: the first turns") ||
		!strings.Contains(out, "**Packed weights:** 38.252 MiB") {
		t.Errorf("report without results:\n%s", out)
	}
}

// TestRenderTextPaths names the results the report comes from and the
// flags that reproduce it.
func TestRenderTextPaths(t *testing.T) {
	g, e := sampleText()
	g.GOMAXPROCS = 4

	var b bytes.Buffer

	p := Paths{Go: "a/go.json", Engine: "a/engine.json", Report: "a/R.md"}

	err := RenderText(&b, g, e, p)
	if err != nil {
		t.Fatal(err)
	}

	out := b.String()
	for _, want := range []string{
		"generated from `a/go.json` (`just bench-text`) and `a/engine.json`",
		"just bench-text -procs 4 -go a/go.json -engine a/engine.json -report a/R.md",
		"just oracle-text bench -out a/engine.json -go a/go.json -report a/R.md",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q:\n%s", want, out)
		}
	}
}

func TestThousands(t *testing.T) {
	for n, want := range map[int]string{0: "0", 999: "999", 1000: "1,000", 40109936: "40,109,936", 123456: "123,456"} {
		if got := thousands(n); got != want {
			t.Errorf("thousands(%d) = %q, want %q", n, got, want)
		}
	}
}

// TestWriteTextReport renders from the paths it is given and leaves out a
// side whose file is missing; the results survive the JSON round trip.
func TestWriteTextReport(t *testing.T) {
	g, _ := sampleText()
	dir := t.TempDir()
	goPath, reportPath := filepath.Join(dir, "text-go.json"), filepath.Join(dir, "new", "R.md")

	err := WriteJSON(goPath, g)
	if err != nil {
		t.Fatal(err)
	}

	var back TextGoResults

	ok, err := ReadJSON(goPath, &back)
	if err != nil || !ok || back.Modes[1].HeapLoadMiB != 41.4 || back.Modes[0].Answers.TTFT.Median != 200 {
		t.Fatalf("round trip: %v, %v, %+v", ok, err, back.Modes)
	}

	err = WriteTextReport(goPath, filepath.Join(dir, "missing.json"), reportPath)
	if err != nil {
		t.Fatal(err)
	}

	out, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(string(out), "| TTFT, median (ms)                         |                  200.0 |") ||
		strings.Contains(string(out), "Go / engine library") {
		t.Errorf("report from the Go side alone:\n%s", out)
	}
}

// TestTextReportCommitted re-renders the committed bench/RESULTS-text.md
// from the committed JSON files: the report is generated, never edited.
func TestTextReportCommitted(t *testing.T) {
	report, err := os.ReadFile("RESULTS-text.md")
	if err != nil {
		t.Skipf("no committed report: %v", err)
	}

	g, e, err := readSides[TextGoResults, TextEngineResults]("results/text-go.json", "results/text-engine.json")
	if err != nil {
		t.Fatal(err)
	}

	var b bytes.Buffer

	err = RenderText(&b, g, e, DefaultTextPaths)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(b.Bytes(), report) {
		t.Errorf("bench/RESULTS-text.md differs from its rendering of the committed results; run " +
			"`go run ./bench/cmd/needle-bench report`")
	}
}
