package bench

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func TestSummarise(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []float64
		want Summary
	}{
		{"empty", nil, Summary{}},
		{"one", []float64{4}, Summary{N: 1, Median: 4, P90: 4, Min: 4, Max: 4}},
		{"even", []float64{4, 1, 3, 2}, Summary{N: 4, Median: 2.5, P90: 4, Min: 1, Max: 4}},
		// ceil(0.9 · 10) = 9: the 9th smallest.
		{"ten", []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, Summary{N: 10, Median: 5.5, P90: 9, Min: 1, Max: 10}},
		// ceil(0.9 · 11) = 10.
		{"eleven", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, Summary{N: 11, Median: 6, P90: 10, Min: 1, Max: 11}},
	} {
		if got := Summarise(tc.in); got != tc.want {
			t.Errorf("%s: %+v, want %+v", tc.name, got, tc.want)
		}
	}
}

func TestSamplesSet(t *testing.T) {
	var s Samples

	// Two passes over two clips of 2 s and 3 s.
	s.Add(0, 2, 10, 100, 40)
	s.Add(0, 3, 20, 200, 60)
	s.Add(1, 2, 30, 300, 50)
	s.Add(1, 3, 40, 400, 100)

	got := s.Set(2)
	if got.Clips != 2 || got.Passes != 2 || got.AudioSeconds != 5 {
		t.Fatalf("clips %d, passes %d, audio %v", got.Clips, got.Passes, got.AudioSeconds)
	}

	if got.TTFT.Median != 25 || got.DecodeTPS.Median != 250 || got.Wall.Median != 55 {
		t.Errorf("medians %v / %v / %v", got.TTFT.Median, got.DecodeTPS.Median, got.Wall.Median)
	}

	// RTF per pass: 0.1 s / 5 s and 0.15 s / 5 s.
	if got.RTF.N != 2 || !near(got.RTF.Min, 0.02) || !near(got.RTF.Max, 0.03) {
		t.Errorf("RTF %+v", got.RTF)
	}
}

func near(a, b float64) bool { return a-b < 1e-12 && b-a < 1e-12 }

func TestTableLayout(t *testing.T) {
	tb := newTable("quantity", "a", "longer head")
	tb.row("x", "1.0", "2")
	tb.row("a longer row label", "10.25", "—")

	want := "" +
		"| quantity           |     a | longer head |\n" +
		"| ------------------ | ----: | ----------: |\n" +
		"| x                  |   1.0 |           2 |\n" +
		"| a longer row label | 10.25 |           — |\n"
	if got := tb.String(); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}

	tb.left = true

	want = "" +
		"| quantity           | a     | longer head |\n" +
		"| ------------------ | ----- | ----------- |\n" +
		"| x                  | 1.0   | 2           |\n" +
		"| a longer row label | 10.25 | —           |\n"
	if got := tb.String(); got != want {
		t.Errorf("left: got\n%s\nwant\n%s", got, want)
	}
}

func sampleSet(ttft, tps float64) Set {
	s := Summary{N: 1, Median: ttft, P90: ttft, Min: ttft, Max: ttft}
	r := Summary{N: 1, Median: tps, P90: tps, Min: tps, Max: tps}

	return Set{Clips: 32, Passes: 5, AudioSeconds: 300, TTFT: s, DecodeTPS: r, Wall: s, RTF: s}
}

func TestRender(t *testing.T) {
	g := &GoResults{
		Run: Run{Date: "2026-10-10T20:00:00+02:00", Commit: "abc", Machine: Machine{CPU: "CPU X", Cores: 8}},
		Modes: []GoMode{
			{Mode: "packed", Clips: sampleSet(60, 800), Long: sampleSet(300, 500), EqualEngine: 31},
			{Mode: "float", Clips: sampleSet(58, 500), Long: sampleSet(290, 400), EqualEngine: 31},
		},
		WER: &WER{Utterances: 200, Words: 4167, PackedErrors: 223, FloatErrors: 223, EngineErrors: 221},
	}
	e := &EngineResults{
		Run: Run{Machine: Machine{CPU: "CPU X", Cores: 8}}, Version: "3.2.0", Threads: "4",
		Library: EngineLibrary{Clips: sampleSet(6, 1500)},
	}

	var b bytes.Buffer

	err := Render(&b, g, e)
	if err != nil {
		t.Fatal(err)
	}

	out := b.String()
	for _, want := range []string{
		"| WER                                       |    5.35 % |   5.35 % |   5.30 % |",
		"Packed − float: **+0.00** percentage points; the Accept allows 0.1, so it **holds**.",
		"| Go packed, this machine                   |      60.0 |            800 |             10.00× |",
		"`whistle.Load` with `WithFloatWeights()`",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report lacks %q:\n%s", want, out)
		}
	}

	g.WER.PackedErrors = 228 // +0.12 points

	b.Reset()

	err = Render(&b, g, e)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(b.String(), "so it **fails**") {
		t.Errorf("a 0.12 point difference does not fail the Accept:\n%s", b.String())
	}
}

func TestRenderOtherMachine(t *testing.T) {
	g := &GoResults{
		Run:   Run{Machine: Machine{CPU: "CPU X", Cores: 8, OS: "os 1", Arch: "arm64"}},
		Modes: []GoMode{{Mode: "packed", Clips: sampleSet(60, 800)}, {Mode: "float", Clips: sampleSet(58, 500)}},
	}

	for name, m := range map[string]Machine{
		"other CPU":     {CPU: "CPU Y", Cores: 8, OS: "os 1", Arch: "arm64"},
		"other cores":   {CPU: "CPU X", Cores: 4, OS: "os 1", Arch: "arm64"},
		"other OS":      {CPU: "CPU X", Cores: 8, OS: "os 2", Arch: "arm64"},
		"other arch":    {CPU: "CPU X", Cores: 8, OS: "os 1", Arch: "amd64"},
		"unknown CPU":   {Cores: 8, OS: "os 1", Arch: "arm64"},
		"engine absent": {},
	} {
		t.Run(name, func(t *testing.T) {
			e := &EngineResults{Run: Run{Machine: m}, Library: EngineLibrary{Clips: sampleSet(6, 1500)}}

			var b bytes.Buffer

			err := Render(&b, g, e)
			if err != nil {
				t.Fatal(err)
			}

			out := b.String()
			if strings.Contains(out, "this machine") || strings.Contains(out, "10.00×") {
				t.Errorf("results from two machines are compared as one:\n%s", out)
			}

			if !strings.Contains(out, "come from different machines") ||
				!strings.Contains(out, "| Go packed, CPU X ") {
				t.Errorf("report does not say the machines differ:\n%s", out)
			}
		})
	}
}

func TestSameMachine(t *testing.T) {
	m := Machine{CPU: "CPU X", Cores: 8, CoreTypes: "4 P + 4 E", OS: "os 1", Arch: "arm64", GoVersion: "go1", MemoryGiB: 16}

	n := m
	n.GoVersion, n.MemoryGiB = "go2", 32 // the toolchain and the memory do not make another machine

	if !SameMachine(m, n) {
		t.Error("same hardware and OS is not the same machine")
	}

	n.CoreTypes = "8 P"
	if SameMachine(m, n) {
		t.Error("other core types count as the same machine")
	}

	if SameMachine(Machine{}, Machine{}) {
		t.Error("two unknown machines count as the same machine")
	}
}

func TestRenderWithoutResults(t *testing.T) {
	var b bytes.Buffer

	err := Render(&b, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(b.String(), "Not measured") || !strings.Contains(b.String(), "| TTFT, median (ms)") {
		t.Errorf("report without results:\n%s", b.String())
	}
}

func TestJSONRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "go.json")
	in := GoResults{GOMAXPROCS: 3, WER: &WER{Words: 7}, Modes: []GoMode{{Mode: "packed", MaxRSSPeakMiB: 1.5}}}

	err := WriteJSON(path, in)
	if err != nil {
		t.Fatal(err)
	}

	var out GoResults

	ok, err := ReadJSON(path, &out)
	if err != nil || !ok {
		t.Fatalf("read: %v, %v", ok, err)
	}

	if out.GOMAXPROCS != 3 || out.WER.Words != 7 || out.Modes[0].MaxRSSPeakMiB != 1.5 {
		t.Errorf("round trip: %+v", out)
	}

	ok, err = ReadJSON(filepath.Join(t.TempDir(), "missing.json"), &out)
	if ok || err != nil {
		t.Errorf("missing file: %v, %v", ok, err)
	}
}

func TestDescribe(t *testing.T) {
	m := Describe(t.Context())
	if m.Cores < 1 || m.GoVersion == "" || m.Arch == "" {
		t.Errorf("Describe: %+v", m)
	}
}
