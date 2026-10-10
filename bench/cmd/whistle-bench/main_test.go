package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/bench"
	"github.com/cwbudde/go-whistle/whistle"
)

const (
	testAudio  = "../../../testdata/audio"
	testOracle = "../../../testdata/oracle"
	testModel  = "../../../models/whistle/whistle.cact"
)

// TestLoadClips reads the committed corpus: 32 clips, digests checked,
// each with the engine's transcript.
func TestLoadClips(t *testing.T) {
	clips, err := loadClips(testAudio, testOracle)
	if err != nil {
		t.Fatal(err)
	}

	if len(clips) != 32 {
		t.Fatalf("%d clips, want 32", len(clips))
	}

	for _, c := range clips {
		if len(c.Samples) == 0 || c.Text == "" || c.Language == "" {
			t.Errorf("%s: %d samples, text %q, language %q", c.ID, len(c.Samples), c.Text, c.Language)
		}
	}
}

func TestTile30s(t *testing.T) {
	pcm := tile30s([]float32{1, 2, 3})
	if len(pcm) != whistle.MaxSamples {
		t.Fatalf("%d samples", len(pcm))
	}

	for i, v := range pcm {
		if v != float32(i%3+1) {
			t.Fatalf("sample %d is %v", i, v)
		}
	}
}

// TestLongClip builds the 30 s clip from the first clip alone: the other
// clips of the manifest are missing, and the result equals the corpus
// path's.
func TestLongClip(t *testing.T) {
	clips, err := loadClips(testAudio, testOracle)
	if err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(filepath.Join(testAudio, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()

	err = os.WriteFile(filepath.Join(dir, "manifest.json"), raw, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	wav, err := os.ReadFile(filepath.Join(testAudio, clips[0].ID+".wav"))
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(filepath.Join(dir, clips[0].ID+".wav"), wav, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	pcm, err := longClip(dir)
	if err != nil {
		t.Fatal(err)
	}

	want := tile30s(clips[0].Samples)
	if len(pcm) != len(want) {
		t.Fatalf("%d samples, want %d", len(pcm), len(want))
	}

	for i := range want {
		if pcm[i] != want[i] {
			t.Fatalf("sample %d is %v, want %v", i, pcm[i], want[i])
		}
	}
}

// TestCheckFixtureModel accepts the committed fixtures for the pinned
// model and rejects them for another one.
func TestCheckFixtureModel(t *testing.T) {
	const pinned = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"

	for _, dir := range []string{testOracle, testOracle + "/librispeech"} {
		err := checkFixtureModel(dir, pinned)
		if err != nil {
			t.Errorf("%s with the pinned model: %v", dir, err)
		}

		err = checkFixtureModel(dir, "0000")
		if err == nil || !strings.Contains(err.Error(), "recorded with model") {
			t.Errorf("%s with another model: %v", dir, err)
		}
	}
}

// TestLoadLibriSpeech reads the committed subset fixtures; the audio is
// not needed.
func TestLoadLibriSpeech(t *testing.T) {
	ls, err := loadLibriSpeech("../../../testdata/oracle/librispeech", "unused")
	if err != nil {
		t.Fatal(err)
	}

	if len(ls.utterances) != 200 || ls.words != 4167 || ls.engineErrors != 221 {
		t.Fatalf("%d utterances, %d words, %d engine errors", len(ls.utterances), ls.words, ls.engineErrors)
	}

	var sum int

	for _, u := range ls.utterances {
		e, _ := wordErrors(u.Transcript, u.engine)
		if e != u.engineErrors {
			t.Errorf("%s: %d errors under the rule, manifest.json says %d", u.ID, e, u.engineErrors)
		}

		sum += e
	}

	if sum != ls.engineErrors {
		t.Errorf("%d errors under the rule, manifest.json says %d", sum, ls.engineErrors)
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
		{"-warmup=0", "-passes=0"}, {"-passes=-1"}, {"-warmup=-1"}, {"-long=0"}, {"-loads=0"}, {"-procs=-2"},
	} {
		err := dispatch(t.Context(), append([]string{"run"}, flags...), &log, &log)
		if err == nil || !strings.Contains(err.Error(), "want") || !strings.Contains(err.Error(), "usage") {
			t.Errorf("run %q: %v", flags, err)
		}
	}

	err := dispatch(t.Context(), []string{"run", "-model", filepath.Join(t.TempDir(), "none.cact")}, &log, &log)
	if err == nil || !strings.Contains(err.Error(), "whistle download") {
		t.Errorf("run without a model: %v", err)
	}
}

// TestReport renders a report without any results.
func TestReport(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "RESULTS.md")

	var log bytes.Buffer

	err := dispatch(t.Context(), []string{
		"report", "-go", filepath.Join(dir, "go.json"), "-engine", filepath.Join(dir, "e.json"),
		"-report", out,
	}, &log, &log)
	if err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.HasPrefix(string(raw), "# Benchmarks") {
		t.Errorf("report:\n%s", raw)
	}
}

// BenchmarkTranscribe is the 32 fixture clips through Transcribe per
// iteration, packed and float, with the median TTFT and decode rate as
// metrics: `go test -run '^$' -bench . ./bench/cmd/whistle-bench/`. It
// skips without the model.
func BenchmarkTranscribe(b *testing.B) {
	if !exists(testModel) {
		b.Skipf("%s missing; run `whistle download`", testModel)
	}

	clips, err := loadClips(testAudio, testOracle)
	if err != nil {
		b.Fatal(err)
	}

	for _, mode := range modes {
		b.Run(mode, func(b *testing.B) {
			m, err := whistle.Load(testModel, options(mode)...)
			if err != nil {
				b.Fatal(err)
			}

			defer func() { _ = m.Close() }()

			var ttft, tps []float64

			for b.Loop() {
				for _, c := range clips {
					r, _, err := transcribe(m, c.Samples)
					if err != nil {
						b.Fatal(err)
					}

					ttft, tps = append(ttft, ms(r.TTFT)), append(tps, r.DecodeTPS)
				}
			}

			b.ReportMetric(bench.Summarise(ttft).Median, "ttft-ms")
			b.ReportMetric(bench.Summarise(tps).Median, "tok/s")
		})
	}
}
