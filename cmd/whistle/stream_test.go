package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
)

const streamDir = "../../testdata/oracle/stream"

// runCmdIO runs the command line with stdin and returns stdout and stderr.
func runCmdIO(t *testing.T, stdin io.Reader, args ...string) (string, string, error) {
	t.Helper()

	root := NewRootCmd()

	var out, errOut bytes.Buffer

	root.SetIn(stdin)
	root.SetOut(&out)
	root.SetErr(&errOut)
	root.SetArgs(args)

	err := root.ExecuteContext(context.Background())

	return out.String(), errOut.String(), err
}

func TestRootHasStream(t *testing.T) {
	cmd, _, err := NewRootCmd().Find([]string{"stream"})
	if err != nil || cmd.Name() != "stream" {
		t.Fatalf("root has no stream subcommand: %v", err)
	}

	for flag, def := range map[string]string{
		"model": defaultModelPath, "language": "", "keywords": "", "chunk": "1", "realtime": "false",
		"json": "false", "commits-only": "false", "format": "s16le",
	} {
		f := cmd.Flags().Lookup(flag)
		if f == nil {
			t.Errorf("no --%s flag", flag)

			continue
		}

		if f.DefValue != def {
			t.Errorf("--%s default = %q, want %q", flag, f.DefValue, def)
		}
	}
}

func TestChunkSamples(t *testing.T) {
	for _, c := range []struct {
		seconds float64
		want    int
	}{{1, 16000}, {0.5, 8000}, {30, whistle.MaxSamples}, {0.0001, 2}} {
		got, err := chunkSamples(c.seconds)
		if err != nil || got != c.want {
			t.Errorf("chunkSamples(%v) = %d, %v; want %d", c.seconds, got, err, c.want)
		}
	}

	for _, s := range []float64{0, -1, 30.001, 31, math.NaN(), math.Inf(1), 1e-6} {
		_, err := chunkSamples(s)
		if exitCode(err) != exitUsage || !strings.Contains(err.Error(), "between 0 and 30") {
			t.Errorf("chunkSamples(%v): err = %v, want a usage error naming the range", s, err)
		}
	}
}

func TestPCMSource(t *testing.T) {
	var s16 bytes.Buffer

	for _, v := range []int16{0, 16384, -32768, 32767, -1} {
		_ = binary.Write(&s16, binary.LittleEndian, v)
	}

	src, err := openSource("-", formatS16, 2, &s16)
	if err != nil {
		t.Fatal(err)
	}

	want := [][]float32{{0, 0.5}, {-1, 32767.0 / 32768}, {-1.0 / 32768}}
	for i, w := range want {
		got, err := src.next()
		if err != nil || !equalPCM(got, w) {
			t.Fatalf("s16le chunk %d = %v, %v; want %v", i, got, err, w)
		}
	}

	_, err = src.next()
	if !errors.Is(err, io.EOF) {
		t.Errorf("after the last chunk: err = %v, want io.EOF", err)
	}

	var f32 bytes.Buffer
	_ = binary.Write(&f32, binary.LittleEndian, []float32{0.25, -0.75, 1})

	src, err = openSource("-", formatF32, 4, &f32)
	if err != nil {
		t.Fatal(err)
	}

	got, err := src.next()
	if err != nil || !equalPCM(got, []float32{0.25, -0.75, 1}) {
		t.Errorf("f32le = %v, %v", got, err)
	}

	src, _ = openSource("-", formatS16, 4, bytes.NewReader([]byte{1, 2, 3}))

	_, err = src.next()
	if err == nil || !strings.Contains(err.Error(), "inside a sample") {
		t.Errorf("odd byte count: err = %v", err)
	}

	_, err = openSource("-", "u8", 4, nil)
	if exitCode(err) != exitUsage {
		t.Errorf("--format u8: err = %v, want a usage error", err)
	}
}

func equalPCM(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// sliceSource plays decoded samples in chunks.
type sliceSource struct {
	pcm   []float32
	chunk int
}

func (s *sliceSource) next() ([]float32, error) {
	if len(s.pcm) == 0 {
		return nil, io.EOF
	}

	n := min(s.chunk, len(s.pcm))
	c := s.pcm[:n]
	s.pcm = s.pcm[n:]

	return c, nil
}

func TestSliceSource(t *testing.T) {
	src := &sliceSource{pcm: make([]float32, 5), chunk: 2}

	var lengths []int

	for {
		c, err := src.next()
		if errors.Is(err, io.EOF) {
			break
		}

		lengths = append(lengths, len(c))
	}

	if len(lengths) != 3 || lengths[0] != 2 || lengths[1] != 2 || lengths[2] != 1 {
		t.Errorf("chunks %v, want [2 2 1]", lengths)
	}
}

// fakeStream records the chunks and answers with canned results.
type fakeStream struct {
	fed     []int
	results []whistle.StreamResult
	stopped int
}

func (f *fakeStream) Process(pcm []float32) (whistle.StreamResult, error) {
	f.fed = append(f.fed, len(pcm))
	r := f.results[min(len(f.fed), len(f.results))-1]

	return r, nil
}

func (f *fakeStream) Stop() whistle.StreamResult {
	f.stopped++

	return whistle.StreamResult{Text: "end.", Words: []whistle.Word{{Text: "end."}}, Language: "en", Received: 2.5}
}

func fakeResults() []whistle.StreamResult {
	return []whistle.StreamResult{
		{Words: []whistle.Word{}, Pending: "Hello", Language: "en", Received: 1},
		{
			Text: "Hello", Words: []whistle.Word{{Text: "Hello", End: 400 * time.Millisecond, Probability: 0.5}},
			Pending: "world", Language: "en", Received: 2, Pass: 1500 * time.Microsecond,
		},
		{Words: []whistle.Word{}, Pending: "world, and", Language: "en", Received: 2.5},
	}
}

func TestRunStreamView(t *testing.T) {
	f := &fakeStream{results: fakeResults()}

	var out bytes.Buffer

	err := runStream(context.Background(), f, &sliceSource{pcm: make([]float32, 40000), chunk: 16000},
		streamConfig{chunk: 16000}, &out)
	if err != nil {
		t.Fatal(err)
	}

	want := `   1.00 s  commit ""  pending "Hello"
   2.00 s  commit "Hello"  pending "world"
   2.50 s  commit ""  pending "world, and"
   2.50 s  stop   "end."
Hello end.
`
	if out.String() != want {
		t.Errorf("view =\n%s\nwant\n%s", out.String(), want)
	}

	if len(f.fed) != 3 || f.fed[2] != 8000 || f.stopped != 1 {
		t.Errorf("fed %v, stopped %d", f.fed, f.stopped)
	}
}

func TestRunStreamJSON(t *testing.T) {
	f := &fakeStream{results: fakeResults()}

	var out bytes.Buffer

	err := runStream(context.Background(), f, &sliceSource{pcm: make([]float32, 32000), chunk: 16000},
		streamConfig{chunk: 16000, asJSON: true, commitsOnly: true}, &out)
	if err != nil {
		t.Fatal(err)
	}

	want := `{"text":"Hello","words":[{"word":"Hello","start":0.00,"end":0.40,"probability":0.500}],` +
		`"pending":"world","language":"en","received":2.00,"pass_ms":1.5}` + "\n" +
		`{"text":"end.","words":[{"word":"end.","start":0.00,"end":0.00,"probability":0.000}],` +
		`"pending":"","language":"en","received":2.50,"pass_ms":0.0}` + "\n"
	if out.String() != want {
		t.Errorf("--json --commits-only =\n%s\nwant\n%s", out.String(), want)
	}
}

// blockingSource never yields, as a stdin nobody writes to.
type blockingSource struct{ release chan struct{} }

func (b blockingSource) next() ([]float32, error) {
	<-b.release

	return nil, io.EOF
}

// Ctrl-C (a cancelled context) stops the stream even while stdin blocks.
// cancelledCtx has ended, but its Err stays nil, so that runStream reaches
// its select with both Done and the reader's closed channel ready.
type cancelledCtx struct{ done chan struct{} }

func (cancelledCtx) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c cancelledCtx) Done() <-chan struct{}     { return c.done }
func (cancelledCtx) Err() error                  { return nil }
func (cancelledCtx) Value(any) any               { return nil }

// A closed reader channel stops the stream; it never feeds an empty chunk.
// The race is random: without the check most runs of 2000 rounds fail.
func TestRunStreamClosedReader(t *testing.T) {
	for range 2000 {
		f := &fakeStream{results: fakeResults()}
		ctx := cancelledCtx{make(chan struct{})}
		close(ctx.done)

		err := runStream(ctx, f, &sliceSource{pcm: make([]float32, 1<<20), chunk: 1600},
			streamConfig{chunk: 1600}, io.Discard)
		if err != nil {
			t.Fatal(err)
		}

		if slices.Contains(f.fed, 0) || f.stopped != 1 {
			t.Fatalf("fed %v, stopped %d", f.fed, f.stopped)
		}
	}
}

func TestRunStreamCancel(t *testing.T) {
	f := &fakeStream{results: fakeResults()}
	src := blockingSource{release: make(chan struct{})}

	t.Cleanup(func() { close(src.release) })

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error)

	var out bytes.Buffer

	go func() { done <- runStream(ctx, f, src, streamConfig{chunk: 16000}, &out) }()

	cancel()

	select {
	case err := <-done:
		if err != nil || f.stopped != 1 || !strings.HasSuffix(out.String(), "end.\n") {
			t.Errorf("err %v, stopped %d, out %q", err, f.stopped, out.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runStream did not return after the context ended")
	}
}

// --realtime feeds no faster than the audio plays.
func TestRunStreamRealtime(t *testing.T) {
	f := &fakeStream{results: fakeResults()}
	start := time.Now()

	err := runStream(context.Background(), f, &sliceSource{pcm: make([]float32, 3200), chunk: 1600},
		streamConfig{chunk: 1600, realtime: true}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}

	if d := time.Since(start); d < 190*time.Millisecond {
		t.Errorf("0.2 s of audio fed in %v", d)
	}
}

func TestStreamCmdUsage(t *testing.T) {
	for _, args := range [][]string{
		{"stream", "a.wav", "b.wav"},
		{"stream", "--chunk", "31", "-"},
		{"stream", "--format", "s16le", clipPath},
		{"stream", "--format", "u8"},
	} {
		_, _, err := runCmdIO(t, strings.NewReader(""), args...)
		if exitCode(err) != exitUsage {
			t.Errorf("%v: err = %v, want a usage error", args, err)
		}
	}
}

func TestStreamCmdMissingWAV(t *testing.T) {
	_, _, err := runCmdIO(t, nil, "stream", filepath.Join(t.TempDir(), "missing.wav"))
	if exitCode(err) != exitFailure || !strings.Contains(err.Error(), "missing.wav") {
		t.Errorf("err = %v, want a failure naming the file", err)
	}
}

// streamRun is one run of testdata/oracle/stream/manifest.json.
type streamRun struct {
	Name         string   `json:"name"`
	Clips        []string `json:"clips"`
	LeadSamples  int      `json:"lead_samples"`
	GapSamples   int      `json:"gap_samples"`
	ChunkSamples int      `json:"chunk_samples"`
	Language     string   `json:"language"`
	Samples      int      `json:"samples"`
	Text         string   `json:"text"`
}

// loadStreamRun reads the run name of the manifest and its samples, or
// skips the test.
func loadStreamRun(t *testing.T, name string) (streamRun, []float32) {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(streamDir, "manifest.json"))
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just oracle-stream`", streamDir)
	}

	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Runs []streamRun `json:"runs"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	for _, r := range manifest.Runs {
		if r.Name != name {
			continue
		}

		x := make([]float32, r.LeadSamples, r.Samples)

		for i, id := range r.Clips {
			pcm, err := frontend.ReadWAV(filepath.Join("../../testdata/audio", id+".wav"))
			if err != nil {
				t.Fatal(err)
			}

			if i > 0 {
				x = append(x, make([]float32, r.GapSamples)...)
			}

			x = append(x, pcm...)
		}

		if len(x) != r.Samples {
			t.Fatalf("%s: %d samples, manifest %d", name, len(x), r.Samples)
		}

		return r, x
	}

	t.Fatalf("no run %s in the manifest", name)

	return streamRun{}, nil
}

var passMs = regexp.MustCompile(`,"pass_ms":\d+\.\d}$`)

// libraryStream is what the stream parity test's Go side does with a run:
// a new Stream fed chunk samples per call, then Stop. It returns every
// call's JSON without pass_ms, and the committed text.
func libraryStream(t *testing.T, m *whistle.Model, r streamRun, x []float32) ([]string, string) {
	t.Helper()

	s, err := m.NewStream(whistle.StreamOptions{Language: r.Language})
	if err != nil {
		t.Fatal(err)
	}

	var lines, text []string

	add := func(res whistle.StreamResult) {
		js, err := res.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}

		lines = append(lines, passMs.ReplaceAllString(string(js), "}"))

		if res.Text != "" {
			text = append(text, res.Text)
		}
	}

	for i := 0; i < len(x); i += r.ChunkSamples {
		res, err := s.Process(x[i:min(i+r.ChunkSamples, len(x))])
		if err != nil {
			t.Fatal(err)
		}

		add(res)
	}

	add(s.Stop())

	return lines, strings.Join(text, " ")
}

// cliLines is the stream command's JSON output without pass_ms, and its
// committed text.
func cliLines(t *testing.T, out string) ([]string, string) {
	t.Helper()

	var lines, text []string

	for line := range strings.Lines(out) {
		line = strings.TrimSuffix(line, "\n")

		var res struct {
			Text string `json:"text"`
		}

		err := json.Unmarshal([]byte(line), &res)
		if err != nil {
			t.Fatalf("line %q: %v", line, err)
		}

		lines = append(lines, passMs.ReplaceAllString(line, "}"))

		if res.Text != "" {
			text = append(text, res.Text)
		}
	}

	return lines, strings.Join(text, " ")
}

// writeWAV writes samples that are int16 / 32768 as a 16-bit mono 16 kHz WAV.
func writeWAV(t *testing.T, path string, x []float32) {
	t.Helper()

	writeSilentWAV(t, path, len(x))

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	copy(raw[44:], s16le(x))

	err = os.WriteFile(path, raw, 0o600)
	if err != nil {
		t.Fatal(err)
	}
}

// s16le is x, int16 / 32768 samples, as raw PCM.
func s16le(x []float32) []byte {
	b := make([]byte, 0, 2*len(x))
	for _, v := range x {
		b = binary.LittleEndian.AppendUint16(b, uint16(int16(math.Round(float64(v)*32768)))) // #nosec G115 -- test PCM.
	}

	return b
}

// The stream command on manifest runs at 1 s chunks, from the WAV file and
// from stdin in both formats (long-clean, 44.91 s, from the WAV only): every
// call equals the library's Stream fed the same chunks
// (TestStreamMatchesEngineAll's Go side), and so does the final committed
// text, which on these runs is also the engine's.
func TestStreamCmdMatchesLibrary(t *testing.T) {
	requireModel(t)

	m, err := whistle.Load(cactPath)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = m.Close() })

	for _, name := range []string{"ls-other-3080-5040-0018.c1", "ls-clean-6930-75918-0000.c1", "long-clean"} {
		t.Run(name, func(t *testing.T) {
			if name == "long-clean" && testing.Short() {
				t.Skip("45 s of audio; skipped with -short")
			}

			r, x := loadStreamRun(t, name)
			if r.ChunkSamples != whistle.SampleRate {
				t.Fatalf("%s: chunks of %d samples, want 1 s", name, r.ChunkSamples)
			}

			want, wantText := libraryStream(t, m, r, x)
			if wantText != r.Text {
				t.Errorf("library text %q, the engine's stream %q", wantText, r.Text)
			}

			wav := filepath.Join(t.TempDir(), name+".wav")
			writeWAV(t, wav, x)

			var f32 bytes.Buffer
			_ = binary.Write(&f32, binary.LittleEndian, x)

			for _, c := range []struct {
				how   string
				stdin io.Reader
				args  []string
			}{
				{"wav", nil, []string{wav}},
				{"s16le", bytes.NewReader(s16le(x)), nil},
				{"f32le", &f32, []string{"--format", "f32le", "-"}},
			} {
				if name == "long-clean" && c.how != "wav" {
					continue // the short runs cover stdin; this one a WAV over 30 s
				}

				out, _, err := runCmdIO(t, c.stdin,
					append([]string{"stream", "--model", cactPath, "--json"}, c.args...)...)
				if err != nil {
					t.Fatalf("%s: %v", c.how, err)
				}

				got, text := cliLines(t, out)
				if text != wantText || strings.Join(got, "\n") != strings.Join(want, "\n") {
					t.Errorf("%s: text %q, want %q; %d lines, want %d:\n%s\nwant\n%s", c.how, text, wantText,
						len(got), len(want), strings.Join(got, "\n"), strings.Join(want, "\n"))
				}
			}
		})
	}
}

// The view ends with the committed text.
func TestStreamCmdView(t *testing.T) {
	requireModel(t)

	out, _, err := runCmdIO(t, nil, "stream", "--model", cactPath, clipPath)
	if err != nil {
		t.Fatal(err)
	}

	lines := strings.Split(strings.TrimSuffix(out, "\n"), "\n")
	if lines[len(lines)-1] != clipText || !strings.Contains(lines[len(lines)-2], "stop") ||
		!regexp.MustCompile(`^ +1\.00 s  commit "`).MatchString(lines[0]) {
		t.Errorf("view =\n%s", out)
	}
}
