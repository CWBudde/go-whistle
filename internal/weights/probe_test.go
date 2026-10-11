//go:build unix

package weights_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/metrics"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/whistle"
)

// The probes of docs/FINDINGS.md F-037: what the packed path costs in
// memory and time against the float path, through the public pipeline.
// They are slow and machine-dependent, so they run on request only.
const (
	memProbeEnv   = "WHISTLE_MEM_PROBE"       // 1: run TestMemProbe (`just mem-probe`)
	memChildEnv   = "WHISTLE_MEM_PROBE_CHILD" // set by TestMemProbe: packed or float
	speedProbeEnv = "WHISTLE_SPEED_PROBE"     // 1: run TestSpeedProbe
	speedWorkers  = "WHISTLE_SPEED_WORKERS"   // n: TestSpeedProbe on tensor.SetWorkers(n) (F-038)

	probeCact    = "../../models/whistle/whistle.cact"
	probeClips   = "../../testdata/audio"
	probeOracle  = "../../testdata/oracle"
	memProbeLine = "MEMPROBE "
)

// memReport is one fresh process's memory after Load and one 30 s
// Transcribe of the first corpus clip tiled to 30 s.
type memReport struct {
	Mode string
	// MaxRSS* are getrusage's ru_maxrss in bytes: the process's peak
	// resident set, anonymous and file-backed pages alike, so the mapped
	// .cact pages count once touched. Base is before Load, Load after it,
	// Peak after the Transcribe.
	MaxRSSBase, MaxRSSLoad, MaxRSSPeak int64
	// RSSNow is the current resident set after a GC at the end, from ps
	// (darwin) or /proc/self/status VmRSS (linux); RSSFile and RSSAnon
	// split it on linux only.
	RSSNow, RSSFile, RSSAnon int64
	// HeapLoad is the Go heap live after a GC right after Load, the clip
	// included (1.9 MB).
	HeapLoad int64
	// HeapLive is the Go heap live after that GC; GoMapped is the memory
	// the Go runtime holds from the OS (total − released), heap free
	// space and metadata included.
	HeapLive, GoMapped int64
	// Mapped is the resident part of the .cact mapping, from vmmap
	// (darwin) or /proc/self/smaps (linux). HeapLive + Mapped is the
	// model's memory: what the process keeps for it, without the binary,
	// the runtime's free heap and the GC's garbage.
	Mapped  int64
	Text    string
	Elapsed time.Duration
}

// TestMemProbe measures packed and float in one fresh child process each:
// the test binary runs itself with memChildEnv set.
func TestMemProbe(t *testing.T) {
	if mode := os.Getenv(memChildEnv); mode != "" {
		memChild(t, mode)

		return
	}

	if os.Getenv(memProbeEnv) == "" {
		t.Skipf("set %s=1 or run `just mem-probe`", memProbeEnv)
	}

	requireProbeInputs(t)

	const mb = 1 << 20 // MiB

	t.Logf("MiB     %6s  %6s  %6s  %6s  %9s  %7s  %6s  %6s  %7s  %9s  %6s  %6s", "base", "load", "heap", "peak",
		"peak−base", "rss now", "file", "anon", "Go heap", "Go mapped", ".cact", "model")

	for _, mode := range []string{"packed", "float"} {
		cmd := exec.Command(os.Args[0], "-test.run=^TestMemProbe$", "-test.count=1", "-test.v")

		cmd.Env = append(os.Environ(), memChildEnv+"="+mode)

		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s child: %v\n%s", mode, err, out)
		}

		r := parseMemReport(t, out)
		t.Logf("%-6s  %6.1f  %6.1f  %6.1f  %6.1f  %9.1f  %7.1f  %6.1f  %6.1f  %7.1f  %9.1f  %6.1f  %6.1f",
			r.Mode, float64(r.MaxRSSBase)/mb, float64(r.MaxRSSLoad)/mb, float64(r.HeapLoad)/mb, float64(r.MaxRSSPeak)/mb,
			float64(r.MaxRSSPeak-r.MaxRSSBase)/mb, float64(r.RSSNow)/mb, float64(r.RSSFile)/mb, float64(r.RSSAnon)/mb,
			float64(r.HeapLive)/mb, float64(r.GoMapped)/mb, float64(r.Mapped)/mb, float64(r.HeapLive+r.Mapped)/mb)
		t.Logf("%-6s  Load + Transcribe %v: %q", r.Mode, r.Elapsed.Round(time.Millisecond), r.Text)
	}
}

func parseMemReport(t *testing.T, out []byte) memReport {
	t.Helper()

	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		_, js, ok := strings.Cut(sc.Text(), memProbeLine)
		if !ok {
			continue
		}

		var r memReport

		err := json.Unmarshal([]byte(js), &r)
		if err != nil {
			t.Fatal(err)
		}

		return r
	}

	t.Fatalf("no %q line in the child's output:\n%s", memProbeLine, out)

	return memReport{}
}

// memChild is the child process: Load, one 30 s Transcribe, report.
func memChild(t *testing.T, mode string) {
	pcm := tiled30s(t)

	var opts []whistle.Option
	if mode == "float" {
		opts = append(opts, whistle.WithFloatWeights())
	}

	runtime.GC()

	r := memReport{Mode: mode, MaxRSSBase: maxRSS(t)}
	start := time.Now()

	m, err := whistle.Load(probeCact, opts...)
	if err != nil {
		t.Fatal(err)
	}

	r.MaxRSSLoad = maxRSS(t)

	runtime.GC()

	r.HeapLoad, _ = goMemory()

	res, err := m.Transcribe(pcm, whistle.TranscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	r.Elapsed, r.Text, r.MaxRSSPeak = time.Since(start), res.Text, maxRSS(t)

	runtime.GC()

	r.RSSNow, r.RSSFile, r.RSSAnon = currentRSS(t)
	r.HeapLive, r.GoMapped = goMemory()
	r.Mapped = mappedResident(t, filepath.Base(probeCact))

	runtime.KeepAlive(m)

	js, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}

	fmt.Println(memProbeLine + string(js)) //nolint:forbidigo // the parent reads this line
}

// maxRSS is ru_maxrss in bytes: darwin reports bytes, linux KiB.
func maxRSS(t *testing.T) int64 {
	t.Helper()

	var ru syscall.Rusage

	err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if err != nil {
		t.Fatal(err)
	}

	if runtime.GOOS == "darwin" {
		return ru.Maxrss
	}

	return ru.Maxrss << 10
}

// currentRSS is the resident set now, and on linux its file-backed and
// anonymous parts.
func currentRSS(t *testing.T) (rss, file, anon int64) {
	t.Helper()

	if runtime.GOOS != "linux" {
		out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
		if err != nil {
			t.Fatal(err)
		}

		kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		if err != nil {
			t.Fatal(err)
		}

		return kb << 10, 0, 0
	}

	raw, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Fatal(err)
	}

	field := func(name string) int64 {
		for line := range strings.Lines(string(raw)) {
			if v, ok := strings.CutPrefix(line, name+":"); ok {
				kb, _ := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), " kB"), 10, 64)

				return kb << 10
			}
		}

		return 0
	}

	return field("VmRSS"), field("RssFile"), field("RssAnon")
}

// mappedResident is the resident size of this process's mappings of the
// file named base.
func mappedResident(t *testing.T, base string) int64 {
	t.Helper()

	if runtime.GOOS == "linux" {
		raw, err := os.ReadFile("/proc/self/smaps")
		if err != nil {
			t.Fatal(err)
		}

		var sum int64

		in := false

		for line := range strings.Lines(string(raw)) {
			fields := strings.Fields(line)
			if len(fields) == 0 {
				continue
			}

			if !strings.HasSuffix(fields[0], ":") { // a mapping's header line
				in = strings.HasSuffix(fields[len(fields)-1], "/"+base)

				continue
			}

			if in && fields[0] == "Rss:" {
				kb, _ := strconv.ParseInt(fields[1], 10, 64)
				sum += kb << 10
			}
		}

		return sum
	}

	out, err := exec.Command("vmmap", "-wide", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Fatal(err)
	}

	var sum int64

	for line := range strings.Lines(string(out)) {
		// mapped file  START-END  [ VSIZE  RSDNT  DIRTY  SWAP] … path
		if !strings.HasPrefix(line, "mapped file") || !strings.HasSuffix(strings.TrimSpace(line), "/"+base) {
			continue
		}

		_, sizes, _ := strings.Cut(line, "[")
		fields := strings.Fields(sizes)
		sum += vmmapSize(t, fields[1])
	}

	return sum
}

// vmmapSize parses vmmap's sizes: 8272K, 16.1M, 1.2G.
func vmmapSize(t *testing.T, s string) int64 {
	t.Helper()

	unit := map[byte]float64{'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30}[s[len(s)-1]]

	v, err := strconv.ParseFloat(s[:len(s)-1], 64)
	if err != nil || unit == 0 {
		t.Fatalf("vmmap size %q", s)
	}

	return int64(v * unit)
}

// goMemory is the live heap and the memory the Go runtime holds.
func goMemory() (live, mapped int64) {
	s := []metrics.Sample{
		{Name: "/gc/heap/live:bytes"},
		{Name: "/memory/classes/total:bytes"},
		{Name: "/memory/classes/heap/released:bytes"},
	}
	metrics.Read(s)

	return int64(s[0].Value.Uint64()), int64(s[1].Value.Uint64() - s[2].Value.Uint64())
}

// probeClip is one corpus clip with the engine's transcript.
type probeClip struct {
	ID, Language, Text string
	Samples            []float32
}

func requireProbeInputs(t *testing.T) {
	t.Helper()

	for _, path := range []string{probeCact, filepath.Join(probeClips, "manifest.json")} {
		_, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			t.Skipf("%s missing; run `whistle download` and `just oracle-clips`", path)
		}
	}
}

// loadProbeClips reads the corpus, the first n clips if n > 0.
func loadProbeClips(t *testing.T, n int) []probeClip {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(probeClips, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Clips []struct {
			ID string `json:"id"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	if n > 0 {
		manifest.Clips = manifest.Clips[:n]
	}

	clips := make([]probeClip, 0, len(manifest.Clips))

	for _, c := range manifest.Clips {
		samples, err := frontend.ReadWAV(filepath.Join(probeClips, c.ID+".wav"))
		if err != nil {
			t.Fatal(err)
		}

		js, err := os.ReadFile(filepath.Join(probeOracle, c.ID+".json"))
		if err != nil {
			t.Fatal(err)
		}

		pc := probeClip{ID: c.ID, Samples: samples}

		err = json.Unmarshal(js, &pc)
		if err != nil {
			t.Fatal(err)
		}

		clips = append(clips, pc)
	}

	return clips
}

// tiled30s is the first corpus clip repeated to 30 s, F-035's 30 s clip.
func tiled30s(t *testing.T) []float32 {
	t.Helper()

	first := loadProbeClips(t, 1)[0].Samples
	pcm := make([]float32, whistle.MaxSamples)

	for i := 0; i < len(pcm); i += len(first) {
		copy(pcm[i:], first)
	}

	return pcm
}

// TestSpeedProbe loads the model packed and float in one process and
// alternates them over the corpus, three passes after a warm-up, and over
// the 30 s clip, six runs each: median TTFT and decode rate per mode, and
// how many transcripts equal the engine's and each other's.
// WHISTLE_SPEED_WORKERS=n runs it on n tensor workers instead of the
// default, GOMAXPROCS; 1 is the single-core kernels (docs/FINDINGS.md
// F-038), with the decoder's beams still on their own goroutines.
func TestSpeedProbe(t *testing.T) {
	if os.Getenv(speedProbeEnv) == "" {
		t.Skipf("set %s=1 to run the packed vs float speed probe", speedProbeEnv)
	}

	requireProbeInputs(t)

	if s := os.Getenv(speedWorkers); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatalf("%s=%q: %v", speedWorkers, s, err)
		}

		prev := tensor.Workers()

		tensor.SetWorkers(n)
		t.Cleanup(func() { tensor.SetWorkers(prev) })
	}

	t.Logf("%d tensor workers, GOMAXPROCS %d", tensor.Workers(), runtime.GOMAXPROCS(0))

	clips := loadProbeClips(t, 0)
	modes := []string{"packed", "float"}
	models := map[string]*whistle.Model{}

	for _, mode := range modes {
		var opts []whistle.Option
		if mode == "float" {
			opts = append(opts, whistle.WithFloatWeights())
		}

		start := time.Now()

		m, err := whistle.Load(probeCact, opts...)
		if err != nil {
			t.Fatal(err)
		}

		t.Cleanup(func() { _ = m.Close() })
		t.Logf("%s: Load %v", mode, time.Since(start).Round(100*time.Microsecond))
		models[mode] = m
	}

	ttft, tps := map[string][]float64{}, map[string][]float64{}
	texts := map[string][]whistle.Result{}

	for pass := range 4 {
		for _, c := range clips {
			for k := range modes {
				mode := modes[(k+pass)%2]

				res, err := models[mode].Transcribe(c.Samples, whistle.TranscribeOptions{})
				if err != nil {
					t.Fatal(err)
				}

				if pass == 0 {
					texts[mode] = append(texts[mode], res)

					continue
				}

				ttft[mode] = append(ttft[mode], float64(res.TTFT.Microseconds())/1000)
				tps[mode] = append(tps[mode], res.DecodeTPS)
			}
		}
	}

	long := map[string][]float64{}
	pcm := tiled30s(t)

	for run := range 7 {
		for k := range modes {
			mode := modes[(k+run)%2]

			res, err := models[mode].Transcribe(pcm, whistle.TranscribeOptions{})
			if err != nil {
				t.Fatal(err)
			}

			if run > 0 {
				long[mode] = append(long[mode], float64(res.TTFT.Microseconds())/1000)
			}
		}
	}

	for _, mode := range modes {
		equal := 0

		for i, c := range clips {
			if r := texts[mode][i]; r.Text == c.Text && r.Language == c.Language {
				equal++
			}
		}

		t.Logf("%-6s median TTFT %.1f ms, decode %.0f tok/s (32 clips × 3); 30 s clip TTFT %.0f ms (6 runs); "+
			"equal to the engine %d/%d", mode, median(ttft[mode]), median(tps[mode]), median(long[mode]), equal, len(clips))
	}

	for i, c := range clips {
		if p, f := texts["packed"][i], texts["float"][i]; p.Text != f.Text || p.Language != f.Language {
			t.Logf("%s: packed [%s] %q, float [%s] %q, engine [%s] %q", c.ID, p.Language, p.Text, f.Language, f.Text,
				c.Language, c.Text)
		}
	}
}

func median(v []float64) float64 {
	s := slices.Sorted(slices.Values(v))
	if len(s) == 0 {
		return 0
	}

	if len(s)%2 == 1 {
		return s[len(s)/2]
	}

	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}
