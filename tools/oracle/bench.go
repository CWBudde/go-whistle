//go:build oracle

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/cwbudde/go-whistle/bench"
)

// benchConfig is one engine benchmark run (docs/FINDINGS.md F-040).
type benchConfig struct {
	runConfig

	out                       string // the results JSON, bench/results/engine.json
	warmUp, runs, long, loads int    // untimed runs, timed passes over the clips, timed 30 s runs, timed loads
}

// benchClip is one clip with its samples, its WAV and the committed
// fixture's transcript.
type benchClip struct {
	id, wav, text, language string
	pcm                     []float32
}

func (c benchClip) seconds() float64 { return float64(len(c.pcm)) / sampleRate }

// engineTiming is the engine's JSON with the timing fields that run.go
// strips from the fixtures (timingKeys).
type engineTiming struct {
	Text      string  `json:"text"`
	Language  string  `json:"language"`
	TTFT      float64 `json:"ttft_ms"`
	DecodeTPS float64 `json:"decode_tps"`
}

// cmdBench times the engine on the 32 corpus clips and on the first one
// tiled to 30 s, the clips whistle-bench times: the needle CLI, one
// process per transcription, and libneedle3 in this process, both without
// word timestamps. It writes bench.EngineResults.
func cmdBench(ctx context.Context, cfg benchConfig, log io.Writer) error {
	res := bench.EngineResults{Run: bench.StartRun(ctx), Threads: cliThreads, WarmUp: cfg.warmUp}

	model, err := pinnedModel(cfg.modelDir)
	if err != nil {
		return err
	}

	pin, err := readPin(filepath.Join(cfg.engineDir, "PIN"))
	if err != nil {
		return fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	res.Version, res.Revision = pin["engine_version"], pin["revision"]

	tmp, err := os.MkdirTemp("", "oracle-bench-")
	if err != nil {
		return fmt.Errorf("temp dir: %w", err)
	}

	defer func() { _ = os.RemoveAll(tmp) }()

	clips, long, err := benchClips(cfg, tmp)
	if err != nil {
		return err
	}

	modelPath := filepath.Join(cfg.modelDir, model.File)

	res.CLI, err = benchCLI(ctx, cfg, filepath.Join(cfg.engineDir, "needle"), modelPath, clips, long, log)
	if err != nil {
		return err
	}

	res.Library, err = benchLibrary(cfg, modelPath, clips, long, log)
	if err != nil {
		return err
	}

	res.Run.Finish(ctx)

	err = bench.WriteJSON(cfg.out, res)
	if err != nil {
		return fmt.Errorf("write results: %w", err)
	}

	_, _ = fmt.Fprintf(log, "wrote %s\n", cfg.out)

	return nil
}

// benchClips reads the corpus clips (checked against the manifest's
// digests) with their fixture transcripts, and writes the 30 s clip, the
// first clip's samples tiled, as a float WAV into tmp so that the CLI
// reads exactly the samples whistle-bench transcribes.
func benchClips(cfg benchConfig, tmp string) ([]benchClip, benchClip, error) {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return nil, benchClip{}, fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	clips := make([]benchClip, 0, len(corpus.Clips))

	for _, c := range corpus.Clips {
		wav := filepath.Join(cfg.audioDir, c.ID+".wav")

		raw, err := readClip(wav, c.SHA256)
		if err != nil {
			return nil, benchClip{}, fmt.Errorf("%s: %w", c.ID, err)
		}

		pcm, _, err := readWAV(raw)
		if err != nil {
			return nil, benchClip{}, fmt.Errorf("%s: %w", c.ID, err)
		}

		var fixture engineTiming

		js, err := os.ReadFile(filepath.Join(cfg.oracleDir, c.ID+".json")) // #nosec G304 -- ids are checked by readCorpus.
		if err != nil {
			return nil, benchClip{}, fmt.Errorf("fixture: %w", err)
		}

		err = json.Unmarshal(js, &fixture)
		if err != nil {
			return nil, benchClip{}, fmt.Errorf("%s fixture: %w", c.ID, err)
		}

		clips = append(clips, benchClip{id: c.ID, wav: wav, text: fixture.Text, language: fixture.Language, pcm: pcm})
	}

	if len(clips) == 0 {
		return nil, benchClip{}, errors.New("the corpus has no clips")
	}

	long := benchClip{id: clips[0].id + "-tiled-30s", wav: filepath.Join(tmp, "tiled-30s.wav")}
	long.pcm = make([]float32, 30*sampleRate)

	for i := 0; i < len(long.pcm); i += len(clips[0].pcm) {
		copy(long.pcm[i:], clips[0].pcm)
	}

	var buf bytes.Buffer

	err = writeWAVFloat(&buf, long.pcm)
	if err != nil {
		return nil, benchClip{}, err
	}

	err = writeFile(long.wav, buf.Bytes())
	if err != nil {
		return nil, benchClip{}, err
	}

	return clips, long, nil
}

// benchCLI runs the needle CLI cfg.warmUp + cfg.runs times per clip, pass
// by pass, and cfg.warmUp + cfg.long times on the 30 s clip.
func benchCLI(ctx context.Context, cfg benchConfig, needle, model string, clips []benchClip, long benchClip,
	log io.Writer,
) (bench.EngineCLI, error) {
	var (
		out     bench.EngineCLI
		samples bench.Samples
	)

	for pass := range cfg.warmUp + cfg.runs {
		for _, c := range clips {
			t, wall, rss, err := cliTranscribe(ctx, needle, model, c.wav)
			if err != nil {
				return out, fmt.Errorf("%s: %w", c.id, err)
			}

			out.MaxRSSClipsMiB = max(out.MaxRSSClipsMiB, rss)

			if pass == 0 && t.Text == c.text && t.Language == c.language {
				out.EqualFixtures++
			}

			if pass >= cfg.warmUp {
				samples.Add(pass-cfg.warmUp, c.seconds(), t.TTFT, t.DecodeTPS, wall)
			}
		}
	}

	out.Clips = samples.Set(len(clips))
	samples = bench.Samples{}

	for run := range cfg.warmUp + cfg.long {
		t, wall, rss, err := cliTranscribe(ctx, needle, model, long.wav)
		if err != nil {
			return out, fmt.Errorf("30 s clip: %w", err)
		}

		out.MaxRSSLongMiB = max(out.MaxRSSLongMiB, rss)

		if run >= cfg.warmUp {
			samples.Add(run-cfg.warmUp, long.seconds(), t.TTFT, t.DecodeTPS, wall)
		}
	}

	out.Long = samples.Set(1)

	_, _ = fmt.Fprintf(log, "CLI     %d clips: ttft_ms median %.1f, p90 %.1f; decode_tps median %.0f; process %.1f ms; "+
		"equal to the fixtures %d/%d\n", len(clips), out.Clips.TTFT.Median, out.Clips.TTFT.P90,
		out.Clips.DecodeTPS.Median, out.Clips.Wall.Median, out.EqualFixtures, len(clips))
	_, _ = fmt.Fprintf(log, "CLI     30 s clip: ttft_ms median %.1f; decode_tps median %.0f; process %.1f ms; max RSS "+
		"%.1f MiB\n", out.Long.TTFT.Median, out.Long.DecodeTPS.Median, out.Long.Wall.Median, out.MaxRSSLongMiB)

	return out, nil
}

// cliTranscribe runs the needle CLI on wav with the pinned threads and no
// word timestamps, and returns its JSON, the process's wall time in ms
// and its ru_maxrss in MiB.
func cliTranscribe(ctx context.Context, needle, model, wav string) (engineTiming, float64, float64, error) {
	cmd := exec.CommandContext(ctx, needle, "--model", model, "--audio", wav, "--threads", cliThreads) // #nosec G204
	cmd.Args[0] = "needle"

	cmd.Env = append(os.Environ(), "NEEDLE_TELEMETRY=0", "DO_NOT_TRACK=1")

	var stdout, stderr bytes.Buffer

	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	start := time.Now()
	err := cmd.Run()
	wall := float64(time.Since(start).Nanoseconds()) / 1e6

	if err != nil {
		return engineTiming{}, 0, 0, fmt.Errorf("needle: %w: %s", err, stderr.String())
	}

	var t engineTiming

	err = json.Unmarshal(stdout.Bytes(), &t)
	if err != nil {
		return engineTiming{}, 0, 0, fmt.Errorf("needle output: %w: %q", err, stdout.Bytes())
	}

	var rss float64
	if ru, ok := cmd.ProcessState.SysUsage().(*syscall.Rusage); ok {
		rss = float64(bench.MaxRSSBytes(ru.Maxrss)) / (1 << 20)
	}

	return t, wall, rss, nil
}

// benchLibrary loads libneedle3 in this process, times cfg.loads more
// needle_load calls, then needle_transcribe without word timestamps
// cfg.warmUp + cfg.runs times per clip, pass by pass, and cfg.warmUp +
// cfg.long times on the 30 s clip. The library runs on its default threads.
func benchLibrary(cfg benchConfig, model string, clips []benchClip, long benchClip, log io.Writer,
) (bench.EngineLibrary, error) {
	var out bench.EngineLibrary

	cact, err := os.ReadFile(model) // #nosec G304 -- modelDir is a flag.
	if err != nil {
		return out, fmt.Errorf("read model: %w", err)
	}

	eng, err := openEngine(cfg.engineDir, cact)
	if err != nil {
		return out, err
	}

	loads := make([]float64, 0, cfg.loads)

	for range cfg.loads {
		start := time.Now()
		rc := eng.load(&cact[0], uint64(len(cact)))

		loads = append(loads, float64(time.Since(start).Nanoseconds())/1e6)

		if rc < 0 {
			return out, fmt.Errorf("needle_load: %s (%d)", eng.lastError(), rc)
		}
	}

	runtime.KeepAlive(cact)

	out.Load = bench.Summarise(loads)

	var (
		samples bench.Samples
		outside []float64
	)

	for pass := range cfg.warmUp + cfg.runs {
		for _, c := range clips {
			o, err := libTranscribe(eng, c, pass-cfg.warmUp, &samples)
			if err != nil {
				return out, err
			}

			if pass >= cfg.warmUp && !math.IsNaN(o) {
				outside = append(outside, o)
			}
		}
	}

	out.Clips = samples.Set(len(clips))
	out.OutsideClock = bench.Summarise(outside)
	samples = bench.Samples{}

	for run := range cfg.warmUp + cfg.long {
		_, err := libTranscribe(eng, long, run-cfg.warmUp, &samples)
		if err != nil {
			return out, err
		}
	}

	out.Long = samples.Set(1)

	_, _ = fmt.Fprintf(log, "library needle_load median %.1f ms; %d clips: ttft_ms median %.1f, p90 %.1f; decode_tps "+
		"median %.0f; call %.1f ms, outside the engine's clock %.2f ms; 30 s clip: ttft_ms %.1f, call %.1f ms\n",
		out.Load.Median, len(clips), out.Clips.TTFT.Median, out.Clips.TTFT.P90, out.Clips.DecodeTPS.Median,
		out.Clips.Wall.Median, out.OutsideClock.Median, out.Long.TTFT.Median, out.Long.Wall.Median)

	return out, nil
}

// libTranscribe times one needle_transcribe of c without word timestamps
// and adds it to samples when pass is not negative (a warm-up). It returns
// the call's time outside the engine's clock (EngineLibrary.OutsideClock),
// NaN without a decode rate.
func libTranscribe(eng *engine, c benchClip, pass int, samples *bench.Samples) (float64, error) {
	buf := make([]byte, transcribeOutCap)

	start := time.Now()
	rc := eng.transcr(&c.pcm[0], int32(len(c.pcm)), nil, nil, 0, &buf[0], int32(len(buf))) // #nosec G115 -- ≤ 30 s.
	wall := float64(time.Since(start).Nanoseconds()) / 1e6

	runtime.KeepAlive(c.pcm)

	r := eng.result(rc, buf)
	if r.RC < 0 {
		return 0, fmt.Errorf("%s: needle_transcribe: %s (%d)", c.id, r.Error, r.RC)
	}

	var t engineTiming

	err := json.Unmarshal(r.out, &t)
	if err != nil {
		return 0, fmt.Errorf("%s: needle_transcribe output: %w: %q", c.id, err, r.out)
	}

	if pass >= 0 {
		samples.Add(pass, c.seconds(), t.TTFT, t.DecodeTPS, wall)
	}

	if t.DecodeTPS <= 0 || rc < 1 {
		return math.NaN(), nil
	}

	return wall - t.TTFT - 1000*float64(rc-1)/t.DecodeTPS, nil
}
