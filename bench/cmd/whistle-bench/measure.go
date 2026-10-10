package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"time"

	"github.com/cwbudde/go-whistle/bench"
	"github.com/cwbudde/go-whistle/whistle"
)

// modes are the two weight modes, in the order of the first transcription.
var modes = [2]string{"packed", "float"}

// options is whistle.Load's options for a mode: packed is the default on
// a .cact, float WithFloatWeights() (docs/FINDINGS.md F-037).
func options(mode string) []whistle.Option {
	if mode == "float" {
		return []whistle.Option{whistle.WithFloatWeights()}
	}

	return nil
}

// config is one Go run.
type config struct {
	model, audio, oracle string
	lsDir, lsAudio       string // the LibriSpeech subset's fixtures and audio
	warmUp, passes, long int    // untimed passes, timed passes over the clips, timed 30 s runs
	loads                int    // timed Loads per mode
	wer                  bool
	log                  io.Writer
	// memory runs one fresh process per mode for the peak RSS.
	memory func(mode string) (memReport, error)
}

// measure is the Go side of the benchmark (docs/FINDINGS.md F-040): Load
// time, the 32 clips and the 30 s clip packed and float alternately in
// one process, the peak RSS in a fresh process per mode and the
// LibriSpeech WER.
func measure(ctx context.Context, cfg config) (*bench.GoResults, error) {
	res := &bench.GoResults{Run: bench.StartRun(ctx), GOMAXPROCS: runtime.GOMAXPROCS(0), WarmUp: cfg.warmUp}

	clips, err := loadClips(cfg.audio, cfg.oracle)
	if err != nil {
		return nil, err
	}

	res.Modes = []bench.GoMode{{Mode: modes[0]}, {Mode: modes[1]}}

	for i := range res.Modes {
		r, err := cfg.memory(res.Modes[i].Mode)
		if err != nil {
			return nil, err
		}

		res.Modes[i].MaxRSSLoadMiB, res.Modes[i].MaxRSSPeakMiB = mebi(r.MaxRSSLoad), mebi(r.MaxRSSPeak)
		_, _ = fmt.Fprintf(cfg.log, "%-6s fresh process: max RSS %.1f MiB after Load, %.1f MiB after a 30 s Transcribe\n",
			r.Mode, res.Modes[i].MaxRSSLoadMiB, res.Modes[i].MaxRSSPeakMiB)
	}

	models, err := loadModels(cfg, res)
	defer func() {
		for _, m := range models {
			if m != nil {
				_ = m.Close()
			}
		}
	}()

	if err != nil {
		return nil, err
	}

	err = timeClips(cfg, res, models, clips)
	if err != nil {
		return nil, err
	}

	err = timeLong(cfg, res, models, tile30s(clips[0].Samples))
	if err != nil {
		return nil, err
	}

	if cfg.wer {
		res.WER, err = measureWER(cfg, models)
		if err != nil {
			return nil, err
		}
	}

	res.Run.Finish(ctx)

	return res, nil
}

// loadModels times cfg.loads Loads per mode, alternating, and keeps the
// last model of each.
func loadModels(cfg config, res *bench.GoResults) ([2]*whistle.Model, error) {
	var (
		models [2]*whistle.Model
		times  [2][]float64
	)

	for i := range max(cfg.loads, 1) {
		for j := range modes {
			k := (j + i) % 2

			if models[k] != nil {
				_ = models[k].Close()
				models[k] = nil
			}

			runtime.GC()

			start := time.Now()

			m, err := whistle.Load(cfg.model, options(modes[k])...)
			if err != nil {
				return models, fmt.Errorf("%s: %w (run `whistle download`)", modes[k], err)
			}

			times[k] = append(times[k], ms(time.Since(start)))
			models[k] = m
		}
	}

	for k := range modes {
		res.Modes[k].Load = bench.Summarise(times[k])
		_, _ = fmt.Fprintf(cfg.log, "%-6s Load: median %.1f ms over %d\n", modes[k], res.Modes[k].Load.Median,
			res.Modes[k].Load.N)
	}

	return models, nil
}

// timeClips transcribes the clips cfg.warmUp + cfg.passes times with both
// models, alternating which goes first per clip and pass. The first
// warm-up pass gives the transcripts.
func timeClips(cfg config, res *bench.GoResults, models [2]*whistle.Model, clips []clip) error {
	var (
		samples [2]bench.Samples
		first   [2][]whistle.Result
	)

	for pass := range cfg.warmUp + cfg.passes {
		for i, c := range clips {
			for j := range modes {
				k := (j + pass + i) % 2

				r, wall, err := transcribe(models[k], c.Samples)
				if err != nil {
					return fmt.Errorf("%s %s: %w", modes[k], c.ID, err)
				}

				if pass == 0 {
					first[k] = append(first[k], r)
				}

				if pass >= cfg.warmUp {
					samples[k].Add(pass-cfg.warmUp, c.seconds(), ms(r.TTFT), r.DecodeTPS, wall)
				}
			}
		}
	}

	for k := range modes {
		m := &res.Modes[k]
		m.Clips = samples[k].Set(len(clips))

		for i, c := range clips {
			if r := first[k][i]; r.Text == c.Text && r.Language == c.Language {
				m.EqualEngine++
			}
		}

		_, _ = fmt.Fprintf(cfg.log, "%-6s %d clips: TTFT median %.1f ms, p90 %.1f ms; decode median %.0f tok/s; "+
			"RTF %.4f; equal to the engine %d/%d\n", m.Mode, len(clips), m.Clips.TTFT.Median, m.Clips.TTFT.P90,
			m.Clips.DecodeTPS.Median, m.Clips.RTF.Median, m.EqualEngine, len(clips))
	}

	for i := range clips {
		p, f := first[0][i], first[1][i]
		res.PackedEqualsFloat += btoi(p.Text == f.Text && p.Language == f.Language)
	}

	return nil
}

// timeLong transcribes the 30 s clip cfg.warmUp + cfg.long times per
// model, alternating which goes first.
func timeLong(cfg config, res *bench.GoResults, models [2]*whistle.Model, pcm []float32) error {
	var samples [2]bench.Samples

	seconds := float64(len(pcm)) / whistle.SampleRate

	for run := range cfg.warmUp + cfg.long {
		for j := range modes {
			k := (j + run) % 2

			r, wall, err := transcribe(models[k], pcm)
			if err != nil {
				return fmt.Errorf("%s 30 s clip: %w", modes[k], err)
			}

			if run >= cfg.warmUp {
				samples[k].Add(run-cfg.warmUp, seconds, ms(r.TTFT), r.DecodeTPS, wall)
			}
		}
	}

	for k := range modes {
		m := &res.Modes[k]
		m.Long = samples[k].Set(1)
		_, _ = fmt.Fprintf(cfg.log, "%-6s 30 s clip: TTFT median %.1f ms, p90 %.1f ms; decode median %.0f tok/s; "+
			"RTF %.4f\n", m.Mode, m.Long.TTFT.Median, m.Long.TTFT.P90, m.Long.DecodeTPS.Median, m.Long.RTF.Median)
	}

	return nil
}

// measureWER transcribes the LibriSpeech subset with both models,
// alternating which goes first, and scores both and the engine's
// transcripts under the manifest's rule.
func measureWER(cfg config, models [2]*whistle.Model) (*bench.WER, error) {
	ls, err := loadLibriSpeech(cfg.lsDir, cfg.lsAudio)
	if err != nil {
		return nil, err
	}

	w := &bench.WER{Utterances: len(ls.utterances)}

	var wall [2]float64

	for i, u := range ls.utterances {
		pcm, err := readWAV(filepath.Join(ls.audioDir, u.ID+".wav"), u.SHA256)
		if err != nil {
			return nil, fmt.Errorf("%w (run `just oracle-librispeech-audio`)", err)
		}

		var out [2]whistle.Result

		for j := range modes {
			k := (j + i) % 2

			r, t, err := transcribe(models[k], pcm)
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", modes[k], u.ID, err)
			}

			out[k], wall[k] = r, wall[k]+t/1000
		}

		pe, n := wordErrors(u.Transcript, out[0].Text)
		fe, _ := wordErrors(u.Transcript, out[1].Text)
		ee, _ := wordErrors(u.Transcript, u.engine)

		if ee != u.engineErrors {
			return nil, fmt.Errorf("%s: the engine's text has %d errors under the rule, manifest.json says %d", u.ID, ee,
				u.engineErrors)
		}

		w.Words += n
		w.PackedErrors += pe
		w.FloatErrors += fe
		w.EngineErrors += ee
		w.AudioSeconds += float64(len(pcm)) / whistle.SampleRate
		w.PackedEqualsFloat += btoi(out[0].Text == out[1].Text && out[0].Language == out[1].Language)
		w.PackedEqualsEngine += btoi(out[0].Text == u.engine)
		w.FloatEqualsEngine += btoi(out[1].Text == u.engine)
	}

	if w.Words != ls.words || w.EngineErrors != ls.engineErrors {
		return nil, fmt.Errorf("the rule gives the engine %d errors in %d words, manifest.json %d in %d",
			w.EngineErrors, w.Words, ls.engineErrors, ls.words)
	}

	w.PackedRTF, w.FloatRTF = wall[0]/w.AudioSeconds, wall[1]/w.AudioSeconds
	_, _ = fmt.Fprintf(cfg.log, "WER over %d utterances: packed %.2f %%, float %.2f %%, engine %.2f %%; packed = float "+
		"%d/%d; RTF packed %.4f, float %.4f\n", w.Utterances, w.Percent(w.PackedErrors), w.Percent(w.FloatErrors),
		w.Percent(w.EngineErrors), w.PackedEqualsFloat, w.Utterances, w.PackedRTF, w.FloatRTF)

	return w, nil
}

// transcribe runs one Transcribe with default options and returns its wall
// time in ms.
func transcribe(m *whistle.Model, pcm []float32) (whistle.Result, float64, error) {
	start := time.Now()

	r, err := m.Transcribe(pcm, whistle.TranscribeOptions{})
	if err != nil {
		return whistle.Result{}, 0, fmt.Errorf("transcribe: %w", err)
	}

	return r, ms(time.Since(start)), nil
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func mebi(bytes int64) float64 { return float64(bytes) / (1 << 20) }

func btoi(b bool) int {
	if b {
		return 1
	}

	return 0
}
