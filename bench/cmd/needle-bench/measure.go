package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"time"

	"github.com/cwbudde/go-whistle/bench"
	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/text"
	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// thinkOpen is what the engine puts at the start of every answer before
// the model's first token, which the decoder feeds with the prefill
// (docs/FINDINGS.md F-052).
const thinkOpen = "<think>\n"

// toolsEnd closes the static prefix, the sink of the engine's KV window
// (docs/FINDINGS.md F-060).
const toolsEnd = "</tools>"

// config is one Go run.
type config struct {
	cact, safetensors, tokenizer string // the model files
	cases, responses             string // the corpus and the engine's recorded answers
	modes                        []string
	warmUp, passes, loads        int // untimed passes, timed passes over the cases, timed loads per mode
	log                          io.Writer
	// memory runs one fresh process per mode for the peak RSS.
	memory func(mode string) (memReport, error)
}

// loaded is one mode's model, ready to answer.
type loaded struct {
	p     *spm.Processor
	m     *text.Model
	d     *toolcall.Decoder
	think []int
}

// load builds mode's model as the bench runs it (docs/FINDINGS.md F-061):
// the tokenizer, the weights and the decoder.
//
//   - safetensors: needle3.safetensors in float32.
//   - packed: needle3.cact's CQ records packed on exact kernels (F-059).
//   - int8: packed with the engine's numerics, int8 and the KV window of
//     256 with the static prefix as its sink (F-060).
//   - int8-lut: int8 on the lut kernel (OpenOptions.LUT).
func load(cfg config, mode string) (*loaded, error) {
	p, err := spm.Load(cfg.tokenizer)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	path, opts, engine := cfg.cact, text.OpenOptions{Packed: true}, false

	switch mode {
	case bench.TextModeSafetensors:
		path, opts = cfg.safetensors, text.OpenOptions{}
	case bench.TextModePacked:
	case bench.TextModeInt8:
		engine = true
	case bench.TextModeInt8LUT:
		opts.LUT, engine = true, true
	default:
		return nil, fmt.Errorf("unknown mode %q", mode)
	}

	src, err := text.OpenWith(path, opts)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", mode, err)
	}

	tc := src.TextConfig()

	var lo text.LoadOptions

	// The engine's numerics (docs/FINDINGS.md F-060).
	if engine {
		end, ok := p.PieceID(toolsEnd)
		if !ok {
			return nil, fmt.Errorf("no piece %s", toolsEnd)
		}

		lo = text.LoadOptions{Int8: true, KVWindow: tc.KVWindow, SinkToken: end}
	}

	m, err := text.LoadWith(src, tc, lo)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", mode, err)
	}

	d, err := toolcall.NewDecoder(m, p)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", mode, err)
	}

	return &loaded{p: p, m: m, d: d, think: p.Encode(thinkOpen)}, nil
}

// answered is one first turn answered and timed.
type answered struct {
	bench.TextAnswer

	calls json.RawMessage // the envelope's function_calls
}

// answer answers case c's first turn as respond does, then times the
// same turn's prefill and decode on a fresh state, replaying the answer's
// tokens one Extend each, since the decoder does not split its time.
func (l *loaded) answer(c textCase) (answered, error) {
	a, prefill, generated, err := l.respond(c)
	if err != nil {
		return answered{}, err
	}

	out, err := l.replay(prefill, generated)
	if err != nil {
		return answered{}, fmt.Errorf("%s: %w", c.ID, err)
	}

	out.Generated, out.WallMS = a.Generated, a.WallMS
	a.TextAnswer = out

	return a, nil
}

// respond answers case c's first turn as the engine does, timed: the
// tools' normalisation, the prompt, the decode under the grammar, the
// confidence and the envelope. It returns the answer's wall time, its
// token count and function_calls, and the replay's input: the prefill up
// to the forced <think>\n and the tokens generated after it. The memory
// child runs it alone, so the bench's replay stays out of the peak RSS
// (docs/FINDINGS.md F-061).
func (l *loaded) respond(c textCase) (answered, []int, []int, error) {
	start := time.Now()

	n, err := toolcall.Normalise(c.Tools)
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	conv := toolcall.NewConversation(l.p, c.System, n.JSON, l.m.Config().MaxSeqLen)

	turn, err := conv.Turn(c.Query, c.Max, true)
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	ids := append(toolcall.StaticPrefix(l.p, c.System, n.JSON), turn...)
	s := l.m.NewState()

	a, err := l.d.Decode(s, ids, c.Max, n.GrammarFor(c.Query))
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	conf, err := l.d.Confidence(s)
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	env, err := n.Envelope(a, c.Query, c.System, nil, conf)
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	wall := ms(time.Since(start))

	js, err := json.Marshal(env)
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	var e struct {
		Calls json.RawMessage `json:"function_calls"`
	}

	err = json.Unmarshal(js, &e)
	if err != nil {
		return answered{}, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
	}

	forced := min(len(l.think), c.Max)
	out := answered{TextAnswer: bench.TextAnswer{Generated: len(a.Tokens), WallMS: wall}, calls: e.Calls}

	return out, append(ids, l.think[:forced]...), a.Tokens[forced:], nil
}

// replay times prefill, the first forward pass and the pick of the first
// generated token, then each generated token after it, one Extend and one
// pick per token, on a fresh state: the decoder's own work but for the
// grammar's mask.
func (l *loaded) replay(prefill, generated []int) (bench.TextAnswer, error) {
	s := l.m.NewState()
	v := l.m.Config().OutVocab

	start := time.Now()

	logits, err := l.m.Extend(s, prefill)
	if err != nil {
		return bench.TextAnswer{}, fmt.Errorf("prefill: %w", err)
	}

	_ = argmax(logits[len(logits)-v:])
	ttft := time.Since(start)

	out := bench.TextAnswer{PrefillTokens: len(prefill), TTFTMS: ms(ttft)}
	out.PrefillTPS = float64(len(prefill)) / ttft.Seconds()

	if len(generated) < 2 {
		return out, nil
	}

	start = time.Now()

	for _, id := range generated[:len(generated)-1] {
		row, err := l.m.Extend(s, []int{id})
		if err != nil {
			return bench.TextAnswer{}, fmt.Errorf("decode: %w", err)
		}

		_ = argmax(row)
	}

	out.DecodeTPS = float64(len(generated)-1) / time.Since(start).Seconds()

	return out, nil
}

// argmax is the index of row's largest value.
func argmax(row []float32) int {
	best := 0

	for i, x := range row {
		if x > row[best] {
			best = i
		}
	}

	return best
}

// measure is the Go side of the text bench (docs/FINDINGS.md F-061): the
// peak RSS in a fresh process per mode, then the loads and the cases with
// every mode in one process, rotating which mode goes first per case and
// pass.
func measure(ctx context.Context, cfg config, cases []textCase) (*bench.TextGoResults, error) {
	res := &bench.TextGoResults{Run: bench.StartRun(ctx), GOMAXPROCS: runtime.GOMAXPROCS(0), WarmUp: cfg.warmUp}

	err := inputs(cfg, res)
	if err != nil {
		return nil, err
	}

	res.Modes = make([]bench.TextGoMode, len(cfg.modes))

	for i, mode := range cfg.modes {
		r, err := cfg.memory(mode)
		if err != nil {
			return nil, err
		}

		m := &res.Modes[i]
		m.Mode = mode
		m.MaxRSSLoadMiB, m.MaxRSSPeakMiB, m.HeapLoadMiB = mebi(r.MaxRSSLoad), mebi(r.MaxRSSPeak), mebi(r.HeapLoad)

		_, _ = fmt.Fprintf(cfg.log, "%-11s fresh process: max RSS %.1f MiB after the load (heap %.1f MiB), %.1f MiB "+
			"after %d answers\n", mode, m.MaxRSSLoadMiB, m.HeapLoadMiB, m.MaxRSSPeakMiB, r.Answers)
	}

	models, err := loadModels(cfg, res)
	if err != nil {
		return nil, err
	}

	err = timeCases(cfg, res, models, cases)
	if err != nil {
		return nil, err
	}

	res.Run.Finish(ctx)

	return res, nil
}

// inputs fingerprints the run's model files and corpus.
func inputs(cfg config, res *bench.TextGoResults) error {
	var err error

	res.Run.Inputs.Model, err = bench.FileSHA256(cfg.cact)
	if err != nil {
		return fmt.Errorf("inputs: %w", err)
	}

	res.Run.Inputs.Clips, err = bench.FileSHA256(cfg.cases)
	if err != nil {
		return fmt.Errorf("inputs: %w", err)
	}

	res.Safetensors, err = bench.FileSHA256(cfg.safetensors)
	if err != nil {
		return fmt.Errorf("inputs: %w", err)
	}

	return nil
}

// loadModels times cfg.loads loads per mode, rotating which mode goes
// first, and keeps the last model of each.
func loadModels(cfg config, res *bench.TextGoResults) ([]*loaded, error) {
	models := make([]*loaded, len(cfg.modes))
	times := make([][]float64, len(cfg.modes))

	for i := range cfg.loads {
		for j := range cfg.modes {
			k := (j + i) % len(cfg.modes)
			models[k] = nil

			runtime.GC()

			start := time.Now()

			l, err := load(cfg, cfg.modes[k])
			if err != nil {
				return nil, err
			}

			times[k] = append(times[k], ms(time.Since(start)))
			models[k] = l
		}
	}

	for k, mode := range cfg.modes {
		res.Modes[k].Load = bench.Summarise(times[k])
		_, _ = fmt.Fprintf(cfg.log, "%-11s load: median %.1f ms over %d\n", mode, res.Modes[k].Load.Median,
			res.Modes[k].Load.N)
	}

	return models, nil
}

// timeCases answers every case cfg.warmUp + cfg.passes times with every
// model, rotating which mode goes first per case and pass. The first pass
// gives the function_calls compared with the engine's.
func timeCases(cfg config, res *bench.TextGoResults, models []*loaded, cases []textCase) error {
	samples := make([]bench.TextSamples, len(models))

	for pass := range cfg.warmUp + cfg.passes {
		for i, c := range cases {
			for j := range models {
				k := (j + pass + i) % len(models)

				a, err := models[k].answer(c)
				if err != nil {
					return fmt.Errorf("%s: %w", cfg.modes[k], err)
				}

				if pass == 0 && bench.FunctionCallsEqual(a.calls, c.EngineCalls) {
					res.Modes[k].FunctionCallsEqual++
				}

				if pass >= cfg.warmUp {
					samples[k].Add(pass-cfg.warmUp, c.ID, a.TextAnswer)
				}
			}
		}

		_, _ = fmt.Fprintf(cfg.log, "pass %d of %d done\n", pass+1, cfg.warmUp+cfg.passes)
	}

	for k := range models {
		m := &res.Modes[k]
		m.Answers, m.Cases = samples[k].Set(), samples[k].Cases()

		_, _ = fmt.Fprintf(cfg.log, "%-11s %d cases: TTFT median %.1f ms, p90 %.1f ms; prefill %.0f tok/s; decode "+
			"median %.0f tok/s; answer %.1f ms; function_calls equal to the engine %d/%d\n", m.Mode, len(cases),
			m.Answers.TTFT.Median, m.Answers.TTFT.P90, m.Answers.PrefillTPS.Median, m.Answers.DecodeTPS.Median,
			m.Answers.Wall.Median, m.FunctionCallsEqual, len(cases))
	}

	return nil
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func mebi(bytes int64) float64 { return float64(bytes) / (1 << 20) }
