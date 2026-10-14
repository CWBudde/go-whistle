//go:build oracle

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/cwbudde/go-whistle/bench"
	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// benchConfig is one engine run of the text bench (docs/FINDINGS.md F-061).
type benchConfig struct {
	config

	out, goResults, report string // the results JSON, and the Go results and report rendered from both
	responses, library     string // the recorded answers and needle_complete counts of the corpus
	warmUp, passes, loads  int    // untimed passes, timed passes over the cases, timed needle_load calls
}

// check rejects counts that leave nothing to time.
func (c benchConfig) check() error {
	switch {
	case c.warmUp < 0:
		return fmt.Errorf("-warmup %d: want 0 or more", c.warmUp)
	case c.passes < 1:
		return fmt.Errorf("-passes %d: want at least 1 timed pass over the cases", c.passes)
	case c.loads < 1:
		return fmt.Errorf("-loads %d: want at least 1 timed needle_load", c.loads)
	}

	return nil
}

// benchUsage is the bench subcommand's usage.
const benchUsage = "usage: oracle/text bench [-out file] [-go file] [-report file] [-warmup n] [-passes n] [-loads n] " +
	"[-cases file] [-needle3 dir] [-engine dir]"

// cmdBenchArgs parses the bench subcommand's flags and runs it.
func cmdBenchArgs(ctx context.Context, args []string, log io.Writer) error {
	plat, err := platformDir()
	if err != nil {
		return err
	}

	fixtures := filepath.Join("testdata", "text", "engine")
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	fs.SetOutput(log)

	cfg := benchConfig{config: config{outDir: fixtures}}
	fs.StringVar(&cfg.out, "out", bench.TextEngineResultsPath, "the engine results to write (JSON)")
	fs.StringVar(&cfg.goResults, "go", bench.TextGoResultsPath, "the Go results (JSON), written by needle-bench")
	fs.StringVar(&cfg.report, "report", bench.TextReportPath, "the report to render")
	fs.IntVar(&cfg.warmUp, "warmup", 1, "untimed passes over the cases")
	fs.IntVar(&cfg.passes, "passes", 5, "timed passes over the cases")
	fs.IntVar(&cfg.loads, "loads", 5, "timed needle_load calls")
	fs.StringVar(&cfg.casesPath, "cases", filepath.Join(fixtures, "cases.json"), "case corpus")
	fs.StringVar(&cfg.responses, "responses", filepath.Join(fixtures, responsesFile), "the CLI's recorded answers")
	fs.StringVar(&cfg.library, "library", filepath.Join(fixtures, libraryFile), "libneedle3's recorded counts")
	fs.StringVar(&cfg.needle3Dir, "needle3", filepath.Join("models", "needle3"), "directory holding needle3.cact")
	fs.StringVar(&cfg.engineDir, "engine", filepath.Join("engine", plat), "directory holding needle and libneedle3")

	err = fs.Parse(args)
	if err != nil {
		return fmt.Errorf("bench: %w\n%s", err, benchUsage)
	}

	err = cfg.check()
	if err != nil {
		return fmt.Errorf("bench: %w\n%s", err, benchUsage)
	}

	return cmdBench(ctx, cfg, log)
}

// benchCase is one first turn of the corpus with what the bench needs
// beside it: the prefill's token count as the Go port renders it, and the
// engine's recorded function_calls and generated-token count.
type benchCase struct {
	k             textCase
	prefillTokens int
	staticTokens  int
	generated     int
	calls         json.RawMessage
}

// engineTimingText is the engine's answer to a text turn, with the timing
// fields that the fixtures drop (timingSuffixes).
type engineTimingText struct {
	Calls      json.RawMessage `json:"function_calls"`
	PrefillTPS float64         `json:"prefill_tps"`
	DecodeTPS  float64         `json:"decode_tps"`
	PeakRAMMB  float64         `json:"peak_ram_mb"`
}

// parseTiming reads an engine answer's function_calls and timing fields;
// an answer without its prefill rate or peak RAM is an error, since the
// bench reports them.
func parseTiming(out []byte) (engineTimingText, error) {
	var t engineTimingText

	err := json.Unmarshal(bytes.TrimSpace(out), &t)
	if err != nil {
		return t, fmt.Errorf("engine output: %w: %q", err, out)
	}

	if t.PrefillTPS <= 0 || t.PeakRAMMB <= 0 {
		return t, fmt.Errorf("engine output without prefill_tps or peak_ram_mb: %q", out)
	}

	return t, nil
}

// answer is the TextAnswer of an engine answer to c with the given TTFT
// (0: unknown) and wall time in ms. Its prefill rate is the prefill's
// tokens over the TTFT when the answer prefilled them all (prefilled), as
// the Go side's; the engine's own prefill_tps is kept beside it.
func (c benchCase) answer(t engineTimingText, ttft, wall float64, prefilled bool) bench.TextAnswer {
	a := bench.TextAnswer{
		PrefillTokens: c.prefillTokens, Generated: c.generated, TTFTMS: ttft, ReportedPrefillTPS: t.PrefillTPS,
		DecodeTPS: t.DecodeTPS, WallMS: wall,
	}

	if prefilled && ttft > 0 {
		a.PrefillTPS = 1000 * float64(c.prefillTokens) / ttft
	}

	return a
}

// outsideDecode is the part of a needle_complete call of call ms that its
// decode clock does not cover: the call less (n − 1) / decode_tps for the
// n tokens it generated. That is the turn's prefill and first token plus
// what follows the last one (the confidence, the envelope). needle_init
// prefills the static prefix before it, and prefill_tps counts the turn's
// tokens alone, so neither gives the TTFT by itself (docs/FINDINGS.md
// F-061).
func outsideDecode(call float64, generated int, decodeTPS float64) float64 {
	if decodeTPS <= 0 || generated < 2 {
		return call
	}

	return call - 1000*float64(generated-1)/decodeTPS
}

// cmdBench times the engine on the first turns of the corpus, the turns
// needle-bench times: the needle CLI, one process per answer at the pinned
// threads, and libneedle3 in this process. It writes
// bench.TextEngineResults and renders the text report.
func cmdBench(ctx context.Context, cfg benchConfig, log io.Writer) error {
	o, err := openOracle(cfg.config, false)
	if err != nil {
		return err
	}
	defer o.close()

	res := bench.TextEngineResults{
		Run: bench.StartRun(ctx), Version: o.pin["engine_version"], Revision: o.pin["revision"], Threads: cliThreads,
		WarmUp: cfg.warmUp,
	}

	res.Run.Inputs.Model, err = bench.FileSHA256(o.needle3Path())
	if err != nil {
		return fmt.Errorf("inputs: %w", err)
	}

	res.Run.Inputs.Clips, err = bench.FileSHA256(cfg.casesPath)
	if err != nil {
		return fmt.Errorf("inputs: %w", err)
	}

	cases, err := o.benchCases(cfg)
	if err != nil {
		return err
	}

	res.CLI, err = o.benchCLI(ctx, cfg, cases, log)
	if err != nil {
		return err
	}

	res.Library, err = o.benchLibrary(cfg, cases, log)
	if err != nil {
		return err
	}

	res.Run.Finish(ctx)

	err = bench.WriteJSON(cfg.out, res)
	if err != nil {
		return fmt.Errorf("write results: %w", err)
	}

	_, _ = fmt.Fprintf(log, "wrote %s\n", cfg.out)

	err = bench.WriteTextReport(cfg.goResults, cfg.out, cfg.report)
	if err != nil {
		return fmt.Errorf("report: %w", err)
	}

	_, _ = fmt.Fprintf(log, "wrote %s\n", cfg.report)

	return nil
}

// benchCases are the corpus's first turns with their prefill token counts
// (the Go port's rendering, whose static prefix needle_init counts alike,
// F-048) and the recorded function_calls and needle_complete counts.
func (o *oracle) benchCases(cfg benchConfig) ([]benchCase, error) {
	p, err := spm.Load(filepath.Join(o.cfg.needle3Dir, "tokenizer.model"))
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	var (
		responses map[string][]engineTimingText
		library   struct {
			Cases map[string]libraryCase `json:"cases"`
		}
	)

	for path, v := range map[string]any{cfg.responses: &responses, cfg.library: &library} {
		raw, err := os.ReadFile(path) // #nosec G304 -- the fixture paths are flags.
		if err != nil {
			return nil, fmt.Errorf("read: %w", err)
		}

		err = json.Unmarshal(raw, v)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	}

	think := len(p.Encode("<think>\n"))
	out := make([]benchCase, 0, len(o.corpus.Cases))

	for _, k := range o.corpus.Cases {
		rec, lib := responses[k.ID], library.Cases[k.ID]
		if len(rec) == 0 || len(lib.Complete) == 0 {
			return nil, fmt.Errorf("%s: no recorded answer", k.ID)
		}

		n, err := toolcall.Normalise([]byte(o.corpus.tools(k.Toolset)))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", k.ID, err)
		}

		system := o.corpus.system(k)

		static := len(toolcall.StaticPrefix(p, system, n.JSON))
		if static != int(lib.Init) {
			return nil, fmt.Errorf("%s: static prefix of %d tokens, needle_init counted %d", k.ID, static, lib.Init)
		}

		turn, err := toolcall.NewConversation(p, system, n.JSON, contextTokens).Turn(k.Query, k.max(), true)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", k.ID, err)
		}

		out = append(out, benchCase{
			k: k, staticTokens: static, prefillTokens: static + len(turn) + min(think, k.max()),
			generated: int(lib.Complete[0]), calls: rec[0].Calls,
		})
	}

	return out, nil
}

// contextTokens is needle3's max_seq_len, the context a conversation's
// turns must fit (docs/FINDINGS.md F-049).
const contextTokens = 8192

// benchCLI answers every case with the needle CLI cfg.warmUp + cfg.passes
// times, pass by pass, one process per answer.
func (o *oracle) benchCLI(ctx context.Context, cfg benchConfig, cases []benchCase, log io.Writer,
) (bench.TextEngineCLI, error) {
	var (
		out      bench.TextEngineCLI
		samples  bench.TextSamples
		ram, rss []float64
	)

	for pass := range cfg.warmUp + cfg.passes {
		for _, c := range cases {
			args, err := o.baseArgs(c.k.Toolset, o.corpus.system(c.k))
			if err != nil {
				return out, err
			}

			t, wall, maxRSS, err := cliAnswer(ctx, o.needle(), append(args, maxArgs(c.k, "--prompt", c.k.Query)...))
			if err != nil {
				return out, fmt.Errorf("%s: %w", c.k.ID, err)
			}

			if pass == 0 && bench.FunctionCallsEqual(t.Calls, c.calls) {
				out.FunctionCallsEqual++
			}

			if pass >= cfg.warmUp {
				samples.Add(pass-cfg.warmUp, c.k.ID, c.answer(t, 0, wall, false))
				ram, rss = append(ram, t.PeakRAMMB), append(rss, maxRSS)
			}
		}
	}

	out.Answers, out.Cases = samples.Set(), samples.Cases()
	out.PeakRAM, out.MaxRSS = bench.Summarise(ram), bench.Summarise(rss)

	_, _ = fmt.Fprintf(log, "CLI     %d cases: prefill_tps %.0f; decode_tps %.0f; process %.1f ms; "+
		"max RSS %.1f MiB; function_calls equal to the recording %d/%d\n", len(cases), out.Answers.ReportedPrefillTPS.Median, out.Answers.DecodeTPS.Median, out.Answers.Wall.Median, out.MaxRSS.Max,
		out.FunctionCallsEqual, len(cases))

	return out, nil
}

// cliAnswer runs the needle CLI with args and returns its answer, the
// process's wall time in ms and its ru_maxrss in MiB.
func cliAnswer(ctx context.Context, needle string, args []string) (engineTimingText, float64, float64, error) {
	cmd := command(ctx, needle, args)

	var stdout, stderr bytes.Buffer

	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	start := time.Now()
	err := cmd.Run()
	wall := float64(time.Since(start).Nanoseconds()) / 1e6

	if err != nil {
		return engineTimingText{}, 0, 0, fmt.Errorf("needle: %w: %s", err, stderr.String())
	}

	t, err := parseTiming(stdout.Bytes())
	if err != nil {
		return engineTimingText{}, 0, 0, err
	}

	var rss float64
	if ru, ok := cmd.ProcessState.SysUsage().(*syscall.Rusage); ok {
		rss = float64(bench.MaxRSSBytes(ru.Maxrss)) / (1 << 20)
	}

	return t, wall, rss, nil
}

// benchLibrary loads libneedle3 in this process, times cfg.loads
// needle_load calls on the .cact bytes in memory, then answers every case
// cfg.warmUp + cfg.passes times, pass by pass: needle_init, needle_reset
// and needle_complete, as the recorded library cases run. Each case is
// answered cold, its static prefix prefilled in the call, and then again
// with the prefix cached.
func (o *oracle) benchLibrary(cfg benchConfig, cases []benchCase, log io.Writer) (bench.TextEngineLibrary, error) {
	var out bench.TextEngineLibrary

	lib, err := openLibrary(o.cfg.engineDir)
	if err != nil {
		return out, err
	}

	cact, err := os.ReadFile(o.needle3Path())
	if err != nil {
		return out, fmt.Errorf("read model: %w", err)
	}

	if len(cact) == 0 {
		return out, errors.New("needle3.cact is empty")
	}

	loads := make([]float64, 0, cfg.loads)

	for range cfg.loads {
		start := time.Now()
		rc := lib.load(&cact[0], uint64(len(cact)))

		loads = append(loads, float64(time.Since(start).Nanoseconds())/1e6)

		if rc < 0 {
			return out, fmt.Errorf("needle_load: %s (%d)", lib.lastError(), rc)
		}
	}

	runtime.KeepAlive(cact)

	out.Load = bench.Summarise(loads)

	var (
		cold, cached     bench.TextSamples
		inits, completes []float64
	)

	for pass := range cfg.warmUp + cfg.passes {
		for _, c := range cases {
			for _, evict := range []bool{true, false} {
				t, init, complete, err := o.libAnswer(lib, c, evict)
				if err != nil {
					return out, fmt.Errorf("%s: %w", c.k.ID, err)
				}

				if pass == 0 && evict && bench.FunctionCallsEqual(t.Calls, c.calls) {
					out.FunctionCallsEqual++
				}

				if pass == 0 && !evict && bench.FunctionCallsEqual(t.Calls, c.calls) {
					out.CachedFunctionCallsEqual++
				}

				if pass < cfg.warmUp {
					continue
				}

				a := c.answer(t, init+outsideDecode(complete, c.generated, t.DecodeTPS), init+complete, evict)

				if evict {
					cold.Add(pass-cfg.warmUp, c.k.ID, a)
					inits, completes = append(inits, init), append(completes, complete)
				} else {
					cached.Add(pass-cfg.warmUp, c.k.ID, a)
				}
			}
		}
	}

	out.Answers, out.Cases, out.Cached = cold.Set(), cold.Cases(), cached.Set()
	out.Init, out.Complete = bench.Summarise(inits), bench.Summarise(completes)

	_, _ = fmt.Fprintf(log, "library needle_load median %.1f ms; %d cases: needle_init %.1f ms, needle_complete %.1f ms; "+
		"TTFT median %.1f ms (prefix cached %.1f ms); prefill_tps %.0f; decode_tps %.0f; function_calls equal to the "+
		"recording %d/%d (prefix cached %d/%d)\n", out.Load.Median, len(cases), out.Init.Median, out.Complete.Median,
		out.Answers.TTFT.Median, out.Cached.TTFT.Median, out.Answers.PrefillTPS.Median, out.Answers.DecodeTPS.Median,
		out.FunctionCallsEqual, len(cases), out.CachedFunctionCallsEqual, len(cases))

	return out, nil
}

// evictInput is the turn the bench answers on another static prefix before
// a cold answer.
const evictInput = "Hello"

// libAnswer answers case c through libneedle3 and returns the answer and
// the wall times of needle_init and needle_complete in ms. needle_init
// prefills the static prefix and keeps it while it gets the same system
// text and tools (F-061). With evict, it first answers one token on
// another system text, so that this answer prefills its static prefix, as
// the CLI and the Go side do.
func (o *oracle) libAnswer(lib *library, c benchCase, evict bool) (engineTimingText, float64, float64, error) {
	system := o.corpus.system(c.k)

	if evict {
		_, err := lib.session(system+"\nbench: another prefix", o.corpus.tools(c.k.Toolset))
		if err != nil {
			return engineTimingText{}, 0, 0, err
		}

		lib.reset()

		_, err = lib.completeText(evictInput, 1)
		if err != nil {
			return engineTimingText{}, 0, 0, err
		}
	}

	start := time.Now()

	rc, err := lib.session(system, o.corpus.tools(c.k.Toolset))
	if err != nil {
		return engineTimingText{}, 0, 0, err
	}

	init := float64(time.Since(start).Nanoseconds()) / 1e6

	if int(rc) != c.staticTokens {
		return engineTimingText{}, 0, 0, fmt.Errorf("needle_init counted %d tokens, the recording %d", rc, c.staticTokens)
	}

	lib.reset()

	start = time.Now()

	comp, err := lib.completeText(c.k.Query, c.k.max())
	if err != nil {
		return engineTimingText{}, 0, 0, err
	}

	complete := float64(time.Since(start).Nanoseconds()) / 1e6

	if int(comp.rc) != c.generated {
		return engineTimingText{}, 0, 0, fmt.Errorf("needle_complete generated %d tokens, the recording %d", comp.rc,
			c.generated)
	}

	t, err := parseTiming(comp.out)

	return t, init, complete, err
}
