// Command needle-bench measures the Needle 3 text model in Go and renders
// the text benchmark report (PLAN.md 11.5's Accept, docs/FINDINGS.md
// F-061): load time, time to first token, prefill and decode rates, the
// answer's wall time and peak RSS on the first turns of the engine corpus
// (testdata/text/engine/cases.json), in four modes: needle3.safetensors in
// float32, needle3.cact packed on exact kernels, packed with the engine's
// int8 numerics and KV window, and the same on the lut kernel. It is
// Whistle's whistle-bench for the text model, a command of its own because
// it loads other models and answers other inputs; it shares the bench
// package's statistics, machine description and table layout. The engine's
// side is `oracle/text bench` (tools/oracle/text); the report puts both
// together.
//
//	go run ./bench/cmd/needle-bench run      # measure: bench/results/text-go.json, then bench/RESULTS-text.md
//	go run ./bench/cmd/needle-bench report   # re-render bench/RESULTS-text.md from both JSON files
//
// It runs from the repository root, where the default paths resolve.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"

	"github.com/cwbudde/go-whistle/bench"
)

const usage = "usage: needle-bench run [flags] | needle-bench report [-go file] [-engine file] [-report file]"

// memLine prefixes the memory child's report on its stdout.
const memLine = "MEMREPORT "

// memReport is a fresh process's memory around the load and one pass over
// the cases.
type memReport struct {
	Mode       string `json:"mode"`
	MaxRSSLoad int64  `json:"max_rss_load"` // ru_maxrss in bytes after the load
	HeapLoad   int64  `json:"heap_load"`    // the live Go heap after the load, in bytes
	MaxRSSPeak int64  `json:"max_rss_peak"` // ru_maxrss after the answers
	Answers    int    `json:"answers"`
}

func main() {
	err := dispatch(context.Background(), os.Args[1:], os.Stdout, os.Stderr)
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "needle-bench:", err)

		os.Exit(1)
	}
}

func dispatch(ctx context.Context, args []string, stdout, log io.Writer) error {
	if len(args) == 0 {
		return errors.New(usage)
	}

	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(log)

	goPath := fs.String("go", bench.TextGoResultsPath, "the Go results (JSON)")
	enginePath := fs.String("engine", bench.TextEngineResultsPath,
		"the engine results (JSON), written by `oracle/text bench`")
	reportPath := fs.String("report", bench.TextReportPath, "the report to write")

	cfg := config{log: log}
	fs.StringVar(&cfg.cact, "cact", "models/needle3/needle3.cact", "the packed model")
	fs.StringVar(&cfg.safetensors, "safetensors", "models/needle3/needle3.safetensors", "the float model")
	fs.StringVar(&cfg.tokenizer, "tokenizer", "models/needle3/tokenizer.model", "the tokenizer")
	fs.StringVar(&cfg.cases, "cases", "testdata/text/engine/cases.json", "the case corpus")
	fs.StringVar(&cfg.responses, "responses", "testdata/text/engine/responses.json", "the engine's recorded answers")

	switch args[0] {
	case "report":
		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("report: %w", err)
		}

		return report(*goPath, *enginePath, *reportPath)
	case "child":
		mode := fs.String("mode", bench.TextModePacked, "the mode to load")

		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("child: %w", err)
		}

		return memChild(stdout, cfg, *mode)
	case "run":
		err := runFlags(fs, &cfg, args[1:])
		if err != nil {
			return err
		}

		cfg.memory = func(mode string) (memReport, error) { return spawnChild(ctx, log, cfg, mode) }

		return run(ctx, cfg, *goPath, *enginePath, *reportPath)
	default:
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
}

// allModes are the modes a run measures by default, in the report's order.
func allModes() []string {
	out := make([]string, 0, len(bench.TextModes))
	for _, m := range bench.TextModes {
		out = append(out, m.Name)
	}

	return out
}

// runFlags parses the run command's own flags into cfg and applies -procs.
func runFlags(fs *flag.FlagSet, cfg *config, args []string) error {
	fs.IntVar(&cfg.warmUp, "warmup", 1, "untimed passes over the cases")
	fs.IntVar(&cfg.passes, "passes", 5, "timed passes over the cases")
	fs.IntVar(&cfg.loads, "loads", 5, "timed loads per mode")
	modes := fs.String("modes", strings.Join(allModes(), ","), "the modes to measure, comma-separated")
	procs := fs.Int("procs", 0, "GOMAXPROCS for the timing process; 0 keeps the default")

	err := fs.Parse(args)
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}

	switch {
	case cfg.warmUp < 0:
		return fmt.Errorf("run: -warmup %d: want 0 or more\n%s", cfg.warmUp, usage)
	case cfg.passes < 1:
		return fmt.Errorf("run: -passes %d: want at least 1 timed pass over the cases\n%s", cfg.passes, usage)
	case cfg.loads < 1:
		return fmt.Errorf("run: -loads %d: want at least 1 timed load\n%s", cfg.loads, usage)
	case *procs < 0:
		return fmt.Errorf("run: -procs %d: want 0 (the default) or more\n%s", *procs, usage)
	}

	cfg.modes = strings.Split(*modes, ",")
	for _, m := range cfg.modes {
		if !slices.Contains(allModes(), m) {
			return fmt.Errorf("run: -modes: unknown mode %q, want some of %s\n%s", m, strings.Join(allModes(), ","),
				usage)
		}
	}

	if *procs > 0 {
		runtime.GOMAXPROCS(*procs)
	}

	return nil
}

// run measures, writes the Go results and renders the report.
func run(ctx context.Context, cfg config, goPath, enginePath, reportPath string) error {
	for _, p := range []string{cfg.cact, cfg.safetensors, cfg.tokenizer} {
		if !exists(p) {
			return fmt.Errorf("%s missing; run `just fetch-needle3`", p)
		}
	}

	cases, err := loadCases(cfg.cases, cfg.responses)
	if err != nil {
		return err
	}

	res, err := measure(ctx, cfg, cases)
	if err != nil {
		return err
	}

	err = bench.WriteJSON(goPath, res)
	if err != nil {
		return fmt.Errorf("write results: %w", err)
	}

	_, _ = fmt.Fprintf(cfg.log, "wrote %s\n", goPath)

	return report(goPath, enginePath, reportPath)
}

// report renders reportPath from the results that exist; a side without
// its JSON file is shown as not run.
func report(goPath, enginePath, reportPath string) error {
	err := bench.WriteTextReport(goPath, enginePath, reportPath)
	if err != nil {
		return fmt.Errorf("report: %w", err)
	}

	return nil
}

// spawnChild runs this binary as a memory child for mode and parses its
// report.
func spawnChild(ctx context.Context, log io.Writer, cfg config, mode string) (memReport, error) {
	self, err := os.Executable()
	if err != nil {
		return memReport{}, fmt.Errorf("memory child: %w", err)
	}

	var stdout bytes.Buffer

	// #nosec G204 -- this binary with its own flags.
	cmd := exec.CommandContext(ctx, self, "child", "-mode", mode, "-cact", cfg.cact, "-safetensors", cfg.safetensors,
		"-tokenizer", cfg.tokenizer, "-cases", cfg.cases, "-responses", cfg.responses)
	cmd.Stdout, cmd.Stderr = &stdout, log

	err = cmd.Run()
	if err != nil {
		return memReport{}, fmt.Errorf("%s memory child: %w", mode, err)
	}

	return parseMemReport(&stdout, mode)
}

// parseMemReport finds the memLine in a child's output.
func parseMemReport(out io.Reader, mode string) (memReport, error) {
	sc := bufio.NewScanner(out)
	for sc.Scan() {
		js, ok := strings.CutPrefix(sc.Text(), memLine)
		if !ok {
			continue
		}

		var r memReport

		err := json.Unmarshal([]byte(js), &r)
		if err != nil {
			return memReport{}, fmt.Errorf("%s memory child: %w", mode, err)
		}

		return r, nil
	}

	return memReport{}, fmt.Errorf("%s memory child: no %q line", mode, memLine)
}

// memChild is the memory child: the load, then one answer to every case,
// with ru_maxrss after each and the live heap after the load, printed as
// one memLine. ru_maxrss is a high-water mark, so the model loads before
// the corpus is read: the sample after the load sees the model alone.
func memChild(stdout io.Writer, cfg config, mode string) error {
	l, err := load(cfg, mode)
	if err != nil {
		return err
	}

	runtime.GC()

	var mem runtime.MemStats

	runtime.ReadMemStats(&mem)

	r := memReport{Mode: mode, MaxRSSLoad: maxRSS(), HeapLoad: int64(mem.HeapAlloc)} // #nosec G115 -- a heap size.

	cases, err := loadCases(cfg.cases, cfg.responses)
	if err != nil {
		return err
	}

	for _, c := range cases {
		_, err := l.answer(c)
		if err != nil {
			return err
		}
	}

	r.MaxRSSPeak, r.Answers = maxRSS(), len(cases)

	js, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("memory report: %w", err)
	}

	_, err = fmt.Fprintln(stdout, memLine+string(js))
	if err != nil {
		return fmt.Errorf("memory report: %w", err)
	}

	return nil
}

func exists(path string) bool {
	_, err := os.Stat(path)

	return err == nil
}
