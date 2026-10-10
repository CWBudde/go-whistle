// Command whistle-bench measures go-whistle against itself and renders the
// benchmark report (PLAN.md 9.6 and the Phase 9 Accept, docs/FINDINGS.md
// F-040): Load time, TTFT, decode rate, real-time factor and peak RSS on
// the 32 fixture clips and a 30 s clip, and the LibriSpeech WER, each
// packed against float. The engine's side is `oracle bench`
// (tools/oracle); the report puts both together.
//
//	go run ./bench/cmd/whistle-bench run      # measure: bench/results/go.json, then bench/RESULTS.md
//	go run ./bench/cmd/whistle-bench report   # re-render bench/RESULTS.md from both JSON files
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
	"strings"

	"github.com/cwbudde/go-whistle/bench"
	"github.com/cwbudde/go-whistle/whistle"
)

const usage = "usage: whistle-bench run [flags] | whistle-bench report [-go file] [-engine file] [-report file]"

// memLine prefixes the memory child's report on its stdout.
const memLine = "MEMREPORT "

// memReport is a fresh process's memory around Load and one 30 s
// Transcribe, as `just mem-probe` measures it (docs/FINDINGS.md F-037).
type memReport struct {
	Mode       string `json:"mode"`
	MaxRSSLoad int64  `json:"max_rss_load"` // ru_maxrss in bytes after Load
	MaxRSSPeak int64  `json:"max_rss_peak"` // and after the Transcribe
	Text       string `json:"text"`
}

func main() {
	err := dispatch(context.Background(), os.Args[1:], os.Stdout, os.Stderr)
	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "whistle-bench:", err)

		os.Exit(1)
	}
}

func dispatch(ctx context.Context, args []string, stdout, log io.Writer) error {
	if len(args) == 0 {
		return errors.New(usage)
	}

	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(log)

	goPath := fs.String("go", bench.GoResultsPath, "the Go results (JSON)")
	enginePath := fs.String("engine", bench.EngineResultsPath, "the engine results (JSON), written by `oracle bench`")
	reportPath := fs.String("report", bench.ReportPath, "the report to write")
	model := fs.String("model", "models/whistle/whistle.cact", "the model")
	audio := fs.String("audio", "testdata/audio", "the fixture clips")
	oracle := fs.String("oracle", "testdata/oracle", "the engine's committed transcripts of the clips")

	switch args[0] {
	case "report":
		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("report: %w", err)
		}

		return report(*goPath, *enginePath, *reportPath)
	case "child":
		mode := fs.String("mode", "packed", "packed or float")

		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("child: %w", err)
		}

		return memChild(stdout, *model, *audio, *mode)
	case "run":
		cfg := config{log: log}

		err := runFlags(fs, &cfg, args[1:])
		if err != nil {
			return err
		}

		cfg.model, cfg.audio, cfg.oracle = *model, *audio, *oracle
		cfg.memory = func(mode string) (memReport, error) { return spawnChild(ctx, log, cfg, mode) }

		return run(ctx, cfg, *goPath, *enginePath, *reportPath)
	default:
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
}

// runFlags parses the run command's own flags into cfg and applies -procs.
func runFlags(fs *flag.FlagSet, cfg *config, args []string) error {
	fs.StringVar(&cfg.lsDir, "librispeech", "testdata/oracle/librispeech", "the LibriSpeech subset's fixtures")
	fs.StringVar(&cfg.lsAudio, "corpus", "models/corpus/librispeech", "the LibriSpeech subset's audio")
	fs.IntVar(&cfg.warmUp, "warmup", 1, "untimed passes over the clips, and untimed 30 s runs")
	fs.IntVar(&cfg.passes, "passes", 5, "timed passes over the clips")
	fs.IntVar(&cfg.long, "long", 10, "timed runs of the 30 s clip")
	fs.IntVar(&cfg.loads, "loads", 5, "timed Loads per mode")
	fs.BoolVar(&cfg.wer, "wer", true, "measure the LibriSpeech WER (needs the corpus audio)")
	procs := fs.Int("procs", 0, "GOMAXPROCS for the timing process; 0 keeps the default")

	err := fs.Parse(args)
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}

	switch {
	case cfg.warmUp < 0:
		return fmt.Errorf("run: -warmup %d: want 0 or more\n%s", cfg.warmUp, usage)
	case cfg.passes < 1:
		return fmt.Errorf("run: -passes %d: want at least 1 timed pass over the clips\n%s", cfg.passes, usage)
	case cfg.long < 1:
		return fmt.Errorf("run: -long %d: want at least 1 timed 30 s run\n%s", cfg.long, usage)
	case cfg.loads < 1:
		return fmt.Errorf("run: -loads %d: want at least 1 timed Load\n%s", cfg.loads, usage)
	case *procs < 0:
		return fmt.Errorf("run: -procs %d: want 0 (the default) or more\n%s", *procs, usage)
	}

	if *procs > 0 {
		runtime.GOMAXPROCS(*procs)
	}

	return nil
}

// run measures, writes the Go results and renders the report.
func run(ctx context.Context, cfg config, goPath, enginePath, reportPath string) error {
	if !exists(cfg.model) {
		return fmt.Errorf("%s missing; run `whistle download`", cfg.model)
	}

	if cfg.wer && !exists(cfg.lsAudio) {
		return fmt.Errorf("%s missing; run `just oracle-librispeech-audio`, or pass -wer=false", cfg.lsAudio)
	}

	res, err := measure(ctx, cfg)
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
	g, e := new(bench.GoResults), new(bench.EngineResults)

	ok, err := bench.ReadJSON(goPath, g)
	if err != nil {
		return fmt.Errorf("report: %w", err)
	}

	if !ok {
		g = nil
	}

	ok, err = bench.ReadJSON(enginePath, e)
	if err != nil {
		return fmt.Errorf("report: %w", err)
	}

	if !ok {
		e = nil
	}

	var b bytes.Buffer

	err = bench.Render(&b, g, e)
	if err != nil {
		return fmt.Errorf("report: %w", err)
	}

	err = os.WriteFile(reportPath, b.Bytes(), 0o600) // #nosec G703 -- a flag or the default path.
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
	cmd := exec.CommandContext(ctx, self, "child", "-mode", mode, "-model", cfg.model, "-audio", cfg.audio, "-oracle", cfg.oracle)
	cmd.Stdout, cmd.Stderr = &stdout, log

	err = cmd.Run()
	if err != nil {
		return memReport{}, fmt.Errorf("%s memory child: %w", mode, err)
	}

	sc := bufio.NewScanner(&stdout)
	for sc.Scan() {
		js, ok := strings.CutPrefix(sc.Text(), memLine)
		if !ok {
			continue
		}

		var r memReport

		err = json.Unmarshal([]byte(js), &r)
		if err != nil {
			return memReport{}, fmt.Errorf("%s memory child: %w", mode, err)
		}

		return r, nil
	}

	return memReport{}, fmt.Errorf("%s memory child: no %q line", mode, memLine)
}

// memChild is the memory child: Load, one Transcribe of the 30 s clip,
// and ru_maxrss after each, printed as one memLine. ru_maxrss is a
// high-water mark, so the model loads before any audio is read: the
// sample after Load sees the model alone, and the 30 s clip is built from
// the first clip only.
func memChild(stdout io.Writer, model, audio, mode string) error {
	m, err := whistle.Load(model, options(mode)...)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}

	defer func() { _ = m.Close() }()

	r := memReport{Mode: mode, MaxRSSLoad: maxRSS()}

	pcm, err := longClip(audio)
	if err != nil {
		return err
	}

	res, err := m.Transcribe(pcm, whistle.TranscribeOptions{})
	if err != nil {
		return fmt.Errorf("transcribe: %w", err)
	}

	r.MaxRSSPeak, r.Text = maxRSS(), res.Text

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
