//go:build oracle

// Command text records the Needle 3 engine's answers to tool-calling requests, the fixtures the Go port of the
// Needle 3 text model is validated against. It reads the case corpus testdata/text/engine/cases.json and writes next
// to it:
//
//	go run -tags oracle ./tools/oracle/text cases  # responses.json, library.json, manifest.json: the CLI's answers
//	                                              # (also checked at other thread counts, with --serve and with
//	                                              # libneedle3)
//	go run -tags oracle ./tools/oracle/text date   # date.json: what the engine does without a date: fact
//	go run -tags oracle ./tools/oracle/text depth  # depth.json: which --depth values load, answers along the ladder
//	go run -tags oracle ./tools/oracle/text audio  # audio.json: speech in, tool calls out (needle3 + whistle)
//	go run -tags oracle ./tools/oracle/text embed  # embed.json, embed.safetensors: needle_embed of short texts
//	go run -tags oracle ./tools/oracle/text retrieval  # retrieval.json: needle_init past 5 tools, the tool index
//	go run -tags oracle ./tools/oracle/text all    # every subcommand above
//	go run -tags oracle ./tools/oracle/text fetch  # download the pinned text model into models/needle3 (-dir)
//
// It runs the closed-source Needle 3 engine (the needle CLI, and libneedle3 through purego), always with
// NEEDLE_TELEMETRY=0 and DO_NOT_TRACK=1, so it lives behind the oracle build tag and never ships with the library or
// the whistle CLI. Every fixture drops the engine's timing fields and is laid out as prettier formats it, so a
// re-run of the same engine diffs clean and treefmt leaves the files alone.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
)

// subcommands in the order `all` runs them.
var subcommands = []struct {
	name string
	run  func(context.Context, config, io.Writer) error
}{
	{"cases", cmdCases},
	{"date", cmdDate},
	{"depth", cmdDepth},
	{"audio", cmdAudio},
	{"embed", cmdEmbed},
	{"retrieval", cmdRetrieval},
}

const usage = "usage: oracle/text cases|date|depth|audio|embed|retrieval|all " +
	"[-out dir] [-cases file] [-needle3 dir] [-whistle dir] [-engine dir] [-audio dir]\n" +
	"       oracle/text fetch [-dir models/needle3]"

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err := dispatch(ctx, os.Args[1:], os.Stderr)

	stop()

	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "oracle/text:", err)

		os.Exit(1)
	}
}

func dispatch(ctx context.Context, args []string, log io.Writer) error {
	if len(args) == 0 {
		return errors.New(usage)
	}

	if args[0] == "fetch" {
		return cmdFetch(ctx, args[1:], log)
	}

	plat, err := platformDir()
	if err != nil {
		return err
	}

	fixtures := filepath.Join("testdata", "text", "engine")
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(log)

	var cfg config

	fs.StringVar(&cfg.outDir, "out", fixtures, "fixture output directory")
	fs.StringVar(&cfg.casesPath, "cases", filepath.Join(fixtures, "cases.json"), "case corpus")
	fs.StringVar(&cfg.needle3Dir, "needle3", filepath.Join("models", "needle3"), "directory holding needle3.cact")
	fs.StringVar(&cfg.whistleDir, "whistle", filepath.Join("models", "whistle"), "directory holding whistle.cact")
	fs.StringVar(&cfg.engineDir, "engine", filepath.Join("engine", plat), "directory holding needle and libneedle3")
	fs.StringVar(&cfg.audioDir, "audio", filepath.Join("testdata", "audio"), "clip corpus directory")

	err = fs.Parse(args[1:])
	if err != nil {
		return fmt.Errorf("%s: %w", args[0], err)
	}

	for _, s := range subcommands {
		if args[0] != s.name && args[0] != "all" {
			continue
		}

		err := s.run(ctx, cfg, log)
		if err != nil {
			return fmt.Errorf("%s: %w", s.name, err)
		}

		if args[0] == s.name {
			return nil
		}
	}

	if args[0] == "all" {
		return nil
	}

	return fmt.Errorf("unknown command %q\n%s", args[0], usage)
}
