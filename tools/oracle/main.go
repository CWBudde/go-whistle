//go:build oracle

// Command oracle builds the engine-oracle fixtures that the Go port is validated against.
//
//	go run -tags oracle ./tools/oracle clips   # fetch the clip corpus into testdata/audio
//	go run -tags oracle ./tools/oracle run     # record needle transcripts and embeddings into testdata/oracle
//	go run -tags oracle ./tools/oracle probes  # record embeddings of the synthetic probes into testdata/oracle/probes
//
// It loads the closed-source Needle 3 engine (the needle CLI and libneedle3 through purego), so it lives behind the
// oracle build tag and never ships with the library or the whistle CLI.
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

// cmdNameProbes is the probes subcommand's name.
const cmdNameProbes = "probes"

const usage = "usage: oracle clips [-audio dir] | oracle run [-audio dir] [-out dir] [-models dir] [-engine dir] | " +
	"oracle probes [-out dir] [-models dir] [-engine dir]"

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err := dispatch(ctx, os.Args[1:], os.Stderr)

	stop()

	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "oracle:", err)

		os.Exit(1)
	}
}

func dispatch(ctx context.Context, args []string, log io.Writer) error {
	if len(args) == 0 {
		return errors.New(usage)
	}

	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(log)
	audio := fs.String("audio", filepath.Join("testdata", "audio"), "clip corpus directory")

	switch args[0] {
	case "clips":
		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("clips: %w", err)
		}

		return cmdClips(ctx, *audio, log)
	case "run", cmdNameProbes:
		plat, err := platformDir()
		if err != nil {
			return err
		}

		defOut := filepath.Join("testdata", "oracle")
		if args[0] == cmdNameProbes {
			defOut = filepath.Join(defOut, "probes")
		}

		out := fs.String("out", defOut, "fixture output directory")
		models := fs.String("models", filepath.Join("models", "whistle"), "directory holding whistle.cact")
		eng := fs.String("engine", filepath.Join("engine", plat), "directory holding needle and libneedle3")

		err = fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("%s: %w", args[0], err)
		}

		cfg := runConfig{audioDir: *audio, outDir: *out, modelDir: *models, engineDir: *eng}
		if args[0] == cmdNameProbes {
			return cmdProbes(cfg, log)
		}

		return cmdRun(ctx, cfg, log)
	default:
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}
}
