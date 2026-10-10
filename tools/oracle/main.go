//go:build oracle

// Command oracle builds the engine-oracle fixtures that the Go port is validated against.
//
//	go run -tags oracle ./tools/oracle clips              # fetch the clip corpus into testdata/audio
//	go run -tags oracle ./tools/oracle run                # record needle transcripts and embeddings into testdata/oracle
//	go run -tags oracle ./tools/oracle probes             # record embeddings of the synthetic probes into testdata/oracle/probes
//	go run -tags oracle ./tools/oracle depth              # transcripts at --audio-depth 2..7 into testdata/oracle/depth
//	go run -tags oracle ./tools/oracle keywords           # keyword-biasing cases into testdata/oracle/keywords
//	go run -tags oracle ./tools/oracle silence            # the empty-JSON sweep into testdata/oracle/silence
//	go run -tags oracle ./tools/oracle limits             # length limits into testdata/oracle/limits
//	go run -tags oracle ./tools/oracle librispeech-audio  # the LibriSpeech subset into models/corpus/librispeech
//	go run -tags oracle ./tools/oracle librispeech        # its transcripts into testdata/oracle/librispeech
//	go run -tags oracle ./tools/oracle stream             # stream API calls into testdata/oracle/stream
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

// Subcommand names.
const (
	cmdNameClips       = "clips"
	cmdNameRun         = "run"
	cmdNameProbes      = "probes"
	cmdNameDepth       = "depth"
	cmdNameKeywords    = "keywords"
	cmdNameSilence     = "silence"
	cmdNameLimits      = "limits"
	cmdNameLSAudio     = "librispeech-audio"
	cmdNameLibriSpeech = "librispeech"
	cmdNameStream      = "stream"
)

const usage = "usage: oracle clips [-audio dir] | oracle run|probes|depth|keywords|silence|limits|librispeech|stream " +
	"[-audio dir] [-out dir] [-oracle dir] [-corpus dir] [-models dir] [-engine dir] | " +
	"oracle librispeech-audio [-corpus dir] [-out dir]"

// engineCmds run the engine and write into testdata/oracle/<subdir>.
var engineCmds = map[string]struct {
	subdir string
	run    func(context.Context, runConfig, io.Writer) error
}{
	cmdNameRun:         {"", cmdRun},
	cmdNameProbes:      {"probes", func(_ context.Context, c runConfig, w io.Writer) error { return cmdProbes(c, w) }},
	cmdNameDepth:       {"depth", cmdDepth},
	cmdNameKeywords:    {"keywords", cmdKeywords},
	cmdNameSilence:     {"silence", cmdSilence},
	cmdNameLimits:      {"limits", cmdLimits},
	cmdNameLibriSpeech: {"librispeech", cmdLibriSpeech},
	cmdNameStream:      {"stream", cmdStream},
}

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

	oracleDir := filepath.Join("testdata", "oracle")
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(log)
	audio := fs.String("audio", filepath.Join("testdata", "audio"), "clip corpus directory")
	corpusDir := fs.String("corpus", filepath.Join("models", "corpus", datasetLibriSpeech),
		"LibriSpeech subset audio directory (git-ignored)")

	switch args[0] {
	case cmdNameClips:
		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("clips: %w", err)
		}

		return cmdClips(ctx, *audio, log)
	case cmdNameLSAudio:
		out := fs.String("out", filepath.Join(oracleDir, datasetLibriSpeech), "manifest directory")

		err := fs.Parse(args[1:])
		if err != nil {
			return fmt.Errorf("%s: %w", args[0], err)
		}

		_, err = ensureLibriSpeech(ctx, *corpusDir, *out, log)

		return err
	}

	sub, ok := engineCmds[args[0]]
	if !ok {
		return fmt.Errorf("unknown command %q\n%s", args[0], usage)
	}

	plat, err := platformDir()
	if err != nil {
		return err
	}

	out := fs.String("out", filepath.Join(oracleDir, sub.subdir), "fixture output directory")
	oracleFlag := fs.String("oracle", oracleDir, "directory holding the committed full-depth clip fixtures")
	models := fs.String("models", filepath.Join("models", "whistle"), "directory holding whistle.cact")
	eng := fs.String("engine", filepath.Join("engine", plat), "directory holding needle and libneedle3")

	err = fs.Parse(args[1:])
	if err != nil {
		return fmt.Errorf("%s: %w", args[0], err)
	}

	return sub.run(ctx, runConfig{
		audioDir: *audio, outDir: *out, oracleDir: *oracleFlag, corpusDir: *corpusDir, modelDir: *models, engineDir: *eng,
	}, log)
}
