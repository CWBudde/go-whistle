//go:build oracle

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-whistle/tools/oracle/probe"
)

// emptyJSON is the engine's exact stdout for silence with word timestamps on (needle.h).
const emptyJSON = `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}`

// sweepManifest is testdata/oracle/silence/manifest.json and testdata/oracle/limits/manifest.json.
type sweepManifest struct {
	fixtureHeader

	Generator string            `json:"generator"`
	Input     string            `json:"input"`
	LibCheck  string            `json:"lib_check"`
	EmptyJSON string            `json:"empty_json"`
	Clips     map[string]string `json:"clips"`
	Signals   []sweepRecord     `json:"signals"`
}

// sweepRecord is one signal: its parameters, the digest of its samples and the engine's outcome. Empty is whether
// the CLI's stdout was exactly emptyJSON. In the limits, Transcript is the signal's key in transcripts.json when the
// CLI succeeded; the silence sweep keeps only the text.
type sweepRecord struct {
	probe.Spec

	PCMSHA256  string     `json:"pcm_sha256"`
	CLI        cliResult  `json:"cli"`
	Lib        *libResult `json:"lib,omitempty"`
	Empty      bool       `json:"empty"`
	Text       string     `json:"text"`
	Language   string     `json:"language"`
	Transcript string     `json:"transcript,omitempty"`
}

const (
	sweepInput = "each signal is written as a 32-bit float mono 16 kHz WAV, so the CLI reads the float32 samples " +
		"whose SHA-256 is pcm_sha256 exactly"
	silenceLibCheck = "needle_transcribe (libneedle3, word_timestamps 1) on the same samples returns the same " +
		"JSON, after dropping the timing fields, on every signal; the run fails otherwise"
	limitsLibCheck = "lib is needle_transcribe (libneedle3, word_timestamps 1) on the same samples: its return " +
		"code (tokens generated, or < 0), needle_last_error, whether it returned exactly empty_json and whether its " +
		"JSON equals the CLI's after dropping the timing fields"
)

// cmdSilence records the engine's outcome for every signal of probe.SilenceSet.
func cmdSilence(ctx context.Context, cfg runConfig, log io.Writer) error {
	return cmdSweep(ctx, cfg, log, cmdNameSilence)
}

// cmdLimits records the CLI's and needle_transcribe's outcome for every signal of probe.LimitSet.
func cmdLimits(ctx context.Context, cfg runConfig, log io.Writer) error {
	return cmdSweep(ctx, cfg, log, cmdNameLimits)
}

func cmdSweep(ctx context.Context, cfg runConfig, log io.Writer, name string) error {
	clips, sums, err := loadSweepClips(cfg.audioDir)
	if err != nil {
		return err
	}

	set, gen, check := probe.SilenceSet, "tools/oracle/probe SilenceSet", silenceLibCheck
	if name == cmdNameLimits {
		set, gen, check = probe.LimitSet, "tools/oracle/probe LimitSet", limitsLibCheck
	}

	specs, err := set(clips)
	if err != nil {
		return err
	}

	o, err := openOracle(cfg, true)
	if err != nil {
		return err
	}

	tmp, err := os.MkdirTemp("", "oracle-"+name+"-")
	if err != nil {
		return fmt.Errorf("temp dir: %w", err)
	}

	defer func() { _ = os.RemoveAll(tmp) }()

	man := sweepManifest{
		fixtureHeader: o.header("needle --audio-word-timestamps"), Generator: gen, Input: sweepInput,
		LibCheck: check, EmptyJSON: emptyJSON, Clips: sums,
	}

	tr := bundle{}

	for _, s := range specs {
		rec, js, err := recordSignal(ctx, o, tmp, s, clips[s.Clip], name == cmdNameLimits)
		if err != nil {
			return fmt.Errorf("%s: %w", s.Name, err)
		}

		if name == cmdNameLimits && js != nil {
			rec.Transcript = s.Name
			tr.add(s.Name, js)
		}

		man.Signals = append(man.Signals, rec)

		out := rec.Text
		if rec.CLI.Exit != 0 {
			out = fmt.Sprintf("exit %d: %s", rec.CLI.Exit, rec.CLI.Stderr)
		} else if rec.Empty {
			out = "(empty)"
		}

		_, _ = fmt.Fprintf(log, "%-22s %6d  %s\n", s.Name, s.Samples, out)
	}

	if name == cmdNameLimits {
		err = tr.write(cfg.outDir)
		if err != nil {
			return err
		}
	}

	return writeManifest(cfg.outDir, man)
}

// recordSignal runs one signal through the CLI and needle_transcribe and returns the CLI's normalised JSON when it
// succeeded. Silence-sweep signals must succeed on both and agree; limit signals keep both outcomes as they are.
func recordSignal(ctx context.Context, o *oracle, tmp string, s probe.Spec, clip []float32, limits bool,
) (sweepRecord, []byte, error) {
	var pcm []float32

	if s.Samples > 0 {
		var err error

		pcm, err = s.SignalWith(clip)
		if err != nil {
			return sweepRecord{}, nil, fmt.Errorf("synthesise: %w", err)
		}
	}

	rec := sweepRecord{Spec: s, PCMSHA256: probe.SHA256(pcm)}

	wav, err := writeTempWAV(tmp, s.Name, pcm)
	if err != nil {
		return rec, nil, err
	}

	rec.CLI, err = o.runIn(ctx, tmp, wav)
	if err != nil {
		return rec, nil, err
	}

	lib := o.eng.transcribePCM(pcm)

	if limits {
		lib.Empty = lib.RC >= 0 && string(bytes.TrimSpace(lib.out)) == emptyJSON
		rec.Lib = &lib
	}

	if rec.CLI.Exit != 0 {
		if !limits {
			return rec, nil, fmt.Errorf("needle: exit %d: %s", rec.CLI.Exit, rec.CLI.Stderr)
		}

		return rec, nil, nil
	}

	rec.Empty = string(bytes.TrimSpace(rec.CLI.stdout)) == emptyJSON

	js, err := normaliseJSON(rec.CLI.stdout)
	if err != nil {
		return rec, nil, err
	}

	err = sameAsLib(js, lib)
	if err != nil && !limits {
		return rec, nil, err
	}

	if limits {
		lib.SameAsCLI = err == nil
	}

	t, err := parseTranscript(js)
	if err != nil {
		return rec, nil, err
	}

	rec.Text, rec.Language = t.Text, t.Language

	return rec, js, nil
}

// sameAsLib requires needle_transcribe's JSON to equal the CLI's normalised JSON.
func sameAsLib(cli []byte, lib libResult) error {
	if lib.RC < 0 {
		return fmt.Errorf("needle_transcribe: %s (%d), the CLI succeeded", lib.Error, lib.RC)
	}

	js, err := normaliseJSON(lib.out)
	if err != nil {
		return fmt.Errorf("needle_transcribe: %w", err)
	}

	if !bytes.Equal(js, cli) {
		return fmt.Errorf("needle_transcribe and the CLI differ:\n%s\n%s", js, cli)
	}

	return nil
}

// loadSweepClips reads probe.SweepClips from the corpus, checked against its manifest, as int16/32768 samples, and
// returns them with their WAV digests.
func loadSweepClips(dir string) (map[string][]float32, map[string]string, error) {
	corpus, err := readCorpus(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	byID := map[string]clip{}
	for _, c := range corpus.Clips {
		byID[c.ID] = c
	}

	pcm, sums := map[string][]float32{}, map[string]string{}

	for _, id := range probe.SweepClips() {
		c, ok := byID[id]
		if !ok {
			return nil, nil, fmt.Errorf("clip %s is not in the corpus", id)
		}

		raw, err := readClip(filepath.Join(dir, id+".wav"), c.SHA256)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", id, err)
		}

		x, rate, err := readWAV(raw)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", id, err)
		}

		if rate != sampleRate {
			return nil, nil, fmt.Errorf("%s: %d Hz, want %d", id, rate, sampleRate)
		}

		pcm[id], sums[id] = x, c.SHA256
	}

	return pcm, sums, nil
}
