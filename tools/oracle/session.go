//go:build oracle

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// fixtureHeader opens every decoding-oracle manifest: what produced the fixtures and how the CLI was run.
type fixtureHeader struct {
	Engine  map[string]string `json:"engine"`
	Model   oracleModel       `json:"model"`
	Command string            `json:"command"`
	Threads string            `json:"threads"`
}

// oracle is the pinned engine and model the decoding-oracle subcommands run: the needle CLI always, libneedle3 when
// the subcommand cross-checks needle_transcribe.
type oracle struct {
	cfg   runConfig
	model oracleModel
	pin   map[string]string
	eng   *engine
}

// openOracle checks the model and engine pins, sets the telemetry opt-outs for libneedle3 (the CLI gets them per
// run), creates the output directory and, with lib set, loads libneedle3.
func openOracle(cfg runConfig, lib bool) (*oracle, error) {
	for k, v := range map[string]string{"NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1"} {
		err := os.Setenv(k, v)
		if err != nil {
			return nil, fmt.Errorf("setenv %s: %w", k, err)
		}
	}

	model, err := pinnedModel(cfg.modelDir)
	if err != nil {
		return nil, err
	}

	pin, err := readPin(filepath.Join(cfg.engineDir, "PIN"))
	if err != nil {
		return nil, fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	o := &oracle{cfg: cfg, model: model, pin: pin}

	if lib {
		cact, err := os.ReadFile(o.modelPath()) // #nosec G304 -- modelDir is a flag.
		if err != nil {
			return nil, fmt.Errorf("read model: %w", err)
		}

		o.eng, err = openEngine(cfg.engineDir, cact)
		if err != nil {
			return nil, err
		}
	}

	err = os.MkdirAll(cfg.outDir, 0o750)
	if err != nil {
		return nil, fmt.Errorf("create %s: %w", cfg.outDir, err)
	}

	return o, nil
}

func (o *oracle) needle() string    { return filepath.Join(o.cfg.engineDir, "needle") }
func (o *oracle) modelPath() string { return filepath.Join(o.cfg.modelDir, o.model.File) }

func (o *oracle) header(command string) fixtureHeader {
	return fixtureHeader{Engine: o.pin, Model: o.model, Command: command, Threads: cliThreads}
}

// cli transcribes wav with the needle CLI (word timestamps on, threads pinned) and the extra flags.
func (o *oracle) cli(ctx context.Context, wav string, extra ...string) ([]byte, error) {
	return transcribe(ctx, o.needle(), o.modelPath(), wav, extra...)
}

// bundle collects a fixture set's normalised engine JSONs by name; it is written as one transcripts.json, an object
// whose values are the objects a per-clip <name>.json would hold. One file instead of hundreds keeps the repository
// small: the file system rounds every file up to a 4 KiB block (docs/FINDINGS.md F-025).
type bundle map[string]json.RawMessage

func (b bundle) add(name string, js []byte) { b[name] = js }

// write writes the bundle to dir/transcripts.json, keys sorted, indented like the per-clip fixtures.
func (b bundle) write(dir string) error {
	js, err := json.MarshalIndent(map[string]json.RawMessage(b), "", "  ")
	if err != nil {
		return fmt.Errorf("transcripts: %w", err)
	}

	return writeFile(filepath.Join(dir, transcriptsFile), append(js, '\n'))
}

// transcriptsFile is the bundle's file name in every fixture set.
const transcriptsFile = "transcripts.json"

// writeManifest writes v as indented JSON to dir/manifest.json.
func writeManifest(dir string, v any) error {
	js, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("manifest: %w", err)
	}

	return writeFile(filepath.Join(dir, "manifest.json"), append(js, '\n'))
}

// runIn runs the needle CLI in dir on the WAV dir/name, so that the engine's messages name the file only by name.
func (o *oracle) runIn(ctx context.Context, dir, name string) (cliResult, error) {
	needle, err := filepath.Abs(o.needle())
	if err != nil {
		return cliResult{}, fmt.Errorf("engine path: %w", err)
	}

	model, err := filepath.Abs(o.modelPath())
	if err != nil {
		return cliResult{}, fmt.Errorf("model path: %w", err)
	}

	return runNeedle(ctx, dir, needle, model, name)
}

// writeTempWAV writes pcm as a float WAV dir/<name>.wav and returns its file name.
func writeTempWAV(dir, name string, pcm []float32) (string, error) {
	var buf bytes.Buffer

	err := writeWAVFloat(&buf, pcm)
	if err != nil {
		return "", err
	}

	return name + ".wav", writeFile(filepath.Join(dir, name+".wav"), buf.Bytes())
}

// transcript is the part of the engine's JSON the manifests summarise.
type transcript struct {
	Text     string `json:"text"`
	Language string `json:"language"`
}

func parseTranscript(js []byte) (transcript, error) {
	var t transcript

	err := json.Unmarshal(js, &t)
	if err != nil {
		return t, fmt.Errorf("transcript: %w", err)
	}

	return t, nil
}

// readFixture returns a committed engine JSON.
func readFixture(p string) ([]byte, error) {
	b, err := os.ReadFile(p) // #nosec G304 -- fixture paths are built from checked clip ids.
	if err != nil {
		return nil, fmt.Errorf("read fixture: %w", err)
	}

	return b, nil
}
