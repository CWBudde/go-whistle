//go:build oracle

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cwbudde/go-whistle/internal/hub"
)

// config is what every subcommand reads and writes.
type config struct {
	outDir     string // testdata/text/engine
	casesPath  string // testdata/text/engine/cases.json
	needle3Dir string // models/needle3, holding needle3.cact
	whistleDir string // models/whistle, holding whistle.cact
	engineDir  string // engine/<platform>
	audioDir   string // testdata/audio, the clip corpus
}

// modelPin is a checked .cact: which hub revision it comes from and its digest.
type modelPin struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	File     string `json:"file"`
	SHA256   string `json:"sha256"`
}

// header opens every fixture file: the engine and models that answered and how the CLI was run.
type header struct {
	Engine  map[string]string `json:"engine"`
	Models  []modelPin        `json:"models"`
	Command string            `json:"command"`
}

// oracle is the pinned engine, the corpus and the scratch files the CLI reads (tool sets and system texts).
type oracle struct {
	cfg     config
	corpus  corpus
	pin     map[string]string
	needle3 modelPin
	whistle *modelPin // set by the subcommands that answer speech
	scratch string
}

// openOracle checks the engine and model pins, reads the corpus, creates the output directory and writes every tool
// set and system text into a scratch directory. close removes it.
func openOracle(cfg config, speech bool) (*oracle, error) {
	pin, err := readPin(filepath.Join(cfg.engineDir, "PIN"))
	if err != nil {
		return nil, fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	needle3, err := checkModel(hub.PinnedNeedle3, cfg.needle3Dir, "needle3.cact")
	if err != nil {
		return nil, err
	}

	c, err := readCorpus(cfg.casesPath)
	if err != nil {
		return nil, err
	}

	o := &oracle{cfg: cfg, corpus: c, pin: pin, needle3: needle3}

	if speech {
		w, err := checkModel(hub.Pinned, cfg.whistleDir, "whistle.cact")
		if err != nil {
			return nil, err
		}

		o.whistle = &w
	}

	err = os.MkdirAll(cfg.outDir, 0o750)
	if err != nil {
		return nil, fmt.Errorf("create %s: %w", cfg.outDir, err)
	}

	o.scratch, err = os.MkdirTemp("", "oracle-text-")
	if err != nil {
		return nil, fmt.Errorf("scratch dir: %w", err)
	}

	for _, name := range c.toolsetNames() {
		err := writeFile(o.toolsPath(name), []byte(c.tools(name)))
		if err != nil {
			o.close()

			return nil, err
		}
	}

	return o, nil
}

func (o *oracle) close() { _ = os.RemoveAll(o.scratch) }

func (o *oracle) needle() string      { return filepath.Join(o.cfg.engineDir, "needle") }
func (o *oracle) needle3Path() string { return filepath.Join(o.cfg.needle3Dir, o.needle3.File) }

func (o *oracle) toolsPath(name string) string {
	return filepath.Join(o.scratch, "tools-"+name+".json")
}

// systemFile writes a system text to the scratch directory, byte for byte (no trailing newline), and returns its
// path: `needle --system` takes a file, not the text.
func (o *oracle) systemFile(text string) (string, error) {
	sum := sha256.Sum256([]byte(text))
	p := filepath.Join(o.scratch, "system-"+hex.EncodeToString(sum[:8])+".txt")

	return p, writeFile(p, []byte(text))
}

// header is the fixture header for a command line that runs the text model alone.
func (o *oracle) header(command string) header {
	return header{Engine: o.pin, Models: []modelPin{o.needle3}, Command: command}
}

// speechHeader is the fixture header for a command line that runs the text and the speech model.
func (o *oracle) speechHeader(command string) header {
	h := o.header(command)
	if o.whistle != nil {
		h.Models = append(h.Models, *o.whistle)
	}

	return h
}

// flagModel is the CLI flag naming a .cact to load.
const flagModel = "--model"

// flagTools is the CLI flag naming the tools file.
const flagTools = "--tools"

// flagSystem is the CLI flag naming the file that holds the system text.
const flagSystem = "--system"

// cliThreads pins the engine's worker threads. Its default is the device's fast cores, at most 4, and unlike
// Whistle's transcripts (docs/FINDINGS.md F-025) the text model's answers depend on it: the confidence moves in the
// third decimal and now and then the reasoning text changes (manifest.json "threads"). 4 is the default on the
// machines with four or more fast cores, which libneedle3 (it has no thread setting) then matches.
const cliThreads = "4"

// baseArgs are the CLI flags that select the text model, a tool set, a system text and the pinned thread count.
func (o *oracle) baseArgs(toolset, system string) ([]string, error) {
	return o.threadArgs(toolset, system, cliThreads)
}

// threadArgs is baseArgs with the given --threads, or none (the engine's default) when threads is empty.
func (o *oracle) threadArgs(toolset, system, threads string) ([]string, error) {
	sys, err := o.systemFile(system)
	if err != nil {
		return nil, err
	}

	args := []string{flagModel, o.needle3Path(), flagTools, o.toolsPath(toolset), flagSystem, sys}
	if threads != "" {
		args = append(args, "--threads", threads)
	}

	return args, nil
}

// checkModel checks dir/local against a pinned hub manifest.
func checkModel(pinned func() (hub.Manifest, error), dir, local string) (modelPin, error) {
	m, err := pinned()
	if err != nil {
		return modelPin{}, fmt.Errorf("pinned manifest: %w", err)
	}

	for _, f := range m.Files {
		if f.Local != local {
			continue
		}

		sum, err := fileSHA256(filepath.Join(dir, f.Local))
		if err != nil {
			return modelPin{}, fmt.Errorf("%w (run `whistle download`)", err)
		}

		if sum != f.SHA256 {
			return modelPin{}, fmt.Errorf("%s: sha256 %s, pinned %s", f.Local, sum, f.SHA256)
		}

		return modelPin{Repo: m.Repo, Revision: m.Revision, File: f.Local, SHA256: sum}, nil
	}

	return modelPin{}, fmt.Errorf("the pinned %s manifest has no %s", m.Repo, local)
}

func fileSHA256(p string) (string, error) {
	f, err := os.Open(p) // #nosec G304 -- the model dirs are flags.
	if err != nil {
		return "", fmt.Errorf("open %s: %w", p, err)
	}

	defer func() { _ = f.Close() }()

	h := sha256.New()

	_, err = io.Copy(h, f)
	if err != nil {
		return "", fmt.Errorf("hash %s: %w", p, err)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// readPin parses engine/<platform>/PIN (key=value lines, written by scripts/fetch-engine.sh).
func readPin(p string) (map[string]string, error) {
	f, err := os.Open(p) // #nosec G304 -- the engine dir is a flag.
	if err != nil {
		return nil, fmt.Errorf("engine pin: %w", err)
	}

	defer func() { _ = f.Close() }()

	out := map[string]string{}
	sc := bufio.NewScanner(f)

	for sc.Scan() {
		k, v, ok := strings.Cut(strings.TrimSpace(sc.Text()), "=")
		if ok {
			out[k] = v
		}
	}

	err = sc.Err()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", p, err)
	}

	return out, nil
}

// writeJSON writes v to dir/name with prettyJSON's layout.
func writeJSON(dir, name string, v any) error {
	js, err := prettyJSON(v)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}

	return writeFile(filepath.Join(dir, name), js)
}

func writeFile(p string, b []byte) error {
	err := os.WriteFile(p, b, 0o600)
	if err != nil {
		return fmt.Errorf("write %s: %w", p, err)
	}

	return nil
}

// equalNodes reports whether two answers encode identically.
func equalNodes(a, b node) bool { return string(compact(a)) == string(compact(b)) }

// itoa is strconv.Itoa, short for the flag lists.
func itoa(i int) string { return strconv.Itoa(i) }
