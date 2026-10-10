//go:build oracle

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-whistle/tools/oracle/probe"
)

// probeManifest is testdata/oracle/probes/manifest.json: what produced the probe rows and how to regenerate the
// signals (docs/FINDINGS.md F-019, F-020).
type probeManifest struct {
	Engine     map[string]string `json:"engine"`
	Model      oracleModel       `json:"model"`
	EmbedWidth int               `json:"embed_width"`
	Generator  string            `json:"generator"`
	Probes     []probeRecord     `json:"probes"`
}

// probeRecord is one probe's parameters, its frame counts and the digest of the samples the engine embedded.
type probeRecord struct {
	probe.Spec

	Frames         int    `json:"frames"`
	InteriorFrames int    `json:"interior_frames"`
	PCMSHA256      string `json:"pcm_sha256"`
}

// cmdProbes records needle_embed for every signal of probe.Set into cfg.outDir.
func cmdProbes(cfg runConfig, log io.Writer) error {
	// The engine's telemetry is off for the oracle (the justfile recipe sets the same).
	for k, v := range map[string]string{"NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1"} {
		err := os.Setenv(k, v)
		if err != nil {
			return fmt.Errorf("setenv %s: %w", k, err)
		}
	}

	model, err := pinnedModel(cfg.modelDir)
	if err != nil {
		return err
	}

	pin, err := readPin(filepath.Join(cfg.engineDir, "PIN"))
	if err != nil {
		return fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	cact, err := os.ReadFile(filepath.Join(cfg.modelDir, model.File)) // #nosec G304 -- modelDir is a flag.
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}

	eng, err := openEngine(cfg.engineDir, cact)
	if err != nil {
		return err
	}

	err = os.MkdirAll(cfg.outDir, 0o750)
	if err != nil {
		return fmt.Errorf("create %s: %w", cfg.outDir, err)
	}

	man := probeManifest{
		Engine: pin, Model: model, EmbedWidth: embedWidth,
		Generator: "tools/oracle/probe Set (go run -tags oracle ./tools/oracle probes)",
	}

	for _, s := range probe.Set() {
		rec, err := recordProbe(cfg.outDir, eng, s, map[string]string{
			"engine_version": pin["engine_version"], "engine_revision": pin["revision"], "model_sha256": model.SHA256,
		})
		if err != nil {
			return fmt.Errorf("%s: %w", s.Name, err)
		}

		man.Probes = append(man.Probes, rec)
		_, _ = fmt.Fprintf(log, "%-22s %6d samples %3d frames (%3d interior)\n", s.Name, s.Samples, rec.Frames,
			rec.InteriorFrames)
	}

	js, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return fmt.Errorf("probe manifest: %w", err)
	}

	return writeFile(filepath.Join(cfg.outDir, "manifest.json"), append(js, '\n'))
}

// recordProbe embeds one probe and writes its rows, the digest of its samples in the metadata.
func recordProbe(dir string, eng *engine, s probe.Spec, meta map[string]string) (probeRecord, error) {
	pcm, err := s.Signal()
	if err != nil {
		return probeRecord{}, fmt.Errorf("synthesise: %w", err)
	}

	emb, err := eng.embedSpeech(pcm)
	if err != nil {
		return probeRecord{}, err
	}

	frames := len(emb) / embedWidth
	if len(emb)%embedWidth != 0 || frames != probe.Frames(s.Samples) {
		return probeRecord{}, fmt.Errorf("needle_embed returned %d floats, want %d frames of %d", len(emb),
			probe.Frames(s.Samples), embedWidth)
	}

	rec := probeRecord{
		Spec: s, Frames: frames, InteriorFrames: probe.InteriorFrames(s.Samples), PCMSHA256: probe.SHA256(pcm),
	}
	meta["probe"], meta["pcm_sha256"] = s.Name, rec.PCMSHA256

	var buf bytes.Buffer

	err = writeSafetensors(&buf, "embed", []int{frames, embedWidth}, emb, meta)
	if err != nil {
		return probeRecord{}, err
	}

	return rec, writeFile(filepath.Join(dir, s.Name+".embed.safetensors"), buf.Bytes())
}
