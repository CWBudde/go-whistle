//go:build oracle

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// kwSetClip is the clip of the keyword-set survey: the German clip of the one keyword search miss, de-paare-missed
// (docs/FINDINGS.md F-029, F-042).
const kwSetClip = "fleurs-de_de-10308608197319017935"

// kwSets are keyword files run on kwSetClip, one keyword per entry. They vary the shared bonus the hypotheses carry
// (single words against phrases, a keyword already in the text, repeated and differently cased keywords), so that a
// bias model must reproduce where the engine writes "Paare" or "Bare" and "Adduktionsplan" or "Adoptionsplan".
var kwSets = [][]string{
	{"Paare"},
	{"Diese Paare"},
	{"Baby"},
	{"können"},
	{"Diese"},
	{"Bare"},
	{"Diese Bare"},
	{"Paare", "Baby"},
	{"Paare", "entscheiden"},
	{"Paare", "können"},
	{"Diese", "Paare"},
	{"Paare", "Diese"},
	{"Paare", "Paare"},
	{"Paare", "Bare"},
	{"Bare", "Paare"},
	{"Diese Paare können"},
	{"Paare können"},
	{"Paare können sich"},
	{"Diese Paare", "Baby"},
	{"diese Paare"},
	{"paare"},
	{"PAARE"},
	{"Pare"},
	{"Paar"},
	{"Adduktionsplan"},
	{"Paare", "Adduktionsplan"},
	{"Diese Paare", "Adduktionsplan"},
	{"einen"},
	{"für ihr Baby"},
	{"Paare", "für ihr Baby"},
	{"Baby entscheiden"},
	{"Paare", "Baby entscheiden"},
}

// kwSetsManifest is testdata/oracle/kwsets/manifest.json.
type kwSetsManifest struct {
	fixtureHeader

	Clip string        `json:"clip"`
	Sets []kwSetRecord `json:"sets"`
}

// kwSetRecord is one keyword file and the engine's text and language with it.
type kwSetRecord struct {
	Keywords []string `json:"keywords"`
	Language string   `json:"language"`
	Text     string   `json:"text"`
}

// cmdKWSets records the engine's text on kwSetClip with every keyword file of kwSets into manifest.json.
func cmdKWSets(ctx context.Context, cfg runConfig, log io.Writer) error {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	var c clip

	for _, cc := range corpus.Clips {
		if cc.ID == kwSetClip {
			c = cc
		}
	}

	if c.ID == "" {
		return fmt.Errorf("keyword sets: clip %s not in the corpus", kwSetClip)
	}

	wav := filepath.Join(cfg.audioDir, c.ID+".wav")

	_, err = readClip(wav, c.SHA256)
	if err != nil {
		return fmt.Errorf("%s: %w", c.ID, err)
	}

	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}

	tmp, err := os.MkdirTemp("", "kwsets")
	if err != nil {
		return fmt.Errorf("keyword sets: %w", err)
	}

	defer func() { _ = os.RemoveAll(tmp) }()

	man := kwSetsManifest{
		fixtureHeader: o.header("needle --audio-word-timestamps --audio-keywords <set>"), Clip: c.ID,
	}
	file := filepath.Join(tmp, "keywords.txt")

	for _, set := range kwSets {
		err = writeFile(file, []byte(strings.Join(set, "\n")+"\n"))
		if err != nil {
			return err
		}

		js, err := o.cli(ctx, wav, "--audio-keywords", file)
		if err != nil {
			return fmt.Errorf("keyword set %q: %w", set, err)
		}

		t, err := parseTranscript(js)
		if err != nil {
			return err
		}

		man.Sets = append(man.Sets, kwSetRecord{Keywords: set, Language: t.Language, Text: t.Text})
		_, _ = fmt.Fprintf(log, "%-40q %s\n", set, t.Text)
	}

	return writeManifest(cfg.outDir, man)
}
