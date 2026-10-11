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

// kwVariant is one keyword file on one clip of the keyword-variant survey.
type kwVariant struct {
	Clip     string
	Keywords []string
}

// kwVariants vary how a keyword is written against the words of keyword cases the engine already answers
// (docs/FINDINGS.md F-029, F-042, F-044): the whole keyword in capitals (does the engine fold case?), a piece from
// inside a word (does a match need a word start?), the start of a word (does it need a word end?) and short phrases
// that our search repeats (loops).
var kwVariants = []kwVariant{
	// Case.
	{"ls-other-3080-5040-0011", []string{"COWLEY'S"}},
	{"ls-other-3080-5040-0011", []string{"cOWLEY'S"}},
	{"fleurs-de_de-10308608197319017935", []string{"ADOPTIONSPLAN"}},
	{"fleurs-de_de-10308608197319017935", []string{"adoptionsplan"}},
	{"fleurs-nl_nl-1004717084892689357", []string{"VARKENS"}},
	{"fleurs-nl_nl-1004717084892689357", []string{"varkens"}},
	{"fleurs-nl_nl-1004717084892689357", []string{"MUGGEN"}},
	{"fleurs-es_419-10236991702517253859", []string{"ALIVIAR"}},
	{"fleurs-es_419-10236991702517253859", []string{"Aliviar"}},
	{"fleurs-it_it-10210638754804006465", []string{"ACCOMUNANO"}},
	{"fleurs-fr_fr-10385751433252343757", []string{"CRISTAUX"}},
	{"fleurs-pl_pl-10013271501230525766", []string{"WYSP"}},
	{"ls-clean-1320-122617-0012", []string{"UNCAS"}},
	// A piece from inside a word.
	{"fleurs-de_de-10308608197319017935", []string{"aare"}},
	{"fleurs-de_de-10308608197319017935", []string{"optionsplan"}},
	{"fleurs-de_de-10308608197319017935", []string{"doptionsplan"}},
	{"ls-other-3080-5040-0011", []string{"owley's"}},
	{"fleurs-nl_nl-1004717084892689357", []string{"arkens"}},
	{"fleurs-it_it-10210638754804006465", []string{"comunano"}},
	// The start of a word.
	{"fleurs-de_de-10308608197319017935", []string{"Adoptions"}},
	{"fleurs-de_de-10308608197319017935", []string{"Adop"}},
	{"fleurs-de_de-10308608197319017935", []string{"Paa"}},
	{"ls-other-3080-5040-0011", []string{"Cowley"}},
	{"fleurs-nl_nl-1004717084892689357", []string{"Varken"}},
	{"fleurs-es_419-10236991702517253859", []string{"alivi"}},
	{"fleurs-it_it-10210638754804006465", []string{"accomun"}},
	// Short phrases.
	{"fleurs-de_de-10308608197319017935", []string{"können sich"}},
	{"fleurs-de_de-10308608197319017935", []string{"für einen"}},
	{"fleurs-de_de-10308608197319017935", []string{"Paare können sich für"}},
	{"ls-other-3080-5040-0011", []string{"some verses"}},
	{"fleurs-es_419-10236991702517253859", []string{"el dolor"}},
}

// kwVariantsManifest is testdata/oracle/kwvariants/manifest.json.
type kwVariantsManifest struct {
	fixtureHeader

	Sets []kwVariantRecord `json:"sets"`
}

// kwVariantRecord is one keyword file on one clip and the engine's text and language with it.
type kwVariantRecord struct {
	Clip     string   `json:"clip"`
	Keywords []string `json:"keywords"`
	Language string   `json:"language"`
	Text     string   `json:"text"`
}

// cmdKWVariants records the engine's text for every entry of kwVariants into manifest.json.
func cmdKWVariants(ctx context.Context, cfg runConfig, log io.Writer) error {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	sums := map[string]string{}
	for _, c := range corpus.Clips {
		sums[c.ID] = c.SHA256
	}

	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}

	tmp, err := os.MkdirTemp("", "kwvariants")
	if err != nil {
		return fmt.Errorf("keyword variants: %w", err)
	}

	defer func() { _ = os.RemoveAll(tmp) }()

	man := kwVariantsManifest{fixtureHeader: o.header("needle --audio-word-timestamps --audio-keywords <set>")}
	file := filepath.Join(tmp, "keywords.txt")

	for _, v := range kwVariants {
		sum, ok := sums[v.Clip]
		if !ok {
			return fmt.Errorf("keyword variants: clip %s not in the corpus", v.Clip)
		}

		wav := filepath.Join(cfg.audioDir, v.Clip+".wav")

		_, err = readClip(wav, sum)
		if err != nil {
			return fmt.Errorf("%s: %w", v.Clip, err)
		}

		err = writeFile(file, []byte(strings.Join(v.Keywords, "\n")+"\n"))
		if err != nil {
			return err
		}

		js, err := o.cli(ctx, wav, "--audio-keywords", file)
		if err != nil {
			return fmt.Errorf("keyword variant %s %q: %w", v.Clip, v.Keywords, err)
		}

		t, err := parseTranscript(js)
		if err != nil {
			return err
		}

		man.Sets = append(man.Sets, kwVariantRecord{Clip: v.Clip, Keywords: v.Keywords, Language: t.Language, Text: t.Text})
		_, _ = fmt.Fprintf(log, "%-36s %-28q %s\n", v.Clip, v.Keywords, t.Text)
	}

	return writeManifest(cfg.outDir, man)
}
