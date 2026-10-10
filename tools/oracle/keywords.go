//go:build oracle

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"unicode"
)

// Keyword case kinds (PLAN.md 6.1).
const (
	kwPresent    = "present"    // a reference word the engine already gets right
	kwMissed     = "missed"     // a reference word the engine's text lacks
	kwDistractor = "distractor" // a wrong but plausible word, in neither the reference nor the engine's text
	kwPhrase     = "phrase"     // a multi-word phrase of the reference
	kwList       = "list"       // several reference words and phrases in one file
)

// keywordCase is one keyword file run against one corpus clip.
type keywordCase struct {
	name     string
	clip     string
	kind     string
	keywords []string
}

// kw is one keyword case of a keywordClip.
type kw struct {
	name     string
	kind     string
	keywords []string
}

// keywordClip groups the cases of one corpus clip.
type keywordClip struct {
	clip  string
	cases []kw
}

// keywordClips are derived from the 32 clips' reference transcripts and the engine's text without keywords
// (testdata/audio/manifest.json, testdata/oracle/<clip>.json); cmdKeywords checks every kind against both.
var keywordClips = []keywordClip{
	{"ls-other-3080-5040-0011", []kw{
		{"en-cowleys-missed", kwMissed, []string{"Cowley's"}},
		{"en-cowleys-lower-missed", kwMissed, []string{"cowley's"}},
		{"en-verses-present", kwPresent, []string{"verses"}},
		{"en-curses-distractor", kwDistractor, []string{"curses"}},
		{"en-verses-of-cowleys-phrase", kwPhrase, []string{"verses of Cowley's"}},
	}},
	{"ls-other-7902-96591-0023", []kw{
		{"en-chance-present", kwPresent, []string{"chance"}},
		{"en-transfer-distractor", kwDistractor, []string{"transfer"}},
		{"en-no-chance-phrase", kwPhrase, []string{"no chance for his body"}},
	}},
	{"ls-clean-5639-40744-0011", []kw{
		{"en-the-door-phrase", kwPhrase, []string{"the door"}},
		{"en-looked-distractor", kwDistractor, []string{"looked"}},
	}},
	{"ls-other-4198-12281-0004", []kw{
		{"en-sanct-missed", kwMissed, []string{"Sanct"}},
		{"en-sanct-james-phrase", kwPhrase, []string{"Sanct James"}},
		{"en-devils-present", kwPresent, []string{"devils"}},
	}},
	{"ls-other-4198-12281-0014", []kw{{"en-did-it-phrase", kwPhrase, []string{"did it"}}}},
	{"ls-clean-1320-122617-0003", []kw{
		{"en-scout-present", kwPresent, []string{"scout"}},
		{"en-stout-distractor", kwDistractor, []string{"stout"}},
	}},
	{"ls-clean-1320-122617-0012", []kw{{"en-uncas-present", kwPresent, []string{"Uncas"}}}},
	{"fleurs-de_de-10308608197319017935", []kw{
		{"de-paare-missed", kwMissed, []string{"Paare"}},
		{"de-adoptionsplan-missed", kwMissed, []string{"Adoptionsplan"}},
		{"de-adaptionsplan-distractor", kwDistractor, []string{"Adaptionsplan"}},
		{"de-baby-present", kwPresent, []string{"Baby"}},
		{"de-diese-paare-phrase", kwPhrase, []string{"Diese Paare"}},
		{"de-paare-adoptionsplan-list", kwList, []string{"Paare", "Adoptionsplan"}},
	}},
	{"fleurs-de_de-10229344228128634115", []kw{
		{"de-fast-jeder-phrase", kwPhrase, []string{"Fast jeder"}},
		{"de-gesellschaft-present", kwPresent, []string{"Gesellschaft"}},
	}},
	{"fleurs-fr_fr-10174413140250646001", []kw{
		{"fr-fisichella-missed", kwMissed, []string{"Fisichella"}},
		{"fr-giancarlo-fisichella-phrase", kwPhrase, []string{"Giancarlo Fisichella"}},
		{"fr-voiture-present", kwPresent, []string{"voiture"}},
	}},
	{"fleurs-fr_fr-10385751433252343757", []kw{
		{"fr-cristaux-missed", kwMissed, []string{"cristaux"}},
		{"fr-renale-missed", kwMissed, []string{"rénale"}},
		{"fr-cristal-distractor", kwDistractor, []string{"cristal"}},
		{"fr-cristaux-renale-declare-list", kwList, []string{"cristaux", "rénale", "ont déclaré"}},
	}},
	{"fleurs-es_419-10236991702517253859", []kw{
		{"es-aliviar-missed", kwMissed, []string{"aliviar"}},
		{"es-aliviar-el-dolor-phrase", kwPhrase, []string{"aliviar el dolor"}},
		{"es-medicamentos-present", kwPresent, []string{"medicamentos"}},
	}},
	{"fleurs-es_419-10149016836336603986", []kw{
		{"es-homenaje-present", kwPresent, []string{"homenaje"}},
		{"es-luchadoras-distractor", kwDistractor, []string{"luchadoras"}},
	}},
	{"fleurs-it_it-10210638754804006465", []kw{
		{"it-accomunano-missed", kwMissed, []string{"accomunano"}},
		{"it-li-accomunano-phrase", kwPhrase, []string{"li accomunano"}},
		{"it-uccelli-present", kwPresent, []string{"uccelli"}},
	}},
	{"fleurs-it_it-1006729518827641482", []kw{
		{"it-montagna-present", kwPresent, []string{"montagna"}},
		{"it-montagne-distractor", kwDistractor, []string{"montagne"}},
	}},
	{"fleurs-nl_nl-1004717084892689357", []kw{
		{"nl-varkens-missed", kwMissed, []string{"Varkens"}},
		{"nl-muggen-missed", kwMissed, []string{"muggen"}},
		{"nl-dragen-de-ziekte-phrase", kwPhrase, []string{"dragen de ziekte"}},
	}},
	{"fleurs-nl_nl-10042455874761876908", []kw{
		{"nl-continuummodel-missed", kwMissed, []string{"continuümmodel"}},
		{"nl-organisaties-present", kwPresent, []string{"organisaties"}},
	}},
	{"fleurs-pl_pl-10047713455444904767", []kw{
		{"pl-gubernatora-missed", kwMissed, []string{"gubernatora"}},
		{"pl-dziewietnastu-policjantow-phrase", kwPhrase, []string{"dziewiętnastu policjantów"}},
		{"pl-zgodnie-present", kwPresent, []string{"Zgodnie"}},
	}},
	{"fleurs-pl_pl-10013271501230525766", []kw{
		{"pl-wysp-missed", kwMissed, []string{"wysp"}},
		{"pl-luksusowe-missed", kwMissed, []string{"luksusowe"}},
		{"pl-francja-present", kwPresent, []string{"Francją"}},
	}},
}

// keywordCases flattens keywordClips in order.
func keywordCases() []keywordCase {
	var out []keywordCase

	for _, kc := range keywordClips {
		for _, c := range kc.cases {
			out = append(out, keywordCase{name: c.name, clip: kc.clip, kind: c.kind, keywords: c.keywords})
		}
	}

	return out
}

// keywordsManifest is testdata/oracle/keywords/manifest.json.
type keywordsManifest struct {
	fixtureHeader

	Match       string          `json:"match"`
	Transcripts string          `json:"transcripts"`
	Cases       []keywordRecord `json:"cases"`
}

// keywordRecord is one case: the keyword file, the engine JSON without it (the committed full-depth fixture, which
// the run re-checks byte for byte) and with it (a key of transcripts.json), and how many keywords each text and the
// reference contain.
type keywordRecord struct {
	Name          string   `json:"name"`
	Clip          string   `json:"clip"`
	Language      string   `json:"language"`
	Kind          string   `json:"kind"`
	Keywords      []string `json:"keywords"`
	KeywordsFile  string   `json:"keywords_file"`
	Without       string   `json:"without"`
	With          string   `json:"with"`
	Reference     string   `json:"reference"`
	TextWithout   string   `json:"text_without"`
	TextWith      string   `json:"text_with"`
	Changed       bool     `json:"changed"`
	HitsReference int      `json:"hits_reference"`
	HitsWithout   int      `json:"hits_without"`
	HitsWith      int      `json:"hits_with"`
}

// matchRule documents the hits_* counts.
const matchRule = "a keyword hits a text when its words occur consecutively in it, after lower-casing and turning " +
	"every character other than a letter, digit or apostrophe into a word break"

// cmdKeywords records, per keyword case, the keyword file <case>.txt and the engine's JSON with it (<case> in
// transcripts.json).
func cmdKeywords(ctx context.Context, cfg runConfig, log io.Writer) error {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	clips := map[string]clip{}
	for _, c := range corpus.Clips {
		clips[c.ID] = c
	}

	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}

	man := keywordsManifest{
		fixtureHeader: o.header("needle --audio-word-timestamps [--audio-keywords <case>.txt]"), Match: matchRule,
		Transcripts: transcriptsFile + ": <case>, the run with the keyword file",
	}
	without := map[string]transcript{}
	tr := bundle{}

	for _, kc := range keywordCases() {
		c, ok := clips[kc.clip]
		if !ok || !validID(kc.name) {
			return fmt.Errorf("keyword case %s: unknown clip %q or invalid name", kc.name, kc.clip)
		}

		wav := filepath.Join(cfg.audioDir, c.ID+".wav")

		if _, seen := without[c.ID]; !seen {
			_, err := readClip(wav, c.SHA256)
			if err != nil {
				return fmt.Errorf("%s: %w", c.ID, err)
			}

			without[c.ID], err = checkWithout(ctx, o, c.ID, wav)
			if err != nil {
				return err
			}
		}

		rec, err := recordKeywordCase(ctx, o, kc, c, wav, without[c.ID], tr)
		if err != nil {
			return fmt.Errorf("keyword case %s: %w", kc.name, err)
		}

		man.Cases = append(man.Cases, rec)
		_, _ = fmt.Fprintf(log, "%-36s %d→%d  %s\n", kc.name, rec.HitsWithout, rec.HitsWith, rec.TextWith)
	}

	err = tr.write(cfg.outDir)
	if err != nil {
		return err
	}

	return writeManifest(cfg.outDir, man)
}

// checkWithout transcribes a clip without keywords and requires the committed testdata/oracle/<id>.json.
func checkWithout(ctx context.Context, o *oracle, id, wav string) (transcript, error) {
	js, err := o.cli(ctx, wav)
	if err != nil {
		return transcript{}, fmt.Errorf("%s: %w", id, err)
	}

	want, err := readFixture(filepath.Join(o.cfg.oracleDir, id+".json"))
	if err != nil {
		return transcript{}, fmt.Errorf("%w (run `just oracle-run`)", err)
	}

	if !bytes.Equal(js, want) {
		return transcript{}, fmt.Errorf("%s: the run without keywords differs from %s/%s.json", id, o.cfg.oracleDir, id)
	}

	return parseTranscript(js)
}

func recordKeywordCase(ctx context.Context, o *oracle, kc keywordCase, c clip, wav string, base transcript,
	tr bundle,
) (keywordRecord, error) {
	rec := keywordRecord{
		Name: kc.name, Clip: c.ID, Language: c.Language, Kind: kc.kind, Keywords: kc.keywords,
		KeywordsFile: kc.name + ".txt", Without: "../" + c.ID + ".json", With: kc.name,
		Reference: c.Transcript, TextWithout: base.Text,
		HitsReference: hits(kc.keywords, c.Transcript), HitsWithout: hits(kc.keywords, base.Text),
	}

	err := checkKind(kc, rec)
	if err != nil {
		return rec, err
	}

	file := filepath.Join(o.cfg.outDir, rec.KeywordsFile)

	err = writeFile(file, []byte(strings.Join(kc.keywords, "\n")+"\n"))
	if err != nil {
		return rec, err
	}

	js, err := o.cli(ctx, wav, "--audio-keywords", file)
	if err != nil {
		return rec, err
	}

	tr.add(rec.With, js)

	t, err := parseTranscript(js)
	if err != nil {
		return rec, err
	}

	rec.TextWith, rec.Changed, rec.HitsWith = t.Text, t.Text != base.Text, hits(kc.keywords, t.Text)

	return rec, nil
}

// kindRules define the kinds by the keyword count n, whether the first keyword has several words, and how many
// keywords the reference and the engine's text without keywords contain.
var kindRules = map[string]func(n int, multiWord bool, ref, base int) bool{
	kwPresent:    func(n int, mw bool, ref, base int) bool { return n == 1 && !mw && ref == 1 && base == 1 },
	kwMissed:     func(n int, mw bool, ref, base int) bool { return n == 1 && !mw && ref == 1 && base == 0 },
	kwDistractor: func(n int, mw bool, ref, base int) bool { return n == 1 && !mw && ref == 0 && base == 0 },
	kwPhrase:     func(n int, mw bool, ref, _ int) bool { return n == 1 && mw && ref == 1 },
	kwList:       func(n int, _ bool, ref, _ int) bool { return n > 1 && ref == n },
}

// checkKind holds a case to its kind's definition against the reference and the engine's text without keywords.
func checkKind(kc keywordCase, r keywordRecord) error {
	n := len(kc.keywords)

	rule, ok := kindRules[kc.kind]
	if !ok || n == 0 || !rule(n, strings.Contains(kc.keywords[0], " "), r.HitsReference, r.HitsWithout) {
		return fmt.Errorf("not a %s case: %d keywords, %d in the reference, %d in the text without them", kc.kind, n,
			r.HitsReference, r.HitsWithout)
	}

	return nil
}

// hits counts the keywords whose words occur consecutively in text (matchRule).
func hits(keywords []string, text string) int {
	t := " " + strings.Join(normWords(text), " ") + " "
	n := 0

	for _, k := range keywords {
		if strings.Contains(t, " "+strings.Join(normWords(k), " ")+" ") {
			n++
		}
	}

	return n
}

// normWords lower-cases s and splits it at every character other than a letter, digit or apostrophe.
func normWords(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\'' && r != '’'
	})
}
