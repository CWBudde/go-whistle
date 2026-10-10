package whistle

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/search"
)

// keywordCase is one entry of testdata/oracle/keywords/manifest.json
// (docs/FINDINGS.md F-025).
type keywordCase struct {
	Name         string `json:"name"`
	Clip         string `json:"clip"`
	Language     string `json:"language"`
	KeywordsFile string `json:"keywords_file"`
	TextWith     string `json:"text_with"`
}

// keywordCasesFast are the keyword cases the search reproduces on short
// clips (internal/search keywordFast, F-029): a lower-cased missed name the
// engine writes capitalised, a distractor that loses against the audio,
// one that wins, and a list of two keywords.
var keywordCasesFast = []string{
	"en-cowleys-lower-missed",
	"en-curses-distractor",
	"en-transfer-distractor",
	"de-paare-adoptionsplan-list",
}

// TestTranscribeKeywordsMatchEngine transcribes keywordCasesFast with the
// keywords of their --audio-keywords files and requires the engine's text
// with them (F-029).
func TestTranscribeKeywordsMatchEngine(t *testing.T) {
	m := loadModel(t)

	var manifest struct {
		Cases []keywordCase `json:"cases"`
	}

	dir := filepath.Join(oracleDir, "keywords")
	readJSON(t, filepath.Join(dir, "manifest.json"), &manifest)

	var cases []keywordCase

	for _, c := range manifest.Cases {
		if slices.Contains(keywordCasesFast, c.Name) {
			cases = append(cases, c)
		}
	}

	if len(cases) != len(keywordCasesFast) {
		t.Fatalf("%d of the %d keyword cases in the manifest", len(cases), len(keywordCasesFast))
	}

	for _, c := range cases {
		raw, err := os.ReadFile(filepath.Join(dir, c.KeywordsFile)) // #nosec G304 -- a fixture path.
		if err != nil {
			t.Fatal(err)
		}

		pcm := loadClips(t, c.Clip)[0].Samples

		res, err := m.Transcribe(pcm, TranscribeOptions{Keywords: search.ParseKeywords(string(raw))})
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}

		if res.Text != c.TextWith || res.Language != c.Language {
			t.Errorf("%s: [%s] %q, the engine [%s] %q", c.Name, res.Language, res.Text, c.Language, c.TextWith)
		}
	}
}

// TestTranscribeKeywordsFormat checks that a keyword list in the engine's
// file format, one entry with newlines, decodes like the split list, and
// that blank entries mean no keywords.
func TestTranscribeKeywordsFormat(t *testing.T) {
	m := loadModel(t)
	pcm := loadClips(t, "fleurs-de_de-10308608197319017935")[0].Samples

	split, err := m.Transcribe(pcm, TranscribeOptions{Keywords: []string{"Paare", "Adoptionsplan"}})
	if err != nil {
		t.Fatal(err)
	}

	joined, err := m.Transcribe(pcm, TranscribeOptions{Keywords: []string{"  Paare\n\nAdoptionsplan \n"}})
	if err != nil {
		t.Fatal(err)
	}

	if split.Text != joined.Text {
		t.Errorf("one entry with newlines %q, two entries %q", joined.Text, split.Text)
	}

	blank, err := m.Transcribe(pcm, TranscribeOptions{Keywords: []string{"", " \n "}})
	if err != nil {
		t.Fatal(err)
	}

	plain, err := m.Transcribe(pcm, TranscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if blank.Text != plain.Text {
		t.Errorf("blank keywords %q, none %q", blank.Text, plain.Text)
	}
}

// depthClipsFast are short clips whose engine text at --audio-depth 2 and 4
// differs from the full depth's and which the ladder reproduces
// (internal/search depthClips, docs/FINDINGS.md F-030).
var depthClipsFast = []string{
	"ls-other-8280-266249-0032",
	"ls-clean-260-123440-0007",
	"ls-clean-7729-102255-0012",
}

// TestTranscribeDepthMatchesEngine transcribes depthClipsFast at depths 2
// and 4 and requires the engine's text at that depth
// (testdata/oracle/depth, F-025).
func TestTranscribeDepthMatchesEngine(t *testing.T) {
	m := loadModel(t)

	var want map[string]struct {
		Language string `json:"language"`
		Text     string `json:"text"`
	}

	readJSON(t, filepath.Join(oracleDir, "depth", "transcripts.json"), &want)

	for _, c := range loadClips(t, depthClipsFast...) {
		for _, depth := range []int{2, 4} {
			res, err := m.Transcribe(c.Samples, TranscribeOptions{Depth: depth})
			if err != nil {
				t.Fatalf("%s d%d: %v", c.ID, depth, err)
			}

			w, ok := want[fmt.Sprintf("%s.d%d", c.ID, depth)]
			if !ok {
				t.Fatalf("%s d%d: no engine transcript", c.ID, depth)
			}

			if res.Text != w.Text || res.Language != w.Language {
				t.Errorf("%s d%d: [%s] %q, the engine [%s] %q", c.ID, depth, res.Language, res.Text, w.Language, w.Text)
			}

			if res.Text == c.Text {
				t.Errorf("%s d%d: equals the full depth's text %q", c.ID, depth, c.Text)
			}
		}
	}
}
