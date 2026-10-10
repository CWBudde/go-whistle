package search

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/encoder"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Pinned inputs as laid out by `whistle download` and `just oracle-run`.
const (
	whistleCactPath = "../../models/whistle/whistle.cact"
	clipDir         = "../../testdata/audio"
	oracleDir       = "../../testdata/oracle"
)

// openModel opens a weight file or skips the test when it is missing.
func openModel(t testing.TB, path string) weights.Source {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", path)
	}

	src, err := weights.Open(path, weights.Options{})
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}

	t.Cleanup(func() { _ = src.Close() })

	return src
}

// clip is one corpus clip with the engine's transcript and its encoder
// memory.
type clip struct {
	ID       string
	Language string // the engine's language
	Text     string // the engine's text
	Mem      *decoder.Memory
}

// loadSetup opens the .cact weights (or skips), loads decoder.Best() and
// encodes the named clips with encoder.Best(), the parity setting
// (docs/FINDINGS.md F-020, F-024).
func loadSetup(t testing.TB, ids ...string) (*decoder.Decoder, []clip) {
	t.Helper()

	src := openModel(t, whistleCactPath)

	d, err := decoder.Load(src, nil, decoder.Best())
	if err != nil {
		t.Fatal(err)
	}

	enc, err := encoder.Load(src, encoder.Best())
	if err != nil {
		t.Fatal(err)
	}

	out := make([]clip, len(ids))

	for i, id := range ids {
		samples, err := frontend.ReadWAV(filepath.Join(clipDir, id+".wav"))
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		js, err := os.ReadFile(filepath.Join(oracleDir, id+".json"))
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		c := clip{ID: id}

		err = json.Unmarshal(js, &struct {
			Language *string `json:"language"`
			Text     *string `json:"text"`
		}{&c.Language, &c.Text})
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		e, frames, err := enc.Embed(samples)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		c.Mem, err = d.Memory(e, frames)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		out[i] = c
	}

	return d, out
}

// depthTexts reads testdata/oracle/depth/transcripts.json: the engine's
// language and text per "<clip>.d<N>" at --audio-depth N
// (docs/FINDINGS.md F-025).
func depthTexts(t testing.TB) map[string]clip {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(oracleDir, "depth", "transcripts.json"))
	if err != nil {
		t.Fatal(err)
	}

	var all map[string]struct {
		Language string `json:"language"`
		Text     string `json:"text"`
	}

	err = json.Unmarshal(raw, &all)
	if err != nil {
		t.Fatal(err)
	}

	out := make(map[string]clip, len(all))
	for k, v := range all {
		out[k] = clip{Language: v.Language, Text: v.Text}
	}

	return out
}

// keywordCase is one case of testdata/oracle/keywords/manifest.json
// (docs/FINDINGS.md F-025): a keyword file on a clip, with the engine's
// text without and with it.
type keywordCase struct {
	Name        string   `json:"name"`
	Clip        string   `json:"clip"`
	Language    string   `json:"language"`
	Kind        string   `json:"kind"` // present, missed, distractor, phrase, list
	Keywords    []string `json:"keywords"`
	File        string   `json:"keywords_file"`
	Reference   string   `json:"reference"`
	TextWithout string   `json:"text_without"`
	TextWith    string   `json:"text_with"`
	Changed     bool     `json:"changed"`
}

// keywordCases reads the keyword manifest, the named cases only when names
// are given, and checks each keyword file against the manifest's list.
func keywordCases(t testing.TB, names ...string) []keywordCase {
	t.Helper()

	dir := filepath.Join(oracleDir, "keywords")

	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Cases []keywordCase `json:"cases"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	var out []keywordCase

	for _, c := range manifest.Cases {
		if len(names) > 0 && !slices.Contains(names, c.Name) {
			continue
		}

		file, err := os.ReadFile(filepath.Join(dir, c.File))
		if err != nil {
			t.Fatal(err)
		}

		if kws := ParseKeywords(string(file)); !slices.Equal(kws, c.Keywords) {
			t.Fatalf("%s: keyword file %q, manifest %q", c.Name, kws, c.Keywords)
		}

		out = append(out, c)
	}

	if len(out) != max(len(names), 1) && len(names) > 0 {
		t.Fatalf("keyword cases %v: found %d", names, len(out))
	}

	return out
}
