package search

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
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
