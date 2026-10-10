package decoder

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-whistle/internal/encoder"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Pinned inputs as laid out by `whistle download` and `just oracle-run`.
const (
	whistleSafetensorsPath = "../../models/whistle/whistle.safetensors"
	whistleCactPath        = "../../models/whistle/whistle.cact"
	clipDir                = "../../testdata/audio"
	oracleDir              = "../../testdata/oracle"
)

// openModel opens a weight file or skips the test when it is missing.
//
//nolint:unparam // the probe harness (build tag oracle) also opens the safetensors weights
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

// clip is one corpus clip with the engine's transcript.
type clip struct {
	ID       string
	Samples  []float32
	Language string // the engine's language
	Text     string // the engine's text
}

// loadClips reads the named clips (all when ids is empty) from
// testdata/audio and testdata/oracle.
func loadClips(t testing.TB, ids ...string) []clip {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(clipDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Clips []struct {
			ID string `json:"id"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	if len(ids) == 0 {
		for _, c := range manifest.Clips {
			ids = append(ids, c.ID)
		}
	}

	out := make([]clip, 0, len(ids))

	for _, id := range ids {
		samples, err := frontend.ReadWAV(filepath.Join(clipDir, id+".wav"))
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		js, err := os.ReadFile(filepath.Join(oracleDir, id+".json"))
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		var tr struct {
			Language string `json:"language"`
			Text     string `json:"text"`
		}

		err = json.Unmarshal(js, &tr)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		out = append(out, clip{ID: id, Samples: samples, Language: tr.Language, Text: tr.Text})
	}

	return out
}

// encode runs the encoder hypothesis encoder.Best() on the clips, with the
// output tap of the Memory axis, and returns one memory input per clip.
//
//nolint:unparam // the probe harness (build tag oracle) also encodes with the lane-mean tap
func encode(t testing.TB, src weights.Source, clips []clip, tap MemoryTap) ([][]float32, []int) {
	t.Helper()

	h := encoder.Best()
	if tap == MemLaneMean {
		h.Output = encoder.OutLaneMean
	}

	enc, err := encoder.Load(src, h)
	if err != nil {
		t.Fatal(err)
	}

	outs, frames := make([][]float32, len(clips)), make([]int, len(clips))

	for i, c := range clips {
		outs[i], frames[i], err = enc.Embed(c.Samples)
		if err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}
	}

	return outs, frames
}
