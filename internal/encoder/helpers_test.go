package encoder

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/safetensors"
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

// clip is one corpus clip with the engine's needle_embed rows.
type clip struct {
	ID      string
	Samples []float32
	Embed   []float32 // [Frames, 512]
	Frames  int
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

		store, err := safetensors.OpenStore(filepath.Join(oracleDir, id+".embed.safetensors"), safetensors.StoreOptions{})
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		ts, err := store.Tensor("embed")

		store.Close()

		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		out = append(out, clip{ID: id, Samples: samples, Embed: ts.Data, Frames: int(ts.Shape[0])})
	}

	return out
}

// frameCosines returns the cosine of each of the frames rows of dim values
// of a against b.
func frameCosines(a, b []float32, frames, dim int) []float64 {
	out := make([]float64, frames)

	for i := range frames {
		var ab, aa, bb float64

		for k := range dim {
			x, y := float64(a[i*dim+k]), float64(b[i*dim+k])
			ab += x * y
			aa += x * x
			bb += y * y
		}

		if aa > 0 && bb > 0 {
			out[i] = ab / math.Sqrt(aa*bb)
		}
	}

	return out
}

// mean is the arithmetic mean, 0 for none.
func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}

	var s float64
	for _, x := range v {
		s += x
	}

	return s / float64(len(v))
}
