package frontend

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const clipDir = "../../testdata/audio"

// TestReadWAVClips decodes the committed oracle clips (16 kHz mono 16-bit
// WAVs written by tools/oracle) and checks them against the manifest.
func TestReadWAVClips(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(clipDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Clips []struct {
			ID      string `json:"id"`
			Samples int    `json:"samples"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	if len(manifest.Clips) == 0 {
		t.Fatal("manifest lists no clips")
	}

	for _, clip := range manifest.Clips {
		path := filepath.Join(clipDir, clip.ID+".wav")

		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}

		w, err := ParseWAV(data)
		if err != nil {
			t.Fatalf("%s: %v", clip.ID, err)
		}

		if want := (Format{Channels: 1, SampleWidth: 2, SampleRate: SampleRate}); w.Format != want {
			t.Errorf("%s: format %+v, want %+v", clip.ID, w.Format, want)
		}

		samples, err := ReadWAV(path)
		if err != nil {
			t.Fatalf("%s: %v", clip.ID, err)
		}

		if len(samples) != clip.Samples {
			t.Errorf("%s: %d samples, manifest %d", clip.ID, len(samples), clip.Samples)
		}

		for i, v := range samples {
			if scaled := float64(v) * 32768; scaled != math.Trunc(scaled) || v < -1 || v >= 1 {
				t.Fatalf("%s: sample %d = %v is not a 16-bit value / 32768", clip.ID, i, v)
			}
		}
	}
}
