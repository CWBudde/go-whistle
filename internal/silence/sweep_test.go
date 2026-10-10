package silence

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/tools/oracle/probe"
)

const (
	clipDir    = "../../testdata/audio"
	silenceSet = "../../testdata/oracle/silence/manifest.json"
	limitsSet  = "../../testdata/oracle/limits/manifest.json"
)

// signal is one sweep or limit signal with the engine's verdict.
type signal struct {
	Name    string
	Set     string
	PCM     []float32
	Empty   bool // the engine returned exactly the empty JSON
	TooLong bool // the engine refused it (over 30 s)
}

type manifestSignal struct {
	probe.Spec

	PCMSHA256 string `json:"pcm_sha256"`
	Empty     bool   `json:"empty"`
	Lib       *struct {
		RC    int  `json:"rc"`
		Empty bool `json:"empty"`
	} `json:"lib"`
}

// loadSignals regenerates the 320 silence-sweep and 22 limit signals from
// their manifests and checks each against its pcm_sha256.
func loadSignals(t testing.TB) []signal {
	t.Helper()

	clips := map[string][]float32{}

	for _, id := range probe.SweepClips() {
		x, err := frontend.ReadWAV(filepath.Join(clipDir, id+".wav"))
		if err != nil {
			t.Fatal(err)
		}

		clips[id] = x
	}

	out := make([]signal, 0, 342)

	for _, set := range []string{silenceSet, limitsSet} {
		raw, err := os.ReadFile(set)
		if err != nil {
			t.Fatal(err)
		}

		var m struct {
			Signals []manifestSignal `json:"signals"`
		}

		err = json.Unmarshal(raw, &m)
		if err != nil {
			t.Fatal(err)
		}

		name := filepath.Base(filepath.Dir(set))

		for _, s := range m.Signals {
			var pcm []float32

			if s.Samples > 0 {
				pcm, err = s.SignalWith(clips[s.Clip])
				if err != nil {
					t.Fatal(err)
				}
			}

			if got := probe.SHA256(pcm); got != s.PCMSHA256 {
				t.Fatalf("%s/%s: regenerated samples %s, manifest %s", name, s.Name, got, s.PCMSHA256)
			}

			sig := signal{Name: s.Name, Set: name, PCM: pcm, Empty: s.Empty}
			if s.Lib != nil {
				// The library's verdict: the CLI refuses an empty WAV that
				// needle_transcribe treats as silence (F-025).
				sig.Empty = s.Lib.Empty
				sig.TooLong = s.Lib.RC < 0
			}

			out = append(out, sig)
		}
	}

	return out
}
