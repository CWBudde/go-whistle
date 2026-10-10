package whistle

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/tools/oracle/probe"
)

// sweepSignal is one signal of the 6.1 silence sweep or length limits,
// regenerated from its manifest (docs/FINDINGS.md F-025).
type sweepSignal struct {
	probe.Spec

	PCMSHA256 string `json:"pcm_sha256"`
	Empty     bool   `json:"empty"`
	Text      string `json:"text"`
	Language  string `json:"language"`
	Lib       *struct {
		RC    int  `json:"rc"`
		Empty bool `json:"empty"`
	} `json:"lib"`

	PCM []float32 `json:"-"`
}

// loadSweep reads testdata/oracle/<set>/manifest.json and regenerates the
// samples of every signal keep selects, checked against pcm_sha256. A limit
// signal takes needle_transcribe's verdict: the CLI refuses an empty WAV
// that the library treats as silence.
func loadSweep(t *testing.T, set string, keep func(sweepSignal) bool) []sweepSignal {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(oracleDir, set, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var m struct {
		Signals []sweepSignal `json:"signals"`
	}

	err = json.Unmarshal(raw, &m)
	if err != nil {
		t.Fatal(err)
	}

	clips := map[string][]float32{}

	var out []sweepSignal

	for _, s := range m.Signals {
		if !keep(s) {
			continue
		}

		if s.Clip != "" && clips[s.Clip] == nil {
			clips[s.Clip], err = frontend.ReadWAV(filepath.Join(clipDir, s.Clip+".wav"))
			if err != nil {
				t.Fatal(err)
			}
		}

		if s.Samples > 0 {
			s.PCM, err = s.SignalWith(clips[s.Clip])
			if err != nil {
				t.Fatal(err)
			}
		}

		if got := probe.SHA256(s.PCM); got != s.PCMSHA256 {
			t.Fatalf("%s/%s: regenerated samples %s, manifest %s", set, s.Name, got, s.PCMSHA256)
		}

		if s.Lib != nil {
			s.Empty = s.Lib.Empty
		}

		out = append(out, s)
	}

	return out
}

// The engine's JSON for silence without word timestamps (F-027).
const silenceNoWordsJSON = `{"text":"","language":"","ttft_ms":0.0,"decode_tps":0.0}`

// checkSilent requires exactly the engine's silence result, undecoded.
func checkSilent(t *testing.T, name string, res Result) {
	t.Helper()

	js, err := res.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	if string(js) != silenceNoWordsJSON {
		t.Errorf("%s: %s, want the engine's silence %s", name, js, silenceNoWordsJSON)
	}
}

// The length limits through Transcribe (F-025): over 30 s is an error before
// the gate, the engine's empty ones are silence, the rest are transcribed.
func TestTranscribeLimits(t *testing.T) {
	m := loadModel(t)
	sigs := loadSweep(t, "limits", func(sweepSignal) bool { return true })
	same := 0

	for _, s := range sigs {
		res, err := m.Transcribe(s.PCM, TranscribeOptions{})

		switch {
		case s.Lib.RC < 0:
			if !errors.Is(err, ErrTooLong) {
				t.Errorf("%s: %v, want ErrTooLong", s.Name, err)
			}

			continue
		case err != nil:
			t.Fatalf("%s: %v", s.Name, err)
		case s.Empty:
			checkSilent(t, s.Name, res)
		case res.Text == "" || res.Language != s.Language:
			t.Errorf("%s: [%s] %q, the engine [%s] %q", s.Name, res.Language, res.Text, s.Language, s.Text)
		}

		if res.Text == s.Text {
			same++
		}

		t.Logf("%-15s engine %-5v [%s] %q; ours [%s] %q", s.Name, s.Empty, s.Language, s.Text, res.Language, res.Text)
	}

	t.Logf("%d limit signals: text equal to the engine's on %d of the %d under 30 s", len(sigs), same, len(sigs)-2)
}

// The silence sweep through Transcribe (F-025, F-031): every signal the
// engine returned the empty JSON for is silence without decoding, and the
// transcribed signals closest to the gate's threshold on either side of
// each family are decoded.
func TestTranscribeSilenceSweep(t *testing.T) {
	m := loadModel(t)

	nearest := map[string]bool{
		"step-m40-2.7": true, "step-m80-2.7": true, "step-m10-2.7": true, "step-m40-m3.5": true,
		"width-560": true, "width-57600": true, "en-snr-m9.75": true, "de-snr-m15": true,
		"en-i16-m87": true, "de-i16-m82.5": true,
	}

	sigs := loadSweep(t, "silence", func(s sweepSignal) bool { return s.Empty || nearest[s.Name] })
	empty := 0

	for _, s := range sigs {
		res, err := m.Transcribe(s.PCM, TranscribeOptions{})
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}

		if s.Empty {
			checkSilent(t, s.Name, res)

			empty++

			continue
		}

		if res.Text == "" || res.Language == "" {
			t.Errorf("%s: [%s] %q, the engine [%s] %q", s.Name, res.Language, res.Text, s.Language, s.Text)
		}

		t.Logf("%-14s engine [%s] %q; ours [%s] %q", s.Name, s.Language, s.Text, res.Language, res.Text)
	}

	if empty != 159 {
		t.Errorf("%d empty sweep signals, want 159", empty)
	}
}

// Silence through TranscribeWAV and the JSON: the recorded engine output for
// 2 s of digital silence, byte for byte.
func TestTranscribeSilenceJSON(t *testing.T) {
	m := loadModel(t)
	path := filepath.Join(t.TempDir(), "silence.wav")
	writeWAV(t, path, make([]int16, 2*SampleRate))

	res, err := m.TranscribeWAV(path, TranscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	js, err := res.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	want, err := os.ReadFile(filepath.Join("testdata", "needle-cli", "silence-2s.stdout"))
	if err != nil {
		t.Fatal(err)
	}

	if string(js)+"\n" != string(want) {
		t.Errorf("2 s of silence: %s, the engine printed %s", js, want)
	}
}
