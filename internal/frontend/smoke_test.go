package frontend

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func clipSamples(t *testing.T) map[string][]float32 {
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

	out := map[string][]float32{}

	for _, c := range manifest.Clips {
		x, err := ReadWAV(filepath.Join(clipDir, c.ID+".wav"))
		if err != nil {
			t.Fatal(err)
		}

		out[c.ID] = x
	}

	if len(out) != 32 {
		t.Fatalf("%d clips, want 32", len(out))
	}

	return out
}

// featureStats checks the shape and finiteness and returns the range.
func featureStats(t *testing.T, name string, fe Features, frames int) (lo, hi float64) {
	t.Helper()

	if fe.Frames != frames || len(fe.Data) != frames*Mels {
		t.Fatalf("%s: %d frames (%d values), want %d", name, fe.Frames, len(fe.Data), frames)
	}

	lo, hi = math.Inf(1), math.Inf(-1)

	for _, v := range fe.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: non-finite value %v", name, v)
		}

		lo, hi = min(lo, float64(v)), max(hi, float64(v))
	}

	return lo, hi
}

// TestLogMelClips runs the committed oracle clips through three hypotheses
// and checks shape, finiteness and a plausible range for speech.
func TestLogMelClips(t *testing.T) {
	plain := stftOptions(HannPeriodic, PadReflect) // ln, no normalisation

	fronts := map[string]*Frontend{
		"ln":      newFrontend(t, plain),
		"whisper": newFrontend(t, Whisper()),
		"nemo":    newFrontend(t, NeMo()),
	}

	lnLo, lnHi, quietest := math.Inf(1), math.Inf(-1), math.Inf(1)

	for id, x := range clipSamples(t) {
		frames := len(x) / Hop

		fe, err := fronts["ln"].LogMel(x)
		if err != nil {
			t.Fatal(err)
		}

		lo, hi := featureStats(t, id+" ln", fe, frames)
		lnLo, lnHi = min(lnLo, lo), max(lnHi, hi)
		quietest = min(quietest, hi)

		// Speech at 16-bit: a peak mel energy between e^-8 and e^10, and
		// at most 5 % of the values at the 1e-10 floor (digital silence).
		floored := 0

		for _, v := range fe.Data {
			if v <= float32(math.Log(1e-10)) {
				floored++
			}
		}

		if hi < -8 || hi > 10 || floored*20 > len(fe.Data) {
			t.Errorf("%s ln: range [%.2f, %.2f], %d of %d values at the floor", id, lo, hi, floored, len(fe.Data))
		}

		fe, err = fronts["whisper"].LogMel(x)
		if err != nil {
			t.Fatal(err)
		}

		// Whisper: the clamp leaves exactly 8 decades, i.e. a spread of at
		// most 2 after (y + 4) / 4.
		lo, hi = featureStats(t, id+" whisper", fe, frames)
		if hi-lo > 2+1e-6 || hi-lo < 1 {
			t.Errorf("%s whisper: range [%.3f, %.3f]", id, lo, hi)
		}

		fe, err = fronts["nemo"].LogMel(x)
		if err != nil {
			t.Fatal(err)
		}

		// NeMo: every channel standardised, so mean 0 and std just below 1.
		featureStats(t, id+" nemo", fe, frames)

		for m := range Mels {
			col := make([]float64, frames)
			for f := range frames {
				col[f] = float64(fe.Data[f*Mels+m])
			}

			mean, std := meanStd(col)
			if math.Abs(mean) > 1e-5 || math.Abs(std-1) > 1e-3 {
				t.Errorf("%s nemo channel %d: mean %.2e, std %.6f", id, m, mean, std)
			}
		}
	}

	t.Logf("ln mel over 32 clips: [%.2f, %.2f], quietest clip peak %.2f", lnLo, lnHi, quietest)
}

// TestLogMelAllVariants runs every hypothesis on 1.5 s of one clip.
func TestLogMelAllVariants(t *testing.T) {
	x := clipSamples(t)["ls-clean-1320-122617-0003"][:24000]

	for _, opts := range Variants() {
		f := newFrontend(t, opts)

		fe, err := f.LogMel(x)
		if err != nil {
			t.Fatalf("%s: %v", opts, err)
		}

		lo, hi := featureStats(t, opts.String(), fe, f.FrameCount(len(x)))
		if lo < -60 || hi > 30 {
			t.Errorf("%s: range [%.2f, %.2f]", opts, lo, hi)
		}
	}
}

// TestLogMel30s: 30 s of audio give the published 3000 frames with centred
// framing and 2998 with PadNone.
func TestLogMel30s(t *testing.T) {
	var x []float32
	for _, c := range clipSamples(t) {
		x = append(x, c...)
	}

	if len(x) < 480000 {
		t.Fatalf("clips hold %d samples", len(x))
	}

	x = x[:480000]

	for _, c := range []struct {
		opts   Options
		frames int
	}{{Whisper(), 3000}, {NeMo(), 3000}, {stftOptions(HannSymmetric, PadNone), 2998}} {
		f := newFrontend(t, c.opts)

		start := time.Now()

		fe, err := f.LogMel(x)
		if err != nil {
			t.Fatal(err)
		}

		featureStats(t, c.opts.String(), fe, c.frames)
		t.Logf("%s: %d × %d in %v", c.opts, fe.Frames, Mels, time.Since(start))
	}
}
