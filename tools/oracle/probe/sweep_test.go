package probe

import (
	"math"
	"strings"
	"testing"
)

// fakeClips stands in for SweepClips: tones of the real clips' lengths.
func fakeClips() map[string][]float32 {
	out := map[string][]float32{}

	for id, n := range map[string]int{ClipEN: 100560, ClipDE: 110400} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(0.3 * math.Sin(2*math.Pi*220*float64(i)/SampleRate))
		}

		out[id] = x
	}

	return out
}

func TestSweepSets(t *testing.T) {
	clips := fakeClips()

	for name, set := range map[string]func(map[string][]float32) ([]Spec, error){
		"SilenceSet": SilenceSet, "LimitSet": LimitSet,
	} {
		specs, err := set(clips)
		if err != nil {
			t.Fatal(err)
		}

		names := map[string]bool{}

		for _, s := range specs {
			if names[s.Name] || strings.ContainsAny(s.Name, "/ ") {
				t.Fatalf("%s: duplicate or unsafe name %q", name, s.Name)
			}

			names[s.Name] = true

			if s.Purpose == "" {
				t.Errorf("%s: no purpose", s.Name)
			}

			if s.Samples == 0 {
				continue // the limits' empty clip
			}

			x, err := s.SignalWith(clips[s.Clip])
			if err != nil {
				t.Fatalf("%s: %v", s.Name, err)
			}

			y, _ := s.SignalWith(clips[s.Clip])
			if SHA256(x) != SHA256(y) {
				t.Fatalf("%s: two syntheses differ", s.Name)
			}

			if len(x) > MaxSamples && s.Name != "silence-480001" && s.Name != "speech-480001" {
				t.Errorf("%s: %d samples, over the engine's limit", s.Name, len(x))
			}
		}

		t.Logf("%s: %d signals", name, len(specs))
	}
}

func TestSilenceSetParameters(t *testing.T) {
	clips := fakeClips()

	specs, err := SilenceSet(clips)
	if err != nil {
		t.Fatal(err)
	}

	byName := map[string]Spec{}
	for _, s := range specs {
		byName[s.Name] = s
	}

	// The SNR family's σ is the clip's RMS scaled by the SNR.
	snr := byName["en-snr-m10"]
	if want := RMS(clips[ClipEN]) * math.Pow(10, 10.0/20); math.Abs(snr.Amplitude-want) > 1e-15 {
		t.Errorf("en-snr-m10 σ = %v, want %v", snr.Amplitude, want)
	}

	if s := byName["step-m40-2.65"]; math.Abs(s.Gain-math.Pow(10, 2.65/20)) > 1e-15 || s.GainDB != 2.65 {
		t.Errorf("step-m40-2.65: gain %v (%v dB)", s.Gain, s.GainDB)
	}

	if s := byName["en-i16-m87.5"]; !s.Quantize16 || s.GainDB != -87.5 {
		t.Errorf("en-i16-m87.5 = %+v", s)
	}

	if s := byName["de-pad-before-23s"]; s.PadBefore != 23*SampleRate || s.Samples != 23*SampleRate+110400 {
		t.Errorf("de-pad-before-23s = %+v", s)
	}
}

func TestPink(t *testing.T) {
	s := Spec{Name: "p", Kind: Pink, Samples: 64000, Amplitude: 1e-3, Seed: 2}

	x, err := s.Signal()
	if err != nil {
		t.Fatal(err)
	}

	if r := RMS(x); math.Abs(r-1e-3) > 1e-9 {
		t.Errorf("pink RMS = %.6g, want 1e-3", r)
	}

	// 1/f noise is strongly correlated from sample to sample; white noise is not.
	white, _ := Spec{Name: "w", Kind: Noise, Samples: 64000, Amplitude: 1e-3, Seed: 2}.Signal()
	if p, w := lag1(x), lag1(white); p < 0.5 || math.Abs(w) > 0.02 {
		t.Errorf("lag-1 correlation: pink %.3f, white %.3f", p, w)
	}

	// The warm-up keeps the start at the steady level.
	if head, tail := RMS(x[:8000]), RMS(x[56000:]); head < tail/2 || head > 2*tail {
		t.Errorf("pink RMS over the first and last 0.5 s: %.3g, %.3g", head, tail)
	}
}

func lag1(x []float32) float64 {
	var num, den float64
	for i := 1; i < len(x); i++ {
		num += float64(x[i]) * float64(x[i-1])
		den += float64(x[i]) * float64(x[i])
	}

	return num / den
}

func TestStep(t *testing.T) {
	s := Spec{Name: "s", Kind: Step, Samples: 64000, Amplitude: 0.01, Seed: 3, From: 24000, To: 40000, Gain: 2}

	x, err := s.Signal()
	if err != nil {
		t.Fatal(err)
	}

	// The same Gaussian stream as Noise, doubled inside [From, To).
	n, _ := Spec{Name: "n", Kind: Noise, Samples: 64000, Amplitude: 0.01, Seed: 3}.Signal()
	if x[23999] != n[23999] || x[24000] != 2*n[24000] || x[39999] != 2*n[39999] || x[40000] != n[40000] {
		t.Error("the step does not scale exactly [From, To)")
	}

	_, err = Spec{Name: "s", Kind: Step, Samples: 10, From: 4, To: 12, Gain: 2}.Signal()
	if err == nil {
		t.Error("a step past the end accepted")
	}
}

func TestSpeech(t *testing.T) {
	clip := []float32{0.5, -0.25, 0.125, 1}
	s := Spec{Name: "s", Kind: Speech, Samples: 6, Clip: "c", ClipFrom: 1, ClipTo: 3, PadBefore: 2, Gain: 0.5}

	x, err := s.SignalWith(clip)
	if err != nil {
		t.Fatal(err)
	}

	if want := []float32{0, 0, -0.125, 0.0625, 0, 0}; !equal(x, want) {
		t.Errorf("speech = %v, want %v", x, want)
	}

	_, err = s.Signal()
	if err == nil {
		t.Error("a speech probe synthesised without its clip")
	}

	// Noise over every sample, the same stream as a Noise probe of the same seed.
	s.Amplitude, s.Seed = 1e-3, 7
	y, _ := s.SignalWith(clip)
	n, _ := Spec{Name: "n", Kind: Noise, Samples: 6, Amplitude: 1e-3, Seed: 7}.Signal()

	if y[0] != n[0] || y[5] != n[5] || y[2] == x[2] {
		t.Errorf("speech under noise = %v, noise %v", y, n)
	}

	for _, bad := range []Spec{
		{Name: "b", Kind: Speech, Samples: 6, ClipTo: 5, Gain: 1},               // past the clip
		{Name: "b", Kind: Speech, Samples: 2, ClipTo: 3, Gain: 1},               // longer than the probe
		{Name: "b", Kind: Speech, Samples: 6, ClipTo: 3, PadBefore: 4, Gain: 1}, // padded past the end
		{Name: "b", Kind: Speech, Samples: 6, ClipTo: 3},                        // no gain
	} {
		_, err := bad.SignalWith(clip)
		if err == nil {
			t.Errorf("%+v: no error", bad)
		}
	}
}

func TestQuantize16(t *testing.T) {
	clip := []float32{0.3, -1, 1, 1e-6, 2e-5, -2e-5}
	s := Spec{Name: "q", Kind: Speech, Samples: 6, ClipTo: 6, Gain: 1, Quantize16: true}

	x, err := s.SignalWith(clip)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{9830.0 / 32768, -1, 32767.0 / 32768, 0, 1.0 / 32768, -1.0 / 32768}
	if !equal(x, want) {
		t.Errorf("quantised = %v, want %v", x, want)
	}
}

func equal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

func TestDBName(t *testing.T) {
	for db, want := range map[float64]string{0: "0", -84: "m84", 2.65: "2.65", -87.5: "m87.5", 30: "30"} {
		if got := dbName(db); got != want {
			t.Errorf("dbName(%v) = %q, want %q", db, got, want)
		}
	}
}
