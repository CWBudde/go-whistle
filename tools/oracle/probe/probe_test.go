package probe

import (
	"math"
	"slices"
	"testing"
)

func TestFrames(t *testing.T) {
	// ceil(floor(n/160)/8), F-012: the same lengths as the encoder's TestStemFrames.
	for n, want := range map[int]int{1280: 1, 1281: 1, 2560: 2, 5120: 4, 16000: 13, 53040: 42, 160000: 125, 480000: 375} {
		if got := Frames(n); got != want {
			t.Errorf("Frames(%d) = %d, want %d", n, got, want)
		}
	}

	// Frame t is interior when [1280t − 1320, 1280t + 1320) lies inside the clip: t ≥ 2 and 1280t + 1320 ≤ n.
	for n, want := range map[int]int{5120: 1, 16000: 10, 32000: 22, 480000: 372, 2560: 0} {
		if got := InteriorFrames(n); got != want {
			t.Errorf("InteriorFrames(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestSignals(t *testing.T) {
	set := Set()
	names := map[string]bool{}
	frames := 0

	for _, s := range set {
		if names[s.Name] {
			t.Fatalf("duplicate probe %s", s.Name)
		}

		names[s.Name] = true
		frames += Frames(s.Samples)

		x, err := s.Signal()
		if err != nil {
			t.Fatal(err)
		}

		y, _ := s.Signal()
		if SHA256(x) != SHA256(y) {
			t.Fatalf("%s: two syntheses differ", s.Name)
		}

		if s.Purpose == "" {
			t.Errorf("%s: no purpose", s.Name)
		}
	}

	// The committed rows are 2 KiB per frame; keep them under 3 MB.
	if frames*512*4 > 3_000_000 {
		t.Errorf("%d frames: %d bytes of rows", frames, frames*512*4)
	}

	t.Logf("%d probes, %d encoder frames", len(set), frames)
}

func TestSignalShapes(t *testing.T) {
	burst, err := Spec{Name: "b", Kind: Burst, Samples: 100, Amplitude: 0.5, Hz: 4000, From: 10, To: 20}.Signal()
	if err != nil {
		t.Fatal(err)
	}

	// 4 kHz at 16 kHz: sin(πi/2) = 0, 1, 0, −1, …
	if burst[9] != 0 || burst[11] != -0.5 || burst[13] != 0.5 || burst[21] != 0 {
		t.Errorf("burst samples 9, 11, 13, 21: %v %v %v %v", burst[9], burst[11], burst[13], burst[21])
	}

	imp, err := Spec{Name: "i", Kind: Impulse, Samples: 10, Offsets: []int{3, 7}, Amplitudes: []float64{1, -2}}.Signal()
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(imp, []float32{0, 0, 0, 1, 0, 0, 0, -2, 0, 0}) {
		t.Errorf("impulses: %v", imp)
	}

	noise, err := Spec{Name: "n", Kind: Noise, Samples: 40000, Amplitude: 1e-3, Seed: 9}.Signal()
	if err != nil {
		t.Fatal(err)
	}

	if sigma := rms(noise); math.Abs(sigma-1e-3) > 2e-5 {
		t.Errorf("noise σ = %.4g, want 1e-3", sigma)
	}

	decay, err := Spec{Name: "d", Kind: Decay, Samples: 64000, FromDB: -20, ToDB: -140, Seed: 3}.Signal()
	if err != nil {
		t.Fatal(err)
	}

	head, tail := rms(decay[:1000]), rms(decay[63000:])
	if db := 20 * math.Log10(head/tail); math.Abs(db-118) > 3 {
		t.Errorf("decay: %.1f dB between the first and last 1000 samples, want about 118", db)
	}

	for _, bad := range []Spec{
		{Name: "z", Kind: Silence},
		{Name: "k", Kind: "chirp", Samples: 4},
		{Name: "b", Kind: Burst, Samples: 4, From: 3, To: 9},
		{Name: "i", Kind: Impulse, Samples: 4, Offsets: []int{4}, Amplitudes: []float64{1}},
		{Name: "i", Kind: Impulse, Samples: 4, Offsets: []int{1}},
	} {
		_, err := bad.Signal()
		if err == nil {
			t.Errorf("%+v: no error", bad)
		}
	}
}

func rms(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}

	return math.Sqrt(s / float64(len(x)))
}
