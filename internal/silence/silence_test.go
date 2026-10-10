package silence

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/go-whistle/internal/encoder"
)

func newGate(t *testing.T) *Gate {
	t.Helper()

	g, err := New()
	if err != nil {
		t.Fatal(err)
	}

	return g
}

// Accept (PLAN.md 6.6, F-031): the gate gives the engine's verdict on every
// signal of the silence sweep and the length limits.
func TestGateMatchesEngine(t *testing.T) {
	g := newGate(t)
	sigs := loadSignals(t)

	var (
		n, tooLong            int
		maxEmpty, minOpen     = math.Inf(-1), math.Inf(1)
		closeEmpty, closeOpen string
	)

	for _, s := range sigs {
		if s.TooLong {
			tooLong++ // an error before the gate (F-025); whistle tests it

			continue
		}

		spread, err := g.Spread(s.PCM)
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}

		silent, err := g.Silent(s.PCM)
		if err != nil || silent != (spread <= ThresholdDB) {
			t.Fatalf("%s: Silent %v, %v; spread %.3f", s.Name, silent, err, spread)
		}

		if silent != s.Empty {
			t.Errorf("%s/%s: spread %.3f dB, silent %v; the engine's JSON empty %v", s.Set, s.Name, spread, silent, s.Empty)
		}

		if s.Empty && spread > maxEmpty {
			maxEmpty, closeEmpty = spread, s.Name
		}

		if !s.Empty && spread < minOpen {
			minOpen, closeOpen = spread, s.Name
		}

		n++
	}

	if n+tooLong != 342 || tooLong != 2 {
		t.Errorf("%d signals with a verdict and %d over 30 s, want 340 and 2", n, tooLong)
	}

	t.Logf("%d signals: largest empty spread %.4f dB (%s), smallest transcribed %.4f dB (%s); threshold %g dB",
		n, maxEmpty, closeEmpty, minOpen, closeOpen, ThresholdDB)
}

func TestFramingIsTheEncoders(t *testing.T) {
	if got, want := Framing, encoder.Best().Frontend; got != want {
		t.Errorf("Framing %v, encoder.Best().Frontend %v", got, want)
	}
}

func TestPercentile(t *testing.T) {
	ten := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	for _, tc := range []struct {
		values []float64
		p      int
		want   float64
	}{
		{ten, 10, 1},
		{ten, 11, 2},
		{ten, 50, 5},
		{ten, 99, 10},
		{ten, 100, 10},
		{ten, 1, 1},
		{[]float64{7}, 10, 7},
		{[]float64{7}, 99, 7},
	} {
		if got := Percentile(tc.values, tc.p); got != tc.want {
			t.Errorf("Percentile(%v, %d) = %v, want %v", tc.values, tc.p, got, tc.want)
		}
	}

	// 400 frames: p99 is the 396th value, so the 4 largest are above it.
	v := make([]float64, 400)
	for i := range v {
		v[i] = float64(i)
	}

	if got := Percentile(v, 99); got != 395 {
		t.Errorf("p99 of 0…399 = %v, want 395", got)
	}
}

func TestSpread(t *testing.T) {
	inf := math.Inf(-1)

	for _, tc := range []struct {
		name   string
		levels []float64
		want   float64
	}{
		{"none", nil, 0},
		{"one", []float64{-20}, 0},
		{"steady", []float64{-30, -30, -30}, 0},
		{"all -Inf", []float64{inf, inf}, 0},
		{"step", []float64{-30, -30, -30, -30, -30, -30, -30, -30, -30, -24}, 6},
		{"unordered", []float64{-24, -30, -30, -30, -30, -30, -30, -30, -30, -30}, 6},
		{"silence below speech", []float64{inf, -40}, math.Inf(1)},
	} {
		if got := Spread(tc.levels); got != tc.want {
			t.Errorf("%s: Spread(%v) = %v, want %v", tc.name, tc.levels, got, tc.want)
		}
	}

	// Spread sorts a copy.
	in := []float64{3, 1, 2}
	Spread(in)

	if in[0] != 3 || in[1] != 1 || in[2] != 2 {
		t.Errorf("Spread changed its input: %v", in)
	}
}

func TestSilentSignals(t *testing.T) {
	g := newGate(t)
	rng := rand.New(rand.NewPCG(1, 2))

	noise := func(n int, level func(i int) float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(level(i) * rng.NormFloat64())
		}

		return x
	}

	steady := func(int) float64 { return 0.01 }

	for _, tc := range []struct {
		name   string
		pcm    []float32
		silent bool
	}{
		{"no samples", nil, true},
		{"159 samples", noise(159, steady), true},
		{"digital silence", make([]float32, 32000), true},
		{"steady noise", noise(32000, steady), true},
		{"quiet steady noise", noise(32000, func(int) float64 { return 1e-6 }), true},
		{"noise with a 10 dB louder second", noise(64000, func(i int) float64 {
			if i >= 24000 && i < 40000 {
				return 0.01 * math.Sqrt(10)
			}

			return 0.01
		}), false},
		{"a burst after silence", append(make([]float32, 16000), noise(16000, steady)...), false},
	} {
		got, err := g.Silent(tc.pcm)
		if err != nil || got != tc.silent {
			s, _ := g.Spread(tc.pcm)
			t.Errorf("%s: Silent = %v, %v (spread %.2f dB); want %v", tc.name, got, err, s, tc.silent)
		}
	}
}

func TestLevelsFrames(t *testing.T) {
	g := newGate(t)

	for _, n := range []int{0, 159, 160, 1279, 16000, 16159} {
		levels, err := g.Levels(make([]float32, n))
		if err != nil || len(levels) != n/160 {
			t.Errorf("Levels(%d samples) = %d frames, %v; want %d", n, len(levels), err, n/160)
		}

		for _, l := range levels {
			if !math.IsInf(l, -1) {
				t.Errorf("Levels of zeros: %v, want -Inf", l)

				break
			}
		}
	}
}
