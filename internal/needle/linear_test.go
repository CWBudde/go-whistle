package needle

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// TestLinear checks the float32 Linear against tensor.MatMulTransB, its
// rows and row views, bit for bit.
func TestLinear(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	w, x := randVec(rng, 7*13, 1), randVec(rng, 3*13, 1)
	l := NewLinear(w, 7, 13)

	if l.Out() != 7 || l.In() != 13 || l.Packed() {
		t.Fatalf("Linear [%d, %d] packed %v", l.Out(), l.In(), l.Packed())
	}

	got, want := make([]float32, 3*7), make([]float32, 3*7)
	l.Apply(got, x, 3)
	tensor.MatMulTransB(want, 7, x, w, 3, 7, 13)

	if !slices.Equal(got, want) {
		t.Error("Apply differs from MatMulTransB")
	}

	row := make([]float32, 13)
	l.Row(row, 4)

	if !slices.Equal(row, w[4*13:5*13]) {
		t.Error("Row 4 differs from the values")
	}

	v := l.Rows(2, 5)
	sub := make([]float32, 3*3)
	v.Apply(sub, x, 3)
	tensor.MatMulTransB(want[:9], 3, x, w[2*13:5*13], 3, 3, 13)

	if v.Out() != 3 || !slices.Equal(sub, want[:9]) {
		t.Error("Rows(2, 5) differs from the slice of the values")
	}

	if (Linear{}).Out() != 0 {
		t.Error("the zero Linear has rows")
	}

	for name, f := range map[string]func(){
		"NewLinear": func() { NewLinear(w, 7, 12) },
		"Rows":      func() { l.Rows(5, 8) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", name)
				}
			}()

			f()
		}()
	}
}

// openWhistlePair opens whistle.cact packed and in float mode, or skips.
func openWhistlePair(t *testing.T) (packed, float weights.Source) {
	t.Helper()

	float = openModel(t, whistleCactPath)

	packed, err := weights.Open(whistleCactPath, weights.Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = packed.Close() })

	return packed, float
}

// TestWhistleLinearPacked loads real CQ2 and CQ4 tensors both ways
// (docs/FINDINGS.md F-037): packed rows equal the float32 ones bit for
// bit, and products agree at 1 − cos ≤ 2e-7 per output row (measured
// 1.02e-7 at worst, on the embedding; F-034 had 7.8e-8 on other inputs).
func TestWhistleLinearPacked(t *testing.T) {
	packed, float := openWhistlePair(t)
	rng := rand.New(rand.NewPCG(3, 4))

	for _, name := range []string{
		"layer00.q_proj", "enc03.pw1", "cross05.v_proj", "stem.out", "embedding", "mhc_phi_res", "engram1.value_proj",
	} {
		p, err := LoadLinear(packed, name, 0, 0)
		if err != nil {
			t.Fatal(err)
		}

		f, err := LoadLinear(float, name, p.Out(), p.In())
		if err != nil {
			t.Fatal(err)
		}

		if !p.Packed() || f.Packed() {
			t.Fatalf("%s: packed %v / %v", name, p.Packed(), f.Packed())
		}

		if name == "mhc_phi_res" { // layer 5's φ_res, as LoadMHC takes it
			p, f = p.Rows(80, 96), f.Rows(80, 96)
		}

		const rows = 5

		x := randVec(rng, rows*p.In(), 1)
		yp, yf := make([]float32, rows*p.Out()), make([]float32, rows*p.Out())
		p.Apply(yp, x, rows)
		f.Apply(yf, x, rows)

		worst := 0.0
		for r := range rows {
			worst = max(worst, 1-cos32(yp[r*p.Out():(r+1)*p.Out()], yf[r*p.Out():(r+1)*p.Out()]))
		}

		t.Logf("%-18s [%5d, %4d]  worst 1 − cos %.2e", name, p.Out(), p.In(), worst)

		if worst > 2e-7 {
			t.Errorf("%s: 1 − cos %.2e > 2e-7", name, worst)
		}

		rp, rf := make([]float32, p.In()), make([]float32, p.In())
		for _, r := range []int{0, p.Out() - 1} {
			p.Row(rp, r)
			f.Row(rf, r)

			if !slices.Equal(rp, rf) {
				t.Errorf("%s: row %d differs", name, r)
			}
		}
	}
}

// TestWhistleModulesPacked runs the modules with a CQ site, the encoder
// attention, a decoder mHC layer with its stacked φ and an engram site
// with its table gathers, on packed and float weights: the outputs agree
// at 1 − cos ≤ 2e-7 (measured ≤ 6.1e-8 with the lut kernel).
func TestWhistleModulesPacked(t *testing.T) {
	packed, float := openWhistlePair(t)
	rng := rand.New(rand.NewPCG(5, 6))
	cfg := packed.Config()

	type run func(src weights.Source) ([]float32, error)

	x := randVec(rng, 40*512, 1)
	stream := randVec(rng, 40*4*512, 1)
	tokens := make([]int, 40)

	for i := range tokens {
		tokens[i] = rng.IntN(8192)
	}

	for _, c := range []struct {
		name string
		run  run
	}{
		{"enc03 attention", func(src weights.Source) ([]float32, error) {
			a, err := LoadAttention(src, "enc03.")
			if err != nil {
				return nil, err
			}

			return a.Forward(x, 40, MaskNone, nil)
		}},
		{"decoder mHC layer 5", func(src weights.Source) ([]float32, error) {
			m, err := LoadMHC(src, "", 5)
			if err != nil {
				return nil, err
			}

			return m.Forward(stream, 40, func(u []float32) ([]float32, error) {
				out := make([]float32, len(u))
				for i, v := range u {
					out[i] = 2 * v
				}

				return out, nil
			})
		}},
		{"engram1", func(src weights.Source) ([]float32, error) {
			e, err := LoadEngram(src, "engram1.", cfg.Orders, cfg.Tables)
			if err != nil {
				return nil, err
			}

			kv, err := e.Forward(tokens)
			if err != nil {
				return nil, err
			}

			return append(kv.K, kv.V...), nil
		}},
	} {
		yp, err := c.run(packed)
		if err != nil {
			t.Fatal(err)
		}

		yf, err := c.run(float)
		if err != nil {
			t.Fatal(err)
		}

		d := 1 - cos32(yp, yf)
		t.Logf("%-20s 1 − cos %.2e", c.name, d)

		if d > 2e-7 {
			t.Errorf("%s: 1 − cos %.2e > 2e-7", c.name, d)
		}
	}
}

func cos32(a, b []float32) float64 {
	var ab, aa, bb float64

	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}

	return ab / math.Sqrt(aa*bb)
}
