package needle

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// expFlush is where the kernel returns 0: below e^−87.33 (exp.go).
const expFlush = 1.2e-38

// expLengths covers every tail of the 4- and 8-wide kernels and the
// lengths attend and HadamardMLP pass (8 ranks, 375 keys).
func expLengths() []int {
	ns := make([]int, 0, 80)
	for n := range 70 {
		ns = append(ns, n)
	}

	return append(ns, 127, 128, 129, 375, 1000, 4099)
}

// checkExpSubSum compares expSubSumF32, the kernel where there is one,
// with the float64 fallback on s: every value within 2 ulps (2·2⁻²³
// relative) or both below expFlush, and the sums within the two float32
// summation bounds plus the values' own error.
func checkExpSubSum(t *testing.T, name string, s []float32, m float32) float64 {
	t.Helper()

	var worst float64

	got, want := slices.Clone(s), slices.Clone(s)
	gotSum := expSubSumF32(got, m)
	wantSum := expSubSumGeneric(want, m)

	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("%s: exp(%v − %v) = %v, want NaN", name, s[i], m, g)
			}

			continue
		}

		if w > expFlush {
			worst = max(worst, math.Abs(g-w)/(0x1p-23*w))
		}

		if math.Abs(g-w) > 2*0x1p-23*w && (w > expFlush || g > expFlush) {
			t.Fatalf("%s: exp(%v − %v) = %v, generic %v (%.2f ulps)", name, s[i], m, g, w,
				math.Abs(g-w)/(0x1p-23*w))
		}
	}

	if math.IsNaN(float64(wantSum)) {
		if !math.IsNaN(float64(gotSum)) {
			t.Fatalf("%s: sum %v, want NaN", name, gotSum)
		}

		return worst
	}

	n := float64(len(s))
	if bound := (2*(n+1)*0x1p-24+2*0x1p-23)*float64(wantSum) + n*expFlush; math.Abs(float64(gotSum-wantSum)) > bound {
		t.Fatalf("%s: sum %v, generic %v (|Δ| %.3g > %.3g)", name, gotSum, wantSum, math.Abs(float64(gotSum-wantSum)),
			bound)
	}

	return worst
}

// TestExpSubSumMatchesGeneric runs the kernel on softmax rows of every
// length at offsets 0–3, with logits spread from 0.1 to 100, so the
// exponents reach past the flush point, and on rows with −Inf and NaN.
func TestExpSubSumMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))

	for _, n := range expLengths() {
		for off := range 4 {
			for _, scale := range []float64{0.1, 3, 30, 100} {
				s := randVec(r, n+off, scale)[off:]

				m := float32(math.Inf(-1))
				for _, v := range s {
					m = max(m, v)
				}

				checkExpSubSum(t, "random", s, m)
			}
		}
	}

	inf, nan := float32(math.Inf(-1)), float32(math.NaN())
	row := []float32{0, -1, inf, -2, -87.3, -87.4, -103, -200, 3, nan, -0.5, 0}
	checkExpSubSum(t, "special", row, 3)
	checkExpSubSum(t, "special, no NaN", slices.Delete(slices.Clone(row), 9, 10), 3)

	// The kernel's own lanes: −Inf and the values below e^−87.33 are 0.
	lanes := []float32{0, -90, inf, -87.3, -1, -2, -3, -4}
	expSubSumF32(lanes, 0)

	if lanes[1] > expFlush || lanes[2] != 0 || lanes[3] < expFlush {
		t.Errorf("flush: exp of 0, −90, −Inf, −87.3 = %v", lanes[:4])
	}
}

// TestExpSubSumSweep checks every exponent on a fine grid of [−90, 0],
// the kernel's range, against math.Exp.
func TestExpSubSumSweep(t *testing.T) {
	const steps = 1 << 16

	s := make([]float32, steps)
	for i := range s {
		s[i] = -90 * float32(i) / steps
	}

	t.Logf("worst error above the flush point: %.2f ulps", checkExpSubSum(t, "sweep", s, 0))

	exact := []float32{0, 0, 0, 0, 0, 0, 0, 0}
	if sum := expSubSumF32(exact, 0); sum != 8 || slices.ContainsFunc(exact, func(v float32) bool { return v != 1 }) {
		t.Errorf("exp(0) gives %v, sum %v; want 1 and 8", exact, sum)
	}
}

func BenchmarkExpSubSum(b *testing.B) {
	s := randVec(rand.New(rand.NewPCG(7, 7)), 375, 2)
	w := make([]float32, len(s))
	m := slices.Max(s)

	b.Run("kernel", func(b *testing.B) {
		for b.Loop() {
			copy(w, s)
			expSubSumF32(w, m)
		}
	})

	b.Run("generic", func(b *testing.B) {
		for b.Loop() {
			copy(w, s)
			expSubSumGeneric(w, m)
		}
	})
}
