package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The tests in this file compare each dispatching kernel (dotF32, axpyF32,
// matMulTransB) with its pure-Go fallback, as AGENTS.md requires for every
// assembly routine. With the purego build tag, or on an architecture without
// assembly, both sides are the fallback and the tests pass trivially; on
// arm64 (NEON) and amd64 with AVX2+FMA they exercise the assembly.

// unitRoundoff is the float32 unit roundoff 2^-24.
const unitRoundoff = 1.0 / (1 << 24)

// asmLengths covers both sides of every SIMD cutover (8, 48), every tail of
// the 4-, 8- and 16-wide loops, and a few long vectors.
func asmLengths() []int {
	ns := make([]int, 0, 136)
	for n := range 130 {
		ns = append(ns, n)
	}

	return append(ns, 255, 256, 257, 1023, 1024, 4099)
}

// TestDotF32MatchesGeneric: the SIMD dot product sums in a different order,
// so it may differ from the fallback by at most the two summation error
// bounds, 2·γ_n·Σ|a_i·b_i| with γ_n ≈ n·u. Offsets 1–3 start the operands
// off any vector alignment.
func TestDotF32MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))

	for _, n := range asmLengths() {
		for off := range 4 {
			a := randGemmData(rng, n+off)[off:]
			b := randGemmData(rng, n+off)[off:]

			var mag float64
			for i := range n {
				mag += math.Abs(float64(a[i]) * float64(b[i]))
			}

			got := float64(dotF32(a, b))
			want := float64(dotF32Generic(a, b))

			if bound := 2*float64(n+1)*unitRoundoff*mag + 1e-30; math.Abs(got-want) > bound {
				t.Fatalf("n=%d off=%d: dotF32 = %v, generic %v (|diff| %g > %g)", n, off, got, want, math.Abs(got-want), bound)
			}
		}
	}
}

// TestAxpyF32MatchesGeneric: each element is one multiply-add, so a fused
// (FMA) kernel and the fallback differ by at most one rounding of the
// product and one of the sum. The slack after n must stay untouched.
func TestAxpyF32MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))

	const slack = 3

	for _, n := range asmLengths() {
		for off := range 4 {
			src := randGemmData(rng, n+off)[off:]
			base := randGemmData(rng, n+off+slack)[off:]
			alpha := rng.Float32()*4 - 2

			got := append([]float32(nil), base...)
			want := append([]float32(nil), base...)

			if n > 0 {
				axpyF32(got[:n], alpha, src)
				axpyF32Generic(want[:n], alpha, src)
			}

			for i := range n {
				prod := math.Abs(float64(alpha) * float64(src[i]))
				bound := 2*unitRoundoff*(prod+math.Abs(float64(base[i]))+math.Abs(float64(want[i]))) + 1e-30

				if d := math.Abs(float64(got[i]) - float64(want[i])); d > bound {
					t.Fatalf("n=%d off=%d: dst[%d] = %v, generic %v (|diff| %g > %g)", n, off, i, got[i], want[i], d, bound)
				}
			}

			for i := n; i < len(got); i++ {
				if got[i] != base[i] {
					t.Fatalf("n=%d off=%d: slack dst[%d] = %v was overwritten (was %v)", n, off, i, got[i], base[i])
				}
			}
		}
	}
}

// TestMatMulTransBMatchesGeneric compares the blocked kernel with the
// per-output fallback element by element, under the dot-product bound, and
// checks that the gaps between output rows stay untouched.
func TestMatMulTransBMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))

	for _, k := range []int{1, 2, 3, 4, 5, 8, 12, 13, 48, 64, 67, 512, 1031} {
		for _, mn := range [][2]int{{1, 1}, {1, 4}, {1, 7}, {4, 4}, {5, 9}, {8, 12}, {13, 3}, {6, 70}} {
			m, n := mn[0], mn[1]
			a := randGemmData(rng, m*k)
			b := randGemmData(rng, n*k)
			ldc := n + 2

			got := make([]float32, m*ldc)
			want := make([]float32, m*ldc)

			for i := range got {
				got[i] = gemmSentinel
				want[i] = gemmSentinel
			}

			matMulTransB(got, ldc, a, b, m, n, k)
			matMulTransBGeneric(want, ldc, a, b, m, n, k)

			for i := range m {
				for j := range n {
					var mag float64
					for p := range k {
						mag += math.Abs(float64(a[i*k+p]) * float64(b[j*k+p]))
					}

					g, w := float64(got[i*ldc+j]), float64(want[i*ldc+j])
					if bound := 2*float64(k+1)*unitRoundoff*mag + 1e-30; math.Abs(g-w) > bound {
						t.Fatalf("m=%d n=%d k=%d: c[%d][%d] = %v, generic %v (|diff| %g > %g)", m, n, k, i, j, g, w, math.Abs(g-w), bound)
					}
				}

				for g := i*ldc + n; g < (i+1)*ldc; g++ {
					if got[g] != gemmSentinel {
						t.Fatalf("m=%d n=%d k=%d: gap c[%d] = %v was overwritten", m, n, k, g, got[g])
					}
				}
			}
		}
	}
}
