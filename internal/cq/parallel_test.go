package cq

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// withWorkers runs fn with tensor.SetWorkers(n) and restores the count.
func withWorkers(n int, fn func()) {
	prev := tensor.Workers()
	defer tensor.SetWorkers(prev)

	tensor.SetWorkers(n)
	fn()
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// parallelShapes are Whistle's shapes where the split runs: the encoder's
// batched products (over activations), the tied head for one row and for
// five (over blocks of rows), and odd sizes with partial blocks and
// groups.
var parallelShapes = []struct {
	out, in, group, bits, rows int
}{
	{384, 512, 128, 2, 375},  // q_proj over 375 frames
	{1024, 512, 128, 2, 101}, // pw1
	{128, 2048, 128, 4, 375}, // φ_res
	{8199, 512, 128, 4, 1},   // head, one beam
	{8199, 512, 128, 4, 5},   // head, five rows
	{4133, 300, 64, 2, 1},    // partial last block and group
	{1031, 200, 32, 4, 37},
}

// TestGEMMWorkers checks that every kernel gives the same bits on one
// worker as on several, also through a Rows view that starts and ends
// inside a block, and that the lut kernel equals its unsplit Go reference
// (docs/FINDINGS.md F-038).
func TestGEMMWorkers(t *testing.T) {
	rng := rand.New(rand.NewPCG(38, 1))

	for _, sh := range parallelShapes {
		m, err := New(randomPacked(rng, sh.out, sh.in, sh.group, sh.bits))
		if err != nil {
			t.Fatal(err)
		}

		x := make([]float32, sh.rows*sh.in)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}

		for _, v := range []*Matrix{m, m.Rows(5, sh.out-7)} {
			for _, k := range kernels {
				if k == kernelLUT && v.best() != kernelLUT {
					continue // the lut kernel's Go reference without assembly: asm_test covers it
				}

				if k == kernelGather && sh.out*sh.rows > 1<<16 {
					continue // the reference is too slow for the large shapes
				}

				var want []float32

				if k == kernelLUT { // the single-threaded kernel before the split, in Go
					want = make([]float32, sh.rows*v.Out())
					lutGEMMGeneric(v, want, x, sh.rows)
				}

				for _, w := range []int{1, 2, 5, 16} {
					got := make([]float32, sh.rows*v.Out())

					withWorkers(w, func() { v.gemm(k, got, x, sh.rows) })

					if want == nil {
						want = got

						continue
					}

					if !sameBits(got, want) {
						t.Fatalf("[%d, %d] CQ%d rows %d from row %d, %v: %d workers differ from the reference",
							v.Out(), sh.in, sh.bits, sh.rows, v.row0, k, w)
					}
				}
			}
		}
	}
}

// TestGEMMsMatchesGEMM checks that GEMMs equals a GEMM per matrix bit for
// bit: matrices that share the tables (q, k, v and gate; the φ as Rows
// views of one stack), a mix that does not (another codebook), and no
// rows, on one worker and on several.
func TestGEMMsMatchesGEMM(t *testing.T) {
	rng := rand.New(rand.NewPCG(38, 2))

	cq2 := func(out int) *Matrix {
		m, err := New(randomPacked(rng, out, 512, 128, 2))
		if err != nil {
			t.Fatal(err)
		}

		return m
	}

	q, k, v, g := cq2(384), cq2(96), cq2(96), cq2(384)
	for _, m := range []*Matrix{k, v, g} {
		m.cb, m.cbMax = q.cb, q.cbMax // one codebook per width, as in .cact
	}

	other := cq2(64) // its own codebook

	phi, err := New(randomPacked(rng, 3*24, 2048, 128, 4))
	if err != nil {
		t.Fatal(err)
	}

	if lutAsm && !q.sharesTables(k) || q.sharesTables(other) {
		t.Fatal("sharesTables: q and k share one codebook, other has its own")
	}

	sets := map[string][]*Matrix{
		"qkvg":  {q, k, v, g},
		"phi":   {phi.Rows(24, 28), phi.Rows(28, 32), phi.Rows(32, 48)},
		"mixed": {q, other, k},
		"one":   {q},
	}

	for name, ms := range sets {
		for _, rows := range []int{0, 1, 3, 375} {
			x := make([]float32, rows*ms[0].In())
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}

			want := make([][]float32, len(ms))

			withWorkers(1, func() {
				for i, m := range ms {
					want[i] = make([]float32, rows*m.Out())
					m.GEMM(want[i], x, rows)
				}
			})

			for _, w := range []int{1, 4, 16} {
				got := make([][]float32, len(ms))
				for i, m := range ms {
					got[i] = make([]float32, rows*m.Out())
				}

				withWorkers(w, func() { GEMMs(ms, got, x, rows) })

				for i := range ms {
					if !sameBits(got[i], want[i]) {
						t.Fatalf("%s, %d rows, %d workers: matrix %d differs from its GEMM", name, rows, w, i)
					}
				}
			}
		}
	}
}

func TestGEMMsPanics(t *testing.T) {
	rng := rand.New(rand.NewPCG(38, 3))

	m, err := New(randomPacked(rng, 32, 128, 128, 2))
	if err != nil {
		t.Fatal(err)
	}

	for name, call := range map[string]func(){
		"outputs": func() { GEMMs([]*Matrix{m}, nil, make([]float32, 128), 1) },
		"short":   func() { GEMMs([]*Matrix{m}, [][]float32{make([]float32, 31)}, make([]float32, 128), 1) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", name)
				}
			}()

			call()
		}()
	}
}
