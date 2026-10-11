package cq

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/cwbudde/go-whistle/internal/tensor"
)

// benchWorkers sets the tensor workers to GOMAXPROCS, which -cpu sets per
// run, for the rest of the (sub-)benchmark: -cpu 1 times one goroutine.
func benchWorkers(b *testing.B) {
	b.Helper()

	prev := tensor.Workers()

	tensor.SetWorkers(runtime.GOMAXPROCS(0))
	b.Cleanup(func() { tensor.SetWorkers(prev) })
}

// benchShapes are Whistle's CQ matrix shapes (F-028). rows is the batch:
// 1 for a decode step, 5 for five beams, 375 for a 30 s encoder pass.
var benchShapes = []struct {
	name          string
	out, in, bits int
	rows          int
}{
	{"k_proj", 96, 512, 2, 1},
	{"q_proj", 384, 512, 2, 1},
	{"out_proj", 512, 512, 2, 1},
	{"pw1", 1024, 512, 2, 1},
	{"stem.out", 512, 1280, 2, 1},
	{"phi_res", 128, 2048, 4, 1},
	{"head", 8199, 512, 4, 1},
	{"q_proj", 384, 512, 2, 5},
	{"head", 8199, 512, 4, 5},
	{"q_proj", 384, 512, 2, 375},
}

// BenchmarkGEMM times the CQ kernels against tensor.MatMulTransB on the
// dequantised matrix, on -cpu workers (F-034's numbers are -cpu 1, F-038's
// also more). The "MB" metric is the matrix's resident size, Matrix.Bytes
// or 4 bytes per weight for float32; it is reported after b.Loop, which
// clears metrics.
func BenchmarkGEMM(b *testing.B) {
	for _, s := range benchShapes {
		rng := rand.New(rand.NewPCG(uint64(s.out), uint64(s.in)))

		m, err := New(randomPacked(rng, s.out, s.in, 128, s.bits))
		if err != nil {
			b.Fatal(err)
		}

		w := make([]float32, s.out*s.in)
		for r := range s.out {
			m.Row(w[r*s.in:(r+1)*s.in], r)
		}

		x := make([]float32, s.rows*s.in)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}

		y := make([]float32, s.rows*s.out)
		prefix := fmt.Sprintf("%s_%dx%d_CQ%d_rows%d", s.name, s.out, s.in, s.bits, s.rows)

		b.Run(prefix+"/float", func(b *testing.B) {
			benchWorkers(b)

			for b.Loop() {
				tensor.MatMulTransB(y, s.out, x, w, s.rows, s.out, s.in)
			}

			b.ReportMetric(float64(4*len(w))/1e6, "MB")
		})

		for _, k := range kernels {
			b.Run(prefix+"/"+k.String(), func(b *testing.B) {
				benchWorkers(b)

				for b.Loop() {
					m.gemm(k, y, x, s.rows)
				}

				b.ReportMetric(float64(m.Bytes())/1e6, "MB")
			})
		}
	}
}

// BenchmarkGEMMs times matrices that read one input, as GEMMs with one
// rotation and table build per activation against a GEMM per matrix
// (F-038): the attention's q, k, v and gate projections and the mHC's
// three φ, for a decode step and a 30 s encoder pass.
func BenchmarkGEMMs(b *testing.B) {
	sets := []struct {
		name     string
		in, bits int
		outs     []int
	}{
		{"qkvg", 512, 2, []int{384, 96, 96, 384}},
		{"phi", 2048, 4, []int{4, 4, 16}},
	}

	for _, set := range sets {
		rng := rand.New(rand.NewPCG(38, uint64(set.in)))
		ms := make([]*Matrix, len(set.outs))

		for k, out := range set.outs {
			m, err := New(randomPacked(rng, out, set.in, 128, set.bits))
			if err != nil {
				b.Fatal(err)
			}

			if k > 0 {
				m.cb, m.cbMax = ms[0].cb, ms[0].cbMax // one codebook per width, as in .cact
			}

			ms[k] = m
		}

		for _, rows := range []int{1, 375} {
			x := make([]float32, rows*set.in)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}

			dsts := make([][]float32, len(ms))
			for k, m := range ms {
				dsts[k] = make([]float32, rows*m.Out())
			}

			b.Run(fmt.Sprintf("%s_rows%d/shared", set.name, rows), func(b *testing.B) {
				benchWorkers(b)

				for b.Loop() {
					GEMMs(ms, dsts, x, rows)
				}
			})

			b.Run(fmt.Sprintf("%s_rows%d/separate", set.name, rows), func(b *testing.B) {
				benchWorkers(b)

				for b.Loop() {
					for k, m := range ms {
						m.GEMM(dsts[k], x, rows)
					}
				}
			})
		}
	}
}
