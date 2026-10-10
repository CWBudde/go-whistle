package cq

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/go-whistle/internal/tensor"
)

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
// dequantised matrix, one goroutine each. The "MB" metric is the matrix's
// resident size, Matrix.Bytes or 4 bytes per weight for float32; it is
// reported after b.Loop, which clears metrics.
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
			for b.Loop() {
				tensor.MatMulTransB(y, s.out, x, w, s.rows, s.out, s.in)
			}

			b.ReportMetric(float64(4*len(w))/1e6, "MB")
		})

		for _, k := range kernels {
			b.Run(prefix+"/"+k.String(), func(b *testing.B) {
				for b.Loop() {
					m.gemm(k, y, x, s.rows)
				}

				b.ReportMetric(float64(m.Bytes())/1e6, "MB")
			})
		}
	}
}
