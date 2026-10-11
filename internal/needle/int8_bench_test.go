package needle

import (
	"math/rand/v2"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
)

// The int8 benchmarks (docs/FINDINGS.md F-041): one row's quantisation at
// the widths Whistle quantises (a query or key head, a value head, the
// mHC's 4 · 512 lanes), and one decoding step's attention of the
// cross-attention's 8 heads over the 375 memory frames of 30 s, float32
// (one dot product and one axpy per frame, cross.go's loop) against
// AttendInt8 on the int8 memory.

func BenchmarkQuantizeRow(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))

	for _, n := range []int{48, 64, 2048} {
		x, q := randVec(r, n, 1), make([]int8, n)

		b.Run(strconv.Itoa(n), func(b *testing.B) {
			b.SetBytes(int64(4 * n))

			for b.Loop() {
				QuantizeRow(q, x)
			}
		})
	}
}

func BenchmarkCrossAttendStep(b *testing.B) {
	const heads, qk, vd, frames = 8, 48, 64, 375

	r := rand.New(rand.NewPCG(3, 4))
	q := randVec(r, heads*qk, 1)
	k, v := randVec(r, frames*heads*qk, 1), randVec(r, frames*heads*vd, 1)
	k8, v8 := QuantizeRows(k, heads*qk, qk), QuantizeRows(v, heads*vd, vd)
	all := func(int) int { return frames }

	b.Run("float32", func(b *testing.B) {
		for b.Loop() {
			attendFloatRef(q, k, v, heads, qk, vd, frames)
		}
	})

	b.Run("int8", func(b *testing.B) {
		for b.Loop() {
			AttendInt8(q, 1, heads, k8, v8, all, 6.928203, true, nil)
		}
	})
}

// attendFloatRef is internal/decoder's cross.attend for one row: a dot
// product per frame for the logits, an axpy per frame for the output.
func attendFloatRef(q, k, v []float32, heads, qk, vd, frames int) []float32 {
	out, s := make([]float32, heads*vd), make([]float32, frames)

	for h := range heads {
		for f := range s {
			s[f] = tensor.DotProduct(q[h*qk:(h+1)*qk], k[(f*heads+h)*qk:(f*heads+h+1)*qk]) / 6.928203
		}

		softmaxInPlace(s)

		for f, p := range s {
			tensor.Axpy(out[h*vd:(h+1)*vd], p, v[(f*heads+h)*vd:(f*heads+h+1)*vd])
		}
	}

	return out
}
