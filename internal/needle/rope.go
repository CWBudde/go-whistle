package needle

import (
	"fmt"
	"math"
)

// Rope holds architecture.py precompute_rope_freqs tables for the
// half-split rotation apply_rope uses. ops.RoPE rotates interleaved pairs,
// so it does not fit (docs/FINDINGS.md F-008, F-011).
type Rope struct {
	cos, sin []float32 // (n, half): angle t / θ^(2i/headDim)
	half, n  int
}

// NewRope builds the tables for headDim (even) and positions 0 … seqLen−1
// with base theta (Whistle's decoder: 1e5). Each step rounds to float32 as
// upstream's float32 jnp code does.
func NewRope(headDim, seqLen int, theta float64) *Rope {
	if headDim <= 0 || headDim%2 != 0 || seqLen < 0 {
		panic(fmt.Sprintf("needle: NewRope(%d, %d)", headDim, seqLen))
	}

	half := headDim / 2
	r := &Rope{cos: make([]float32, seqLen*half), sin: make([]float32, seqLen*half), half: half, n: seqLen}

	freqs := make([]float32, half)
	for i := range freqs {
		e := float32(2*i) / float32(headDim)
		freqs[i] = 1 / float32(math.Pow(theta, float64(e)))
	}

	for t := range seqLen {
		for i, f := range freqs {
			angle := float64(float32(t) * f)
			r.cos[t*half+i] = float32(math.Cos(angle))
			r.sin[t*half+i] = float32(math.Sin(angle))
		}
	}

	return r
}

// HeadDim is the head width the tables rotate.
func (r *Rope) HeadDim() int { return 2 * r.half }

// Len is the number of positions the tables cover.
func (r *Rope) Len() int { return r.n }

// apply rotates one head vector x (HeadDim values) at position pos in
// place: with x1, x2 its first and last halves, [x1·cos − x2·sin,
// x2·cos + x1·sin].
func (r *Rope) apply(x []float32, pos int) {
	h := r.half
	cs, sn := r.cos[pos*h:(pos+1)*h], r.sin[pos*h:(pos+1)*h]
	x1, x2 := x[:h], x[h:2*h]

	for i := range h {
		a, b := x1[i], x2[i]
		x1[i] = a*cs[i] - b*sn[i]
		x2[i] = b*cs[i] + a*sn[i]
	}
}
