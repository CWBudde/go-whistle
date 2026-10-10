package needle

import (
	"fmt"

	"github.com/cwbudde/go-whistle/internal/cq"
	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Linear is a weight matrix [out, in] in the .cact layout of a Flax Dense
// kernel, for y = x · Wᵀ: float32 values, or a CQ matrix that multiplies
// straight from its packed indices (internal/cq, docs/FINDINGS.md F-037).
// Both give the same products up to the order of the float32 sums and the
// lut kernel's table rounding (1 − cos ≤ 7.8e-8 on Whistle's tensors,
// F-034). The zero Linear has no rows. It is safe for concurrent use.
type Linear struct {
	out, in int
	w       []float32  // [out, in]; nil when packed
	q       *cq.Matrix // nil for float32
}

// NewLinear wraps float32 values w, out rows of in, without copying them.
func NewLinear(w []float32, out, in int) Linear {
	if len(w) != out*in {
		panic(fmt.Sprintf("needle: %d values for a [%d, %d] linear", len(w), out, in))
	}

	return Linear{out: out, in: in, w: w}
}

// LoadLinear reads the rank-2 tensor name, which must have out rows and in
// columns where those are positive. A packed CQ tensor
// (weights.PackedMatrix) stays packed and is prepared for products; any
// other is read as float32 with Tensor.
func LoadLinear(src weights.Source, name string, out, in int) (Linear, error) {
	l := &loader{src: src}
	w := l.linear(name, out, in)

	return w, l.err
}

// Out is the number of rows, the output width.
func (l Linear) Out() int { return l.out }

// In is the number of columns, the input width.
func (l Linear) In() int { return l.in }

// Packed reports whether the matrix multiplies from packed CQ indices.
func (l Linear) Packed() bool { return l.q != nil }

// Apply sets dst = x · Wᵀ for each of the rows of x: x holds rows × In
// values and dst rows × Out, both row-major.
func (l Linear) Apply(dst, x []float32, rows int) {
	if l.q != nil {
		l.q.GEMM(dst[:rows*l.out], x[:rows*l.in], rows)

		return
	}

	tensor.MatMulTransB(dst, l.out, x, l.w, rows, l.out, l.in)
}

// Row writes row r of W to dst (len In): the values for float32, and the
// row dequantised as cact.CQ does it, bit for bit, when packed. It serves
// the gathers: the token embedding and the engram tables.
func (l Linear) Row(dst []float32, r int) {
	if l.q != nil {
		l.q.Row(dst, r)

		return
	}

	copy(dst, l.w[r*l.in:(r+1)*l.in])
}

// Rows returns rows [lo, hi) of W as a Linear sharing its storage: one
// layer's mHC φ from the stacked tensor (docs/FINDINGS.md F-028).
func (l Linear) Rows(lo, hi int) Linear {
	if lo < 0 || hi < lo || hi > l.out {
		panic(fmt.Sprintf("needle: Rows(%d, %d) of %d rows", lo, hi, l.out))
	}

	if l.q != nil {
		return Linear{out: hi - lo, in: l.in, q: l.q.Rows(lo, hi)}
	}

	return Linear{out: hi - lo, in: l.in, w: l.w[lo*l.in : hi*l.in : hi*l.in]}
}

// linear reads a rank-2 tensor as a Linear with the shape checks of
// matrix, packed where the source serves it packed (LoadLinear). Its
// product is prepared at load time.
func (l *loader) linear(name string, rows, cols int) Linear {
	w := l.table(name, rows, cols)
	if w.q != nil {
		w.q.Prepare()
	}

	return w
}

// table is linear for a matrix that is read by rows (Row) only, such as
// an engram table: a packed one gets no lut copy unless a product needs
// it (cq.NewGather).
func (l *loader) table(name string, rows, cols int) Linear {
	if l.err != nil {
		return Linear{}
	}

	q, ok, err := weights.PackedMatrix(l.src, l.prefix+name)
	if err != nil {
		l.failf("%s%s: %w", l.prefix, name, err)

		return Linear{}
	}

	if !ok {
		data, out, in := l.matrix(name, rows, cols)
		if data == nil {
			return Linear{}
		}

		return NewLinear(data, out, in)
	}

	if (rows > 0 && q.Out() != rows) || (cols > 0 && q.In() != cols) {
		l.failf("%s%s has shape (%d, %d), want (%d, %d)", l.prefix, name, q.Out(), q.In(), rows, cols)

		return Linear{}
	}

	return Linear{out: q.Out(), in: q.In(), q: q}
}
