package needle

import (
	"fmt"

	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// HadamardMLP is architecture.py HadamardMLP, the Monarch FFN: three
// Kronecker-factored mixes of n = na·nb (Whistle: 16 × 32 = 512) with
// diagonal scales, a SiLU after the first and the permutations hada_p1 and
// hada_p2 after the first two, and a rank-r conditional scale (r = 8):
//
//	cond = 1 + softmax(x @ cond_v) @ cond_u
//	z1   = kron(d1 · x; w1a, w1b)[p1]
//	z2   = kron(silu(d2 · cond · z1 + b2); w2a, w2b)[p2]
//	out  = d4 · kron(d3 · z2; w3a, w3b)
//
// Whistle's model width is n, so upstream's zero padding to the next power
// of two never applies; LoadHadamardMLP takes the width from d1
// (docs/FINDINGS.md F-011).
type HadamardMLP struct {
	n, na, nb, rank int

	d1, d2, b2, d3, d4 []float32       // (n)
	w                  [3][2][]float32 // stage s: a (na, na), b (nb, nb)
	wT                 [3][2][]float32 // the factors transposed, for kronApply
	condVT             []float32       // cond_v transposed to (rank, n) for linear
	condU              []float32       // (rank, n)
	perm               [2][]int        // hada_p1, hada_p2
}

// LoadHadamardMLP loads the block with the names prefix+"d1", …
// ("layer00." for the decoder's layer 0, "enc00.hada_0." and "enc00.hada."
// for the encoder's two) and the shared permutations hada_p1/p2. The .cact
// backend serves those from records 226/227, the safetensors backend from
// its numpy MT19937 port (docs/FINDINGS.md F-006, F-007).
func LoadHadamardMLP(src weights.Source, prefix string) (*HadamardMLP, error) {
	l := &loader{src: src, prefix: prefix}
	h := &HadamardMLP{}

	h.d1 = l.vector("d1", 0)
	h.n = len(h.d1)
	h.d2 = l.vector("d2", h.n)
	h.b2 = l.vector("b2", h.n)
	h.d3 = l.vector("d3", h.n)
	h.d4 = l.vector("d4", h.n)
	h.loadFactors(l)

	condV, _, rank := l.matrix("cond_v", h.n, 0)
	h.rank = rank
	h.condVT = make([]float32, len(condV))

	for i := range len(condV) / max(rank, 1) {
		for k := range rank {
			h.condVT[k*h.n+i] = condV[i*rank+k]
		}
	}

	h.condU, _, _ = l.matrix("cond_u", rank, h.n)

	shared := &loader{src: src}
	h.perm[0] = shared.perm("hada_p1", h.n)
	h.perm[1] = shared.perm("hada_p2", h.n)

	for _, err := range []error{l.err, shared.err} {
		if err != nil {
			return nil, err
		}
	}

	return h, nil
}

// Dim is the model width of the input and the output.
func (h *HadamardMLP) Dim() int { return h.n }

// Forward runs the block on x, t rows of Dim values, and returns t rows of
// Dim values.
func (h *HadamardMLP) Forward(x []float32, t int) ([]float32, error) {
	return h.forward(x, t, nil)
}

// loadFactors loads w1a … w3b: square, the a factors na × na and the b
// factors nb × nb as w1a and w1b, with na·nb = n.
func (h *HadamardMLP) loadFactors(l *loader) {
	var cols int

	h.w[0][0], h.na, cols = l.matrix("w1a", 0, 0)
	if l.err == nil && cols != h.na {
		l.failf("%sw1a is %d × %d, want square", l.prefix, h.na, cols)
	}

	h.w[0][1], h.nb, cols = l.matrix("w1b", 0, 0)
	if l.err == nil && cols != h.nb {
		l.failf("%sw1b is %d × %d, want square", l.prefix, h.nb, cols)
	}

	if l.err == nil && h.na*h.nb != h.n {
		l.failf("%sw1a/w1b blocks %d × %d do not tile width %d", l.prefix, h.na, h.nb, h.n)
	}

	for s := 1; s < 3; s++ {
		h.w[s][0], _, _ = l.matrix(fmt.Sprintf("w%da", s+1), h.na, h.na)
		h.w[s][1], _, _ = l.matrix(fmt.Sprintf("w%db", s+1), h.nb, h.nb)
	}

	if l.err != nil {
		return
	}

	for s := range h.w {
		h.wT[s][0], h.wT[s][1] = transposed(h.w[s][0], h.na), transposed(h.w[s][1], h.nb)
	}
}

// transposed returns the transpose of the n × n matrix m.
func transposed(m []float32, n int) []float32 {
	t := make([]float32, n*n)

	for i := range n {
		for j := range n {
			t[j*n+i] = m[i*n+j]
		}
	}

	return t
}

// perm reads a permutation of n stored as float32 and checks it.
func (l *loader) perm(name string, n int) []int {
	v := l.vector(name, n)
	if v == nil {
		return nil
	}

	p := make([]int, n)
	seen := make([]bool, n)

	for i, f := range v {
		k := int(f)
		if float32(k) != f || k < 0 || k >= n || seen[k] {
			l.failf("%s%s is not a permutation of %d (entry %d = %v)", l.prefix, name, n, i, f)

			return nil
		}

		seen[k] = true
		p[i] = k
	}

	return p
}

// hadaTrace receives copies of the intermediates for the parity tests.
type hadaTrace struct {
	condLogits, cond, z1, a2, z2, z3 []float32
}

func (h *HadamardMLP) forward(x []float32, t int, tr *hadaTrace) ([]float32, error) {
	if t < 0 || len(x) != t*h.n {
		return nil, fmt.Errorf("needle: HadamardMLP input has %d values, want %d rows of %d", len(x), t, h.n)
	}

	n := h.n
	logits := make([]float32, t*h.rank)
	linear(logits, x, t, h.condVT, h.rank, n)

	if tr != nil {
		tr.condLogits = append([]float32(nil), logits...)
	}

	out := make([]float32, t*n)

	// Rows are independent: the trace collects them in order, so only an
	// untraced run splits them (docs/FINDINGS.md F-038).
	if tr != nil {
		h.rows(x, out, logits, 0, t, tr)
	} else {
		forRows(t, hadamardRows, func(lo, hi int) { h.rows(x, out, logits, lo, hi, nil) })
	}

	return out, nil
}

// rows runs forward's rows [lo, hi) of x into out, with the conditional
// logits of all rows in logits.
func (h *HadamardMLP) rows(x, out, logits []float32, lo, hi int, tr *hadaTrace) {
	n := h.n
	cond, z, y, k := make([]float32, n), make([]float32, n), make([]float32, n), make([]float32, n)

	for i := lo; i < hi; i++ {
		xi, oi := x[i*n:(i+1)*n], out[i*n:(i+1)*n]

		// cond = 1 + softmax(x @ cond_v) @ cond_u
		s := logits[i*h.rank : (i+1)*h.rank]
		softmaxInPlace(s)
		clear(cond)

		for r, p := range s {
			tensor.Axpy(cond, p, h.condU[r*n:(r+1)*n])
		}

		for c := range cond {
			cond[c]++
		}

		// z1 = kron(d1 · x; w1a, w1b)[p1]
		for c, d := range h.d1 {
			z[c] = d * xi[c]
		}

		h.mix(oi, z, 0, y, k)

		if tr != nil {
			tr.cond = append(tr.cond, cond...)
			tr.z1 = append(tr.z1, oi...)
		}

		// z2 = kron(silu(d2 · cond · z1 + b2); w2a, w2b)[p2]
		for c := range z {
			a := h.d2[c]*cond[c]*oi[c] + h.b2[c]
			if tr != nil {
				tr.a2 = append(tr.a2, a)
			}

			z[c] = a * sigmoid(a)
		}

		h.mix(oi, z, 1, y, k)

		if tr != nil {
			tr.z2 = append(tr.z2, oi...)
		}

		// out = d4 · kron(d3 · z2; w3a, w3b)
		for c, d := range h.d3 {
			z[c] = d * oi[c]
		}

		h.mix(oi, z, 2, y, k)

		if tr != nil {
			tr.z3 = append(tr.z3, oi...)
		}

		for c, d := range h.d4 {
			oi[c] *= d
		}
	}
}

// mix writes stage s's Kronecker apply of z to dst, gathered through the
// stage's permutation for the first two stages (dst[i] = kron(z)[p[i]]).
// y and k are n-value scratch buffers distinct from dst and z.
func (h *HadamardMLP) mix(dst, z []float32, s int, y, k []float32) {
	aT, bT := h.wT[s][0], h.wT[s][1]
	if s == 2 {
		kronApply(dst, z, aT, h.na, bT, h.nb, y)

		return
	}

	kronApply(k, z, aT, h.na, bT, h.nb, y)

	for i, p := range h.perm[s] {
		dst[i] = k[p]
	}
}

// kronApply is architecture.py _kron_apply for one vector: with
// Z = z.reshape(na, nb), out[k, l] = Σ_ij Z[i, j] · a[i, k] · b[j, l]
// (einsum "ij,ik,jl->kl"), i.e. out = aᵀ · Z · b. That is z · (a ⊗ b),
// computed in na·nb² + na²·nb multiply-adds instead of (na·nb)², as two
// GEMMs on the transposed factors aT and bT, so that both contract over
// contiguous rows: Yᵀ = bᵀ · Zᵀ, then out = aᵀ · Y. yT is scratch for Yᵀ
// (nb × na); dst, z and yT must not overlap. The GEMM sums each output in
// its own order, not the einsum's (docs/FINDINGS.md F-035).
func kronApply(dst, z, aT []float32, na int, bT []float32, nb int, yT []float32) {
	tensor.MatMulTransB(yT, na, bT, z, nb, na, nb)
	tensor.MatMulTransB(dst, nb, aT, yT, na, nb, na)
}
