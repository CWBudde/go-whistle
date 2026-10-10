package needle

import (
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Mask selects which keys a query attends to.
type Mask int

const (
	// MaskNone lets every query see every key (the encoder).
	MaskNone Mask = iota
	// MaskCausal lets the query at step t see keys 0 … t (the decoder).
	MaskCausal
)

func (m Mask) String() string {
	switch m {
	case MaskNone:
		return "none"
	case MaskCausal:
		return "causal"
	default:
		return fmt.Sprintf("Mask(%d)", int(m))
	}
}

// Attention is architecture.py MultiHeadAttention on the explicit softmax
// path (flash=False, quant=False): Q/K/V projections, optional causal
// taps, per-head q/k ZCRMSNorm, optional half-split RoPE, softmax(qkᵀ/√qk)
// with GQA, the sigmoid output gate and the output projection. The
// decoder's self-attention has taps and runs causal with RoPE; the
// encoder's has no taps and runs unmasked (docs/FINDINGS.md F-011).
type Attention struct {
	dim, heads, kvHeads, qk, v, taps int

	qProj, kProj, vProj []float32 // [heads·qk | kvHeads·qk | kvHeads·v, dim]
	qTaps, kTaps, vTaps []float32 // [taps, width], nil without taps
	qNorm, kNorm        ZCRMSNorm // (qk)
	gateProj            []float32 // [heads·v, dim]
	outProj             []float32 // [dim, heads·v]
}

// LoadAttention loads the attention with the names prefix+"q_proj", …
// ("layer00." for the decoder's layer 0, "enc00." for the encoder's). The
// head counts and widths come from the shapes: qk from q_norm, the query
// heads from q_proj, the KV heads from k_proj and v from v_proj. The block
// has taps exactly when prefix+"q_taps" exists.
func LoadAttention(src weights.Source, prefix string) (*Attention, error) {
	l := &loader{src: src, prefix: prefix}
	a := &Attention{}

	var qW, kW, vW int

	qNorm := l.vector("q_norm", 0)
	a.qProj, qW, a.dim = l.matrix("q_proj", 0, 0)
	a.kProj, kW, _ = l.matrix("k_proj", 0, a.dim)
	a.vProj, vW, _ = l.matrix("v_proj", 0, a.dim)
	a.qk = len(qNorm)

	if l.err == nil {
		a.setHeads(l, qW, kW, vW)
	}

	a.qNorm = NewZCRMSNorm(qNorm)
	a.kNorm = NewZCRMSNorm(l.vector("k_norm", a.qk))
	a.gateProj, _, _ = l.matrix("gate_proj", a.heads*a.v, a.dim)
	a.outProj, _, _ = l.matrix("out_proj", a.dim, a.heads*a.v)

	if src.Has(prefix + "q_taps") {
		a.qTaps, a.taps, _ = l.matrix("q_taps", 0, qW)
		a.kTaps, _, _ = l.matrix("k_taps", a.taps, kW)
		a.vTaps, _, _ = l.matrix("v_taps", a.taps, vW)
	}

	if l.err != nil {
		return nil, l.err
	}

	return a, nil
}

// Dim is the model width of the input and the output.
func (a *Attention) Dim() int { return a.dim }

// Heads is the number of query heads.
func (a *Attention) Heads() int { return a.heads }

// KVHeads is the number of key/value heads.
func (a *Attention) KVHeads() int { return a.kvHeads }

// QKDim is the query/key head width.
func (a *Attention) QKDim() int { return a.qk }

// VDim is the value head width.
func (a *Attention) VDim() int { return a.v }

// Taps is the number of causal Q/K/V taps, 0 without.
func (a *Attention) Taps() int { return a.taps }

// Forward runs the attention on x, t rows of Dim values, and returns t
// rows of Dim values. rope is nil for no rotary embedding; otherwise its
// head width must be QKDim and it must cover t positions, row t taking
// position t.
func (a *Attention) Forward(x []float32, t int, mask Mask, rope *Rope) ([]float32, error) {
	return a.forward(x, t, mask, rope, nil)
}

// setHeads derives the head counts and the value width from the
// projection widths.
func (a *Attention) setHeads(l *loader, qW, kW, vW int) {
	if a.qk == 0 || kW == 0 || qW%a.qk != 0 || kW%a.qk != 0 {
		l.failf("%sq_proj/k_proj widths %d/%d are not multiples of qk %d", l.prefix, qW, kW, a.qk)

		return
	}

	a.heads, a.kvHeads = qW/a.qk, kW/a.qk
	if a.heads%a.kvHeads != 0 || vW == 0 || vW%a.kvHeads != 0 {
		l.failf("%s: %d query heads, %d KV heads and v width %d do not group", l.prefix, a.heads, a.kvHeads, vW)

		return
	}

	a.v = vW / a.kvHeads
}

// kvHead is the KV head query head h reads. Upstream repeats each KV head
// heads/kvHeads times (jnp.repeat along the head axis), so it is
// h / (heads/kvHeads): heads 0–3 read KV head 0 and 4–7 KV head 1.
func (a *Attention) kvHead(h int) int { return h / (a.heads / a.kvHeads) }

// attnTrace receives copies of the intermediates, in row layout except
// probs (heads, T, T); the parity tests compare them with the fixture.
type attnTrace struct {
	qProj, kProj, vProj       []float32
	qTapped, kTapped, vTapped []float32
	qNormed, kNormed          []float32
	qRope, kRope              []float32
	probs                     []float32
	attn, gate, gated         []float32
}

func (a *Attention) check(x []float32, t int, mask Mask, rope *Rope) error {
	if t < 0 || len(x) != t*a.dim {
		return fmt.Errorf("needle: attention input has %d values, want %d rows of %d", len(x), t, a.dim)
	}

	if mask != MaskNone && mask != MaskCausal {
		return fmt.Errorf("needle: unknown attention mask %v", mask)
	}

	if rope != nil && (rope.HeadDim() != a.qk || rope.Len() < t) {
		return fmt.Errorf("needle: RoPE tables for head width %d and %d positions, want %d and %d",
			rope.HeadDim(), rope.Len(), a.qk, t)
	}

	return nil
}

func (a *Attention) forward(x []float32, t int, mask Mask, rope *Rope, tr *attnTrace) ([]float32, error) {
	err := a.check(x, t, mask, rope)
	if err != nil {
		return nil, err
	}

	q, k, v := a.project(x, t, tr)
	a.normRotate(q, k, t, 0, rope, tr)

	return a.output(x, t, a.attend(q, k, v, t, 0, mask, tr), tr), nil
}

// output is the sigmoid gate σ(gate_proj(x)) on the attention result out,
// t rows of heads·v values, then the output projection.
func (a *Attention) output(x []float32, t int, out []float32, tr *attnTrace) []float32 {
	oW := a.heads * a.v
	gate := make([]float32, t*oW)
	linear(gate, x, t, a.gateProj, oW, a.dim)

	for i, g := range gate {
		gate[i] = sigmoid(g)
		out[i] *= gate[i]
	}

	if tr != nil {
		tr.gate, tr.gated = gate, slices.Clone(out)
	}

	y := make([]float32, t*a.dim)
	linear(y, out, t, a.outProj, a.dim, oW)

	return y
}

// project returns q, k and v in row layout: the projections, then the
// causal taps if the block has them.
func (a *Attention) project(x []float32, t int, tr *attnTrace) (q, k, v []float32) {
	qW, kW, vW := a.heads*a.qk, a.kvHeads*a.qk, a.kvHeads*a.v
	q, k, v = make([]float32, t*qW), make([]float32, t*kW), make([]float32, t*vW)
	linear(q, x, t, a.qProj, qW, a.dim)
	linear(k, x, t, a.kProj, kW, a.dim)
	linear(v, x, t, a.vProj, vW, a.dim)

	if tr != nil {
		tr.qProj, tr.kProj, tr.vProj = slices.Clone(q), slices.Clone(k), slices.Clone(v)
	}

	if a.taps == 0 {
		return q, k, v
	}

	q = applyTaps(q, t, qW, a.qTaps, a.taps, 1)
	k = applyTaps(k, t, kW, a.kTaps, a.taps, 1)
	v = applyTaps(v, t, vW, a.vTaps, a.taps, 1)

	if tr != nil {
		tr.qTapped, tr.kTapped, tr.vTapped = slices.Clone(q), slices.Clone(k), slices.Clone(v)
	}

	return q, k, v
}

// normRotate applies the per-head q/k norms and, if rope is set, the
// rotation at each row's position, off + row, in place.
func (a *Attention) normRotate(q, k []float32, t, off int, rope *Rope, tr *attnTrace) {
	// Every qk-wide chunk of a row is one head.
	a.qNorm.Apply(q, q)
	a.kNorm.Apply(k, k)

	if tr != nil {
		tr.qNormed, tr.kNormed = slices.Clone(q), slices.Clone(k)
	}

	if rope == nil {
		return
	}

	for i := range t {
		for h := range a.heads {
			o := (i*a.heads + h) * a.qk
			rope.apply(q[o:o+a.qk], off+i)
		}

		for h := range a.kvHeads {
			o := (i*a.kvHeads + h) * a.qk
			rope.apply(k[o:o+a.qk], off+i)
		}
	}

	if tr != nil {
		tr.qRope, tr.kRope = slices.Clone(q), slices.Clone(k)
	}
}

// attendGEMMRows is the query block from which attend batches each head
// into two GEMMs; below it, one dot product and one axpy per key is
// cheaper than gathering the head's keys and values (docs/FINDINGS.md
// F-035).
const attendGEMMRows = 8

// attend is softmax(q kᵀ / √qk) v per query head with its KV head, the
// heads concatenated per row, for t query rows at positions off … off+t−1
// against the keys and values of k and v (off + t rows when causal).
// Masked logits are left out, which equals upstream's float32-min fill:
// their weights are exactly 0.
func (a *Attention) attend(q, k, v []float32, t, off int, mask Mask, tr *attnTrace) []float32 {
	keys := len(k) / (a.kvHeads * a.qk)

	if tr != nil {
		tr.probs = make([]float32, a.heads*t*keys)
	}

	var out []float32
	if t >= attendGEMMRows {
		out = a.attendGEMM(q, k, v, t, off, mask, tr)
	} else {
		out = a.attendRows(q, k, v, t, off, mask, tr)
	}

	if tr != nil {
		tr.attn = slices.Clone(out)
	}

	return out
}

// attendScale is the logits' divisor, jnp.sqrt(float32(qk)); upstream
// divides by it.
func (a *Attention) attendScale() float32 { return float32(math.Sqrt(float64(a.qk))) }

// keysAt is the number of keys query row i at position off + i sees.
func keysAt(keys, off, i int, mask Mask) int {
	if mask == MaskCausal {
		return off + i + 1
	}

	return keys
}

// attendRows is attend one query row at a time: a dot product per key for
// the logits and an axpy per key for the weighted values.
func (a *Attention) attendRows(q, k, v []float32, t, off int, mask Mask, tr *attnTrace) []float32 {
	qW, kW, vW, oW := a.heads*a.qk, a.kvHeads*a.qk, a.kvHeads*a.v, a.heads*a.v
	keys := len(k) / kW
	out := make([]float32, t*oW)
	scores := make([]float32, keys)
	scale := a.attendScale()

	for h := range a.heads {
		kv := a.kvHead(h)

		for i := range t {
			qi := q[i*qW+h*a.qk : i*qW+(h+1)*a.qk]
			s := scores[:keysAt(keys, off, i, mask)]

			for j := range s {
				s[j] = tensor.DotProduct(qi, k[j*kW+kv*a.qk:j*kW+(kv+1)*a.qk]) / scale
			}

			softmaxInPlace(s)

			o := out[i*oW+h*a.v : i*oW+(h+1)*a.v]
			for j, p := range s {
				tensor.Axpy(o, p, v[j*vW+kv*a.v:j*vW+(kv+1)*a.v])
			}

			if tr != nil {
				copy(tr.probs[(h*t+i)*keys:], s)
			}
		}
	}

	return out
}

// attendGEMM is attend for a block of query rows: per KV head it gathers
// the keys (keys × qk) and the transposed values (v × keys) once, and per
// query head it computes all logits as Q·Kᵀ and all outputs as P·V with
// tensor.MatMulTransB. The weights of masked keys are set to 0, so they
// add exact zeros to P·V. The sums run in the GEMM's order instead of one
// key after the other.
func (a *Attention) attendGEMM(q, k, v []float32, t, off int, mask Mask, tr *attnTrace) []float32 {
	qW, kW, vW, oW := a.heads*a.qk, a.kvHeads*a.qk, a.kvHeads*a.v, a.heads*a.v
	keys := len(k) / kW
	out := make([]float32, t*oW)
	qh, kh, vT := make([]float32, t*a.qk), make([]float32, keys*a.qk), make([]float32, a.v*keys)
	probs := make([]float32, t*keys)
	scale := a.attendScale()
	group := a.heads / a.kvHeads

	for kv := range a.kvHeads {
		for j := range keys {
			copy(kh[j*a.qk:(j+1)*a.qk], k[j*kW+kv*a.qk:j*kW+(kv+1)*a.qk])

			for d, x := range v[j*vW+kv*a.v : j*vW+(kv+1)*a.v] {
				vT[d*keys+j] = x
			}
		}

		for h := kv * group; h < (kv+1)*group; h++ {
			for i := range t {
				copy(qh[i*a.qk:(i+1)*a.qk], q[i*qW+h*a.qk:i*qW+(h+1)*a.qk])
			}

			tensor.MatMulTransB(probs, keys, qh, kh, t, keys, a.qk)

			for i := range t {
				row := probs[i*keys : (i+1)*keys]
				n := keysAt(keys, off, i, mask)

				for j := range row[:n] {
					row[j] /= scale
				}

				softmaxInPlace(row[:n])
				clear(row[n:])

				if tr != nil {
					copy(tr.probs[(h*t+i)*keys:], row)
				}
			}

			tensor.MatMulTransB(out[h*a.v:], oW, probs, vT, t, a.v, keys)
		}
	}

	return out
}

// applyTaps is the causal tap conv Σ_j taps[j] · shift_right(z, d·j) over
// t rows of width values, summed in upstream's order j = 0, 1, …, with
// dilation d (1 for the attention taps, the largest n-gram order for
// Engram's). Upstream also multiplies each shifted term by the mask's
// (d·j)-th diagonal (tap_ok); for the causal mask that is 1 wherever the
// shift is defined and the shift fills zeros elsewhere, and without a mask
// there is no factor, so neither changes the sum (docs/FINDINGS.md F-011,
// F-013).
func applyTaps(z []float32, t, width int, taps []float32, n, dilation int) []float32 {
	out := make([]float32, len(z))

	for j := range n {
		tap := taps[j*width : (j+1)*width]
		s := dilation * j

		for i := s; i < t; i++ {
			dst, src := out[i*width:(i+1)*width], z[(i-s)*width:(i-s+1)*width]
			for c, w := range tap {
				dst[c] += w * src[c]
			}
		}
	}

	return out
}
