package needle

import (
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Block is architecture.py Block, the decoder layer that MHC wraps:
//
//	x += α · ev,  α = σ(⟨rms_unit(x), rms_unit(ek)⟩ / √d)     engram sites only
//	h   = x + σ(attn_gate) · post_attn_norm(attn(ZCRMSNorm_0(x)))
//	out = h + HadamardMLP(pre_hada_norm(h))
//
// Whistle's decoder layers add a gated cross-attention, which upstream's
// Block does not contain; it is not part of this block (docs/FINDINGS.md
// F-013). Extend's Hooks let the decoder insert it (F-021).
type Block struct {
	dim int

	normIn, postNorm, preHada ZCRMSNorm
	attn                      *Attention
	gate                      float32 // σ(attn_gate)
	mlp                       *HadamardMLP
}

// LoadBlock loads the block with the names prefix+"norm_in", … ("layer00."
// … "layer07."): the attention and the MLP as LoadAttention and
// LoadHadamardMLP, plus norm_in, post_norm, attn_gate and pre_hada.
func LoadBlock(src weights.Source, prefix string) (*Block, error) {
	attn, err := LoadAttention(src, prefix)
	if err != nil {
		return nil, err
	}

	mlp, err := LoadHadamardMLP(src, prefix)
	if err != nil {
		return nil, err
	}

	if mlp.Dim() != attn.Dim() {
		return nil, fmt.Errorf("needle: %s: attention width %d, MLP width %d", prefix, attn.Dim(), mlp.Dim())
	}

	l := &loader{src: src, prefix: prefix}
	b := &Block{dim: attn.Dim(), attn: attn, mlp: mlp}
	b.normIn = NewZCRMSNorm(l.vector("norm_in", b.dim))
	b.postNorm = NewZCRMSNorm(l.vector("post_norm", b.dim))
	b.preHada = NewZCRMSNorm(l.vector("pre_hada", b.dim))

	gate := l.vector("attn_gate", 1)
	if l.err != nil {
		return nil, l.err
	}

	b.gate = sigmoid(gate[0])

	return b, nil
}

// Dim is the model width of the input and the output.
func (b *Block) Dim() int { return b.dim }

// Forward runs the block on x, t rows of Dim values, and returns t rows of
// Dim values; x is not written. mask and rope are the attention's (the
// decoder: MaskCausal and θ = 1e5 tables). engram is the site's keys and
// values at an engram layer (Engram.Forward), nil elsewhere.
func (b *Block) Forward(x []float32, t int, mask Mask, rope *Rope, engram *EngramKV) ([]float32, error) {
	return b.forward(x, t, mask, rope, engram, nil)
}

// Hook adds a residual branch to h, t rows of the block's width, in place.
type Hook func(h []float32, t int) error

// Hooks are residual branches Extend inserts into the block; a nil hook is
// skipped. Whistle's decoder puts its cross-attention at one of them
// (docs/FINDINGS.md F-021).
type Hooks struct {
	BeforeAttn Hook // after the engram gate, before the self-attention
	AfterAttn  Hook // between the self-attention and the MLP
	AfterMLP   Hook // after the MLP's residual add
}

// Extend runs the block causally on x, t new rows of Dim values at the
// positions c.Len() … c.Len()+t−1, with the self-attention's keys and
// values cached in c, and returns t rows of Dim values. engram is the
// site's keys and values of the new rows (Engram.Extend), nil away from
// the sites. With empty hooks and an empty cache it equals Forward with
// MaskCausal (docs/FINDINGS.md F-022).
func (b *Block) Extend(x []float32, t int, c *KVCache, rope *Rope, engram *EngramKV, hooks Hooks,
) ([]float32, error) {
	attn := func(n []float32) ([]float32, error) { return b.attn.Extend(n, t, c, rope) }

	return b.run(x, t, attn, engram, hooks, nil)
}

// blockTrace receives the intermediates for the parity tests.
type blockTrace struct {
	normIn, attn, postAttn, h, preHada, mlp []float32
}

func (b *Block) forward(x []float32, t int, mask Mask, rope *Rope, engram *EngramKV, tr *blockTrace,
) ([]float32, error) {
	attn := func(n []float32) ([]float32, error) { return b.attn.Forward(n, t, mask, rope) }

	return b.run(x, t, attn, engram, Hooks{}, tr)
}

// run is the block around the self-attention attn, with the hooks.
func (b *Block) run(x []float32, t int, attn func([]float32) ([]float32, error), engram *EngramKV, hooks Hooks,
	tr *blockTrace,
) ([]float32, error) {
	if t < 0 || len(x) != t*b.dim {
		return nil, fmt.Errorf("needle: block input has %d values, want %d rows of %d", len(x), t, b.dim)
	}

	h := slices.Clone(x)

	if engram != nil {
		if len(engram.K) != len(x) || len(engram.V) != len(x) {
			return nil, fmt.Errorf("needle: engram k/v have %d/%d values, want %d", len(engram.K), len(engram.V),
				len(x))
		}

		applyEngram(h, engram, t, b.dim, make([]float32, t))
	}

	err := runHook(hooks.BeforeAttn, h, t)
	if err != nil {
		return nil, err
	}

	n := make([]float32, len(h))
	b.normIn.Apply(n, h)

	a, err := attn(n)
	if err != nil {
		return nil, err
	}

	if tr != nil {
		tr.normIn, tr.attn = n, slices.Clone(a)
	}

	b.postNorm.Apply(a, a)

	// skip + σ(attn_gate) · x: the product is rounded first, no FMA.
	for i, v := range a {
		h[i] += float32(b.gate * v)
	}

	err = runHook(hooks.AfterAttn, h, t)
	if err != nil {
		return nil, err
	}

	p := make([]float32, len(h))
	b.preHada.Apply(p, h)

	m, err := b.mlp.Forward(p, t)
	if err != nil {
		return nil, err
	}

	if tr != nil {
		tr.postAttn, tr.h, tr.preHada, tr.mlp = a, slices.Clone(h), p, slices.Clone(m)
	}

	for i, v := range h {
		m[i] += v
	}

	err = runHook(hooks.AfterMLP, m, t)
	if err != nil {
		return nil, err
	}

	return m, nil
}

func runHook(hook Hook, h []float32, t int) error {
	if hook == nil {
		return nil
	}

	return hook(h, t)
}

// applyEngram is Block's engram gate for one site in place on x, t rows of
// dim values: x += α · v with α = σ(⟨rms_unit(x), rms_unit(k)⟩ / √dim)
// per row, written to alpha (t values). Upstream sums the gated values of
// every site times its flag, which is 1 for the layer's own site and 0 for
// the others, so one site is exact.
func applyEngram(x []float32, kv *EngramKV, t, dim int, alpha []float32) {
	ux, uk := make([]float32, dim), make([]float32, dim)
	scale := float32(math.Sqrt(float64(dim))) // math.sqrt(d_model), divided by as upstream

	for i := range t {
		xi, vi := x[i*dim:(i+1)*dim], kv.V[i*dim:(i+1)*dim]
		RMSUnit(ux, xi, dim)
		RMSUnit(uk, kv.K[i*dim:(i+1)*dim], dim)

		a := sigmoid(tensor.DotProduct(ux, uk) / scale)
		alpha[i] = a

		for c, v := range vi {
			xi[c] += float32(a * v)
		}
	}
}
