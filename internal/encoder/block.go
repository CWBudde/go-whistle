package encoder

import (
	"fmt"

	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// block is one encoder layer, a macaron block built from the Needle 3
// modules and the six norms the checkpoint holds (PLAN.md card, step 4):
//
//	x += s · hadamard_mlp_0(pre_hada_norm_0(x))                       FFN1
//	x += σ(attn_gate) · post_attn_norm(attn(ZCRMSNorm_0(x)))          attention, no mask
//	x += pw2(… dw(glu(pw1(conv_norm(x)))) …)                          conv module
//	x += s · hadamard_mlp(pre_hada_norm(x))                           FFN2
//
// s, the GLU, the activation and the place of conv_out_norm in the conv
// module, its padding, the module order and the attention's rotation are
// Hypothesis axes (docs/FINDINGS.md F-014). No tensor here has a bias.
type block struct {
	h   Hypothesis
	dim int

	preHada0, normIn, postNorm, convNorm, convOut, preHada needle.ZCRMSNorm

	attn      *needle.Attention
	gate      float32 // σ(attn_gate)
	hada0     *needle.HadamardMLP
	hada      *needle.HadamardMLP
	pw1       []float32 // [glu, dim]
	pw2       []float32 // [dim, dim]
	dw        []float32 // [dwKernel, half]
	dwKernel  int
	glu, half int // pw1's width and the GLU output's
}

func loadBlock(src weights.Source, prefix string, h Hypothesis) (*block, error) {
	attn, err := needle.LoadAttention(src, prefix)
	if err != nil {
		return nil, err
	}

	b := &block{h: h, dim: attn.Dim(), attn: attn}

	b.hada0, err = needle.LoadHadamardMLP(src, prefix+"hada_0.")
	if err != nil {
		return nil, err
	}

	b.hada, err = needle.LoadHadamardMLP(src, prefix+"hada.")
	if err != nil {
		return nil, err
	}

	norms := []struct {
		dst  *needle.ZCRMSNorm
		name string
	}{
		{&b.preHada0, "pre_hada_0"},
		{&b.normIn, "norm_in"},
		{&b.postNorm, "post_norm"},
		{&b.convNorm, "conv_norm"},
		{&b.convOut, "conv_out_norm"},
		{&b.preHada, "pre_hada"},
	}

	for _, n := range norms {
		*n.dst, err = needle.LoadZCRMSNorm(src, prefix+n.name)
		if err != nil {
			return nil, err
		}
	}

	gate, _, err := src.Tensor(prefix + "attn_gate")
	if err != nil {
		return nil, fmt.Errorf("encoder: %sattn_gate: %w", prefix, err)
	}

	if len(gate) != 1 {
		return nil, fmt.Errorf("encoder: %sattn_gate has %d values, want 1", prefix, len(gate))
	}

	b.gate = sigmoid(gate[0])

	err = b.loadConv(src, prefix)
	if err != nil {
		return nil, err
	}

	return b, nil
}

// loadConv reads pw1, pw2 and dw and checks that the GLU output fits dw.
func (b *block) loadConv(src weights.Source, prefix string) error {
	shape, err := src.Shape(prefix + "pw1")
	if err != nil {
		return fmt.Errorf("encoder: %spw1: %w", prefix, err)
	}

	b.glu = shape[0]

	b.pw1, err = matrix(src, prefix+"pw1", b.glu, b.dim)
	if err != nil {
		return err
	}

	b.pw2, err = matrix(src, prefix+"pw2", b.dim, b.dim)
	if err != nil {
		return err
	}

	dwShape, err := src.Shape(prefix + "dw")
	if err != nil {
		return fmt.Errorf("encoder: %sdw: %w", prefix, err)
	}

	b.dwKernel = dwShape[0]

	b.half = b.glu / 2
	if b.h.GLU == GLUNone {
		b.half = b.glu
	}

	if dwShape[1] != b.half || b.half != b.dim {
		return fmt.Errorf("%w: glu=%s gives %d channels, %sdw is %v and pw2 takes %d", ErrRuledOut,
			axes[11].Names[b.h.GLU-1], b.half, prefix, dwShape, b.dim)
	}

	b.dw, err = matrix(src, prefix+"dw", b.dwKernel, b.half)

	return err
}

// forward runs the block on x, t rows of dim values; x is not written.
func (b *block) forward(x []float32, t int, rope *needle.Rope) ([]float32, error) {
	x = append([]float32(nil), x...)

	scale := float32(1)
	if b.h.FFNScale == FFNHalf {
		scale = 0.5
	}

	err := b.ffn(x, t, b.preHada0, b.hada0, scale)
	if err != nil {
		return nil, err
	}

	if b.h.Order == OrderConvFirst {
		b.conv(x, t)

		err = b.attend(x, t, rope)
	} else {
		err = b.attend(x, t, rope)

		b.conv(x, t)
	}

	if err != nil {
		return nil, err
	}

	err = b.ffn(x, t, b.preHada, b.hada, scale)
	if err != nil {
		return nil, err
	}

	return x, nil
}

// ffn is x += scale · mlp(norm(x)).
func (b *block) ffn(x []float32, t int, norm needle.ZCRMSNorm, mlp *needle.HadamardMLP, scale float32) error {
	n := make([]float32, len(x))
	norm.Apply(n, x)

	m, err := mlp.Forward(n, t)
	if err != nil {
		return err
	}

	for i, v := range m {
		x[i] += float32(scale * v)
	}

	return nil
}

// attend is x += σ(attn_gate) · post_attn_norm(attn(norm_in(x))).
func (b *block) attend(x []float32, t int, rope *needle.Rope) error {
	n := make([]float32, len(x))
	b.normIn.Apply(n, x)

	a, err := b.attn.Forward(n, t, needle.MaskNone, rope)
	if err != nil {
		return err
	}

	b.postNorm.Apply(a, a)

	for i, v := range a {
		x[i] += float32(b.gate * v)
	}

	return nil
}

// conv is x += the conv module of conv_norm(x).
func (b *block) conv(x []float32, t int) {
	d := b.dim
	n := make([]float32, len(x))
	b.convNorm.Apply(n, x)

	p := make([]float32, t*b.glu)
	tensor.MatMulTransB(p, b.glu, n, b.pw1, t, b.glu, d)

	c := b.depthwise(b.gated(p, t), t)

	switch b.h.ConvNorm {
	case NormBeforeAct:
		b.convOut.Apply(c, c)
		applyAct(c, b.h.ConvAct)
	case NormAfterAct:
		applyAct(c, b.h.ConvAct)
		b.convOut.Apply(c, c)
	case NormAfterPW2:
		applyAct(c, b.h.ConvAct)
	}

	o := make([]float32, t*d)
	tensor.MatMulTransB(o, d, c, b.pw2, t, d, d)

	if b.h.ConvNorm == NormAfterPW2 {
		b.convOut.Apply(o, o)
	}

	for i, v := range o {
		x[i] += v
	}
}

// gated applies the GLU to t rows of pw1's output.
func (b *block) gated(p []float32, t int) []float32 {
	if b.h.GLU == GLUNone {
		return p
	}

	h := b.half
	out := make([]float32, t*h)

	for i := range t {
		a, g := p[i*b.glu:i*b.glu+h], p[i*b.glu+h:(i+1)*b.glu]
		o := out[i*h : (i+1)*h]

		for k := range o {
			switch b.h.GLU {
			case GLUValueGate:
				o[k] = a[k] * sigmoid(g[k])
			case GLUGateValue:
				o[k] = sigmoid(a[k]) * g[k]
			case GLUSwishValue:
				o[k] = a[k] * sigmoid(a[k]) * g[k]
			case GLUValueSwish:
				o[k] = a[k] * g[k] * sigmoid(g[k])
			}
		}
	}

	return out
}

// depthwise is the dwKernel-tap depthwise conv over time on t rows of half
// channels: dw[j, c] weights frame i − pad + j, zeros outside the clip.
func (b *block) depthwise(x []float32, t int) []float32 {
	h, k := b.half, b.dwKernel

	pad := (k - 1) / 2
	if b.h.DWPad == DWCausal {
		pad = k - 1
	}

	out := make([]float32, t*h)

	for i := range t {
		o := out[i*h : (i+1)*h]

		for j := range k {
			src := i - pad + j
			if src < 0 || src >= t {
				continue
			}

			w, row := b.dw[j*h:(j+1)*h], x[src*h:(src+1)*h]
			for c := range o {
				o[c] += w[c] * row[c]
			}
		}
	}

	return out
}
