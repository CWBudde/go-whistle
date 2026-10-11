package decoder

import (
	"fmt"
	"math"

	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// cross is one decoder layer's gated cross-attention over the encoder
// output (docs/FINDINGS.md F-021). Its tensors are .cact crossNN.*:
// cross_norm, q/k/v/gate/out projections, q_norm, k_norm, post_cross_norm
// and cross_gate. q and k are 8 heads of 48 (no GQA), v 8 heads of 64:
//
//	x   = cross_norm(h)
//	c   = softmax(q̂(x) k̂(m)ᵀ / √48) v(m) · σ(gate_proj(x))
//	h  += σ(cross_gate) · post_cross_norm(out_proj(c))
//
// with the Hypothesis' alternatives for the norms, the gate and the
// residual.
type cross struct {
	h                   Hypothesis
	dim, heads, qk, v   int
	norm, post          needle.ZCRMSNorm
	qNorm, kNorm        needle.ZCRMSNorm
	qProj, kProj, vProj needle.Linear // [heads·qk | heads·qk | heads·v, dim]
	gateProj, outProj   needle.Linear // [heads·v, dim], [dim, heads·v]
	resid               float32       // the residual factor from cross_gate
	noNorm              bool
	int8                needle.Int8 // setInt8
}

// loadCross loads the cross-attention with the names prefix+"norm_in", …
// ("cross00." … "cross07.").
func loadCross(src weights.Source, prefix string, h Hypothesis) (*cross, error) {
	c := &cross{h: h}
	l := &loader{src: src, prefix: prefix}

	normIn := l.vector("norm_in")
	post := l.vector("post_norm")
	qNorm := l.vector("q_norm")
	kNorm := l.vector("k_norm")
	gate := l.vector("attn_gate")
	c.dim, c.qk = len(normIn), len(qNorm)

	c.qProj = l.matrix("q_proj", c.dim)
	c.kProj = l.matrix("k_proj", c.dim)
	c.vProj = l.matrix("v_proj", c.dim)
	c.gateProj = l.matrix("gate_proj", c.dim)
	c.outProj = l.matrix("out_proj", c.vProj.Out())

	if l.err != nil {
		return nil, l.err
	}

	qW, vW := c.qProj.Out(), c.vProj.Out()

	if c.qk == 0 || qW%c.qk != 0 || len(kNorm) != c.qk || len(post) != c.dim || len(gate) != 1 {
		return nil, fmt.Errorf("decoder: %s: q_proj width %d, norms %d/%d/%d", prefix, qW, c.qk, len(kNorm), len(post))
	}

	c.heads = qW / c.qk
	c.v = vW / c.heads

	if vW%c.heads != 0 || c.kProj.Out() != qW || c.gateProj.Out() != vW || c.outProj.Out() != c.dim {
		return nil, fmt.Errorf("decoder: %s projections do not fit %d heads of %d/%d", prefix, c.heads, c.qk, c.v)
	}

	c.norm, c.post = needle.NewZCRMSNorm(normIn), needle.NewZCRMSNorm(post)
	c.setQKNorm(qNorm, kNorm)
	c.setResidual(gate[0])

	return c, nil
}

// loader reads tensors prefix+name; the first error sticks.
type loader struct {
	src    weights.Source
	prefix string
	err    error
}

// vector returns a rank-1 tensor.
func (l *loader) vector(name string) []float32 {
	if l.err != nil {
		return nil
	}

	data, shape, err := l.src.Tensor(l.prefix + name)
	if err != nil {
		l.err = fmt.Errorf("decoder: %s%s: %w", l.prefix, name, err)

		return nil
	}

	if len(shape) != 1 {
		l.err = fmt.Errorf("decoder: %s%s has shape %v, want rank 1", l.prefix, name, shape)

		return nil
	}

	return data
}

// matrix returns a [out, in] matrix with in columns, packed where the
// source serves it packed (needle.LoadLinear).
func (l *loader) matrix(name string, in int) needle.Linear {
	if l.err != nil {
		return needle.Linear{}
	}

	if in <= 0 {
		l.err = fmt.Errorf("decoder: %s%s: no rows of %d", l.prefix, name, in)

		return needle.Linear{}
	}

	w, err := needle.LoadLinear(l.src, l.prefix+name, 0, in)
	if err != nil {
		l.err = fmt.Errorf("decoder: %w", err)
	}

	return w
}

// setInt8 switches the cross-attention to the engine's int8 numerics
// (docs/FINDINGS.md F-041), as needle.Attention.SetInt8: with q.Act, the
// queries are quantised per head; with q.KV, the memory holds its keys and
// values in int8 per head and frame.
func (c *cross) setInt8(q needle.Int8) { c.int8 = q }

// setQKNorm builds the q/k norms of the QKNorm axis.
func (c *cross) setQKNorm(q, k []float32) {
	switch c.h.QKNorm {
	case QKNormPlain:
		// gain = scale: ZCRMSNorm with scale − 1.
		q2, k2 := make([]float32, len(q)), make([]float32, len(k))
		for i := range q {
			q2[i], k2[i] = q[i]-1, k[i]-1
		}

		c.qNorm, c.kNorm = needle.NewZCRMSNorm(q2), needle.NewZCRMSNorm(k2)
	case QKNormUnit:
		c.qNorm, c.kNorm = needle.NewZCRMSNorm(make([]float32, len(q))), needle.NewZCRMSNorm(make([]float32, len(k)))
	case QKNormNone:
		c.noNorm = true
	default:
		c.qNorm, c.kNorm = needle.NewZCRMSNorm(q), needle.NewZCRMSNorm(k)
	}
}

// setResidual sets the residual factor of the Residual axis from the
// cross_gate parameter.
func (c *cross) setResidual(g float32) {
	switch c.h.Residual {
	case ResTanh:
		c.resid = float32(math.Tanh(float64(g)))
	case ResRaw:
		c.resid = g
	default:
		c.resid = sigmoid(g)
	}
}

// memory is a cross-attention's keys (normed, maybe rotated) and values
// for one encoder output of frames rows, projected once per clip: float32,
// or int8 per head and frame with needle.Int8.KV (docs/FINDINGS.md F-041).
type memory struct {
	k, v   []float32 // [frames, heads·qk], [frames, heads·v]; nil in int8
	k8, v8 needle.Int8Rows
	frames int
}

// project computes the memory keys and values of enc, frames rows of dim
// values. rope rotates frame f at position f when set. Both projections
// read enc, so packed ones share its rotation and tables
// (needle.ApplyShared), with the same bits as one Apply each
// (docs/FINDINGS.md F-038, F-041).
func (c *cross) project(enc []float32, frames int, rope *needle.Rope) memory {
	kW, vW := c.heads*c.qk, c.heads*c.v
	m := memory{k: make([]float32, frames*kW), v: make([]float32, frames*vW), frames: frames}
	needle.ApplyShared([][]float32{m.k, m.v}, enc, frames, c.kProj, c.vProj)

	if !c.noNorm {
		c.kNorm.Apply(m.k, m.k)
	}

	if rope != nil {
		for f := range frames {
			for hd := range c.heads {
				rope.Apply(m.k[(f*c.heads+hd)*c.qk:(f*c.heads+hd+1)*c.qk], f)
			}
		}
	}

	if c.int8.KV {
		m.k8, m.v8 = needle.QuantizeRows(m.k, kW, c.qk), needle.QuantizeRows(m.v, vW, c.v)
		m.k, m.v = nil, nil
	}

	return m
}

// branch returns the term the residual adds for h, t rows of dim values at
// the token positions off … off+t−1, against the memory m. A non-nil keep
// receives the attention probabilities and the output gate (Attention).
func (c *cross) branch(h []float32, t, off int, m memory, rope *needle.Rope, keep *Attention) []float32 {
	qW, oW := c.heads*c.qk, c.heads*c.v
	x := make([]float32, len(h))
	c.norm.Apply(x, h)

	q := make([]float32, t*qW)
	c.qProj.Apply(q, x, t)

	if !c.noNorm {
		c.qNorm.Apply(q, q)
	}

	if rope != nil {
		for i := range t {
			for hd := range c.heads {
				rope.Apply(q[(i*c.heads+hd)*c.qk:(i*c.heads+hd+1)*c.qk], off+i)
			}
		}
	}

	out := c.attend(q, t, m, keep)

	if c.h.Gate != GateNone {
		gin := x
		if c.h.Gate == GateSigmoidResidual {
			gin = h
		}

		gate := make([]float32, t*oW)
		c.gateProj.Apply(gate, gin, t)

		for i, g := range gate {
			if c.h.Gate == GateSiLU {
				gate[i] = g * sigmoid(g)
			} else {
				gate[i] = sigmoid(g)
			}

			out[i] *= gate[i]
		}

		if keep != nil {
			keep.Gate = gate
		}
	}

	y := make([]float32, t*c.dim)
	c.outProj.Apply(y, out, t)

	if c.h.Residual != ResNoPostNorm {
		c.post.Apply(y, y)
	}

	for i, v := range y {
		y[i] = c.resid * v
	}

	return y
}

// attend is softmax(q kᵀ / √qk) v per head over every memory frame. A
// non-nil keep receives the probabilities, [t, heads, frames]. An int8
// memory runs needle.AttendInt8, which quantises the queries per head with
// needle.Int8.Act (docs/FINDINGS.md F-041).
func (c *cross) attend(q []float32, t int, m memory, keep *Attention) []float32 {
	qW, vW := c.heads*c.qk, c.heads*c.v
	scale := float32(math.Sqrt(float64(c.qk)))

	if keep != nil {
		keep.Probs = make([]float32, t*c.heads*m.frames)
	}

	if c.int8.KV {
		var probs []float32
		if keep != nil {
			probs = keep.Probs
		}

		all := func(int) int { return m.frames }

		return needle.AttendInt8(q, t, c.heads, m.k8, m.v8, all, scale, c.int8.Act, probs)
	}

	if c.int8.Act {
		needle.FakeQuant(q, c.qk)
	}

	out := make([]float32, t*vW)
	scores := make([]float32, m.frames)

	for hd := range c.heads {
		for i := range t {
			qi := q[i*qW+hd*c.qk : i*qW+(hd+1)*c.qk]
			for f := range scores {
				scores[f] = tensor.DotProduct(qi, m.k[f*qW+hd*c.qk:f*qW+(hd+1)*c.qk]) / scale
			}

			softmax(scores)

			if keep != nil {
				copy(keep.Probs[(i*c.heads+hd)*m.frames:], scores)
			}

			o := out[i*vW+hd*c.v : i*vW+(hd+1)*c.v]
			for f, p := range scores {
				tensor.Axpy(o, p, m.v[f*vW+hd*c.v:f*vW+(hd+1)*c.v])
			}
		}
	}

	return out
}

// sigmoid is jax.nn.sigmoid in float32.
func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// softmax is jax.nn.softmax over s in place: exp(s − max) / Σ.
func softmax(s []float32) {
	m := float32(math.Inf(-1))
	for _, v := range s {
		m = max(m, v)
	}

	var sum float32

	for i, v := range s {
		e := float32(math.Exp(float64(v - m)))
		s[i] = e
		sum += e
	}

	for i := range s {
		s[i] /= sum
	}
}
