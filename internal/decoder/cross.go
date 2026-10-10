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
	qProj, kProj, vProj []float32 // [heads·qk | heads·qk | heads·v, dim]
	gateProj, outProj   []float32 // [heads·v, dim], [dim, heads·v]
	resid               float32   // the residual factor from cross_gate
	noNorm              bool
}

// loadCross loads the cross-attention with the names prefix+"norm_in", …
// ("cross00." … "cross07.").
func loadCross(src weights.Source, prefix string, h Hypothesis) (*cross, error) {
	c := &cross{h: h}
	l := &loader{src: src, prefix: prefix}

	normIn := l.tensor("norm_in", 1)
	post := l.tensor("post_norm", 1)
	qNorm := l.tensor("q_norm", 1)
	kNorm := l.tensor("k_norm", 1)
	gate := l.tensor("attn_gate", 1)
	c.dim, c.qk = len(normIn), len(qNorm)

	var qW, vW int

	c.qProj, qW = l.matrix("q_proj", c.dim)
	c.kProj, _ = l.matrix("k_proj", c.dim)
	c.vProj, vW = l.matrix("v_proj", c.dim)
	c.gateProj, _ = l.matrix("gate_proj", c.dim)
	c.outProj, _ = l.matrix("out_proj", vW)

	if l.err != nil {
		return nil, l.err
	}

	if c.qk == 0 || qW%c.qk != 0 || len(kNorm) != c.qk || len(post) != c.dim || len(gate) != 1 {
		return nil, fmt.Errorf("decoder: %s: q_proj width %d, norms %d/%d/%d", prefix, qW, c.qk, len(kNorm), len(post))
	}

	c.heads = qW / c.qk
	c.v = vW / c.heads

	if vW%c.heads != 0 || len(c.kProj) != qW*c.dim || len(c.gateProj) != vW*c.dim || len(c.outProj) != c.dim*vW {
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

// tensor returns a tensor of the given rank.
func (l *loader) tensor(name string, rank int) []float32 {
	if l.err != nil {
		return nil
	}

	data, shape, err := l.src.Tensor(l.prefix + name)
	if err != nil {
		l.err = fmt.Errorf("decoder: %s%s: %w", l.prefix, name, err)

		return nil
	}

	if len(shape) != rank {
		l.err = fmt.Errorf("decoder: %s%s has shape %v, want rank %d", l.prefix, name, shape, rank)

		return nil
	}

	return data
}

// matrix returns a [out, in] matrix with in columns and its row count.
func (l *loader) matrix(name string, in int) ([]float32, int) {
	data := l.tensor(name, 2)
	if data == nil {
		return nil, 0
	}

	if in <= 0 || len(data)%in != 0 {
		l.err = fmt.Errorf("decoder: %s%s has %d values, not rows of %d", l.prefix, name, len(data), in)

		return nil, 0
	}

	return data, len(data) / in
}

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
// for one encoder output of frames rows, projected once per clip.
type memory struct {
	k, v   []float32 // [frames, heads·qk], [frames, heads·v]
	frames int
}

// project computes the memory keys and values of enc, frames rows of dim
// values. rope rotates frame f at position f when set.
func (c *cross) project(enc []float32, frames int, rope *needle.Rope) memory {
	kW, vW := c.heads*c.qk, c.heads*c.v
	m := memory{k: make([]float32, frames*kW), v: make([]float32, frames*vW), frames: frames}
	linear(m.k, enc, frames, c.kProj, kW, c.dim)
	linear(m.v, enc, frames, c.vProj, vW, c.dim)

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

	return m
}

// branch returns the term the residual adds for h, t rows of dim values at
// the token positions off … off+t−1, against the memory m.
func (c *cross) branch(h []float32, t, off int, m memory, rope *needle.Rope) []float32 {
	qW, oW := c.heads*c.qk, c.heads*c.v
	x := make([]float32, len(h))
	c.norm.Apply(x, h)

	q := make([]float32, t*qW)
	linear(q, x, t, c.qProj, qW, c.dim)

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

	out := c.attend(q, t, m)

	if c.h.Gate != GateNone {
		gin := x
		if c.h.Gate == GateSigmoidResidual {
			gin = h
		}

		gate := make([]float32, t*oW)
		linear(gate, gin, t, c.gateProj, oW, c.dim)

		for i, g := range gate {
			if c.h.Gate == GateSiLU {
				out[i] *= g * sigmoid(g)
			} else {
				out[i] *= sigmoid(g)
			}
		}
	}

	y := make([]float32, t*c.dim)
	linear(y, out, t, c.outProj, c.dim, oW)

	if c.h.Residual != ResNoPostNorm {
		c.post.Apply(y, y)
	}

	for i, v := range y {
		y[i] = c.resid * v
	}

	return y
}

// attend is softmax(q kᵀ / √qk) v per head over every memory frame.
func (c *cross) attend(q []float32, t int, m memory) []float32 {
	qW, vW := c.heads*c.qk, c.heads*c.v
	out := make([]float32, t*vW)
	scores := make([]float32, m.frames)
	scale := float32(math.Sqrt(float64(c.qk)))

	for hd := range c.heads {
		for i := range t {
			qi := q[i*qW+hd*c.qk : i*qW+(hd+1)*c.qk]
			for f := range scores {
				scores[f] = tensor.DotProduct(qi, m.k[f*qW+hd*c.qk:f*qW+(hd+1)*c.qk]) / scale
			}

			softmax(scores)

			o := out[i*vW+hd*c.v : i*vW+(hd+1)*c.v]
			for f, p := range scores {
				tensor.Axpy(o, p, m.v[f*vW+hd*c.v:f*vW+(hd+1)*c.v])
			}
		}
	}

	return out
}

// linear computes dst = x · wᵀ for rows of x of width in and w [out, in].
func linear(dst, x []float32, rows int, w []float32, out, in int) {
	tensor.MatMulTransB(dst, out, x, w, rows, out, in)
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
