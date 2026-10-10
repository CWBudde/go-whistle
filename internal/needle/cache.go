package needle

import (
	"fmt"
	"slices"
)

// KVCache is one causal self-attention's decoding state (docs/FINDINGS.md
// F-022): the raw Q/K/V projections of the last Taps − 1 positions, which
// the causal taps of the next position read, and the normed, rotated keys
// and the values of every position so far. The zero value is an empty
// cache.
type KVCache struct {
	n                   int       // positions so far
	qHist, kHist, vHist []float32 // raw projections of the last Taps − 1 positions, oldest first
	k, v                []float32 // all positions, row layout
}

// Len is the number of positions in the cache.
func (c *KVCache) Len() int { return c.n }

// Fork returns a cache with c's positions that extends independently of c
// (docs/FINDINGS.md F-026). The two share the keys and values so far
// without a copy: Extend only appends to them, and the fork's slices are
// capped at their length, so its first append moves its rows to a new
// array while c may keep appending in place. The tap history is replaced,
// never written, on every Extend, so it is shared as is.
func (c *KVCache) Fork() KVCache {
	return KVCache{
		n: c.n, qHist: c.qHist, kHist: c.kHist, vHist: c.vHist,
		k: slices.Clip(c.k), v: slices.Clip(c.v),
	}
}

// Extend runs the causal attention on x, t new rows of Dim values at the
// positions Len() … Len()+t−1, appends their keys and values to c and
// returns t rows of Dim values. On an empty cache it equals Forward with
// MaskCausal, and any split of a sequence into Extend calls gives the same
// rows as one call, up to float32 summation order. rope is nil for no
// rotary embedding; otherwise it must cover Len()+t positions.
func (a *Attention) Extend(x []float32, t int, c *KVCache, rope *Rope) ([]float32, error) {
	err := a.check(x, t, MaskCausal, nil)
	if err != nil {
		return nil, err
	}

	if rope != nil && (rope.HeadDim() != a.qk || rope.Len() < c.n+t) {
		return nil, fmt.Errorf("needle: RoPE tables for head width %d and %d positions, want %d and %d",
			rope.HeadDim(), rope.Len(), a.qk, c.n+t)
	}

	qW, kW, vW := a.heads*a.qk, a.kvHeads*a.qk, a.kvHeads*a.v
	q, k, v := make([]float32, t*qW), make([]float32, t*kW), make([]float32, t*vW)
	linear(q, x, t, a.qProj, qW, a.dim)
	linear(k, x, t, a.kProj, kW, a.dim)
	linear(v, x, t, a.vProj, vW, a.dim)

	if a.taps > 0 {
		q = a.extendTaps(q, t, qW, a.qTaps, &c.qHist)
		k = a.extendTaps(k, t, kW, a.kTaps, &c.kHist)
		v = a.extendTaps(v, t, vW, a.vTaps, &c.vHist)
	}

	a.normRotate(q, k, t, c.n, rope, nil)

	c.k, c.v = append(c.k, k...), append(c.v, v...)
	out := a.attend(q, c.k, c.v, t, c.n, MaskCausal, nil)
	c.n += t

	return a.output(x, t, out, nil), nil
}

// extendTaps applies the causal taps to t new raw rows z of width values,
// prefixed by the history hist of earlier raw rows, and keeps the last
// Taps − 1 raw rows in hist. The history holds every row a tap can reach,
// and positions before the first count as zero, as in applyTaps.
func (a *Attention) extendTaps(z []float32, t, width int, taps []float32, hist *[]float32) []float32 {
	h := len(*hist) / width
	all := append(slices.Clone(*hist), z...)
	out := applyTaps(all, h+t, width, taps, a.taps, 1)

	keep := min(a.taps-1, h+t)
	*hist = slices.Clone(all[(h+t-keep)*width:])

	return out[h*width:]
}

// EngramCache is one engram site's decoding state (docs/FINDINGS.md
// F-022): the last MaxOrder − 1 tokens, which the n-gram hashes of the
// next position read, and the raw values (e @ value_proj) of the last
// (taps − 1) · dilation positions, which the dilated taps read. The zero
// value is an empty cache.
type EngramCache struct {
	n    int       // positions so far
	toks []int     // the last MaxOrder − 1 tokens, oldest first
	vRaw []float32 // raw values of the last (taps − 1) · dilation positions
}

// Len is the number of positions in the cache.
func (c *EngramCache) Len() int { return c.n }

// Fork returns a cache with c's positions that extends independently of c
// (docs/FINDINGS.md F-026). Extend replaces the token and value history
// instead of writing into it, so the fork shares both.
func (c *EngramCache) Fork() EngramCache { return *c }

// Extend returns the keys and values of the tokens that continue the
// sequence in c, each len(tokens) rows of Dim values, and advances c. On
// an empty cache it equals Forward, and any split of a sequence into
// Extend calls gives the same rows as one call.
func (e *Engram) Extend(tokens []int, c *EngramCache) (*EngramKV, error) {
	for i, tok := range tokens {
		if tok < 0 {
			return nil, fmt.Errorf("needle: engram token %d at %d is negative", tok, i)
		}
	}

	t, n := len(tokens), len(e.orders)
	w := n * e.sub

	hist := append(slices.Clone(c.toks), tokens...)
	h := len(c.toks)
	emb := make([]float32, t*w)
	idx := make([]int, n)

	for i := range t {
		pos := c.n + i
		e.IndicesAt(idx, hist[max(0, h+i+1-e.maxOrder):h+i+1])

		// fetched · ngram_ok: positions before order − 1 keep zero rows.
		for col, order := range e.orders {
			if pos+1 < order {
				continue
			}

			row := col*e.slots + idx[col]
			copy(emb[i*w+col*e.sub:i*w+(col+1)*e.sub], e.tables[row*e.sub:(row+1)*e.sub])
		}
	}

	k, vRaw := make([]float32, t*e.dim), make([]float32, t*e.dim)
	linear(k, emb, t, e.keyProj, e.dim, w)
	linear(vRaw, emb, t, e.valProj, e.dim, w)

	rh := len(c.vRaw) / e.dim
	all := append(slices.Clone(c.vRaw), vRaw...)
	v := applyTaps(all, rh+t, e.dim, e.tapW, e.taps, e.dilation)[rh*e.dim:]

	keepV := min((e.taps-1)*e.dilation, rh+t)
	c.vRaw = slices.Clone(all[(rh+t-keepV)*e.dim:])
	keepT := min(e.maxOrder-1, len(hist))
	c.toks = slices.Clone(hist[len(hist)-keepT:])
	c.n += t

	return &EngramKV{K: k, V: v}, nil
}
