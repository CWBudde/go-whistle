package needle

import (
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// mhcSinkhornIters is architecture.py _sinkhorn's iteration count.
const mhcSinkhornIters = 20

// MHC is one layer of architecture.py's manifold hyper-connections, the
// residual stream of n lanes (Whistle: 4) that _ScanBody's advance carries
// around a block:
//
//	nx     = rms_unit(stream.reshape(T, n·C))
//	h_pre  = σ(a_pre · (nx @ φ_pre) + b_pre + pre_off)
//	u      = Σ_k h_pre[k] · stream[k]                    the block input
//	y      = block(u) − u
//	h_post = 2 · σ(a_post · (nx @ φ_post) + b_post + post_off)
//	h_res  = sinkhorn(a_res · (nx @ φ_res).reshape(n, n) + b_res)
//	out[i] = Σ_j h_res[i, j] · stream[j] + h_post[i] · y
//
// with lane = eye(n)[layer % n], pre_off = 8·lane − 4 and post_off =
// −4·(1 − lane). Stack expands the embedding to n equal lanes before the
// first layer (ExpandLanes) and takes the lane mean after the last
// (CollapseLanes) (docs/FINDINGS.md F-013).
type MHC struct {
	lanes, dim int

	aPre, aPost, aRes float32
	bPre, bPost       []float32 // (lanes)
	bRes              []float32 // (lanes, lanes)
	phiPre, phiPost   Linear    // [lanes, lanes·dim]
	phiRes            Linear    // [lanes², lanes·dim], row n·i + j is h_res[i, j]
	preOff, postOff   []float32 // (lanes)
}

// LoadMHC loads layer's heads from the stacked tensors prefix+"mhc_a_pre",
// … ("" for the decoder, "enc." for the encoder). The lane count comes from
// mhc_b_pre and the width from φ, which the Source serves lane-major: rows
// layer·n … layer·n + n − 1 of mhc_phi_pre are the layer's n columns
// (docs/FINDINGS.md F-006).
func LoadMHC(src weights.Source, prefix string, layer int) (*MHC, error) {
	l := &loader{src: src, prefix: prefix}

	aPre := l.vector("mhc_a_pre", 0)
	layers := len(aPre)

	if l.err == nil && (layer < 0 || layer >= layers) {
		l.failf("%smhc: layer %d of %d", prefix, layer, layers)
	}

	aPost := l.vector("mhc_a_post", layers)
	aRes := l.vector("mhc_a_res", layers)
	bPre, _, n := l.matrix("mhc_b_pre", layers, 0)
	bPost, _, _ := l.matrix("mhc_b_post", layers, n)
	bRes := l.cube("mhc_b_res", layers, n, n)
	phiPre := l.linear("mhc_phi_pre", layers*n, 0)
	nC := phiPre.In()

	if l.err == nil && (n == 0 || nC%n != 0) {
		l.failf("%smhc_phi_pre width %d is not a multiple of %d lanes", prefix, nC, n)
	}

	phiPost := l.linear("mhc_phi_post", layers*n, nC)
	phiRes := l.linear("mhc_phi_res", layers*n*n, nC)

	if l.err != nil {
		return nil, l.err
	}

	m := &MHC{lanes: n, dim: nC / n}
	m.aPre, m.aPost, m.aRes = aPre[layer], aPost[layer], aRes[layer]
	m.bPre, m.bPost = bPre[layer*n:(layer+1)*n], bPost[layer*n:(layer+1)*n]
	m.bRes = bRes[layer*n*n : (layer+1)*n*n]
	m.phiPre = phiPre.Rows(layer*n, (layer+1)*n)
	m.phiPost = phiPost.Rows(layer*n, (layer+1)*n)
	m.phiRes = phiRes.Rows(layer*n*n, (layer+1)*n*n)
	m.preOff, m.postOff = laneOffsets(n, layer)

	return m, nil
}

// WithLane returns m with the lane offsets of position lane instead of its
// layer's: the heads and weights are shared. A ladder sub-network (Stack
// with exit_depth) gives its i-th active layer lane eye(n)[i % n]
// (docs/FINDINGS.md F-030).
func (m *MHC) WithLane(lane int) *MHC {
	c := *m
	c.preOff, c.postOff = laneOffsets(m.lanes, lane)

	return &c
}

// Lanes is the number of residual lanes.
func (m *MHC) Lanes() int { return m.lanes }

// Dim is the model width of one lane and of the block.
func (m *MHC) Dim() int { return m.dim }

// Forward advances stream, t rows of Lanes × Dim values (lane-major within
// a row), through block and returns the new stream in the same layout.
// block gets u, t rows of Dim values, and must return t rows of Dim values
// without writing to u.
func (m *MHC) Forward(stream []float32, t int, block func(u []float32) ([]float32, error)) ([]float32, error) {
	return m.forward(stream, t, block, nil)
}

// laneOffsets is Stack's pre_off = 8·lane − 4 and post_off = −4·(1 − lane)
// for lane = eye(n)[layer % n]: the layer's own lane gets +4 and 0, the
// others −4 and −4.
func laneOffsets(n, layer int) (pre, post []float32) {
	pre, post = make([]float32, n), make([]float32, n)

	for i := range n {
		pre[i], post[i] = -4, -4
	}

	pre[layer%n], post[layer%n] = 4, 0

	return pre, post
}

// ExpandLanes is Stack's lane expand: every row of dim values of x
// broadcast to lanes equal lanes, giving rows of lanes × dim values. It
// panics unless len(x) is a multiple of dim.
func ExpandLanes(x []float32, lanes, dim int) []float32 {
	checkRows("ExpandLanes", x, x, dim)

	out := make([]float32, 0, len(x)*lanes)
	for r := 0; r < len(x); r += dim {
		for range lanes {
			out = append(out, x[r:r+dim]...)
		}
	}

	return out
}

// CollapseLanes is Stack's ending jnp.mean(stream, axis=2): the mean over
// the lanes of each row of lanes × dim values, giving rows of dim values.
// It panics unless len(stream) is a multiple of lanes × dim.
func CollapseLanes(stream []float32, lanes, dim int) []float32 {
	checkRows("CollapseLanes", stream, stream, lanes*dim)

	out := make([]float32, len(stream)/lanes)
	for r := range len(out) / dim {
		o := out[r*dim : (r+1)*dim]
		for k := range lanes {
			off := (r*lanes + k) * dim
			for c, v := range stream[off : off+dim] {
				o[c] += v
			}
		}

		for c := range o {
			o[c] /= float32(lanes)
		}
	}

	return out
}

// mhcTrace receives the intermediates in row layout for the parity tests.
type mhcTrace struct {
	nx, preLogits, hPre, u, blockOut, y, postLogits, hPost, resLogits, hRes []float32
}

func (m *MHC) forward(stream []float32, t int, block func([]float32) ([]float32, error), tr *mhcTrace,
) ([]float32, error) {
	n, c := m.lanes, m.dim
	nC := n * c

	if t < 0 || len(stream) != t*nC {
		return nil, fmt.Errorf("needle: mHC stream has %d values, want %d rows of %d lanes × %d",
			len(stream), t, n, c)
	}

	nx := make([]float32, len(stream))
	RMSUnit(nx, stream, nC)

	hPre, hPost, hRes := m.logits(nx, t)
	if tr != nil {
		tr.nx = nx
		tr.preLogits, tr.postLogits, tr.resLogits = slices.Clone(hPre), slices.Clone(hPost), slices.Clone(hRes)
	}

	u := make([]float32, t*c)

	for i := range t {
		ui := u[i*c : (i+1)*c]
		for k := range n {
			hPre[i*n+k] = sigmoid(hPre[i*n+k])
			tensor.Axpy(ui, hPre[i*n+k], stream[(i*n+k)*c:(i*n+k+1)*c])
			hPost[i*n+k] = 2 * sigmoid(hPost[i*n+k])
		}

		sinkhorn(hRes[i*n*n:(i+1)*n*n], n, mhcSinkhornIters)
	}

	if tr != nil {
		tr.hPre, tr.hPost, tr.hRes, tr.u = hPre, hPost, hRes, slices.Clone(u)
	}

	out, err := block(u)
	if err != nil {
		return nil, err
	}

	if len(out) != len(u) {
		return nil, fmt.Errorf("needle: mHC block returned %d values for %d rows of %d", len(out), t, c)
	}

	y := make([]float32, len(u))
	for i, v := range u {
		y[i] = out[i] - v
	}

	if tr != nil {
		tr.blockOut, tr.y = slices.Clone(out), y
	}

	return m.mix(stream, t, hRes, hPost, y), nil
}

// logits returns the three heads' logits for the rows of nx: h_pre and
// h_post (t, n), h_res (t, n, n). The products are rounded before the adds
// (the conversions rule out FMA), as upstream's elementwise jnp ops are.
func (m *MHC) logits(nx []float32, t int) (pre, post, res []float32) {
	n := m.lanes
	pre, post, res = make([]float32, t*n), make([]float32, t*n), make([]float32, t*n*n)
	m.phiPre.Apply(pre, nx, t)
	m.phiPost.Apply(post, nx, t)
	m.phiRes.Apply(res, nx, t)

	for i := range t {
		for k := range n {
			pre[i*n+k] = float32(m.aPre*pre[i*n+k]) + m.bPre[k] + m.preOff[k]
			post[i*n+k] = float32(m.aPost*post[i*n+k]) + m.bPost[k] + m.postOff[k]
		}

		for k := range n * n {
			res[i*n*n+k] = float32(m.aRes*res[i*n*n+k]) + m.bRes[k]
		}
	}

	return pre, post, res
}

// mix is out[i] = Σ_j h_res[i, j] · stream[j] + h_post[i] · y per row.
func (m *MHC) mix(stream []float32, t int, hRes, hPost, y []float32) []float32 {
	n, c := m.lanes, m.dim
	out := make([]float32, len(stream))

	for i := range t {
		yi := y[i*c : (i+1)*c]

		for a := range n {
			o := out[(i*n+a)*c : (i*n+a+1)*c]
			for b := range n {
				tensor.Axpy(o, hRes[(i*n+a)*n+b], stream[(i*n+b)*c:(i*n+b+1)*c])
			}

			hp := hPost[i*n+a]
			for k, v := range yi {
				o[k] += float32(hp * v)
			}
		}
	}

	return out
}

// sinkhorn is architecture.py _sinkhorn on one n×n matrix of logits, in
// place: iters × (subtract each row's logsumexp, then each column's), then
// exp. The columns of the result sum to 1 up to float32 rounding, its rows
// up to the residual of iters iterations.
//
// Subtracting a row's logsumexp is dividing the row of exp(logits) by its
// sum, so sinkhornExp runs the same iteration on exp(logits − max) in
// float64, with n² exps instead of 2·n·(n+1)·iters exps and logs. Where
// the logits are not finite or spread too far for float64, sinkhornLog
// runs upstream's log-domain form (docs/FINDINGS.md F-035).
func sinkhorn(k []float32, n, iters int) {
	if !sinkhornExp(k, n, iters) {
		sinkhornLog(k, n, iters)
	}
}

// sinkhornExpRange bounds the logits' spread plus 2·iters·ln n for
// sinkhornExp. Every entry of exp(logits − max) starts in [e^−spread, 1]
// and stays at most 1, and each normalisation divides it by a sum of at
// most n, so all entries stay above e^−700 ≈ 1e-304, inside float64's
// normal range, and no sum is 0.
const sinkhornExpRange = 700

// sinkhornExp is sinkhorn in the exp domain in float64. It returns false
// without writing k if the logits are out of sinkhornExpRange.
func sinkhornExp(k []float32, n, iters int) bool {
	lo, hi := slices.Min(k), slices.Max(k)
	spread := float64(hi) - float64(lo)

	if math.IsInf(spread, 0) || math.IsNaN(spread) ||
		spread+2*float64(iters)*math.Log(float64(n)) > sinkhornExpRange {
		return false
	}

	var buf [16]float64

	e := buf[:]
	if n*n > len(buf) {
		e = make([]float64, n*n)
	}

	e = e[:n*n]
	for i, v := range k {
		e[i] = math.Exp(float64(v) - float64(hi))
	}

	for range iters {
		for r := range n {
			row := e[r*n : (r+1)*n]

			var s float64
			for _, v := range row {
				s += v
			}

			for c := range row {
				row[c] /= s
			}
		}

		for c := range n {
			var s float64
			for r := range n {
				s += e[r*n+c]
			}

			for r := range n {
				e[r*n+c] /= s
			}
		}
	}

	for i, v := range e {
		k[i] = float32(v)
	}

	return true
}

// sinkhornLog is sinkhorn in the log domain in float32, as upstream
// computes it. Staying in the log domain keeps any finite logits finite,
// where a direct exp overflows float32 above 88.7; each logsumexp shifts
// by its maximum as jax.nn.logsumexp does.
func sinkhornLog(k []float32, n, iters int) {
	for range iters {
		for r := range n {
			row := k[r*n : (r+1)*n]
			l := logSumExp(row, 0, 1, n)

			for c := range row {
				row[c] -= l
			}
		}

		for c := range n {
			l := logSumExp(k, c, n, n)

			for r := range n {
				k[r*n+c] -= l
			}
		}
	}

	for i, v := range k {
		k[i] = float32(math.Exp(float64(v)))
	}
}

// logSumExp is log Σ exp over the n values k[off], k[off+stride], …:
// max + log Σ exp(v − max) in float32, as jax.nn.logsumexp.
func logSumExp(k []float32, off, stride, n int) float32 {
	m := float32(math.Inf(-1))
	for i := range n {
		m = max(m, k[off+i*stride])
	}

	if math.IsInf(float64(m), 0) {
		m = 0 // jax.nn.logsumexp uses 0 where the max is not finite
	}

	var s float32
	for i := range n {
		s += float32(math.Exp(float64(k[off+i*stride] - m)))
	}

	return m + float32(math.Log(float64(s)))
}
