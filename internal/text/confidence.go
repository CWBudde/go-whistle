package text

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// ErrNoHead reports a probe head the model's weights do not carry.
var ErrNoHead = errors.New("text: the model has no such probe head")

// confidenceKey is the ConfidenceHead's name prefix (cact.TextNames).
const confidenceKey = "confidence_head"

// probeHead is one of upstream's ProbeHeads (needle/model/architecture.py
// @ af2654dffa: probe_pool, ProbeHead, HeadProjection) in the .cact layout
// of export.py, in float32: k probes and a gain per cell row (the scaled
// embedding, then every block), q queries with a bias per row and probe,
// and the projection of the q pooled vectors to out values.
type probeHead struct {
	rows, k, q, out, dim int

	probes  []float32 // (rows·k, dim): probes[l, j]
	gain    []float32 // (rows, k)
	query   []float32 // (q, dim)
	rowBias []float32 // (q, rows, k)
	proj    []float32 // (out, q·dim): HeadProjection's kernel transposed
	bias    []float32 // (out)
}

// loadProbeHead loads the head key from src, or returns nil when src does
// not carry it. rows is the number of cell rows, Layers + 1.
func loadProbeHead(src weights.Source, key string, rows, dim int) (*probeHead, error) {
	if !src.Has(key + ".probes") {
		return nil, nil //nolint:nilnil // no head is not an error
	}

	h := &probeHead{rows: rows, dim: dim}

	var shapes [6][]int

	for i, p := range []*[]float32{&h.probes, &h.gain, &h.query, &h.rowBias, &h.proj, &h.bias} {
		name := key + "." + [...]string{"probes", "gain", "query", "row_bias", "proj", "bias"}[i]

		v, shape, err := src.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("text: %w", err)
		}

		*p, shapes[i] = v, shape
	}

	probes, query, proj := shapes[0], shapes[2], shapes[4]
	if len(probes) != 2 || len(query) != 2 || len(proj) != 2 || probes[0]%rows != 0 {
		return nil, fmt.Errorf("text: %s shapes %v", key, shapes)
	}

	h.k, h.q, h.out = probes[0]/rows, query[0], proj[0]

	want := [6][]int{{rows * h.k, dim}, {rows, h.k}, {h.q, dim}, {h.q, rows, h.k}, {h.out, h.q * dim}, {h.out}}
	for i := range shapes {
		if !slices.Equal(shapes[i], want[i]) {
			return nil, fmt.Errorf("text: %s shapes %v, want %v", key, shapes, want)
		}
	}

	return h, nil
}

// pool is probe_pool for one sequence over the first keep positions of
// cells (rows slices of at least keep·dim values) with every row kept
// (row_keep None): per row l and probe j an attention over the positions,
//
//	s[l,j,t] = cells[t,l] · probes[l,j] / √d,  r[l,j] = Σₜ softmaxₜ(s) cells[t,l]
//	r[l,j]   = rms_unit(r[l,j]) · gain[l,j]
//	u[q,l,j] = r[l,j] · query[q] / √d + row_bias[q,l,j]
//	out[q]   = Σ_{l,j} softmax_{l,j}(u[q]) r[l,j]
//
// returned as q·dim values, query-major.
func (h *probeHead) pool(cells [][]float32, keep int) []float32 {
	d, m := h.dim, h.rows*h.k
	inv := 1 / math.Sqrt(float64(d))
	r := make([]float32, m*d)
	scores := make([]float64, keep)
	acc := make([]float64, d)

	for l := range h.rows {
		cell := cells[l]

		for j := range h.k {
			probe := h.probes[(l*h.k+j)*d : (l*h.k+j+1)*d]
			for t := range keep {
				scores[t] = dot(cell[t*d:(t+1)*d], probe) * inv
			}

			softmax(scores)
			clear(acc)

			for t := range keep {
				row, w := cell[t*d:(t+1)*d], scores[t]
				for c, x := range row {
					acc[c] += w * float64(x)
				}
			}

			rmsUnit(r[(l*h.k+j)*d:(l*h.k+j+1)*d], acc, h.gain[l*h.k+j])
		}
	}

	out := make([]float32, h.q*d)
	u := make([]float64, m)

	for q := range h.q {
		query := h.query[q*d : (q+1)*d]
		for i := range m {
			u[i] = dot(r[i*d:(i+1)*d], query)*inv + float64(h.rowBias[q*m+i])
		}

		softmax(u)
		clear(acc)

		for i, w := range u {
			for c, x := range r[i*d : (i+1)*d] {
				acc[c] += w * float64(x)
			}
		}

		for c, v := range acc {
			out[q*d+c] = float32(v)
		}
	}

	return out
}

// project is HeadProjection: pooled · kernel + bias.
func (h *probeHead) project(pooled []float32) []float32 {
	out := make([]float32, h.out)

	for o := range out {
		n := len(pooled)
		out[o] = float32(dot(pooled, h.proj[o*n:(o+1)*n]) + float64(h.bias[o]))
	}

	return out
}

// rmsUnit sets dst = x · rsqrt(mean(x²) + 1e-6) · gain, upstream's
// _rms_unit (no learned scale) and the probe gain.
func rmsUnit(dst []float32, x []float64, gain float32) {
	var ss float64
	for _, v := range x {
		ss += v * v
	}

	s := float64(gain) / math.Sqrt(ss/float64(len(x))+1e-6)
	for c, v := range x {
		dst[c] = float32(v * s)
	}
}

// softmax replaces x by its softmax.
func softmax(x []float64) {
	peak := math.Inf(-1)
	for _, v := range x {
		peak = max(peak, v)
	}

	var sum float64

	for i, v := range x {
		x[i] = math.Exp(v - peak)
		sum += x[i]
	}

	for i := range x {
		x[i] /= sum
	}
}

func dot(a, b []float32) float64 {
	var s float64
	for i, v := range a {
		s += float64(v) * float64(b[i])
	}

	return s
}

// HasConfidence reports whether the model carries the ConfidenceHead.
func (m *Model) HasConfidence() bool { return m.confidence != nil }

// Confidence returns the ConfidenceHead's raw logit (upstream's
// forward_confidence, before any sigmoid) pooled over the first keep
// positions of out, a Forward output with the per-block lane means
// (hidden true). The cells are upstream's hidden_cells: the scaled input
// embedding, then the lane mean after each of the Layers blocks.
//
// Upstream pools over every position whose token is not PAD; keep chooses
// the span instead, e.g. the prompt or the prompt and its completion. The
// mask is causal, so the first keep positions of a longer output pool as
// forward_confidence on those keep tokens does. The tokens must not hold
// PAD (id 0), which upstream would leave out. Without the head Confidence
// fails with ErrNoHead.
func (m *Model) Confidence(out *Output, keep int) (float32, error) {
	h := m.confidence
	if h == nil {
		return 0, fmt.Errorf("%w: %s", ErrNoHead, confidenceKey)
	}

	d := m.cfg.Dim
	t := len(out.Embed) / d

	if len(out.Hidden) != m.cfg.Layers {
		return 0, fmt.Errorf("text: confidence needs the %d per-block hidden states (Forward with hidden), have %d",
			m.cfg.Layers, len(out.Hidden))
	}

	if keep < 1 || keep > t {
		return 0, fmt.Errorf("text: confidence over %d of %d positions", keep, t)
	}

	cells := make([][]float32, 0, h.rows)
	cells = append(cells, out.Embed)
	cells = append(cells, out.Hidden...)

	return h.project(h.pool(cells, keep))[0], nil
}
