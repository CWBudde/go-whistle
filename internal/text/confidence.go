package text

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/needle"
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
//
// The three matrices are read by rows: a copy of the float32 values, or
// from a packed source (CQ4 in needle3.cact) the row dequantised as
// cact.CQ does it, bit for bit, one row at a time (docs/FINDINGS.md F-059).
type probeHead struct {
	rows, k, q, out, dim int

	probes  needle.Linear // (rows·k, dim): probes[l, j]
	gain    []float32     // (rows, k)
	query   needle.Linear // (q, dim)
	rowBias []float32     // (q, rows, k)
	proj    needle.Linear // (out, q·dim): HeadProjection's kernel transposed
	bias    []float32     // (out)
}

// probeTensors are a probe head's names after its key, in the order
// loadProbeHead checks their shapes.
var probeTensors = [...]string{"probes", "gain", "query", "row_bias", "proj", "bias"}

// loadProbeHead loads the head key from src, or returns nil when src does
// not carry it. rows is the number of cell rows, Layers + 1.
func loadProbeHead(src weights.Source, key string, rows, dim int) (*probeHead, error) {
	if !src.Has(key + ".probes") {
		return nil, nil //nolint:nilnil // no head is not an error
	}

	var shapes [len(probeTensors)][]int

	for i, name := range probeTensors {
		shape, err := src.Shape(key + "." + name)
		if err != nil {
			return nil, fmt.Errorf("text: %w", err)
		}

		shapes[i] = shape
	}

	probes, query, proj := shapes[0], shapes[2], shapes[4]
	if len(probes) != 2 || len(query) != 2 || len(proj) != 2 || probes[0]%rows != 0 {
		return nil, fmt.Errorf("text: %s shapes %v", key, shapes)
	}

	h := &probeHead{rows: rows, dim: dim, k: probes[0] / rows, q: query[0], out: proj[0]}

	want := [len(probeTensors)][]int{
		{rows * h.k, dim}, {rows, h.k}, {h.q, dim}, {h.q, rows, h.k}, {h.out, h.q * dim}, {h.out},
	}
	for i := range shapes {
		if !slices.Equal(shapes[i], want[i]) {
			return nil, fmt.Errorf("text: %s shapes %v, want %v", key, shapes, want)
		}
	}

	var err error

	table := func(dst *needle.Linear, i int) {
		if err == nil {
			*dst, err = needle.LoadTable(src, key+"."+probeTensors[i], want[i][0], want[i][1])
		}
	}
	vector := func(dst *[]float32, i int) {
		if err == nil {
			*dst, _, err = src.Tensor(key + "." + probeTensors[i])
		}
	}

	table(&h.probes, 0)
	vector(&h.gain, 1)
	table(&h.query, 2)
	vector(&h.rowBias, 3)
	table(&h.proj, 4)
	vector(&h.bias, 5)

	if err != nil {
		return nil, fmt.Errorf("text: %w", err)
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
	probe := make([]float32, d)

	for l := range h.rows {
		cell := cells[l]

		for j := range h.k {
			h.probes.Row(probe, l*h.k+j)

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
	query := probe

	for q := range h.q {
		h.query.Row(query, q)

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
	row := make([]float32, len(pooled))

	for o := range out {
		h.proj.Row(row, o)
		out[o] = float32(dot(pooled, row) + float64(h.bias[o]))
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
