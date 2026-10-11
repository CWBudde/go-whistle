package needle

import (
	"fmt"
	"math"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// normEps is ZCRMSNorm's and _rms_unit's epsilon.
const normEps = 1e-6

// ZCRMSNorm is architecture.py ZCRMSNorm, the zero-centred RMS norm over
// the last axis: y = (1 + scale) · x / sqrt(mean(x²) + 1e-6). There is no
// mean subtraction and no bias.
type ZCRMSNorm struct {
	gain []float32 // 1 + scale
}

// NewZCRMSNorm returns the norm with the given scale (the stored
// parameter, not 1 + scale).
func NewZCRMSNorm(scale []float32) ZCRMSNorm {
	gain := make([]float32, len(scale))
	for i, s := range scale {
		gain[i] = 1 + s
	}

	return ZCRMSNorm{gain: gain}
}

// LoadZCRMSNorm loads the norm whose scale is the (n) tensor name, e.g.
// "layer00.norm_in" or "enc00.q_norm".
func LoadZCRMSNorm(src weights.Source, name string) (ZCRMSNorm, error) {
	l := &loader{src: src}

	scale := l.vector(name, 0)
	if l.err != nil {
		return ZCRMSNorm{}, l.err
	}

	return NewZCRMSNorm(scale), nil
}

// Dim is the width the norm runs over.
func (n ZCRMSNorm) Dim() int { return len(n.gain) }

// Apply normalises each row of Dim values of x into dst. dst may be x; it
// panics unless len(dst) = len(x) is a multiple of Dim.
func (n ZCRMSNorm) Apply(dst, x []float32) {
	d := len(n.gain)
	checkRows("ZCRMSNorm", dst, x, d)

	forRows(len(x)/d, normPieceValues/d, func(lo, hi int) {
		for r := lo * d; r < hi*d; r += d {
			row, out := x[r:r+d], dst[r:r+d]
			rms := rowRMS(row)

			for c, g := range n.gain {
				out[c] = g * row[c] / rms
			}
		}
	})
}

// RMSUnit is architecture.py _rms_unit: each row of dim values of x
// scaled to unit RMS, x · rsqrt(mean(x²) + 1e-6), written to dst. dst may
// be x; it panics unless len(dst) = len(x) is a multiple of dim.
func RMSUnit(dst, x []float32, dim int) {
	checkRows("RMSUnit", dst, x, dim)

	forRows(len(x)/dim, normPieceValues/dim, func(lo, hi int) {
		for r := lo * dim; r < hi*dim; r += dim {
			row, out := x[r:r+dim], dst[r:r+dim]
			inv := float32(1 / math.Sqrt(float64(meanSquare(row)+normEps)))

			for c, v := range row {
				out[c] = v * inv
			}
		}
	})
}

// rowRMS is sqrt(mean(row²) + eps), rounded like upstream's float32.
func rowRMS(row []float32) float32 {
	return float32(math.Sqrt(float64(meanSquare(row) + normEps)))
}

// meanSquare is mean(row²), accumulated in float64 and rounded to float32
// (docs/FINDINGS.md F-011). Four partial sums break the chain of
// dependent adds; in float64 the order does not show in the float32
// result (F-035).
func meanSquare(row []float32) float32 {
	var s0, s1, s2, s3 float64

	i := 0
	for ; i+3 < len(row); i += 4 {
		v := row[i : i+4 : i+4]
		s0 += float64(v[0]) * float64(v[0])
		s1 += float64(v[1]) * float64(v[1])
		s2 += float64(v[2]) * float64(v[2])
		s3 += float64(v[3]) * float64(v[3])
	}

	for ; i < len(row); i++ {
		s0 += float64(row[i]) * float64(row[i])
	}

	return float32((s0 + s1 + s2 + s3) / float64(len(row)))
}

func checkRows(op string, dst, x []float32, dim int) {
	if dim <= 0 || len(x)%dim != 0 || len(dst) != len(x) {
		panic(fmt.Sprintf("needle: %s on %d values into %d with dim %d", op, len(x), len(dst), dim))
	}
}
