package cact

import (
	"encoding/binary"
	"math"
)

// Internals for the external cact_test package (names_whistle_test.go),
// which also imports internal/tokenizer and so cannot live in package cact.
var (
	Float32ToFloat16 = float32ToFloat16
	Float16ToFloat32 = float16ToFloat32
	DequantCQ        = dequantCQ
)

// Blob returns tensor i's raw record bytes, whatever its dtype.
func (f *File) Blob(i int) []byte {
	rec := f.records[i]

	return f.data[rec.Offset : rec.Offset+rec.NBytes]
}

// RequantCQ is portPack's blob for 1-4 bit widths, with the rotation done by
// a float64 fast Walsh-Hadamard transform instead of the explicit matrix:
// H is symmetric, so the butterflies compute g @ H in O(group log group).
// margin[n] is BoundaryMargin of index n's unit coordinate, in the order the
// blob packs the indices. TestRequantCQMatchesPortPack pins it to portPack.
func RequantCQ(w []float32, out, in, width, group int, cb []float32) (blob []byte, margin []float64) {
	groups := (in + group - 1) / group
	inPad := groups * group
	invSqrt := 1 / math.Sqrt(float64(group))

	codes := make([]uint8, out*inPad)
	margin = make([]float64, out*inPad)
	norms := make([]byte, 0, 2*out*groups)
	rot := make([]float64, group)

	for r := range out {
		for gi := range groups {
			for k := range rot {
				rot[k] = 0
				if col := gi*group + k; col < in {
					rot[k] = float64(w[r*in+col])
				}
			}

			fwht64(rot)

			var sq float64

			for k := range rot {
				rot[k] *= invSqrt
				sq += rot[k] * rot[k]
			}

			n := math.Sqrt(sq)
			norms = binary.LittleEndian.AppendUint16(norms, float32ToFloat16(float32(n)))

			for k, v := range rot {
				u := v / math.Max(n, 1e-12)
				codes[r*inPad+gi*group+k] = nearestLevel(u, cb)
				margin[r*inPad+gi*group+k] = BoundaryMargin(u, cb)
			}
		}
	}

	return append(packLSB(codes, width), norms...), margin
}

// BoundaryMargin is the distance from u to the nearest decision boundary of
// the sorted codebook cb, the midpoint between two adjacent levels. Within
// that distance a rounding error can move u to the neighbouring level.
func BoundaryMargin(u float64, cb []float32) float64 {
	m := math.Inf(1)
	for k := 1; k < len(cb); k++ {
		m = min(m, math.Abs(u-(float64(cb[k-1])+float64(cb[k]))/2))
	}

	return m
}

// fwht64 is fwht in float64: the unnormalised Sylvester transform.
func fwht64(x []float64) {
	for h := 1; h < len(x); h *= 2 {
		for i := 0; i < len(x); i += 2 * h {
			for j := i; j < i+h; j++ {
				a, b := x[j], x[j+h]
				x[j], x[j+h] = a+b, a-b
			}
		}
	}
}
