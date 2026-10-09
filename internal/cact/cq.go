package cact

import (
	"encoding/binary"
	"fmt"
	"math"
)

// ternaryRecordBits is export.py TERNARY_RECORD_BITS: the Record.Bits value
// of a ternary tensor, whose trits are stored as 2-bit crumbs.
const ternaryRecordBits = 5

// CQ dequantises a CQ tensor into a row-major [out][in] matrix, as export.py
// _cq_unpack does (docs/FINDINGS.md F-004). Each row holds in_pad = in
// rounded up to the group size indices in one LSB-first bitstream; per
// group, the addressed levels are scaled by the FP16 norm and rotated back
// by the normalised Sylvester Hadamard H/sqrt(group). The padding columns
// are dropped.
func (f *File) CQ(i int) ([]float32, error) {
	blob, err := f.blob(i, CQ)
	if err != nil {
		return nil, err
	}

	rec := f.records[i]

	cb, err := f.cqLevels(rec)
	if err != nil {
		return nil, fmt.Errorf("cact: tensor %d: %w", i, err)
	}

	w, err := dequantCQ(rec, blob, cb)
	if err != nil {
		return nil, fmt.Errorf("cact: tensor %d: %w", i, err)
	}

	return w, nil
}

// cqLevels returns the codebook a record's indices address. 2 to 4 bits use
// the header codebook, which _pack_cact scales by 1/sqrt(group) of the
// export; it writes one group size for the whole file (F-004). The 1-bit
// and ternary levels are analytic, as in quantize.py _cq_codebook_np.
func (f *File) cqLevels(rec Record) ([]float32, error) {
	var unit []float64

	switch rec.Bits {
	case 1:
		c := math.Sqrt(2 / math.Pi)
		unit = []float64{-c, c}
	case ternaryRecordBits:
		unit = []float64{-1.2240064, 0, 1.2240064} // quantize.py _TERNARY_CB
	default:
		return f.Codebook(int(rec.Bits))
	}

	cb := make([]float32, len(unit))
	for k, c := range unit {
		cb[k] = float32(c / math.Sqrt(float64(rec.Group)))
	}

	return cb, nil
}

// dequantCQ decodes a CQ blob, which Parse has sized per
// _packed_row_bytes: out rows of packed indices, then out*groups FP16 norms.
func dequantCQ(rec Record, blob []byte, cb []float32) ([]float32, error) {
	out, in, group := int(rec.Shape[0]), int(rec.Shape[1]), int(rec.Group)

	// _cq_hadamard_np doubles H until it covers the group, so any other size
	// fails the matmul upstream.
	if group&(group-1) != 0 {
		return nil, fmt.Errorf("CQ group %d is not a power of two", group)
	}

	groups := (in + group - 1) / group
	inPad := groups * group

	width := int(rec.Bits)
	if rec.Bits == ternaryRecordBits {
		width = 2
	}

	rowBytes := inPad * width / 8
	packed, norms := blob[:out*rowBytes], blob[out*rowBytes:]

	// H/sqrt(group) is folded into the norm; fwht applies the ±1 part.
	scale := float32(1 / math.Sqrt(float64(group)))

	w := make([]float32, out*in)
	idx := make([]uint8, inPad)
	buf := make([]float32, group)

	for r := range out {
		unpackLSB(idx, packed[r*rowBytes:(r+1)*rowBytes], width)

		if rec.Bits == ternaryRecordBits {
			err := crumbsToTrits(idx)
			if err != nil {
				return nil, fmt.Errorf("row %d: %w", r, err)
			}
		}

		for g := range groups {
			s := float16ToFloat32(binary.LittleEndian.Uint16(norms[2*(r*groups+g):])) * scale
			for k, j := range idx[g*group : (g+1)*group] {
				buf[k] = cb[j] * s
			}

			fwht(buf)

			col := g * group
			copy(w[r*in+col:r*in+min(col+group, in)], buf)
		}
	}

	return w, nil
}

// unpackLSB is export.py _unpack_lsb for one row: index k occupies bits
// [k*width, (k+1)*width) of an LSB-first bitstream. Upstream reads it in
// chunks of 8 indices from width little-endian bytes, and so does this.
// len(idx) is a multiple of 8 and len(packed) = len(idx)*width/8.
func unpackLSB(idx []uint8, packed []byte, width int) {
	mask := uint64(1)<<width - 1

	for c := range len(idx) / 8 {
		var word uint64
		for b, v := range packed[c*width : (c+1)*width] {
			word |= uint64(v) << (8 * b)
		}

		for k := range 8 {
			idx[8*c+k] = uint8(word >> (k * width) & mask)
		}
	}
}

// crumbsToTrits maps the stored crumbs 3, 0, 1 onto the trits 0, 1, 2, as
// export.py _unpack_ternary_crumbs. Crumb 2 encodes nothing; upstream would
// index past the 3-level codebook.
func crumbsToTrits(idx []uint8) error {
	for k, c := range idx {
		switch c {
		case 3:
			idx[k] = 0
		case 0, 1:
			idx[k] = c + 1
		default:
			return fmt.Errorf("ternary crumb %d at column %d is not 0, 1 or 3", c, k)
		}
	}

	return nil
}

// fwht multiplies x in place by the ±1 Sylvester Hadamard matrix of
// _cq_hadamard_np, H_2n = [[H_n, H_n], [H_n, -H_n]]. H is symmetric, so x@H
// and H@x agree; len(x) is a power of two.
func fwht(x []float32) {
	for h := 1; h < len(x); h *= 2 {
		for i := 0; i < len(x); i += 2 * h {
			for j := i; j < i+h; j++ {
				a, b := x[j], x[j+h]
				x[j], x[j+h] = a+b, a-b
			}
		}
	}
}
