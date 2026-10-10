package cact

import (
	"encoding/binary"
	"fmt"
)

// Packed is a CQ tensor in its stored form, for kernels that multiply by it
// without dequantising (internal/cq, docs/FINDINGS.md F-028). Index and Norms
// alias the File's data and must not be modified.
//
// Row r of the [Out, In] matrix is Groups() groups of Group weights; the last
// group is zero-padded past In. Per group g, the dequantised weights are
//
//	w[r, g·Group : (g+1)·Group] = H · (Codebook[idx] · Norm(r, g) / √Group)
//
// with H the ±1 Sylvester Hadamard matrix, as CQ computes them (F-004).
type Packed struct {
	Out, In  int
	Group    int       // weights per norm, a power of two
	Bits     int       // Record.Bits: 1 to 4, or 5 for ternary crumbs
	Width    int       // bits per stored index: Bits, or 2 for ternary
	Codebook []float32 // the levels an index addresses; for ternary, by trit
	// Index holds Out rows of RowBytes() bytes: the LSB-first bitstream of
	// the group-padded row's indices. Ternary rows hold crumbs 3, 0, 1 for
	// the trits 0, 1, 2.
	Index []byte
	Norms []byte // Out·Groups() little-endian FP16 norms, [out][groups]
}

// Groups is the number of groups per row.
func (p Packed) Groups() int { return (p.In + p.Group - 1) / p.Group }

// RowBytes is the length of one row's index bitstream.
func (p Packed) RowBytes() int { return p.Groups() * p.Group * p.Width / 8 }

// Norm returns the FP16 norm of row r, group g, widened to float32.
func (p Packed) Norm(r, g int) float32 {
	return float16ToFloat32(binary.LittleEndian.Uint16(p.Norms[2*(r*p.Groups()+g):]))
}

// CQPacked returns CQ tensor i without dequantising it: the index bitstream
// and the FP16 norms as stored, and the codebook CQ would use.
func (f *File) CQPacked(i int) (Packed, error) {
	blob, err := f.blob(i, CQ)
	if err != nil {
		return Packed{}, err
	}

	rec := f.records[i]

	// As in dequantCQ: no Sylvester Hadamard for other group sizes.
	if rec.Group&(rec.Group-1) != 0 {
		return Packed{}, fmt.Errorf("cact: tensor %d: CQ group %d is not a power of two", i, rec.Group)
	}

	cb, err := f.cqLevels(rec)
	if err != nil {
		return Packed{}, fmt.Errorf("cact: tensor %d: %w", i, err)
	}

	p := Packed{
		Out: int(rec.Shape[0]), In: int(rec.Shape[1]), Group: int(rec.Group),
		Bits: int(rec.Bits), Width: int(rec.Bits), Codebook: cb,
	}
	if rec.Bits == ternaryRecordBits {
		p.Width = 2
	}

	// Parse has sized the blob per _packed_row_bytes.
	n := p.Out * p.RowBytes()
	p.Index, p.Norms = blob[:n:n], blob[n:]

	return p, nil
}
