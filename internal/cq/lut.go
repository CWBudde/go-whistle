package cq

import (
	"math"
	"sync"
)

// The lut kernel (docs/FINDINGS.md F-034) is the nibble kernel recast for
// byte shuffles (NEON TBL, AVX2 VPSHUFB): one instruction looks up 16 or
// 32 rows at once, where the scalar kernels look up one.
//
//   - Tables. Per activation group, every nibble table entry is rounded to
//     an integer v in [−lutQ, lutQ] with one scale per group, M/lutQ, where
//     M bounds the group's entries. v is stored as two byte planes, lo =
//     v & 127 and hi = v>>7 + lutBias, both in [0, 127], so a shuffle reads
//     each and two of them add without overflowing a byte.
//   - Index. The index is repacked once into blocks of lutRows rows: per
//     block and index byte, the rows' bytes are adjacent, so one vector load
//     holds one index byte of 16 or 32 rows.
//   - Sums. Per row and group, the lo and hi lookups sum exactly in 16 bits;
//     the group's integer 128·Σhi + Σlo − bias times norm · scale is added
//     to y in float32. The integer sums are exact, so the assembly and the
//     Go reference give the same bits.
//
// The rotation is the plain FWHT; its 1/√G goes into the group scale.

const (
	lutRows = 32   // rows per block of the repacked index
	lutQ    = 8191 // table entries are rounded to integers in [−lutQ, lutQ]
	lutBias = 64   // offset of the hi plane: v>>7 ∈ [−64, 63] is stored + 64

	// lutMaxGroupBytes bounds the index bytes per group: a group sums
	// 2·groupBytes entries of each plane, at most 127 each, in uint16.
	lutMaxGroupBytes = 256
)

// lutLayout is a matrix's index and norms in blocks of lutRows rows, built
// once and shared by the matrix and its Rows views. Rows past the last are
// zero.
type lutLayout struct {
	once                  sync.Once
	index                 []byte    // the root matrix's packed rows
	norms                 []float32 // the root matrix's [out][groups] norms
	out, rowBytes, groups int

	// idx[(b·rowBytes + j)·lutRows + i] is index byte j of row b·lutRows + i.
	idx []byte
	// norm[(b·groups + g)·lutRows + i] is the norm of row b·lutRows + i, group g.
	norm []float32
}

// blocks returns the repacked index and norms, building them on first use.
func (l *lutLayout) blocks() (idx []byte, norm []float32) {
	l.once.Do(func() {
		nb := (l.out + lutRows - 1) / lutRows
		l.idx = make([]byte, nb*l.rowBytes*lutRows)
		l.norm = make([]float32, nb*l.groups*lutRows)

		for r := range l.out {
			b, i := r/lutRows, r%lutRows

			for j, v := range l.index[r*l.rowBytes : (r+1)*l.rowBytes] {
				l.idx[(b*l.rowBytes+j)*lutRows+i] = v
			}

			for g, n := range l.norms[r*l.groups : (r+1)*l.groups] {
				l.norm[(b*l.groups+g)*lutRows+i] = n
			}
		}
	})

	return l.idx, l.norm
}

// lutSpan is the range of blocks [b0, b1) that holds rows [row0, row0+out)
// of the root matrix.
func (m *Matrix) lutSpan() (b0, b1 int) {
	return m.row0 / lutRows, (m.row0 + m.out + lutRows - 1) / lutRows
}

// gemmLUT runs the lut kernel for each activation: rotate, build the
// tables group by group, look up every block of the matrix's rows.
func (m *Matrix) gemmLUT(s *scratch, dst, x []float32, rows int) {
	idx, norm := m.lut.blocks()
	b0, b1 := m.lutSpan()
	nb := b1 - b0
	idx = idx[b0*m.rowBytes*lutRows : b1*m.rowBytes*lutRows]
	norm = norm[b0*m.groups*lutRows : b1*m.groups*lutRows]

	if cap(s.y) < nb*lutRows {
		s.y = make([]float32, nb*lutRows)
	}

	if s.tab == nil {
		s.tab = make([]byte, 64*m.rowBytes)
	}

	if len(s.steps) < m.groups { // matrices of one row layout can differ in group size
		s.steps = make([]float32, m.groups)
	}

	y := s.y[:nb*lutRows]
	off := m.row0 - b0*lutRows
	tabBytes := 64 * m.groupBytes
	pad := m.groups * m.group

	for i := range rows {
		xh := s.xh[:pad]
		copy(xh, x[i*m.in:(i+1)*m.in])
		clear(xh[m.in:])

		for g := range m.groups {
			xg := xh[g*m.group : (g+1)*m.group]
			fwhtF32(xg)

			tg := s.tab[g*tabBytes : (g+1)*tabBytes]
			if m.width == 2 {
				s.steps[g] = lutTables2(tg, xg, m.cb, m.cbMax) * m.lutStep
			} else {
				s.steps[g] = lutTables4(tg, xg, m.cb, m.cbMax) * m.lutStep
			}
		}

		lutLookup(y, idx, s.tab, norm, s.steps, nb, m.groups, m.groupBytes)
		copy(dst[i*m.out:(i+1)*m.out], y[off:off+m.out])
	}
}

// lutStepOf is the factor from a group's bound M to its integer step in y:
// M/lutQ for the rounding and 1/√G for the rotation, rounded once.
func lutStepOf(group int) float32 {
	return float32(1 / (lutQ * math.Sqrt(float64(group))))
}

// lutInv is the factor from a table entry to its integer, lutQ/M, and 0
// for an all-zero group.
func lutInv(m float32) float32 {
	if m > 0 {
		return lutQ / m
	}

	return 0
}

// lutPut stores entry c of a table, e already scaled to [−lutQ, lutQ],
// rounded to nearest even as the vector conversions do.
func lutPut(t []byte, c int, e float32) {
	v := int32(math.RoundToEven(float64(e)))
	t[c] = byte(v & 127)
	t[16+c] = byte(v>>7 + lutBias)
}

// lutTables2Generic builds the 2-bit tables of one rotated group x: per
// index pair q, 32 bytes, lo then hi plane, for the 16 codes hi<<2|lo of
// cb[lo]·x[2q] + cb[hi]·x[2q+1]. It returns the group's bound M,
// cbMax · max_q (|x[2q]| + |x[2q+1]|).
func lutTables2Generic(tab []byte, x, cb []float32, cbMax float32) float32 {
	var mx float32
	for q := 0; q < len(x); q += 2 {
		mx = max(mx, abs32(x[q])+abs32(x[q+1]))
	}

	m := cbMax * mx
	inv := lutInv(m)

	for q := range len(x) / 2 {
		a, b := x[2*q]*inv, x[2*q+1]*inv
		t := tab[32*q : 32*q+32]

		for c := range 16 {
			lutPut(t, c, float32(cb[c&3]*a)+float32(cb[c>>2]*b))
		}
	}

	return m
}

// lutTables4Generic builds the 4-bit tables of one rotated group x: per
// index q, 32 bytes for the 16 codes c of cb[c]·x[q]. It returns the
// group's bound M, cbMax · max_q |x[q]|.
func lutTables4Generic(tab []byte, x, cb []float32, cbMax float32) float32 {
	var mx float32
	for _, v := range x {
		mx = max(mx, abs32(v))
	}

	m := cbMax * mx
	inv := lutInv(m)

	for q, v := range x {
		a := v * inv
		t := tab[32*q : 32*q+32]

		for c := range 16 {
			lutPut(t, c, cb[c]*a)
		}
	}

	return m
}

// lutLookupGeneric sets y[b·lutRows + i] for the first `blocks` blocks of
// the repacked index: Σ_g float32(128·Σhi + Σlo − bias) · (norm · steps[g]),
// with the sums over the group's index bytes j, the low nibble in table 2j
// and the high one in table 2j+1. bias removes lutBias from the 2·gb hi
// entries of a group.
func lutLookupGeneric(y []float32, idx, tab []byte, norm, steps []float32, blocks, groups, gb int) {
	bias := int32(128 * lutBias * 2 * gb)
	rowBytes := groups * gb

	for b := range blocks {
		yb := y[b*lutRows : (b+1)*lutRows]
		clear(yb)

		for g, step := range steps[:groups] {
			for i := range yb {
				var lo, hi int32

				for j := g * gb; j < (g+1)*gb; j++ {
					v := idx[(b*rowBytes+j)*lutRows+i]
					t := tab[64*j : 64*j+64]
					lo += int32(t[v&15]) + int32(t[32+v>>4])
					hi += int32(t[16+v&15]) + int32(t[48+v>>4])
				}

				coef := norm[(b*groups+g)*lutRows+i] * step
				yb[i] += float32(float32(hi<<7+lo-bias) * coef)
			}
		}
	}
}

func abs32(v float32) float32 { return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)) }
