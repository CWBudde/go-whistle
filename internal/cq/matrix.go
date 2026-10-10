// Package cq multiplies by CQ matrices straight from their packed .cact form,
// in the rotated (Hadamard) domain, without dequantising the weights
// (docs/FINDINGS.md F-028).
//
// Per row r and group g, cact.CQ dequantises w[r, g] = H · c[r, g] · n[r, g]/√G
// with c the addressed codebook levels, n the FP16 norm and H the symmetric
// ±1 Sylvester Hadamard matrix of the group size G (F-004). So
//
//	y[r] = Σ_g w[r, g] · x_g = Σ_g n[r, g] · (c[r, g] · x̂_g),   x̂_g = H · x_g / √G
//
// and a product needs one FWHT per activation group instead of one per
// weight group. The rest is a table lookup per packed index byte.
package cq

import (
	"fmt"
	"sync"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// Matrix is a 2- or 4-bit CQ matrix [out, in] for y = W · x, as .cact stores
// it (F-028 lists the Whistle tensors this covers). It is safe for
// concurrent use.
type Matrix struct {
	out, in    int
	group      int
	groups     int
	width      int // bits per index, 2 or 4
	rowBytes   int
	groupBytes int
	index      []byte    // out rows of rowBytes; aliases cact.Packed.Index
	norms      []float32 // [out][groups], widened from FP16
	cb         []float32
	cbMax      float32    // max |cb|, for the lut kernel's table bound
	lutStep    float32    // lutStepOf(group)
	lut        *lutLayout // the lut kernel's blocked rows, shared with Rows views
	row0       int        // first row in lut: non-zero for a Rows view
	scratch    *sync.Pool // *scratch, shared by every Matrix of the same row layout
}

// New builds a Matrix from a packed CQ tensor. Index stays shared with p;
// the norms are widened to float32 once, and the lut kernel's blocked copy
// is built (Prepare).
func New(p cact.Packed) (*Matrix, error) {
	m, err := NewGather(p)
	if err != nil {
		return nil, err
	}

	m.Prepare()

	return m, nil
}

// NewGather builds a Matrix like New but leaves the lut kernel's blocked
// copy to the first GEMM or Prepare, for a tensor that Row reads and no
// product needs: the engram tables (docs/FINDINGS.md F-037).
func NewGather(p cact.Packed) (*Matrix, error) {
	switch {
	case p.Bits != 2 && p.Bits != 4, p.Width != p.Bits:
		return nil, fmt.Errorf("cq: %d-bit CQ (index width %d) not supported, only 2 and 4", p.Bits, p.Width)
	case p.Group < 8 || p.Group&(p.Group-1) != 0:
		return nil, fmt.Errorf("cq: group %d is not a power of two >= 8", p.Group)
	case p.Out < 0 || p.In <= 0:
		return nil, fmt.Errorf("cq: bad shape [%d, %d]", p.Out, p.In)
	case len(p.Codebook) != 1<<p.Bits:
		return nil, fmt.Errorf("cq: %d codebook levels for %d bits", len(p.Codebook), p.Bits)
	case len(p.Index) != p.Out*p.RowBytes() || len(p.Norms) != 2*p.Out*p.Groups():
		return nil, fmt.Errorf("cq: %d index and %d norm bytes for [%d, %d], want %d and %d",
			len(p.Index), len(p.Norms), p.Out, p.In, p.Out*p.RowBytes(), 2*p.Out*p.Groups())
	}

	m := &Matrix{
		out: p.Out, in: p.In, group: p.Group, groups: p.Groups(), width: p.Bits,
		rowBytes: p.RowBytes(), groupBytes: p.Group * p.Bits / 8,
		index: p.Index, norms: make([]float32, p.Out*p.Groups()), cb: p.Codebook,
	}

	for r := range m.out {
		for g := range m.groups {
			m.norms[r*m.groups+g] = p.Norm(r, g)
		}
	}

	for _, c := range m.cb {
		m.cbMax = max(m.cbMax, abs32(c))
	}

	m.lutStep = lutStepOf(m.group)
	m.lut = &lutLayout{index: m.index, norms: m.norms, out: m.out, rowBytes: m.rowBytes, groups: m.groups}
	m.scratch = scratchPool(m.groups*m.group, m.rowBytes)

	return m, nil
}

// Out is the number of rows, the length of y.
func (m *Matrix) Out() int { return m.out }

// In is the number of columns, the length of x.
func (m *Matrix) In() int { return m.in }

// Bits is the index width, 2 or 4.
func (m *Matrix) Bits() int { return m.width }

// Prepare builds the blocked copy of the index and the norms that the lut
// kernel multiplies by (F-034), if that kernel runs and the copy is not
// built yet. A GEMM calls it; calling it at load time keeps the work out
// of the first product. It is safe for concurrent use.
func (m *Matrix) Prepare() {
	if m.best() == kernelLUT {
		m.lut.blocks()
	}
}

// Bytes is the memory the matrix holds: the packed indices, the float32
// norms and the codebook, and once built the lut kernel's blocked copy of
// its rows' indices and norms (F-034). Index bytes shared with a file or
// another view count in full.
func (m *Matrix) Bytes() int {
	n := len(m.index) + 4*len(m.norms) + 4*len(m.cb)
	if m.best() == kernelLUT && m.lut.built.Load() {
		b0, b1 := m.lutSpan()
		n += (b1 - b0) * lutRows * (m.rowBytes + 4*m.groups)
	}

	return n
}

// Rows returns rows [lo, hi) as a Matrix that shares m's storage, for a
// tensor that stacks several matrices (the mHC φ of every layer, F-028).
func (m *Matrix) Rows(lo, hi int) *Matrix {
	if lo < 0 || hi < lo || hi > m.out {
		panic(fmt.Sprintf("cq: Rows(%d, %d) of %d rows", lo, hi, m.out))
	}

	v := *m
	v.out = hi - lo
	v.row0 = m.row0 + lo
	v.index = m.index[lo*m.rowBytes : hi*m.rowBytes : hi*m.rowBytes]
	v.norms = m.norms[lo*m.groups : hi*m.groups : hi*m.groups]

	return &v
}

// Row dequantises row r into dst (len In), with cact.CQ's arithmetic and
// so its result bit for bit: for gathers such as an embedding lookup.
func (m *Matrix) Row(dst []float32, r int) {
	if r < 0 || r >= m.out || len(dst) != m.in {
		panic(fmt.Sprintf("cq: Row(len %d, %d) of [%d, %d]", len(dst), r, m.out, m.in))
	}

	buf := make([]float32, m.group)
	row := m.index[r*m.rowBytes : (r+1)*m.rowBytes]
	scale := invSqrt(m.group)

	for g := range m.groups {
		s := m.norms[r*m.groups+g] * scale
		for k := range buf {
			buf[k] = m.cb[m.at(row, g*m.group+k)] * s
		}

		fwht(buf)

		col := g * m.group
		copy(dst[col:min(col+m.group, m.in)], buf)
	}
}

// at is index k of a packed row, LSB first. width divides 8, so an index
// never straddles a byte.
func (m *Matrix) at(row []byte, k int) uint8 {
	bit := k * m.width

	return row[bit/8] >> (bit % 8) & (1<<m.width - 1)
}
