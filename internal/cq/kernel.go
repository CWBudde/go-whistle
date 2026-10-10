package cq

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// kernel selects the inner product over the packed indices. All three give
// the same sum up to float32 rounding order (F-028 has the timings).
type kernel uint8

const (
	// kernelGather is the reference: per index, cb[idx] · x̂[k].
	kernelGather kernel = iota
	// kernelNibble looks up 16-entry tables, one per index nibble, built
	// per activation: two lookups per index byte.
	kernelNibble
	// kernelNibble4 interleaves the nibble tables of four activations, so
	// one index decode feeds four lookups; leftover rows use kernelNibble.
	kernelNibble4
	// kernelLUT rounds the nibble tables to 14-bit integers in two byte
	// planes and looks up a block of 32 rows per shuffle (lut.go, F-034).
	kernelLUT
)

func (k kernel) String() string {
	switch k {
	case kernelGather:
		return "gather"
	case kernelNibble:
		return "nibble"
	case kernelNibble4:
		return "nibble4"
	case kernelLUT:
		return "lut"
	default:
		return fmt.Sprintf("kernel %d", uint8(k))
	}
}

// scratch holds one product's rotated activations and lookup tables for a
// row layout: pad = Groups·Group activation floats, rowBytes index bytes.
type scratch struct {
	xh   []float32           // x̂ of up to four activations, rotated and group-padded
	nib  [][16]float32       // per index nibble, its value for each of the 16 codes; nil until a nibble kernel runs
	nib4 [][2][16][4]float32 // per index byte, both nib tables of four activations, interleaved; nil until a GEMM needs it

	// The lut kernel's state, nil until it runs: the byte-plane tables of
	// one activation, the group steps and the blocked output.
	tab   []byte
	steps []float32
	y     []float32
}

// scratchPools holds one *sync.Pool per row layout, so matrices of the
// same width share their scratch: Whistle has five layouts for 152
// matrices.
var scratchPools sync.Map // [2]int{pad, rowBytes} → *sync.Pool

func scratchPool(pad, rowBytes int) *sync.Pool {
	if p, ok := scratchPools.Load([2]int{pad, rowBytes}); ok {
		return p.(*sync.Pool) //nolint:forcetypeassert // only *sync.Pool is stored
	}

	p, _ := scratchPools.LoadOrStore([2]int{pad, rowBytes}, &sync.Pool{New: func() any {
		return &scratch{xh: make([]float32, 4*pad)}
	}})

	return p.(*sync.Pool) //nolint:forcetypeassert // only *sync.Pool is stored
}

// GEMV sets dst[r] = Σ_c W[r, c] · x[c] for every row r: len(dst) = Out,
// len(x) = In.
func (m *Matrix) GEMV(dst, x []float32) {
	m.GEMM(dst, x, 1)
}

// GEMM runs GEMV for each of rows activations: x is [rows, In] and dst
// [rows, Out], both row-major. Each activation is rotated once, and four
// at a time share one pass over the packed indices.
func (m *Matrix) GEMM(dst, x []float32, rows int) {
	m.gemm(m.best(), dst, x, rows)
}

// best is the fastest kernel measured on Whistle's shapes: lut where the
// assembly runs it (F-034), else nibble4 (F-028).
func (m *Matrix) best() kernel {
	if lutAsm && m.groupBytes <= lutMaxGroupBytes {
		return kernelLUT
	}

	return kernelNibble4
}

func (m *Matrix) gemm(k kernel, dst, x []float32, rows int) {
	if rows < 0 || len(x) != rows*m.in || len(dst) != rows*m.out {
		panic(fmt.Sprintf("cq: GEMM(len %d, len %d, %d rows) on [%d, %d]", len(dst), len(x), rows, m.out, m.in))
	}

	s, _ := m.scratch.Get().(*scratch)
	defer m.scratch.Put(s)

	if k == kernelLUT {
		m.gemmLUT(s, dst, x, rows)
		return
	}

	pad := m.groups * m.group
	i := 0

	if s.nib == nil {
		s.nib = make([][16]float32, 2*m.rowBytes)
	}

	if k == kernelNibble4 && rows >= 4 {
		if s.nib4 == nil {
			s.nib4 = make([][2][16][4]float32, m.rowBytes)
		}

		for ; i+4 <= rows; i += 4 {
			for a := range 4 {
				m.rotate(s.xh[a*pad:(a+1)*pad], x[(i+a)*m.in:(i+a+1)*m.in])
			}

			m.nibbleTables4(s.nib4, s.nib, s.xh)
			m.lookupNibble4(dst[i*m.out:(i+4)*m.out], s.nib4)
		}
	}

	for ; i < rows; i++ {
		xh := s.xh[:pad]
		m.rotate(xh, x[i*m.in:(i+1)*m.in])

		y := dst[i*m.out : (i+1)*m.out]
		if k == kernelGather {
			m.gather(y, xh)
			continue
		}

		m.nibbleTables(s.nib, xh)
		m.lookupNibble(y, s.nib)
	}
}

// rotate sets xh to x̂: each group of x, zero-padded past In, times
// H/√G. cact.CQ drops the padding columns' weights; the zeros in x drop
// them here.
func (m *Matrix) rotate(xh, x []float32) {
	copy(xh, x)
	clear(xh[m.in:])

	scale := invSqrt(m.group)

	for g := range m.groups {
		xg := xh[g*m.group : (g+1)*m.group]
		fwhtF32(xg)

		for k := range xg {
			xg[k] *= scale
		}
	}
}

// gather is the reference kernel: y[r] = Σ_g n[r, g] · Σ_k cb[idx] · x̂_g[k].
func (m *Matrix) gather(y, xh []float32) {
	for r := range m.out {
		row := m.index[r*m.rowBytes : (r+1)*m.rowBytes]

		var acc float32

		for g := range m.groups {
			var dot float32
			for k := range m.group {
				dot += m.cb[m.at(row, g*m.group+k)] * xh[g*m.group+k]
			}

			acc += dot * m.norms[r*m.groups+g]
		}

		y[r] = acc
	}
}

// nibbleTables sets nib[q][code] to the contribution of nibble q of a row
// when it holds code: for 4 bits cb[code] · x̂[q], for 2 bits the two
// indices cb[code&3] · x̂[2q] + cb[code>>2] · x̂[2q+1] (LSB first).
func (m *Matrix) nibbleTables(nib [][16]float32, xh []float32) {
	if m.width == 4 {
		for q, v := range xh {
			t := &nib[q]
			for c, l := range m.cb {
				t[c] = l * v
			}
		}

		return
	}

	for q := range nib {
		a, b := xh[2*q], xh[2*q+1]
		t := &nib[q]

		for hi := range 4 {
			hb := m.cb[hi] * b
			for lo := range 4 {
				t[hi<<2|lo] = m.cb[lo]*a + hb
			}
		}
	}
}

// lookupNibble sums the nibble tables over each group's index bytes: the
// low nibble is the even table, the high one the odd table. Eight index
// bytes are one little-endian word; the array pointers leave one bounds
// check per word.
func (m *Matrix) lookupNibble(y []float32, nib [][16]float32) {
	gb := m.groupBytes

	for r := range m.out {
		row := m.index[r*m.rowBytes : (r+1)*m.rowBytes]
		norms := m.norms[r*m.groups : (r+1)*m.groups]

		var acc float32

		for g, n := range norms {
			rg := row[g*gb : (g+1)*gb]
			tg := nib[2*g*gb : 2*(g+1)*gb]

			var a0, a1, a2, a3, a4, a5, a6, a7 float32

			j := 0
			for ; j+8 <= len(rg); j += 8 {
				w := binary.LittleEndian.Uint64(rg[j:])
				t := (*[16][16]float32)(tg[2*j:])
				a0 += t[0][w&15] + t[8][w>>32&15]
				a1 += t[1][w>>4&15] + t[9][w>>36&15]
				a2 += t[2][w>>8&15] + t[10][w>>40&15]
				a3 += t[3][w>>12&15] + t[11][w>>44&15]
				a4 += t[4][w>>16&15] + t[12][w>>48&15]
				a5 += t[5][w>>20&15] + t[13][w>>52&15]
				a6 += t[6][w>>24&15] + t[14][w>>56&15]
				a7 += t[7][w>>28&15] + t[15][w>>60]
			}

			for ; j < len(rg); j++ { // groups under 8 bytes
				a0 += tg[2*j][rg[j]&15] + tg[2*j+1][rg[j]>>4]
			}

			acc += (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) * n
		}

		y[r] = acc
	}
}

// nibbleTables4 interleaves the nibble tables of the four activations in
// xh: nib4[q/2][q%2][code][a] is activation a's nib[q][code]. nib is
// scratch.
func (m *Matrix) nibbleTables4(nib4 [][2][16][4]float32, nib [][16]float32, xh []float32) {
	pad := m.groups * m.group

	for a := range 4 {
		m.nibbleTables(nib, xh[a*pad:(a+1)*pad])

		for q := range nib {
			for c, v := range &nib[q] {
				nib4[q/2][q%2][c][a] = v
			}
		}
	}
}

// lookupNibble4 is lookupNibble for four activations: each index nibble
// selects four adjacent entries, one per activation. dst is [4, Out].
func (m *Matrix) lookupNibble4(dst []float32, nib4 [][2][16][4]float32) {
	gb := m.groupBytes
	y0, y1, y2, y3 := dst[:m.out], dst[m.out:2*m.out], dst[2*m.out:3*m.out], dst[3*m.out:4*m.out]

	for r := range m.out {
		row := m.index[r*m.rowBytes : (r+1)*m.rowBytes]
		norms := m.norms[r*m.groups : (r+1)*m.groups]

		var c0, c1, c2, c3 float32

		for g, n := range norms {
			rg := row[g*gb : (g+1)*gb]
			tg := nib4[g*gb : (g+1)*gb]

			// a holds the low nibbles' sums, b the high ones'.
			var a0, a1, a2, a3, b0, b1, b2, b3 float32

			j := 0
			for ; j+8 <= len(rg); j += 8 {
				w := binary.LittleEndian.Uint64(rg[j:])
				t := (*[8][2][16][4]float32)(tg[j:])

				for k := range t {
					e, f := &t[k][0][w&15], &t[k][1][w>>4&15]
					w >>= 8
					a0, a1, a2, a3 = a0+e[0], a1+e[1], a2+e[2], a3+e[3]
					b0, b1, b2, b3 = b0+f[0], b1+f[1], b2+f[2], b3+f[3]
				}
			}

			for ; j < len(rg); j++ { // groups under 8 bytes
				e, f := &tg[j][0][rg[j]&15], &tg[j][1][rg[j]>>4]
				a0, a1, a2, a3 = a0+e[0], a1+e[1], a2+e[2], a3+e[3]
				b0, b1, b2, b3 = b0+f[0], b1+f[1], b2+f[2], b3+f[3]
			}

			c0 += (a0 + b0) * n
			c1 += (a1 + b1) * n
			c2 += (a2 + b2) * n
			c3 += (a3 + b3) * n
		}

		y0[r], y1[r], y2[r], y3[r] = c0, c1, c2, c3
	}
}

// invSqrt is cact.CQ's 1/√G, rounded to float32 once.
func invSqrt(group int) float32 { return float32(1 / math.Sqrt(float64(group))) }

// fwht multiplies x in place by the ±1 Sylvester Hadamard matrix,
// H_2n = [[H_n, H_n], [H_n, -H_n]]; len(x) is a power of two. Duplicated
// from internal/cact (cq.go fwht), which keeps it unexported; this copy runs
// on activations, and fwhtF32 runs it in assembly with the same stages in
// the same order, so with the same bits (F-034).
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
