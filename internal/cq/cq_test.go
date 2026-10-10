package cq

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
)

var kernels = []kernel{kernelGather, kernelNibble, kernelNibble4}

// randomPacked is a [out, in] CQ tensor with uniform random indices, FP16
// norms between 2^-4 and 2^4 and a sorted random codebook, laid out as
// cact.CQPacked returns it.
func randomPacked(rng *rand.Rand, out, in, group, bits int) cact.Packed {
	p := cact.Packed{Out: out, In: in, Group: group, Bits: bits, Width: bits}

	p.Codebook = make([]float32, 1<<bits)
	for i := range p.Codebook {
		p.Codebook[i] = float32(rng.NormFloat64() / math.Sqrt(float64(group)))
	}

	slices.Sort(p.Codebook)

	p.Index = make([]byte, out*p.RowBytes())
	for i := range p.Index {
		p.Index[i] = byte(rng.Uint32())
	}

	for range out * p.Groups() {
		exp := uint16(11 + rng.IntN(9)) // 2^-4 .. 2^4
		p.Norms = binary.LittleEndian.AppendUint16(p.Norms, exp<<10|uint16(rng.IntN(1024)))
	}

	return p
}

// sylvester is the ±1 Hadamard matrix of order n.
func sylvester(n int) [][]float64 {
	h := [][]float64{{1}}
	for len(h) < n {
		k := len(h)
		next := make([][]float64, 2*k)

		for i := range next {
			next[i] = make([]float64, 2*k)
			for j := range next[i] {
				v := h[i%k][j%k]
				if i >= k && j >= k {
					v = -v
				}

				next[i][j] = v
			}
		}

		h = next
	}

	return h
}

// dequant64 is the packed form's documented meaning in float64, with an
// explicit Hadamard matrix: w[r, g] = H · cb[idx] · norm / √G, cut to In.
func dequant64(p cact.Packed) [][]float64 {
	h := sylvester(p.Group)
	sqrtG := math.Sqrt(float64(p.Group))
	w := make([][]float64, p.Out)

	for r := range p.Out {
		w[r] = make([]float64, p.In)
		row := p.Index[r*p.RowBytes():]

		for col := range p.In {
			g, j := col/p.Group, col%p.Group

			for k := range p.Group {
				bit := (g*p.Group + k) * p.Width
				idx := row[bit/8] >> (bit % 8) & (1<<p.Width - 1)
				w[r][col] += float64(p.Codebook[idx]) * float64(p.Norm(r, g)) * h[j][k] / sqrtG
			}
		}
	}

	return w
}

// cosine of two vectors in float64.
func cosine(a []float32, b []float64) float64 {
	var ab, aa, bb float64

	for i := range a {
		ab += float64(a[i]) * b[i]
		aa += float64(a[i]) * float64(a[i])
		bb += b[i] * b[i]
	}

	return ab / math.Sqrt(aa*bb)
}

// rotatedBound bounds the float32 error of each y[r] in the rotated domain.
// Every term is cb[idx] · n · x̂_k with |x̂_k| ≤ Σ_j |x_g,j| / √G; the FWHT
// rounds log2(G) times, the scale, the lookup tables and the products a few
// more, and the sum over In terms once each, so |Δ| ≤ γ(In + log2(G) + 8) ·
// Σ_g n · Σ_k |cb[idx]| · Σ_j |x_g,j| / √G, with γ(k) = ku / (1 − ku).
func rotatedBound(p cact.Packed, x []float32) []float64 {
	k := float64(p.Groups()*p.Group) + math.Log2(float64(p.Group)) + 8
	gamma := k * 0x1p-24 / (1 - k*0x1p-24)
	bound := make([]float64, p.Out)

	for r := range p.Out {
		row := p.Index[r*p.RowBytes():]

		for g := range p.Groups() {
			var xAbs, cAbs float64
			for c := g * p.Group; c < min((g+1)*p.Group, p.In); c++ {
				xAbs += math.Abs(float64(x[c]))
			}

			for j := range p.Group {
				bit := (g*p.Group + j) * p.Width
				cAbs += math.Abs(float64(p.Codebook[row[bit/8]>>(bit%8)&(1<<p.Width-1)]))
			}

			bound[r] += gamma * float64(p.Norm(r, g)) * cAbs * xAbs / math.Sqrt(float64(p.Group))
		}
	}

	return bound
}

// testShapes cover one and several groups, padding (In not a multiple of
// the group, down to one real column in the last group) and groups 8 to
// 128, for both widths.
var testShapes = []struct{ out, in, group int }{
	{1, 8, 8},
	{3, 16, 8},
	{4, 13, 8},
	{5, 40, 32},
	{7, 200, 128},
	{6, 384, 128},
	{9, 129, 128},
}

// TestKernels compares every kernel's GEMM with the float64 product of the
// explicit dequantisation, on random packed matrices: per element within
// a float32 bound, and by cosine.
func TestKernels(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	for _, s := range testShapes {
		for _, bits := range []int{2, 4} {
			p := randomPacked(rng, s.out, s.in, s.group, bits)

			m, err := New(p)
			if err != nil {
				t.Fatalf("%v bits %d: New: %v", s, bits, err)
			}

			w := dequant64(p)

			const rows = 6 // one block of four, two leftover rows

			x := make([]float32, rows*s.in)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}

			for _, k := range kernels {
				got := make([]float32, rows*s.out)
				m.gemm(k, got, x, rows)

				for i := range rows {
					xi := x[i*s.in : (i+1)*s.in]
					bound := rotatedBound(p, xi)
					want := make([]float64, s.out)

					for r := range s.out {
						for c := range s.in {
							want[r] += w[r][c] * float64(xi[c])
						}

						if d := math.Abs(float64(got[i*s.out+r]) - want[r]); d > bound[r] {
							t.Errorf("%v bits %d %s: y[%d][%d] = %v, want %v (|Δ| %.3g > %.3g)",
								s, bits, k, i, r, got[i*s.out+r], want[r], d, bound[r])
						}
					}

					if c := cosine(got[i*s.out:(i+1)*s.out], want); c < 0.999999 {
						t.Errorf("%v bits %d %s: row %d cosine %.9f", s, bits, k, i, c)
					}
				}
			}
		}
	}
}

// TestGEMMRows checks that a GEMM row does not depend on the other rows:
// reversing the rows of a block reverses the result bit for bit. GEMV runs
// the one-row kernel, whose sums are ordered differently, so it agrees with
// GEMM to rounding.
func TestGEMMRows(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))

	m, err := New(randomPacked(rng, 33, 256, 128, 2))
	if err != nil {
		t.Fatal(err)
	}

	const rows = 4

	x, rev := make([]float32, rows*256), make([]float32, rows*256)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	for i := range rows {
		copy(rev[i*256:(i+1)*256], x[(rows-1-i)*256:(rows-i)*256])
	}

	all, back := make([]float32, rows*33), make([]float32, rows*33)
	m.GEMM(all, x, rows)
	m.GEMM(back, rev, rows)

	for i := range rows {
		if !slices.Equal(all[i*33:(i+1)*33], back[(rows-1-i)*33:(rows-i)*33]) {
			t.Errorf("GEMM row %d changes with the row order", i)
		}

		one := make([]float32, 33)
		m.GEMV(one, x[i*256:(i+1)*256])

		want := make([]float64, 33)
		for r, v := range all[i*33 : (i+1)*33] {
			want[r] = float64(v)
		}

		if c := cosine(one, want); c < 0.9999999 {
			t.Errorf("GEMV row %d: cosine %.9f to GEMM", i, c)
		}
	}
}

// TestRow checks Row against the float64 dequantisation and Rows against
// the full matrix.
func TestRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))

	for _, s := range testShapes {
		for _, bits := range []int{2, 4} {
			p := randomPacked(rng, s.out, s.in, s.group, bits)

			m, err := New(p)
			if err != nil {
				t.Fatal(err)
			}

			w := dequant64(p)
			got := make([]float32, s.in)

			for r := range s.out {
				m.Row(got, r)

				if c := cosine(got, w[r]); c < 0.9999999 {
					t.Errorf("%v bits %d: Row(%d) cosine %.9f", s, bits, r, c)
				}
			}

			lo, hi := s.out/3, s.out-s.out/4
			v := m.Rows(lo, hi)

			x := make([]float32, s.in)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}

			full, part := make([]float32, s.out), make([]float32, hi-lo)
			m.GEMV(full, x)
			v.GEMV(part, x)

			if v.Out() != hi-lo || !slices.Equal(part, full[lo:hi]) {
				t.Errorf("%v bits %d: Rows(%d, %d) differs from the full product", s, bits, lo, hi)
			}

			row := make([]float32, s.in)

			for r := lo; r < hi; r++ {
				m.Row(got, r)
				v.Row(row, r-lo)

				if !slices.Equal(got, row) {
					t.Errorf("%v bits %d: Rows(%d, %d).Row(%d) differs", s, bits, lo, hi, r-lo)
				}
			}
		}
	}
}

func TestNewErrors(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	good := randomPacked(rng, 2, 128, 128, 2)

	for _, tc := range []struct {
		name string
		edit func(p *cact.Packed)
		want string
	}{
		{"3 bits", func(p *cact.Packed) { p.Bits, p.Width = 3, 3 }, "not supported"},
		{"ternary", func(p *cact.Packed) { p.Bits = 5 }, "not supported"},
		{"group 24", func(p *cact.Packed) { p.Group = 24 }, "power of two"},
		{"codebook", func(p *cact.Packed) { p.Codebook = p.Codebook[:3] }, "codebook"},
		{"index", func(p *cact.Packed) { p.Index = p.Index[1:] }, "index"},
		{"norms", func(p *cact.Packed) { p.Norms = p.Norms[2:] }, "norm"},
	} {
		p := good
		tc.edit(&p)

		_, err := New(p)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one mentioning %q", tc.name, err, tc.want)
		}
	}
}

func TestGEMMPanics(t *testing.T) {
	m, err := New(randomPacked(rand.New(rand.NewPCG(9, 10)), 4, 128, 128, 4))
	if err != nil {
		t.Fatal(err)
	}

	for name, f := range map[string]func(){
		"short x":   func() { m.GEMV(make([]float32, 4), make([]float32, 127)) },
		"short dst": func() { m.GEMV(make([]float32, 3), make([]float32, 128)) },
		"rows":      func() { m.GEMM(make([]float32, 8), make([]float32, 128), 2) },
		"row":       func() { m.Row(make([]float32, 128), 4) },
		"view":      func() { m.Rows(3, 5) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", name)
				}
			}()

			f()
		}()
	}
}

// TestFWHT pins fwht to the explicit matrix.
func TestFWHT(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))

	for _, n := range []int{1, 2, 8, 128} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.IntN(17) - 8)
		}

		h := sylvester(n)
		want := make([]float32, n)

		for i := range n {
			var s float64
			for j := range n {
				s += h[i][j] * float64(x[j])
			}

			want[i] = float32(s)
		}

		fwht(x)

		if !slices.Equal(x, want) {
			t.Errorf("fwht n=%d = %v, want %v", n, x, want)
		}
	}
}
