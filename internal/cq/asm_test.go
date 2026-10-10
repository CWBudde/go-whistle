package cq

import (
	"bytes"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// The tests in this file compare each dispatching routine of the lut kernel
// (fwhtF32, lutTables2, lutTables4, lutLookup) with its pure-Go reference,
// as AGENTS.md requires for every assembly routine. The assembly does the
// same float32 operations in the same order, so the results must be equal
// bit for bit (F-034). With the purego build tag, on an architecture
// without assembly, or on amd64 without AVX2, both sides are the reference
// and the tests pass trivially.

// asmVector is n random floats starting at offset off of a larger slice,
// so the assembly also sees operands off any vector alignment. heavy gives
// heavy tails (Gaussian cubed); about one value in 16 is an exact zero.
func asmVector(rng *rand.Rand, n, off int, heavy bool) []float32 {
	x := make([]float32, n+off)[off:]
	for i := range x {
		v := rng.NormFloat64()
		if heavy {
			v = v * v * v
		}

		if rng.IntN(16) == 0 {
			v = 0
		}

		x[i] = float32(v)
	}

	return x
}

func TestFWHTMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))

	for n := 1; n <= 1024; n *= 2 {
		for off := range 3 {
			x := asmVector(rng, n, off, off == 2)
			want := slices.Clone(x)

			fwhtF32(x)
			fwht(want)

			if !slices.Equal(x, want) {
				t.Fatalf("n=%d off=%d: fwhtF32 differs from fwht", n, off)
			}
		}
	}
}

// TestLUTTablesMatchGeneric covers group sizes 8 to 256, both widths, an
// all-zero group (M = 0) and heavy tails; the table bytes and M must be
// equal, and the bytes past the tables untouched.
func TestLUTTablesMatchGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 24))

	for _, bits := range []int{2, 4} {
		for n := 8; n <= 256; n *= 2 {
			for trial := range 4 {
				cb := make([]float32, 1<<bits)
				for i := range cb {
					cb[i] = float32(rng.NormFloat64())
				}

				slices.Sort(cb)

				var cbMax float32
				for _, c := range cb {
					cbMax = max(cbMax, abs32(c))
				}

				x := asmVector(rng, n, trial%3, trial == 3)
				if trial == 0 {
					clear(x)
				}

				size := 32 * n * bits / 4
				got := bytes.Repeat([]byte{0xAA}, size+5)[1:]
				want := bytes.Repeat([]byte{0xAA}, size+4)

				var mGot, mWant float32
				if bits == 2 {
					mGot = lutTables2(got[:size], x, cb, cbMax)
					mWant = lutTables2Generic(want[:size], x, cb, cbMax)
				} else {
					mGot = lutTables4(got[:size], x, cb, cbMax)
					mWant = lutTables4Generic(want[:size], x, cb, cbMax)
				}

				if math.Float32bits(mGot) != math.Float32bits(mWant) {
					t.Fatalf("bits %d n=%d trial %d: M = %v, generic %v", bits, n, trial, mGot, mWant)
				}

				if i := firstDiff(got, want); i >= 0 {
					t.Fatalf("bits %d n=%d trial %d: table byte %d = %#x, generic %#x", bits, n, trial, i, got[i], want[i])
				}
			}
		}
	}
}

// TestLUTLookupMatchesGeneric runs the lookup on random repacked indices
// and tables (plane bytes in [0, 127], as lutPut writes them), across
// blocks, groups and group widths up to lutMaxGroupBytes; one case has all
// entries at 127, where the uint16 sums are fullest. The float past the
// output must stay untouched.
func TestLUTLookupMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(25, 26))

	for _, gb := range []int{1, 2, 3, 4, 8, 16, 32, 64, 256} {
		for _, groups := range []int{1, 2, 5} {
			for _, blocks := range []int{1, 2, 3} {
				full := gb == 256 && groups == 1

				idx := make([]byte, blocks*groups*gb*lutRows)
				for i := range idx {
					idx[i] = byte(rng.Uint32())
				}

				tab := make([]byte, 64*groups*gb)
				for i := range tab {
					tab[i] = byte(rng.IntN(128))
					if full {
						tab[i] = 127
					}
				}

				norm := asmVector(rng, blocks*groups*lutRows, 0, false)
				steps := asmVector(rng, groups, 0, false)

				got := asmVector(rng, blocks*lutRows+1, 0, false)
				want := slices.Clone(got)

				lutLookup(got[:blocks*lutRows], idx, tab, norm, steps, blocks, groups, gb)
				lutLookupGeneric(want[:blocks*lutRows], idx, tab, norm, steps, blocks, groups, gb)

				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("gb=%d groups=%d blocks=%d: y[%d] = %v, generic %v", gb, groups, blocks, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestLUTKernelMatchesGeneric runs the whole lut GEMM, through the
// dispatching routines and through the Go references, on random matrices
// and on Rows views that start and end inside a block: the outputs must be
// equal bit for bit.
func TestLUTKernelMatchesGeneric(t *testing.T) {
	t.Logf("lut assembly: %v", lutAsm)

	rng := rand.New(rand.NewPCG(27, 28))

	for _, s := range []struct{ out, in, group, bits int }{
		{1, 8, 8, 2}, {33, 40, 8, 4}, {70, 200, 128, 2}, {96, 512, 128, 2}, {45, 2048, 128, 4},
	} {
		m, err := New(randomPacked(rng, s.out, s.in, s.group, s.bits))
		if err != nil {
			t.Fatal(err)
		}

		for _, v := range []*Matrix{m, m.Rows(s.out/3, s.out-s.out/5)} {
			const rows = 3

			x := asmVector(rng, rows*v.in, 0, false)
			got := make([]float32, rows*v.out)
			want := make([]float32, rows*v.out)

			v.gemm(kernelLUT, got, x, rows)
			lutGEMMGeneric(v, want, x, rows)

			if !slices.Equal(got, want) {
				t.Errorf("%v rows [%d, %d): lut GEMM differs from the Go reference", s, v.row0, v.row0+v.out)
			}
		}
	}
}

// lutGEMMGeneric is gemmLUT with the Go reference routines.
func lutGEMMGeneric(m *Matrix, dst, x []float32, rows int) {
	idx, norm := m.lut.blocks()
	b0, b1 := m.lutSpan()
	nb := b1 - b0
	idx = idx[b0*m.rowBytes*lutRows : b1*m.rowBytes*lutRows]
	norm = norm[b0*m.groups*lutRows : b1*m.groups*lutRows]

	y := make([]float32, nb*lutRows)
	tab := make([]byte, 64*m.rowBytes)
	steps := make([]float32, m.groups)
	tabBytes := 64 * m.groupBytes

	for i := range rows {
		xh := make([]float32, m.groups*m.group)
		copy(xh, x[i*m.in:(i+1)*m.in])

		for g := range m.groups {
			xg := xh[g*m.group : (g+1)*m.group]
			fwht(xg)

			tg := tab[g*tabBytes : (g+1)*tabBytes]
			if m.width == 2 {
				steps[g] = lutTables2Generic(tg, xg, m.cb, m.cbMax) * m.lutStep
			} else {
				steps[g] = lutTables4Generic(tg, xg, m.cb, m.cbMax) * m.lutStep
			}
		}

		lutLookupGeneric(y, idx, tab, norm, steps, nb, m.groups, m.groupBytes)

		off := m.row0 - b0*lutRows
		copy(dst[i*m.out:(i+1)*m.out], y[off:off+m.out])
	}
}

// firstDiff is the first index where a and b differ, or -1.
func firstDiff(a, b []byte) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}

	if len(a) != len(b) {
		return min(len(a), len(b))
	}

	return -1
}
