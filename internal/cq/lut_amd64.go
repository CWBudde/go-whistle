//go:build amd64 && !purego

package cq

import "golang.org/x/sys/cpu"

// lutAsm reports whether the lut kernel runs in assembly: on amd64 with
// AVX2 (lut_amd64.s, F-034). Without it the Go reference runs, and best()
// keeps the nibble4 kernel.
var lutAsm = cpu.X86.HasAVX2

// fwhtF32 is fwht; the AVX2 kernel takes n >= 8.
func fwhtF32(x []float32) {
	if !lutAsm || len(x) < 8 {
		fwht(x)
		return
	}

	fwhtAVX2(&x[0], len(x))
}

func lutTables2(tab []byte, x, cb []float32, cbMax float32) float32 {
	if !lutAsm {
		return lutTables2Generic(tab, x, cb, cbMax)
	}

	_ = tab[16*len(x)-1] // len(x)/2 tables of 32 bytes
	_ = cb[3]

	return lutTables2AVX2(&tab[0], &x[0], len(x), &cb[0], cbMax)
}

func lutTables4(tab []byte, x, cb []float32, cbMax float32) float32 {
	if !lutAsm {
		return lutTables4Generic(tab, x, cb, cbMax)
	}

	_ = tab[32*len(x)-1]
	_ = cb[15]

	return lutTables4AVX2(&tab[0], &x[0], len(x), &cb[0], cbMax)
}

func lutLookup(y []float32, idx, tab []byte, norm, steps []float32, blocks, groups, gb int) {
	if !lutAsm {
		lutLookupGeneric(y, idx, tab, norm, steps, blocks, groups, gb)
		return
	}

	if blocks == 0 {
		return
	}

	_ = y[blocks*lutRows-1]
	_ = idx[blocks*groups*gb*lutRows-1]
	_ = tab[64*groups*gb-1]
	_ = norm[blocks*groups*lutRows-1]
	_ = steps[groups-1]

	lutLookupAVX2(&y[0], &idx[0], &tab[0], &norm[0], &steps[0], blocks, groups, gb)
}

// fwhtAVX2 sets x[:n] to H_n · x[:n] with fwht's butterflies in fwht's
// order; n is a power of two >= 8.
//
//go:noescape
func fwhtAVX2(x *float32, n int)

// lutTables2AVX2 is lutTables2Generic for len(x) = n, a multiple of 8.
//
//go:noescape
func lutTables2AVX2(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32

// lutTables4AVX2 is lutTables4Generic for len(x) = n, a multiple of 8.
//
//go:noescape
func lutTables4AVX2(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32

// lutLookupAVX2 is lutLookupGeneric; blocks, groups and gb are >= 1.
//
//go:noescape
func lutLookupAVX2(y *float32, idx, tab *byte, norm, steps *float32, blocks, groups, gb int)
