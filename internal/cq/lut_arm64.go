//go:build arm64 && !purego

package cq

// lutAsm reports whether the lut kernel runs in assembly: on arm64 always,
// as every AArch64 CPU has NEON (lut_arm64.s, F-034).
const lutAsm = true

// fwhtF32 is fwht; the NEON kernel takes n >= 8.
func fwhtF32(x []float32) {
	if len(x) < 8 {
		fwht(x)
		return
	}

	fwhtNEON(&x[0], len(x))
}

func lutTables2(tab []byte, x, cb []float32, cbMax float32) float32 {
	_ = tab[16*len(x)-1] // len(x)/2 tables of 32 bytes
	_ = cb[3]

	return lutTables2NEON(&tab[0], &x[0], len(x), &cb[0], cbMax)
}

func lutTables4(tab []byte, x, cb []float32, cbMax float32) float32 {
	_ = tab[32*len(x)-1]
	_ = cb[15]

	return lutTables4NEON(&tab[0], &x[0], len(x), &cb[0], cbMax)
}

func lutLookup(y []float32, idx, tab []byte, norm, steps []float32, blocks, groups, gb int) {
	_ = y[blocks*lutRows-1]
	_ = idx[blocks*groups*gb*lutRows-1]
	_ = tab[64*groups*gb-1]
	_ = norm[blocks*groups*lutRows-1]
	_ = steps[groups-1]

	lutLookupNEON(&y[0], &idx[0], &tab[0], &norm[0], &steps[0], blocks, groups, gb)
}

// fwhtNEON sets x[:n] to H_n · x[:n] with fwht's butterflies in fwht's
// order; n is a power of two >= 8.
//
//go:noescape
func fwhtNEON(x *float32, n int)

// lutTables2NEON is lutTables2Generic for len(x) = n, a multiple of 8.
//
//go:noescape
func lutTables2NEON(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32

// lutTables4NEON is lutTables4Generic for len(x) = n, a multiple of 4.
//
//go:noescape
func lutTables4NEON(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32

// lutLookupNEON is lutLookupGeneric; blocks, groups and gb are >= 1.
//
//go:noescape
func lutLookupNEON(y *float32, idx, tab *byte, norm, steps *float32, blocks, groups, gb int)
