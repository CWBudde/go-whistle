//go:build (!amd64 && !arm64) || purego

package cq

// lutAsm reports whether the lut kernel runs in assembly; without it the
// Go reference runs, and best() keeps the nibble4 kernel.
const lutAsm = false

func fwhtF32(x []float32) { fwht(x) }

func lutTables2(tab []byte, x, cb []float32, cbMax float32) float32 {
	return lutTables2Generic(tab, x, cb, cbMax)
}

func lutTables4(tab []byte, x, cb []float32, cbMax float32) float32 {
	return lutTables4Generic(tab, x, cb, cbMax)
}

func lutLookup(y []float32, idx, tab []byte, norm, steps []float32, blocks, groups, gb int) {
	lutLookupGeneric(y, idx, tab, norm, steps, blocks, groups, gb)
}
