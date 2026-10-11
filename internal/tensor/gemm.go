// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor/gemm.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: large products split over Parallel (docs/FINDINGS.md F-038).

package tensor

import "fmt"

// MatMulTransB computes the m×n product of a [m×k] and the transpose of
// b [n×k], both row-major with row stride k:
//
//	c[i*ldc+j] = dot(a[i*k:(i+1)*k], b[j*k:(j+1)*k])   for i < m, j < n
//
// Entries of c outside those positions are left untouched. Callers add bias
// or accumulate themselves.
//
// On arm64 a register-blocked NEON kernel computes 4×4 blocks of c, so each
// loaded vector feeds four FMAs instead of one. Elsewhere it is one
// DotProduct per output, the same arithmetic as before.
//
// A product of at least 2·gemmPieceFMAs multiply-adds runs on Parallel:
// over rows from 8 rows of a on, else over rows of b, in pieces that start
// at multiples of 4. Every 4×4 and 1×4 block then stays as it is in one
// call, so every output is the same bit for bit for any worker count
// (docs/FINDINGS.md F-038).
func MatMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	if m < 0 || n < 0 || k < 0 || ldc < n {
		panic(fmt.Sprintf("tensor: MatMulTransB bad dims m=%d n=%d k=%d ldc=%d", m, n, k, ldc))
	}

	if m == 0 || n == 0 {
		return
	}

	if len(a) < m*k || len(b) < n*k || len(c) < (m-1)*ldc+n {
		panic(fmt.Sprintf("tensor: MatMulTransB short slices: len(a)=%d len(b)=%d len(c)=%d for m=%d n=%d k=%d ldc=%d",
			len(a), len(b), len(c), m, n, k, ldc))
	}

	if getWorkers() > 1 && m*n*k >= 2*gemmPieceFMAs {
		matMulTransBParallel(c, ldc, a, b, m, n, k)
		return
	}

	matMulTransB(c, ldc, a, b, m, n, k)
}

// gemmPieceFMAs is the least work of one Parallel piece of MatMulTransB,
// about 15 µs on one NEON core: below it, waking a helper costs more than
// it saves (docs/FINDINGS.md F-038).
const gemmPieceFMAs = 1 << 19

// matMulTransBParallel is matMulTransB split over the rows of a (from 8
// rows on) or else of b, in pieces of whole 4-row blocks.
func matMulTransBParallel(c []float32, ldc int, a, b []float32, m, n, k int) {
	if m >= 8 {
		Parallel(m, gemmGrain(n*k), func(lo, hi int) {
			matMulTransB(c[lo*ldc:], ldc, a[lo*k:hi*k], b, hi-lo, n, k)
		})

		return
	}

	Parallel(n, gemmGrain(m*k), func(lo, hi int) {
		matMulTransB(c[lo:], ldc, a, b[lo*k:hi*k], m, hi-lo, k)
	})
}

// gemmGrain is the number of rows, a multiple of 4, that holds at least
// gemmPieceFMAs multiply-adds at perRow each.
func gemmGrain(perRow int) int {
	g := (gemmPieceFMAs + perRow - 1) / perRow

	return (g + 3) &^ 3
}

func matMulTransBGeneric(c []float32, ldc int, a, b []float32, m, n, k int) {
	for i := range m {
		aRow := a[i*k : (i+1)*k]
		cRow := c[i*ldc : i*ldc+n]

		for j := range cRow {
			cRow[j] = dotF32(aRow, b[j*k:(j+1)*k])
		}
	}
}
