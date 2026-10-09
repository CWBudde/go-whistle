//go:build amd64 && !purego

// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor/axpy_amd64.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: !purego added to the build constraint.

package tensor

func axpyF32(dst []float32, alpha float32, src []float32) {
	n := len(dst)
	if useAVX2FMA && n >= 8 {
		n8 := n &^ 7
		axpyF32AVX2(&dst[0], &src[0], alpha, n8)

		if n8 == n {
			return
		}

		axpyF32Generic(dst[n8:], alpha, src[n8:])

		return
	}

	axpyF32Generic(dst, alpha, src)
}

//go:noescape
func axpyF32AVX2(dst, src *float32, alpha float32, n int)
