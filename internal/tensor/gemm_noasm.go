//go:build !arm64 || purego

// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor/gemm_noarm64.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: renamed, || purego added to the build constraint.

package tensor

func matMulTransB(c []float32, ldc int, a, b []float32, m, n, k int) {
	matMulTransBGeneric(c, ldc, a, b, m, n, k)
}
