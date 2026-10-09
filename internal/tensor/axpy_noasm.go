//go:build (!amd64 && !arm64) || purego

// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor/axpy_noamd64.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: renamed, || purego added to the build constraint.

package tensor

func axpyF32(dst []float32, alpha float32, src []float32) {
	axpyF32Generic(dst, alpha, src)
}
