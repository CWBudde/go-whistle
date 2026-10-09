//go:build (!amd64 && !arm64) || purego

// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor/dot_noasm.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: || purego added to the build constraint.

package tensor

func dotF32(a, b []float32) float32 {
	return dotF32Generic(a, b)
}
