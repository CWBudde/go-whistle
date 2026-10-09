// Package tensor is a dense, row-major float32 tensor with the kernels the
// model blocks build on: MatMulTransB (GEMM), DotProduct and Axpy, plus
// Linear, MatMul, Softmax, broadcasting and shape operations.
//
// The kernels have NEON assembly on arm64 and AVX2+FMA assembly on amd64
// (chosen at run time from golang.org/x/sys/cpu). Every assembly routine has
// a pure-Go fallback that other architectures use and that the purego build
// tag forces everywhere; asm_test.go compares each kernel with its fallback.
//
// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor @
// 9faba38 without LayerNorm; each file names its source (docs/FINDINGS.md
// F-008).
package tensor
