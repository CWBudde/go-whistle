//go:build amd64 && !purego

package needle

import "golang.org/x/sys/cpu"

// useAVX2 is true when the CPU has AVX2, which the int8 kernels need
// (VPMOVSXBW, VPMADDWD, VPMOVSXBD on 256-bit registers).
var useAVX2 = cpu.X86.HasAVX2

// dotsI8Kernel runs the AVX2 kernel when the CPU has it and the row width
// is a multiple of 16, the Go loop otherwise.
func dotsI8Kernel(dst []int32, q, k []int8, stride int) {
	if !useAVX2 || len(q) == 0 || len(q)%16 != 0 {
		dotsI8Generic(dst, q, k, stride)

		return
	}

	dotsI8AVX2(&dst[0], &q[0], &k[0], len(dst), len(q), stride)
}

// axpysI8Kernel runs the AVX2 kernel when the CPU has it and the output
// width is a multiple of 16, the Go loop otherwise.
func axpysI8Kernel(out, w []float32, v []int8, stride int) {
	if !useAVX2 || len(out) == 0 || len(out)%16 != 0 {
		axpysI8Generic(out, w, v, stride)

		return
	}

	axpysI8AVX2(&out[0], &w[0], &v[0], len(w), len(out), stride)
}

// dotsI8AVX2 is dotsI8 for n ≥ 1 keys and a width that is a positive
// multiple of 16.
//
//go:noescape
func dotsI8AVX2(dst *int32, q, k *int8, n, width, stride int)

// axpysI8AVX2 is axpysI8 for n ≥ 1 keys and a width that is a positive
// multiple of 16.
//
//go:noescape
func axpysI8AVX2(out, w *float32, v *int8, n, width, stride int)
