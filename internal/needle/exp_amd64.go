//go:build amd64 && !purego

package needle

import "golang.org/x/sys/cpu"

// useAVX2FMA is true when the CPU has AVX2 and FMA, which the kernel
// needs (VFMADD213PS, VPSLLD on 256-bit registers).
var useAVX2FMA = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// expSubSumF32 runs the AVX2 kernel over the multiple-of-8 part of s and
// the Go loop over the rest, or the Go loop alone without AVX2 and FMA.
func expSubSumF32(s []float32, m float32) float32 {
	n8 := len(s) &^ 7
	if !useAVX2FMA || n8 == 0 {
		return expSubSumGeneric(s, m)
	}

	sum := expSubSumAVX2(&s[0], m, n8)
	if n8 == len(s) {
		return sum
	}

	return sum + expSubSumGeneric(s[n8:], m)
}

// expSubSumAVX2 sets the n floats at s to exp(s[i] − m) and returns their
// sum. n must be a positive multiple of 8.
//
//go:noescape
func expSubSumAVX2(s *float32, m float32, n int) float32
