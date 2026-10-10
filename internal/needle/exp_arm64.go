//go:build arm64 && !purego

package needle

// expSubSumF32 runs the NEON kernel over the multiple-of-4 part of s and
// the Go loop over the rest.
func expSubSumF32(s []float32, m float32) float32 {
	n4 := len(s) &^ 3
	if n4 == 0 {
		return expSubSumGeneric(s, m)
	}

	sum := expSubSumNEON(&s[0], m, n4)
	if n4 == len(s) {
		return sum
	}

	return sum + expSubSumGeneric(s[n4:], m)
}

// expSubSumNEON sets the n floats at s to exp(s[i] − m) and returns their
// sum. n must be a positive multiple of 4.
//
//go:noescape
func expSubSumNEON(s *float32, m float32, n int) float32
