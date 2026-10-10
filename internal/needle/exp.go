package needle

import "math"

// expSubSum sets s[i] = exp(s[i] − m) and returns Σ s[i], the numerator
// and denominator of a softmax. m must be at least every s[i], so every
// exponent is at most 0. On arm64 (NEON) and on amd64 with AVX2 and FMA,
// an assembly kernel computes vectors of exponents with Cephes' expf: the
// range reduction x = n·ln 2 + r in two parts, a degree-6 polynomial in r
// and 2ⁿ added to the exponent bits. It is within 2 float32 ulps of
// math.Exp, and it returns 0 below e^−87.33, just above float32's
// smallest normal, where math.Exp still gives up to 1.2e-38. The kernel
// sums in vector lanes; the tail and the fallback sum one value after the
// other (docs/FINDINGS.md F-035).
func expSubSum(s []float32, m float32) float32 {
	return expSubSumF32(s, m)
}

// expSubSumGeneric is expSubSum in Go: float64 math.Exp rounded to
// float32, summed in order.
func expSubSumGeneric(s []float32, m float32) float32 {
	var sum float32

	for i, v := range s {
		e := float32(math.Exp(float64(v - m)))
		s[i] = e
		sum += e
	}

	return sum
}
