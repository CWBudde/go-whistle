//go:build (!amd64 && !arm64) || purego

package needle

func expSubSumF32(s []float32, m float32) float32 {
	return expSubSumGeneric(s, m)
}
