package cact

import "math"

// float16ToFloat32 widens an IEEE 754 binary16 value, including subnormals,
// infinities and NaN. Copied from internal/safetensors/store.go, which in turn
// comes from github.com/cwbudde/go-pocket-tts internal/safetensors @ 9faba38.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x03ff)

	var bits uint32

	switch exp {
	case 0:
		if frac == 0 {
			bits = sign << 31
		} else {
			// Subnormal: normalize.
			exp32 := uint32(113)

			for (frac & 0x0400) == 0 {
				frac <<= 1
				exp32--
			}

			frac &= 0x03ff
			bits = (sign << 31) | (exp32 << 23) | (frac << 13)
		}
	case 0x1f:
		// Inf / NaN.
		bits = (sign << 31) | 0x7f800000 | (frac << 13)
	default:
		exp32 := exp + (127 - 15)
		bits = (sign << 31) | (exp32 << 23) | (frac << 13)
	}

	return math.Float32frombits(bits)
}
