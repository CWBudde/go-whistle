//go:build arm64 && !purego

#include "textflag.h"

// func expSubSumNEON(s *float32, m float32, n int) float32
//
// s[i] = exp(s[i] − m) for i < n, returns Σ s[i]; n is a positive
// multiple of 4. Cephes expf per lane (see exp.go):
//
//	lt = x < lo,  x = max(x, lo)      lo = −87.33, flushed to 0 at the end
//	k  = round(x · log2e)
//	r  = x − k·ln2hi − k·ln2lo        Cody–Waite, ln2hi exact in 9 bits
//	p  = ((((c6·r + c5)·r + c4)·r + c3)·r + c2)·r + c1
//	y  = p·r² + r + 1                 e^r for |r| ≤ ln2/2
//	e  = y · 2^k                      as bits(y) + k<<23; k ≥ −126 and
//	                                  r ≥ 0 there, so the result is normal
//
// A NaN passes FMAX and the polynomial and stays NaN; lo > NaN is false,
// so it is not flushed.
//
// Args (ABI0): s +0(FP), m +8(FP), n +16(FP), ret +24(FP); 28 bytes.
//
// Registers:
//   R0       s (post-incremented by the store), R2 remaining count
//   V0       four partial sums, V1 m in all lanes
//   V2       x, then r, then e;  V3 the lt mask;  V4 k;  V5 r²
//   V6, V7   the Horner steps
//   V16–V26  log2e, ln2hi, ln2lo, c6 … c1, lo, 1.0 in all lanes
TEXT ·expSubSumNEON(SB), NOSPLIT, $0-28
	MOVD  s+0(FP), R0
	FMOVS m+8(FP), F1
	MOVD  n+16(FP), R2

	VDUP V1.S[0], V1.S4

	MOVW $0x3FB8AA3B, R3 // log2e
	VDUP R3, V16.S4
	MOVW $0x3F318000, R3 // ln2hi = 0.693359375
	VDUP R3, V17.S4
	MOVW $0xB95E8083, R3 // ln2lo = −2.12194440e-4
	VDUP R3, V18.S4
	MOVW $0x39506967, R3 // c6 = 1.9875691500e-4
	VDUP R3, V19.S4
	MOVW $0x3AB743CE, R3 // c5 = 1.3981999507e-3
	VDUP R3, V20.S4
	MOVW $0x3C088908, R3 // c4 = 8.3334519073e-3
	VDUP R3, V21.S4
	MOVW $0x3D2AA9C1, R3 // c3 = 4.1665795894e-2
	VDUP R3, V22.S4
	MOVW $0x3E2AAAAA, R3 // c2 = 1.6666665459e-1
	VDUP R3, V23.S4
	MOVW $0x3F000000, R3 // c1 = 0.5
	VDUP R3, V24.S4
	MOVW $0xC2AEA8F6, R3 // lo = −87.33
	VDUP R3, V25.S4
	MOVW $0x3F800000, R3 // 1.0
	VDUP R3, V26.S4

	VEOR V0.B16, V0.B16, V0.B16

loop:
	VLD1  (R0), [V2.S4]
	VFSUB V1.S4, V2.S4, V2.S4   // x = s − m
	VFCMGT V2.S4, V25.S4, V3.S4 // lt = lo > x
	VFMAX V25.S4, V2.S4, V2.S4  // x = max(x, lo)

	VFMUL   V16.S4, V2.S4, V4.S4 // k = round(x · log2e)
	VFRINTN V4.S4, V4.S4
	VFMLS   V17.S4, V4.S4, V2.S4 // r = x − k·ln2hi
	VFMLS   V18.S4, V4.S4, V2.S4 // r −= k·ln2lo
	VFMUL   V2.S4, V2.S4, V5.S4  // r²

	VORR V20.B16, V20.B16, V6.B16 // p = c5 + c6·r
	VFMLA V19.S4, V2.S4, V6.S4
	VORR V21.B16, V21.B16, V7.B16 // p = c4 + p·r
	VFMLA V6.S4, V2.S4, V7.S4
	VORR V22.B16, V22.B16, V6.B16 // p = c3 + p·r
	VFMLA V7.S4, V2.S4, V6.S4
	VORR V23.B16, V23.B16, V7.B16 // p = c2 + p·r
	VFMLA V6.S4, V2.S4, V7.S4
	VORR V24.B16, V24.B16, V6.B16 // p = c1 + p·r
	VFMLA V7.S4, V2.S4, V6.S4

	VFMLA V6.S4, V5.S4, V2.S4   // y = r + p·r²
	VFADD V26.S4, V2.S4, V2.S4  // y += 1

	VFCVTZS V4.S4, V4.S4        // e = y · 2^k
	VSHL    $23, V4.S4, V4.S4
	VADD    V4.S4, V2.S4, V2.S4
	VBIC    V3.B16, V2.B16, V2.B16 // e = 0 where x < lo

	VST1.P [V2.S4], 16(R0)
	VFADD  V2.S4, V0.S4, V0.S4

	SUBS $4, R2, R2
	BNE  loop

	VFADDP V0.S4, V0.S4, V0.S4
	VFADDP V0.S4, V0.S4, V0.S4
	FMOVS  F0, ret+24(FP)
	RET
