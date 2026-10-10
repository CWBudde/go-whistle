//go:build amd64 && !purego

#include "textflag.h"

// expConsts holds the kernel's constants, each in all 8 lanes of a YMM
// word: log2e, ln2hi, ln2lo, c6 … c1, lo, 1.0 (see exp_arm64.s).
DATA expConsts<>+0(SB)/8, $0x3FB8AA3B3FB8AA3B   // log2e
DATA expConsts<>+8(SB)/8, $0x3FB8AA3B3FB8AA3B
DATA expConsts<>+16(SB)/8, $0x3FB8AA3B3FB8AA3B
DATA expConsts<>+24(SB)/8, $0x3FB8AA3B3FB8AA3B
DATA expConsts<>+32(SB)/8, $0x3F3180003F318000  // ln2hi = 0.693359375
DATA expConsts<>+40(SB)/8, $0x3F3180003F318000
DATA expConsts<>+48(SB)/8, $0x3F3180003F318000
DATA expConsts<>+56(SB)/8, $0x3F3180003F318000
DATA expConsts<>+64(SB)/8, $0xB95E8083B95E8083  // ln2lo = −2.12194440e-4
DATA expConsts<>+72(SB)/8, $0xB95E8083B95E8083
DATA expConsts<>+80(SB)/8, $0xB95E8083B95E8083
DATA expConsts<>+88(SB)/8, $0xB95E8083B95E8083
DATA expConsts<>+96(SB)/8, $0x3950696739506967  // c6 = 1.9875691500e-4
DATA expConsts<>+104(SB)/8, $0x3950696739506967
DATA expConsts<>+112(SB)/8, $0x3950696739506967
DATA expConsts<>+120(SB)/8, $0x3950696739506967
DATA expConsts<>+128(SB)/8, $0x3AB743CE3AB743CE // c5 = 1.3981999507e-3
DATA expConsts<>+136(SB)/8, $0x3AB743CE3AB743CE
DATA expConsts<>+144(SB)/8, $0x3AB743CE3AB743CE
DATA expConsts<>+152(SB)/8, $0x3AB743CE3AB743CE
DATA expConsts<>+160(SB)/8, $0x3C0889083C088908 // c4 = 8.3334519073e-3
DATA expConsts<>+168(SB)/8, $0x3C0889083C088908
DATA expConsts<>+176(SB)/8, $0x3C0889083C088908
DATA expConsts<>+184(SB)/8, $0x3C0889083C088908
DATA expConsts<>+192(SB)/8, $0x3D2AA9C13D2AA9C1 // c3 = 4.1665795894e-2
DATA expConsts<>+200(SB)/8, $0x3D2AA9C13D2AA9C1
DATA expConsts<>+208(SB)/8, $0x3D2AA9C13D2AA9C1
DATA expConsts<>+216(SB)/8, $0x3D2AA9C13D2AA9C1
DATA expConsts<>+224(SB)/8, $0x3E2AAAAA3E2AAAAA // c2 = 1.6666665459e-1
DATA expConsts<>+232(SB)/8, $0x3E2AAAAA3E2AAAAA
DATA expConsts<>+240(SB)/8, $0x3E2AAAAA3E2AAAAA
DATA expConsts<>+248(SB)/8, $0x3E2AAAAA3E2AAAAA
DATA expConsts<>+256(SB)/8, $0x3F0000003F000000 // c1 = 0.5
DATA expConsts<>+264(SB)/8, $0x3F0000003F000000
DATA expConsts<>+272(SB)/8, $0x3F0000003F000000
DATA expConsts<>+280(SB)/8, $0x3F0000003F000000
DATA expConsts<>+288(SB)/8, $0xC2AEA8F6C2AEA8F6 // lo = −87.33
DATA expConsts<>+296(SB)/8, $0xC2AEA8F6C2AEA8F6
DATA expConsts<>+304(SB)/8, $0xC2AEA8F6C2AEA8F6
DATA expConsts<>+312(SB)/8, $0xC2AEA8F6C2AEA8F6
DATA expConsts<>+320(SB)/8, $0x3F8000003F800000 // 1.0
DATA expConsts<>+328(SB)/8, $0x3F8000003F800000
DATA expConsts<>+336(SB)/8, $0x3F8000003F800000
DATA expConsts<>+344(SB)/8, $0x3F8000003F800000
GLOBL expConsts<>(SB), RODATA|NOPTR, $352

// func expSubSumAVX2(s *float32, m float32, n int) float32
//
// s[i] = exp(s[i] − m) for i < n, returns Σ s[i]; n is a positive
// multiple of 8. The same Cephes expf as exp_arm64.s, 8 lanes at a time.
// VMAXPS returns its second (Intel) source if either is NaN, so with x
// there a NaN stays NaN; x < lo (VCMPPS predicate 1, LT_OS) is false for
// NaN, so it is not flushed.
//
// Args (ABI0): s +0(FP), m +8(FP), n +16(FP), ret +24(FP); 28 bytes.
//
// Registers:
//   DI       s, CX remaining count
//   Y0       eight partial sums, Y1 m in all lanes
//   Y2       x, then r;  Y3 the lt mask;  Y4 k;  Y5 r²;  Y6 p, then e
//   Y8–Y15   log2e, ln2hi, ln2lo, lo, 1.0, c1, c2, c3; c4–c6 from memory
TEXT ·expSubSumAVX2(SB), NOSPLIT, $0-28
	MOVQ         s+0(FP), DI
	VBROADCASTSS m+8(FP), Y1
	MOVQ         n+16(FP), CX

	VMOVUPS expConsts<>+0(SB), Y8    // log2e
	VMOVUPS expConsts<>+32(SB), Y9   // ln2hi
	VMOVUPS expConsts<>+64(SB), Y10  // ln2lo
	VMOVUPS expConsts<>+288(SB), Y11 // lo
	VMOVUPS expConsts<>+320(SB), Y12 // 1.0
	VMOVUPS expConsts<>+256(SB), Y13 // c1
	VMOVUPS expConsts<>+224(SB), Y14 // c2
	VMOVUPS expConsts<>+192(SB), Y15 // c3

	VXORPS Y0, Y0, Y0

loop:
	VMOVUPS (DI), Y2
	VSUBPS  Y1, Y2, Y2      // x = s − m
	VCMPPS  $1, Y11, Y2, Y3 // lt = x < lo
	VMAXPS  Y2, Y11, Y2     // x = max(lo, x), NaN stays

	VMULPS       Y8, Y2, Y4  // k = round(x · log2e)
	VROUNDPS     $0, Y4, Y4
	VFNMADD231PS Y9, Y4, Y2  // r = x − k·ln2hi
	VFNMADD231PS Y10, Y4, Y2 // r −= k·ln2lo
	VMULPS       Y2, Y2, Y5  // r²

	VMOVUPS     expConsts<>+96(SB), Y6  // p = c6
	VFMADD213PS expConsts<>+128(SB), Y2, Y6 // p = p·r + c5
	VFMADD213PS expConsts<>+160(SB), Y2, Y6 // p = p·r + c4
	VFMADD213PS Y15, Y2, Y6 // p = p·r + c3
	VFMADD213PS Y14, Y2, Y6 // p = p·r + c2
	VFMADD213PS Y13, Y2, Y6 // p = p·r + c1

	VFMADD213PS Y2, Y5, Y6 // y = p·r² + r
	VADDPS      Y12, Y6, Y6 // y += 1

	VCVTTPS2DQ Y4, Y4      // e = y · 2^k
	VPSLLD     $23, Y4, Y4
	VPADDD     Y4, Y6, Y6
	VANDNPS    Y6, Y3, Y6  // e = 0 where x < lo

	VMOVUPS Y6, (DI)
	VADDPS  Y6, Y0, Y0

	ADDQ $32, DI
	SUBQ $8, CX
	JNZ  loop

	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER
	VMOVSS       X0, ret+24(FP)
	RET
