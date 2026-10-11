//go:build amd64 && !purego

#include "textflag.h"

// The int8 kernels of AttendInt8 (int8kernel.go, docs/FINDINGS.md F-041)
// in AVX2.

// func dotsI8AVX2(dst *int32, q, k *int8, n, width, stride int)
//
// dst[j] = Σ_{i < width} q[i] · k[j·stride + i] for j < n; width is a
// positive multiple of 16, n ≥ 1. Per 16-byte chunk, VPMOVSXBW widens q
// and k to int16 and VPMADDWD multiplies and adds pairs into eight int32
// lanes (|q · k| ≤ 2¹⁴, so the sums are exact); the lanes are summed at
// the end of the row.
//
// Args (ABI0): dst +0, q +8, k +16, n +24, width +32, stride +40; 48 bytes.
//
// Registers: DI dst, SI q, DX k (key j), CX keys left, R8 width, R9 stride;
// R10 the q cursor, R11 the key cursor, R12 bytes left in a row; Y0/Y1
// widened chunks, Y2 products, Y3 the sums.
TEXT ·dotsI8AVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ q+8(FP), SI
	MOVQ k+16(FP), DX
	MOVQ n+24(FP), CX
	MOVQ width+32(FP), R8
	MOVQ stride+40(FP), R9

key:
	MOVQ   SI, R10
	MOVQ   DX, R11
	MOVQ   R8, R12
	VPXOR  Y3, Y3, Y3

chunk:
	VPMOVSXBW (R10), Y0
	VPMOVSXBW (R11), Y1
	VPMADDWD  Y1, Y0, Y2
	VPADDD    Y2, Y3, Y3
	ADDQ      $16, R10
	ADDQ      $16, R11
	SUBQ      $16, R12
	JNZ       chunk

	VEXTRACTI128 $1, Y3, X4
	VPADDD       X4, X3, X3
	VPSHUFD      $0x4E, X3, X4
	VPADDD       X4, X3, X3
	VPSHUFD      $0xB1, X3, X4
	VPADDD       X4, X3, X3
	VMOVD        X3, (DI)

	ADDQ $4, DI
	ADDQ R9, DX
	DECQ CX
	JNZ  key

	VZEROUPPER
	RET

// func axpysI8AVX2(out, w *float32, v *int8, n, width, stride int)
//
// out[d] += Σ_{j < n} w[j] · v[j·stride + d] for d < width; width is a
// positive multiple of 16, n ≥ 1. Per chunk of 16 outputs, two
// accumulators stay in Y0/Y1 while the keys run in order; each key's 16
// codes are sign-extended to int32 (VPMOVSXBD), converted (VCVTDQ2PS),
// multiplied by w[j] and added: VMULPS then VADDPS, no FMA, as
// axpysI8Generic's float32(w · c) + out.
//
// Args (ABI0): out +0, w +8, v +16, n +24, width +32, stride +40; 48 bytes.
//
// Registers: DI out chunk, SI w, DX the chunk's column in v, CX n,
// R8 outputs left, R9 stride; R10 the w cursor, R11 the v cursor, R12 keys
// left; Y0/Y1 sums, Y2 w[j], Y3/Y4 the codes.
TEXT ·axpysI8AVX2(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ w+8(FP), SI
	MOVQ v+16(FP), DX
	MOVQ n+24(FP), CX
	MOVQ width+32(FP), R8
	MOVQ stride+40(FP), R9

chunk:
	VMOVUPS (DI), Y0
	VMOVUPS 32(DI), Y1
	MOVQ    SI, R10
	MOVQ    DX, R11
	MOVQ    CX, R12

key:
	VBROADCASTSS (R10), Y2
	VPMOVSXBD    (R11), Y3
	VPMOVSXBD    8(R11), Y4
	VCVTDQ2PS    Y3, Y3
	VCVTDQ2PS    Y4, Y4
	VMULPS       Y2, Y3, Y3
	VMULPS       Y2, Y4, Y4
	VADDPS       Y3, Y0, Y0
	VADDPS       Y4, Y1, Y1
	ADDQ         $4, R10
	ADDQ         R9, R11
	DECQ         R12
	JNZ          key

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	ADDQ    $64, DI
	ADDQ    $16, DX
	SUBQ    $16, R8
	JNZ     chunk

	VZEROUPPER
	RET
