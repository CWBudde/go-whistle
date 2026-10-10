//go:build amd64 && !purego

#include "textflag.h"

// AVX2 kernels of the lut path (lut.go, docs/FINDINGS.md F-034). Each one
// does the same float32 operations in the same order as its Go reference,
// so the results are equal bit for bit (asm_test.go): no FMA, and
// VCVTPS2DQ rounds to nearest even under Go's default MXCSR, as
// math.RoundToEven does. Operands are in Go's order: sources, then the
// destination.

// func fwhtAVX2(x *float32, n int)
//
// x[:n] = H_n · x[:n], n a power of two >= 8, with fwht's butterflies in
// fwht's order. Pass 1 runs h = 1, 2, 4 in registers per 8 floats: the
// partner of every element is one permutation away (VPERMILPS,
// VPERM2F128); the sum is kept in the lower element of each pair, the
// difference lower − upper in the upper one (VBLENDPS). Then one pass over
// memory per h = 8 … n/2, 8 floats per step.
TEXT ·fwhtAVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX

	MOVQ SI, DI
	MOVQ CX, DX

fwhtBlock8:
	VMOVUPS    (DI), Y0
	VPERMILPS  $0xB1, Y0, Y1       // partners at h = 1
	VADDPS     Y1, Y0, Y2
	VSUBPS     Y0, Y1, Y3          // x[j] − x[j+1] in the upper elements
	VBLENDPS   $0xAA, Y3, Y2, Y0
	VPERMILPS  $0x4E, Y0, Y1       // h = 2
	VADDPS     Y1, Y0, Y2
	VSUBPS     Y0, Y1, Y3
	VBLENDPS   $0xCC, Y3, Y2, Y0
	VPERM2F128 $0x01, Y0, Y0, Y1   // h = 4
	VADDPS     Y1, Y0, Y2
	VSUBPS     Y0, Y1, Y3
	VBLENDPS   $0xF0, Y3, Y2, Y0
	VMOVUPS    Y0, (DI)
	ADDQ       $32, DI
	SUBQ       $8, DX
	JNZ        fwhtBlock8

	MOVQ $32, R8                   // h in bytes
	SHLQ $2, CX                    // n in bytes
	LEAQ (SI)(CX*1), R9            // end of x

fwhtPass:
	CMPQ R8, CX
	JGE  fwhtDone
	MOVQ SI, R10                   // start of the block of 2h

fwhtBlock:
	MOVQ R10, DI                   // &x[j]
	LEAQ (R10)(R8*1), DX           // &x[j+h]
	MOVQ R8, R11

fwhtStep:
	VMOVUPS (DI), Y0
	VMOVUPS (DX), Y1
	VADDPS  Y1, Y0, Y2
	VSUBPS  Y1, Y0, Y3
	VMOVUPS Y2, (DI)
	VMOVUPS Y3, (DX)
	ADDQ    $32, DI
	ADDQ    $32, DX
	SUBQ    $32, R11
	JNZ     fwhtStep

	LEAQ (R10)(R8*2), R10
	CMPQ R10, R9
	JLT  fwhtBlock

	SHLQ $1, R8
	JMP  fwhtPass

fwhtDone:
	VZEROUPPER
	RET

// lutConsts sets Y13 = lutBias and Y14 = 127 per int16, Y15 = 0x7FFFFFFF
// per float (|x|). Clobbers AX.
#define lutConsts \
	MOVL         $0x00400040, AX \
	VMOVD        AX, X13 \
	VPBROADCASTD X13, Y13 \
	MOVL         $0x007F007F, AX \
	VMOVD        AX, X14 \
	VPBROADCASTD X14, Y14 \
	MOVL         $0x7FFFFFFF, AX \
	VMOVD        AX, X15 \
	VPBROADCASTD X15, Y15

// lutBound sets X5 = M, cbMax (X3) times the largest element of Y4.
#define lutBound \
	VEXTRACTF128 $1, Y4, X5 \
	VMAXPS       X5, X4, X4 \
	VPERMILPS    $0x4E, X4, X5 \
	VMAXPS       X5, X4, X4 \
	VPERMILPS    $0xB1, X4, X5 \
	VMAXPS       X5, X4, X4 \
	VMULSS       X4, X3, X5

// lutInv sets Y6 = lutQ/M in every element, or 0 unless M (X5) > 0
// (lutInv). Clobbers AX, X7.
#define lutInv \
	MOVL         $0x45FFF800, AX \
	VMOVD        AX, X6 \
	VDIVSS       X5, X6, X6 \
	VXORPS       X7, X7, X7 \
	VCMPSS       $0x0E, X7, X5, X7 \
	VANDPS       X7, X6, X6 \
	VBROADCASTSS X6, Y6

// lutQuantStore rounds the 16 entries in Y0 (codes 0–7) and Y1 (8–15) to
// integers v and stores the 32-byte table at DI, lo plane then hi plane
// (lutPut); the packs work per 128-bit lane, VPERMQ restores the order.
// Clobbers Y0, Y1.
#define lutQuantStore \
	VCVTPS2DQ Y0, Y0 \
	VCVTPS2DQ Y1, Y1 \
	VPACKSSDW Y1, Y0, Y0 \
	VPERMQ    $0xD8, Y0, Y0 \
	VPSRAW    $7, Y0, Y1 \
	VPADDW    Y13, Y1, Y1 \
	VPAND     Y14, Y0, Y0 \
	VPACKUSWB Y1, Y0, Y0 \
	VPERMQ    $0xD8, Y0, Y0 \
	VMOVDQU   Y0, (DI) \
	ADDQ      $32, DI

// func lutTables2AVX2(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32
//
// lutTables2Generic for one group x[:n], n a multiple of 8: returns M and
// writes n/2 tables of 32 bytes to tab. cb points to the 4 levels.
//
// Registers: DI tab, SI x, CX count; Y8 cb[0..3] twice, Y9 cb[0] ×4 and
// cb[1] ×4, Y10 cb[2] ×4 and cb[3] ×4; Y6 lutQ/M.
TEXT ·lutTables2AVX2(SB), NOSPLIT, $0-44
	MOVQ   tab+0(FP), DI
	MOVQ   x+8(FP), SI
	MOVQ   n+16(FP), CX
	MOVQ   cb+24(FP), BX
	VMOVSS cbMax+32(FP), X3
	lutConsts

	MOVQ   SI, R8
	MOVQ   CX, DX
	VXORPS Y4, Y4, Y4

lut2Max:
	VANDPS  (R8), Y15, Y0
	VHADDPS Y0, Y0, Y0             // |x[2q]| + |x[2q+1]|
	VMAXPS  Y0, Y4, Y4
	ADDQ    $32, R8
	SUBQ    $8, DX
	JNZ     lut2Max

	lutBound
	lutInv

	VBROADCASTF128 (BX), Y8
	VBROADCASTSS   (BX), Y9
	VBROADCASTSS   4(BX), Y0
	VBLENDPS       $0xF0, Y0, Y9, Y9
	VBROADCASTSS   8(BX), Y10
	VBROADCASTSS   12(BX), Y0
	VBLENDPS       $0xF0, Y0, Y10, Y10

lut2Build:
	VBROADCASTSS (SI), Y2
	VMULPS       Y6, Y2, Y2        // a = x[2q] · lutQ/M
	VBROADCASTSS 4(SI), Y3
	VMULPS       Y6, Y3, Y3        // b = x[2q+1] · lutQ/M
	VMULPS       Y8, Y2, Y2        // cb[c&3] · a
	VMULPS       Y9, Y3, Y0        // cb[c>>2] · b, codes 0–7
	VMULPS       Y10, Y3, Y1       // codes 8–15
	VADDPS       Y0, Y2, Y0
	VADDPS       Y1, Y2, Y1
	lutQuantStore
	ADDQ         $8, SI
	SUBQ         $2, CX
	JNZ          lut2Build

	VMOVSS X5, ret+40(FP)
	VZEROUPPER
	RET

// func lutTables4AVX2(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32
//
// lutTables4Generic for one group x[:n], n a multiple of 8: returns M and
// writes n tables of 32 bytes to tab. cb points to the 16 levels.
//
// Registers: DI tab, SI x, CX count; Y8, Y9 cb[0..7], cb[8..15]; Y6 lutQ/M.
TEXT ·lutTables4AVX2(SB), NOSPLIT, $0-44
	MOVQ   tab+0(FP), DI
	MOVQ   x+8(FP), SI
	MOVQ   n+16(FP), CX
	MOVQ   cb+24(FP), BX
	VMOVSS cbMax+32(FP), X3
	lutConsts

	MOVQ   SI, R8
	MOVQ   CX, DX
	VXORPS Y4, Y4, Y4

lut4Max:
	VANDPS (R8), Y15, Y0
	VMAXPS Y0, Y4, Y4
	ADDQ   $32, R8
	SUBQ   $8, DX
	JNZ    lut4Max

	lutBound
	lutInv

	VMOVUPS (BX), Y8
	VMOVUPS 32(BX), Y9

lut4Build:
	VBROADCASTSS (SI), Y2
	VMULPS       Y6, Y2, Y2        // a = x[q] · lutQ/M
	VMULPS       Y8, Y2, Y0        // cb[c] · a, codes 0–7
	VMULPS       Y9, Y2, Y1        // codes 8–15
	lutQuantStore
	ADDQ         $4, SI
	DECQ         CX
	JNZ          lut4Build

	VMOVSS X5, ret+40(FP)
	VZEROUPPER
	RET

// func lutLookupAVX2(y *float32, idx, tab *byte, norm, steps *float32, blocks, groups, gb int)
//
// lutLookupGeneric. Per block of 32 rows and index byte j, one load holds
// the byte of all 32 rows; AND and a shift split the nibbles, and four
// VPSHUFBs look them up in the lo and hi planes of tables 2j and 2j+1,
// each broadcast to both 128-bit lanes. Each plane's two lookups add in a
// byte (at most 254) and widen into uint16 sums per row; the in-lane
// unpacks keep rows 0–7 and 16–23 in one register, 8–15 and 24–31 in the
// other. At the end of a group, 128·Σhi + Σlo − bias is converted to
// float32 (VPERM2I128 restores the row order) and added to y, in memory,
// times norm · step.
//
// Registers:
//   AX y, BX idx, CX tab, DX norm, SI steps (BX, DX advance throughout)
//   DI blocks left, R8 groups, R9 gb, R10 table pointer, R11 step pointer,
//   R12 groups left, R13 bytes left in the group
//   Y10/Y11  Σlo of rows 0–7 and 16–23 / 8–15 and 24–31 (uint16)
//   Y12/Y13  Σhi, the same rows
//   Y8 bias (int32), Y9 step, Y14 zero, Y15 0x0F; Y0–Y7 temporaries
TEXT ·lutLookupAVX2(SB), NOSPLIT, $0-64
	MOVQ y+0(FP), AX
	MOVQ idx+8(FP), BX
	MOVQ tab+16(FP), CX
	MOVQ norm+24(FP), DX
	MOVQ steps+32(FP), SI
	MOVQ blocks+40(FP), DI
	MOVQ groups+48(FP), R8
	MOVQ gb+56(FP), R9

	MOVQ         R9, R12
	SHLQ         $14, R12          // bias = 128 · lutBias · 2·gb
	VMOVD        R12, X8
	VPBROADCASTD X8, Y8
	MOVL         $0x0F0F0F0F, R12
	VMOVD        R12, X15
	VPBROADCASTD X15, Y15
	VPXOR        Y14, Y14, Y14

lookupBlock:
	VMOVUPS Y14, (AX)
	VMOVUPS Y14, 32(AX)
	VMOVUPS Y14, 64(AX)
	VMOVUPS Y14, 96(AX)
	MOVQ    CX, R10
	MOVQ    SI, R11
	MOVQ    R8, R12

lookupGroup:
	VPXOR Y10, Y10, Y10
	VPXOR Y11, Y11, Y11
	VPXOR Y12, Y12, Y12
	VPXOR Y13, Y13, Y13
	MOVQ  R9, R13

lookupByte:
	VMOVDQU        (BX), Y0
	VPAND          Y15, Y0, Y1     // low nibbles
	VPSRLW         $4, Y0, Y0
	VPAND          Y15, Y0, Y0     // high nibbles
	VBROADCASTI128 (R10), Y2       // lo plane, table 2j
	VBROADCASTI128 16(R10), Y3     // hi plane, table 2j
	VBROADCASTI128 32(R10), Y4     // lo plane, table 2j+1
	VBROADCASTI128 48(R10), Y5     // hi plane, table 2j+1
	VPSHUFB        Y1, Y2, Y2
	VPSHUFB        Y0, Y4, Y4
	VPADDB         Y4, Y2, Y2      // lo
	VPSHUFB        Y1, Y3, Y3
	VPSHUFB        Y0, Y5, Y5
	VPADDB         Y5, Y3, Y3      // hi
	VPUNPCKLBW     Y14, Y2, Y4
	VPADDW         Y4, Y10, Y10
	VPUNPCKHBW     Y14, Y2, Y2
	VPADDW         Y2, Y11, Y11
	VPUNPCKLBW     Y14, Y3, Y4
	VPADDW         Y4, Y12, Y12
	VPUNPCKHBW     Y14, Y3, Y3
	VPADDW         Y3, Y13, Y13
	ADDQ           $32, BX
	ADDQ           $64, R10
	DECQ           R13
	JNZ            lookupByte

	VBROADCASTSS (R11), Y9         // step of the group
	ADDQ         $4, R11

	// Rows 0–7 and 16–23.
	VPUNPCKLWD Y14, Y10, Y4        // Σlo, rows 0–3 | 16–19
	VPUNPCKHWD Y14, Y10, Y5        // rows 4–7 | 20–23
	VPUNPCKLWD Y14, Y12, Y6        // Σhi
	VPUNPCKHWD Y14, Y12, Y7
	VPSLLD     $7, Y6, Y6
	VPSLLD     $7, Y7, Y7
	VPADDD     Y6, Y4, Y4
	VPADDD     Y7, Y5, Y5
	VPSUBD     Y8, Y4, Y4
	VPSUBD     Y8, Y5, Y5
	VPERM2I128 $0x20, Y5, Y4, Y6   // rows 0–7
	VPERM2I128 $0x31, Y5, Y4, Y7   // rows 16–23
	VCVTDQ2PS  Y6, Y6
	VCVTDQ2PS  Y7, Y7
	VMULPS     (DX), Y9, Y4        // norm · step
	VMULPS     64(DX), Y9, Y5
	VMULPS     Y4, Y6, Y6
	VMULPS     Y5, Y7, Y7
	VADDPS     (AX), Y6, Y6
	VADDPS     64(AX), Y7, Y7
	VMOVUPS    Y6, (AX)
	VMOVUPS    Y7, 64(AX)

	// Rows 8–15 and 24–31.
	VPUNPCKLWD Y14, Y11, Y4
	VPUNPCKHWD Y14, Y11, Y5
	VPUNPCKLWD Y14, Y13, Y6
	VPUNPCKHWD Y14, Y13, Y7
	VPSLLD     $7, Y6, Y6
	VPSLLD     $7, Y7, Y7
	VPADDD     Y6, Y4, Y4
	VPADDD     Y7, Y5, Y5
	VPSUBD     Y8, Y4, Y4
	VPSUBD     Y8, Y5, Y5
	VPERM2I128 $0x20, Y5, Y4, Y6   // rows 8–15
	VPERM2I128 $0x31, Y5, Y4, Y7   // rows 24–31
	VCVTDQ2PS  Y6, Y6
	VCVTDQ2PS  Y7, Y7
	VMULPS     32(DX), Y9, Y4
	VMULPS     96(DX), Y9, Y5
	VMULPS     Y4, Y6, Y6
	VMULPS     Y5, Y7, Y7
	VADDPS     32(AX), Y6, Y6
	VADDPS     96(AX), Y7, Y7
	VMOVUPS    Y6, 32(AX)
	VMOVUPS    Y7, 96(AX)

	ADDQ $128, DX
	DECQ R12
	JNZ  lookupGroup

	ADDQ $128, AX
	DECQ DI
	JNZ  lookupBlock

	VZEROUPPER
	RET
