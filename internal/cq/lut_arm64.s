//go:build arm64 && !purego

#include "textflag.h"

// NEON kernels of the lut path (lut.go, docs/FINDINGS.md F-034). Each one
// does the same float32 operations in the same order as its Go reference,
// so the results are equal bit for bit (asm_test.go).
//
// Go 1.25's arm64 assembler lacks the vector float instructions and SHRN,
// so they are WORD-encoded with the instruction in the comment, as in
// internal/tensor:
//   FADD   Vd.4S, Vn.4S, Vm.4S     0x4E20D400 | m<<16 | n<<5 | d
//   FSUB   Vd.4S, Vn.4S, Vm.4S     0x4EA0D400 | m<<16 | n<<5 | d
//   FMUL   Vd.4S, Vn.4S, Vm.4S     0x6E20DC00 | m<<16 | n<<5 | d
//   FMUL   Vd.4S, Vn.4S, Vm.S[i]   0x4F809000 | (i&1)<<21 | m<<16 | (i>>1)<<11 | n<<5 | d
//   FADDP  Vd.4S, Vn.4S, Vm.4S     0x6E20D400 | m<<16 | n<<5 | d
//   FMAX   Vd.4S, Vn.4S, Vm.4S     0x4E20F400 | m<<16 | n<<5 | d
//   FMAXV  Sd, Vn.4S               0x6E30F800 | n<<5 | d
//   FABS   Vd.4S, Vn.4S            0x4EA0F800 | n<<5 | d
//   FCVTNS Vd.4S, Vn.4S            0x4E21A800 | n<<5 | d
//   SCVTF  Vd.4S, Vn.4S            0x4E21D800 | n<<5 | d
//   SHRN   Vd.8B, Vn.8H, #7        0x0F098400 | n<<5 | d   (SHRN2 Vd.16B: 0x4F098400)

// func fwhtNEON(x *float32, n int)
//
// x[:n] = H_n · x[:n], n a power of two >= 8, with fwht's butterflies in
// fwht's order. Pass 1 runs h = 1, 2, 4 in registers per 8 floats: UZP1/
// UZP2 split even and odd elements, so h = 1 and h = 2 become vector
// butterflies, and a last UZP pair restores the order. Then one pass over
// memory per h = 8 … n/2, 8 floats per step.
TEXT ·fwhtNEON(SB), NOSPLIT, $0-16
	MOVD x+0(FP), R0
	MOVD n+8(FP), R1
	MOVD R0, R2
	MOVD R1, R3

fwhtBlock8:
	VLD1  (R2), [V0.S4, V1.S4]             // x0..x7
	VUZP1 V1.S4, V0.S4, V2.S4              // [x0 x2 x4 x6]
	VUZP2 V1.S4, V0.S4, V3.S4              // [x1 x3 x5 x7]
	WORD $0x4E23D444 // FADD V4.4S, V2.4S, V3.4S  (h = 1: [y0 y2 y4 y6])
	WORD $0x4EA3D445 // FSUB V5.4S, V2.4S, V3.4S  ([y1 y3 y5 y7])
	VUZP1 V5.S4, V4.S4, V6.S4              // [y0 y4 y1 y5]
	VUZP2 V5.S4, V4.S4, V7.S4              // [y2 y6 y3 y7]
	WORD $0x4E27D4C8 // FADD V8.4S, V6.4S, V7.4S  (h = 2: [z0 z4 z1 z5])
	WORD $0x4EA7D4C9 // FSUB V9.4S, V6.4S, V7.4S  ([z2 z6 z3 z7])
	VUZP1 V9.S4, V8.S4, V10.S4             // [z0 z1 z2 z3]
	VUZP2 V9.S4, V8.S4, V11.S4             // [z4 z5 z6 z7]
	WORD $0x4E2BD540 // FADD V0.4S, V10.4S, V11.4S  (h = 4)
	WORD $0x4EABD541 // FSUB V1.4S, V10.4S, V11.4S
	VST1.P [V0.S4, V1.S4], 32(R2)
	SUBS  $8, R3, R3
	BNE   fwhtBlock8

	MOVD $32, R4                        // h in bytes
	LSL  $2, R1, R1                     // n in bytes
	ADD  R0, R1, R7                     // end of x

fwhtPass:
	CMP  R1, R4
	BGE  fwhtDone
	MOVD R0, R6                         // start of the block of 2h

fwhtBlock:
	MOVD R6, R8                         // &x[j]
	ADD  R4, R6, R9                     // &x[j+h]
	MOVD R4, R10

fwhtStep:
	VLD1  (R8), [V0.S4, V1.S4]
	VLD1  (R9), [V2.S4, V3.S4]
	WORD $0x4E22D404 // FADD V4.4S, V0.4S, V2.4S
	WORD $0x4E23D425 // FADD V5.4S, V1.4S, V3.4S
	WORD $0x4EA2D406 // FSUB V6.4S, V0.4S, V2.4S
	WORD $0x4EA3D427 // FSUB V7.4S, V1.4S, V3.4S
	VST1.P [V4.S4, V5.S4], 32(R8)
	VST1.P [V6.S4, V7.S4], 32(R9)
	SUBS  $32, R10, R10
	BNE   fwhtStep

	ADD  R4<<1, R6, R6
	CMP  R7, R6
	BLT  fwhtBlock

	LSL  $1, R4, R4
	B    fwhtPass

fwhtDone:
	RET

// func lutTables2NEON(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32
//
// lutTables2Generic for one group x[:n], n a multiple of 8: returns M and
// writes n/2 tables of 32 bytes to tab. cb points to the 4 levels.
//
// Registers: R0 tab, R1 x, R2 count; V16 cb[0..3], V17–V20 cb[k] in every
// lane; F28 lutQ/M; V25 = 127, V26 = lutBias per byte.
TEXT ·lutTables2NEON(SB), NOSPLIT, $0-44
	MOVD  tab+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  n+16(FP), R2
	MOVD  cb+24(FP), R3
	FMOVS cbMax+32(FP), F30

	MOVD  R1, R4
	MOVD  R2, R5
	VEOR  V31.B16, V31.B16, V31.B16

lut2Max:
	VLD1.P 32(R4), [V0.S4, V1.S4]
	WORD $0x4EA0F800 // FABS V0.4S, V0.4S
	WORD $0x4EA0F821 // FABS V1.4S, V1.4S
	WORD $0x6E21D402 // FADDP V2.4S, V0.4S, V1.4S  (|x[2q]| + |x[2q+1]|)
	WORD $0x4E22F7FF // FMAX V31.4S, V31.4S, V2.4S
	SUBS  $8, R5, R5
	BNE   lut2Max

	WORD $0x6E30FBFF // FMAXV S31, V31.4S
	FMULS F31, F30, F30                    // M = cbMax · max
	MOVW   $0x45FFF800, R10                // 8191.0 = lutQ
	FMOVS  R10, F29
	FDIVS  F30, F29, F28                   // lutQ / M
	VEOR   V27.B16, V27.B16, V27.B16
	FCMPS  F27, F30                        // M against 0
	FCSELS GT, F28, F27, F28               // lutInv: 0 unless M > 0

	VLD1  (R3), [V16.S4]
	VDUP  V16.S[0], V17.S4
	VDUP  V16.S[1], V18.S4
	VDUP  V16.S[2], V19.S4
	VDUP  V16.S[3], V20.S4
	VMOVI $127, V25.B16
	VMOVI $64, V26.B16

lut2Build:
	VLD1.P 16(R1), [V0.S4]                 // [a0 b0 a1 b1], two pairs
	WORD $0x4F9C9000 // FMUL V0.4S, V0.4S, V28.S[0]  (times lutQ/M)
	WORD $0x4F809201 // FMUL V1.4S, V16.4S, V0.S[0]  (T = cb[0..3] · a0)
	WORD $0x4FA09222 // FMUL V2.4S, V17.4S, V0.S[1]  (U_k = cb[k] · b0)
	WORD $0x4FA09243 // FMUL V3.4S, V18.4S, V0.S[1]
	WORD $0x4FA09264 // FMUL V4.4S, V19.4S, V0.S[1]
	WORD $0x4FA09285 // FMUL V5.4S, V20.4S, V0.S[1]
	WORD $0x4E22D422 // FADD V2.4S, V1.4S, V2.4S  (entries 4k..4k+3 = T + U_k)
	WORD $0x4E23D423 // FADD V3.4S, V1.4S, V3.4S
	WORD $0x4E24D424 // FADD V4.4S, V1.4S, V4.4S
	WORD $0x4E25D425 // FADD V5.4S, V1.4S, V5.4S
	WORD $0x4E21A842 // FCVTNS V2.4S, V2.4S  (round to nearest even)
	WORD $0x4E21A863 // FCVTNS V3.4S, V3.4S
	WORD $0x4E21A884 // FCVTNS V4.4S, V4.4S
	WORD $0x4E21A8A5 // FCVTNS V5.4S, V5.4S
	VUZP1 V3.H8, V2.H8, V6.H8              // v as int16, entries 0–7
	VUZP1 V5.H8, V4.H8, V7.H8              // entries 8–15
	WORD $0x0F0984C9 // SHRN V9.8B, V6.8H, #7  (v >> 7)
	WORD $0x4F0984E9 // SHRN2 V9.16B, V7.8H, #7
	VADD  V26.B16, V9.B16, V9.B16          // hi = v>>7 + lutBias
	VUZP1 V7.B16, V6.B16, V8.B16           // low bytes of v
	VAND  V25.B16, V8.B16, V8.B16          // lo = v & 127
	VST1.P [V8.B16, V9.B16], 32(R0)
	WORD $0x4F809A01 // FMUL V1.4S, V16.4S, V0.S[2]  (T = cb[0..3] · a1)
	WORD $0x4FA09A22 // FMUL V2.4S, V17.4S, V0.S[3]  (U_k = cb[k] · b1)
	WORD $0x4FA09A43 // FMUL V3.4S, V18.4S, V0.S[3]
	WORD $0x4FA09A64 // FMUL V4.4S, V19.4S, V0.S[3]
	WORD $0x4FA09A85 // FMUL V5.4S, V20.4S, V0.S[3]
	WORD $0x4E22D422 // FADD V2.4S, V1.4S, V2.4S  (entries 4k..4k+3 = T + U_k)
	WORD $0x4E23D423 // FADD V3.4S, V1.4S, V3.4S
	WORD $0x4E24D424 // FADD V4.4S, V1.4S, V4.4S
	WORD $0x4E25D425 // FADD V5.4S, V1.4S, V5.4S
	WORD $0x4E21A842 // FCVTNS V2.4S, V2.4S  (round to nearest even)
	WORD $0x4E21A863 // FCVTNS V3.4S, V3.4S
	WORD $0x4E21A884 // FCVTNS V4.4S, V4.4S
	WORD $0x4E21A8A5 // FCVTNS V5.4S, V5.4S
	VUZP1 V3.H8, V2.H8, V6.H8              // v as int16, entries 0–7
	VUZP1 V5.H8, V4.H8, V7.H8              // entries 8–15
	WORD $0x0F0984C9 // SHRN V9.8B, V6.8H, #7  (v >> 7)
	WORD $0x4F0984E9 // SHRN2 V9.16B, V7.8H, #7
	VADD  V26.B16, V9.B16, V9.B16          // hi = v>>7 + lutBias
	VUZP1 V7.B16, V6.B16, V8.B16           // low bytes of v
	VAND  V25.B16, V8.B16, V8.B16          // lo = v & 127
	VST1.P [V8.B16, V9.B16], 32(R0)
	SUBS  $4, R2, R2
	BNE   lut2Build

	FMOVS F30, ret+40(FP)
	RET

// func lutTables4NEON(tab *byte, x *float32, n int, cb *float32, cbMax float32) float32
//
// lutTables4Generic for one group x[:n], n a multiple of 4: returns M and
// writes n tables of 32 bytes to tab. cb points to the 16 levels.
//
// Registers: R0 tab, R1 x, R2 count; V16–V19 cb[0..15]; F28 lutQ/M;
// V25 = 127, V26 = lutBias per byte.
TEXT ·lutTables4NEON(SB), NOSPLIT, $0-44
	MOVD  tab+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  n+16(FP), R2
	MOVD  cb+24(FP), R3
	FMOVS cbMax+32(FP), F30

	MOVD  R1, R4
	MOVD  R2, R5
	VEOR  V31.B16, V31.B16, V31.B16

lut4Max:
	VLD1.P 16(R4), [V0.S4]
	WORD $0x4EA0F800 // FABS V0.4S, V0.4S
	WORD $0x4E20F7FF // FMAX V31.4S, V31.4S, V0.4S
	SUBS  $4, R5, R5
	BNE   lut4Max

	WORD $0x6E30FBFF // FMAXV S31, V31.4S
	FMULS F31, F30, F30                    // M = cbMax · max
	MOVW   $0x45FFF800, R10                // 8191.0 = lutQ
	FMOVS  R10, F29
	FDIVS  F30, F29, F28                   // lutQ / M
	VEOR   V27.B16, V27.B16, V27.B16
	FCMPS  F27, F30                        // M against 0
	FCSELS GT, F28, F27, F28               // lutInv: 0 unless M > 0

	VLD1  (R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VMOVI $127, V25.B16
	VMOVI $64, V26.B16

lut4Build:
	VLD1.P 16(R1), [V0.S4]                 // four indices' x
	WORD $0x4F9C9000 // FMUL V0.4S, V0.4S, V28.S[0]  (times lutQ/M)
	WORD $0x4F809202 // FMUL V2.4S, V16.4S, V0.S[0]  (entries 4k..4k+3 = cb[4k..4k+3] · x0)
	WORD $0x4F809223 // FMUL V3.4S, V17.4S, V0.S[0]
	WORD $0x4F809244 // FMUL V4.4S, V18.4S, V0.S[0]
	WORD $0x4F809265 // FMUL V5.4S, V19.4S, V0.S[0]
	WORD $0x4E21A842 // FCVTNS V2.4S, V2.4S  (round to nearest even)
	WORD $0x4E21A863 // FCVTNS V3.4S, V3.4S
	WORD $0x4E21A884 // FCVTNS V4.4S, V4.4S
	WORD $0x4E21A8A5 // FCVTNS V5.4S, V5.4S
	VUZP1 V3.H8, V2.H8, V6.H8              // v as int16, entries 0–7
	VUZP1 V5.H8, V4.H8, V7.H8              // entries 8–15
	WORD $0x0F0984C9 // SHRN V9.8B, V6.8H, #7  (v >> 7)
	WORD $0x4F0984E9 // SHRN2 V9.16B, V7.8H, #7
	VADD  V26.B16, V9.B16, V9.B16          // hi = v>>7 + lutBias
	VUZP1 V7.B16, V6.B16, V8.B16           // low bytes of v
	VAND  V25.B16, V8.B16, V8.B16          // lo = v & 127
	VST1.P [V8.B16, V9.B16], 32(R0)
	WORD $0x4FA09202 // FMUL V2.4S, V16.4S, V0.S[1]  (entries 4k..4k+3 = cb[4k..4k+3] · x1)
	WORD $0x4FA09223 // FMUL V3.4S, V17.4S, V0.S[1]
	WORD $0x4FA09244 // FMUL V4.4S, V18.4S, V0.S[1]
	WORD $0x4FA09265 // FMUL V5.4S, V19.4S, V0.S[1]
	WORD $0x4E21A842 // FCVTNS V2.4S, V2.4S  (round to nearest even)
	WORD $0x4E21A863 // FCVTNS V3.4S, V3.4S
	WORD $0x4E21A884 // FCVTNS V4.4S, V4.4S
	WORD $0x4E21A8A5 // FCVTNS V5.4S, V5.4S
	VUZP1 V3.H8, V2.H8, V6.H8              // v as int16, entries 0–7
	VUZP1 V5.H8, V4.H8, V7.H8              // entries 8–15
	WORD $0x0F0984C9 // SHRN V9.8B, V6.8H, #7  (v >> 7)
	WORD $0x4F0984E9 // SHRN2 V9.16B, V7.8H, #7
	VADD  V26.B16, V9.B16, V9.B16          // hi = v>>7 + lutBias
	VUZP1 V7.B16, V6.B16, V8.B16           // low bytes of v
	VAND  V25.B16, V8.B16, V8.B16          // lo = v & 127
	VST1.P [V8.B16, V9.B16], 32(R0)
	WORD $0x4F809A02 // FMUL V2.4S, V16.4S, V0.S[2]  (entries 4k..4k+3 = cb[4k..4k+3] · x2)
	WORD $0x4F809A23 // FMUL V3.4S, V17.4S, V0.S[2]
	WORD $0x4F809A44 // FMUL V4.4S, V18.4S, V0.S[2]
	WORD $0x4F809A65 // FMUL V5.4S, V19.4S, V0.S[2]
	WORD $0x4E21A842 // FCVTNS V2.4S, V2.4S  (round to nearest even)
	WORD $0x4E21A863 // FCVTNS V3.4S, V3.4S
	WORD $0x4E21A884 // FCVTNS V4.4S, V4.4S
	WORD $0x4E21A8A5 // FCVTNS V5.4S, V5.4S
	VUZP1 V3.H8, V2.H8, V6.H8              // v as int16, entries 0–7
	VUZP1 V5.H8, V4.H8, V7.H8              // entries 8–15
	WORD $0x0F0984C9 // SHRN V9.8B, V6.8H, #7  (v >> 7)
	WORD $0x4F0984E9 // SHRN2 V9.16B, V7.8H, #7
	VADD  V26.B16, V9.B16, V9.B16          // hi = v>>7 + lutBias
	VUZP1 V7.B16, V6.B16, V8.B16           // low bytes of v
	VAND  V25.B16, V8.B16, V8.B16          // lo = v & 127
	VST1.P [V8.B16, V9.B16], 32(R0)
	WORD $0x4FA09A02 // FMUL V2.4S, V16.4S, V0.S[3]  (entries 4k..4k+3 = cb[4k..4k+3] · x3)
	WORD $0x4FA09A23 // FMUL V3.4S, V17.4S, V0.S[3]
	WORD $0x4FA09A44 // FMUL V4.4S, V18.4S, V0.S[3]
	WORD $0x4FA09A65 // FMUL V5.4S, V19.4S, V0.S[3]
	WORD $0x4E21A842 // FCVTNS V2.4S, V2.4S  (round to nearest even)
	WORD $0x4E21A863 // FCVTNS V3.4S, V3.4S
	WORD $0x4E21A884 // FCVTNS V4.4S, V4.4S
	WORD $0x4E21A8A5 // FCVTNS V5.4S, V5.4S
	VUZP1 V3.H8, V2.H8, V6.H8              // v as int16, entries 0–7
	VUZP1 V5.H8, V4.H8, V7.H8              // entries 8–15
	WORD $0x0F0984C9 // SHRN V9.8B, V6.8H, #7  (v >> 7)
	WORD $0x4F0984E9 // SHRN2 V9.16B, V7.8H, #7
	VADD  V26.B16, V9.B16, V9.B16          // hi = v>>7 + lutBias
	VUZP1 V7.B16, V6.B16, V8.B16           // low bytes of v
	VAND  V25.B16, V8.B16, V8.B16          // lo = v & 127
	VST1.P [V8.B16, V9.B16], 32(R0)
	SUBS  $4, R2, R2
	BNE   lut4Build

	FMOVS F30, ret+40(FP)
	RET

// func lutLookupNEON(y *float32, idx, tab *byte, norm, steps *float32, blocks, groups, gb int)
//
// lutLookupGeneric. Per block of 32 rows and index byte j, two loads hold
// the byte of rows 0–15 and 16–31; AND and USHR split the nibbles, and
// eight TBLs look them up in the lo and hi planes of tables 2j and 2j+1.
// Each plane's two lookups add in a byte (at most 254) and widen into
// uint16 sums per row. At the end of a group, 128·Σhi + Σlo − bias is
// converted to float32 and added to y times norm · step.
//
// Registers:
//   R0 y, R1 idx, R2 tab, R3 norm, R4 steps (R1, R3 advance throughout)
//   R5 blocks left, R6 groups, R7 gb, R8 table pointer, R9 step pointer,
//   R10 groups left, R11 bytes left in the group
//   V0–V7    y of rows 0–31 (float32)
//   V8–V11   Σlo of rows 0–7, 8–15, 16–23, 24–31 (uint16)
//   V12–V15  Σhi, the same rows
//   V16/V17  index bytes of rows 0–15/16–31, then their high nibbles
//   V20–V23  lo and hi plane of table 2j, lo and hi plane of table 2j+1
//   V24/V25  low nibbles; V26–V29 lookups; V30 bias (int32); V31 0x0F
TEXT ·lutLookupNEON(SB), NOSPLIT, $0-64
	MOVD y+0(FP), R0
	MOVD idx+8(FP), R1
	MOVD tab+16(FP), R2
	MOVD norm+24(FP), R3
	MOVD steps+32(FP), R4
	MOVD blocks+40(FP), R5
	MOVD groups+48(FP), R6
	MOVD gb+56(FP), R7

	LSL   $14, R7, R12                // bias = 128 · lutBias · 2·gb
	VDUP  R12, V30.S4
	VMOVI $15, V31.B16

lookupBlock:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	MOVD R2, R8
	MOVD R4, R9
	MOVD R6, R10

lookupGroup:
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	MOVD R7, R11

lookupByte:
	VLD1.P 32(R1), [V16.B16, V17.B16]
	VLD1.P 64(R8), [V20.B16, V21.B16, V22.B16, V23.B16]
	VAND  V31.B16, V16.B16, V24.B16     // low nibbles, rows 0–15
	VAND  V31.B16, V17.B16, V25.B16     // rows 16–31
	VUSHR $4, V16.B16, V16.B16          // high nibbles
	VUSHR $4, V17.B16, V17.B16
	VTBL  V24.B16, [V20.B16], V26.B16   // lo, rows 0–15
	VTBL  V16.B16, [V22.B16], V27.B16
	VADD  V27.B16, V26.B16, V26.B16
	VTBL  V25.B16, [V20.B16], V27.B16   // lo, rows 16–31
	VTBL  V17.B16, [V22.B16], V28.B16
	VADD  V28.B16, V27.B16, V27.B16
	VTBL  V24.B16, [V21.B16], V28.B16   // hi, rows 0–15
	VTBL  V16.B16, [V23.B16], V29.B16
	VADD  V29.B16, V28.B16, V28.B16
	VTBL  V25.B16, [V21.B16], V29.B16   // hi, rows 16–31
	VTBL  V17.B16, [V23.B16], V24.B16
	VADD  V24.B16, V29.B16, V29.B16
	VUADDW  V26.B8, V8.H8, V8.H8
	VUADDW2 V26.B16, V9.H8, V9.H8
	VUADDW  V27.B8, V10.H8, V10.H8
	VUADDW2 V27.B16, V11.H8, V11.H8
	VUADDW  V28.B8, V12.H8, V12.H8
	VUADDW2 V28.B16, V13.H8, V13.H8
	VUADDW  V29.B8, V14.H8, V14.H8
	VUADDW2 V29.B16, V15.H8, V15.H8
	SUBS  $1, R11, R11
	BNE   lookupByte

	FMOVS (R9), F18                     // step of the group
	ADD   $4, R9
	// rows 0–7
	VUSHLL  $7, V12.H4, V24.S4
	VUSHLL2 $7, V12.H8, V25.S4
	VUADDW  V8.H4, V24.S4, V24.S4
	VUADDW2 V8.H8, V25.S4, V25.S4
	VSUB    V30.S4, V24.S4, V24.S4
	VSUB    V30.S4, V25.S4, V25.S4
	WORD $0x4E21DB18 // SCVTF V24.4S, V24.4S
	WORD $0x4E21DB39 // SCVTF V25.4S, V25.4S
	VLD1.P  32(R3), [V26.S4, V27.S4]
	WORD $0x4F92935A // FMUL V26.4S, V26.4S, V18.S[0]  (norm · step)
	WORD $0x4F92937B // FMUL V27.4S, V27.4S, V18.S[0]
	WORD $0x6E3ADF18 // FMUL V24.4S, V24.4S, V26.4S
	WORD $0x6E3BDF39 // FMUL V25.4S, V25.4S, V27.4S
	WORD $0x4E38D400 // FADD V0.4S, V0.4S, V24.4S
	WORD $0x4E39D421 // FADD V1.4S, V1.4S, V25.4S
	// rows 8–15
	VUSHLL  $7, V13.H4, V24.S4
	VUSHLL2 $7, V13.H8, V25.S4
	VUADDW  V9.H4, V24.S4, V24.S4
	VUADDW2 V9.H8, V25.S4, V25.S4
	VSUB    V30.S4, V24.S4, V24.S4
	VSUB    V30.S4, V25.S4, V25.S4
	WORD $0x4E21DB18 // SCVTF V24.4S, V24.4S
	WORD $0x4E21DB39 // SCVTF V25.4S, V25.4S
	VLD1.P  32(R3), [V26.S4, V27.S4]
	WORD $0x4F92935A // FMUL V26.4S, V26.4S, V18.S[0]  (norm · step)
	WORD $0x4F92937B // FMUL V27.4S, V27.4S, V18.S[0]
	WORD $0x6E3ADF18 // FMUL V24.4S, V24.4S, V26.4S
	WORD $0x6E3BDF39 // FMUL V25.4S, V25.4S, V27.4S
	WORD $0x4E38D442 // FADD V2.4S, V2.4S, V24.4S
	WORD $0x4E39D463 // FADD V3.4S, V3.4S, V25.4S
	// rows 16–23
	VUSHLL  $7, V14.H4, V24.S4
	VUSHLL2 $7, V14.H8, V25.S4
	VUADDW  V10.H4, V24.S4, V24.S4
	VUADDW2 V10.H8, V25.S4, V25.S4
	VSUB    V30.S4, V24.S4, V24.S4
	VSUB    V30.S4, V25.S4, V25.S4
	WORD $0x4E21DB18 // SCVTF V24.4S, V24.4S
	WORD $0x4E21DB39 // SCVTF V25.4S, V25.4S
	VLD1.P  32(R3), [V26.S4, V27.S4]
	WORD $0x4F92935A // FMUL V26.4S, V26.4S, V18.S[0]  (norm · step)
	WORD $0x4F92937B // FMUL V27.4S, V27.4S, V18.S[0]
	WORD $0x6E3ADF18 // FMUL V24.4S, V24.4S, V26.4S
	WORD $0x6E3BDF39 // FMUL V25.4S, V25.4S, V27.4S
	WORD $0x4E38D484 // FADD V4.4S, V4.4S, V24.4S
	WORD $0x4E39D4A5 // FADD V5.4S, V5.4S, V25.4S
	// rows 24–31
	VUSHLL  $7, V15.H4, V24.S4
	VUSHLL2 $7, V15.H8, V25.S4
	VUADDW  V11.H4, V24.S4, V24.S4
	VUADDW2 V11.H8, V25.S4, V25.S4
	VSUB    V30.S4, V24.S4, V24.S4
	VSUB    V30.S4, V25.S4, V25.S4
	WORD $0x4E21DB18 // SCVTF V24.4S, V24.4S
	WORD $0x4E21DB39 // SCVTF V25.4S, V25.4S
	VLD1.P  32(R3), [V26.S4, V27.S4]
	WORD $0x4F92935A // FMUL V26.4S, V26.4S, V18.S[0]  (norm · step)
	WORD $0x4F92937B // FMUL V27.4S, V27.4S, V18.S[0]
	WORD $0x6E3ADF18 // FMUL V24.4S, V24.4S, V26.4S
	WORD $0x6E3BDF39 // FMUL V25.4S, V25.4S, V27.4S
	WORD $0x4E38D4C6 // FADD V6.4S, V6.4S, V24.4S
	WORD $0x4E39D4E7 // FADD V7.4S, V7.4S, V25.4S
	SUBS  $1, R10, R10
	BNE   lookupGroup

	VST1.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R0)
	SUBS  $1, R5, R5
	BNE   lookupBlock
	RET
