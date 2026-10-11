//go:build arm64 && !purego

#include "textflag.h"

// The int8 kernels of AttendInt8 (int8kernel.go, docs/FINDINGS.md F-041).
// Go 1.25's assembler lacks the integer widening and pairwise NEON
// instructions and the vector float ones, so they are WORD-encoded with
// the instruction in the comment:
//
//   SMULL   Vd.8H, Vn.8B, Vm.8B     0x0E20C000 | m<<16 | n<<5 | d
//   SMULL2  Vd.8H, Vn.16B, Vm.16B   0x4E20C000 | m<<16 | n<<5 | d
//   SADALP  Vd.4S, Vn.8H            0x4E606800 | n<<5 | d
//   ADDP    Vd.4S, Vn.4S, Vm.4S     0x4EA0BC00 | m<<16 | n<<5 | d
//   ADDV    Sd, Vn.4S               0x4EB1B800 | n<<5 | d
//   SSHLL   Vd.8H, Vn.8B, #0        0x0F08A400 | n<<5 | d   (SSHLL2 Vn.16B: 0x4F08A400)
//   SSHLL   Vd.4S, Vn.4H, #0        0x0F10A400 | n<<5 | d   (SSHLL2 Vn.8H: 0x4F10A400)
//   SCVTF   Vd.4S, Vn.4S            0x4E21D800 | n<<5 | d
//   FMUL    Vd.4S, Vn.4S, Vm.S[0]   0x4F809000 | m<<16 | n<<5 | d
//   FADD    Vd.4S, Vn.4S, Vm.4S     0x4E20D400 | m<<16 | n<<5 | d

// func dotsI8NEON(dst *int32, q, k *int8, n, width, stride int)
//
// dst[j] = Σ_{i < width} q[i] · k[j·stride + i] for j < n; width is a
// positive multiple of 16, n ≥ 1. Each 16-byte chunk is two SMULLs to
// int16 (|q · k| ≤ 2¹⁴) and two SADALPs into four int32 lanes, so the sums
// are exact. Four keys run at once and ADDP reduces their lanes to one
// vector of four results.
//
// Args (ABI0): dst +0, q +8, k +16, n +24, width +32, stride +40; 48 bytes.
//
// Registers: R0 dst, R1 q, R2 k (key j), R3 keys left, R4 width, R5 stride;
// R6–R9 the four key cursors, R10 the q cursor, R11 bytes left in a row;
// V0 the q chunk, V1–V4 the key chunks, V5/V6 products, V20–V23 sums.
TEXT ·dotsI8NEON(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD q+8(FP), R1
	MOVD k+16(FP), R2
	MOVD n+24(FP), R3
	MOVD width+32(FP), R4
	MOVD stride+40(FP), R5

four:
	CMP $4, R3
	BLT one

	MOVD R2, R6
	ADD  R5, R6, R7
	ADD  R5, R7, R8
	ADD  R5, R8, R9
	MOVD R1, R10
	MOVD R4, R11
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

fourChunk:
	VLD1.P 16(R10), [V0.B16]
	VLD1.P 16(R6), [V1.B16]
	VLD1.P 16(R7), [V2.B16]
	VLD1.P 16(R8), [V3.B16]
	VLD1.P 16(R9), [V4.B16]
	WORD $0x0E21C005 // SMULL  V5.8H, V0.8B, V1.8B
	WORD $0x4E21C006 // SMULL2 V6.8H, V0.16B, V1.16B
	WORD $0x4E6068B4 // SADALP V20.4S, V5.8H
	WORD $0x4E6068D4 // SADALP V20.4S, V6.8H
	WORD $0x0E22C005 // SMULL  V5.8H, V0.8B, V2.8B
	WORD $0x4E22C006 // SMULL2 V6.8H, V0.16B, V2.16B
	WORD $0x4E6068B5 // SADALP V21.4S, V5.8H
	WORD $0x4E6068D5 // SADALP V21.4S, V6.8H
	WORD $0x0E23C005 // SMULL  V5.8H, V0.8B, V3.8B
	WORD $0x4E23C006 // SMULL2 V6.8H, V0.16B, V3.16B
	WORD $0x4E6068B6 // SADALP V22.4S, V5.8H
	WORD $0x4E6068D6 // SADALP V22.4S, V6.8H
	WORD $0x0E24C005 // SMULL  V5.8H, V0.8B, V4.8B
	WORD $0x4E24C006 // SMULL2 V6.8H, V0.16B, V4.16B
	WORD $0x4E6068B7 // SADALP V23.4S, V5.8H
	WORD $0x4E6068D7 // SADALP V23.4S, V6.8H
	SUBS $16, R11, R11
	BNE  fourChunk

	WORD $0x4EB5BE98 // ADDP V24.4S, V20.4S, V21.4S  [a01 a23 b01 b23]
	WORD $0x4EB7BED9 // ADDP V25.4S, V22.4S, V23.4S  [c01 c23 d01 d23]
	WORD $0x4EB9BF1A // ADDP V26.4S, V24.4S, V25.4S  [a b c d]
	VST1.P [V26.S4], 16(R0)

	ADD  R5<<2, R2, R2
	SUB  $4, R3, R3
	B    four

one:
	CBZ R3, done

	MOVD R2, R6
	MOVD R1, R10
	MOVD R4, R11
	VEOR V20.B16, V20.B16, V20.B16

oneChunk:
	VLD1.P 16(R10), [V0.B16]
	VLD1.P 16(R6), [V1.B16]
	WORD $0x0E21C005 // SMULL  V5.8H, V0.8B, V1.8B
	WORD $0x4E21C006 // SMULL2 V6.8H, V0.16B, V1.16B
	WORD $0x4E6068B4 // SADALP V20.4S, V5.8H
	WORD $0x4E6068D4 // SADALP V20.4S, V6.8H
	SUBS $16, R11, R11
	BNE  oneChunk

	WORD   $0x4EB1BA94 // ADDV S20, V20.4S
	FMOVS  F20, (R0)
	ADD    $4, R0
	ADD    R5, R2, R2
	SUB    $1, R3, R3
	B      one

done:
	RET

// func axpysI8NEON(out, w *float32, v *int8, n, width, stride int)
//
// out[d] += Σ_{j < n} w[j] · v[j·stride + d] for d < width; width is a
// positive multiple of 16, n ≥ 1. Per chunk of 16 outputs, the four
// accumulators stay in V16–V19 while the keys run in order; each key's 16
// codes are sign-extended to int32 (SSHLL twice), converted (SCVTF),
// multiplied by w[j] and added: FMUL then FADD, no FMA, as
// axpysI8Generic's float32(w · c) + out.
//
// Args (ABI0): out +0, w +8, v +16, n +24, width +32, stride +40; 48 bytes.
//
// Registers: R0 out, R1 w, R2 v, R3 n, R4 width, R5 stride; R6 the out
// chunk, R7 outputs left, R8 the chunk's column in v, R9 the w cursor,
// R10 the v cursor, R11 keys left; F31 w[j]; V0 codes, V1–V6 widened.
TEXT ·axpysI8NEON(SB), NOSPLIT, $0-48
	MOVD out+0(FP), R0
	MOVD w+8(FP), R1
	MOVD v+16(FP), R2
	MOVD n+24(FP), R3
	MOVD width+32(FP), R4
	MOVD stride+40(FP), R5

	MOVD R0, R6
	MOVD R4, R7
	MOVD R2, R8

chunk:
	VLD1 (R6), [V16.S4, V17.S4, V18.S4, V19.S4]
	MOVD R1, R9
	MOVD R8, R10
	MOVD R3, R11

key:
	FMOVS.P 4(R9), F31
	VLD1    (R10), [V0.B16]
	ADD     R5, R10, R10
	WORD $0x0F08A401 // SSHLL  V1.8H, V0.8B, #0
	WORD $0x4F08A402 // SSHLL2 V2.8H, V0.16B, #0
	WORD $0x0F10A423 // SSHLL  V3.4S, V1.4H, #0
	WORD $0x4F10A424 // SSHLL2 V4.4S, V1.8H, #0
	WORD $0x0F10A445 // SSHLL  V5.4S, V2.4H, #0
	WORD $0x4F10A446 // SSHLL2 V6.4S, V2.8H, #0
	WORD $0x4E21D863 // SCVTF  V3.4S, V3.4S
	WORD $0x4E21D884 // SCVTF  V4.4S, V4.4S
	WORD $0x4E21D8A5 // SCVTF  V5.4S, V5.4S
	WORD $0x4E21D8C6 // SCVTF  V6.4S, V6.4S
	WORD $0x4F9F9063 // FMUL   V3.4S, V3.4S, V31.S[0]
	WORD $0x4F9F9084 // FMUL   V4.4S, V4.4S, V31.S[0]
	WORD $0x4F9F90A5 // FMUL   V5.4S, V5.4S, V31.S[0]
	WORD $0x4F9F90C6 // FMUL   V6.4S, V6.4S, V31.S[0]
	WORD $0x4E23D610 // FADD   V16.4S, V16.4S, V3.4S
	WORD $0x4E24D631 // FADD   V17.4S, V17.4S, V4.4S
	WORD $0x4E25D652 // FADD   V18.4S, V18.4S, V5.4S
	WORD $0x4E26D673 // FADD   V19.4S, V19.4S, V6.4S
	SUBS $1, R11, R11
	BNE  key

	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R6)
	ADD    $16, R8, R8
	SUBS   $16, R7, R7
	BNE    chunk

	RET
