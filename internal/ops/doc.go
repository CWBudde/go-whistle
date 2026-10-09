// Package ops holds the neural-network operators on top of package tensor:
// scaled dot-product attention with a causal mask and offset, interleaved
// RoPE from precomputed cos/sin tables, and Conv1D (im2col + GEMM for
// groups=1, a direct loop for grouped and depthwise convolutions).
//
// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/ops @
// 9faba38 without ConvTranspose1D, MLP and the Pocket-TTS position/context
// attention; each file names its source (docs/FINDINGS.md F-008).
package ops
