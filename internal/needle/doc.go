// Package needle implements the Needle 3 blocks that Whistle's encoder and
// decoder share, in float32, with upstream's semantics
// (needle/model/architecture.py @ af2654dffa; PLAN.md "Shared Needle 3
// blocks"; docs/FINDINGS.md F-011, F-013):
//
//   - ZCRMSNorm and RMSUnit (architecture.py ZCRMSNorm, _rms_unit);
//   - Attention (MultiHeadAttention with flash=False): projections,
//     optional causal Q/K/V taps, per-head q/k ZCRMSNorm, half-split RoPE,
//     GQA and the sigmoid output gate, causal or unmasked;
//   - HadamardMLP: the Monarch Kronecker FFN with its rank-8 conditional
//     scale and the permutations hada_p1/p2;
//   - Block: the decoder layer (engram gate, ZCRMSNorm → Attention →
//     σ(attn_gate) · post_attn_norm, pre_hada_norm → HadamardMLP), without
//     the cross-attention, which upstream's Block lacks;
//   - MHC (_ScanBody's advance): the n-lane residual stream around a block,
//     its pre/post/res heads and the 20-iteration log-domain Sinkhorn, with
//     ExpandLanes and CollapseLanes for Stack's start and end;
//   - Engram: the uint32 n-gram hash indices, the table lookup, the
//     key/value projections and the dilated value taps.
//
// Every block loads its weights from a weights.Source by name prefix
// ("layer00.", "enc00.", "enc00.hada_0.", ...), so the encoder and the
// decoder share the code, and the .cact and safetensors formats serve the
// same layout ([out, in] projections, docs/FINDINGS.md F-006). Weight
// slices are the Source's cached values and are never written.
//
// Activations are one sequence at a time, row-major (T, width): row t is
// time step t, and a per-head tensor keeps head h of row t at columns
// h·d … (h+1)·d, which is upstream's reshape(B, T, heads, d) before its
// transpose. The parity tests hold every block to the fixture contract of
// testdata/needle/README.md.
package needle
