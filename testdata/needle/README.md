# Needle 3 block fixtures

Reference inputs, outputs and intermediates of the shared Needle 3 blocks, for `internal/needle`. They are written by
`scripts/needle_fixtures.py` from upstream's Flax modules (`needle/model/architecture.py` @ `af2654dffa`), bound to the
real Whistle parameters of `whistle.safetensors` @ `b358ddad`. This file is the contract the Go blocks are held to;
`docs/FINDINGS.md` F-009 records how it was built.

## Regenerate

```bash
just needle-fixtures              # writes the seven files below
just needle-fixtures --f64-check  # writes nothing; prints each tensor's distance to a float64 rerun
```

This needs `models/whistle/whistle.safetensors` (`whistle download`) and `original/needle` (`just fetch-upstream`). The
run is deterministic: two runs give byte-identical files. The inputs come from `np.random.default_rng(seed)` with the
seed in each file's metadata. The script sorts the `__metadata__` keys itself, because safetensors writes them in hash
map order.

## Numerics

- float32 everywhere: every module gets `dtype=float32` (upstream defaults to bfloat16), `jax_enable_x64` is off,
  `jax_default_matmul_precision` is `"highest"`, and the platform is CPU.
- `quant=False`: no fake quantisation of activations, K/V or weights. The engine's W2A8 numerics are not modelled.
- `MultiHeadAttention` runs with `flash=False`, the explicit softmax path. `flash=True`, which goes through
  `jax.nn.dot_product_attention`, differs by up to 1.3e-4 at |out| ≤ 329. That is float32 noise of a different
  summation order.
- Every intermediate is either a submodule's own output (Flax `capture_intermediates`) or a recomputation from
  upstream's helpers that reproduces the module's output exactly (0 difference) before the file is written.

## Weights

No weights are stored. A test loads them from `whistle.safetensors` by the Flax path in the tables below, sliced at
the layer in brackets. Each file's `weights` metadata lists the same paths together with the `.cact` record from
`cact.WhistleNames()`.

**Use the FP32 safetensors weights.** The `.cact` records are CQ2/CQ4 (cosine 0.93–0.995 against the source) or FP16
(the norm scales). Neither can meet the tolerance below. The `.cact` names are listed to show which record a block
reads in the quantised path.

Flax Dense kernels are `(in, out)` and compute `y = x @ kernel`; `.cact` stores them transposed as `(out, in)`.

## Tolerance

Compare every float tensor by both of these:

- max |go − fixture| ≤ 1e-5 · max(1, max |fixture|), the 1e-5 taken relative to the tensor's largest magnitude;
- cosine(go, fixture) ≥ 0.99999.

Compare I64 tensors exactly.

A flat 1e-5 absolute bound is not achievable. With the real weights, block outputs reach about 1.5e3, and the float32
fixtures themselves lie up to 1.3e-3 from a float64 rerun of the same code. Measured by `--f64-check`:

| file                | max \|value\| | max abs to float64 | worst (abs to float64) / max(1, max \|tensor\|) | min cosine     |
| ------------------- | ------------- | ------------------ | ----------------------------------------------- | -------------- |
| `zcrmsnorm`         | 281           | 3.9e-7             | 1.1e-7 (`rms_unit`)                             | 1.000000000000 |
| `attention_decoder` | 329           | 5.1e-4             | 3.4e-6 (`gate`)                                 | 1.000000000000 |
| `attention_encoder` | 1200          | 4.2e-4             | 3.9e-6 (`gate`)                                 | 1.000000000000 |
| `hadamard_mlp`      | 1480          | 9.7e-4             | 6.6e-7 (`out`)                                  | 1.000000000000 |
| `mhc`               | 1160          | 7.7e-4             | 2.0e-6 (`layer1.stream_out`)                    | 0.999999999999 |
| `engram`            | 152           | 7.2e-5             | 1.2e-6 (`v_raw`)                                | 1.000000000000 |
| `block`             | 1460          | 1.3e-3             | 1.8e-6 (`h`)                                    | 1.000000000000 |

A correct float32 Go port carries noise of the same size, so the bound leaves about 2.5× headroom over the sum of both.

## Layout

Row-major, upstream's shapes: activations are `(B, T, ·)`, and per-head tensors are `(B, heads, T, head_dim)`. Heads
are split as `reshape(B, T, heads, head_dim)` followed by the transpose. GQA uses `jnp.repeat`, so query head `h` reads
KV head `h // 4`. `shift_right(z, j)` shifts along T and fills with zeros. All masks are causal (`mask[q, k] = k ≤ q`)
unless a file says `none`.

Common metadata (all strings): `block`, `layer`, `seed`, `upstream` (`af2654dffa`), `weights_pin` (`b358ddad`),
`weights` (JSON list), `shapes` (JSON map of name to shape), `dtype` (`float32`), `matmul_precision`, `versions` (jax,
flax, numpy), plus the block-specific keys named below.

## `zcrmsnorm.safetensors`

`ZCRMSNorm(eps=1e-6)` and `_rms_unit`. Weight: `stack/layers/block/ZCRMSNorm_0/scale[0]` = `.cact` 1 `layer00.norm_in`.

| tensor     | shape       | meaning                                                                                                                                                           |
| ---------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x`        | [2, 8, 512] | input. Row scales `[[1, 2, 0.5, 10, 100, 0.1, 0.01, 1], [1e-3, 1e-4, 1e-5, 0, 1, 1, 1, 1]]`; row (1, 3) is all zeros, and row (1, 5) has +3 added (non-zero mean) |
| `rms`      | [2, 8, 1]   | `sqrt(mean(x², -1) + 1e-6)`                                                                                                                                       |
| `y`        | [2, 8, 512] | `(1 + scale) · x / rms`                                                                                                                                           |
| `rms_unit` | [2, 8, 512] | `x · rsqrt(mean(x², -1) + 1e-6)`, no scale                                                                                                                        |

## `attention_decoder.safetensors`

`MultiHeadAttention`, decoder layer 0: 8 query heads and 2 KV heads, qk 48, v 64, 3 taps, causal mask, RoPE θ = 1e5.
Metadata: `mask`, `taps`, `rope_theta`, `flash`.

| tensor                             | shape                                | meaning                                                                                           |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `x`                                | [2, 8, 512]                          | input (in a Block, the output of `ZCRMSNorm_0`)                                                   |
| `q_proj`, `k_proj`, `v_proj`       | [2, 8, 384], [2, 8, 96], [2, 8, 128] | `x @ kernel`                                                                                      |
| `q_tapped`, `k_tapped`, `v_tapped` | as above                             | `Σ_{j<3} taps[j] · shift_right(z, j)`; the causal tap mask is all ones where the shift is defined |
| `q_normed`, `k_normed`             | [2, 8, 8, 48], [2, 2, 8, 48]         | per-head `ZCRMSNorm` (`q_norm`, `k_norm`) of the tapped values                                    |
| `rope_cos`, `rope_sin`             | [8, 24]                              | `precompute_rope_freqs(48, 8, 1e5)`: angle `t / θ^(2i/48)`                                        |
| `q_rope`, `k_rope`                 | as `*_normed`                        | half split: `[x1·cos − x2·sin, x2·cos + x1·sin]` with `x1, x2` the first and last 24 dims         |
| `probs`                            | [2, 8, 8, 8]                         | `softmax(q kᵀ / √48)` over keys, masked logits set to float32 min (so exactly 0)                  |
| `attn`                             | [2, 8, 512]                          | `probs @ v`, heads concatenated (8 × 64)                                                          |
| `gate`                             | [2, 8, 512]                          | `sigmoid(x @ gate_proj)`                                                                          |
| `gated`                            | [2, 8, 512]                          | `attn · gate`                                                                                     |
| `out`                              | [2, 8, 512]                          | `gated @ out_proj`, the module output                                                             |

| Flax path (`stack/layers/block/self_attn/…[0]`)   | `.cact`                                                    |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `q_proj/kernel`, `k_proj/kernel`, `v_proj/kernel` | 2 `layer00.q_proj`, 3 `layer00.k_proj`, 4 `layer00.v_proj` |
| `q_taps`, `k_taps`, `v_taps`                      | 5 `layer00.q_taps`, 6 `layer00.k_taps`, 7 `layer00.v_taps` |
| `q_norm/scale`, `k_norm/scale`                    | 8 `layer00.q_norm`, 9 `layer00.k_norm`                     |
| `gate_proj/kernel`, `out_proj/kernel`             | 10 `layer00.gate_proj`, 11 `layer00.out_proj`              |

## `attention_encoder.safetensors`

The same module with encoder layer 0's attention weights, `qkv_conv_taps=0` and no mask. The encoder has no taps and
the safetensors file has none. `MultiHeadAttention` expresses it exactly. Whether the encoder applies RoPE is still
open (PLAN.md U3), so both variants are stored. Metadata: `mask` (`none`), `taps` (`0`), `rope_theta`.

| tensor                                                                                             | shape       | meaning                                                          |
| -------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------- |
| `x`, `q_proj`, `k_proj`, `v_proj`, `q_normed`, `k_normed`, `probs`, `attn`, `gate`, `gated`, `out` | as above    | as in `attention_decoder`, without taps, RoPE or mask            |
| `out_rope`                                                                                         | [2, 8, 512] | the module output with the decoder's RoPE tables (θ = 1e5) added |

| Flax path (`encoder/layers/block/self_attn/…[0]`) | `.cact`                                                    |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `q_proj/kernel`, `k_proj/kernel`, `v_proj/kernel` | 325 `enc00.q_proj`, 326 `enc00.k_proj`, 327 `enc00.v_proj` |
| `gate_proj/kernel`, `out_proj/kernel`             | 328 `enc00.gate_proj`, 329 `enc00.out_proj`                |
| `q_norm/scale`, `k_norm/scale`                    | 330 `enc00.q_norm`, 331 `enc00.k_norm`                     |

## `hadamard_mlp.safetensors`

`HadamardMLP(512)`, decoder layer 0: Monarch blocks 16 × 32, conditional rank 8, no split permutations. Metadata:
`cond_rank`, `blocks`, `split_perms`. `z[p]` is a gather: `out[i] = z[p[i]]`. `kron(z; a, b)` is `_kron_apply`:

```text
Z = z.reshape(16, 32);  out[k, l] = Σ_ij Z[i, j] · a[i, k] · b[j, l];  out.reshape(512)
```

| tensor        | shape       | meaning                                                                                         |
| ------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| `x`           | [2, 8, 512] | input (in a Block, the output of `pre_hada_norm`)                                               |
| `cond_logits` | [2, 8, 8]   | `x @ cond_v`                                                                                    |
| `cond`        | [2, 8, 512] | `1 + softmax(cond_logits) @ cond_u`                                                             |
| `z1`          | [2, 8, 512] | `kron(d1 · x; w1a, w1b)[p1]`                                                                    |
| `a2`          | [2, 8, 512] | `d2 · cond · z1 + b2`, the SiLU input                                                           |
| `z2`          | [2, 8, 512] | `kron(silu(a2); w2a, w2b)[p2]`                                                                  |
| `z3`          | [2, 8, 512] | `kron(d3 · z2; w3a, w3b)`                                                                       |
| `out`         | [2, 8, 512] | `d4 · z3`, the module output                                                                    |
| `p1`, `p2`    | [512] I64   | `_hada_perms(512)`: `RandomState(11 / 13).permutation(512)`, constants equal to `.cact` 226/227 |

Weights `stack/layers/block/hadamard_mlp/…[0]`: `d1` 15, `d2` 16, `b2` 17, `d3` 18, `d4` 19, `w1a` 20, `w1b` 21,
`w2a` 22, `w2b` 23, `w3a` 24, `w3b` 25, `cond_v` 26, `cond_u` 27 (`.cact` `layer00.<name>`).

## `mhc.safetensors`

One `_ScanBody` step with 4 lanes and real mHC heads, in two cases with batch 1 and T = 8. **`layer0`** starts from
`Stack`'s lane expand (`x` broadcast to 4 lanes; that stream is not stored) and also collapses the result as `Stack`
ends: the lane mean and `final_norm`, applied here after one layer. **`layer1`** feeds four different lanes, scaled 1,
0.5, 2 and 0.25, into layer 1. That exercises the Sinkhorn mix and the offsets of lane 1. Engram is off in both cases,
since the site flags are 0 at layers 0 and 1. Metadata: `lanes`, `sinkhorn_iters`, `mask`, `rope_theta`, `engram`.

`lane = eye(4)[layer % 4]`, `pre_off = 8·lane − 4`, `post_off = −4·(1 − lane)`.

| tensor (`layerN.` prefix) | shape          | meaning                                                                               |
| ------------------------- | -------------- | ------------------------------------------------------------------------------------- |
| `x` (no prefix)           | [1, 8, 512]    | `layer0` input before the lane expand                                                 |
| `layer1.stream_in`        | [1, 8, 4, 512] | `layer1` input stream                                                                 |
| `nx`                      | [1, 8, 2048]   | `rms_unit(stream.reshape(B, T, 4·512))`, lanes concatenated lane-major                |
| `h_pre_logits`, `h_pre`   | [1, 8, 4]      | `a_pre · (nx @ phi_pre) + b_pre + pre_off`, then the sigmoid                          |
| `u`                       | [1, 8, 512]    | `Σ_n h_pre[n] · stream[n]`, the block input                                           |
| `block_out`               | [1, 8, 512]    | `Block_layer(u)`: causal, RoPE, taps, no engram                                       |
| `y`                       | [1, 8, 512]    | `block_out − u`                                                                       |
| `h_post_logits`, `h_post` | [1, 8, 4]      | `a_post · (nx @ phi_post) + b_post + post_off`, then `2 · sigmoid`                    |
| `h_res_logits`            | [1, 8, 4, 4]   | `a_res · (nx @ phi_res).reshape(4, 4) + b_res`; entry `[i, j]` is column `4i + j`     |
| `h_res`                   | [1, 8, 4, 4]   | `_sinkhorn`: 20 × (subtract the row logsumexp, then the column logsumexp), then `exp` |
| `stream_out`              | [1, 8, 4, 512] | lane `i`: `Σ_j h_res[i, j] · stream[j] + h_post[i] · y`, the `_ScanBody` output       |
| `layer0.collapsed`        | [1, 8, 512]    | mean over the lanes of `layer0.stream_out`                                            |
| `layer0.final`            | [1, 8, 512]    | `ZCRMSNorm(final_norm)(layer0.collapsed)`                                             |

| Flax path                                                   | `.cact`                                                                                      |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `stack/mhc_a_pre`, `a_post`, `a_res` `[L]`                  | 217, 218, 219, element L                                                                     |
| `stack/mhc_b_pre`, `b_post` `[L]` (4); `b_res` `[L]` (4, 4) | 220, 221, 222, row L                                                                         |
| `stack/mhc_phi_pre`, `phi_post` `[L]` (2048, 4)             | 223, 224 lane-major: rows 4L … 4L+3 are the 4 columns                                        |
| `stack/mhc_phi_res` `[L]` (2048, 16)                        | 225 lane-major: rows 16L … 16L+15 are the 16 columns                                         |
| `stack/layers/block/…[L]`                                   | the 27 block weights of layer L (`.cact` 1–27 for layer 0, 28–54 for layer 1), as in `block` |
| `stack/final_norm/scale`                                    | 236 `final_norm` (`layer0.final` only)                                                       |

## `engram.safetensors`

`Engram` at site 0 (decoder layer 3, weights `engrams_0`), batch 2, T = 12, plus the gate that `Block` applies at
layer 3. Orders (2, 3), 2 hashes each, 4 tables of 18432 × 128, conv dilation 3. Metadata: `site`, `orders`, `hashes`,
`slots`, `dilation`, `mask`.

| tensor     | shape          | meaning                                                                                                                                                                                                                                     |
| ---------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tokens`   | [2, 12] I64    | input ids in [0, 8199). `tokens[0, 6:9]` repeats `tokens[0, 1:4]`, and `tokens[1, 0]` is the pad id 0                                                                                                                                       |
| `indices`  | [2, 12, 4] I64 | `engram_indices`: column `c = 2·order_index + hash`; `acc = (0x9E3779B9 · (c + 1)) mod 2³²`, then for `j < order`: `acc = (acc ^ tok[t − j]) · 0x01000193 mod 2³²` (`tok[t − j] = 0` before the start); `acc ^= acc >> 15`; `acc mod 18432` |
| `ngram_ok` | [12, 4]        | 1 where `t ≥ order − 1`. Upstream's shape is (1, T, 4). Indices at earlier positions exist (hashed over zero padding) but are masked                                                                                                        |
| `tap_ok`   | [4, 12]        | 1 where `t ≥ 3j`. Upstream's shape is (4, 1, T)                                                                                                                                                                                             |
| `e`        | [2, 12, 512]   | `e[b, t, 128c : 128(c+1)] = tables[c, indices[b, t, c]] · ngram_ok[t, c]`                                                                                                                                                                   |
| `k`        | [2, 12, 512]   | `e @ key_proj`                                                                                                                                                                                                                              |
| `v_raw`    | [2, 12, 512]   | `e @ value_proj`                                                                                                                                                                                                                            |
| `v`        | [2, 12, 512]   | `Σ_{j<4} taps[j] · shift_right(v_raw, 3j) · tap_ok[j]`                                                                                                                                                                                      |
| `x`        | [2, 12, 512]   | hidden state entering `Block` at layer 3                                                                                                                                                                                                    |
| `alpha`    | [2, 12]        | `sigmoid(⟨rms_unit(x), rms_unit(k)⟩ / √512)`                                                                                                                                                                                                |
| `x_out`    | [2, 12, 512]   | `x + alpha · v` (site flag 1). The script checks that `Block_3(x, engram)` equals `Block_3(x_out)`                                                                                                                                          |

| Flax path                     | `.cact`                                          |
| ----------------------------- | ------------------------------------------------ |
| `engrams_0/embedding`         | 228 `engram0.tables`, stored as (4 · 18432, 128) |
| `engrams_0/key_proj/kernel`   | 229 `engram0.key_proj`                           |
| `engrams_0/value_proj/kernel` | 230 `engram0.value_proj`                         |
| `engrams_0/taps`              | 231 `engram0.taps`                               |

## `block.safetensors`

One decoder `Block` at layer 0: causal mask, RoPE θ = 1e5 and 3 taps. It has no cross-attention, which upstream's
`Block` does not contain, and no engram, since layer 0 is not a site. Metadata: `mask`, `taps`, `rope_theta`, `flash`,
`cross_attention`, `engram`.

| tensor      | shape       | meaning                              |
| ----------- | ----------- | ------------------------------------ |
| `x`         | [2, 8, 512] | input                                |
| `norm_in`   | [2, 8, 512] | `ZCRMSNorm_0(x)`                     |
| `attn`      | [2, 8, 512] | `self_attn(norm_in)`                 |
| `post_attn` | [2, 8, 512] | `post_attn_norm(attn)`               |
| `h`         | [2, 8, 512] | `x + sigmoid(attn_gate) · post_attn` |
| `pre_hada`  | [2, 8, 512] | `pre_hada_norm(h)`                   |
| `mlp`       | [2, 8, 512] | `hadamard_mlp(pre_hada)`             |
| `out`       | [2, 8, 512] | `h + mlp`, the block output          |

Weights `stack/layers/block/…[0]`, all 27 of `.cact` 1–27: `ZCRMSNorm_0/scale` 1 `layer00.norm_in`, the attention
weights as in `attention_decoder`, `post_attn_norm/scale` 12 `layer00.post_norm`, `attn_gate` 13 `layer00.attn_gate`,
`pre_hada_norm/scale` 14 `layer00.pre_hada`, and the MLP weights as in `hadamard_mlp`. The safetensors file also holds
the decoder's cross-attention (`cross_attn/*`, `cross_norm`, `post_cross_norm`, `cross_gate`), which no upstream
module reads.
