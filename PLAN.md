# PLAN

> Roadmap for porting **Whistle** (Cactus Compute speech-to-text) to Go, using as much native Go as possible: no cgo, no
> Python and no ONNX at runtime.

## Why this is not a straight translation

In [cactus-compute/needle](https://github.com/cactus-compute/needle), Whistle is `needle/agent/whistle.py`: a ctypes
wrapper (187 lines, plus about 300 lines of WAV I/O and terminal UI) around the **closed-source C++ engine**
`libneedle3`. That engine ships only as prebuilt binaries in HF `Cactus-Compute/needle3`. The public sources contain no
speech model code at all: no mel front end, no encoder and no decoding. So the port is a **re-implementation from the
weights**, using these sources:

| Source                                                                 | What it gives us                                                                                                                                       |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HF `Cactus-Compute/whistle` `checkpoints/whistle.safetensors`          | All 55,151,433 parameters as FP32 with real tensor names; the model config is in `__metadata__["config"]`                                              |
| HF `Cactus-Compute/whistle` `whistle.cact`                             | The shipped 16.9 MB quantised model. It also carries the **mel filterbank**, the CQ **codebooks**, the Hadamard **permutations** and the **tokenizer** |
| `needle/model/architecture.py` (JAX/Flax)                              | The reference implementation of the shared Needle 3 blocks: ZCRMSNorm, attention, HadamardMLP, mHC, engram. The decoder **is** this code               |
| `needle/model/export.py`, `quantize.py`, `tokenizer.py`                | The `.cact` container format, CQ pack/unpack and the SentencePiece reference tokenizer                                                                 |
| `needle/agent/whistle.py`, `tests/test_whistle.py`, `llms.txt`         | The C ABI, the JSON schemas, WAV conversion, edge cases (silence, the 30 s limit)                                                                      |
| cactuscompute.com/whistle, `/blog/cact-format`, `/blog/porting-needle` | Prose on the pipeline (stem, conformer encoder, cross-attention, beam search, keyword biasing, streaming)                                              |
| Prebuilt `needle` binary / `libneedle3`                                | A **black-box oracle**: `needle_embed` returns the encoder output per 80 ms frame, and `needle_transcribe` returns the final JSON                      |

Everything the sources leave open is listed under [Unknowns](#unknowns-to-resolve-against-the-oracle). Each one is
settled against the oracle and recorded in `docs/FINDINGS.md`.

## Upstream pins

|                 | Where                                   | Pin                                                         |
| --------------- | --------------------------------------- | ----------------------------------------------------------- |
| Python / JAX    | `github.com/cactus-compute/needle`      | `af2654dffa` (2026-10-07), package `cactus-needle` 3.1.0    |
| Weights         | `huggingface.co/Cactus-Compute/whistle` | `b358ddad`, `weights_version` 2.0.0                         |
| Engine (oracle) | `huggingface.co/Cactus-Compute/needle3` | engine 3.2.0 (`ENGINE_VERSIONS` in `needle/agent/fetch.py`) |

Licenses: the code and weights are Apache-2.0. The engine binaries have no published source licence, so they are used
**locally as a test oracle only**, are never redistributed and never become a runtime dependency. Do not use
`cactus-compute/cactus` (the C++ runtime under a non-commercial licence) as a reference; it contains no Whistle code anyway.

---

## Model reference card

### Config (`whistle.safetensors` → `__metadata__["config"]`)

```json
{
  "text": 8192,
  "dim": 512,
  "kv_heads": 2,
  "qk_dim": 48,
  "v_dim": 64,
  "enc_layers": 8,
  "dec_layers": 8,
  "conv_kernel": 9,
  "taps": 3,
  "lanes": 4,
  "enc_lanes": 4,
  "sites": [3, 7],
  "orders": [2, 3],
  "slots": 18432,
  "hashes": 2,
  "stem": 128,
  "stem_stages": 3,
  "rope": 100000.0,
  "mels": 80,
  "rate": 16000,
  "fft": 512,
  "win": 400,
  "hop": 160,
  "max_tokens": 320,
  "pad": 0,
  "eos": 1,
  "bos": 2,
  "dtype": "bfloat16",
  "freq_masks": 2,
  "freq_width": 17,
  "time_masks": 10,
  "time_width": 0.05,
  "dither": 1e-5,
  "per_augment_prob": 0.1,
  "speed": [0.9, 1.1]
}
```

- The run metadata includes `"stage": "qapt"` and
  `weight_bits "embedding=4,stack/mhc=4,encoder/mhc=4,default=2"`, `kv_bits 8`, `act_bits 8`. The engine runs W2A8
  with 4-bit embeddings and mHC.
- 8 query heads: `q_proj` is 512→384 = 8×48. GQA is 8q:2kv with qk 48 and v 64.
- The masks, speed, dither and augment keys are **training-only**; ignore them.

### Pipeline

1. **Input**: 16 kHz mono float32 in [-1, 1], at most 30 s (480,000 samples) per pass.
2. **Front end**:
   - 80-bin log-mel; n_fft 512; Hann window of 400 samples, zero-padded to 512; hop 160.
   - 30 s gives 3000 frames.
   - The filterbank is stored in `.cact` tensor 679, FP32 `(257, 80)`, Slaney-normalised over the full 0–8 kHz.
3. **Conv stem** (2-D over time × mel):
   - `stem/w [3,3,1,128]` is a full 3×3 conv from 1 to 128 channels.
   - Then two depthwise-separable stages: `stem/dw_{1,2} [3,3,1,128]` + `stem/pw_{1,2} [128,128]`.
   - Three stride-2 halvings take 3000×80 down to 375×10, i.e. 80 ms frames.
   - Flatten to 1280 features, then `stem/out/kernel [1280,512]`.
4. **Encoder**: 8 non-causal blocks, width 512, 4 mHC lanes, then `encoder/final_norm`. Per block, in Conformer
   macaron order:
   - `pre_hada_norm_0 → hadamard_mlp_0` (FFN1)
   - `ZCRMSNorm_0 → self-attention (GQA 8:2, q/k ZCRMSNorm, sigmoid output gate, no QKV taps) → post_attn_norm · σ(attn_gate)`
   - `conv_norm → pw1 (512→1024) → [GLU?] → depthwise dw (kernel 9) → conv_out_norm → pw2 (512→512)`
   - `pre_hada_norm → hadamard_mlp` (FFN2)
   - Also a global scalar `pe_gate` ≈ −6.11, so σ ≈ 0.002. Its use is unknown.
5. **Decoder**: 8 Needle 3 `Block`s, width 512, with:
   - a 3-tap causal conv on Q/K/V and RoPE θ = 1e5;
   - engram memory at layers 3 and 7;
   - 4 mHC lanes;
   - **plus gated cross-attention per layer**: `x ← x + σ(cross_gate) · softmax(q̂ K̂ᵀ/√48) V`.
     The per-layer tensors are `cross_norm`, `q_proj 512→384`, `k_proj 512→384` (8 heads, not GQA),
     `v_proj 512→512`, `gate_proj`, `out_proj`, `q_norm`, `k_norm`, `post_cross_norm` and `cross_gate`.
   - Encoder K/V are projected once per clip and shared by all beams.
   - The embedding `[8199, 512]` is tied with the output head. Input scaling is `sqrt(512)`.
6. **Decoding**:
   - Beam search with 5 beams and length-normalised log-probability.
   - Optional keyword biasing through an Aho-Corasick automaton.
   - At most 320 tokens.
   - The language is emitted as a token: `<|en|>`=8192, de, fr, es, it, nl, `<|pl|>`=8198.
7. **Extras**:
   - Word timestamps come "from the decoder's own attention".
   - A silence gate measures the clip's loudness range before decoding.
   - The `--audio-depth N` ladder uses decoder layers in bisection order **0, 7, 3, 5, 1, 2, 4, 6**; the encoder always runs all 8.
   - Streaming re-transcribes about 1 s chunks and commits words that two consecutive passes agree on.

### Shared Needle 3 blocks (from `architecture.py`, verbatim semantics)

```python
# ZCRMSNorm, eps 1e-6, zero-centred scale
rms = sqrt(mean(x**2, -1) + eps);  y = (1 + scale) * x / rms

# Attention: q/k/v proj → optional causal taps q = Σ_j qt[j]·shift_right(q, j)
# → heads → ZCRMSNorm on q and k per head → RoPE (half split: x1*cos - x2*sin, x2*cos + x1*sin,
#   freqs = 1/θ^(arange(0,hd,2)/hd)) → softmax(qkᵀ/√48)·v (GQA repeat)
out = out * sigmoid(gate_proj(x));  return out_proj(out)

# HadamardMLP (Monarch, n=512 → 16×32 blocks, HADA_COND_RANK = 8,
# perms = np.random.RandomState(11 / 13).permutation(512))
cond = 1 + softmax(x @ cond_v) @ cond_u
z = kron_apply(d1 * z, *f[0])[..., p1]
z = kron_apply(silu(d2 * cond * z + b2), *f[1])[..., p2]
z = kron_apply(d3 * z, *f[2]);  return (d4 * z)[..., :d_model]
# kron_apply: z.reshape(..., a.n, b.n); einsum("...ij,ik,jl->...kl", z, a, b)

# Block (decoder)
if engram: x += Σ_s site_flag[s] · σ(<rms_unit(x), rms_unit(ek)>/√d) · ev
skip = x; x = attn(ZCRMSNorm(x)); x = skip + σ(attn_gate) · post_attn_norm(x)
skip = x; x = skip + HadamardMLP(pre_hada_norm(x))

# mHC (4 lanes, lane = eye(n)[layer % n], pre_off = 8·lane − 4, post_off = −4·(1 − lane), Sinkhorn 20 iterations)
nx = rms_unit(stream.reshape(B, T, n·C))
hpre = σ(a_pre·(nx @ φ_pre) + b_pre + pre_off);  u = Σ_n hpre·x_n
y = block(u) − u
hpost = 2σ(a_post·(nx @ φ_post) + b_post + post_off)
hres = sinkhorn(a_res·(nx @ φ_res).reshape(n, n) + b_res)
stream' = hres @ stream + hpost ⊗ y          # input: broadcast to n lanes; output: lane mean, then final_norm

# Engram (seed 0x9E3779B9, prime 0x01000193, uint32 arithmetic; 2 orders × 2 hashes = 4 tables, sub_dim 128,
# 4 conv taps, dilation = max(orders) = 3)
seed = (SEED * (oi*stride + h + 1)) & 0xFFFFFFFF;  acc = seed
for j in range(order): acc = (acc ^ shift_right(u, j)) * PRIME
acc ^= acc >> 15;  idx = acc % slots
k = key_proj(e); v = value_proj(e); v = Σ_j taps[j] · shift_right(v, 3j)
```

### `.cact` container

- Header: 196 bytes (`<48If`), tag `0x05E12A84`. **Note:** the `export.py` docstring wrongly says 120 bytes.
- Then 28 FP32 codebook floats: CQ2[4] | CQ3[8] | CQ4[16].
- Then 681 × 44-byte records (`<BBHIIIIQQII`: dtype, ndim, pad, shape[4], offset, nbytes, group, bits).
- Blobs are 64-byte aligned, and matrices are stored `[out, in]`. Dtypes: 1=FP16, 2=FP32, 3=CQ, 4=RAW.
- **CQ dequantisation** per 128-weight group: `w = (codebook[idx] · fp16_norm) @ H128`, where H128 is the normalised
  Walsh-Hadamard matrix. The indices are an LSB-first bitstream per row. CQ2 costs 2.125 bits/weight and CQ4 4.125.
- **Read the codebooks from the header.** Upstream derives them from `np.random.RandomState(0).randn(400000)`
  (Lloyd-Max), which would otherwise mean porting numpy's MT19937 + Gaussian generator.

Whistle tensor map. It was **derived by value-matching against safetensors and must be verified in Phase 1**:

| Index   | Contents                                                                                                                                                                     |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0       | embedding CQ4 `(8199, 512)`                                                                                                                                                  |
| 1–216   | 8 decoder layers × 27, in Needle order: norm_in, q, k, v, q/k/v taps, q_norm, k_norm, gate, out, post_norm, attn_gate, pre_hada, d1, d2, b2, d3, d4, w1a…w3b, cond_v, cond_u |
| 217–225 | decoder mHC: 6 FP16 scalars/vectors, then φ_pre, φ_post, φ_res CQ4 `(32,2048)/(32,2048)/(128,2048)`                                                                          |
| 226–227 | `hada_p1`, `hada_p2` FP32 `(512)`, the Hadamard permutations                                                                                                                 |
| 228–235 | engram sites 3 and 7: tables CQ2 `(73728, 128)`, key_proj, value_proj, taps `(4, 512)`                                                                                       |
| 236     | decoder final_norm                                                                                                                                                           |
| 237     | **audio manifest** FP32(13) = `[3246, 1, 8192, 8, 4, 9, 128, 3, 80, 16000, 512, 400, 160]`. The first two values are unknown                                                 |
| 238–317 | 8 × cross-attention: cross_norm, q, k, v, gate, out, q_norm, k_norm, post_cross_norm, cross_gate                                                                             |
| 318     | `pe_gate`                                                                                                                                                                    |
| 319–662 | 8 encoder layers × 43: six norms, q(384), k(96), v(128), gate, out, q_norm, k_norm, attn_gate, pw1 `(1024,512)`, pw2, dw `(9,512)`, hadamard_mlp_0 (13), hadamard_mlp (13)   |
| 663–671 | encoder mHC                                                                                                                                                                  |
| 672     | encoder final_norm                                                                                                                                                           |
| 673–678 | stem: w `(9,128)`, dw_1, pw_1 CQ2, dw_2, pw_2, out CQ2 `(512,1280)`                                                                                                          |
| 679     | mel filterbank FP32 `(257, 80)`                                                                                                                                              |
| 680     | RAW tokenizer blob                                                                                                                                                           |

### Tokenizer

- SentencePiece BPE with 8199 pieces. pad/eos/bos/unk = 0/1/2/3. `add_dummy_prefix=0`, `byte_fallback=1`.
- Ids 4–27 are chat markers, including `<speech>`=18 and `</speech>`=19. Ids 8192–8198 are the language tokens.
- ASR mainly needs **decode**: skip CONTROL/UNK, turn byte pieces into bytes, `▁` becomes a space.
- **Encode** is needed only for keyword biasing; port `RefTokenizer` (greedy best-score pair merge with byte fallback).

### External interface to mirror

- `transcribe` JSON:
  - `{"text", "language", "ttft_ms", "decode_tps"}`, plus `"words": [{"word", "start", "end", "probability"}]` when timestamps are on.
  - Silence returns exactly `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}`.
  - More than 480,000 samples is an error whose message mentions "30 s".
- `stream` JSON: `{"text", "words", "pending", "language", "received", "pass_ms"}`.
- `embed`: float32 rows, one per 80 ms encoder frame.
- WAV conversion as in `whistle.py`: average the channels; 8-bit is unsigned with offset 128;
  `scale = 2^(8·width−1) · channels`; resample to 16 kHz when needed. Upstream uses soxr HQ.

---

## Architecture of the Go port

```
go-whistle/
├── whistle/                 public API: Load(path, opts), (*Model).Transcribe/Stream/Embed, Options, Result JSON types
├── cmd/whistle/             CLI (cobra): transcribe, stream, embed, download, doctor
├── internal/
│   ├── tensor/              copied from go-pocket-tts internal/runtime/tensor (GEMM, dot, axpy + arm64/amd64 asm)
│   ├── ops/                 adapted from go-pocket-tts internal/runtime/ops (attention, RoPE, conv) + depthwise 1-D/2-D
│   ├── safetensors/         copied from go-pocket-tts internal/safetensors
│   ├── cact/                .cact reader, CQ bit unpack, FWHT-128, FP16, tensor-name map
│   ├── weights/             one interface over safetensors and .cact (named lookup, lazy dequant)
│   ├── tokenizer/           SentencePiece pieces from the .cact RAW blob; decode + RefTokenizer encode
│   ├── frontend/            WAV → mono f32 → resample → STFT → mel → log/normalise
│   │                        (uses github.com/cwbudde/algo-dsp dsp/{resample,stft,window} and github.com/cwbudde/algo-fft)
│   ├── needle/              shared blocks: ZCRMSNorm, Attention (taps, q/k norm, gate), HadamardMLP, MHC + Sinkhorn, Engram
│   ├── encoder/             stem + conformer blocks + final_norm
│   ├── decoder/             Needle Block + cross-attention, KV cache, tied head, depth ladder
│   ├── search/              beam search, Aho-Corasick keyword bias, timestamps, silence gate, streaming commit
│   └── testutil/            fixture loading, cosine/top-1 helpers, "skip if models missing"
├── tools/oracle/            //go:build oracle — runs the needle binary / dlopens libneedle3 via purego, dumps fixtures
├── scripts/                 uv-managed Python: JAX block fixtures, safetensors↔cact comparison, WER evaluation
├── testdata/                small fixtures (safetensors/JSON), short audio clips
└── docs/FINDINGS.md         log of every reverse-engineered decision
```

Code copied from go-pocket-tts keeps a header comment naming the source commit, so later fixes can be ported in
either direction.

## Validation strategy: three oracles

1. **JAX reference** (`architecture.py` loaded with the safetensors weights). Gives per-block input/output fixtures for
   everything the decoder shares with Needle 3. Float32 against float32, tolerance about 1e-5 (max abs) or cosine ≥ 0.99999.
2. **safetensors ↔ `.cact`**. Each dequantised `.cact` tensor must match its FP32 twin by cosine. Expect ≥ 0.99 for
   CQ4 and ≥ 0.95 for CQ2; the exact values are recorded. This checks the CQ path **and** the positional name map.
3. **Engine** (`needle` CLI / `libneedle3`):
   - `needle_embed` gives encoder output per frame. Target cosine ≥ 0.99, since the engine runs W2A8.
   - `needle_transcribe` gives the final JSON. Target: transcript equality, the same language and timestamps within one frame.
   - Fixture set: LibriSpeech test-clean/other clips, FLEURS clips for de/fr/es/it/nl/pl, silence, a 30 s+1 clip and a keyword-biasing case.
   - WER on a LibriSpeech test-clean subset, compared with the published 4.31 (test-other 10.49).

Following the upstream porting blog, logits are compared by **cosine and top-1**, not by relative error.

---

## Phase 0 — Bootstrap

- [x] Repo, `go.mod`, `justfile`, `treefmt.toml`, `.golangci.yml`, `AGENTS.md`, `PLAN.md`
- [ ] GitHub Actions: `just ci` on linux/amd64 and macos/arm64. Model-dependent tests skip when `models/` is absent.
- [ ] `internal/hub`: a minimal pure-Go Hugging Face downloader. It resolves `revision`, resumes downloads, verifies
      SHA-256 against a committed `models/manifest.lock.json`, and honours `HF_TOKEN`/`HF_HUB_OFFLINE`. Pin
      `whistle.cact`, `checkpoints/whistle.safetensors` and `config.json` at `b358ddad`.
- [ ] `whistle download` command, with the local layout `models/whistle/{whistle.cact, whistle.safetensors, config.json}`
- [ ] `scripts/fetch-engine.sh`: fetch the platform `needle` binary and `libneedle3` from `Cactus-Compute/needle3` into
      `engine/<platform>/`. Set `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` for every oracle run.
- [ ] Snapshot upstream into `original/needle` at `af2654dffa` and set up `uv` (`jax`, `flax`, `safetensors`,
      `sentencepiece`, `numpy`) for `scripts/`
- [ ] `tools/oracle` (build tag `oracle`): run `needle --model … --audio clip.wav --audio-word-timestamps` and
      `needle_embed`, and write `testdata/oracle/<clip>.{json,embed.safetensors}`.
      **Accept:** fixtures exist for at least 20 English and 2 clips per other language.

## Phase 1 — Weights & tokenizer

- [ ] Copy `internal/safetensors` from go-pocket-tts. Parse `__metadata__` `config` and `run` into a typed `Config`.
- [ ] `internal/cact`: the header (196 B, tag check, all fields), the 28 codebook floats, the 44-byte records, 64-byte
      aligned blobs, and readers for FP32, FP16 and RAW.
- [ ] CQ dequant: the LSB-first bitstream unpack for 2/3/4 bits, the per-group FP16 norm and FWHT-128 (normalised).
      Unit-test it against a tiny tensor packed by a Go port of `export.py` `_cq_pack`.
- [ ] Tensor name map (index → safetensors name). **Accept:** `scripts/compare_cact.py` (or a Go test) reports a cosine
      for all 681 tensors, every tensor passes its threshold, and the map is pinned in a test.
- [ ] `internal/weights`: one lookup interface over both formats, with lazy dequant and an mmap where the platform allows it.
- [ ] `internal/tokenizer`: load the pieces, scores and types from the RAW blob (format per `export.py`). Decode
      (control/unk skipped, byte fallback, `▁`→space) and `RefTokenizer` encode. **Accept:** round-trip on multilingual
      sample text; decode of known id sequences matches `sentencepiece` (fixture from `scripts/`).

## Phase 2 — Shared Needle 3 blocks (float32)

- [ ] Copy `internal/tensor` and `internal/ops` from go-pocket-tts; drop what is not needed (convtranspose, Mimi-specific ops).
- [ ] `ZCRMSNorm`, `rms_unit`
- [ ] `Attention`: Q/K/V projections, optional causal taps, per-head q/k ZCRMSNorm, half-split RoPE (θ = 1e5), GQA
      8:2, sigmoid output gate. Masks: causal and none.
- [ ] `HadamardMLP`: Monarch 16×32 Kronecker apply, conditional rank-8 scale, SiLU, the permutations p1/p2. Read the
      permutations from `.cact` 226/227. For safetensors-only loads, either reuse the `.cact` values or port numpy MT19937 +
      `permutation`, and pin both against each other in a test.
- [ ] `MHC`: lane expand/collapse, pre/post/res heads, a numerically stable log-domain 20-iteration Sinkhorn.
- [ ] `Engram`: uint32 multi-head hashing over the token history, table lookup, key/value projections, dilated
      4-tap conv, sigmoid gating against `rms_unit(x)`.
- [ ] `scripts/needle_fixtures.py`: run each Flax module with real Whistle decoder weights on random inputs and save
      the inputs and outputs. **Accept:** every block matches within 1e-5 (max abs) / cosine ≥ 0.99999.

## Phase 3 — Front end

- [ ] WAV decode exactly as in `whistle.py`: 8/16/24/32-bit, channel averaging, 8-bit offset 128.
      **Accept:** the upstream matrix (16000/1/2, 8000/1/2, 44100/2/2, 16000/1/1, 22050/2/3, 48000/1/4). A 220 Hz tone
      at 0.3 gives 8000±1 samples, a peak of 0.28–0.31 and 108–111 zero crossings.
- [ ] Resampling with `algo-dsp/dsp/resample` (windowed-sinc / polyphase). Measure the deviation from soxr HQ on test
      tones and document it.
- [ ] STFT: 400-sample window zero-padded to 512, hop 160 → 257 bins → power → mel from `.cact` tensor 679.
- [ ] Hypothesis set for what the docs leave open: window (Hann periodic or symmetric), centre padding (reflect, zero
      or none), power or magnitude, log flavour (`ln`, `log10`, `log(x + eps)` or `clamp` à la Whisper),
      normalisation (per mel channel over time, global, or none) and the "250–3500 Hz band-limit". Resolve them in Phase 4
      against the encoder oracle, because no mel-level oracle exists.

## Phase 4 — Encoder (reverse engineering)

- [ ] Stem: a 2-D conv with candidate strides (2,2) on all three convs vs. stride only on the depthwise stages; padding
      SAME or VALID; activation (SiLU, GELU or none) after each stage. Pin the frame count: 3000 → 375.
- [ ] Encoder block candidates:
  - FFN residual scale 1 or ½ (macaron)
  - conv module: GLU over pw1's 1024 channels or not; SiLU after dw; any normalisation between dw and pw2
  - attention: full non-causal; RoPE in the encoder or not
  - where `pe_gate` applies: a sinusoidal or RoPE mix-in gated by σ(pe_gate), or on the stem output
  - how mHC wraps the block (same as the decoder, `enc_lanes` 4)
- [ ] Hypothesis harness `internal/encoder/probe_test.go` (build tag `oracle`): enumerate the combinations, run the
      encoder on the fixture clips and score each by the mean per-frame cosine against `needle_embed`. Prune greedily by stage.
- [ ] Confirm what `needle_embed` actually returns: the shape, and whether it is before or after `final_norm` and the lane mean.
- [ ] **Accept:** mean per-frame cosine ≥ 0.99 against the engine on all fixture clips. Every resolved unknown is
      written to `docs/FINDINGS.md` with its metrics.

## Phase 5 — Decoder + greedy decoding

- [ ] Cross-attention: per-layer `cross_norm → q`, encoder `k/v` projected once per clip and cached, q/k norm,
      sigmoid gate, `post_cross_norm`, σ(`cross_gate`) residual. Decide where it sits in the block (after
      self-attention is the most likely) by fixture accuracy.
- [ ] KV cache for self-attention, plus a rolling history for the Q/K/V taps and the engram n-grams.
- [ ] Tied output head (logits = x · Eᵀ) and `sqrt(d)` input scaling.
- [ ] Prompt-format candidates: `[bos]`, `[bos, <speech>]`, `[bos, <|lang|>]`, … Pick the one where greedy decoding
      reproduces the engine's language token and the first words.
- [ ] Language detection: argmax (or softmax over 8192–8198) at the language step; a fixed language forces the token.
- [ ] **Accept:** greedy transcripts are sensible on all fixtures and the language matches the engine on 100% of them.

## Phase 6 — Engine-parity decoding

- [ ] Beam search: 5 beams, with candidate length normalisations (`logp / len`, `logp / len^α`, GNMT); EOS handling;
      max 320 tokens; encoder K/V and cross-attention keys shared across beams; copy-on-write KV per beam.
- [ ] Keyword biasing: tokenise the keywords, build an Aho-Corasick automaton over token ids, add a bonus as a match
      advances and revoke it on failure. Tune the weight against the engine using fixture clips with and without keywords.
- [ ] Silence gate: a loudness-range measure (e.g. a percentile spread of frame RMS in dB). Pick the threshold from
      oracle sweeps (silence, very quiet speech, noise). Return the exact empty JSON for silence.
- [ ] The 30 s limit: an error whose message contains "30 s" for more than 480,000 samples.
- [ ] `Depth` option: run the decoder layers given by the bisection order 0, 7, 3, 5, 1, 2, 4, 6, truncated to N.
      Port `_ladder_layer_order` and test that it matches the engine's `--audio-depth`.
- [ ] **Accept:** transcripts equal the engine's on ≥ 95% of fixtures (with the rest explained in FINDINGS), and
      LibriSpeech test-clean WER is within 0.3 absolute of the engine on the same subset.

## Phase 7 — Word timestamps & output JSON

- [ ] Word grouping: tokens starting with `▁` begin words; byte pieces are merged.
- [ ] Timestamps from cross-attention: per token, aggregate the attention over heads and layers (candidates: all
      layers, the last layer, the best-aligned heads) and run a monotonic DTW alignment over the 80 ms frames.
- [ ] `probability` per word: candidates are the mean or the product of the token probabilities. Match the engine.
- [ ] `ttft_ms` and `decode_tps` measured the same way as the engine (TTFT includes the front end and the encoder).
- [ ] **Accept:** timestamps are monotone and within [0, duration], start/end are within 80 ms of the engine on ≥ 90% of
      words, and the JSON is byte-compatible in field names and order.

## Phase 8 — Streaming & embeddings

- [ ] `Stream`: buffer the incoming chunks; every ~1 s re-transcribe the buffer; commit the longest word prefix on
      which two consecutive passes agree; report `pending`, `received` and `pass_ms`; `Stop()` flushes. Support
      buffers longer than 30 s by trimming committed audio (match the engine's behaviour).
- [ ] `Embed`: the encoder output per 80 ms frame, the same layout as `needle_embed`.
- [ ] **Accept:** on fixture clips fed in 1 s chunks, the final committed text equals the offline transcript, and
      the embeddings match the engine at cosine ≥ 0.99.

## Phase 9 — Native quantised path & performance

- [ ] Run straight from the packed `.cact`: a 2/4-bit LUT GEMV in the rotated (Hadamard) domain. Apply FWHT-128 to the
      activation groups once and multiply by the codebook values, instead of dequantising the weights. Memory target:
      under 64 MB RSS for the model.
- [ ] Optional int8 activations and an int8 KV cache (`act_bits 8`, `kv_bits 8`) to mirror the engine's numerics, behind a flag.
- [ ] Plan 9 assembly (NEON/AVX2) for the LUT GEMV, the FWHT, the Monarch Kronecker products and the softmax/Sinkhorn
      hot loops, with pure-Go fallbacks and equivalence tests (see `go-plan9-assembly`).
- [ ] Parallelism: the encoder over time blocks and heads; the decoder over beams.
- [ ] Benchmarks (`bench/`): TTFT and decode tok/s against the engine on the same machine. The engine publishes
      11.1 ms TTFT and 1319 tok/s on an M4 Pro with 5 beams. Also the real-time factor for 30 s clips.
- [ ] **Accept:** WER unchanged within 0.1 against the float path, with numbers recorded in `bench/RESULTS.md`.

## Phase 10 — Distribution & extras

- [ ] CLI polish: `whistle transcribe [--language] [--keywords] [--word-timestamps] [--depth] [--json]`,
      `whistle stream` (stdin PCM or a WAV played in chunks), `whistle embed`, `whistle doctor`
- [ ] README quickstart, examples and Go API docs
- [ ] Optional: an HTTP server (`/transcribe`, `/stream` over WebSocket)
- [ ] Optional: a js/wasm build (as in go-pocket-tts) with a small web demo
- [ ] Optional: the Needle 3 text model (tool calling) on top of `internal/needle` and `.cact`, which reuses most of the work
- [ ] Optional: extract `internal/tensor`/`ops` into a shared module once go-whistle and go-pocket-tts have stabilised

---

## Unknowns to resolve against the oracle

| #   | Unknown                                                                                     | Phase | Resolved by                       |
| --- | ------------------------------------------------------------------------------------------- | ----- | --------------------------------- |
| U1  | Mel log flavour, floor, normalisation; the "250–3500 Hz" claim vs. the full-band filterbank | 3/4   | encoder cosine vs. `needle_embed` |
| U2  | Stem strides, padding, activations                                                          | 4     | frame count + encoder cosine      |
| U3  | Encoder conv module (GLU?), macaron scale, `pe_gate` usage, positions                       | 4     | encoder cosine                    |
| U4  | Meaning of the audio manifest values `3246, 1`                                              | 4/5   | inspection + experiments          |
| U5  | Whether the encoder and decoder share the Hadamard permutations                             | 2/4   | block parity / encoder cosine     |
| U6  | Cross-attention position in the block                                                       | 5     | greedy transcript parity          |
| U7  | Decoder prompt format, language detection                                                   | 5     | language + first-token parity     |
| U8  | Beam length normalisation, EOS rules                                                        | 6     | transcript parity                 |
| U9  | Keyword bias weight and revocation                                                          | 6     | keyword fixture parity            |
| U10 | Silence gate measure and threshold                                                          | 6     | threshold sweep                   |
| U11 | Timestamp algorithm (heads/layers, DTW), word probability                                   | 7     | timestamp parity                  |
| U12 | Streaming commit rule details, behaviour beyond 30 s                                        | 8     | stream parity                     |

## Non-goals

- Training, fine-tuning, quantisation-aware training (the JAX side of upstream)
- Telemetry (upstream's is on by default; this port sends nothing)
- Live microphone capture in the core module, because it needs cgo. Streaming takes PCM chunks; a separate optional
  module can add audio capture later.
- Bit-exact equality with the engine

## Risks

- **The oracle is a moving, closed target.** Pin the engine version, keep the fixtures in the repo, and never fetch
  "latest" in tests.
- **Unknowns might not be identifiable from the encoder output alone**, e.g. the mel normalisation interacts with the
  stem. Mitigation: staged probing, and if needed per-layer probes via the engine's `needle_complete`/`embed` paths.
- **The `.cact` tensor map is derived, not documented.** Name-map tests against safetensors catch drift.
- **Performance without cgo.** A 2-bit LUT GEMV in pure Go can still be a few times slower than the engine. Assembly
  in Phase 9 narrows the gap, but parity with the engine's speed is not a hard goal.
