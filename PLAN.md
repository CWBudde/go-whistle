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

|                 | Where                                   | Pin                                                           |
| --------------- | --------------------------------------- | ------------------------------------------------------------- |
| Python / JAX    | `github.com/cactus-compute/needle`      | `af2654dffa` (2026-10-07), package `cactus-needle` 3.1.0      |
| Weights         | `huggingface.co/Cactus-Compute/whistle` | `b358ddad`, `weights_version` 2.0.0                           |
| Engine (oracle) | `huggingface.co/Cactus-Compute/needle3` | engine 3.2.0 (`ENGINE_VERSIONS` in `needle/agent/fetch.py`)   |
| Text model      | `huggingface.co/Cactus-Compute/needle3` | `93e88ede` (`needle3.cact`, `needle3.safetensors`, tokenizer) |

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

Whistle tensor map, verified per tensor in Phase 1 (`cact.WhistleNames()`, `docs/FINDINGS.md` F-006):

| Index   | Contents                                                                                                                                                                     |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0       | embedding CQ4 `(8199, 512)`                                                                                                                                                  |
| 1–216   | 8 decoder layers × 27, in Needle order: norm_in, q, k, v, q/k/v taps, q_norm, k_norm, gate, out, post_norm, attn_gate, pre_hada, d1, d2, b2, d3, d4, w1a…w3b, cond_v, cond_u |
| 217–225 | decoder mHC: 6 FP16 scalars/vectors, then φ_pre, φ_post, φ_res CQ4 `(32,2048)/(32,2048)/(128,2048)`                                                                          |
| 226–227 | `hada_p1`, `hada_p2` FP32 `(512)`, the Hadamard permutations                                                                                                                 |
| 228–235 | engram sites 3 and 7: tables CQ2 `(73728, 128)`, key_proj, value_proj, taps `(4, 512)`                                                                                       |
| 236     | decoder final_norm                                                                                                                                                           |
| 237     | **audio manifest** FP32(13) = `[3246, 1, 8192, 8, 4, 9, 128, 3, 80, 16000, 512, 400, 160]`: magic 0xCAE, version 1, then config                                              |
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
├── cmd/whistle/             CLI (cobra): transcribe, stream, embed, serve, download, doctor
├── cmd/whistle-wasm/        js/wasm build for the web demo in web/
├── internal/
│   ├── hub/                 pure-Go Hugging Face downloader for the pinned weights
│   ├── safetensors/         copied from go-pocket-tts internal/safetensors
│   ├── cact/                .cact reader, CQ bit unpack, FWHT-128, FP16, tensor-name map
│   ├── weights/             one interface over safetensors and .cact (named lookup, lazy dequant)
│   ├── tokenizer/           SentencePiece pieces from the .cact RAW blob; decode + RefTokenizer encode
│   ├── frontend/            WAV → mono f32 → resample → STFT → mel → log/normalise
│   │                        (uses github.com/cwbudde/algo-dsp dsp/{resample,stft,window} and github.com/cwbudde/algo-fft)
│   ├── needle/              shared blocks: ZCRMSNorm, Attention (taps, q/k norm, gate), HadamardMLP, MHC + Sinkhorn, Engram
│   ├── encoder/             stem + conformer blocks + final_norm
│   ├── decoder/             Needle Block + cross-attention, KV cache, tied head, depth ladder
│   ├── search/              beam search (copy-on-write KV per beam), Aho-Corasick keyword bias
│   ├── align/               word grouping, word probability, cross-attention DTW timestamps
│   ├── cq/                  CQ LUT GEMV in the rotated (Hadamard) domain, straight from the packed .cact
│   ├── eval/                WER/CER and edit distance shared by the parity tests and the LibriSpeech harness
│   ├── silence/             the engine's silence gate
│   ├── pcm/                 raw PCM for `whistle stream` from stdin
│   ├── server/              `whistle serve`: HTTP and WebSocket
│   ├── spm/                 pure-Go sentencepiece for the text model
│   ├── text/                the Needle 3 text model: source over both formats, forward pass, state
│   └── toolcall/            the text model's prompt: rendering, static prefix, context budget
├── tools/oracle/            //go:build oracle — runs the needle binary / dlopens libneedle3 via purego, dumps fixtures
│   └── text/                the same for the text model
├── scripts/                 uv-managed Python: JAX block fixtures, safetensors↔cact comparison, WER evaluation
├── bench/                   `just bench` and the results against the engine
├── web/                     the browser demo
├── testdata/                small fixtures (safetensors/JSON), short audio clips
└── docs/FINDINGS.md         log of every reverse-engineered decision
```

The tensor kernels (GEMM, dot, axpy, attention, RoPE, conv) live in `github.com/cwbudde/algo-tensor`, extracted from
go-pocket-tts and this repo.

Code copied from go-pocket-tts keeps a header comment naming the source commit, so later fixes can be ported in
either direction.

## Validation strategy: three oracles

1. **JAX reference** (`architecture.py` loaded with the safetensors weights). Gives per-block input/output fixtures for
   everything the decoder shares with Needle 3. Float32 against float32, tolerance about 1e-5 (max abs) or cosine ≥ 0.99999.
2. **safetensors ↔ `.cact`**. Each dequantised `.cact` tensor must match its FP32 twin by cosine. Measured: CQ4
   0.9943–0.9954, CQ2 0.9319–0.9401 (F-006; the earlier "≥ 0.95 for CQ2" guess was wrong). This checks the CQ path **and** the positional name map.
3. **Engine** (`needle` CLI / `libneedle3`):
   - `needle_embed` gives encoder output per frame. Target cosine ≥ 0.99, since the engine runs W2A8.
   - `needle_transcribe` gives the final JSON. Target: transcript equality, the same language and timestamps within one frame.
   - Fixture set: LibriSpeech test-clean/other clips, FLEURS clips for de/fr/es/it/nl/pl, silence, a 30 s+1 clip and a keyword-biasing case.
   - WER on a LibriSpeech test-clean subset, compared with the published 4.31 (test-other 10.49).

Following the upstream porting blog, logits are compared by **cosine and top-1**, not by relative error.

---

Finished work is kept short: what exists, where, and the finding that holds the evidence. The measurements, the
alternatives tried and the dated history are in `docs/FINDINGS.md` and the git history (the full plan as of
`1628347`).

## Phase 0 — Bootstrap — ✅ DONE (2026-10-09)

- [x] Repo, `go.mod`, `justfile`, `treefmt.toml`, `.golangci.yml`, `AGENTS.md`, `PLAN.md`; CI (`just ci`) on
      ubuntu-latest only.
- [x] `internal/hub`: a pure-Go Hugging Face downloader (pinned revision, resume, SHA-256 against the embedded
      `internal/hub/manifest.lock.json`, `HF_TOKEN`, `HF_HUB_OFFLINE`, `HF_ENDPOINT`).
- [x] `whistle download` into `models/whistle/{whistle.cact, whistle.safetensors, config.json}`.
- [x] `just fetch-engine` (`scripts/fetch-engine.sh`): `needle` and `libneedle3` (from the 3.2.0 wheel) into
      `engine/<platform>/`, every file pinned by SHA-256.
- [x] `just fetch-upstream` (`original/needle` at the pin) and the `uv` project in `scripts/`.
- [x] `tools/oracle`: `just oracle-clips` (32 clips in `testdata/audio`: 20 LibriSpeech, 2 each of de/fr/es/it/nl/pl)
      and `just oracle-run` (engine JSON and `needle_embed` in `testdata/oracle/`) (F-001, F-002).

## Phase 1 — Weights & tokenizer — ✅ DONE (2026-10-09)

- [x] `internal/safetensors` (from go-pocket-tts `9faba38`) with a typed, strict `Config` and `Run`.
- [x] `internal/cact`: header, codebooks, records, FP32/FP16/RAW readers (F-003).
- [x] CQ dequantisation (`File.CQ`); the header's codebooks equal upstream's, so numpy's RNG need not be ported (F-004).
- [x] Tensor name map `cact.WhistleNames()` (golden `internal/cact/testdata/whistle_names.txt`): all 681 records
      checked against safetensors (F-006).
- [x] `internal/weights`: one `Source` over `.cact` and safetensors, lazy cached dequantisation, mmap with an
      `os.ReadFile` fallback (F-007).
- [x] `internal/tokenizer`: the pieces from the RAW blob, sentencepiece's decode, `RefTokenizer`'s encode (F-005).

## Phase 2 — Shared Needle 3 blocks (float32) — ✅ DONE (2026-10-10)

- [x] `internal/tensor` and `internal/ops` from go-pocket-tts, since moved to `github.com/cwbudde/algo-tensor`
      (Phase 10) (F-008).
- [x] `internal/needle`: `ZCRMSNorm`/`RMSUnit`, `Attention` (taps, q/k norm, half-split RoPE, GQA, gate),
      `HadamardMLP` (Monarch; permutations from `.cact` or the numpy MT19937 port), `MHC` (20 Sinkhorn iterations
      are part of the model), `Engram` (F-011, F-013).
- [x] **Accept** met: `just needle-fixtures` (`testdata/needle/`); every block within 1e-5 · max(1, max |ref|) and
      cosine ≥ 0.99999 of float32 JAX, worst 0.41 of the bound (F-009).

## Phase 3 — Front end — ✅ DONE (2026-10-10)

- [x] WAV decoding as `whistle.py` does it (`frontend.ParseWAV`, `DecodeWAV`), the upstream WAV matrix (F-010).
- [x] Resampling with algo-dsp, measured against soxr HQ (`testdata/frontend/soxr_hq.safetensors`) (F-010).
- [x] STFT and mel from `.cact` tensor 679 (`frontend.New`, `LogMel`), against a direct DFT and numpy (F-012).
- [x] The front-end hypothesis set (`frontend.Options`), resolved in Phase 4 (F-012).

## Phase 4 — Encoder (reverse engineering) — ✅ DONE (2026-10-10)

- [x] Stem: stride 2 on all three convs, zero padding 1, SiLU after `stem/w` and the pointwise convs (F-016).
- [x] Encoder block: macaron FFN ½, GLU a·σ(b), SiLU after `dw`, `conv_out_norm`, non-causal attention with RoPE 1e5,
      mHC as in the decoder, shared permutations; `pe_gate` is unused (F-017, F-019).
- [x] Hypothesis harness `just encoder-probe` (tag `oracle`); fast regression `TestBestMatchesEngine` (F-014).
- [x] `needle_embed` returns `enc.final_norm` of the lane mean, `[frames, 512]` (F-015).
- [x] **Accept** met: `encoder.Best()` on the `.cact` weights, mean per-frame cosine 0.999828 against `needle_embed`,
      every clip ≥ 0.999033. Front end: power spectrum, a soft log knee 76 dB below the peak, per-channel norm, full
      band (F-018, F-020).

## Phase 5 — Decoder + greedy decoding — ✅ DONE (2026-10-10)

- [x] Cross-attention between the self-attention and the MLP, inside the mHC block (F-021).
- [x] KV cache and rolling tap/engram histories (`needle.KVCache`, `needle.EngramCache`, `decoder.State`) (F-022).
- [x] Tied head, √512 input scale, `stack/final_norm` of the lane mean (F-023).
- [x] Prompt `[<s>]`, then `<|lang|>`, the text, `</s>` (F-023).
- [x] Language by argmax over the 7 language tokens, or forced (`decoder.Greedy`) (F-024).
- [x] **Accept** met: language 32/32, greedy text equal to the engine's on 22/32 (`TestGreedyMatchesEngine`).

## Phase 6 — Engine-parity decoding — ✅ DONE (2026-10-10)

### 6.1 Decoding oracle corpus (`tools/oracle`, `testdata/oracle/`, `justfile`)

- [x] `just oracle-depth`, `oracle-keywords` (53 cases), `oracle-silence` (320 signals), `oracle-limits` and
      `oracle-librispeech` (200 test-clean utterances, audio git-ignored in `models/corpus/`, engine WER 5.30 %)
      (F-025).
- [x] **Accept** met: every recipe re-runs without a diff; 1.37 MB.

### 6.2 Beam search (`internal/search`, `internal/decoder`, `internal/needle`)

- [x] `decoder.State.Fork`: copy-on-write KV per beam, shared encoder memory (F-026).
- [x] `search.Beam` with `search.Default()`: Whisper's search, 5 beams, Σ log p / text tokens, stop at 5 finished,
      language by argmax (`just search-probe`) (F-026).
- [x] `internal/eval`: edit distance, WER, CER.
- [x] **Accept** met: 31/32 transcripts equal to the engine's; the miss, `fleurs-pl_pl-10047713455444904767`, is a
      near-tie (`TestBeamMatchesEngine`).

### 6.3 Keyword biasing (`internal/search`, `internal/tokenizer`; needs 6.1)

- [x] `search.NewKeywords`: an Aho-Corasick bonus over token ids, revoked when a match fails (`just keywords-probe`);
      `DefaultBias` was re-fitted in Phase 10 (F-029, F-044).
- [x] **Accept** (reworded by the user's decision) met: 46/53 keyword cases at the time, 47/53 now, every mismatch
      explained (`TestKeywordsMatchEngine`).

### 6.4 Depth ladder (`internal/decoder`; needs 6.1)

- [x] `decoder.WithDepth(n)` over the bisection order 0, 7, 3, 5, 1, 2, 4, 6, lanes by position (F-030).
- [x] **Accept** (reworded by the user's decision) met: 161/192 depth transcripts equal, the other 31 within
      0.0288 nats per text token (`TestDepthMatchesEngine`).

### 6.5 Public pipeline & limits (`whistle/`, `cmd/whistle`)

- [x] `whistle.Load`, `Transcribe`, `TranscribeWAV`, `Embed` (F-027).
- [x] The 30 s limit: `ErrTooLong`, the engine's text `audio limit is 30 s`.
- [x] `Result` JSON byte-compatible with the engine; `whistle.Silence()`.
- [x] `ttft_ms` and `decode_tps` on the engine's clock (F-027).
- [x] `whistle transcribe <wav> [--language] [--json]`.
- [x] **Accept** met: `Transcribe` equals `decoder.Greedy`; engine JSON round-trips (`TestResultRoundTripsEngineOutput`).

### 6.6 Silence gate & integration (`whistle/`; needs 6.1, 6.2, 6.5)

- [x] `internal/silence`: p99 − p10 of the 250–3500 Hz mel frame energy ≤ 5 dB, all 340 sweep verdicts as the
      engine's (F-031).
- [x] Beam search, keywords and depth wired into `Transcribe` (F-031).
- [x] `just eval-librispeech` (`TestEvalLibriSpeech`).
- [x] **Accept** met: 31/32 fixtures equal (`TestTranscribeMatchesEngine`), LibriSpeech WER 5.35 % against the
      engine's 5.30 % (F-031).

## Phase 7 — Word timestamps & output JSON — ✅ DONE (2026-10-10)

- [x] 7.1 Cross-attention capture (`decoder.State.SetCapture`), one teacher-forced pass over the winning text (F-033).
- [x] 7.2 `align.Words`: the engine's words on 477/477 transcripts; probability is Whisper's mean token p (F-033).
- [x] 7.3 Timestamps: Whisper's `find_alignment` over the last half of the layers, DTW, an energy onset for the first
      start (F-033).
- [x] 7.4 `TranscribeOptions.WordTimestamps`, `whistle transcribe --word-timestamps`.
- [x] **Accept** met: start and end within 80 ms on 97.4 % of the engine's words, JSON byte-compatible
      (`TestTranscribeWordsMatchEngine`, `TestTranscribeWordsJSON`).

## Phase 8 — Streaming & embeddings — ✅ DONE (2026-10-10)

- [x] 8.1 `just oracle-stream`: 108 runs, 822 calls in `testdata/oracle/stream/` (F-032).
- [x] 8.2 `Embed` in `needle_embed`'s layout, every clip ≥ 0.999033 (F-027).
- [x] 8.3 `NewStream`, `Process`, `Stop`: LocalAgreement-2, the 15 s sentence trim, the engine's stream JSON;
      `just stream-parity` 797/930 calls equal (F-036).
- [x] 8.4 Buffers past 30 s: the engine's sentence trim. With nothing to trim the engine most likely passes the whole
      buffer; the port deliberately feeds only the last 30 s, so `long-silence-lead`'s calls at 34 and 36 s still differ
      (F-036).
- [x] **Accept** (reworded by the user's decision) met: at 1 s chunks the final text equals the offline transcript on
      18/32 clips, as often as the engine's own stream (`TestStreamAccept`).

## Phase 9 — Native quantised path & performance — ✅ DONE (2026-10-11)

- [x] 9.1 `internal/cq`: the CQ LUT GEMV/GEMM in the Hadamard domain, `cact.File.CQPacked` (F-028).
- [x] 9.2 The packed `.cact` path (`weights.Packer`, `needle.Linear`) is `whistle.Load`'s default
      (`WithFloatWeights()` keeps float); the weights take 34.1 MiB (target reworded by the user's decision)
      (F-037, F-038).
- [x] 9.3 `whistle.WithInt8()`, off by default: int8 where the engine quantises, int8 KV and memory (F-041).
- [x] 9.4 NEON/AVX2 assembly for the LUT kernels, the FWHT and the softmax exp, each with a pure-Go fallback and an
      equivalence test; the Monarch products run on algo-tensor's GEMM, NEON on arm64 and Go on amd64 (12.3)
      (F-034, F-035).
- [x] 9.5 `tensor.Parallel`: the encoder over time blocks and heads, the decoder over beams, bit-identical for any
      worker count (F-038).
- [x] 9.6 `just bench` and `just oracle-bench` (`bench/RESULTS.md`): TTFT 18.8 against the engine's 6.9 ms, decode
      1085 against 1499 tok/s (F-040).
- [x] **Accept** met: packed and float WER 5.35 % each, all 200 LibriSpeech transcripts equal (F-040).

## Phase 10 — Distribution & extras

- [x] CLI: `transcribe` (several files, keywords, word timestamps, depth, JSON), `stream` (a WAV or stdin PCM),
      `embed` (JSON or `.npy`), `doctor`.
- [x] README quickstart, `whistle/doc.go`, `whistle/example_test.go`.
- [x] Follow-up from 6.3, `de-paare-missed`: closed by the user's decision. `DefaultBias` is `w=2.15 complete=0`
      with the case spellings and `spent`; the engine's own rule was not found (F-042, F-044).
- [x] `whistle serve` (`internal/server`): `POST /transcribe`, a `GET /stream` WebSocket, `/healthz`, `/v1/info`.
- [x] js/wasm: `cmd/whistle-wasm`, the `web/` demo, `just wasm`, `just wasm-test` (F-043).
- [ ] Optional: the Needle 3 text model (tool calling) on top of `internal/needle` and `.cact`, which reuses most of the work
      (2026-10-11) — moved to Phase 11, which plans it in sub-phases 11.1–11.9; 11.1 is done, 11.2 in part.
- [x] `internal/tensor` and `internal/ops` extracted into `github.com/cwbudde/algo-tensor`, which go-pocket-tts uses
      too.

## Phase 11 — Needle 3 text model (tool calling)

The Needle 3 text model is the 121M-parameter tool-calling model Whistle's decoder was cut from: the same `Block`,
`Stack`, mHC, engram and HadamardMLP, at width 768 and 20 layers, without cross-attention. Unlike Whistle's encoder,
its whole model side is open upstream: `architecture.py` (`SimpleAttentionNetwork`, the probe heads, the ladder),
`export.py` (the `.cact` layout including the heads), `quantize.py` (CQ, the int8 fake quantisation),
`tokenizer.py` (sentencepiece) and `finetune.py` (the prompt format). Those parts are a **translation** checked
against JAX. What only the engine can tell: the decoding (the byte-level grammar the engine compiles from the
schemas, greedy or beam, the stop rules), the response envelope and its post-processing (confidence, suppression,
grounding, `triggers`, tool retrieval with more than 5 tools), multi-turn prompts, and the W2A8 numerics with the
engine's KV window. Those are fitted against the engine, as Phases 4–8 did for Whistle.

### Text model reference card

Pins: HF `Cactus-Compute/needle3` @ `93e88ede` (`needle3.cact`, `checkpoints/needle3.safetensors`, `config.json`,
`tokenizer/tokenizer.model`; `internal/hub/needle3.lock.json`). `config.json` names engine 3.1.0; the local oracle is
engine 3.2.0.

Config (`needle3.safetensors` → `__metadata__["config"]`, upstream's `TransformerConfig` names; the `.cact` header
states the same geometry):

```json
{
  "vocab_size": 8192,
  "d_model": 768,
  "num_heads": 12,
  "num_kv_heads": 2,
  "num_layers": 20,
  "qk_head_dim": 48,
  "v_head_dim": 64,
  "max_seq_len": 8192,
  "dtype": "bfloat16",
  "engram_slots": 18432,
  "engram_layers": [3, 7, 11, 15, 19],
  "sliding_window": 1024,
  "global_layers": [4, 9, 14, 19],
  "kv_window": 256,
  "mhc_lanes": 4,
  "qkv_conv_taps": 3,
  "out_vocab": 8192,
  "kv_bits": 8,
  "act_bits": 8,
  "weight_bits": "embedding=4,stack/mhc=4,default=2",
  "confidence_probes": 4,
  "confidence_queries": 4
}
```

- Left at `TransformerConfig` defaults: `engram_orders` (2, 3), `engram_heads` 0, so 3 hashes per order and 6 tables
  of 18432 × 128 per site; `rope_theta` 1e5; no `ladder_*` (the ladder is the default bisection order). The `image_*`
  keys are 0. The run metadata: `stage` qapt from `needle3_rl.pkl`, 10,000 steps, bits annealed 4 → 2,
  `tokenizer_hash` = MD5 of `tokenizer.model`, a merged `confidence_head`.
- 121,019,861 F16 parameters in 64 tensors (Whistle: F32). `config.json`'s 121,021,910 also counts the derived
  `.cact` records `hada_p1`, `hada_p2` (1024 each) and `heads.manifest` (1).
- 12 query heads (`q_proj` 768 → 576), GQA 12:2, qk 48, v 64. HadamardMLP width 1024 = 32 × 32: the 768-wide input is
  zero-padded to 1024 and the output cut back (Whistle: 512 = 16 × 32, no padding).
- Layers outside `global_layers` attend within a sliding window of 1024 positions. `kv_window` 256 is the window the
  engine's KV cache was post-trained for (export.py); the float model does not use it.
- Probe heads: only the `confidence_head` (code 2; `ConfidenceHead`, out 1, probes over the embedding and all 20
  blocks). No `embedding_head`, no `router_head`.

Pipeline, prompt to JSON:

1. **Prompt** (`finetune.render_example`, `run.py`): `<|im_start|>system\n{facts}<|im_end|>\n` (only with facts;
   `Needle` prefixes `date: YYYY-MM-DD Day HH:MM` unless one is given) `<|im_start|>user\n<tools>{tools JSON,
compact separators, ensure_ascii=False}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n`; ids are
   `[BOS] + sentencepiece.encode(prompt)`.
2. **Tokenizer**: sentencepiece BPE, 8192 pieces, identity normaliser, byte fallback, 24 user-defined markers (ids
   4–27, the chat markers plus `<image>`, `<speech>`, `<tts>`, `<imagen>` pairs); pad/eos/bos/unk 0/1/2/3. The
   `.cact` RAW record holds the same pieces.
3. **Forward**: `embedding · √768` → 20 mHC-wrapped blocks (engram at 3, 7, 11, 15, 19) → lane mean → `final_norm` →
   tied head over the first `out_vocab` rows.
4. **Decoding** (engine only): an optional `<think>\n…\n</think>\n` reasoning, then
   `<tool_call>[{"name":…,"arguments":{…}}]</tool_call><|im_end|>`, constrained by "a byte-level grammar compiled
   from your schemas" (`llms.txt`).
5. **Envelope** (engine only): `{"type", "success", "error", "error_code", "reason", "function_calls",
"suppressed_calls", "reasoning", "confidence", "prefill_tps", "decode_tps", "peak_ram_mb", "validation":
{"ungrounded", "negation"}}`. `confidence` is "the min of a post-hoc calibration head and the decode probability
   of the call"; calls below 0.1 move to `suppressed_calls`.

`.cact` map (`cact.TextNames`, golden `internal/cact/testdata/needle3_names.txt`), 581 records: 0 embedding CQ4
`(8192, 768)`; 1–540 20 layers × 27 in Whistle's decoder order; 541–549 mHC (φ CQ4); 550–551 `hada_p1/p2` FP32
`(1024)`; 552–571 5 engram sites (tables CQ2 `(110592, 128)`); 572 `final_norm`; 573 `heads.manifest` FP16 `[2]`;
574–579 the confidence head (probes CQ4 `(84, 768)`, gain, query CQ4, row_bias, proj CQ4 `(1, 3072)`, bias); 580 the
RAW tokenizer.

### 11.1 Weights & tokenizer (`internal/hub`, `internal/safetensors`, `internal/cact`, `internal/text`, `internal/spm`; no prerequisites) — ✅ DONE (2026-10-11)

- [x] Pin `Cactus-Compute/needle3` @ `93e88ede` (`internal/hub/needle3.lock.json`); `just fetch-needle3` into
      `models/needle3/`.
- [x] `Store.TextConfig()` and `TextRun()` for `needle3.safetensors` (F-045).
- [x] `cact.TextNames` and `TextNamesFor` (golden `internal/cact/testdata/needle3_names.txt`) (F-045).
- [x] `text.Source` over both formats under the `.cact` names (F-045).
- [x] `internal/spm`: pure-Go sentencepiece, equal to Python's on `testdata/text/tokenizer/needle3.jsonl` (F-046).
- [x] **Accept** met: all 581 records by the F-006 checks, the same geometry and names from both formats, the
      tokenizer equal to sentencepiece.

### 11.2 Float forward pass (`internal/text`, `internal/needle` additive; needs 11.1) — ✅ DONE (2026-10-12)

- [x] HadamardMLP below its power of two (768 in 1024); Whistle's outputs unchanged.
- [x] `text.Load`, `Model.Forward`; JAX fixtures from `just text-forward-fixtures` (`testdata/text/forward/`) (F-045).
- [x] The banded mask of the local layers (`Attention.SetWindow`), the fixture `long_tools` past 1024 positions
      (F-045).
- [x] Incremental decoding: `text.State`, `Model.Extend`, `State.Fork`.
- [x] **Accept** met: final hidden states cosine ≥ 0.999999974 per position and top-1 equal at every position,
      `long_tools` included (`TestForwardMatchesJAX`, `TestExtendMatchesForward`).

### 11.3 Prompt format (`internal/text` or a new `internal/toolcall`; needs 11.1)

A translation of `render_example` and `Needle.__init__`, then fitted where the engine renders more than upstream
shows (multi-turn, tool results, retrieval).

- [x] Render as `render_example` (`toolcall.Render`, `Encode`, `ToolsJSON`, `WithDateFact`), equal to Python on
      `testdata/text/prompt/render.json` (F-048).
- [x] `toolcall.StaticPrefix`, `[BOS]` and the rendering up to `</tools>`, has `needle_init`'s count on every engine
      case (`TestStaticPrefixMatchesEngine`) (F-048).
- [ ] Multi-turn (`complete` after `complete`, `run`'s tool results, `<tool_result>`), `reset`, and the engine's turn
      limit and trimming (`--fail-input-overflow`).
      Note (2026-10-12): upstream's porting notes (cactuscompute.com/blog/porting-needle) give the wire format to
      check: "The first user turn continues the open `user` message; later turns open their own", a later turn as
      `\n…<|im_end|>\n<|im_start|>assistant\n` and a tool result as
      `\n<|im_start|>tool\n<tool_result>{…}</tool_result><|im_end|>\n<|im_start|>assistant\n`; the Python docs add
      that a turn after a call is a tool result only when it parses as JSON (F-049).
      (2026-10-12) — partial: the turns the engine adds are pinned. `just oracle-text turns` (`turns.json`, 53
      conversations over `needle --serve --fail-input-overflow`, byte-identical on a re-run) reads each probe's length
      and room from the refusal. `toolcall.Conversation` renders the first turn, later user turns and tool results
      (JSON after an answer with a call, minified with every literal as written), counts a refused turn, resets, and
      refuses past context − static prefix − `--max` with the engine's numbers: 41 of 41 refusals equal
      (`TestConversationMatchesEngine`); without the flag an oversized turn keeps its end (`TestConversationTrim`).
      Remains: how the engine drops earlier turns and renders the assistant's turns in the history (no count reports
      the history), how much of a cut turn it keeps, and its spelling of `<`, `&` and `>` in a tool result's strings
      (12 probes count more than the minified text) (F-051).
      (2026-10-15) — partial: `just oracle-text history` (`history.json`, 30 wire probes, 107 conversations and a
      `--max` scan, byte-identical on a re-run) reads the history from the later turns' answers and confidence. The
      `<`, `&` and `>` counts were the engine's request parser, which reads Go's `<` escapes as their hex digits;
      sent unescaped they count as the port renders them, so `TestConversationMatchesEngine` has 53 of 53 refusals.
      `Decoder.Decode` leaves the answer open less its ending token and `Conversation.Turn` closes it before the next
      turn: `<|im_end|>` after a call block before a tool result (F-052's form, kept for `timer_then_result`) and
      after any other answer, `</tool_call><|im_end|>` after a call block before a user turn; a refused turn feeds
      nothing. Without the flag an oversized turn is cut to its room and the conversation before it is dropped
      (`Restarted`): a first turn keeps the last tokens of its rendering, a later one its markers. Under int8 the later
      turns have the engine's reasoning on 52 and calls on 112 of 154, confidence within 0.02 on 37 of the 47 decoded
      as the engine's, token counts on 41 of 41 (`TestHistoryMatchesEngine`; F-063). Remains: the tool result after a
      call (the whole answer fits more turns in aggregate but breaks `timer_then_result`), the turn after a `respond`
      answer (confidence off), the history after a truncated answer (the engine's decode under a small limit is not the
      port's), and which positions the engine keeps past the context (the KV window hides them).
- [x] Tool retrieval: nothing ranks the tools. `needle3.cact` has no embedding head, so engine 3.2.0 renders every
      declared tool (1–239 measured); its one limit is the 8192-token context, `toolcall.CheckPrefix`
      (`TestStaticPrefixPastFiveTools`, `TestRetrievalOff`) (F-049).
- [x] Tool normalisation before rendering, from upstream's porting notes (F-049): unwrap OpenAI-style
      `{"type":"function","function":{…}}` entries, turn tool and property names into snake*case, drop keys the model
      was not trained on (`triggers` among them), and keep an alias table so the calls carry the caller's names. The
      corpus's tools need none of it; `needle_init`'s counts on tools that do (OpenAI-style, camelCase, `triggers`)
      pin the rendering, the answers the aliasing.
      (2026-10-12) — `toolcall.Normalise`: an entry's `function` object, only a tool's top-level `triggers` dropped,
      `toolcall.Snake` on tool and property names (non-ASCII names untouched) down nested properties, `items`,
      `anyOf`/`oneOf`/`allOf` and `additionalProperties` but not `$defs`, `required` following, and suffixes for
      colliding names; `CallerName`/`CallerArgs` map the calls back. `just oracle-text normalise` (`normalise.json`,
      58 sets, byte-identical on a re-run): the normalised static prefix has `needle_init`'s count on all 58, 33 of
      which differ without it (`TestNormaliseMatchesEngine`), and the engine's calls over camelCase, OpenAI-style and
      colliding tools carry the caller's names as the alias table gives them (`TestNormaliseAliasesMatchEngine`).
      Separators that tokenise like `*` and the suffixes' digits are not identified by the counts (F-050).
- [ ] **Accept:** the static-prefix token count equals `needle_init`'s on every engine case, and a float greedy decode
      of the rendered prompt gives the engine's `function_calls` on at least the cases where the float model's
      top-1 is not a near-tie.
      (2026-10-12) — partial: the static-prefix half is met (above). The decode half waits for 11.4, by the user's
      decision: the float model's greedy calls (`TestFloatGreedyAgainstEngine`, the case's token budget) equal the
      engine's on 7 of 19 cases; the other 12 differ where the engine works after or around the decode, not at
      near-ties: it capitalises strings (5), re-resolves dates (2), rewrites a message (1), holds the arguments to the
      schema (3) and suppresses calls below confidence 0.1 (2; F-048).
      (2026-10-13) — partial: with the grammar and the repair, the float model's calls after the envelope equal the
      engine's on 20 of 24 cases (`TestFloatGreedyAgainstEngine`). Of the other 4 only `negation_door` (margin 0.018)
      is a near-tie; `alarm_morning` (0.080, a label), `message_and_timer` (0.088, "I'm running late" kept) and
      `catalogue_email` (0.241, one recipient) are not (F-056).

### 11.4 Decoding and the response envelope (new `internal/toolcall`; needs 11.2, 11.3, the engine corpus)

Engine-only: fitted against `testdata/text/engine/`.

- [x] Engine corpus (`just oracle-text`, `testdata/text/engine/`): 24 cases with a fixed date fact, the answers
      byte-identical with the timings dropped at `--threads 4` (they depend on the thread count); also the depth
      runs, fused audio answers, `needle_init`'s prefix counts and `needle_embed` of text (F-047, F-049).
- [x] Greedy against beam, the stop tokens (`<|im_end|>`, `</tool_call>`, EOS) and `--max`.
      (2026-10-12) — `toolcall.Decoder`: a forced `<think>\n`, then greedy; `</tool_call>` counts and ends the answer,
      `<|im_end|>` and EOS end it uncounted, `--max` counts the forced tokens; the history keeps the answer less its
      last token and `<|im_end|>`. With the dequantised `.cact` weights the reasoning is the engine's on 21 of 24
      finished turns (the other three near-ties, margins ≤ 0.054) and the token count `needle_complete`'s on all but
      three the grammar changes; `timer_budget` truncates at 8 as the engine (`TestDecodeMatchesEngine`; F-052).
- [ ] The byte-level grammar compiled from the JSON schemas (types, `enum`, `const`, bounds, `format: date`, required
      keys, `anyOf`/`$ref`), applied as a token mask over byte-fallback pieces; the unconstrained `<think>` part.
      (2026-10-13) — partial: `toolcall.Grammar` (`Normalised.Grammar`, `GrammarFor`), a byte-level pushdown automaton
      over the call block compiled from the rendered schemas (the item's keywords, `oneOf`/`allOf`, local recursive
      `$ref`, upstream's caps of 20-character numbers and 64 items), and `tokenMask` over normal and byte-fallback
      pieces; `Decoder.Decode` keeps the reasoning free and masks from `<tool_call>` on. Model-free tests per keyword,
      the mask against brute force over Needle 3's vocabulary, and no reachable dead end (`TestGrammar*`,
      `TestTokenMask*`, `TestGrammarNoDeadEnds`). Against the engine: `lights_and_lock` and `catalogue_lock` now
      have its calls and count, `spanish_lights_music` its calls; `just oracle-text grammar` (`grammar.json`, 15
      schema-breaking requests, byte-identical on a re-run) gives its calls on 8 (`TestGrammarMatchesEngine`); the
      float greedy calls go from 10 to 14 of 24 (F-055). Remains: `thermostat_decimal` is one token short of the
      engine's count, and four probes the engine truncates at 512 tokens, which show nothing of how its grammar
      traps the model there.
- [x] Parse the call block into `function_calls`; the envelope's `type` (`call`, `respond`, refusal as an empty
      call), `success`/`error`/`error_code`/`reason`, `suppressed_calls` below confidence 0.1, `validation`
      (`ungrounded`, `negation`), `triggers` regexes.
      (2026-10-12) — partial: `toolcall.Normalised.Envelope`, `NoTools`, `Triggered`. `just oracle-text triggers`
      (`triggers.json`, byte-identical on a re-run) adds eleven trigger probes and the answer without tools. On all
      35 corpus and unrouted probe turns the type, the failure fields, the split into function and suppressed calls
      (below 0.1 unless a trigger matched an un-negated request), `negation` and `ungrounded` equal the engine's
      (`TestEnvelopeMatchesEngine`; F-054). Remains: routing a triggered request to its tool, which needs the grammar
      (three probes), and negation cues beyond "don't".
      (2026-10-13) — routing done: a trigger restricts the grammar to its tools and requires a call
      (`GrammarFor`); `capital_upper` has the engine's call, `weather_to_timer` and `pizza_to_lights` differ only
      where the repair step works (a capital, an unstated optional), with confidence within 0.002 on all three
      (`TestEnvelopeMatchesEngine`; F-055). Remains: negation cues beyond "don't".
      (2026-10-13) — `validation` done: `just oracle-text validation` (`validation.json`, 163 probes and 171 answers,
      byte-identical on a re-run) pins the cues and the grounding. `negation` reads `don't`, `dont `, `do not`, `never`,
      `shouldn't`, `should not`, `mustn't` and `must not` in any case, not the curly `don’t`, `not`, `won't`, `can't` or
      German, Spanish and French, and not before `forget`, `worry`, `hesitate`, `mind` or `disturb`; the drop of negated
      calls takes whole tokens and the verb within three; calls that are all negated or excluded are suppressed, not
      dropped. `ungrounded` lists top-level scalars once per call, grounded in the conversation's user turns and tool
      results, not in the system facts (`Conversation.Inputs`). `negation` equals the engine's on all 171 answers,
      `ungrounded` on all 151 whose calls are the engine's (`TestValidationMatchesEngine`; F-057).
- [ ] `confidence`: `ConfidenceHead` (`probe_pool` over the embedding and the 20 block outputs, a translation) and
      the min with the call's decode probability (engine).
      (2026-10-12) — partial: `text.Model.Confidence` equals JAX within 8.2e-6 (`TestConfidenceMatchesJAX`);
      `toolcall.Decoder.Confidence`, the sigmoid over the conversation as fed, is within 0.02 of the engine on every
      turn whose decode is the engine's but `event_tomorrow` (0.026) and the `respond` turn (0.927 against 0.637)
      (`TestEnvelopeMatchesEngine`; F-053). Remains: the `respond` turn, and the min with the decode probability,
      never binding in the corpus.
      Note (2026-10-13): the engine scores 0.0000 on every call with an object-valued argument
      (`ground_nested_time` and exploratory requests, F-057), which the head does not reproduce.
      (2026-10-14) — partial: `just oracle-text confidence` (`confidence.json`, 199 probes and 258 answers,
      byte-identical on a re-run) pins the min. The envelope's `confidence` is min(head, `Decoded.Probability`): for a
      leading call block the smallest probability of a block token among the tokens the grammar allows (it binds on
      probes from far off, e.g. 0.507 against the engine's 0.506 where the head says 0.84; never in the corpus), for a
      `respond` answer the geometric mean of every token's probability with its stop token (the head never sets it);
      suppression reads the min. The `respond` turn is now within 0.02. Of the turns decoded as the engine's,
      confidence is within 0.02 on 106 of 135 probes and 19 of 22 corpus turns in float32, 128 of 148 and 20 of 22
      under int8 (`TestConfidenceMatchesEngine`, `TestEnvelopeMatchesEngine`; F-062). Remains: the 0.0000, which is
      not an envelope rule but comes from the engine's own decode inside objects and cut integers and is not pinned;
      a decimal for an integer (the port's grammar forbids the fraction, so the min binds on three answers the
      engine scores by the head); `event_tomorrow` (−0.043) and `catalogue_lock` (+0.067) under int8, traced to the
      weight path's implementation noise of F-060.
- [ ] The engine's post-decode repair (llms.txt:110: capitalisation, date re-resolution, enum/polarity flips,
      dropping unevidenced optionals, duplicate and negation drops, rewrites), which F-048's 7 of the 12 differences
      come from.
      (2026-10-13) — partial: `toolcall` repairs the calls after parsing and before suppression and validation, one
      function per rule (`repair.go`; `Envelope` takes the system facts for the `date:` fact).
      `just oracle-text repair` (`repair.json`, 90 probes, each rule firing and held back, byte-identical on a re-run)
      gives the engine's calls on 85 (`TestRepairMatchesEngine`); every repair difference of the corpus and the
      grammar probes is gone (`callsDiffer` empty, `TestEnvelopeMatchesEngine`, `TestGrammarMatchesEngine`), with
      model-free tests per rule (`TestRepair*`). Pinned and implemented: the name split, phone completion, quoted titles, a place's following
      address, weekday and `March 4th` dates, polar booleans and enums in the call's own clause, `from X to Y`, the
      paired tool, `by N`, unstated optional numbers, empty optional strings, duplicates, excluded and negated calls,
      and the capitalisation of seven keys (F-056). Remains: the invented required slot's refill (not identified),
      a required number's default (never fires in the probes), unstated optional strings and a place query's dropped
      words (not pinned), the enum's named option (the probes still differ, the engine adds `oscillate: true`), and a
      boolean flipped from another clause.
      Note (2026-10-13): the paired tool keeps its tool when `not`, `never`, `dont` or `don't` start within 13 bytes
      before the verb, a third reading of negation; the port's clause rule gets 7 of 12 `paired` probes of
      `validation.json` wrong (F-057).
      (2026-10-14) — partial: the grounding gates of upstream's confidence guide withhold calls as the engine does,
      one function per gate (`repair.go`): a reported quote, an unnamed required enum, a control word or an
      unmentioned place in a required slot, no quantity for a required number, the destination copied into the
      origin. `just oracle-text gates` (`gates.json`, 200 probes, byte-identical on a re-run) gives the engine's calls
      and split on 193 of 204 answers (`TestGatesMatchesEngine`, the 11 others listed with a reason). Under a matching
      trigger only the quote gate (with negated and excluded calls) still withholds. The paired tool and polarity read
      the 13-byte window instead of the clause rule: all 12 `paired` probes equal (`TestValidationMatchesEngine`).
      Remains as before, less the required number (now the quantity gate); `place_named`'s lower-cased `hamburg` is
      a repair not identified (F-058).
- [ ] **Accept:** on the engine corpus, `type` and `function_calls` (name and arguments, as parsed JSON) equal the
      engine's on ≥ 95 % of the cases with every miss explained by a log-probability margin, `reasoning` equal on
      ≥ 90 %, `confidence` within 0.02.
      (2026-10-14) — partial, read by the user's decisions: the engine's numerics (packed + int8 + the KV window,
      `NEEDLE_TEST_NUMERICS=int8`), "the engine corpus" the 25 turns of `cases.json`, the confidence on every turn
      decoded as the engine's. `TestEnvelopeMatchesEngine` under `NEEDLE_TEST_NUMERICS=int8`: `type` and
      `function_calls` equal on 24 of 25 (96 %; the miss, `message_and_timer`, at a margin of 0.005), met; `reasoning`
      equal on 22 of 25 (88 %; misses at margins 0.005, 0.014 and 0.076), not met; confidence within 0.02 on 20 of
      the 22 turns decoded as the engine's, not met (`event_tomorrow`, `catalogue_lock`; F-062).

### 11.5 Native quantised path (`internal/text`, `internal/weights` or a packed `text.Source`, `internal/cq`; needs 11.2) — ✅ DONE (2026-10-14)

- [x] Serve the 122 CQ records packed (`cq.Matrix`) and run the model on them, as 9.2 does for Whistle.
      (2026-10-14) — `text.OpenWith(path, text.OpenOptions{Packed: true})` serves all 122 CQ records as `cq.Matrix`
      with no float32 copy (`TestSourcePacked`, `TestLoadPackedNoFloatCopy`); `text.Open` stays float. On exact
      kernels (`cq.Matrix.Exact`, the default packed mode) the logits are within 1 − cos 8.4e-14 of the dequantised
      model with top-1 equal at 1732 of 1732 positions and the confidence within 3.7e-6 (`TestForwardCactPacked`,
      `TestExtendCactPacked`); the weights take 37.2–38.3 MiB by run (`TestPackedMemory`; F-059 records 38.252 MiB,
      under 40 MiB and 0.3 % over 40 × 10⁶ bytes). Whistle's lut kernel (`LUT: true`) is 3.5× faster on a 120-token
      forward but moves the confidence by up to 3.5e-4 and adds 15 MiB; the exact path is 3.4× slower than float32
      (F-059).
- [x] The engine's int8 activations and KV (`act_bits` 8, `kv_bits` 8, `Int8` of F-041) and its KV window of 256.
      (2026-10-14) — `text.LoadWith(src, cfg, LoadOptions{Int8, KVWindow, SinkToken})`, opt-in; `Load` and the float
      and packed numerics unchanged. The int8 KV cache takes the sliding window and the KV window with a sink
      (`Attention.SetKVWindow`, `KVCache.SetSink`, `AttendInt8Spans`). By F-041's ρ on the confidence logit over 386
      answers, the engine quantises the queries, keys and values per head (the mix 0.86 ± 0.12 dequantised, 1.00 ±
      0.14 packed), not the attention input, the gated output or the heads' inputs; the mHC input and the engram rows
      are too small to read and stay float32. The KV window is 256 positions on every layer with the static prefix
      (BOS … `</tools>`) kept in view: on `just oracle-text window` (`window.json`, 22 answers whose fact lies 90–1100
      tokens back, byte-identical on a re-run) it gives the engine's calls on 17 (F-060: 12 without a window), and the
      engine's made-up calls where the window cuts the fact (`TestWindowMatchesEngine`). Packed + int8 + window
      against float32 (`NEEDLE_TEST_NUMERICS=int8`): reasoning 33 against 29 and confidence within 0.02 on 31 against
      26 of 36 envelope turns, validation calls 162 against 159, gates 194 against 193, grammar 9 against 10. Open:
      past 1024 positions, the window's edge, prompts without `</tools>` (F-060).
- [x] **Accept:** with the `.cact` weights, `function_calls` equal the engine's at least as often as with the float
      weights; memory and speed against the engine (`just bench`-style), packed weights under 40 MB.
      (2026-10-14) — met, read by the user's decisions: the `.cact` side is the engine's numerics (packed + int8 + the
      KV window), the float side `needle3.safetensors`, the count summed over the engine test sets; "40 MB" is 40 MiB
      as in F-059. `NEEDLE_TEST_NUMERICS=safetensors` against `int8` over the six disjoint sets (538 answers):
      `function_calls` equal on 507 against 426, at or above safetensors in every set (envelope 35/23, validation
      162/127, gates 197/177, repair 84/76, grammar 9/7, window 20/16); the first turns alone 23 against 20. The
      packed weights take 38.25 MiB (`TestPackedMemory`). `just bench-text` and `just oracle-text bench`
      (`bench/RESULTS-text.md`, the 24 first turns): the engine-numerics mode on the lut kernel reaches a TTFT of 111
      against the engine's 73 ms cold (5.6 ms with its prefix cached), decode 411 against 1767 tok/s, an answer 425
      against 107 ms, peak RSS 492 against 103 MiB; on the exact kernel TTFT 350 ms and decode 116 tok/s. Speed and
      memory are reported, not bounded; the levers are in F-061's Open.

### 11.6 Depth ladder (`internal/text`; needs 11.4)

- [ ] `--depth N`: the bisection order over 20 layers (`_ladder_layer_order`), the sub-network lanes (F-030), the
      confidence head's `row_keep` for the dropped blocks.
- [ ] **Accept:** per depth the engine accepts, `function_calls` equal the engine's at the 11.4 rate.

### 11.7 Public API, CLI and parity corpus (a public `needle` package, `cmd/`; needs 11.4)

- [ ] A Go API mirroring `needle.Needle` (`Complete`, `Run` with Go tool functions, `Extract`, `Reset`), returning the
      engine's envelope.
- [ ] A CLI mirroring the engine's (`--tools`, `--system`, `--prompt`, `--max`, `--depth`, `--serve`).
- [ ] The parity corpus: the six `needle/environments` suites (`TEST_CASES`) recorded with the engine.
- [ ] **Accept:** the envelope JSON equals the engine's on ≥ 95 % of the corpus without the timings, and every
      environment suite passes exactly when the engine's passes.

### 11.8 Audio in → tool calls out (`whistle/` + the text model; needs 11.7)

- [ ] `needle --model needle3.cact --model whistle.cact --tools … --audio clip.wav`: transcribe with Whistle, answer
      the transcript, merge the speech fields under the `audio_` prefix.
- [ ] **Accept:** the fused JSON equals the engine's on the audio fixtures without the timings.

### 11.9 Text embeddings (optional; needs 11.2)

- [ ] `needle_embed` for text: find what it pools without an embedding head in `needle3.cact`.
      Note (2026-10-12): upstream's porting notes call it "four probe queries over width 768", "not a contrastively
      trained embedding", and give a shortlist recipe over it (`argsort(-(index @ q))[:k]`) for callers with many tools,
      since the engine does not retrieve without an embedding head (F-049).
- [ ] **Accept:** cosine ≥ 0.99 against `needle_embed` on the recorded inputs.

## Phase 12 — Public release & follow-ups from the first write-up

The repository went public on 2026-10-11; writing the first blog post about it turned up the items below.

### 12.1 Docs and release — ✅ DONE (2026-10-12)

- [x] README for the public module: `go install github.com/cwbudde/go-whistle/cmd/whistle@latest` and
      `go get github.com/cwbudde/go-whistle/whistle`.
- [x] Releases: `.github/workflows/release.yaml` builds the CLI for four platforms from a `v*` tag; `v0.1.1` is the
      first working release (`v0.1.0`'s arm64 build failed; `docs/releases/`).
- [x] The README links the hosted demo, https://cwbudde.github.io/go-whistle/.

### 12.2 Web demo — ✅ DONE (2026-10-11)

- [x] Insecure contexts: the SHA-256 is computed in Go (`whistle.sha256`) without `crypto.subtle`; the microphone is
      disabled with a note.
- [x] GitHub Pages (`.github/workflows/pages.yaml`) deploys `web/` on every push to `main`.
- [x] The stream line shows the passes, the last pass's time and the sum.

### 12.3 Performance on amd64

- [ ] GEMM in AVX2+FMA: `algo-tensor` has `gemm_arm64.s` but no amd64 GEMM, and `whistle doctor` says so on x86
      ("AVX2: CQ LUT GEMV; AVX2+FMA: dot, axpy, softmax exp; GEMM in Go"). The float products (the Monarch
      Kronecker products, the float path) therefore run in pure Go there. 9.4's note that they run on "the existing
      NEON/AVX2 GEMM" is only true on arm64; correct it. The kernel belongs in `algo-tensor`: its PLAN 1.4 (after 1.2,
      the AVX2 port of algo-transcribe's kernels) gives `tensor.MatMulTransB` an AVX2 + FMA kernel as a numerics
      change, released as v0.4.0; then bump the module here. **Accept:** with algo-tensor v0.4.0 (its 1.4, the
      AVX2 + FMA `MatMulTransB`), the parity numbers on amd64 and arm64 stay at
      least as good as before (`TestTranscribeMatchesEngine`, the keyword fixtures, the depth ladder, the stream
      decisions, the LibriSpeech WER and the `Embed` cosine against the engine), the `Embed` and `Transcribe` outputs
      that change are named with their largest difference and their fixtures regenerated in the same change, and a
      measured TTFT gain on an x86 machine.
      (2026-10-12) — partial: 9.4's note now says the GEMM is NEON on arm64 and pure Go on amd64. The kernel waits for
      algo-tensor 1.1 (its PR #6), 1.2 (after algo-transcribe PR #9), v0.2.0 and 1.3. The amd64 numbers it has to hold
      are those of the next item: 31/32, keywords 47/53, depth 161/192, stream 797/930, WER 5.35 %, `Embed` 0.999828.
      (2026-10-12) — the gate moved: algo-tensor released v0.2.0 after its 1.1 (`b33d487`, without 1.2), and its PR #7
      splits 1.3: the `Conv2D` routing (v0.3.0, not on amd64) stays 1.3, the AVX2 + FMA `MatMulTransB` becomes 1.4
      after 1.2. By the user's decision 1.2 and 1.4 ship together as v0.4.0 (1.2 also moves bits on AVX2 machines, a minor
      release by algo-tensor's rules; algo-tensor PR #8), and the Accept names v0.4.0. The
      kernel now waits for 1.2 (after algo-transcribe PR #9) and 1.4; adopting v0.3.0 is a 12.4 item.
- [x] Benchmarks on linux/amd64: `bench/RESULTS-linux-amd64.md` (2× Xeon Gold 5520+, KVM guest): TTFT 47.3 against
      the engine's 160.9 ms, decode 547 against 523 tok/s; parity as on arm64 (F-040 "linux/amd64").

### 12.4 Remaining speed gap (optional)

- [ ] TTFT against the engine: 18.8 against 6.9 ms (2.72×), decode 1085 against 1499 tok/s (0.72×), median over the
      32 clips (F-040). Profile where the TTFT goes (front end, stem, encoder, memory, first decoder steps) and take
      up the open levers of F-037 and F-041. Engine speed is still not a hard goal (Risks); the aim is a measured
      breakdown and whichever levers pay off without moving the parity numbers.
- [x] algo-tensor v0.3.1 (`Conv2D` through `Conv2DPlan` on arm64 and js/wasm): parity unchanged, amd64 and purego
      outputs byte-identical (F-038).
- [ ] js/wasm speed (F-043 "Open"): about 22 tok/s and a TTFT of 0.3–0.8 s on one thread without SIMD. Try several
      workers, each with its own 17 MB model, to run the beams or the encoder blocks in parallel. A hand-written wasm
      SIMD module for the LUT kernels would leave pure Go; decide first whether it is worth that.

---

## Unknowns to resolve against the oracle

All resolved for Whistle.

| #   | Unknown                                                  | Resolution                                                                                     |
| --- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| U1  | Mel log flavour, floor, normalisation; "250–3500 Hz"     | power, soft log knee 76 dB below the peak, per-channel norm, full band (F-020)                 |
| U2  | Stem strides, padding, activations                       | stride 2 on all three convs, zero padding 1, SiLU (F-016)                                      |
| U3  | Encoder conv module, macaron scale, `pe_gate`, positions | GLU a·σ(b), FFN ½, RoPE only, `pe_gate` unused (F-017, F-019)                                  |
| U4  | The audio manifest values `3246, 1`                      | magic 0xCAE and version 1 (F-006)                                                              |
| U5  | Shared Hadamard permutations between encoder and decoder | shared (F-017)                                                                                 |
| U6  | Cross-attention position in the block                    | after the self-attention (F-021)                                                               |
| U7  | Decoder prompt format, language detection                | `[<s>]`, language by argmax (F-023, F-024)                                                     |
| U8  | Beam length normalisation, EOS rules                     | Whisper's at 5 beams, Σ log p / len (F-026)                                                    |
| U9  | Keyword bias weight and revocation                       | 2.15 nats per token, revoked on failure; the engine's rule not found (F-044)                   |
| U10 | Silence gate                                             | p99 − p10 of 250–3500 Hz mel frame energy ≤ 5 dB (F-031)                                       |
| U11 | Word timestamps and probabilities                        | Whisper's alignment over the last half of the layers (F-033)                                   |
| U12 | Stream commit rule                                       | LocalAgreement-2, 15 s trim; past 30 s the port keeps 30 s, the engine most likely all (F-036) |

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
