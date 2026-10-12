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
│   ├── search/              beam search (copy-on-write KV per beam), Aho-Corasick keyword bias
│   ├── align/               word grouping, word probability, cross-attention DTW timestamps
│   ├── cq/                  CQ LUT GEMV in the rotated (Hadamard) domain, straight from the packed .cact
│   ├── eval/                WER/CER and edit distance shared by the parity tests and the LibriSpeech harness
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
2. **safetensors ↔ `.cact`**. Each dequantised `.cact` tensor must match its FP32 twin by cosine. Measured: CQ4
   0.9943–0.9954, CQ2 0.9319–0.9401 (F-006; the earlier "≥ 0.95 for CQ2" guess was wrong). This checks the CQ path **and** the positional name map.
3. **Engine** (`needle` CLI / `libneedle3`):
   - `needle_embed` gives encoder output per frame. Target cosine ≥ 0.99, since the engine runs W2A8.
   - `needle_transcribe` gives the final JSON. Target: transcript equality, the same language and timestamps within one frame.
   - Fixture set: LibriSpeech test-clean/other clips, FLEURS clips for de/fr/es/it/nl/pl, silence, a 30 s+1 clip and a keyword-biasing case.
   - WER on a LibriSpeech test-clean subset, compared with the published 4.31 (test-other 10.49).

Following the upstream porting blog, logits are compared by **cosine and top-1**, not by relative error.

---

## Phase 0 — Bootstrap — ✅ DONE (2026-10-09)

- [x] Repo, `go.mod`, `justfile`, `treefmt.toml`, `.golangci.yml`, `AGENTS.md`, `PLAN.md`
- [x] GitHub Actions: `just ci` on linux/amd64 and macos/arm64. Model-dependent tests skip when `models/` is absent.
      (2026-10-09) — `.github/workflows/ci.yml` (initially an ubuntu-latest + macos-latest matrix, treefmt 2.6.0,
      golangci-lint 2.13.2) and a doc-only `whistle` package, without which `go test`/`golangci-lint` fail on an empty
      module. First run on `CWBudde/go-whistle` PR #1 (run 37953639555): both jobs green, macos 1m37s, ubuntu 1m41s.
      macOS was then dropped: CI runs ubuntu-latest only, because macOS minutes bill at 10x on a private repo and the
      arm64 assembly copied from go-pocket-tts is built and tested there.
- [x] `internal/hub`: a minimal pure-Go Hugging Face downloader. It resolves `revision`, resumes downloads, verifies
      SHA-256 against a committed `models/manifest.lock.json`, and honours `HF_TOKEN`/`HF_HUB_OFFLINE`. Pin
      `whistle.cact`, `checkpoints/whistle.safetensors` and `config.json` at `b358ddad`.
      (2026-10-09) — the manifest lives in `internal/hub/manifest.lock.json`, embedded with `go:embed`, rather than in
      `models/manifest.lock.json`: `models/` is git-ignored and an installed binary has no checkout to read from.
      `b358ddad` is pinned as `b358ddadd89b7a713b5aa131f23032d3cca1b251`; `config.json` is
      `ac2dbcb9…a3d43` (603 B). `Download` re-resolves the pin, skips verified files, resumes from `<local>.part`
      via `Range` (206 append, 200 restart, 416 re-verify), and checks size + SHA-256 before the rename.
      `HF_ENDPOINT` is honoured too. httptest suite plus opt-in `WHISTLE_HUB_LIVE=1` test. Against real HF: all three
      files verified, and a `.part` truncated to 100 MB resumed to the right `whistle.safetensors` digest.
- [x] `whistle download` command, with the local layout `models/whistle/{whistle.cact, whistle.safetensors, config.json}`
      (2026-10-09) — `cmd/whistle` (cobra v1.10.2): `whistle download [--dir models/whistle] [--offline]` drives
      `hub.Pinned()` + `Client.Download`. Progress goes to stderr, and one `whistle b358ddad ready in <dir>` line goes
      to stdout. `HF_TOKEN` comes from the environment only (no flag), and `--offline` is OR'd with `HF_HUB_OFFLINE`.
      Ctrl-C cancels the download and keeps the `.part` file for resume. Tests cover the flags, the pinned layout, a
      download from a fake Hub, and offline failure without a request. Real run: a fresh download in 22 s with all
      three digests verified; the re-run and `--offline` only verify.
- [x] `scripts/fetch-engine.sh`: fetch the platform `needle` binary and `libneedle3` from `Cactus-Compute/needle3` into
      `engine/<platform>/`. Set `NEEDLE_TELEMETRY=0` and `DO_NOT_TRACK=1` for every oracle run.
      (2026-10-09) — `just fetch-engine [--platform …]` for macos-arm64, linux-x86_64 and linux-arm64, pinned to HF
      revision `2ae11323dc` with a SHA-256 for every file. `libneedle3.{dylib,so}` ships only inside
      `python/cactus_needle-3.2.0-py3-none-<tag>.whl`, so the script extracts it from there: upstream's `<platform>/`
      dirs on HF hold only `needle`, the static `libneedle.a` (not fetched) and `needle.h`. `engine/<platform>/` ends
      up with `needle`, `needle.h`, the wheel, the extracted `libneedle3.{dylib,so}` and a `PIN` file. The CLI has no
      `--version` and no version string; its upload (`80896a6582`, 2026-10-05) has the same date as the 3.2.0 wheel.
      Only macos-arm64 was run here.
- [x] Snapshot upstream into `original/needle` at `af2654dffa` and set up `uv` (`jax`, `flax`, `safetensors`,
      `sentencepiece`, `numpy`) for `scripts/`
      (2026-10-09) — `just fetch-upstream` (shallow fetch of the pinned commit, idempotent, HEAD verified);
      `scripts/pyproject.toml` + `uv.lock` (Python 3.12; resolved jax 0.11.2, flax 0.12.10, numpy 2.5.3).
- [x] `tools/oracle` (build tag `oracle`): run `needle --model … --audio clip.wav --audio-word-timestamps` and
      `needle_embed`, and write `testdata/oracle/<clip>.{json,embed.safetensors}`.
      **Accept:** fixtures exist for at least 20 English and 2 clips per other language.
      (2026-10-09) — `just oracle-clips` stream-reads the tarballs and closes them once enough clips are found:
      LibriSpeech test-clean/test-other from openslr (FLAC decoded with `mewkiz/flac`) and FLEURS `test.tar.gz` at
      `70bb2e84b9`. It commits 32 clips of 2–8 s as 16-bit WAV (5.5 MB) in `testdata/audio`, with `manifest.json`
      (archive, member, transcript, sha256) and `ATTRIBUTION.md` (CC BY 4.0). The selection is deterministic:
      10 clips per LibriSpeech split from 10 speakers (≤ 2 each), and 2 different sentences per de/fr/es/it/nl/pl.
      `just oracle-run` runs the CLI and gets `needle_embed` from `libneedle3` through purego. It writes
      `<id>.json` (timing fields dropped) and `<id>.embed.safetensors` (F32 `[frames, 512]`), plus `manifest.json`
      with the engine and model pins (4.6 MB in total). Count: en 20, de/fr/es/it/nl/pl 2 each. Detected language
      32/32. A re-run is byte-identical. Embed shape and CLI output are in `docs/FINDINGS.md` F-001/F-002.
      `testdata/{audio,oracle}` are excluded from treefmt. Only macos-arm64 has been run.

## Phase 1 — Weights & tokenizer — ✅ DONE (2026-10-09)

- [x] Copy `internal/safetensors` from go-pocket-tts. Parse `__metadata__` `config` and `run` into a typed `Config`.
      (2026-10-09) — copied from go-pocket-tts `9faba38` (Store with F32/F16/BF16/I64 decode and key remapping,
      `EncodeTensors`/`WriteFile`, `LoadFirstTensor`, plus tests), without the voice-state and Mimi helpers.
      `Store.Metadata()` exposes the `__metadata__` string map, and non-string values are rejected.
      `EncodeTensorsWithMetadata` writes one. `Store.Config()` and `Store.Run()` decode strictly: unknown and missing
      keys are errors, and the training-only keys are kept so the decode is lossless. `Store.Step()` reads the
      training step. The model-gated `TestWhistleCheckpoint` pins `whistle.safetensors`: 116 F32 tensors, 55,151,433
      parameters, `embedding/embedding` `[8199, 512]`, `Config` equal to the reference card, `Run` = qapt/kv 8/act 8,
      and step 25000. Loading still reads the whole file; mmap belongs to `internal/weights`.
- [x] `internal/cact`: the header (196 B, tag check, all fields), the 28 codebook floats, the 44-byte records, 64-byte
      aligned blobs, and readers for FP32, FP16 and RAW. (2026-10-09) — `Parse`/`Open` decode the 49-field `Header`
      and every `Record` with `binary.Decode` and validate the whole directory up front: tag, codebook length, engram
      order/site counts, pad, ndim and trailing dims, 64-byte alignment, blob order/overlap/EOF, and nbytes per dtype
      (CQ per `_packed_row_bytes`, ternary included), with overflow-checked sizes. `Codebook(2|3|4)`, `FP32`, `FP16`
      (FP16 widening copied from `internal/safetensors`), `Float32` and `Raw`; the wrong dtype yields `ErrDType`.
      Unit tests use a Go encoder that mirrors `_pack_cact`. The model-gated `TestWhistleCact` pins the full header,
      all 28 codebook floats, the record histogram (FP16 524, FP32 4, CQ2 145, CQ4 7, RAW 1), records 0/226/227/237/
      679/680, the audio manifest and the 8199-piece tokenizer blob; `TestWhistleCactMatchesConfig` maps the header
      onto the safetensors config (`docs/FINDINGS.md` F-003).
- [x] CQ dequant: the LSB-first bitstream unpack for 2/3/4 bits, the per-group FP16 norm and FWHT-128 (normalised).
      Unit-test it against a tiny tensor packed by a Go port of `export.py` `_cq_pack`. (2026-10-09) — `File.CQ`
      mirrors `_cq_unpack` and returns `[out][in]` without the padding columns. It does the LSB-first unpack (read in
      8-index chunks), looks up the stored codebook, scales by the FP16 norm and runs an in-place float32 FWHT with
      1/√group folded into the norm. Groups must be a power of two. 1-bit and ternary (record bits 5, crumbs 3/0/1 →
      trits 0/1/2) are implemented as upstream does them, but they are only unit-tested because Whistle has none.
      Crumb 2 and non-power-of-two groups are errors. Unit tests pin hand-computed bitstreams for 1–4 bits and one-hot
      Sylvester rows. A float64 matrix reference checks `CQ` within the float32 bound γ(log2 g + 3)·Σ|x|/√g. The
      `_cq_pack` port, generalised to every width, runs a round trip over 7 shapes (groups 8–128, `in` not a multiple
      of the group, zero rows). Its per-group bound is n·‖levelBound‖ + |n − n16|·‖q‖ + fp32 slack, where
      `levelBound` is half the codebook cell gap or the overshoot past an end level. The measured relative errors (0.60
      / 0.34 / 0.17 / 0.09 for 1–4 bits) match Gaussian Lloyd-Max theory, and the bound catches a 5 % scale error.
      The model-gated `TestWhistleCQ` dequantises all 145 CQ2 and 7 CQ4 tensors (finite, RMS 1e-3…10, max |w| 69.29).
      It pins row norms and values of tensors 0, 2 and 671 against numpy `_cq_unpack` to 1e-5 of the row norm; the
      measured gap is about 1e-8. `TestWhistleCQMatchesSafetensors` pins cos(`CQ(0)`, `embedding/embedding`) at
      0.99537221, with a worst row of 0.98980336. Discovery: the header codebooks equal `_cq_codebook_np(b, 128)` bit
      for bit, so the RNG need not be ported. The header holds one codebook for one export group size, which is fine
      for every upstream writer (`docs/FINDINGS.md` F-004). Dequant is eager and single-threaded for now; laziness
      belongs to `internal/weights`.
- [x] Tensor name map (index → safetensors name). **Accept:** `scripts/compare_cact.py` (or a Go test) reports a cosine
      for all 681 tensors, every tensor passes its threshold, and the map is pinned in a test.
      (2026-10-09) — `cact.WhistleNames()` (`internal/cact/names.go`) gives every index an upstream-style name
      (`layer03.q_proj`, `cross03.*`, `enc03.hada_0.d1`, `stem.out`, …), its safetensors source, a layer slice and a
      transform: reshape, transpose (Flax `(in, out)` → `(out, in)`) or lane-major (mHC φ), or derived.
      `Entry.Apply` performs the transform and is unit-tested. A brute-force cosine search over all shape-compatible
      slices and transposes, which assumed no order, found the order: embedding; 8 decoder layers × 27 in exact
      `_tensors` order; decoder mHC; `hada_p1/p2`; engram sites 3 and 7; `final_norm`; the audio manifest at 237
      (where `heads.manifest` would go); 8 cross-attentions × 10; `pe_gate`; 8 encoder layers × 43 (six norms in
      macaron order, attention, pw1/pw2/dw, `hadamard_mlp_0`, `hadamard_mlp`); encoder mHC; `enc.final_norm`; stem;
      mel bank; tokenizer. All 116 safetensors tensors are used. The map is pinned by the golden
      `internal/cact/testdata/whistle_names.txt` plus structural invariants.
      The model-gated `TestWhistleNameMap` logs a 681-row table (cosine, threshold, check) in about 1.5 s. FP16
      (524): all 458,057 values are the RNE FP16 of the source, and each cosine meets the rounding bound
      √(1 − ‖h‖²/‖x‖²) (0.99999991–0.99999996). CQ (145 CQ2, 7 CQ4): each record equals a Go `_cq_pack`
      requantisation of its source bit for bit, except 11 norms that differ by 1 ulp and 16 indices that flip at
      float32 near-ties (≤ 4e-8 from a decision boundary; the derived tolerance is 1.5e-5). The cosine threshold is
      cos(Q(src), src) less the angle of 1-ulp norms. Pinned minimum cosines: FP16 0.999999963923, CQ2 0.931853685259,
      CQ4 0.994312193718. Derived records equal a Go recomputation: `hada_p*` a numpy MT19937 `permutation` port, the
      mel bank librosa Slaney in float64 rounded once to float32 (bit-exact), the manifest
      `[0xCAE magic, version 1, config…]` (patched copies show the engine needs both). The tokenizer parses and its
      sha256 equals F-005's.
      Every CQ width matches the run's `weight_bits`. Discovery: the Gaussian Lloyd-Max cosine (0.9394) is not a
      floor, since 143/145 CQ2 fall below it (0.9319–0.9401), so the "≥ 0.95 for CQ2" expectation above was wrong.
      The `.cact` file holds nothing beyond the checkpoint, its config and two constants (`docs/FINDINGS.md` F-006).
      The RAW tokenizer has no cosine; its check is the parse plus the hash.
- [x] `internal/weights`: one lookup interface over both formats, with lazy dequant and an mmap where the platform allows it.
      **Accept** (set for this item): one interface with a `.cact` and a safetensors backend under the F-006 names;
      equal shapes for every name, and a model-gated test comparing the backends for every name by the F-006
      thresholds; nothing dequantised at open, CQ dequantised on first access, cached and safe for concurrent callers
      (`-race`); a pure-Go mmap with an `os.ReadFile` fallback, both paths identical, `Close` unmaps, and nothing
      handed out aliases the mapping; a tested decision for the derived records; unit tests on synthetic files.
      (2026-10-09) — `weights.Source` (`Names`, `Has`, `Shape`, `Tensor(name) ([]float32, []int, error)`, `Raw`,
      `Config`, `Release`, `Close`) is implemented by `Cact` and `Safetensors`, and `Open` picks one by the `.cact`
      tag. The safetensors side applies `Entry.Apply`, so both return the `.cact` layout. Both check every entry's
      shape at open time without reading tensor data; `safetensors.Store.Shape` was added for this. `weights.Config`
      holds the geometry both formats state (F-003). The safetensors side derives `Heads` from `layer00.q_proj`, and
      both configs are identical for Whistle. Cache: a per-name `sync.Once`, unbounded until `Release(name)` or
      `Close`. The cached slice is the one the model keeps, so it costs nothing extra; callers share it read-only.
      mmap: `syscall.Mmap` behind `//go:build unix`, with `os.ReadFile` in `mmap_nommap.go` (windows, js/wasm,
      wasip1, plan9) and `Options.NoMmap` to force it. Opening `whistle.safetensors` takes 0.6 ms and +0.2 MB of heap
      mapped, against 12–23 ms and +221 MB read. Derived records: the safetensors side recomputes `hada_p1/p2`, the
      mel bank and the manifest; the recomputations moved to `internal/cact/derived.go`. The tokenizer is
      `ErrUnavailable`, because the checkpoint has none and it is data, not a function of the config. The
      model-gated `TestWhistleCrossFormat` covers all 681 names: shapes equal, derived records bit-equal, 524 FP16
      records within half an ulp and above the rounding bound, and per-type minimum cosines equal to F-006 to 1e-9
      (FP16 0.999999963923, CQ2 0.931853685259, CQ4 0.994312193718). Each backend also equals its plain reader bit
      for bit, and `OpenCact` decodes 0 tensors. `TestWhistleMmapFallback` finds 680/680 tensors identical mapped and
      read, per file. Unit tests on synthetic files of both formats cover laziness, 32 concurrent callers, `Release`,
      close during decode, address-range aliasing and open errors. Gap: CQ records are checked against F-006's
      per-width floor here; the per-record requantisation threshold stays in `TestWhistleNameMap`, which applies
      because the backends equal the plain readers (`docs/FINDINGS.md` F-007).
- [x] `internal/tokenizer`: load the pieces, scores and types from the RAW blob (format per `export.py`). Decode
      (control/unk skipped, byte fallback, `▁`→space) and `RefTokenizer` encode. **Accept:** round-trip on multilingual
      sample text; decode of known id sequences matches `sentencepiece` (fixture from `scripts/`).
      (2026-10-09) — `Parse`/`FromCact` read the blob (header, 8199 records, strict validation) into pieces, scores
      and types. `Decode` follows sentencepiece: CONTROL/UNK vanish, `▁`→space in piece surfaces, and byte runs use
      one U+FFFD per invalid byte. `Encode` is a straight port of `RefTokenizer` (markers longest first, best-score BPE,
      byte fallback). Fixture route: `scripts/tokenizer_fixture.py` (`just tokenizer-fixture`) rebuilds a
      sentencepiece `ModelProto` from the blob itself. It cross-checks against needle3's `tokenizer.model`: 0/8192
      pieces differ, and 104/104 decodes plus 24/24 encodes over the shared ids are equal. The only change is
      `unk_surface` set to `""`. The model-gated tests match sentencepiece on 109 sequences (2622 ids) and on the
      decode of every single id (hash), match `RefTokenizer` encode on 25/25 multilingual samples, and round-trip all
      25 plus 11 more texts. `RefTokenizer` decode disagrees with sentencepiece on 11 byte-fallback corner cases;
      `docs/FINDINGS.md` F-005 lists them. Gap: the engine's behaviour on those cases is not yet measured.

## Phase 2 — Shared Needle 3 blocks (float32) — ✅ DONE (2026-10-10)

- [x] Copy `internal/tensor` and `internal/ops` from go-pocket-tts; drop what is not needed (convtranspose, Mimi-specific ops).
      (2026-10-10) — copied from go-pocket-tts `internal/runtime/{tensor,ops}` @ `9faba38`; every file names its source
      and, where it differs, the change. Kept: `Tensor` with broadcast and shape ops, `MatMulTransB` (NEON 4×4/1×4),
      `DotProduct` and `Axpy` (NEON, AVX2+FMA via `golang.org/x/sys/cpu`), `Linear`, `MatMul`, `Softmax`, `Attention`
      (causal + offset or none), `CausalMask`, `RoPE`, `Conv1D`/`Conv1DLeftPad` and the conv worker/scratch pools.
      Dropped: `ConvTranspose1D` (Mimi), `MLP` (FlowLM; Whistle uses `HadamardMLP`), `AttentionWithPositions` (Pocket-TTS
      streaming mask), `LayerNorm` (Needle norms are ZCRMSNorm) and the FlowLM/Mimi benchmarks, now Whistle-shaped. A
      `purego` build tag forces the pure-Go fallbacks on any arch. The new `asm_test.go` compares all six asm routines
      with their fallbacks under float32 rounding bounds (dot/GEMM 2(n+1)·2⁻²⁴·Σ|ab|, axpy one FMA rounding). It passes
      natively on arm64 and under linux/amd64 emulation with AVX2 detected, in both tag variants. The tolerances keep
      their values (1e-4 matmul/linear/softmax, 0 causal_mask, 2e-4 rope/attention/conv1d). The entries of the dropped
      kernels are gone, and `TestKernelTolerancesPinned` pins the rest. A 32-bit overflow in a copied test is fixed.
      Follow-ups (`docs/FINDINGS.md` F-008): `RoPE` is interleaved with caller-built cos/sin tables (θ is the caller's,
      no builder); the `Attention` item needs the half-split layout with θ = 1e5 and GQA 8:2 (`Attention` wants equal
      head counts). The depthwise kernel-9 conv runs on the scalar grouped path. The stem needs a `Conv2D` (stride 2,
      SAME/VALID, groups 1 and 128, HWIO kernels), which is not built yet.
- [x] `ZCRMSNorm`, `rms_unit`
      (2026-10-10) — `internal/needle`: `ZCRMSNorm` (`NewZCRMSNorm`, `LoadZCRMSNorm(src, name)`, `Apply` over rows of
      the scale's width, in place allowed) and `RMSUnit(dst, x, dim)`; mean(x²) accumulates in float64. Against
      `zcrmsnorm.safetensors` with `layer00.norm_in` (zero row, ×1e-5 … ×100 rows, a non-zero-mean row): `y` max abs
      8.9e-8, `rms_unit` 4.8e-7, `rms` 1.2e-7, at most 0.01 of the F-009 bound, cosine 1.000000000000. Unit tests on
      tiny vectors (zero row, in place) need no model (`docs/FINDINGS.md` F-011).
- [x] `Attention`: Q/K/V projections, optional causal taps, per-head q/k ZCRMSNorm, half-split RoPE (θ = 1e5), GQA
      8:2, sigmoid output gate. Masks: causal and none.
      (2026-10-10) — `needle.LoadAttention(src, prefix)` derives heads, KV heads, qk and v from the shapes and has taps
      exactly when `<prefix>q_taps` exists, so `layer00.` is the decoder flavour and `enc00.` the encoder flavour;
      `Forward(x, T, mask, rope)` with `MaskNone`/`MaskCausal` and an optional `*Rope` (`NewRope(48, T, 1e5)`, block-local
      half split; `ops.RoPE` is interleaved and stays unchanged). GQA reads KV head h // 4 (`jnp.repeat`), no copies;
      the core is a per-head loop over `tensor.DotProduct`/`Axpy` (`ops.Attention` needs equal head counts). Every
      intermediate of both fixtures is within the contract, cosine 1.000000000000: `attention_decoder` worst `gate`
      3.9e-6 (0.39 of the bound), `out` 7.0e-4 at |ref| 329 (0.21); `attention_encoder` worst `gate` 4.1e-6 (0.41),
      `out` and `out_rope` 4.9e-4 at |ref| ≤ 1202 (0.04). The `gate` gap is the fixture's own float32 noise (a
      float64-accumulated gate sits 3.4e-6 / 3.8e-6 from it, as F-009's float64 rerun did). Unit tests: hand-computed
      RoPE, the fixture's RoPE tables (1.2e-7), the GQA head map, a float64 transcription of upstream on random
      tiny weights for all four mask/taps/RoPE combinations, exact causality, load and call errors. Gap: no RoPE
      offset or KV cache yet (decoder branch); padding masks are not supported (F-011).
- [x] `HadamardMLP`: Monarch 16×32 Kronecker apply, conditional rank-8 scale, SiLU, the permutations p1/p2. Read the
      permutations from `.cact` 226/227. For safetensors-only loads, either reuse the `.cact` values or port numpy MT19937 +
      `permutation`, and pin both against each other in a test.
      (2026-10-10) — `needle.LoadHadamardMLP(src, prefix)` (`layer00.`, `enc00.hada_0.`, `enc00.hada.`) takes n, the
      16 × 32 blocks and rank 8 from the shapes and reads `hada_p1`/`hada_p2` from the `Source`: `.cact` 226/227, or
      the numpy MT19937 port the safetensors backend derives them with, checked to be permutations. `Forward(x, T)`
      applies each Kronecker stage as aᵀ · Z · b (24,576 multiply-adds per row and stage instead of 262,144;
      `TestKronApplyMatchesDense` against the dense `np.kron` matrix). Against `hadamard_mlp.safetensors`: `out` 2.0e-3
      at |ref| 1481, worst `cond` 0.14 of the bound, cosine 1.000000000000 for `cond_logits`, `cond`, `z1`, `a2`,
      `z2`, `z3` and `out`. Permutation pin: `TestHadaPermsFixture` (fixture `p1`/`p2` = `cact.NumpyPermutation`, no
      model needed) and `TestHadaPermsBothBackends` (layer 0 through `.cact` and safetensors, identical and equal to
      the fixture); `internal/weights` `TestWhistleCrossFormat` compares all derived records too.
- [x] `MHC`: lane expand/collapse, pre/post/res heads, a numerically stable log-domain 20-iteration Sinkhorn.
      (2026-10-10) — `needle.LoadMHC(src, prefix, layer)` (`""` decoder, `enc.` encoder) slices the layer's heads from
      the stacked `mhc_*` tensors (φ lane-major, so each head is one GEMM) and derives the lanes and the width from the
      shapes. `Forward(stream, T, block)` takes the block as a function, so the encoder's Conformer block fits later.
      `ExpandLanes`/`CollapseLanes` are `Stack`'s broadcast and lane mean. The Sinkhorn is log-domain float32 with
      max-shifted logsumexp; it stays finite at logits of 1e30. Discovery: 20 iterations do not converge on Whistle's
      `h_res` (logits up to 70.8). The columns sum to 1, the rows deviate by up to 6.7 % (layer 0), so the count is
      part of the model: 19 or 21 iterations miss the fixture's `h_res` by about 3e-3, 300 times the bound. Against
      `mhc.safetensors`: every head and the stream of both layers, plus `collapsed` and `final`. Worst
      `layer1.stream_out` 0.18 of the bound (5.8e-4 at 317), `h_res` ≤ 0.14, cosine ≥ 0.999999999999. Unit tests:
      column sums, float64 transcriptions of `_sinkhorn` and of `advance`, large logits, the lane offsets, and the
      expand/collapse bookkeeping (`docs/FINDINGS.md` F-013).
- [x] `Engram`: uint32 multi-head hashing over the token history, table lookup, key/value projections, dilated
      4-tap conv, sigmoid gating against `rms_unit(x)`.
      (2026-10-10) — `needle.LoadEngram(src, "engram0.", cfg.Orders, cfg.Tables)`, `Forward(tokens)` → `*EngramKV`;
      the gate is applied inside `Block.Forward(…, engram)`. The hash is plain Go `uint32` arithmetic (seed stride =
      hashes, since `engram_seed_heads` = 0) and is pinned to seven hand-computed wraparound cases. Incremental form:
      `IndicesAt(dst, hist)` reads the last `MaxOrder` = 3 tokens, and a rolling window reproduces the sequence's
      indices at every position. The value taps reuse `applyTaps`, which now takes a dilation (3). Against
      `engram.safetensors`: all 96 `indices` exact, `ngram_ok`/`tap_ok` as the causal rule, `e` exact, `k`, `v_raw`,
      `v`, `alpha`, `x_out` ≤ 0.12 of the bound, cosine 1.000000000000. Gap: the rolling `v_raw` history for the
      taps during decoding belongs to the Phase 5 cache (F-013).
- [x] `scripts/needle_fixtures.py`: run each Flax module with real Whistle decoder weights on random inputs and save
      the inputs and outputs. **Accept:** every block matches within 1e-5 (max abs) / cosine ≥ 0.99999.
      (2026-10-10) — partial: the fixture half exists. `just needle-fixtures` imports upstream's Flax modules and binds
      them to the FP32 safetensors parameters by Flax path, with every module leaf present at the exact shape. It runs
      in float32 (`highest` matmul precision, `flash=False`, `quant=False`) and writes seven files to
      `testdata/needle/` (1.97 MB in total, 89 tensors, no weights): `zcrmsnorm`, `attention_decoder` (layer 0: taps,
      causal mask, RoPE 1e5, GQA, gate), `attention_encoder` (encoder layer 0 weights, no mask or taps, with and
      without RoPE), `hadamard_mlp`, `mhc` (layer 0 from the lane expand to the collapse + `final_norm`, and layer 1
      on four distinct lanes, with h_pre/h_post/h_res and the block output), `engram` (site 0 / layer 3, I64 ids and
      hash indices, the gate) and `block` (layer 0). Every intermediate either is a submodule output or reproduces
      the module output exactly. Two runs are byte-identical; safetensors writes `__metadata__` in hash-map order, so
      the script sorts it. All files load with `internal/safetensors`. `testdata/needle/README.md` is the contract.
      Discovery: block outputs reach 1.5e3 and the float32 fixtures lie up to 1.3e-3 from a float64 rerun, so a flat
      1e-5 max abs is unreachable. The contract uses 1e-5 · max(1, max |ref|) per tensor plus cosine ≥ 0.99999; the
      measured float32 noise is ≤ 3.9e-6 on that scale (`docs/FINDINGS.md` F-009). (2026-10-10) — the Go blocks
      `ZCRMSNorm`/`rms_unit`, `Attention` (decoder and encoder fixtures, with and without RoPE) and `HadamardMLP` pass
      `zcrmsnorm`, `attention_decoder`, `attention_encoder` and `hadamard_mlp` within this contract, every
      intermediate, worst 0.41 of the bound (F-011). Pending: `MHC`, `Engram` and `Block` against `mhc`, `engram`
      and `block`. (2026-10-10) — every fixture file passes the contract with every tensor. `MHC`, `Engram` and
      `Block` (upstream's decoder `Block` without cross-attention: the engram add, `norm_in` → attention →
      σ(attn_gate)·`post_norm`, `pre_hada` → HadamardMLP) close the last three. Worst per file, over
      `go test -v -run Fixture ./internal/needle/` (FP32 safetensors weights, Apple M5 Pro):

      | fixture             | tensors | worst (of bound)         | max \|Δ\|  | bound   | min cosine     |
      | ------------------- | ------- | ------------------------ | ---------- | ------- | -------------- |
      | `zcrmsnorm`         | 3       | `rms_unit` 0.01          | 4.8e-7     | 3.6e-5  | 1.000000000000 |
      | `attention_decoder` | 15      | `gate` 0.39              | 3.9e-6     | 1.0e-5  | 1.000000000000 |
      | `attention_encoder` | 11      | `gate` 0.41              | 4.1e-6     | 1.0e-5  | 1.000000000000 |
      | `hadamard_mlp`      | 7       | `cond` 0.14              | 7.9e-6     | 5.6e-5  | 1.000000000000 |
      | `mhc`               | 24      | `layer1.stream_out` 0.18 | 5.8e-4     | 3.2e-3  | 0.999999999999 |
      | `engram`            | 7 + I64 | `v_raw` 0.12             | 5.3e-5     | 4.6e-4  | 1.000000000000 |
      | `block`             | 7       | `h` 0.18                 | 9.9e-5     | 5.6e-4  | 0.999999999999 |

      The I64 tensors are exact (`engram.indices` 96/96, `hadamard_mlp.p1`/`p2`), and `-tags purego` gives the same
      picture (worst 0.40). No fixture was regenerated. F-009's remark that `h_res` does not matter on the lane
      expand is only approximate, since 20 Sinkhorn iterations leave the rows unconverged; the fixture is right as
      it is, and `layer0` pins the Sinkhorn too (F-013).

## Phase 3 — Front end — ✅ DONE (2026-10-10)

- [x] WAV decode exactly as in `whistle.py`: 8/16/24/32-bit, channel averaging, 8-bit offset 128.
      **Accept:** the upstream matrix (16000/1/2, 8000/1/2, 44100/2/2, 16000/1/1, 22050/2/3, 48000/1/4). A 220 Hz tone
      at 0.3 gives 8000±1 samples, a peak of 0.28–0.31 and 108–111 zero crossings. (2026-10-10) —
      `internal/frontend`: `ParseWAV` walks RIFF chunks as Python's `wave` does (reads bounded by the RIFF size,
      word-aligned chunks, width from bits per sample, last fmt before data) and rejects what it rejects with typed
      errors: `*FormatError` for non-PCM (IEEE float included) and `ErrNotRIFF`, `ErrNotWAVE`, `ErrMissingChunk`,
      `ErrDataBeforeFmt`, `ErrTruncated`, `ErrChannels`, `ErrSampleWidth`. `WAV.Mono` is `_read_wav`'s float64
      expression rounded to float32, bit for bit. `DecodeWAV`/`ReadWAV` resample anything not at 16 kHz.
      `TestUpstreamWAVMatrix` writes the six files like `_write_wav` (bytes pinned to Python's writer): 8000 samples
      each, peaks ±0.29688 to ±0.30000, 109 or 110 crossings; upstream with soxr gives 109 everywhere. The 30 s
      limit is not part of this item. All 32 committed clips decode to their manifest sample counts (F-010).
- [x] Resampling with `algo-dsp/dsp/resample` (windowed-sinc / polyphase). Measure the deviation from soxr HQ on test
      tones and document it. (2026-10-10) — algo-dsp v0.12.4 `NewRational` with taps per phase scaled to
      `ceil(128 · max(up, down) / up)`, cutoff 0.96, Kaiser β 11. The group delay is removed by prepending zeros
      instead of `ProcessAligned`'s linear interpolation (−14 dB at 7 kHz). The residual is 0 for odd prototypes
      (48 kHz), 0.0011 output samples from 22.05/44.1 kHz and half a sample from 8 kHz. Fixture
      `testdata/frontend/soxr_hq.safetensors` (`just resample-fixture`, soxr 1.1.0 HQ): tones 220 Hz–9.5 kHz, a chirp
      and noise from 8/22.05/44.1/48 kHz, plus soxr's levels for float tones at 8.1–9.5 kHz. Against soxr: passband
      tones 56–80 dB SNR (residual-limited) from 22.05/44.1 kHz and 129–136 dB from 48 kHz; 7 kHz 49.9 / 112 dB;
      gain within 0.0065 dB; aligned passband ≥ 66.9 dB everywhere, 8 kHz included; noise's 7.4–8 kHz transition
      band 25–29 dB; aliases −87 dB at 8.1 kHz and below −110 dB from 8.2 kHz (soxr −130 to −144 dB). 30 s takes
      29–121 ms (soxr 0.5–2 ms). The presets lose 1.7–5.9 dB at 7 kHz, and length 64 lets 8.1 kHz alias at −29 dB.
      Rates with `down` > 16384 (e.g. 47999 Hz) use algo-dsp's ratio approximation; rates below 1 kHz are rejected
      (F-010).
- [x] STFT: 400-sample window zero-padded to 512, hop 160 → 257 bins → power → mel from `.cact` tensor 679.
      (2026-10-10) — `internal/frontend` `New(bank, Options)` → `Spectrogram`/`LogMel` (`[frames, 80]`, time × mel).
      Framing runs on algo-dsp `dsp/stft` `STFT32` with a 512-long custom window holding the `dsp/window` Hann at offset
      56 (centred, frame i over `[160i − 200, 160i + 200)`) or 96 in frame i+1 (`none`, over `[160i, 160i + 400)`;
      algo-dsp's own `PadNone` would lose the last frame). Power in float64 from the complex64 bins; the mel product
      accumulates in float64 over each channel's non-zero bins. `LoadMelBank` reads `audio.mel_filters` `(257, 80)`
      from a `weights.Source`; the model-gated `TestMelBankBackends` finds `.cact` and safetensors bit-identical to each
      other and to `cact.SlaneyMel`. Verified for 2 windows × 3 paddings against a float64 direct DFT (max |Δ|/peak
      ≤ 3.2·10⁻⁷, max relative error ≤ 4.7·10⁻⁵ in bins within 60 dB of the frame peak) and a numpy `rfft` fixture
      (`just mel-fixture` → `testdata/frontend/stft.safetensors`, 183 KB, byte-identical reruns; ≤ 2.5·10⁻⁷ and
      ≤ 3.7·10⁻⁵), pinned at 1e-6 / 2e-4. Frame counts: centred `n/160` rounded down (3000 for 30 s), `none`
      `1 + (n − 400)/160` (2998). Discovery: three round-up stride-2 halvings of `floor(n/160)` reproduce
      `needle_embed`'s length for all 32 corpus clips and F-001's seven lengths (39/39); `ceil(n/160)`, `n/160 + 1` and
      the `none` count do not. F-001's `(n + 640)/1280` misses 7 corpus clips and is corrected. Smoke test: all 32
      clips finite at `[n/160, 80]` under three hypotheses; 30 s takes about 5 ms. Reflect padding needs more than 256
      samples (`ErrTooShort`), below the engine's 640-sample minimum for one frame (`docs/FINDINGS.md` F-012).
- [x] Hypothesis set for what the docs leave open: window (Hann periodic or symmetric), centre padding (reflect, zero
      or none), power or magnitude, log flavour (`ln`, `log10`, `log(x + eps)` or `clamp` à la Whisper),
      normalisation (per mel channel over time, global, or none) and the "250–3500 Hz band-limit". Resolve them in Phase 4
      against the encoder oracle, because no mel-level oracle exists.
      (2026-10-10) — set implemented and tested; **the resolution stays open for Phase 4**. `frontend.Options` has one
      typed enum per axis (zero value invalid, no default): `Window`, `Padding`, `Spectrum`, `Log` (`ln(max(x, 1e-10))`,
      `log10`, `ln(x + 2⁻²⁴)`, Whisper's `max(y, max(y) − 8)` then `(y + 4)/4`), `Norm` (`(x − mean)/(std + 1e-5)`,
      n−1 std, per channel or global), `Band` (`full`; `bins`: FFT bins outside [250, 3500] Hz zeroed before the bank,
      an ideal pre-filter; `channels`: mel channels with centres outside zeroed, 6–58 stay) and `Preemphasis` (none or
      0.97). Pre-emphasis was added because NeMo's Conformer preprocessor matches every published constant
      (512/400/160, Slaney mels with Slaney norm, dither 1e-5, "normalised per channel"); presets `NeMo()` and
      `Whisper()`. `Variants()` gives all 864 combinations in a stable order, with `String`/`ParseOptions` keys. A unit
      test per axis shows each value doing what it says; all 864 run finite on a real clip. Not implemented: a bank
      re-derived over 250–3500 Hz (the shipped bank is full band) and a time-domain band-pass (no stated shape)
      (`docs/FINDINGS.md` F-012).

## Phase 4 — Encoder (reverse engineering) — ✅ DONE (2026-10-10)

- [x] Stem: a 2-D conv with candidate strides (2,2) on all three convs vs. stride only on the depthwise stages; padding
      SAME or VALID; activation (SiLU, GELU or none) after each stage. Pin the frame count: 3000 → 375.
      (2026-10-10) — `internal/encoder` stem on the new `ops.Conv2D` (NHWC/HWIO, strides, per-side padding, groups;
      float64-reference and hand-computed tests, `ops.SamePadding` = lax SAME). Stride-2 on all three convs: stride
      only on the depthwise stages (20 mel positions) and VALID (9) do not fit `stem/out`'s 1280 = 10 × 128 and are
      rejected at load (`ErrRuledOut`); 3000 mel frames → 375 and the engine's frame rule hold (`TestStemFrames`,
      `TestEmbed30s`). Chosen by score (inner clips, `.cact`, leader 0.999819): zero padding 1 on every side, torch
      style (lax SAME −0.243827); kernel [time, mel] (swapped −0.509124); channel-major NCHW flatten, mel fastest
      (NHWC −0.713620); SiLU after `stem/w` (GELU −0.000569, none −0.063222), nothing after the depthwise convs (SiLU
      −0.092369), SiLU after the pointwise convs (GELU −0.027121, none −0.815920); no input scale (×√512 −0.837042)
      (`docs/FINDINGS.md` F-016).
- [x] Encoder block candidates:
  - FFN residual scale 1 or ½ (macaron)
  - conv module: GLU over pw1's 1024 channels or not; SiLU after dw; any normalisation between dw and pw2
  - attention: full non-causal; RoPE in the encoder or not
  - where `pe_gate` applies: a sinusoidal or RoPE mix-in gated by σ(pe_gate), or on the stem output
  - how mHC wraps the block (same as the decoder, `enc_lanes` 4)

  (2026-10-10) — everything but `pe_gate` is resolved by score (F-017). FFN ½ (1: −0.622692); GLU a·σ(b) over pw1's
  halves ("no GLU" ruled out by the 512-wide `dw`; a·silu(b) −0.403596, σ(a)·b −0.978467); SiLU after `dw` (GELU
  −0.005362, none −0.820849); `conv_out_norm` between `dw` and the SiLU (after it −0.185564, after `pw2` −0.896173);
  centred kernel-9 `dw` (causal −0.907688); macaron order (conv first −0.415660); full non-causal attention with
  half-split RoPE θ = 1e5 (θ 1e4 −0.029625, none −0.221335); mHC exactly as the decoder, 4 lanes (plain stack
  −0.678340); shared `hada_p1/p2`, so U5 is settled (split −0.980344, identity −0.977999). On speech, σ(pe_gate) =
  0.0022 puts a sinusoid −70 dB below the stem output, so the corpus cannot decide `pe_gate`.
  (2026-10-10) — `pe_gate` decided by synthetic probes: **unused** (F-019). `just oracle-probes` records
  `needle_embed` for 30 signals from `tools/oracle/probe` (silence 0.32–30 s, DC, quiet tones and bursts, noise, a
  −20 → −140 dB decay, impulses at sub-frame offsets and falling levels): 1217 frames, 2.5 MB in
  `testdata/oracle/probes` with a manifest of every generator parameter and each signal's SHA-256. On digital silence
  the stem output is exactly 0, so a position term would be the stack's only input. `pe=none` reproduces the engine's
  silence rows at 0.999946 (every length ≥ 0.999916, the engine's own reproducibility). Every tested placement loses
  at least 0.0088 (`TestProbePEGate`): sinusoid on the stem input, before or after `stem/out`, on every block's input,
  each as `[sin | cos]` or interleaved; the RoPE mix; and an ungated control. The best of them, interleaved after
  `stem/out`, scores 0.991178 (0.977333 on 10 s). On the 32 clips they all tie within 6e-6 except the control and the
  RoPE mix. So the encoder has no position term beyond RoPE (silence: no RoPE 0.942122, θ 1e4 0.985391).
  `TestBestMatchesProbes` pins it.

- [x] Hypothesis harness `internal/encoder/probe_test.go` (build tag `oracle`): enumerate the combinations, run the
      encoder on the fixture clips and score each by the mean per-frame cosine against `needle_embed`. Prune greedily by stage.
      (2026-10-10) — `just encoder-probe`. `encoder.Hypothesis` = `frontend.Options` + 19 typed encoder axes with
      named values (the PLAN candidates plus kernel axis order, flatten order, input scale, GLU forms, depthwise
      padding, module order, mHC on/off, permutation sets, output tap). The harness rules out by shapes and by frame
      count, caches the mel and the stem output, and runs pairs on all cores (32 clips in about 0.3 s). Baselines:
      the mean engine frame of the other clips scores 0.2349 and a random mel projection −0.0048, so about 0.23 means
      no signal. Search: 7 stages (stem geometry, stem activations, front end log/norm/spec, conv module, block
      structure, positions/output, front end window/padding/band/pre), each the full product around a beam of 4.
      Then coordinate descent with escalating moves (single axes, then axis pairs, then whole stages; any change
      restarts from single axes) until no level moves the leader, then a per-axis sensitivity table. On `.cact`
      weights it goes 0.2485 → 0.7749 (stem geometry) → 0.9992 (RoPE) → 0.9998 (descent, `log`) in about 3 min.
      Started from a hypothesis wrong on all 26 axes (0.0179), it reaches the same leader in 7.5 min; without the
      pair and stage moves it stalled at 0.753. Fast regression without the tag: `TestBestMatchesEngine` (F-014).
- [x] Confirm what `needle_embed` actually returns: the shape, and whether it is before or after `final_norm` and the lane mean.
      (2026-10-10) — `[frames, 512]`, `enc.final_norm` of the lane mean. Every one of the 2205 engine rows divided by
      (1 + the `.cact` FP16 `enc.final_norm` gain) has RMS 1 within 1.8e-7, which only a ZCRMSNorm output has
      (`TestEmbedIsFinalNormOutput`). Before `final_norm`, the lane mean scores −0.396920 (F-015).
- [x] **Accept:** mean per-frame cosine ≥ 0.99 against the engine on all fixture clips. Every resolved unknown is
      written to `docs/FINDINGS.md` with its metrics.
      (2026-10-10) — `encoder.Best()` with the dequantised `.cact` weights: mean per-frame cosine 0.999539 over the 32
      clips, every clip ≥ 0.992932 (fleurs-fr_fr-10174413140250646001; next 0.9981, all others ≥ 0.9992). The float
      safetensors weights give 0.921019 (min 0.881934), so the engine runs the quantised weights. The frames below
      0.995 are near-silent (peak |x| < 3e-3): the log floor for quiet frames is the remaining gap. Front end (U1)
      decided as far as the embeddings allow: power spectrum, Whisper log clamp, per-channel norm, full band (the
      "250–3500 Hz" mask is ruled out), no pre-emphasis; window and reflect vs zero padding tie (F-018). Findings
      F-014 to F-018.
      (2026-10-10) — the near-silent gap was the Whisper clamp's hard edge (F-020). `Best()` now takes
      `log=soft-76db`, log10(x + 10^−7.6 · max x): mean 0.999828, every clip ≥ 0.999033 (ls-clean-260-123440-0018);
      the old worst clip rises to 0.999713, and 7 frames stay below 0.995 instead of 13 (worst 0.9558 instead of
      0.6753). Safetensors weights: 0.921134 (min 0.881205). The absolute floor never binds on speech: the engine is
      scale invariant on tones, bursts, noise and DC over 2–4 decades, down to 1e-5. Every soft depth from 7.35 to
      8.2 beats the hard clamp, and the fitted depth holds on held-out halves. Zero centring is decided by the DC probes (reflect
      0.8775); dither is ruled out by silence (0.915996). The window still ties (Δ 1e-6). Open, below the target:
      the per-clip optimal knee depth varies (7.5–8.55), and impulses below about 1e-5 behave as if under an absolute
      floor that tones do not show. Findings F-014 to F-020.

## Phase 5 — Decoder + greedy decoding — ✅ DONE (2026-10-10)

- [x] Cross-attention: per-layer `cross_norm → q`, encoder `k/v` projected once per clip and cached, q/k norm,
      sigmoid gate, `post_cross_norm`, σ(`cross_gate`) residual. Decide where it sits in the block (after
      self-attention is the most likely) by fixture accuracy.
      (2026-10-10) — between the self-attention and the MLP, inside the mHC block function
      (`needle.Block.Extend` with `Hooks.AfterAttn`):
      `x += σ(cross_gate)·post_cross_norm(out_proj(softmax(q̂k̂ᵀ/√48)v · σ(gate_proj(cross_norm(x)))))`, 8 heads of
      48/64, per-head q/k ZCRMSNorm, no RoPE, over `needle_embed`
      (`enc.final_norm` of the lane mean). Decided by teacher-forced mean log-probability of the engine transcripts,
      32 clips, `.cact` (leader −0.1302, top-1 0.977): after the MLP −0.1484 (language 20/32), before the
      self-attention −1.0900, outside mHC −1.5188; q/k norm none −0.2098, without gain −0.7983, gain = scale −2.7037;
      gate from the un-normed residual −0.0863, none −0.7254, silu −1.4487; no post norm −4.2525, raw `cross_gate`
      −8.9044; RoPE in the cross-attention −0.3912; memory before `enc.final_norm` −3.6391. σ and tanh of
      `cross_gate` (22.0–22.8) are both 1 and tie exactly (`docs/FINDINGS.md` F-021).
- [x] KV cache for self-attention, plus a rolling history for the Q/K/V taps and the engram n-grams.
      (2026-10-10) — `needle.KVCache` (keys/values of every position plus the last 2 raw Q/K/V rows for the taps)
      with `Attention.Extend`, and `needle.EngramCache` (last 2 tokens, last 9 raw value rows for the dilated taps)
      with `Engram.Extend`; `decoder.State` holds one per layer and site. One call equals `Forward` exactly; row by
      row within 6e-7 on random weights, 6.1e-5 at |ref| 379 on the real layer 3 with engram
      (`TestWhistleDecoderIncremental`). `TestIncrementalMatchesFullRun`: a greedy decode through the caches and a
      one-pass re-run agree on all 38 × 8199 logits within the F-009 contract and on every argmax (F-022).
- [x] Tied output head (logits = x · Eᵀ) and `sqrt(d)` input scaling.
      (2026-10-10) — embedding × √512 (× 1: −4.7279), `stack/final_norm` of the lane mean (without: −37885), tied head
      x · Eᵀ over all 8199 pieces (÷ √512: −7.8698, same top-1) (F-023).
- [x] Prompt-format candidates: `[bos]`, `[bos, <speech>]`, `[bos, <|lang|>]`, … Pick the one where greedy decoding
      reproduces the engine's language token and the first words.
      (2026-10-10) — `[<s>]`, then `<|lang|>`, the text without a leading `▁`, `</s>`. `[<s>, <speech>]` −0.4236 and
      language 3/32, `[<pad>]` −0.6570, `[<speech>]` −0.7252, `[</s>]` −0.8315, the other `<speech>`/`</speech>` forms
      worse; `▁` before the first word −0.3202. The embedding rows agree: `<speech>`, `</speech>`, `<pad>` and all 24
      chat markers sit at the untrained RMS 8.3, `<s>`, `</s>` and the language tokens at 2.5–3.5 like text (F-023).
- [x] Language detection: argmax (or softmax over 8192–8198) at the language step; a fixed language forces the token.
      (2026-10-10) — `decoder.Greedy(enc, frames, lang)`: argmax over the 7 language tokens after `[<s>]`, or the
      forced one; then argmax until `</s>` or 320 tokens, decoded with `internal/tokenizer`. The language token is
      also the unrestricted argmax (rank 0) on 32/32 clips with either weight source (F-024).
- [x] **Accept:** greedy transcripts are sensible on all fixtures and the language matches the engine on 100% of them.
      (2026-10-10) — `.cact` weights: language 32/32, text identical on 21/32, every clip stops at `</s>`, mean WER
      0.0496 / CER 0.0188 against the engine's text, max CER 0.1295 (fleurs-pl_pl-10013271501230525766, where the
      engine's own text has non-words). Safetensors: language 32/32, exact 10/32, mean CER 0.0543, max 0.2235. Search
      harness `just decoder-probe` (beam 4 over four stages, then single-axis/pair/stage descent; 2.6 min from the
      prior, 3.6 min from a start wrong on all 12 axes, both end at `decoder.Best()`); fast regression
      `TestGreedyMatchesEngine` (three clips exact, no tag) (F-021 to F-024). Re-run on the F-020 encoder
      (`log=soft-76db`) after the merge: language 32/32, exact 22/32, every clip stops at `</s>`, mean WER 0.0583 /
      CER 0.0228, max CER 0.1655 (fleurs-pl_pl-10013271501230525766).

## Phase 6 — Engine-parity decoding — ✅ DONE (2026-10-10)

Split into sub-phases with disjoint package scopes so that independent work can run in parallel. 6.1, 6.2 and 6.5 have
no prerequisites; 6.3 and 6.4 need the 6.1 fixtures; 6.6 integrates everything and carries the phase **Accept**.

### 6.1 Decoding oracle corpus (`tools/oracle`, `testdata/oracle/`, `justfile`)

- [x] Depth runs: engine transcripts at `--audio-depth 2..7` for the 32 fixture clips (`testdata/oracle/depth/`).
      (2026-10-10) — `just oracle-depth`: 192 transcripts in `depth/transcripts.json` (`<clip>.d<N>`). Each run
      re-checks that `--audio-depth 8` equals `testdata/oracle/<clip>.json` byte for byte (32/32); 1 is refused, 9
      fails to load. Text equal to the full depth: d2 7, d3 11, d4 11, d5 16, d6 15, d7 19 of 32 (F-025).
- [x] Keyword runs (`testdata/oracle/keywords/`): per case a keyword file and the transcript with and without it. Cases:
      a word the engine already gets right, a word it misses against the reference transcript, a wrong but plausible
      word, a multi-word phrase.
      (2026-10-10) — `just oracle-keywords`: 53 cases over 19 clips in all 7 languages (15 present, 16 missed, 8
      distractor, 12 phrase, 2 lists), each a `<case>.txt` plus its transcript in `transcripts.json`; without is the
      committed `testdata/oracle/<clip>.json`, re-checked byte for byte. Each kind is checked against the reference and
      the engine's text. The text changes on 24 cases; 9 of 16 missed words appear, 2 of 8 distractors win (F-025).
- [x] Silence sweep (`testdata/oracle/silence/`), signals from `tools/oracle/probe`: digital silence of several
      lengths, white/pink noise at several levels, a speech clip attenuated in steps down to −90 dBFS, speech padded
      with long silence. Record whether the engine returns the empty JSON.
      (2026-10-10) — `just oracle-silence`: 320 signals of `probe.SilenceSet` (new kinds pink, step, speech with gain,
      padding, noise, 16-bit grid); every parameter exact in the manifest plus the samples' SHA-256. 159 return exactly
      the empty JSON; `needle_transcribe` agrees with the CLI on all 320. Silence of any length and steady white or pink
      noise at any level: empty. Empty until a noise step of +2.65 dB (+2.70 opens) at −80, −40 and −10 dBFS alike; a
      +10 dB section opens it from 560 samples to 90 % of the clip; 16-bit speech empty from −87.5 dB (en) / −83 dB
      (de); SNR empty from −10 dB (en) / −15.25 dB (de); exact float gain and padding never empty (F-025).
- [x] Limits (`testdata/oracle/limits/`): 480,000 vs 480,001 samples (the exact error text), 0 and a handful of samples.
      (2026-10-10) — `just oracle-limits`, CLI and `needle_transcribe`: 480,000 samples transcribe; 480,001 give CLI
      exit 1 `needle: audio limit is 30 s` and rc −1 with `needle_last_error` `audio limit is 30 s`, for silence too
      (the length check precedes the silence gate). 0 samples: CLI exit 1 `needle: <file> holds no audio`, library rc 0
      and the empty JSON. 1–161 samples of speech: the empty JSON; 1279–1281: "you" (F-025).
- [x] LibriSpeech test-clean subset (`testdata/oracle/librispeech/`): about 200 utterances, seeded and spread over the
      speakers, with ids, SHA-256, reference text and the engine JSON. The audio stays git-ignored under
      `models/corpus/` and a `just` recipe fetches it.
      (2026-10-10) — `just oracle-librispeech-audio` streams all of test-clean (archive SHA-256 pinned) and keeps per
      speaker the 5 utterances ≤ 30 s with the smallest SHA-256 of `"go-whistle 6.1/<id>"`: 40 × 5 = 200, 1.5–28.4 s,
      WAVs in `models/corpus/librispeech/`, `subset.json` with ids, SHA-256 and references. `just oracle-librispeech`
      fetches when needed and records the engine JSON; engine WER 221/4167 = 5.30 % with upper-casing and punctuation
      removal only (F-025).
- [x] **Accept:** every `just oracle-*` recipe regenerates its fixtures deterministically (a re-run leaves no diff), and
      the repo grows by less than 2 MB.
      (2026-10-10) — Each new recipe run twice, `git status` clean after the second run; `oracle-clips`, `oracle-run`
      and `oracle-probes` re-run with no diff; the LibriSpeech audio re-fetched byte-identically. The five sets hold
      1,368,294 bytes in 63 files, `du -sk` 1568 (one `transcripts.json` per set instead of 843 small files, which
      took 3.8 MB in 4 KiB blocks) (F-025).

### 6.2 Beam search (`internal/search`, `internal/decoder`, `internal/needle`)

- [x] `decoder.State.Fork`: copy-on-write KV per beam; the encoder memory (cross K/V) is shared by all beams.
      (2026-10-10) — `needle.KVCache.Fork` shares the prefix's keys and values and caps the fork's slices, so the
      first append copies them; the tap and engram histories are replaced on every `Extend` and shared as is; the
      `decoder.Memory` pointer is shared. Three interleaved branches each equal a fresh state fed their whole
      sequence (`TestForkMatchesFullRun`, worst max |Δ| 4.6e-5 at |logit| 102; `TestAttentionFork`,
      `TestEngramFork` ≤ 2.4e-7) (`docs/FINDINGS.md` F-026).
- [x] Beam search: 5 beams, max 320 tokens. Candidates for the length normalisation (`logp / len`, `logp / len^α`,
      GNMT), the EOS rule (all beams finished, best finished beats best alive, K finished) and whether the language
      step is beamed or argmax (U8).
      (2026-10-10) — `search.Beam(d, mem, search.Default())`, with per-token log-probs, the score and an `Adjust`
      hook for 6.3. `just search-probe` scores 1218 variants (none / len / len^α / GNMT with α 0.2–3 × L of text,
      text + EOS or the whole sequence × all-finished / best-beats-alive / 5-finished / HF × two EOS admission rules ×
      argmax / scored / beamed language) on a shared prefix memo in 3.6 s. Winner: Whisper's search, Σ log p / text
      tokens, stop at 5 finished, every EOS ranked above the last live beam admitted, language by argmax: 31/32
      exact. The best alternatives: GNMT α 1.2 31/32 (higher CER); all-finished, best-beats-alive, HF 29; EOS
      admitted only among the step's best 5: 30. 12 variants tie with the winner (L with EOS, any language step) on
      all 32 transcripts. Width: 1 → 22, 3 → 29, 5 → 31, 6 → 29, 10 → 27 (F-026).
- [x] Move `editDistance`/`wer`/`cer` from the decoder tests into `internal/eval` for reuse.
      (2026-10-10) — `eval.EditDistance`, `eval.Words`, `eval.WER`, `eval.CER` with unit tests; the decoder probe
      uses them.
- [x] **Accept:** the variant is picked by evidence on the 32 clips (`.cact` weights); target ≥ 31/32 transcripts equal
      to the engine's, every remaining mismatch explained by its log-probability margin.
      (2026-10-10) — 31/32 exact (greedy 22/32), mean WER 0.0208 / CER 0.0085 against the engine's text; the real
      search equals the probe's memo on 32/32. The mismatch, `fleurs-pl_pl-10047713455444904767` (neither text is
      Polish), is a near-tie: the engine's text in its most likely spelling scores Σ log p −14.2704 (−0.2854 per
      text token) under our model against our −15.6215 (−0.2947), and its prefix leaves our beam at text token 43,
      0.1350 nats behind the fifth live beam. Beam 61 ms per clip, greedy 46 ms. Fast regression
      `TestBeamMatchesEngine` (three clips greedy misses, exact, no tag) (F-026).

### 6.3 Keyword biasing (`internal/search`, `internal/tokenizer`; needs 6.1)

- [x] Tokenise the keywords, build an Aho-Corasick automaton over token ids, add a bonus as a match advances and revoke
      it on failure. Tune the weight and the revocation against the 6.1 keyword fixtures (U9).
      (2026-10-10) — `search.NewKeywords(tok, keywords, search.DefaultBias())`, whose `Adjust` plugs into
      `Options.Adjust`; `search.ParseKeywords` reads the engine's newline format. `just keywords-probe` scores 3846
      variants in three grids (weight per token or per character, revocation, completion bonus, spellings, phrase
      splitting, a renormalised logit bias, the final pick by the model, five search rules) on a memo that is exact for
      biased searches; the real search equals it on 53/53. Winner: 2 nats per matched token, all of it revoked when a
      partial match fails, +1 on completion, the keyword after a space and bare, as written and with its first letter
      upper- and lower-cased, on Whisper's search unchanged: 45/53. Alternatives: no revocation 42, half 44; per
      character 43; split phrases 43; completion bonus alone 34; logit bias 29; model-only final pick 41; other search
      rules 44 (and 47/53 without keywords). `internal/tokenizer` needed nothing new (F-029).
- [x] **Accept:** at least 85 % of the keyword fixture cases equal the engine's transcript, and F-029 explains every
      mismatch by its log-probability margin (a near-tie), a miss the case shares with the run without keywords, or a
      documented search miss.
      (2026-10-10) — reworded by the user's decision; was: "transcripts with keywords equal the engine's on the keyword
      fixtures at the same rate as without". Before the PR #23 review fixes none of the 3846 grid variants reached it
      (best 45/53 against 50/53); the grids were not re-run after the fixes, which bring `DefaultBias` to 46/53. Met
      at 46/53 (86.8 %): of the 7 mismatches, 3 are the Polish clip missed without keywords, 3 near-ties (0.0036,
      0.0125, 0.0247 nats per token) and 1 search miss (`de-paare-missed`, a follow-up in Phase 10).
      (2026-10-10) — with keywords 45/53 (84.9 %) against 50/53 (94.3 %) without on the same cases; the
      reference rate is 31/32 (96.9 %, F-026). Per kind, with / without: present 14/14, missed 13/15, distractor 7/8,
      phrase 10/11, list 1/2 of 15/16/8/12/2. Of the 8 mismatches, 3 are on the Polish clip we already miss without
      keywords; 3 are near-ties or small model gaps (0.0036, 0.0125, 0.0247 nats per token); `de-diese-paare-phrase` is
      a model gap (the engine writes a non-keyword our model puts 2.4 nats lower); `de-paare-missed` is a search miss:
      weaker hypotheses with the shared keyword bonus end one token earlier and fill Whisper's 5 finished slots. No
      variant reaches 46. Fast regression `TestKeywordsMatchEngine` (F-029).
      (2026-10-10) — after the PR #23 review fixes (suffix completions, order-independent bonus, bare spellings
      anchored at the text start, byte pieces under `PerChar`) 46/53 (86.8 %): the anchoring makes
      `de-diese-paare-phrase` equal; the Polish pair changes text but still differs (F-029).

### 6.4 Depth ladder (`internal/decoder`; needs 6.1)

- [x] `Depth` option: run the decoder layers given by the bisection order 0, 7, 3, 5, 1, 2, 4, 6, truncated to N.
      Port `_ladder_layer_order` (`architecture.py`).
      (2026-10-10) — `decoder.LadderOrder`, `decoder.LadderLayers` and `(*Decoder).WithDepth(n)`: upstream's
      sub-network (`Stack` with `exit_depth`, `ladder_slice`). The chosen layers run in order with their own weights,
      the others pass the stream on, the i-th active layer takes mHC lane i mod 4 (`needle.MHC.WithLane`), an engram
      site acts only at an active layer, the final norm stays; weights and `Memory` are shared, depth 8 equals the
      decoder logit for logit. Lanes by position beat the layer's own lane at every depth (d2–d7: 24/27/29/29/28/24
      against 13/10/15/15/22/23 of 32) (F-030).
- [x] **Accept:** every transcript at depths 2–7 that differs from the 6.1 depth fixtures is a near-tie: the engine's
      text and ours are at most 0.03 nats per text token apart under our model.
      (2026-10-10) — reworded by the user's decision; was: "transcripts at depths 2–7 equal the 6.1 depth fixtures at
      the same rate as the full depth does". A truncated network magnifies tiny numeric differences: the engine's own
      depth-N text equals its full-depth text on only 7–19 of 32 clips. Met: all 31 mismatches of 192 are at most
      0.0288 nats per text token apart (numbers below).
      (2026-10-10) — `just depth-probe`, `search.Default()` on `WithDepth`: depth 2 24/32, 3 27/32, 4 29/32,
      5 29/32, 6 28/32, 7 24/32 (161/192 = 83.9 %) against 31/32 at full depth; greedy 18–22/32. All 31 mismatches
      are near-ties: 23 model gaps of at most 0.0288 nats per text token (median 0.0025) and 8 search misses of at
      most 0.0196. The engine's own text at depth N equals its full-depth text on only 7–19 of 32. Fast regression
      `TestDepthMatchesEngine` (F-030).

### 6.5 Public pipeline & limits (`whistle/`, `cmd/whistle`)

- [x] `whistle.Load(path, opts)` over `.cact` or safetensors; `(*Model).Transcribe(pcm, opts)`, `TranscribeWAV`,
      `Embed`; built only from the exported `frontend`, `encoder` and `decoder` APIs. Greedy decoding until 6.6.
      (2026-10-10) — `Load(path, ...Option)` (`WithVocabulary`, `WithoutMmap`; a safetensors checkpoint takes the
      tokenizer from the `whistle.cact` next to it) builds `encoder.Best()` and `decoder.Best()`. `Transcribe` takes
      `TranscribeOptions{Language, Keywords, WordTimestamps, Depth}`: keywords, timestamps and depths 2–7 return
      `ErrNotSupported`, an unknown language or a depth outside 2..8 an error. The decoding sits behind an internal
      `strategy` seam; the greedy one is `decoder.GreedyMemory` step for step over the exported API, with the clock
      between the steps. `Embed` returns `Embedding{Values, Frames, Dim}` (`TestLoadSafetensors`,
      `TestTranscribeOptionsRejected`, `docs/FINDINGS.md` F-027).
- [x] The 30 s limit: an error whose message contains "30 s" for more than 480,000 samples.
      (2026-10-10) — `ErrTooLong` is the engine's own text, `audio limit is 30 s` (recorded stderr for 480,001
      samples). `Transcribe`, `Embed` and the CLI reject 480,001 samples; 480,000 give 375 frames
      (`TestTranscribeLimit`, `TestTranscribeCmdLimit`, `TestTooLongIsTheEngineText`).
- [x] Result JSON byte-compatible with the engine in field names, order and number formatting; silence returns exactly
      `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}`.
      (2026-10-10) — `Result.MarshalJSON`/`UnmarshalJSON` by hand: `text`, `language`, `words` (only when non-nil),
      `ttft_ms`, `decode_tps`; `%.1f` for the timing, `%.2f` for times, `%.3f` for probabilities; `"` escaped,
      non-ASCII raw. The engine prints `words` only with `--audio-word-timestamps`, for silence too, so the exact
      JSON above is the timestamps-on form, `whistle.Silence()`; `Result{}` is the engine's form without timestamps.
      Gating real silence to it is 6.6 (F-027).
- [x] `ttft_ms` and `decode_tps` measured the same way as the engine (TTFT includes the front end and the encoder).
      (2026-10-10) — TTFT from the PCM in hand to the first token after the language token; `decode_tps` is the
      tokens after it, EOS included, over their time. The engine's TTFT is 0.66 + 1.084 ms per second of audio
      (r 0.980 over the 32 clips, median of 3 runs), so it includes the encoder, as ours does (3.30 + 13.04 ms/s,
      r 0.991). Its `decode_tps` does not depend on the text length (r −0.147). Medians: engine 7.5 ms / 1401 tok/s
      (5 beams), Go 85.5 ms / 677 tok/s (greedy). Whether the engine stops the TTFT clock at the language token or at
      the first text token is below the fits' resolution (F-027).
- [x] Minimal `whistle transcribe <wav> [--language] [--json]`.
      (2026-10-10) — plus `--model` (default `models/whistle/whistle.cact`); prints the text, or the JSON line with
      `--json` (`TestTranscribeCmdText`, `TestTranscribeCmdJSON`).
- [x] **Accept:** the transcripts and `language` of `Transcribe` equal `decoder.Greedy` on the 32 clips, and the JSON
      round-trips byte-identical against recorded engine output.
      (2026-10-10) — `TestTranscribeMatchesGreedy`: text and language equal on 32/32 (`-short`: 3 clips); text equal
      to the engine's on 22/32, as in Phase 5. `TestResultRoundTripsEngineOutput`: the ten raw engine outputs in
      `whistle/testdata/needle-cli/` (with and without timestamps, quotes, Polish, silence, a 30 s clip) unmarshal and
      marshal back byte-identical (F-027).

### 6.6 Silence gate & integration (`whistle/`; needs 6.1, 6.2, 6.5)

- [x] Silence gate: a loudness-range measure (e.g. a percentile spread of frame RMS in dB). Pick the measure and the
      threshold from the 6.1 silence sweep (U10); return the exact empty JSON for silence.
      (2026-10-10) — `internal/silence`: per front-end frame (Hann 400 centred every 160, zero padded) the mel energy
      of channels 6–57 (centres 261–3307 Hz, the site's "250-3500 Hz") in dB; silence iff the nearest-rank p99 − p10
      spread is ≤ 5 dB. Classifies all 340 sweep and limit verdicts like the engine, empty up to 4.985 dB
      (step +2.65 dB), transcribed from 5.011 dB (en SNR −9.75 dB). The other 2 of the 342 are over 30 s and rejected
      first. Leaderboard of 340 measures (`go test -tags oracle -run TestFitGate ./internal/silence/`): the full band
      15 wrong (pink noise), flat 250–3500 Hz bins 12, the time-domain RMS 29; band edges ±1 channel and p9/p10 also
      separate, the window and padding are not identified. `Transcribe` runs it after the length check and returns
      `Result{}` (the engine's JSON without timestamps; `Silence()` once Phase 7 adds them) without decoding
      (`TestGateMatchesEngine`, `TestTranscribeSilenceSweep`, `TestTranscribeLimits`; `docs/FINDINGS.md` F-031).
- [x] Beam search wired into `Transcribe` (with keywords and depth once 6.3/6.4 are merged).
      (2026-10-10) — beam search part done: `search.Beam(d, mem, search.Default())` behind the `strategy` seam, greedy
      kept for tests. The F-027 clock reads the search's steps through `Options.Adjust` (TTFT at the first step after
      the language token, `decode_tps` over the steps), chained before any keyword hook (`TestBeamClock`) (F-031).
      (2026-10-10) — keywords and depth wired: `TranscribeOptions.Keywords` builds `search.NewKeywords` under
      `search.DefaultBias()` (entries trimmed, newlines split like the engine's file), `Depth` 2–7 runs the
      `decoder.WithDepth` rung built at `Load` on the same `Memory`; `whistle transcribe --keywords file --depth N`.
      Through `Transcribe` the F-029 fast keyword cases and the F-030 depth-2/4 clips equal the engine
      (`TestTranscribeKeywordsMatchEngine`, `TestTranscribeDepthMatchesEngine`).
- [x] `just eval-librispeech`: WER of the Go pipeline and of the engine on the 6.1 subset.
      (2026-10-10) — `TestEvalLibriSpeech` (runs only with `WHISTLE_EVAL_LIBRISPEECH`; skips without the model or
      audio) transcribes the 200 utterances with `TranscribeWAV` and scores both sides with the manifest's `wer_rule`
      through `internal/eval` (it reproduces the engine's 221 errors exactly): Go 223/4167 = 5.35 %, engine 5.30 %,
      text equal on 181/200, language 200/200, 42.5–66.0 s for 1549 s of audio on a shared machine (F-031).
- [x] **Accept:** transcripts equal the engine's on ≥ 95% of fixtures (with the rest explained in FINDINGS), and
      LibriSpeech test-clean WER is within 0.3 absolute of the engine on the same subset.
      (2026-10-10) — Fixtures: 31/32 (96.9 %) equal in text and language through `Transcribe`
      (`TestTranscribeMatchesEngine`); the miss, `fleurs-pl_pl-10047713455444904767`, is F-026's search miss (the
      engine's text scores higher under our model, its prefix leaves the beam 0.135 nats behind the fifth). LibriSpeech
      subset: 5.35 % against 5.30 %, +0.05 percentage points. Of the 19 texts that differ from the engine's, 8 only in
      punctuation; under teacher forcing our model prefers ours on 17 by at most 0.025 nats per token, the other 2 are
      search misses by under 0.001 (F-031). Holds without keywords at full depth; 6.3/6.4 add those.

## Phase 7 — Word timestamps & output JSON — ✅ DONE (2026-10-10)

One session: `internal/decoder` (capture hook), `internal/align` (new) and `whistle/`. The 32 fixture JSONs already hold
the engine's word times and probabilities, so this phase needs no new oracle runs; teacher-force the engine's text.

- [x] 7.1 Cross-attention capture: per decoded token, the attention probabilities per layer and head over the 80 ms
      frames, through a decoder hook that costs nothing when off.
      (2026-10-10) — `decoder.State.SetCapture(fn)` reports per `Extend` call and layer that runs a
      `decoder.Attention` (probabilities `[tokens, heads, frames]` and the output gate); off it is one nil check per
      layer, on the logits are bit-identical, a row-by-row feed is within 1.1e-6 of one pass, a fork does not inherit
      it, a ladder rung reports only its layers (`TestCapture`). Timing uses one extra teacher-forced pass over the
      winning transcript instead of capturing in every beam: 26.0 ms on a 6.3 s clip, `Transcribe` 143.9 → 170.4 ms
      (+18 %, `BenchmarkWordTimestamps`), 93 % of it the decoder pass (`docs/FINDINGS.md` F-033).
- [x] 7.2 Word grouping: tokens starting with `▁` begin words; byte pieces are merged. `probability` per word:
      candidates are the mean or the product of the token probabilities (U11). Match the engine.
      (2026-10-10) — `align.Words`: the engine's word strings on 477 of 477 transcripts (8156 words: fixtures,
      LibriSpeech, keyword and depth runs). Time and probability come from the word's core tokens, without Whisper's
      punctuation words (`string.punctuation` substring test: `,` `.` `!"` out, `."` `?"` in). `just align-probe`
      probability leaderboard, 8156 words: **Whisper's mean of the core tokens' p over the whole vocabulary** 50.6 %
      equal to three decimals, 91.1 % within 0.01, mean |Δ| 0.0036; renormalised over text pieces ties (50.6 / 91.2);
      keyword-adjusted 50.3 / 90.5 (on the keyword cases 42.2 / 80.7 against 44.5 / 87.2 raw); geometric mean 49.7 /
      85.5; mean over all tokens 45.8 / 83.6; product, min, max, first/last lower. The rest is the engine's W2A8 token
      probabilities (F-033).
- [x] 7.3 Timestamps: aggregate the attention over heads and layers (candidates: all layers, the last layer, the
      best-aligned heads) and run a monotonic DTW alignment over the 80 ms frames (U11).
      (2026-10-10) — Whisper's `find_alignment` with its default alignment heads, every head of the last half of the
      layers that run (4–7; on a ladder rung its last half): per-head z-score per frame over all positions, no median
      filter, rows = the queries predicting the language token, each text token and EOS, `dtw(−matrix)`, jump times.
      Grid of 69,120 variants plus whole-layer subsets, greedy head fits and ladder rungs, fitted on LibriSpeech:
      exact start / end 94.6 % / 98.8 % (layers 4–7) against ≤ 85 % / 90 % for any other layer subset, 85.3 % /
      88.5 % for a fitted 16-head set, 77.7 % / 82.3 % for the last layer, 71.9 % / 78.8 % for all heads; depth runs
      95.0 % / 98.0 % with the rung's last half. Every start but the first is exact on 99.0 %, every end on 98.8 %.
      The first word's start is not identified; an energy onset (silence gate levels within 14 dB of p99, fitted)
      floors it. Teacher forced, `align.Default`, start and end within 80 ms: LibriSpeech (train) 99.1 %, held out
      99.3 % (fixtures 99.3, keywords 99.2, depth 99.3); without the onset 94.6 % held out (F-033).
- [x] 7.4 `words` in `whistle.Result` and `--word-timestamps` in the CLI.
      (2026-10-10) — `TranscribeOptions.WordTimestamps` fills `Result.Words` (one capture pass on the search's rung and
      `Memory`, then `align.Align`); silence returns `Silence()`; `whistle transcribe --word-timestamps` prints a line
      per word, or with `--json` the engine's line (`TestTranscribeCmdWordTimestamps`).
- [x] **Accept:** timestamps are monotone and within [0, duration], start/end are within 80 ms of the engine on ≥ 90% of
      words, and the JSON is byte-compatible in field names and order.
      (2026-10-10) — Through `Transcribe` on the 32 fixtures (`TestTranscribeWordsMatchEngine`): every result's words
      spell its text, are monotone (start ≤ end ≤ next start), on the 80 ms grid and within [0, duration]; start and
      end within 80 ms on 97.4 % of the engine's 459 words (start 97.8 %, end 97.6 %; 451 paired by string, the
      Polish search miss of F-026 counting as failures), exact 93.9 % / 97.2 %, probability within 0.01 on 90.6 %.
      `TestTranscribeWordsJSON`: the raw engine outputs with timestamps in `whistle/testdata/needle-cli/` equal ours
      with the numbers masked (fleurs-de, ls-clean, `silence-2s` exactly; `tiled-30s`, whose transcript differs,
      in structure) (F-033).

## Phase 8 — Streaming & embeddings — ✅ DONE (2026-10-10)

- [x] 8.1 Stream oracle (`tools/oracle`, `testdata/oracle/stream/`): `needle_stream_transcribe_process`/`_stop` with
      1 s chunks over the 32 clips, recording every call's output, plus two concatenated streams longer than 30 s.
      (2026-10-10) — `just oracle-stream` (purego, exact chunks): 108 runs, 822 calls in `calls.json` (sample range,
      rc, JSON without `pass_ms`) plus `manifest.json` (620 KB). The runs: the 32 clips at 1 s, 0.5 s and 2 s, 4
      forced-language runs, `long-clean` (44.91 s), `long-other-gaps` (42.16 s), `long-silence-lead` (36.01 s) and 5
      edge cases. Every run equals the CLI's `--audio-stream`; two runs are byte-identical. Each pass is
      `needle_transcribe` of `[t0, received)` (470/470 committing calls), commits by LocalAgreement-2 on exact words
      (816/816) and trims at the second-to-last committed sentence once the buffer exceeds 15 s (all 470 t0). The
      final text equals offline on 18/32 clips at 1 s (14/32 at 0.5 s, 27/32 at 2 s), so the Accept needs the stream
      fixtures. Beyond 30 s without commits is open (F-032).
- [x] 8.2 `Embed`: the encoder output per 80 ms frame, the same layout as `needle_embed` (delivered with 6.5).
      (2026-10-10) — `(*Model).Embed(pcm)`: `Frames` rows of 512 values (`enc.final_norm` of the lane mean), at most
      30 s, no frame under 160 samples. Against `testdata/oracle/<clip>.embed.safetensors` (`TestEmbedMatchesEngine`):
      mean per-frame cosine 0.999828, every clip ≥ 0.999033, so the phase's "embeddings match at cosine ≥ 0.99"
      holds (F-027).
- [x] 8.3 `Stream`: buffer the incoming chunks; every ~1 s re-transcribe the buffer; commit the longest word prefix on
      which two consecutive passes agree; report `pending`, `received` and `pass_ms`; `Stop()` flushes (U12).
      (2026-10-10) — `(*Model).NewStream(StreamOptions{Language, Keywords})`, `Process(pcm) (StreamResult, error)`,
      `Stop() StreamResult`; `StreamResult` marshals to the engine's stream JSON (field order, `received` to two
      decimals, `pass_ms` to one, the words through F-027's marshaller). One pass per call over the buffer with
      `Transcribe`'s decoding and word timestamps, F-032's dropping (end − 0.1 s, shortest repeated n-gram; longest
      gives the same on all 108 runs), LocalAgreement-2 on exact strings, the 15 s second-to-last-sentence trim, `Stop`
      without a pass, F-032 item 9's edges. `received` is the engine's float arithmetic (word times
      `float32(frames) · 0.08f` plus a float64 offset), the one variant of 64 that prints all 929 values. A `Stream`
      belongs to one goroutine; the `Model` stays safe for concurrent use. `just stream-parity` replays all 108 runs
      against `calls.json`: the decision (rc, text, pending, language, received) equals the engine's on 794 of 930
      calls (clips 193/224, chunks 454/535, language 21/32, long 115/128, edge 11/11), every decision of 69 of 108
      runs, `received` and `language` on 930/930; committed words both times within 80 ms 98.8 %, probability within
      0.01 93.4 %; the final text equals the engine's stream on 98 of 108 runs. Every run's first differing call
      follows a pass that spells other words than the engine's (38 runs, 32 within ±0.07 nats per token) or another
      word time (1), none the bookkeeping (`docs/FINDINGS.md` F-036). After 9.2 made the packed
      path the default, the same replay gives 797/930 calls (clips 194, chunks 456, others unchanged), the Accept
      numbers unchanged.
- [x] 8.4 Buffers longer than 30 s: trim committed audio the way the engine does (U12).
      (2026-10-10) — The sentence trim keeps the long runs under 30 s as in the engine: all of F-032's inferred
      buffer starts are reproduced up to each run's first divergence (466/471; the five others follow it in
      `long-clean`), `received` included ("42.15"). With nothing to trim (`long-silence-lead`), the evidence (word
      times on the 80 ms grid from 0, `received` 36.01, no ≤ 30 s window fits on the engine) says the engine passes
      the whole buffer. The port never feeds more than 30 s: it drops the oldest audio in whole frames and passes the
      last 30 s on the grid. Under our model both give the same 3 of 5 passes after the silence; the engine's 34 s and
      36 s passes are reached from neither (`TestStreamWindowScan`), so that run's last calls still differ (F-036).
- [x] **Accept:** on fixture clips fed in 1 s chunks, the final committed text equals the offline transcript at least
      as often as the engine's own stream does, with the call-by-call parity against the engine's stream reported, and
      the embeddings match the engine at cosine ≥ 0.99.
      (2026-10-10) — reworded by the user's decision; was: "on fixture clips fed in 1 s chunks, the final committed text
      equals the offline transcript, and the embeddings match the engine at cosine ≥ 0.99". The engine's own stream
      meets that on 18 of 32 clips, so no engine-faithful stream can. Met: 18 of 32 against the engine's 18; the
      final text equals the engine's stream on 98 of 108 runs, and 797 of 930 calls (85.7 %) equal the engine's (8.3,
      F-036); embeddings ≥ 0.999033.
      (2026-10-10) — at 1 s chunks our final committed text equals the offline transcript on 18 of 32 clips
      (normalised 24 of 32, 25 word errors in 459), against the engine's offline text and against our own
      `Transcribe` alike; the engine's own stream also gives 18 of 32 (24 normalised, 19 word errors), 13 of the 14
      misses shared (`TestStreamAccept`, F-036). The embedding half holds since 8.2 (F-027: every clip ≥ 0.999033).

## Phase 9 — Native quantised path & performance — ✅ DONE (2026-10-11)

- [x] 9.1 CQ LUT GEMV kernel (`internal/cq`, plus an additive raw-blob accessor in `internal/cact`): a 2/4-bit LUT GEMV
      in the rotated (Hadamard) domain. Apply FWHT-128 to the activation groups once and multiply by the codebook
      values, instead of dequantising the weights. Pure Go. **Accept:** equal to dequantise + GEMV at cosine
      ≥ 0.99999 on real `.cact` tensors, with a benchmark against the float GEMV.
      (2026-10-10) — `cact.File.CQPacked` returns a CQ record as stored (index bitstream and FP16 norms aliasing the
      file, widths, group, shape, codebook); `TestWhistleCQPacked` decodes all 152 records from it to exactly `CQ`.
      `cq.Matrix` (`New`, `GEMV`, `GEMM`, `Rows`, `Row`) rotates each activation group once by FWHT-128 / √128 and
      sums per-activation 16-entry nibble tables over the packed index words; `GEMM` interleaves the tables of four
      activations so one index decode feeds four lookups. The `gather` reference stays; a 256-entry byte table lost
      on every shape but `pw1` and was removed. All 152 CQ tensors are stored `[out, in]` and qualify as is; the mHC
      φ need `Rows` per layer, and the embedding and engram gathers use `Row`. `TestWhistleGEMM` on 11 real tensors
      (CQ2 q/k/v/out/pw1/stem/engram table, CQ4 embedding head and φ) against `CQ` + `tensor.MatMulTransB`: worst
      1 − cos 1.7e-13 for any kernel, max |Δ| ≤ 5.7e-7 of max |y|. Single-threaded on the M5 Pro, the pure-Go kernel
      is 1.15–2.4× faster than the pure-Go float loop on 9 of 10 shapes, but 2.2–5.7× slower than the NEON float
      GEMM (q_proj 20.1 vs 7.7 µs, head 652 vs 192 µs). Memory: 16.63 MB for all CQ matrices against 218.77 MB as
      float32 (13.2×). Details and what 9.2 needs: `docs/FINDINGS.md` F-028.
- [x] 9.2 Run straight from the packed `.cact`: wire the 9.1 kernel into `needle`, `encoder` and `decoder` behind
      `weights.Source`. Memory target: the model's weights under 64 MB, counted as the live Go heap plus the resident
      `.cact` pages (`just mem-probe`).
      (2026-10-10) — target reworded by the user's decision; was: "under 64 MB RSS for the model". Process RSS is not
      the measure: the weights are, 34.1 MiB (numbers below). The 127.6 MiB process peak of a 30 s `Transcribe`
      (activations and GC headroom) is a separate goal that does not block 9.2 (F-037 "Open").
      (2026-10-10) — the packed path runs the whole model and is `whistle.Load`'s default for `.cact`
      (`WithFloatWeights()` keeps float). `weights.Options.Packed` makes the `.cact` source a `weights.Packer` whose
      `Matrix(name)` serves each CQ record as a cached `cq.Matrix` without a float32 copy. `needle.Linear` (`Apply`,
      `Row`, `Rows`) runs all 152 CQ tensors: the attention and cross-attention projections, `pw1`/`pw2`, the stem, the
      mHC φ, the engram tables and projections, and the embedding with its tied head. `cq.NewGather` skips the lut copy
      of the gather-only engram tables. Parity, packed against float: every transcript equal (32 clips, 192 depth cases,
      53 keyword cases, 200 LibriSpeech utterances). Equal to the engine: 31/32 both ways. WER 5.35 % both (engine
      5.30 %). `Embed` against `needle_embed`: 0.999827 against 0.999828. Memory (`just mem-probe`, M5 Pro): the model,
      i.e. the live Go heap plus the resident `.cact` pages after `Load` and a 30 s `Transcribe`, is 34.1 MiB against
      230.6 MiB, and the process is at 41.7 MiB after `Load`. But a 30 s `Transcribe` peaks at 127.6 MiB process RSS
      (float 488.3 MiB): the stem's 29 MiB activation spike, the 10 MiB cross-attention memory and `GOGC` headroom. So
      the 64 MB target holds for the weights, not for a 30 s run. Speed, the same process alternating: decode 1.56–1.59×
      faster, `Load` 10 against 165 ms, but TTFT 2–8 % slower (batched products, the 2048-wide CQ4 φ tables). Open
      levers for the peak and the TTFT: `docs/FINDINGS.md` F-037.
      (2026-10-10) — 9.5 runs the stem in chunks of 8 output frames: a 30 s `Transcribe` peaks at 106.4–108.1 MiB
      process RSS packed (was 126.8–127.6), 97.5–97.7 MiB on one core; the model stays at 34.7–34.9 MiB. The rest is
      `GOGC` headroom over a 29–40 MiB live heap during the decode. `docs/FINDINGS.md` F-038.
- [x] 9.3 Optional int8 activations and an int8 KV cache (`act_bits 8`, `kv_bits 8`) to mirror the engine's numerics,
      behind a flag.
      (2026-10-11) — `whistle.WithInt8()`, off by default; with it off, the `Embed` rows and the `Transcribe` texts,
      word times and probabilities hash as on `main`. Upstream's `fake_quant` (s = max|x| / 127 per row, round half to
      even) was tried at every site and scored against the engine's continuous outputs (`needle_embed`, 7383 word
      probabilities): the engine applies it to the mHC heads' input and to the attention's queries, keys and values
      per head (share of our int8 step it takes too ≈ 1), but not to the other CQ products' inputs (≈ 0; per group,
      rotated and bfloat16 variants too). So the flag quantises exactly those, and the decoder's self-attention cache and
      the cross-attention memory hold int8 codes with a scale per head and row (`needle.Int8Rows`), attended with exact
      int32 dot products (`needle.AttendInt8`, NEON/AVX2 kernels with Go fallbacks). Flag on against off: word
      probabilities' gap to the engine −8 %, depth ladder 174 against 161/192, stream 812 against 797/930 decisions,
      WER 5.33 % against 5.35 % (engine 5.30 %), keywords 46/53 both; but 30 against 31/32 clips equal and `Embed`
      0.999719 against 0.999827 (one fragile frame). Speed: decode 6–9 % faster, TTFT 5–9 % slower; the 30 s peak
      99.7–103.6 against 107.2–107.9 MiB. The flag stays off by default. The memory's k and v projections now share one
      rotation (`ApplyShared`, bit for bit). `docs/FINDINGS.md` F-041.
- [x] 9.4 Plan 9 assembly (NEON/AVX2) for the LUT GEMV, the FWHT, the Monarch Kronecker products and the
      softmax/Sinkhorn hot loops, with pure-Go fallbacks and equivalence tests (see `go-plan9-assembly`).
      (2026-10-10) — the `internal/needle` half. Profiled `Transcribe` over the 32 clips first. The Monarch
      `kronApply` now runs as two `tensor.MatMulTransB` calls on transposed factors (4.8 → 0.68 µs per mix). `attend`
      batches each head into Q·Kᵀ and P·V GEMMs from 8 query rows. Sinkhorn runs in the exp domain in float64, with a
      log-domain fallback (4.46 → 0.52 µs). `meanSquare` uses four partial sums. The softmax gets a NEON/AVX2 Cephes
      expf kernel (`exp_arm64.s`, `exp_amd64.s`, float64 `purego` fallback, ≤ 0.99 ulp). `TestTranscribeMatchesEngine`,
      one run each: median TTFT 82.5 → 57.8 ms, decode 561 → 609 tok/s, still 31/32, all 32 transcripts unchanged; a
      30 s clip's TTFT 513 → 281 ms. The remaining time is `tensor`'s GEMM and `search`'s goroutines.
      `docs/FINDINGS.md` F-035.
      (2026-10-10) — the `internal/cq` half: a new `lut` kernel with NEON (`lut_arm64.s`) and AVX2
      (`lut_amd64.s`, chosen at run time) assembly for the FWHT, the table build and the lookup, each with a Go
      reference that it equals bit for bit (`asm_test.go`); `purego` and other architectures keep `nibble4`. Nibble
      tables are rounded to 14-bit integers in two byte planes, and `cq.New` repacks the index into blocks of 32 rows,
      so one `TBL`/`VPSHUFB` looks up 16 or 32 rows. Accuracy on the 11 real tensors: worst 1 − cos 7.8e-8 in every
      build (arm64, purego, amd64 AVX2 in Docker). Single-threaded on the M5 Pro, the GEMV beats the NEON float GEMM
      on all seven F-028 shapes: 1.05× (φ_res) to 3.2× (pw1), q_proj 2.91 vs 7.52 µs, head 97 vs 192 µs; 4.8–6.7×
      faster than `nibble4`. Batched: 1.40× at 5 rows of q_proj, 1.11× at 375, but 1.06× slower on the 5-beam head.
      The blocked copy doubles the CQ memory to 33.27 MB (still 6.6× below float32). Details: `docs/FINDINGS.md`
      F-034. Both halves merged: the LUT GEMV, the FWHT and the softmax exp have NEON/AVX2 assembly; the Monarch
      Kronecker products run on `internal/tensor`'s NEON GEMM on arm64 and in pure Go on amd64, which has no GEMM
      kernel yet (12.3); Sinkhorn stayed in Go (8.6× faster in the
      exp domain, under 1 % of the profile, F-035). Every kernel has a pure-Go fallback and an equivalence test, and the
      amd64 builds pass in linux/amd64 Docker with AVX2.
- [x] 9.5 Parallelism: the encoder over time blocks and heads; the decoder over beams.
      (2026-10-10) — `tensor.Parallel` runs pieces on the caller and helpers from one budget for the process
      (`Workers() − 1`, default GOMAXPROCS), so nested and concurrent callers (beams × kernels) stay bounded. Every
      split keeps each output's sums in their order, so outputs are bit for bit the same for any worker count and equal
      to `origin/main`'s (`Embed` of 32 clips and `Transcribe` texts, word times and probabilities hashed in both trees).
      `MatMulTransB` and the lut GEMM split over rows (aligned to 4) or over the head's blocks of 32 rows; `cq.GEMMs` and
      `needle.ApplyShared` rotate and tabulate an activation once for q/k/v/gate, the three φ and engram k/v (1.25× and
      1.77× on one core). The encoder runs over time blocks (the stem in chunks of 8 frames, the row loops) and over
      (head, row block) pairs in the attention; the decoder runs over beams as before (233–270 tok/s on one core,
      960–1060 on all). `TestSpeedProbe`, old and new alternating at a load average of 4–7: median TTFT 65.3–67.1 →
      21.2–21.3 ms packed, 60.7–62.9 → 23.4–24.0 ms float; the 30 s clip 332–337 → 66–67 ms and 311–314 → 72–75 ms;
      decode 819–947 → 960–961 tok/s packed. Parity unchanged: 31/32, `Embed` 0.999827, WER 5.35 %, stream 797/930.
      `docs/FINDINGS.md` F-038.
- [x] 9.6 Benchmarks (`bench/`): TTFT and decode tok/s against the engine on the same machine. The engine publishes
      11.1 ms TTFT and 1319 tok/s on an M4 Pro with 5 beams. Also the real-time factor for 30 s clips.
      (2026-10-10) — `just bench` (`bench/cmd/whistle-bench`, public `whistle` API) times packed and float alternately
      in one process over the 32 fixture clips (1 warm-up + 5 passes) and the first clip tiled to 30 s (1 + 10 runs):
      `Load`, TTFT and decode rate (median, p90), wall time, real-time factor, plus peak RSS from a fresh child per mode
      and the LibriSpeech WER. `just oracle-bench` (`tools/oracle bench`) times the engine on the same samples, the CLI
      (`--threads 4`, one process per run) and `libneedle3` in process. Both write `bench/results/*.json`, and
      `bench/RESULTS.md` is rendered from them. The engine's `ttft_ms` is the same clock in the CLI and the library and
      leaves out process start, model load and WAV reading; a library call is `ttft_ms` + (n − 1) / `decode_tps` plus
      about two decode steps. So like with like is `Transcribe` against `needle_transcribe`. M5 Pro, commit
      `fe5884e4e6`, load average 5–8 from other sessions, medians: TTFT 60.4 ms packed / 57.5 ms float against the
      engine's 7.5 ms (8.1×); decode 1013 / 604 against 1383 tok/s (0.73×); 30 s clip TTFT 310 / 290 against 36 ms,
      real-time factor 0.0165 / 0.0192 against 0.0048; `Load` 7.0 / 139 ms against `needle_load`'s 2.7 ms; peak RSS of
      `Load` + 30 s 125 / 489 MiB against the CLI's 70 MiB. The engine here beats its published M4 Pro TTFT (7.5 against
      11.1 ms) at about its published decode rate. (2026-10-11) — re-run after 9.5 (both sides, load average 2–6):
      TTFT 19.3 ms packed / 21.6 ms float against the engine's 6.9 ms (2.8×); decode 1071 / 597 against 1499 tok/s
      (0.71×); 30 s clip TTFT 60 / 67 against 35 ms, real-time factor 0.0081 / 0.0116 against 0.0045; `Load` 6.7 / 136
      ms against 2.4 ms; peak RSS 106 / 493 MiB against 70 MiB; WER unchanged. (2026-10-11) — `just bench` after 9.3
      (int8 off, the memory's k and v sharing one rotation): TTFT 18.8 / 21.3 ms, decode 1085 / 610 tok/s, 30 s clip
      TTFT 61 / 69 ms, peak RSS 108 / 495 MiB, WER 5.35 / 5.35 %; the engine side unchanged. `docs/FINDINGS.md` F-040.
- [x] **Accept:** WER unchanged within 0.1 against the float path, with numbers recorded in `bench/RESULTS.md`.
      (2026-10-10) — the float path is `whistle.Load` with `WithFloatWeights()`, packed its default on `whistle.cact`.
      LibriSpeech test-clean subset (200 utterances, 4167 words, `just bench`): packed 223/4167 = 5.35 %, float
      223/4167 = 5.35 %, difference 0.00 points; all 200 transcripts equal. Engine 5.30 %. Recorded in
      `bench/RESULTS.md`; F-040.

## Phase 10 — Distribution & extras

- [x] CLI polish on top of 6.5: `whistle transcribe [--language] [--keywords] [--word-timestamps] [--depth] [--json]`,
      `whistle stream` (stdin PCM or a WAV played in chunks), `whistle embed`, `whistle doctor`
      (2026-10-10) — `transcribe` takes several files on one model load: the text after each path, or a JSON line each
      (the engine's JSON unchanged); a failing file goes to stderr, the rest go on. A file over 30 s fails with
      `ErrTooLong`, its path and a pointer to `stream`. Exit codes: 0, 1 on failure, 2 for a wrong command line (flags,
      arguments, unknown commands). `stream [wav]` feeds a WAV file in `--chunk` seconds (default 1, (0, 30] as the
      engine's `--audio-chunk`), or without a file raw 16 kHz mono PCM from stdin (`--format s16le|f32le`), one
      `Process` per chunk and `Stop` at EOF or Ctrl-C (a WAV is decoded a chunk at a time, `frontend.OpenWAV`);
      `--realtime` paces a file, `--json` prints each call's stream JSON, `--commits-only` the committing calls and the
      stop as the engine's `--audio-stream`. The CLI's calls equal the library's `Stream` fed the same chunks, byte for
      byte without `pass_ms`, on three manifest runs at 1 s (WAV, s16le and f32le; `long-clean`, 44.91 s, from the WAV),
      and so their final text, which is also the engine's stream text (`TestStreamCmdMatchesLibrary`). `embed <wav>`
      writes JSON with `frames`, `dim`, `frame_ms` and `embedding` (shortest float32 decimals) or `.npy` (`-o`); equal
      to `Embed` in both formats, cosine ≥ 0.99 per frame against `needle_embed`, 30 s and empty files rejected
      (`TestEmbedCmd`). `doctor` prints the model's path, size, format, SHA-256 against the pinned manifest, the other
      pinned files next to it, the weights mode, the Go version and platform, the assembly kernels (NEON; AVX2/FMA by
      CPU on amd64; none under `purego`) and GOMAXPROCS, then loads the model, runs a 2 s tone burst through `Embed` and
      `Transcribe` and transcribes `testdata/audio/ls-other-3080-5040-0018.wav` when present; it exits 1 on any failed
      check (`TestDoctorPasses`, `TestDoctorWrongFile`, `TestDoctorMissingModel`). Model-free tests cover the PCM
      readers, chunking, the view, Ctrl-C, pacing, the `.npy` and JSON writers and the exit codes.
- [x] README quickstart, examples and Go API docs
      (2026-10-10) — README: status, quickstart (install, `download`, `doctor`, `transcribe`, `stream`, `embed`),
      a Go API program, the parity table with links to FINDINGS and the limits; every command and the program were
      run and their output pasted. The repository is not public yet, so the README points `go get` at a clone via
      `replace`. `whistle/doc.go` is a package overview (loading, transcribing, streaming, embeddings, parity), and
      `whistle/example_test.go` has 10 examples: 4 model-free with `// Output:` (the result and stream JSON, reading
      engine JSON, embedding rows), 6 that need the model and only compile under `go test` (each was run against
      the model with its output checked once).
- [x] Follow-up from 6.3: `de-paare-missed`, the one keyword search miss. Weaker hypotheses with the shared keyword
      bonus end one token earlier and fill Whisper's 5 finished slots (F-029); find the engine's rule that keeps the
      right one.
      (2026-10-11) — partial: no finished-slot rule is the engine's. Nine variants behind a test-only switch
      (`finRule`: a finished text takes one slot, one live hypothesis per text, 6/7/10 slots, the final pick with the
      live ones, only the step's best 5 EOS count, a stop only when no live one outscores the best finished) on the 53
      keyword cases, 32 clips and 192 depth cases: every one that makes `de-paare-missed` equal loses its mirror
      `de-diese-paare-phrase` (44–46/53, the slot variants also 29–30/32 clips and 153–158/192). Both cases have the same
      beam and differ only by a bonus all hypotheses share, which no rank-based finished rule can tell apart.
      `oracle kwsets` records the engine on the clip with 32 keyword files (`testdata/oracle/kwsets`): the miss is in
      the bias, whose single-word bonus prunes the competing "Bare" spelling in our search but not in the engine's.
      273 biases (weights, completion, revocation, spellings, transient, icefall-style and once-per-keyword
      completions): 142 make the case equal, none without losing a keyword case. The best, w 2.1 + completion 0.25,
      gives 47/53 and 26/32 sets (`DefaultBias` 17/32) but loses `pl-francja-present`, a 0.0033-nat near-tie; w 1.5
      fits 31/32 sets but 44/53 cases. Defaults unchanged, so the guard set is unchanged. Open: case-insensitive
      matching (`PAARE` = `Paare` in the engine), phrase loops, the capitalised bare spelling (F-042).
      (2026-10-11) — partial, second round: `oracle kwvariants` records 31 more keyword files on 8 clips
      (`testdata/oracle/kwvariants`): the engine ignores case (13 of 13 capitals), needs a word start, no word end,
      and matches token ids (`Adop` = `▁Ad op` takes effect, `Paa` and `alivi` do not). `TestProbeFold` scores 308
      biases on the 53 cases, 32 sets and 31 variants: case spellings on token ids (`fold=cases`) cost nothing and add
      9 variants; a matcher over folded bytes is 42/53 at best, a bonus per character 40–42/53; `spent` stops the
      loops but turns the variant `Cowley` into a finished-slot search miss; the capitalised bare spelling is the
      weight, not the form. `DefaultBias` is now `w=1.8 complete=1.5` with the case spellings: 47/53 cases
      (`it-montagne-distractor` won, nothing lost), 18/32 sets, 29/31 variants, the rest of the guard set unchanged.
      `de-paare-missed` still differs: no bias makes it equal without a loss. Two trades are left to the user:
      `w=2.15 complete=0` (`de-paare-missed` equal, 28/32 sets) for `pl-francja-present`, and `spent` for the
      variant `Cowley`. Open: what the engine's bonus depends on, since `accomun` and `Paare` both have 3 tokens and
      want bonuses on either side of the threshold that prunes "Bare" (F-044).
      (2026-10-12) — closed by the user's decision on the two trades: `DefaultBias` is `w=2.15 complete=0` with the
      case spellings and `spent`. `de-paare-missed` is equal; `TestProbeFold` (grid 4): 47/53 cases (`pl-francja-present`
      lost for it), 28/32 keyword sets (was 18/32), 28/31 variants (`Cowley` lost), no phrase loops. The engine's rule
      itself was not found (F-044 "Decision"). `TestKeywordsMatchEngine` keeps the engine's text on its four cases;
      two score floors moved.
- [x] Optional: an HTTP server (`/transcribe`, `/stream` over WebSocket)
      (2026-10-11) — `whistle serve` (`internal/server`) loads the model once. `POST /transcribe` takes a WAV body
      or raw 16 kHz mono PCM (`?format=s16le|f32le`) with the options as query parameters (`language`, repeatable
      `keywords`, `depth`, `word_timestamps`) and answers with the engine's JSON; 400 for bad options, unknown
      parameters or bad audio, 408 for a body slower than `--read-timeout`, 413 over `--max-body-mib` (16), 422 over
      30 s, 503 (before the body is read) when `--concurrency` (2) passes run and `--queue` (8) wait, 405 for other
      methods. `GET /stream` is a WebSocket with one `Stream` per
      connection: a binary message is one `Process` chunk (at most 30 s), answered by its stream JSON; the text
      message `{"type":"stop"}` sends the `Stop` result and closes with 1000, and the server sends it before its own
      closes (1001 on shutdown, 1008 after `--stream-idle`). `GET /healthz`, `GET /v1/info` (SHA-256, weights mode,
      languages, limits), a log line per request, graceful shutdown on SIGINT/SIGTERM within `--shutdown-timeout`.
      The model is safe for concurrent use (8 concurrent requests at `--concurrency 4` equal the sequential ones
      under `-race`); the server bounds the passes anyway, since each uses every core.
      With the model, `/transcribe` equals `whistle transcribe --json` but for the timing on five option sets and an
      s16le body (`TestServeMatchesTranscribeCmd`), and `/stream` in 1 s chunks equals `whistle stream --json`
      without `pass_ms`, line for line, on two manifest runs in s16le and f32le (`TestServeStreamMatchesStreamCmd`).
      Model-free tests cover the status codes, limits, queue, framing, stop, idle, shutdown and its deadline
      against a fake model. WebSocket: `github.com/coder/websocket`, only in `internal/server`.
- [x] Optional: a js/wasm build (as in go-pocket-tts) with a small web demo
      (2026-10-11) — `GOOS=js GOARCH=wasm` builds every package (mmap was already build-tagged, the kernels have their
      `*_noasm.go`). `whistle.LoadBytes` loads a `.cact` (or, with `WithVocabulary`, a safetensors checkpoint) from
      bytes. `cmd/whistle-wasm` installs `globalThis.whistle` (`load`, `decodeWAV`, `transcribe` → the engine JSON,
      `streamStart/streamProcess/streamStop` → the stream JSON, errors as rejected promises), which `web/worker.js` runs
      in a Web Worker. `web/` (plain HTML/JS/CSS): the model from the pinned Hugging Face URL (SHA-256 checked, kept in
      Cache Storage) or a file picker, language, keywords, depth and word timestamps, a WAV transcribed with the timed
      words (integer PCM through the CLI's WAV reader in Go, anything else decoded by the browser), the same file
      streamed in 1 s chunks, and the microphone through an AudioWorklet resampling to 16 kHz, committed and pending
      text live. `just wasm`, `just wasm-serve` (`cmd/whistle-wasm/serve`), `just wasm-test`; CI runs `wasm-test`
      without the model. Under Node every package but `cmd/whistle` and `internal/server` (sockets, 4 GB) passes, the
      whole `whistle` package with the 32-clip parity included; on 4 clips the page's path gives the JSON of
      `whistle transcribe --json --word-timestamps` byte for byte without the timings, and a 1 s stream the calls of
      `whistle stream --json` (`cmd/whistle-wasm/node/headless.mjs`); in Chromium the page gave the CLI's JSON, the
      stream, the cache and SHA-256 check, the browser decoder and the microphone worked. One thread, no SIMD, no FMA:
      TTFT 0.3–0.8 s and 24–27 tok/s under Node against 13–21 ms and 885–1121 tok/s native; two thirds of the time in
      the CQ lookup kernels (F-043).
- [ ] Optional: the Needle 3 text model (tool calling) on top of `internal/needle` and `.cact`, which reuses most of the work
      (2026-10-11) — moved to Phase 11, which plans it in sub-phases 11.1–11.9; 11.1 is done, 11.2 in part.
- [x] Optional: extract `internal/tensor`/`ops` into a shared module once go-whistle and go-pocket-tts have stabilised
      (2026-10-11) — `github.com/cwbudde/algo-tensor` (v0.1.0, public): the union of both copies, go-whistle's version
      where both have one, golden hashes equal on arm64, amd64 and js/wasm; its `MIGRATION.md` lists every identifier
      either project uses. go-whistle imports it and `internal/tensor`, `internal/ops` are gone: `Embed` and
      `Transcribe` (32 clips packed with word timestamps, 3 clips float, at depth 4 and float `Embed`) give the same
      bytes as before, with and without `-tags purego`. go-pocket-tts switched too
      (CWBudde/go-pocket-tts#46): it keeps one worker by default, and its WAVs are byte-identical to before at 1 and 4 workers.

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

- [x] Pin and download: `Cactus-Compute/needle3` @ `93e88ede` with SHA-256 next to Whistle's manifest, a download into
      `models/needle3/`, the pin in the upstream table.
      (2026-10-11) — `internal/hub/needle3.lock.json` (embedded, `hub.PinnedNeedle3`) pins `93e88ede867e…` with
      `needle3.cact` (35,335,380 B), `checkpoints/needle3.safetensors` (242,047,978 B), `config.json` (1,273 B) and
      `tokenizer/tokenizer.model` (126,520 B); the LFS digests equal the Hub's. `just fetch-needle3`
      (`tools/oracle/text fetch`, the same resuming, verifying `Client.Download`) lays them out as
      `models/needle3/{needle3.cact, needle3.safetensors, config.json, tokenizer.model}`. Tests: the pin, both
      manifests disjoint, an offline verify of the local files, and an opt-in live download of the three small files
      (`WHISTLE_HUB_LIVE=1`, all verified).
- [x] `needle3.safetensors` through `internal/safetensors`: the text config and run, typed and strict.
      (2026-10-11) — `Store.TextConfig()`/`TextRun()` decode upstream's `TransformerConfig` keys strictly (31 keys,
      the 7 `image_*` included), with `Orders`, `Rope`, `Hashes`, `Tables`, `SubDim` and `HadaN` supplying
      `from_saved`'s defaults. `TestNeedle3Checkpoint`: 64 F16 tensors, 121,019,861 parameters, the config above,
      the run (qapt, `needle3_rl.pkl`, `tokenizer_hash` = MD5 of `tokenizer.model`), step 10000; Whistle's `Config()`
      rejects it. The unchanged reader handles F16 (F-045).
- [x] `needle3.cact` through `internal/cact`: the header, the tensor map with the probe heads, every record against
      its safetensors source.
      (2026-10-11) — the reader opens it unchanged: 581 records, kv_window 256, global mask 4/9/14/19, 5 sites,
      6 tables, hada_n 1024, the same 28 codebook floats as Whistle. `cact.TextNames(f)` maps it from the header and
      the `heads.manifest`; `TextNamesFor(h, heads)` builds the same map without a file. The names are Whistle's
      wherever both models have the tensor (`TestTextNamesWhistleTrunk` maps whistle.cact's decoder trunk 1:1).
      `TestNeedle3NameMap` checks all 581 records as F-006 does: FP16 456/456 exact, CQ 122/122 = the CQ quantiser's
      output of the source (29 near-tie indices with margin ≤ 3.2e-8, 31 norms ±1 ulp), `hada_p1/p2` = numpy's
      `RandomState(11/13).permutation(1024)`, `heads.manifest` [2], the tokenizer 8192 pieces; all 64 sources used.
      Minimum cosines: CQ2 0.933338362129 (`layer17.out_proj`), CQ4 0.994432977749 (`confidence_head.proj`). The
      probe-head matrices are CQ4 though the plan says `default=2` (`quantize.py` `HEAD_BITS`). Golden map
      `internal/cact/testdata/needle3_names.txt` (F-045).
- [x] One source over both formats under the `.cact` names, so that `internal/needle`'s loaders read it.
      (2026-10-11) — `text.Source` (`Open`, `OpenCact`, `OpenSafetensors`) implements `weights.Source` without
      touching `internal/weights`: lazy decode and dequant, cache, `Release`, `Close`. The safetensors side derives
      the permutations, the heads.manifest and the embedding head's zero bias; its tokenizer is `ErrUnavailable`.
      `TestSourceBothFormats`: the same 581 names and shapes, 458 records bit-equal, 122 CQ within the F-045
      cosines. `text.Config` from either format is equal, and the header a text export of the safetensors config
      writes equals needle3.cact's field for field (`TestConfigBothFormats`).
- [x] Pure-Go sentencepiece over `tokenizer.model`: encode and decode equal to upstream on a multilingual corpus of
      tool-calling prompts; compared with `RefTokenizer` over the `.cact` blob.
      (2026-10-11) — `internal/spm`: a hand-written ModelProto reader and sentencepiece 0.2.2's BPE (identity
      normaliser, whole-piece user-defined symbols, score-ordered pair merges leftmost on ties, byte fallback, U+FFFD
      for invalid UTF-8, `" ⁇ "` for unk). `just text-tokenizer-fixture` (`testdata/text/tokenizer/needle3.jsonl`,
      88 KB, byte-identical re-run): 117 texts (46 `render_example` prompts and targets in en, de, fr, es, it, nl,
      pl, pt, tr, ja, zh, ko, ru, el, ar, hi plus edge cases and invalid UTF-8) with 5,255 ids, 80 decode sequences
      with 1,413 ids and a digest of every single id's decode: all equal to Python sentencepiece. The `.cact` blob
      has the same 8192 pieces, scores bit for bit and types. `RefTokenizer` (`internal/tokenizer`, read-only)
      encodes 109/109 valid texts and 20,000 random ones like sentencepiece; it differs only on invalid UTF-8 (8/8)
      and on decoding `<unk>` (5/80) (F-046).
- [x] **Accept:** every one of the 581 records equals its source by the F-006 checks (FP16 exact, CQ = the CQ
      quantiser's output of the source), both formats give the same geometry and names, and the tokenizer equals
      sentencepiece on every fixture text and id sequence.
      (2026-10-11) — met; numbers in the items above.

### 11.2 Float forward pass (`internal/text`, `internal/needle` additive; needs 11.1) — ✅ DONE (2026-10-12)

A translation of `SimpleAttentionNetwork.__call__` on the shared blocks.

- [x] HadamardMLP at a width below its power of two (768 in 1024), additively; Whistle's numerics unchanged.
      (2026-10-11) — `LoadHadamardMLP` takes n from `d1` and the model width from `cond_v`; below n a row runs
      zero-padded in a scratch buffer and the first `Dim` values are copied out (`Width()` reports n). At width = n
      the code path is the old one. Whistle unchanged: SHA-256 of `Transcribe` (packed and float, with word
      timestamps and at depth 4) and of `Embed` on all 32 clips equal the base commit's, natively and with
      `-tags purego`; `TestTranscribeMatchesEngine` 31/32, `TestEmbedMatchesEngine` mean 0.999827.
- [x] The forward pass: embedding, 20 mHC blocks with 5 engram sites, `final_norm`, the tied head; JAX fixtures
      (`scripts/text_forward_fixtures.py`, `testdata/text/forward/`) on tool-calling prompts with their greedy
      completions.
      (2026-10-11) — `text.Load(src, cfg)` and `(*Model).Forward(tokens, hidden)` on `needle.Block`, `MHC`,
      `Engram`, `Rope` and `LoadLinear`; nothing else in `internal/needle` changed. `just text-forward-fixtures` runs
      upstream's module in float32 (`highest` precision, `flash=False`, no quantisation) on three
      `render_example` prompts (en one tool, de two tools, en two calls; 88/156/156 tokens) plus their 32-token
      greedy completions, and a 6-token plain text with every layer (1.74 MB, byte-identical re-run). The float
      reference answers `{"name":"set_timer","arguments":{"minutes":10,"label":"pasta"}}` where the engine says
      `"Pasta"`. Measured against it from `needle3.safetensors`: final hidden states cosine ≥ 0.999999974 per
      position (496 positions), the last two positions' logits and every block's hidden state 1.000000000, the tiny
      case every layer and position 1.000000000 at |stream| up to 1.7e7, top-1 equal at 496/496 positions (smallest
      JAX margin 0.0016). The dequantised `.cact` weights alone (float activations) give final cosine 0.939–0.946
      and the reference's top-1 on 440/496 positions (F-045).
- [x] The banded mask of the local layers past 1024 positions, with a fixture longer than the window.
      (2026-10-12) — `needle.Attention.SetWindow`/`Block.SetWindow` narrow the causal mask to upstream's band
      (pos_q − pos_k) < w in both attend paths and in `Extend`, additively: Whistle's `Transcribe` and `Embed` hashes on
      the 32 clips equal `main`'s, natively and with `-tags purego`. `text.Load` sets 1024 on the 16 local layers, and
      `Forward` takes up to `max_seq_len` (`ErrTooLong`; `ErrWindow` is gone). The new fixture case `long_tools` (16
      tools, 1198 + 32 tokens, the needed tool ~1160 positions before the answer): final worst row 0.999999974, top-1
      1230/1230 (`TestForwardMatchesJAX`). Without the band, JAX and Go both fall to 0.8876 with 15 top-1 changes past
      1024 (`TestForwardBandMatters`), so the case tells the two masks apart (F-045).
- [x] Incremental decoding: `KVCache`/`EngramCache` per layer (`Block.Extend`), equal to the full forward at every
      position, fork for beams.
      (2026-10-12) — `text.State`, `Model.NewState`, `Model.Extend` (logits per new token) and `State.Fork`. On all four
      cases, one call, the prompt then one token per call, and uneven chunks (1000 + 30 + 200 across the window) give
      `Forward`'s logits at cosine 1.000000000000 at every position, and JAX's top-1 (`TestExtendMatchesForward`); a
      parent and two forks interleaved token by token each equal their own `Forward` (`TestExtendFork`); rejected
      calls leave the state unchanged (`TestExtendErrors`).
- [x] **Accept:** against float32 JAX, cosine ≥ 0.9999 for the final hidden states (per position), the per-block
      hidden states and the logits, and the next-token top-1 equal at every position of every fixture.
      partial: (2026-10-11) — met for sequences up to the 1024-position window (numbers above); the banded mask and
      the incremental path are open.
      (2026-10-12) — met on every fixture case, `long_tools` past the window included: final worst row 0.999999974,
      `logits_last` and `hidden_last` 1.000000000, top-1 equal at 1230/1230 positions; the incremental path gives the
      same logits.

### 11.3 Prompt format (`internal/text` or a new `internal/toolcall`; needs 11.1)

A translation of `render_example` and `Needle.__init__`, then fitted where the engine renders more than upstream
shows (multi-turn, tool results, retrieval).

- [x] Render system facts, tools and query as `render_example`, the tools JSON byte for byte as Python's
      `json.dumps(…, separators=(",", ":"), ensure_ascii=False)` with the caller's key order; the date fact of
      `_with_date_fact` behind an option (no clock in tests).
      (2026-10-12) — `internal/toolcall`: `Render`, `Encode` (`[BOS]` + sentencepiece), `ToolsJSON` (the caller's key
      order, a repeated key in its first place with its last value, Python's escapes, integers in full, floats in
      Python's repr, `Infinity` past the range) and `WithDateFact(system, now)`. `scripts/text_prompt_fixtures.py`
      runs upstream's `render_example` and `_with_date_fact` (fixed clock) on the 19 engine cases, the 4 forward
      prompts, 10 edge cases and 10 date cases (`testdata/text/prompt/render.json`, byte-identical on a re-run): every
      prompt, id sequence and date fact equal (`TestRenderMatchesPython`, `TestEncodeMatchesPython`,
      `TestWithDateFactMatchesPython`; F-048).
- [x] `needle_init` returns the token count of the static prefix (system and tools): match it on the engine corpus to
      pin down the engine's prefix rendering. The engine side is recorded (`testdata/text/engine/manifest.json`:
      314, 243 and 91 tokens for the three tool sets, +6 for a `; locale: xx-XX` suffix).
      (2026-10-12) — `toolcall.StaticPrefix`, `[BOS]` and the rendering up to `</tools>`, has `needle_init`'s count on
      all 19 cases (9 × 314, 6 × 243, 1 × 91, 2 × 320, 1 × 249; `TestStaticPrefixMatchesEngine`), and its ids start
      the whole prompt's. The rendering up to the query without `[BOS]` fits the counts too (the `\n` after `</tools>`
      is one token); the counts cannot separate the two, and both give the same input for a whole prompt (F-048).
- [ ] Multi-turn (`complete` after `complete`, `run`'s tool results, `<tool_result>`), `reset`, and the engine's turn
      limit and trimming (`--fail-input-overflow`).
- [ ] Tool retrieval: with more than 5 tools only the top 5 are rendered ("a built-in retrieval head"); find what
      ranks them (`needle_embed`, `--tool-index`), since `needle3.cact` has no embedding or router head.
- [ ] **Accept:** the static-prefix token count equals `needle_init`'s on every engine case, and a float greedy decode
      of the rendered prompt gives the engine's `function_calls` on at least the cases where the float model's
      top-1 is not a near-tie.
      (2026-10-12) — partial: the static-prefix half is met (above). The decode half waits for 11.4, by the user's
      decision: the float model's greedy calls (`TestFloatGreedyAgainstEngine`, the case's token budget) equal the
      engine's on 7 of 19 cases; the other 12 differ where the engine works after or around the decode, not at
      near-ties: it capitalises strings (5), re-resolves dates (2), rewrites a message (1), holds the arguments to the
      schema (3) and suppresses calls below confidence 0.1 (2; F-048).

### 11.4 Decoding and the response envelope (new `internal/toolcall`; needs 11.2, 11.3, the engine corpus)

Engine-only: fitted against `testdata/text/engine/`.

- [x] Engine corpus (`tools/oracle/text`, `testdata/text/engine/`, `just oracle-text`).
      (2026-10-11) — engine 3.2.0 loads the pinned `needle3.cact` alone and with `whistle.cact` (`needle_models` 3).
      19 cases, 20 turns: one and two calls, de/fr/es, dates, decimals, two refusals, a negation, an extraction
      schema, a tool result fed back, `--max 8`; every case passes a fixed `date: 2026-10-11 Sun 09:30` fact, since
      neither the CLI nor the library adds one (`--system` takes a file; literal text is silently ignored). Types:
      19 `call`, 1 `respond`; a refusal is `call` with no calls; 2 calls suppressed (confidence 0.092, 0.0315);
      `--max 8` gives `success:false, error_code:truncated`. With the timings dropped the answers are
      byte-identical across runs, `needle_complete` equals the CLI and `--serve`. Unlike Whistle, the answers depend
      on `--threads` (confidence in the 2nd–4th decimal, `reasoning` in 3 cases; the calls never), so the fixtures
      pin 4 threads. Also recorded: `--depth 2..20` (0/1 refused, 21 fails to load), three fused audio answers
      (each the text answer to `audio_text`), `needle_init`'s prefix counts, and `needle_embed` of text (3072
      L2-normalised floats = 4 queries × 768, the shape of a probe head's pooled output). 112 KB (F-047).
- [ ] Greedy against beam, the stop tokens (`<|im_end|>`, `</tool_call>`, EOS) and `--max`.
- [ ] The byte-level grammar compiled from the JSON schemas (types, `enum`, `const`, bounds, `format: date`, required
      keys, `anyOf`/`$ref`), applied as a token mask over byte-fallback pieces; the unconstrained `<think>` part.
- [ ] Parse the call block into `function_calls`; the envelope's `type` (`call`, `respond`, refusal as an empty
      call), `success`/`error`/`error_code`/`reason`, `suppressed_calls` below confidence 0.1, `validation`
      (`ungrounded`, `negation`), `triggers` regexes.
- [ ] `confidence`: `ConfidenceHead` (`probe_pool` over the embedding and the 20 block outputs, a translation) and
      the min with the call's decode probability (engine).
- [ ] **Accept:** on the engine corpus, `type` and `function_calls` (name and arguments, as parsed JSON) equal the
      engine's on ≥ 95 % of the cases with every miss explained by a log-probability margin, `reasoning` equal on
      ≥ 90 %, `confidence` within 0.02.

### 11.5 Native quantised path (`internal/text`, `internal/weights` or a packed `text.Source`, `internal/cq`; needs 11.2)

- [ ] Serve the 122 CQ records packed (`cq.Matrix`) and run the model on them, as 9.2 does for Whistle.
- [ ] The engine's int8 activations and KV (`act_bits` 8, `kv_bits` 8, `Int8` of F-041) and its KV window of 256.
- [ ] **Accept:** with the `.cact` weights, `function_calls` equal the engine's at least as often as with the float
      weights; memory and speed against the engine (`just bench`-style), packed weights under 40 MB.

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
- [ ] **Accept:** cosine ≥ 0.99 against `needle_embed` on the recorded inputs.

## Phase 12 — Public release & follow-ups from the first write-up

The repository went public on 2026-10-11. Writing the first blog post about it (running the CLI from the module path
on a Linux/amd64 laptop, an AMD Ryzen 5 4600H, and hosting the web demo under a sub-path) turned up the gaps below.
The `de-paare-missed` follow-up stays in Phase 10, and the Needle 3 text model is Phase 11.

### 12.1 Docs and release

- [x] README after going public: drop "the repository is not public yet" (Quickstart, Go API) and the `replace`
      instructions; install with `go install github.com/cwbudde/go-whistle/cmd/whistle@latest` and use the library
      with a plain `go get github.com/cwbudde/go-whistle`. The status note and "Limits → Speed" still say Phase 9 is
      under way. Phase 9 is done, so point them at the numbers in `bench/RESULTS.md` and at 12.3 instead.
      **Accept:** `go install …@latest` and the README's Go API program work from a clean module without a clone.
      (2026-10-11) — Quickstart installs with `go install …@latest` and fetches the two example clips with `curl`
      when there is no clone; the Go API section is one `go get github.com/cwbudde/go-whistle/whistle`. The package
      path, not the module root: with the default `-mod=readonly`, `go get` of the root leaves go.sum without the
      dependencies of `whistle` and `go run` fails. The status note and "Limits → Speed" cite `bench/RESULTS.md` and
      12.3. Checked in a temporary `GOPATH`, `GOMODCACHE` and `GOCACHE` outside the repository: `@latest`
      (`v0.0.0-20261011164734-4aafa14c5ca7`, main before this change) installs, `whistle download`, `doctor` and
      the README's `transcribe` lines give the README's output, and the Go API program in a new module prints the
      README's output, with `GOFLAGS=-mod=mod` and without.
- [x] A first tagged release (`v0.1.0`) with short release notes (what works, parity, limits), so `@latest` resolves
      to a version and not to a pseudo-version of `main`. Optional: prebuilt CLI binaries for linux/amd64,
      linux/arm64, darwin/arm64 and windows/amd64 on the GitHub release.
      (2026-10-11) — partial: the notes are in `docs/releases/v0.1.0.md`, and `.github/workflows/release.yaml` turns
      a pushed `v*` tag into a release with them and the CLI for the four platforms (cross-compiled on one Linux
      runner, `.tar.gz`/`.zip` with LICENSE and README.md, `SHA256SUMS`); built locally, the darwin/arm64
      `whistle doctor` passes. The tag follows the merge.
      (2026-10-12) — `v0.1.0` is tagged on `24828b8`, but its release workflow stopped on the arm64 builds:
      `internal/needle/exp_arm64.s` uses Go 1.27 mnemonics while `go.mod` said Go 1.25 (F-035). The tag stays, since
      Go versions are immutable. #49 requires Go 1.27, adds `just check-cross` to CI and the `v0.1.1` notes, and
      `v0.1.1` (`fff05e3`) is the release:
      https://github.com/CWBudde/go-whistle/releases/tag/v0.1.1, with the four archives and `SHA256SUMS`. The
      darwin/arm64 and linux/amd64 archives match `SHA256SUMS`, and their `whistle doctor` passes on an M5 Pro and on
      a Xeon. proxy.golang.org serves `v0.1.1`, and `go install …/cmd/whistle@latest` in a clean `GOPATH` installs
      `v0.1.1`.
- [x] Link the hosted web demo (12.2) from the README's "Web demo" section.
      (2026-10-11) — done with #43: https://cwbudde.github.io/go-whistle/.

### 12.2 Web demo

- [x] Insecure contexts: `fetchModel` (`web/whistle-client.js`) always calls `crypto.subtle.digest`, which exists only
      in secure contexts (https, localhost). Served over plain http (a LAN IP, an http-only host), "Fetch and load"
      fails with "Cannot read properties of undefined (reading 'digest')", and so does picking a local `.cact`. Fall
      back to a SHA-256 computed in Go (an exported `whistle.sha256(bytes)` in `cmd/whistle-wasm`, from
      `crypto/sha256`) when `crypto.subtle` is missing, and treat a missing Cache Storage as "no cache", as
      `caches === undefined` already is. **Accept:** `just wasm-test` (or a headless run against
      `http://<lan-ip>:<port>/`, where `isSecureContext` is false) loads the model with the SHA-256 checked and
      transcribes a clip.
      (2026-10-11) — `whistle.sha256(bytes)` hashes in 1 MiB blocks, so the model is never copied into Go whole;
      `sha256Of` in the client uses Web Crypto where there is one and else hands the buffer to the worker, which
      gives it back with the digest (transferred both ways, no copy). Both the fetch and the local-file path use it;
      a missing Cache Storage was already a miss. Over plain http the model status says the model is neither cached
      nor hashed by Web Crypto, and the microphone (no `navigator.mediaDevices`, no `audioWorklet` on the context) is
      disabled with a status line saying it needs https or localhost. `just wasm-test` now runs the page's path once
      more without `crypto.subtle` and `caches`: Go hashes 3 MiB of bytes and gives them back unchanged, a wrong
      hash is rejected, and with the model the pinned SHA-256 is checked in Go and `ls-other-3080-5040-0018` gives
      the secure run's JSON (without timings). Also driven in headless Chrome against `http://192.168.213.61:8090/`
      (`just wasm-serve`, `isSecureContext` false): "Fetch and load" from Hugging Face with the SHA-256 checked in
      Go, a wrong hash rejected, the local `.cact` picked, and the German FLEURS clip transcribed with the CLI's JSON
      (timings aside).
- [x] A hosted demo: a GitHub Pages workflow that runs `just wasm` and publishes `web/` (the model still comes from
      Hugging Face, whose CDN sends CORS headers). Serve `.wasm` as `application/wasm` so `instantiateStreaming` is
      used. The build is 6.8 MiB, or 1.8 MiB gzipped.
      (2026-10-11) — `.github/workflows/pages.yaml` (#43) runs `test-wasm`, builds with `just wasm` and deploys `web/`
      on every push to `main`. GitHub Pages serves over https, so the secure-context item above does not bite there.
- [x] Stream result line: after `Stop`, the page shows "last pass 0 ms", the stop's `pass_ms`. Show the last real
      pass (or the total) instead.
      (2026-10-11) — `StreamSession` counts the passes (the `streamProcess` calls; the stop runs none) and keeps the
      last pass's time and the sum, so the line reads e.g. "7 passes, the last 2254 ms, 9.65 s in all" after the
      stop too (headless Chrome, the German clip). `just wasm-test` checks it on a fake and on a real 1 s stream.

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
- [x] Benchmarks on linux/amd64: `bench/RESULTS.md` has only the Apple M5 Pro. Add a `just bench` run on an x86
      machine (with the CPU, cores and governor recorded), and the engine side where a linux/amd64 engine build
      exists. On the 4600H laptop, `transcribe --json` reported a TTFT of 101–108 ms and 150–270 tok/s on five fixture
      clips, about 5× the M5 Pro's TTFT, under background load and not measured with `just bench`.
      (2026-10-12) — `bench/RESULTS-linux-amd64.md`, from `bench/results/linux-amd64/` (`just bench -procs 16` and
      `just oracle-bench` with `-go`, `-engine`/`-out` and `-report`). The machine is 2× Xeon Gold 5520+, 80 vCPUs, a
      KVM guest; the governor row says "none (no cpufreq; kvm guest)", which `bench` now records on linux. The engine
      ran its 3.2.0 `linux-x86_64` build. Median over the 32 clips, Go packed against the engine library: TTFT 47.3
      against 160.9 ms (0.29×), and decode 547 against 523 tok/s (1.05×). With 4 workers, as the engine uses: 51.4 ms
      and 363 tok/s (0.69×). The float path decodes at 130 tok/s, since the GEMM is in Go. On amd64, parity is the
      same as on arm64. `docs/FINDINGS.md` F-040 "linux/amd64".

### 12.4 Remaining speed gap (optional)

- [ ] TTFT against the engine: 18.8 against 6.9 ms (2.72×), decode 1085 against 1499 tok/s (0.72×), median over the
      32 clips (F-040). Profile where the TTFT goes (front end, stem, encoder, memory, first decoder steps) and take
      up the open levers of F-037 and F-041. Engine speed is still not a hard goal (Risks); the aim is a measured
      breakdown and whichever levers pay off without moving the parity numbers.
- [ ] Adopt algo-tensor v0.3.0 (its 1.3, algo-tensor PR #7): `ops.Conv2D` with groups = 1 runs through
      `Conv2DPlan` (im2col and `MatMulTransBBias`) on arm64 and js/wasm, not on amd64. Whistle's stem
      (`internal/encoder/stem.go`) is such a conv: algo-tensor measured it 9.51 → 3.93 ms on an Apple M5 Pro (one
      worker), and its `Conv2D` bits change there (golden case max |Δ| 9.5 × 10⁻⁷). **Accept:** with v0.3.0, the
      parity numbers on arm64 stay at least as good as before (`TestTranscribeMatchesEngine`, the keyword fixtures,
      the depth ladder, the stream decisions, the LibriSpeech WER and the `Embed` cosine against the engine), the
      `Embed` and `Transcribe` outputs that change are named with their largest difference and their fixtures
      regenerated in the same change, amd64 outputs are unchanged, and `just bench` gives the TTFT before and after.
      js/wasm changes too, and CI skips its model-gated tests: with the model, `just wasm-test ""` (no `-short`) passes,
      the `whistle` package's engine parity under Node and the headless page's JSON against the CLI's on its 4 clips
      included.
- [ ] js/wasm speed (F-043 "Open"): about 22 tok/s and a TTFT of 0.3–0.8 s on one thread without SIMD. Try several
      workers, each with its own 17 MB model, to run the beams or the encoder blocks in parallel. A hand-written wasm
      SIMD module for the LUT kernels would leave pure Go; decide first whether it is worth that.

---

## Unknowns to resolve against the oracle

| #   | Unknown                                                                                     | Phase   | Resolved by                       |
| --- | ------------------------------------------------------------------------------------------- | ------- | --------------------------------- |
| U1  | Mel log flavour, floor, normalisation; the "250–3500 Hz" claim vs. the full-band filterbank | 3/4     | encoder cosine vs. `needle_embed` |
| U2  | Stem strides, padding, activations                                                          | 4       | frame count + encoder cosine      |
| U3  | Encoder conv module (GLU?), macaron scale, `pe_gate` usage, positions                       | 4       | encoder cosine                    |
| U4  | Meaning of the audio manifest values `3246, 1`                                              | 4/5     | inspection + experiments          |
| U5  | Whether the encoder and decoder share the Hadamard permutations                             | 2/4     | block parity / encoder cosine     |
| U6  | Cross-attention position in the block — resolved: after the self-attention, F-021           | 5       | teacher-forced transcript score   |
| U7  | Decoder prompt format, language detection — resolved: `[<s>]`, argmax language, F-023/F-024 | 5       | language + teacher-forced score   |
| U8  | Beam length normalisation, EOS rules — resolved: Whisper's at 5 beams, Σ log p / len, F-026 | 6.2     | transcript parity                 |
| U9  | Keyword bias weight and revocation — best fit: 2 nats per token, full revocation, F-029     | 6.3     | keyword fixture parity            |
| U10 | Silence gate — resolved: p99 − p10 of 250–3500 Hz mel frame energy ≤ 5 dB, F-031            | 6.6     | threshold sweep                   |
| U11 | Timestamps — Whisper's alignment, last half of the layers; first start fitted, F-033        | 7.2–7.3 | timestamp parity                  |
| U12 | Stream commit rule — resolved: LocalAgreement-2, 15 s trim, F-032; > 30 s: last 30 s, F-036 | 8.3–8.4 | stream parity                     |

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
