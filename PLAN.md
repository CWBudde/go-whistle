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

## Phase 4 — Encoder (reverse engineering)

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
- [ ] Encoder block candidates:
  - FFN residual scale 1 or ½ (macaron)
  - conv module: GLU over pw1's 1024 channels or not; SiLU after dw; any normalisation between dw and pw2
  - attention: full non-causal; RoPE in the encoder or not
  - where `pe_gate` applies: a sinusoidal or RoPE mix-in gated by σ(pe_gate), or on the stem output
  - how mHC wraps the block (same as the decoder, `enc_lanes` 4)

  (2026-10-10) — partial: everything but `pe_gate` is resolved by score (F-017). FFN ½ (1: −0.622692); GLU a·σ(b)
  over pw1's halves ("no GLU" ruled out by the 512-wide `dw`; a·silu(b) −0.403596, σ(a)·b −0.978467); SiLU after
  `dw` (GELU −0.005362, none −0.820849); `conv_out_norm` between `dw` and the SiLU (after it −0.185564, after `pw2`
  −0.896173); centred kernel-9 `dw` (causal −0.907688); macaron order (conv first −0.415660); full non-causal
  attention with half-split RoPE θ = 1e5 (θ 1e4 −0.029625, none −0.221335); mHC exactly as the decoder, 4 lanes
  (plain stack −0.678340); shared `hada_p1/p2`, so U5 is settled (split −0.980344, identity −0.977999). Open:
  `pe_gate`. σ(pe_gate) = 0.0022 makes a sinusoid −70 dB below the stem output (RMS 0.0016 against 4.6–5.1), so
  adding it to the stem output either way changes the score by ≤ 1e-6. A RoPE mix replaces the winning RoPE
  (−0.221535). `Best()` leaves it unused; the embeddings cannot decide it.

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

## Phase 5 — Decoder + greedy decoding — ✅ DONE (2026-10-10)

- [x] Cross-attention: per-layer `cross_norm → q`, encoder `k/v` projected once per clip and cached, q/k norm,
      sigmoid gate, `post_cross_norm`, σ(`cross_gate`) residual. Decide where it sits in the block (after
      self-attention is the most likely) by fixture accuracy.
      (2026-10-10) — between the self-attention and the MLP, inside the mHC block function
      (`needle.Block.Extend` with `Hooks.AfterAttn`): `x += σ(cross_gate)·post_cross_norm(out_proj(softmax(q̂k̂ᵀ/√48)v
    · σ(gate_proj(cross_norm(x)))))`, 8 heads of 48/64, per-head q/k ZCRMSNorm, no RoPE, over `needle_embed`
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
      `TestGreedyMatchesEngine` (three clips exact, no tag) (F-021 to F-024).

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
| U6  | Cross-attention position in the block — resolved: after the self-attention, F-021           | 5     | teacher-forced transcript score   |
| U7  | Decoder prompt format, language detection — resolved: `[<s>]`, argmax language, F-023/F-024 | 5     | language + teacher-forced score   |
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
