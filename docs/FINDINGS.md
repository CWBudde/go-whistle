# Findings

Every reverse-engineered decision of the port, with the hypothesis, the alternatives tried, the metric against the
oracle and the winner. Code comments cite entries by id (`F-001`).

Oracle: Needle 3 engine 3.2.0 (`Cactus-Compute/needle3` @ `2ae11323dc`, macos-arm64) with `whistle.cact` from
`Cactus-Compute/whistle` @ `b358ddad`.

## F-001 — `needle_embed` speech output shape (2026-10-09)

**Question.** `needle.h` says a speech embedding is "one row of that width per 80 ms frame, flattened" but gives
neither the width nor how a partial frame counts.

**Measured.** A null-output call returns the float count without computing. On synthetic PCM:

| samples | 1280 | 1281 | 2560 | 16000 | 32000 | 48000 | 160000 |
| ------- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| floats  | 512  | 512  | 1024 | 6656  | 12800 | 19456 | 64000  |
| frames  | 1    | 1    | 2    | 13    | 25    | 38    | 125    |

**Alternatives.**

- `ceil(samples / 1280)`: rejected, because 1281 samples give 1 frame, not 2.
- `floor(samples / 1280)`: rejected, because 16000 samples give 13 frames, not 12.
- `(samples + 640) / 1280` in integer division (round half up): fits all seven lengths.

**Winner.** The width is **512 floats per frame**, and frames = `(samples + 640) / 1280` in integer division. That is
an 80 ms hop at 16 kHz. All 32 corpus clips (`testdata/oracle`) return a multiple of 512 floats.

The engine does not say whether 512 is the encoder width or a projection of it; Phase 4 settles that against the
safetensors weights. `tools/oracle` stores each embedding as F32 `embed` `[frames, 512]` in
`testdata/oracle/<clip>.embed.safetensors`.

## F-002 — `needle` CLI transcription output (2026-10-09)

**Measured.** `needle --model whistle.cact --audio clip.wav --audio-word-timestamps` prints one JSON object on stdout
and nothing on stderr:
`{"text", "language", "words": [{"word", "start", "end", "probability"}], "ttft_ms", "decode_tps"}`.

- Times are in seconds with two decimals. All 918 word boundaries in the corpus lie on the 80 ms grid, the same hop
  as F-001.
- Probabilities have three decimals.
- `words` keeps punctuation attached to the word (`"an,"`, `"Nichtgewachsen."`).

**Determinism.** `ttft_ms` and `decode_tps` are wall-clock measurements, so `tools/oracle` drops them. With those
removed, two runs of the whole corpus give byte-identical JSON and embeddings (macos-arm64, default threads).

**Language detection** matched the corpus language on 32 of 32 clips.

## F-003 — `.cact` header vs the safetensors config (2026-10-09)

**Question.** The `.cact` header (`export.py` `_HDR_FMT "<48If"`) is written for Needle's text model
(`num_heads`, `hada_n`, `engram_*`), while `whistle.safetensors` stores the Whistle config as JSON. Which header field
corresponds to which config key, and does the header describe the encoder at all?

**Measured.** `internal/cact` `TestWhistleCactMatchesConfig` reads both files at `b358ddad`:

| `.cact` header                  | value        | safetensors config / run      |
| ------------------------------- | ------------ | ----------------------------- |
| `vocab`                         | 8199         | `text` + 7 language tokens    |
| `d_model`                       | 512          | `dim`                         |
| `num_kv_heads`                  | 2            | `kv_heads`                    |
| `num_layers`                    | 8            | `dec_layers` (= `enc_layers`) |
| `qk_head_dim`, `v_head_dim`     | 48, 64       | `qk_dim`, `v_dim`             |
| `max_seq_len`                   | 320          | `max_tokens`                  |
| `hada_n`                        | 512          | next power of two of `dim`    |
| `mhc_lanes`                     | 4            | `lanes` (= `enc_lanes`)       |
| `qkv_conv_taps`                 | 3            | `taps`                        |
| `engram_slots`                  | 18432        | `slots`                       |
| `num_engram_tables`             | 4            | `len(orders)` × `hashes`      |
| `engram_sub_dim`                | 128          | `dim` / tables                |
| `engram_conv_dilation`          | 3            | `max(orders)`                 |
| `engram_orders`, `engram_sites` | [2,3], [3,7] | `orders`, `sites`             |
| `rope_theta`                    | 1e5          | `rope`                        |
| `kv_bits`                       | 8            | `run.kv_bits`                 |

The header holds three values the config does not state: `num_heads` = 8, `engram_conv_taps` = 4 (the `export.py`
constant `ENGRAM_CONV_TAPS`) and `kv_window` = 0. `out_vocab`, `sliding_window`, the global mask and
`engram_seed_heads` are all 0.

**Conclusion.** The header carries the fields `export.py` writes for Needle's text model, so it describes the
**decoder**: vocab, engram and the Q/K/V taps exist only there. Whistle's encoder and decoder both have 8 layers and 4
mHC lanes, so `num_layers` and `mhc_lanes` cannot tell the two apart; the decoder reading is the one consistent with
the rest of the header. The encoder (`conv_kernel`, `stem`, `stem_stages`) and the front end (`mels`, `rate`, `fft`,
`win`, `hop`) are not in the header. They appear in the 13-float FP32 tensor #237,
`[3246, 1, 8192, 8, 4, 9, 128, 3, 80, 16000, 512, 400, 160]`: from the third value on it equals `text`,
`enc_layers`, `enc_lanes`, `conv_kernel`, `stem`, `stem_stages`, `mels`, `rate`, `fft`, `win` and `hop` (asserted by
the same test). F-006 explains the first two values: a magic and a format version.

Also measured: the header is 196 bytes, not the 120 the `export.py` docstring states; the codebooks are stored already
scaled by 1/√128 (cb2 = ±0.0399, ±0.1333); blobs are 64-byte aligned and contiguous, and the last one ends exactly at
EOF. Whistle uses only CQ2 (145 tensors) and CQ4 (7), both with group 128; there is no CQ3, ternary or binary tensor.

## F-004 — CQ dequantisation (2026-10-09)

**Question.** `export.py` documents the CQ blob, and `_cq_unpack` decodes it, but only `_cq_pack` at
`WEIGHT_BITS` = 4 writes one. Does Whistle's `.cact` follow that code exactly: which codebook, which Hadamard order,
which bit and norm layout? And what should the Go reader do for the widths Whistle does not use?

**Codebook.** `_cq_codebook_np(bits, group)` draws 400,000 samples from numpy's `RandomState(0)` and runs 200 Lloyd-Max
iterations. Rerunning that code with numpy reproduces all 28 header floats of `whistle.cact` bit for bit as
`_cq_codebook_np(2|3|4, 128)`. Winner: `File.CQ` uses the stored codebook and does not port the RNG; the tests carry
the float64 unit levels. Caveat: the header holds one codebook, scaled by 1/√group of the export. `_pack_cact` writes
a single group size for every CQ tensor, and every Whistle CQ tensor has group 128, but a file that mixed group sizes
would decode its 2–4-bit tensors with the wrong scale. The 1-bit (±√(2/π)) and ternary (±1.2240064, 0) levels are not
stored. They are scaled per record, as `_cq_codebook_np` does.

**Layout.** Measured as the cosine between `CQ(0)` (CQ4, `[8199, 512]`) and the F32 `embedding/embedding` of
`whistle.safetensors`:

| decoding                                                                 | cosine      |
| ------------------------------------------------------------------------ | ----------- |
| `_cq_unpack`: LSB-first, low nibble = even column, norms `[out][groups]` | **0.99537** |
| high nibble = even column                                                | −0.01339    |
| norms stored group-major, `[groups][out]`                                | 0.91511     |

The winner's worst row cosine is 0.98980. For a 4-bit Gaussian Lloyd-Max quantiser, the expected cosine is
√(1 − 0.0095) ≈ 0.9952. The unit round trip (`TestCQ_RoundTrip`) also lands on the textbook distortions, with a
relative error of 0.60 / 0.34 / 0.17 / 0.09 for 1 / 2 / 3 / 4 bits.

**Rotation.** `_cq_hadamard_np` builds Sylvester's matrix (`np.block([[H, H], [H, -H]])`) in natural order. It is
symmetric and orthonormal, so the inverse of the pack rotation `g @ H` is `@ H` again. `fwht` applies the ±1 part with
in-place butterflies; the 1/√group factor is folded into the norm. The explicit matrix stays in the tests as the
reference. The ternary one-hot tests pin the row order by hand.

**Numerics.** float32, as upstream. A float64 matrix reference agrees with `CQ` within the float32 bound
γ(log2(group) + 3) · Σ|x| / √group. On `whistle.cact`, numpy's `_cq_unpack` and `CQ` differ by about 1e-8 of the row
norm, where the test allows 1e-5. All 145 CQ2 and 7 CQ4 tensors are finite, with an RMS between 1e-3 and 10, and the
largest |w| over all of them is 69.29.

**Widths outside Whistle.** 1-bit and ternary are implemented as `_cq_unpack` does them: eight 1-bit indices per
byte, and ternary crumbs 3/0/1 → trits 0/1/2. They are only covered by unit tests. Two inputs fail upstream, so they
are errors here: crumb 2 (numpy raises `IndexError` on the 3-level codebook), and a group that is not a power of two
(`_cq_hadamard_np` overshoots the group and the matmul fails).

## F-005 — Tokenizer blob, decode rules and the sentencepiece fixture (2026-10-09)

**Question.** Does the RAW blob (`.cact` #680) follow `export.py` `_tokenizer_blob`? `Cactus-Compute/whistle` ships no
`tokenizer.model`, so what is a trustworthy sentencepiece reference for decode? And where sentencepiece and upstream
`RefTokenizer` disagree, which one does the port follow?

**Measured (blob).** 109,231 bytes in the `_TK_HDR`/`_TK_REC` layout, with the last record ending exactly at the end
of the blob. Header: 8199 pieces, pad/eos/bos/unk = 0/1/2/3, `add_dummy_prefix` 0, `byte_fallback` 1. Types: 7908
NORMAL, 256 BYTE (ids 28–283, `<0x00>`…`<0xFF>`), 31 USER_DEFINED, 3 CONTROL (0–2) and 1 UNKNOWN (3). NORMAL scores
are minus the merge rank, 0 to −7907, all distinct; every other piece scores 0. The USER_DEFINED pieces are ids 4–27
(24 chat markers, including `<image>`, `<speech>`, `<tts>` and `<imagen>`) and the language tokens 8192–8198
(`<|en|>`, `<|de|>`, `<|fr|>`, `<|es|>`, `<|it|>`, `<|nl|>`, `<|pl|>`).

Upstream `_tokenizer_blob` marks only the 16 `CHAT_MARKERS` as USER_DEFINED. It would have typed `<image>`,
`<speech>`, `<tts>`, `<imagen>` and the language tokens as NORMAL, so the pinned exporter did not write Whistle's blob.
The blob's types match the sentencepiece model's own flags instead: needle3's `user_defined_symbols` lists all 24
markers. As a result, `Encode` cuts `<speech>` and the language tokens out whole.

**Fixture route.** `scripts/tokenizer_fixture.py` rebuilds a sentencepiece BPE `ModelProto` from the blob's pieces,
scores and types. Its normalizer and trainer settings copy needle3's model: `identity` normalizer, no dummy prefix, no
whitespace collapsing, `byte_fallback`, `split_by_whitespace` and a maximum piece length of 16. The script asserts that
the loaded model reports the blob's piece, score and type back for all 8199 ids. needle3's `tokenizer.model` (@
`2ae11323dc`, sha256 `97dfd566…1b25`) served as a cross-check: the script's `--needle3` flag runs that comparison and
only prints, so the fixture does not depend on the download. Results:

- Pieces, scores and types: 0 of 8192 differ from blob ids 0–8191.
- Decode: 104/104 fixture sequences that use only shared ids are equal, and so are 8192/8192 single ids.
- Encode: 24/24 samples without language tokens are equal.

The rebuilt model and the published needle3 model therefore agree everywhere they overlap. The rebuilt model also
covers the 7 language tokens. The decode logic is sentencepiece's own (0.2.2).

One setting deviates on purpose: `unk_surface` is `""`. sentencepiece prints `" ⁇ "` for `<unk>` by default, while
`RefTokenizer` and PLAN.md skip it. Only sequences containing id 3 are affected.

**Disagreements.** `RefTokenizer.decode` differs from sentencepiece on 11 of the 109 fixture sequences. All 11 are
byte-fallback corner cases:

1. Invalid UTF-8. sentencepiece writes one U+FFFD per invalid byte (`E2 82` → `��`). Python's `errors="replace"` writes
   one per maximal subpart (`E2 82` → `�`).
2. Byte runs. sentencepiece decodes each run of consecutive BYTE pieces on its own, so a CONTROL or UNKNOWN piece
   between two runs ends the first one (`<0xC3> <pad> <0xA9>` → `��`). `RefTokenizer` joins all bytes first (→ `é`).
3. `▁` spelled by bytes. sentencepiece turns `▁` into a space only in piece surfaces (`<0xE2><0x96><0x81>` → `▁`).
   `RefTokenizer` replaces it after the join (→ a space).

**Winner.** `Decode` follows sentencepiece, as the Accept criterion asks; for case 1 that is also what
`utf8.DecodeRune` gives. Well-formed output never reaches these cases. The engine's behaviour on them is unknown, so
revisit this if a transcript ever differs on one. `Encode` is a straight port of `RefTokenizer`. Unlike
sentencepiece, it does not split at whitespace before the BPE, but on all 25 fixture samples (Latin, accented
European, Cyrillic, CJK, Arabic, Hebrew, Devanagari, emoji, markers, language tokens, whitespace and control
characters) `RefTokenizer` and sentencepiece give identical ids. Go matches `RefTokenizer` on 25/25 and round-trips all
25, plus 11 more texts in `TestWhistleRoundTrip`. Text containing a literal `▁` cannot round-trip, because encode
cannot tell it apart from an escaped space.

The Go side differs from upstream in three more places:

- `Parse` is stricter than `parse_tokenizer_blob`. It rejects trailing bytes, unknown types, malformed or repeated byte
  pieces, out-of-range special ids, and byte fallback without all 256 byte pieces.
- `Encode` carries invalid UTF-8 input, which a Python `str` cannot hold, byte by byte into byte pieces.
- `Decode` returns an error for an id out of range, where sentencepiece raises.

Regenerate with `just tokenizer-fixture [--needle3 tokenizer.model]`. The model-gated tests in `internal/tokenizer`
read `testdata/tokenizer/whistle.json`.

## F-006 — Whistle's `.cact` tensor order, transforms and per-tensor checks (2026-10-09)

**Question.** `.cact` records are nameless and positional. Upstream's public exporter (`export.py` `_tensors`) writes
the Needle 3 text model only, and Whistle's audio exporter is not published. Which safetensors tensor feeds each of the
681 records, through which transform? And how is each record shown to be correct?

**Method.** A throw-away Go search scored every record against every shape-compatible candidate by cosine: each of the
116 safetensors tensors whole, each layer slice of the 80 stacked `[8, …]` tensors (unit axes squeezed), plus every
2-D candidate transposed and every mHC φ in export.py's lane-major layout. The search assumed no order. Every sourced
record had one clear winner:

| record type | winner cosine            | best runner-up cosine                                                  |
| ----------- | ------------------------ | ---------------------------------------------------------------------- |
| FP16 vector | ≥ 0.99999996             | 0.999983 (`cross01.k_norm` vs `cross_attn/q_norm[1]`)                  |
| CQ2         | 0.9319–0.9401            | 0.069                                                                  |
| CQ4         | 0.9943–0.9954            | 0.014                                                                  |
| FP16 scalar | ±1 (cosine is only sign) | decided by value: the FP16 of the winner equals the record bit for bit |

The winners form the order below. `internal/cact` `WhistleNames()` encodes it and `testdata/whistle_names.txt` pins it.

| index   | name                     | contents                                                                                                                                 |
| ------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 0       | `embedding`              | `embedding/embedding`                                                                                                                    |
| 1–216   | `layerNN.*`, 8 × 27      | exactly `_tensors`' per-layer order with `qkv_conv_taps` = 3, from `stack/layers/block`                                                  |
| 217–225 | `mhc_*`                  | `stack/mhc_*`, a/b whole, φ lane-major                                                                                                   |
| 226–227 | `hada_p1`, `hada_p2`     | derived                                                                                                                                  |
| 228–235 | `engram0.*`, `engram1.*` | `engrams_0` (site 3) and `engrams_1` (site 7): tables, key_proj, value_proj, taps                                                        |
| 236     | `final_norm`             | `stack/final_norm/scale`                                                                                                                 |
| 237     | `audio.manifest`         | derived, where `_tensors` would put `heads.manifest`                                                                                     |
| 238–317 | `crossNN.*`, 8 × 10      | `cross_norm`, `cross_attn/{q,k,v,gate,out}_proj`, `q_norm`, `k_norm`, `post_cross_norm`, `cross_gate`, named with the decoder's suffixes |
| 318     | `pe_gate`                | `pe_gate`, stored `(1,)`                                                                                                                 |
| 319–662 | `encNN.*`, 8 × 43        | six norms, q/k/v/gate/out, q_norm, k_norm, attn_gate, pw1, pw2, dw, `hadamard_mlp_0` (13), `hadamard_mlp` (13)                           |
| 663–671 | `enc.mhc_*`              | `encoder/mhc_*`                                                                                                                          |
| 672     | `enc.final_norm`         | `encoder/final_norm/scale`                                                                                                               |
| 673–678 | `stem.*`                 | `w`, `dw_1`, `pw_1`, `dw_2`, `pw_2`, `out`                                                                                               |
| 679     | `audio.mel_filters`      | derived                                                                                                                                  |
| 680     | `tokenizer`              | RAW (F-005)                                                                                                                              |

The encoder's six norms come in macaron execution order: `pre_hada_norm_0`, `ZCRMSNorm_0`, `post_attn_norm`,
`conv_norm`, `conv_out_norm`, `pre_hada_norm`. All 116 safetensors tensors are used; none is training-only. Every HadamardMLP, the encoder's included, keeps
`_tensors`' inner order (d1, d2, b2, d3, d4, w1a … w3b, cond_v, cond_u). The decoder layers come first and the audio parts follow, each as its own block of eight
layers.

**Transforms.** Four transforms cover every record (`cact.Transform`, applied by `Entry.Apply` after the layer slice):

- `reshape`: the values stay in order. Unit axes are squeezed (`dw` `(9, 1, 512)` → `(9, 512)`), the 3×3 stem kernels
  `(3, 3, 1, 128)` become `(9, 128)`, the engram tables `(4, 18432, 128)` become `(73728, 128)`, and layer scalars
  plus `pe_gate` are stored as `(1,)`.
- `transpose`: every Flax Dense kernel `(in, out)` is stored `(out, in)`. This includes `pw1`/`pw2`, the stem's 1×1
  kernels and `stem/out` `(1280, 512)` → `(512, 1280)`.
- `lane-major`: an mHC φ `(L, n·C, lanes)` is stored as `transpose(0, 2, 1).reshape(L·lanes, n·C)`, as `_tensors`
  does.
- `derived`: records 226, 227, 237, 679 and 680 have no safetensors source.

Read untransposed, no square kernel scores better than the CQ runner-ups (≤ 0.069), so the transposition is not in
doubt.

**Derived records.** Each one is checked by equality:

- `hada_p1`/`hada_p2` equal `np.random.RandomState(11 / 13).permutation(512)` (`_hada_perms` without the split, since
  the config has no `ladder_widths`). A Go port of MT19937 (`init_genrand`) and numpy's legacy shuffle
  (`random_interval` with masked rejection) reproduces both bit for bit. It is pinned to numpy 2.5.3 output and to the
  C++ standard's 10000th `mt19937` value. One pair serves every HadamardMLP, encoder and decoder alike.
- `audio.mel_filters` equals `librosa.filters.mel(sr=16000, n_fft=512, n_mels=80)` with Slaney scale and Slaney norm,
  computed in float64 with numpy's `rfftfreq`/`linspace` arithmetic and rounded **once** to float32: 0 of 20,560
  values differ. librosa's own default path, float32 weights later multiplied by the float64 `enorm`, misses 142
  values by 1 ulp; doing everything in float32 misses 182. The exporter therefore built the bank in float64.
- `audio.manifest` equals 3246 and 1, followed by the eleven config values F-003 lists (`text` … `hop`). F-003 left
  the first two values open. 3246 is `0xCAE`, a magic. The engine (`needle` 3.2.0) was run on a copy of
  `whistle.cact` with one manifest value patched:

  | patched value                   | engine result                                         |
  | ------------------------------- | ----------------------------------------------------- |
  | `[0]` = 3247 or 0, `[9]` = 8000 | "no speech model loaded" (the file is not recognised) |
  | `[1]` = 2 or 0                  | "failed to load .cact speech model"                   |
  | `[2]` = 8000, `[3]` = 7         | "failed to load .cact speech model"                   |
  | unpatched                       | the normal transcript                                 |

  So the engine finds the manifest by content, as the FP32 `(13,)` record with the magic and a 16 kHz rate. `[1]` is
  a format version it accepts only as 1, and the geometry must agree with the tensors. A static look at
  `libneedle3.dylib`'s `needle_load` with `objdump` had shown the same comparisons (3246.0f, 1.0f, 16000.0f) and led
  to the experiment; no engine code was copied.

- `tokenizer` parses with `internal/tokenizer` (8199 pieces = header `vocab` = `text` + 7, pad/eos/bos 0/1/2 = the
  config). Its sha256 equals the `blob_sha256` of F-005's sentencepiece fixture.

**Thresholds.** They are derived per record, not tuned:

- **FP16** (524 records, 458,057 values). Every value must be the round-to-nearest-even FP16 of its F32 source, and
  all 458,057 are. The cosine must also meet the rounding bound: with |x̂ᵢ − xᵢ| ≤ hᵢ, half an FP16 ulp at xᵢ (2⁻²⁵ in
  the subnormal range), sin θ ≤ ‖h‖/‖x‖, so cos ≥ √(1 − ‖h‖²/‖x‖²). The bounds range
  from 0.99999991 to 0.99999996. The 25 scalars (attention gates, cross gates, `pe_gate`) rest on the exact-value check, since their
  cosine is ±1.
- **CQ** (145 CQ2, 7 CQ4). A Go port of `_cq_pack` (rotate by H/√128 via a float64 FWHT, split off the norm, nearest
  Lloyd-Max level, FP16 norm) requantises each source. The record must equal that blob. The FP16 norms may differ by
  1 ulp, which happens for 11 norms in 10 records. A packed index may differ only by one level, and only where the exact
  unit coordinate lies within 260·2⁻²⁴ ≈ 1.5·10⁻⁵ of the decision boundary between the two levels. That tolerance is
  the float32 error of upstream's rotation: H's rounded entries, γ₁₂₈ for the 128-term sum, and as much again for the
  norm. 16 indices in 14 records differ, all within 2·10⁻¹⁰ to 4·10⁻⁸ of the boundary; every other index of the
  54.7 M CQ weights is identical. The cosine threshold is then cos(acos(cos(Q(src), src)) + asin(2⁻¹⁰)): the quantiser's
  own error on these weights, plus the angle that 1-ulp norm changes (a group scaled by at most 2⁻¹⁰) can add.
  A near-tie flip barely moves the cosine, because u lies about equally far from both levels.
- **Derived FP32 and RAW**: exact equality, as listed above.

**Rejected thresholds.**

- A Gaussian Lloyd-Max floor, √(1 − D) = 0.93941 for CQ2 (D = 0.1175) and 0.99524 for CQ4 (D = 0.009497), would fail
  143 of 145 CQ2 and 5 of 7 CQ4 records. QAT did **not** pull these weights closer to the grid; the rotated groups
  quantise slightly worse than i.i.d. Gaussians.
- PLAN.md's "≥ 0.95 for CQ2" is wrong for the same reason. The measured CQ2 range is 0.93185–0.94008 (median 0.93465),
  and the CQ4 range is 0.99431–0.99537.

**Measured** (`TestWhistleNameMap`): all 681 records pass. The minimum cosine per type is pinned to 1e-9:

| type | records | min cosine     | at                 |
| ---- | ------- | -------------- | ------------------ |
| FP16 | 524     | 0.999999963923 | `enc00.q_norm`     |
| CQ2  | 145     | 0.931853685259 | `cross07.k_proj`   |
| CQ4  | 7       | 0.994312193718 | `enc.mhc_phi_post` |

Every CQ record's width also matches the run's `weight_bits` plan (`embedding=4,stack/mhc=4,encoder/mhc=4,default=2`).
The test runs in about 1.5 s; `RequantCQ` uses an FWHT and is pinned to the matrix reference `portPack`.

**Consequence.** The `.cact` file holds no information beyond the safetensors checkpoint, its config and two constants
(magic and version). Every CQ record is a deterministic function of its source, up to float32 rounding ties. A
safetensors-only load can therefore rebuild the deployment numerics exactly; the derived records need only the
permutation port and the mel recomputation, both in `internal/cact`'s tests.

## F-009 — Needle 3 block fixtures from the Flax modules (2026-10-10)

**Question.** Can upstream's Flax modules (`architecture.py` @ `af2654dffa`) run as-is with Whistle's safetensors
parameters, so that their outputs become the Go blocks' reference? How do the Flax paths map onto the module
parameters, and what does the checkpoint lack? And is PLAN.md's Phase 2 tolerance (1e-5 max abs) achievable?

**Binding.** `scripts/needle_fixtures.py` builds each module's parameter tree with `jax.eval_shape(module.init, …)`.
It fills each leaf at Flax path `p` from safetensors `<prefix>/<p>`, sliced at the layer for the scanned `[8, …]`
tensors, and asserts that every shape matches. The prefixes:

| module                               | safetensors prefix                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| `Block`, layer L                     | `stack/layers/block/…[L]`, 27 leaves (exactly `.cact` `layerLL.*`)                   |
| `MultiHeadAttention` (decoder)       | `stack/layers/block/self_attn/…[L]`                                                  |
| `MultiHeadAttention` (encoder-style) | `encoder/layers/block/self_attn/…[L]` with `qkv_conv_taps=0`                         |
| `HadamardMLP`                        | `stack/layers/block/hadamard_mlp/…[L]`                                               |
| `_ScanBody` (mHC)                    | `stack/layers/block/…[L]`; the `hc` dict from `stack/mhc_*[L]` plus the lane offsets |
| `Engram`, site s                     | `engrams_s/…` (no layer axis)                                                        |
| final `ZCRMSNorm`                    | `stack/final_norm/scale`                                                             |

Every parameter the modules declare exists with the exact shape, so nothing is missing. The rest of what they need is
computed: the RoPE tables (θ = `rope`), `_hada_perms(512)`, the mHC `pre_off`/`post_off` from `lane = eye(4)[L % 4]`,
the engram hash constants, and the masks. Two config facts are implicit:

- `num_heads` is not in the config. It is the `q_proj` width over `qk_dim`, 384 / 48 = 8.
- `hashes` (2) is `engram_heads`. It equals `engram_geometry`'s default, `512 // (2 · 128)`.

The only unread checkpoint tensors in the decoder are the cross-attention tensors (`cross_attn/*`, `cross_norm`,
`post_cross_norm`, `cross_gate`), which upstream's `Block` does not contain. The encoder's attention fits
`MultiHeadAttention` exactly (q/k norm, gate, GQA 8:2, no taps); its other parts (conv module, macaron FFN) have no
upstream module.

**Intermediates.** Flax's `capture_intermediates` gives every submodule's output: the projections, `q_norm`/`k_norm`,
`gate_proj`, the Block's norms, `self_attn` and `hadamard_mlp`, and `value_proj`. Everything in between is recomputed
in the script from upstream's helpers (`_shift_right`, `_mask_diag`, `apply_rope`, `_kron_apply`, `_hada_perms`,
`_rms_unit`, `_sinkhorn`, `engram_indices`) in upstream's expression order. Before a file is written, each
recomputation must reproduce the module's own output, and all do with **0** difference. For the engram gate, which
lives inside `Block`, the script checks `Block_3(x, engram_kv)` = `Block_3(x + α·v)`.

**Choices.**

- `flash=False` (explicit softmax) is the reference path. `flash=True` uses `jax.nn.dot_product_attention`, which pads
  qk 48 to 64 and rescales q by √(64/48). It differs by 1.3e-4 at |out| ≤ 329 (4e-7 relative), and it keeps its
  softmax in float32 even in an x64 run (2.2e-5 from flash=False there). It is mathematically the same attention.
- `quant=False` throughout. `_aq`, `maybe_quant_query` and `maybe_quant_kv` are then identities.
- The encoder-style fixture stores the output both without and with RoPE, because U3 is open.
- The mHC fixture runs layer 0 on the `Stack` lane expand, where every doubly stochastic `h_res` gives the same
  mix. It also runs layer 1 on four distinct lanes, where the Sinkhorn mix and lane 1's offsets matter. Layer 0's
  output is then collapsed (lane mean, `final_norm`) to pin `Stack`'s ending. Engram is off, since the site flags at
  layers 0 and 1 are 0 and `x + 0·…` is exact.
- Engram ids include a repeated trigram, so the indices repeat: order 2 at t = 7, 8 equals t = 2, 3, and order 3 at
  t = 8 equals t = 3. They also include the pad id 0. Positions t < order − 1 get a hash over zero padding;
  `ngram_ok` zeroes their table rows.

**Determinism.** safetensors 0.8.0 serialises `__metadata__` from a Rust `HashMap`. Its key order, and so the file
bytes, changes from run to run: three runs of the same `save_file` gave three digests. The script rewrites the header
with the metadata keys sorted. The JSON keeps its length, so the data offsets stay valid. It then reloads the file and
compares every tensor and the metadata. Two `just needle-fixtures` runs are byte-identical. The seven files total
1,972,688 bytes and load with `internal/safetensors`: all 89 tensors have the shapes in their `shapes` metadata, and
the per-tensor sums equal numpy's.

**Tolerance.** PLAN.md's "1e-5 max abs" cannot be met as an absolute bound. With the real weights, the decoder's
activations are large: `hadamard_mlp` and `block` outputs reach 1.5e3, attention outputs 330 (decoder) and 1.2e3
(encoder weights), and the mHC stream 1.2e3. `--f64-check` reruns every block in float64. For that rerun,
`jnp.float32` is mapped to `float64` inside `architecture.py`, because upstream casts to float32 explicitly. The
float32 fixtures lie up to 1.3e-3 max abs from the float64 values (`block.mlp`). Relative to each tensor's largest
magnitude, the gap is at most 3.9e-6 (`attention_encoder.gate`, whose max is 1), typically 1e-6. Every cosine is
≥ 0.999999999999. A float32 Go port has noise of the same order, so the contract in `testdata/needle/README.md` is:

- max |go − fixture| ≤ 1e-5 · max(1, max |fixture|) per tensor;
- cosine ≥ 0.99999;
- I64 exactly.

Alternatives considered: a flat 1e-5 absolute bound fails for any float32 implementation, the fixtures included. A
per-element relative bound breaks down near zero (`probs`, `y − u` cancellations). Cosine alone would miss a
uniformly wrong scale.

**Not covered.** The Go blocks do not exist yet; the fixtures are the contract for the next Phase 2 branches.
