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

**Correction (F-012).** The frame rule is `ceil(floor(samples / 160) / 8)`, not `(samples + 640) / 1280`. Both fit the
seven lengths above, but 7 of the 32 corpus clips follow only the former (e.g. 53040 samples give 42 frames).

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

## F-007 — One lookup over both formats: interface, derived records, cache and mmap (2026-10-09)

**Question.** Model code must read Whistle's tensors the same way from the deployed `.cact` (CQ2/CQ4, FP16) and from
the float32 safetensors checkpoint, which Phase 2 uses as its reference. What does the safetensors side serve for the
five records that have no source (F-006)? What does the cache keep, and for how long? How is the file held, and what
may a caller keep after `Close`?

**Interface.** `internal/weights` `Source`: `Names`, `Has`, `Shape`, `Tensor(name) ([]float32, []int, error)`, `Raw`
(byte blobs), `Config`, `Release` and `Close`. `Cact` and `Safetensors` implement it, and `Open` picks one by the `.cact`
tag. Names are F-006's (`cact.WhistleNames`, or any map passed in `Options.Names`), so model code never sees record
indices or Flax paths. Both backends return the `.cact` layout: `(out, in)` kernels, a layer slice per block, and
lane-major mHC φ. The safetensors side gets there through `Entry.Apply`. At open time, each backend checks every
entry's shape: against its record (`.cact`) or against its source's header shape (safetensors). A wrong map therefore
fails at once, not on first use, and the check reads no tensor data. `safetensors.Store` gained `Shape(name)` for this.

**Config.** `weights.Config` holds the inference geometry that both formats state, mapped as in F-003. For `.cact` it
comes from the header plus the audio manifest; the manifest needs the magic and version 1, as the engine does. For
safetensors it comes from the config, where `Vocab` = `text` + 7, `HadaN` = the next power of two of `dim`, and `Tables`
= `len(orders)` × `hashes`. The checkpoint config has no head count. `Heads` is the row count of `layer00.q_proj` over
`qk_dim` (384 / 48 = 8), checked against the source shape at open time. It equals the header's `num_heads`. Left out:
the token ids (the tokenizer holds them), the KV width (run metadata) and the training keys. Both backends return
identical `Config` values for Whistle.

**Derived records.** The candidates:

| option                                       | `hada_p*`, mel bank, manifest                         | tokenizer                   |
| -------------------------------------------- | ----------------------------------------------------- | --------------------------- |
| recompute in Go                              | exact: F-006 showed each is a function of the config  | impossible: data, no config |
| typed "not available from this format" error | Phase 2 and 3 would need a `.cact` for a float32 load | honest                      |
| read the records from a `.cact` beside it    | a hidden second-file dependency                       | same                        |

Winner: the safetensors backend recomputes the permutations (numpy MT19937), the Slaney mel bank (float64, rounded
once) and the manifest. For the tokenizer, `Raw` returns `ErrUnavailable`, and so does `Tensor` for any derived name it
cannot recompute. The recomputations moved from `internal/cact`'s tests to `internal/cact/derived.go`. They sit beside
`WhistleNames`, which defines those records, and stay in a leaf package that the later `needle` and `frontend` packages
can import without a cycle. On Whistle, all four recomputed records equal the `.cact` ones bit for bit.

**Cache.** The candidates:

- No cache. Every call decodes again; the CQ4 embedding alone takes 15–18 ms to dequantise.
- A bounded LRU. A model keeps every tensor it loads, so evicting one frees nothing while the model holds it.
- Unbounded until `Release(name)` or `Close`. **Winner.**

The cached slice is the very slice the model keeps, so while the model lives the cache costs no extra memory. `Release`
lets code that repacks a tensor (the Phase 9 LUT GEMV) drop the float32 copy. Callers therefore share one slice, which
they must not modify. A per-name `sync.Once` makes concurrent first calls decode once. An `RWMutex` lets `Close` wait for
decodes in flight. A decode error stays cached until `Release`. Measured on an Apple M5 Pro: a cached hit takes under
1 µs. The whole model decodes to 221 MB of float32 in either format, in 0.12–0.14 s from `.cact` and 0.21 s from
safetensors.

**mmap.** Unix (`//go:build unix`) maps the file read-only and shared with `syscall.Mmap`. Cross-vetting showed
`syscall.Mmap` builds for linux, darwin, the BSDs, solaris/illumos, aix and android. Elsewhere (windows, js/wasm,
wasip1, plan9), `mmap_nommap.go` reads the file with `os.ReadFile`. Windows mapping through `CreateFileMapping` is left
for when it is needed. Empty files, and files the system refuses to map, are read too; `Options.NoMmap` forces the read.

| file (M5 Pro)         | open, mapped | heap after open | open, read | heap after open |
| --------------------- | ------------ | --------------- | ---------- | --------------- |
| `whistle.cact`        | 0.4 ms       | +0.1 MB         | 1.6 ms     | +17 MB          |
| `whistle.safetensors` | 0.6 ms       | +0.2 MB         | 12–23 ms   | +221 MB         |

Decoding everything from the read safetensors file holds 442 MB on the heap, against 221 MB when mapped.

**Aliasing rule.** Nothing handed out aliases the file data. The `.cact` readers allocate (FP16/FP32 widening, CQ
dequantisation, and `Raw`, which clones). `Store.Tensor` decodes into a new slice and `Apply` copies. Names, metadata
and the config are copied when the JSON header is decoded. `Close` drops the cache and unmaps; afterwards `Tensor` and
`Raw` return `ErrClosed`, while every slice handed out stays valid. A test checks that no returned slice lies inside the
mapping's address range, and reads every slice again after the unmap. Caveat, as for any mmap reader: truncating the
file under an open source faults the process.

**Cross-format check.** `TestWhistleCrossFormat` covers all 681 names. Shapes are equal for every name and both
`Config` values are equal. The 4 recomputed records are bit-equal. All 524 FP16 records lie within half an FP16 ulp of
their source and meet F-006's rounding bound. For CQ, the minimum cosine per type equals F-006's to 1e-9: FP16
0.999999963923, CQ2 0.931853685259 and CQ4 0.994312193718. Each backend also equals its format's plain reader bit for
bit (`cact.File`; `Store.Tensor` + `Entry.Apply`), so `TestWhistleNameMap`'s per-record CQ thresholds carry over. The
test runs in 0.8 s. Gap: this package checks CQ records against F-006's per-width floor, not against a per-record
requantisation, because the `_cq_pack` port lives in `internal/cact`'s test files.

## F-008 — `internal/tensor` and `internal/ops` copied from go-pocket-tts (2026-10-10)

Nothing here is reverse-engineered. This entry records the copy source, what was kept and dropped, the fallback build
tag and the tolerances, so later kernel work can tell copied behaviour from go-whistle decisions.

Since 2026-10-11 both packages live in the module `github.com/cwbudde/algo-tensor` (v0.1.0, the union of
go-whistle's and go-pocket-tts' copies; its comments keep these F-numbers). Since 2026-10-12 go-whistle takes
v0.3.1, whose `Conv2D` runs groups = 1 through im2col on arm64 and js/wasm (F-038, "The stem's conv through
algo-tensor v0.3.1"). Paths such as `internal/tensor/asm_test.go` below name the files as they were here; the module
has them under `tensor/` and `ops/`.

**Source.** `github.com/cwbudde/go-pocket-tts` `internal/runtime/tensor` and `internal/runtime/ops` at `9faba38`.
Every copied file starts with a comment that names its source file and commit, plus a `Change:` line when the file
differs. Only the import paths changed in the other files. Fixes can therefore be ported in either direction.

**Kept and dropped.**

| what                                                         | verdict | reason                                                                                                                                       |
| ------------------------------------------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tensor`, broadcast, `Narrow`/`Gather`/`Transpose`/`Concat`  | kept    | the generic base; `Gather` is the embedding lookup                                                                                           |
| `MatMulTransB` (NEON), `DotProduct`, `Axpy` (NEON, AVX2+FMA) | kept    | every linear, attention and conv in Phases 2–5 rests on them                                                                                 |
| `Linear`, `MatMul`, `Softmax`                                | kept    | projections, scores, the language softmax                                                                                                    |
| `Attention`, `CausalMask`                                    | kept    | causal with an offset (decoder plus KV cache) and none (encoder, cross-attention)                                                            |
| `RoPE`                                                       | kept    | but see the RoPE gap below                                                                                                                   |
| `Conv1D`, `Conv1DLeftPad`, conv worker pool, scratch pool    | kept    | depthwise kernel 9 in the encoder; left padding is the causal form of the Engram taps                                                        |
| `LayerNorm`                                                  | dropped | every Needle 3 norm is a ZCRMSNorm (scale only, no mean, no bias); the next item builds it                                                   |
| `MLP` (linear → SiLU → linear)                               | dropped | FlowLM's FFN; Whistle's FFNs are `HadamardMLP`, which brings its own SiLU                                                                    |
| `AttentionWithPositions`, `AttentionNoContext`               | dropped | Pocket-TTS streaming mask (absolute positions, ring-buffer keys, context window); Whistle re-runs ~1 s chunks and needs only causal + offset |
| `ConvTranspose1D` and its prepacked variants                 | dropped | Mimi decoder upsampling; Whistle has no transposed conv                                                                                      |
| `PERFORMANCE.md`, FlowLM/Mimi benchmarks                     | dropped | the benchmarks now use Whistle shapes (375 frames, 8 heads of 48/64, depthwise k 9)                                                          |

**Fallback build tag.** In the source, the fallbacks were selected by architecture alone, so on arm64 and amd64 no
build could run them. Alternatives: keep that and test only through `*Generic` calls, or add a tag. A custom `noasm`
tag was rejected in favour of `purego`, the tag the standard library's `crypto` packages and `x/crypto` use for
this. The asm files now require `<arch> && !purego`, and `dot_noasm.go`, `axpy_noasm.go` (was `axpy_noamd64.go`) and
`gemm_noasm.go` (was `gemm_noarm64.go`) require `(!amd64 && !arm64) || purego` (gemm: `!arm64 || purego`).
`go test -tags purego` runs the pure-Go code anywhere.

**Assembly vs fallback.** The six routines are `dotF32NEON`, `axpyF32NEON`, `gemm4x4NEON`, `gemm1x4NEON` (arm64), plus
`dotF32AVX2` and `axpyF32AVX2` (amd64, only with AVX2 and FMA, detected via `golang.org/x/sys/cpu`). amd64 has no GEMM
kernel: `MatMulTransB` is the dot loop there. The copied tests check the public functions against float64 references.
None compared a kernel with its fallback, so `internal/tensor/asm_test.go` does that, for lengths 0–129, 255–257,
1023/1024 and 4099 at offsets 0–3:

- `dotF32` vs `dotF32Generic`: |Δ| ≤ 2(n+1)·2⁻²⁴·Σ|aᵢbᵢ|, two float32 summation bounds, since the orders differ.
- `axpyF32` vs `axpyF32Generic`: |Δ| ≤ 2·2⁻²⁴·(|αxᵢ| + |yᵢ| + |result|), one fused vs separate multiply-add. The
  slack after n stays untouched.
- `matMulTransB` vs `matMulTransBGeneric`: the dot bound per element over 13 k values × 8 (m, n) shapes. The row gaps of
  a wider `ldc` stay untouched.

Run on arm64 (Apple M5 Pro, native) and on linux/amd64 under Docker Desktop's emulation, where `useAVX2FMA` was
`true`. That host has no Rosetta, so `GOARCH=amd64 go test` cannot run natively. All pass in both tag variants. On an
amd64 CPU without AVX2, or with `purego`, both sides are the fallback and the tests pass trivially.

**Tolerances.** `ops.KernelTolerances` keeps go-pocket-tts' values: 1e-4 abs/rel for matmul, linear and softmax, 0 for
causal_mask, and 2e-4 for rope, attention and conv1d. The entries for the dropped kernels (`layer_norm`, `mlp`,
`convtranspose1d`) are gone. `TestKernelTolerancesPinned` pins the map, so any change must come back here. These were
ONNX parity targets in go-pocket-tts. Nothing in go-whistle uses them yet. They are kernel-level and looser than the
Phase 2 block acceptance (1e-5 max abs, cosine ≥ 0.99999), which the `needle_fixtures` item enforces on its own.

**Gaps for later items.**

- **RoPE.** `ops.RoPE` rotates _interleaved_ pairs (x₂ⱼ, x₂ⱼ₊₁) with caller-supplied cos/sin tables `[max_seq, dim/2]`.
  θ is therefore whatever the caller's tables use; there is no θ parameter and no table builder. Needle 3 uses the
  _half-split_ layout (xⱼ, xⱼ₊₄₈/₂): x1·cos − x2·sin, x2·cos + x1·sin with freqs = 1/θ^(arange(0, hd, 2)/hd), θ = 1e5,
  hd = 48. Follow-up for the `Attention` item: a half-split variant and a table builder for θ.
- **GQA.** `Attention` needs q, k and v with equal head counts. Whistle's 8:2 self-attention needs either repeated k/v
  heads or a head map (kv head = h / 4) in `attention4D`. The scale is 1/√(q depth) = 1/√48, as Needle's is, and v may
  be wider than q (64 vs 48).
- **Depthwise convs.** The encoder's kernel-9 depthwise conv is `Conv1D` with groups = channels. That takes the scalar
  `conv1DGrouped` loop (1.08 ms for 512 × 375 on the M5 Pro), which is correct but not vectorised. The 2-D stem will
  need a `Conv2D` over (time, mel) with per-axis stride (2, 2), SAME or VALID padding (the Phase 4 hypotheses), groups 1
  for `stem/w` (1 → 128) and groups 128 for `stem/dw_{1,2}`, with Flax HWIO kernels `[3, 3, 1, 128]`. The pointwise
  stages are `Linear`. Not built here.
- **Linear.** `tensor.Linear` runs one `dotF32` per output and does not use `MatMulTransB`'s 4×4 blocks (4.8 ms for the
  375 × 512 → 384 `q_proj`). Routing it through the GEMM is Phase 9 work.

**Other changes.** `TestSetConvWorkersClamp` passed `maxInt32 + 123`, which overflows a 32-bit `int`, so the package
did not build on 386 or arm. It now passes the largest `int`. Lint: the copied `//nolint` comments did not cover
every finding, because go-pocket-tts excluded `cyclop`/`gocognit` by path (`internal/runtime/`). Five functions (three
kernels, `MatMul` and `prepareConv1D`) gained narrowly scoped `cyclop` or `gocognit` nolints with reasons;
`.golangci.yml` is unchanged.

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

## F-010 — WAV decode and resampling to 16 kHz vs `whistle.py` and soxr HQ (2026-10-10)

**Question.** `whistle.py` `_read_wav` turns a WAV file into 16 kHz mono float32 with Python's `wave` module, its own
sample conversion and `soxr.resample(mono, rate, 16000, quality="HQ")`. How exactly can the port follow it in pure Go,
which files does it have to reject, and how far is a resampler built on `algo-dsp/dsp/resample` (v0.12.4) from soxr?

**Decode (`internal/frontend/wav.go`).** `ParseWAV` follows `wave.py` (CPython 3.12) rule by rule:

- Every chunk is read through the RIFF chunk, so nothing past the declared RIFF size is seen. Chunks are word-aligned;
  a chunk whose skip would end past the RIFF chunk is an error (`wave.py` raises from its seek).
- The format tag must be PCM (1) or EXTENSIBLE (0xFFFE) with the PCM sub-format GUID. Anything else, IEEE float
  included, is a `*FormatError`, as `wave.py`'s "unknown format".
- The sample width is `(bits + 7) // 8` from `wBitsPerSample`; block align and byte rate are ignored. Width 0 and no
  channels are errors; so is a width above 4, which `_read_wav` has no decoder for (a `KeyError` upstream).
- The last fmt chunk before the first data chunk counts; a data chunk before any fmt chunk is an error; anything after
  the data chunk is ignored. The frame count is `data size // frame size` (a trailing partial frame is dropped), and
  the data may be shorter than declared (streamed files with size 0xFFFFFFFF work).

`Mono` is `_read_wav`'s expression, `float32((sum(frame) - offset·channels) / (2^(8w−1)·channels))` in float64 with
offset 128 for unsigned 8-bit samples, bit for bit (`TestMonoFormula` checks all widths and 1–3 channels). Only two
malformed-input cases differ: a data chunk cut inside a sample is `ErrTruncated` for every width above 8 bits, where
upstream raises for 16 and 32 bits but decodes a cut 24-bit sample from the bytes it has; and rates are checked (below).

**Accept matrix.** `TestUpstreamWAVMatrix` writes the six WAVs as `test_whistle.py` `_write_wav` does
(`TestUpstreamWAVBytes` pins the bytes to Python's writer) and runs `DecodeWAV`. The same files through upstream
`_read_wav` with soxr 1.1.0:

| rate/channels/width | samples (Go / upstream) | max, min (Go)    | max, min (upstream) | crossings (Go / upstream) |
| ------------------- | ----------------------- | ---------------- | ------------------- | ------------------------- |
| 16000/1/2           | 8000 / 8000             | 0.29999 −0.29999 | 0.29999 −0.29999    | 109 / 109                 |
| 8000/1/2            | 8000 / 8000             | 0.29998 −0.29998 | 0.29999 −0.29999    | 110 / 109                 |
| 44100/2/2           | 8000 / 8000             | 0.29997 −0.29997 | 0.29997 −0.29997    | 109 / 109                 |
| 16000/1/1           | 8000 / 8000             | 0.29688 −0.29688 | 0.29688 −0.29688    | 109 / 109                 |
| 22050/2/3           | 8000 / 8000             | 0.30000 −0.30000 | 0.30000 −0.30000    | 109 / 109                 |
| 48000/1/4           | 8000 / 8000             | 0.30000 −0.30000 | 0.30000 −0.30000    | 109 / 109                 |

All six meet the criterion (8000±1 samples, peaks inside 0.28–0.31, 108–111 crossings). The 8 kHz crossing count
differs by one because of the half-sample residual below.

**What soxr does.** The one-shot `soxr.resample` is linear phase with the delay removed (input time t maps to output
time t) and returns `round(n·16000/rate)` samples (22050 Hz: 100 and 101 input samples both give 73). Measured from
its impulse response, HQ is flat to 7.35–7.39 kHz (−0.01 dB), −3 dB at 7.60 kHz, −6 dB at 7.66 kHz and −100 dB at
7.98 kHz when downsampling to 16 kHz (3.70 / 3.80 / 3.83 / 3.99 kHz from 8 kHz). Tones just above 8 kHz come out at
−130 to −144 dB.

**Fixture.** `scripts/resample_fixture.py` (`just resample-fixture`, deterministic) stores soxr HQ's output for
220 Hz, 1 kHz, 3.5 kHz, 7 kHz and 9.5 kHz tones, a chirp to 0.98 of the lower Nyquist and seeded white noise, 0.4 s
each at amplitude 0.5, from 8000, 22050, 44100 and 48000 Hz (26 cases, 16-bit inputs regenerated in Go and checked by
SHA-256, 675 KB). Because 16-bit quantisation noise floors anything below about −95 dB, it also stores soxr's output
level for float32 tones at 8.1, 8.2, 8.4 and 9.5 kHz. "Near-Nyquist" means the top quarter of the lower Nyquist:
6–8 kHz, or 3–4 kHz from 8 kHz. The Go side drops 400 output samples at each end (different filter lengths give
different start and end transients).

**Alignment.** algo-dsp's polyphase filter is causal, and its prototype has `taps per phase · up` taps. Two built-in
ways of removing the delay were rejected: `ProcessAligned` interpolates the fractional part linearly, which took a
further 14 dB off a 7 kHz tone (−15.9 dB total with `QualityBest`), and `Stream` rounds the delay up, leaving up to one
output sample of offset (not measured). `Resample` instead prepends `lead` zeros so that the integer part of the delay,
`floor((n−1)/2) + lead·up`, is a multiple of `down`, and takes the raw outputs from there on: no interpolation. That is
exact for an odd prototype (forced when `up` is odd, e.g. 48 kHz). An even prototype leaves half an upsampled step,
`0.5/down` output samples: 0.0011 from 22.05 and 44.1 kHz (up 320 or 160), but half a sample (31 µs) from 8 kHz,
because `up` = 2 makes every prototype even. algo-dsp has no hook for a fractional start phase or a custom
prototype, so the residual stays. The measured lag equals it to within 0.0011 samples on every case but the
stopband tones.

**Filter.** algo-dsp's presets fix taps per phase, so their transition band widens by `down/up` when downsampling.
`filterSpec.length` scales taps per phase to `ceil(length · max(up, down) / up)` instead, so the prototype always spans
`length` samples of the lower rate. Alternatives, worst case over 22.05/44.1/48 kHz on the fixture, alias levels for
float input at 48 kHz, time for 30 s of 44.1 kHz (Apple M5 Pro, one core):

| setting                           | 7 kHz gain vs soxr | 9.5 kHz (16-bit) | 8.1 / 8.2 kHz alias | 30 s @ 44.1 kHz |
| --------------------------------- | ------------------ | ---------------- | ------------------- | --------------- |
| `QualityFast` (16 taps/phase)     | −5.92 dB           | −17.1 dB         |                     | 4 ms            |
| `QualityBalanced` (32)            | −4.39 dB           | −25.8 dB         |                     | 6 ms            |
| `QualityBest` (64)                | −1.68 dB           | −51.7 dB         |                     | 14 ms           |
| `ProcessAligned`, `QualityBest`   | −15.87 dB          | −62.4 dB         |                     | 14 ms           |
| length 64, cutoff 0.96, β 9       | ≤ 0.005 dB         | −92.9 dB         | −29.3 / −40.2 dB    | 46 ms           |
| **length 128, cutoff 0.96, β 11** | ≤ 0.007 dB         | −93.6 dB         | −86.9 / −112.1 dB   | 111 ms          |
| length 160, cutoff 0.96, β 12     | ≤ 0.007 dB         | −93.6 dB         | −121.8 / −139.8 dB  | 144 ms          |
| soxr HQ (python-soxr 1.1.0)       | 0                  | −93.6 dB         | −139.4 / −130.6 dB  | 2 ms            |

**Winner: length 128, cutoff 0.96, Kaiser β 11** (`hqSpec`). It is the shortest setting that reproduces soxr's band
edge (−0.11 dB at 7.4 kHz and −2.78 dB at 7.6 kHz; soxr −0.02 and −2.89 dB) and keeps aliases at −87 dB from 8.1 kHz
and below −110 dB from 8.2 kHz. Length 64 takes less than half the time but lets 8.1 kHz alias at −29 dB into the top mel bands;
length 160 gains about 2 dB on noise for 30 % more time. Cutoffs 0.95 or β 9 at length 128 fail the pinned bounds.
Time per 30 s: 29 ms from 8 kHz, 46 ms from 22.05 kHz, 111 ms from 44.1 kHz and 121 ms from 48 kHz (270–1000× real
time, about 30–60× slower than soxr's SIMD multi-stage code), dominated by `taps per phase` = 128 to 385
multiply-adds per output sample.

**Measured** (`TestResampleMatchesSoxr`, interior, `hqSpec`; "aligned" removes the expected residual first, so it
isolates the filter shape; STFT is the mean |Δ| of log power over 400-sample Hann frames, hop 160, in bins within
60 dB of the frame peak):

| input                     | SNR dB      | max abs       | lag (residual)       | aligned SNR dB   | STFT dB            |
| ------------------------- | ----------- | ------------- | -------------------- | ---------------- | ------------------ |
| 8k: 220 Hz / 1 kHz        | 27.3 / 14.2 | 0.022 / 0.098 | 0.501 (0.5)          | 67.1 / 67.1      | ≤ 0.063            |
| 8k: 3.5 kHz (near)        | 3.4         | 0.34          | 0.500 (0.5)          | 67.1             | 0.0000             |
| 8k: chirp                 | 7.4         | 0.35          | 0.500 (0.5)          | 67.2 / 49.3      | 0.052 / 0.052      |
| 8k: noise                 | 7.4         | 0.40          | 0.500 (0.5)          | 66.9 / 27.4      | 0.039 / 1.49       |
| 22.05k, 44.1k: tones < 6k | 56.1–80.2   | ≤ 7.8·10⁻⁴    | 0.001–0.002 (0.0011) | ≥ 116.6          | ≤ 0.0001           |
| 22.05k, 44.1k: 7 kHz      | 49.9        | 1.6·10⁻³      | 0.0003 (0.0011)      | 62.5             | 0.0065             |
| 22.05k, 44.1k: chirp      | 54.2        | 1.7·10⁻³      | 0.0005 (0.0011)      | 92.0 / 72.4      | 0.0003 / 0.0042    |
| 22.05k, 44.1k: noise      | 32.9–33.6   | ≤ 0.016       | 0.0006 (0.0011)      | 85.7 / 25.7–29.1 | 0.0003 / 1.24–1.27 |
| 48k: tones < 6k           | 129.1–136.0 | ≤ 4.5·10⁻⁷    | ≤ 0.001 (0)          | ≥ 131.1          | 0.0000             |
| 48k: 7 kHz                | 112.0       | 1.3·10⁻⁶      | 0.0000 (0)           | 112.0            | 0.0000             |
| 48k: chirp                | 70.7        | 2.2·10⁻³      | 0.0000 (0)           | 131.3 / 93.8     | 0.0000 / 0.0002    |
| 48k: noise                | 32.9        | 0.012         | 0.0000 (0)           | 127.1 / 25.4     | 0.0000 / 1.21      |

Pairs are passband / near-Nyquist. Gain differs by at most 0.0065 dB on every non-stopband case, and every
non-stopband cosine is ≥ 0.99974 except at 8 kHz (0.77–0.999, the half-sample residual). Two differences remain.
First, the residual: from 8 kHz the output is 31 µs late, which costs nothing in magnitude (STFT ≤ 0.063 dB, aligned
SNR ≥ 66.9 dB) but dominates any sample-wise comparison. Second, the 7.4–8 kHz transition band, where noise differs by
1.2–1.5 dB per STFT bin on average and noise's near-Nyquist band sits at 25–29 dB SNR; the aligned passband is
≥ 66.9 dB everywhere. A 16 kHz WAV is not resampled, exactly as upstream.

**Rates.** The ratio `16000/rate` is reduced exactly when `down` ≤ 16384, which covers every common rate (11025:
640/441, 44056: 2000/5507, 192000: 1/12) and bounds the prototype at about two million taps (35 MB, 70 ms to
design). Rates that need a larger `down` (47999 Hz: 16000/47999) get algo-dsp's continued-fraction approximation with
denominator ≤ 16384, here 1/3: a 2·10⁻⁵ rate error, 10 samples of drift over 30 s. Two guards reject rates soxr
takes, both as `ErrSampleRate`: rates below 1 kHz (soxr only refuses 0), so that a small file with a near-zero rate
field cannot expand more than 16-fold into gigabytes of output, and rates from about 262 MHz up, which have no
approximation. Pinned in `TestResamplerRatio`, `TestResampleTone`, `TestResampleLength` and
`TestResampleImpulseAlignment`.

Regenerate the fixture with `just resample-fixture`; `internal/frontend` reads `testdata/frontend/soxr_hq.safetensors`.

## F-011 — Needle 3 core blocks in Go: ZCRMSNorm, Attention, HadamardMLP (2026-10-10)

**Question.** Can float32 Go ports of `ZCRMSNorm`/`_rms_unit`, `MultiHeadAttention` and `HadamardMLP` meet the F-009
fixture contract (max |Δ| ≤ 1e-5 · max(1, max |ref|) per tensor and cosine ≥ 0.99999) on every intermediate, and which
upstream details does a port have to get right to do so?

**API (`internal/needle`).** `ZCRMSNorm` (`NewZCRMSNorm`, `LoadZCRMSNorm(src, name)`, `Apply`), `RMSUnit`, `Rope`
(`NewRope(headDim, seqLen, θ)`), `Attention` (`LoadAttention(src, prefix)`, `Forward(x, T, mask, rope)`, masks
`MaskNone` and `MaskCausal`) and `HadamardMLP` (`LoadHadamardMLP(src, prefix)`, `Forward(x, T)`). Activations are one
sequence, row-major (T, width); a per-head tensor keeps head h of row t at columns h·d … (h+1)·d. Blocks load by name
prefix from a `weights.Source` (`layer00.`, `enc00.`, `enc00.hada_0.`, `enc00.hada.`), so encoder and decoder share
the code. Dimensions come from the shapes, not from the config: qk from `q_norm`, the query heads from `q_proj`, the
KV heads from `k_proj`, v from `v_proj`, the Monarch blocks from `w1a`/`w1b`, the rank from `cond_v`. An attention has
taps exactly when `<prefix>q_taps` exists, so `LoadAttention` gives the decoder flavour for `layer00.` and the encoder
flavour for `enc00.` without a flag. The mask and RoPE are per call, because U3 (does the encoder rotate?) is open.

**Details a port must match.**

- **Half-split RoPE.** `apply_rope` rotates (xᵢ, xᵢ₊₂₄), not interleaved pairs, so `ops.RoPE` (F-008) does not fit.
  It stays untouched; `needle.Rope` is block-local. The tables round every step to float32 as the jnp code does:
  e = float32(2i)/48, f = 1/float32(θ^e), angle = float32(t)·f, then cos/sin. They match the fixture's `rope_cos` and
  `rope_sin` within 1.2e-7. Position = row index; a decoding offset (KV cache) is the decoder branch's addition.
- **GQA.** `jnp.repeat(k, 4, axis=1)` repeats each KV head in place, so query head h reads KV head h // 4 (heads 0–3
  → 0, 4–7 → 1), not h mod 2 as `jnp.tile` would. `kvHead` maps the index; nothing is copied. `ops.Attention` needs
  equal head counts and has no per-head hooks for the trace, so the attention core is a per-head loop over
  `tensor.DotProduct` and `tensor.Axpy`; `internal/ops` is unchanged.
- **Taps and the mask.** Upstream multiplies the j-th shifted term by the mask's j-th diagonal (`_mask_diag`). For the
  causal mask that is 1 wherever `shift_right` is defined (zeros elsewhere), and without a mask there is no factor, so
  `applyTaps` is Σⱼ taps[j] · shift_right(z, j), summed j = 0, 1, 2 as Python's `sum`. A padding mask would zero taps
  across padding; no Whistle path uses one, and `Forward` accepts only `MaskNone` and `MaskCausal`.
- **Masked logits.** Upstream fills them with float32 min before the softmax; exp underflows to exactly 0, and every
  causal row has an unmasked key. The port skips masked keys, which gives the same probabilities (the fixture's
  `probs` are 0 there, and so are ours).
- **Scale.** `qkᵀ / sqrt(float32(48))`, a division as upstream, not a multiplication by the reciprocal.
- **Norm.** mean(x²) is accumulated in float64 and rounded to float32 before `+ 1e-6` and the square root. Every
  qk-wide chunk of a row is one head, so the per-head q/k norms are `ZCRMSNorm.Apply` over the whole row buffer.
- **Kronecker apply.** `_kron_apply`'s einsum `"ij,ik,jl->kl"` is out = aᵀ · Z · b for Z = z.reshape(16, 32), which
  equals z · (a ⊗ b) with `np.kron`'s layout (`TestKronApplyMatchesDense`, 2×3, 4×4 and 16×32 against the dense
  matrix). The port computes Y = Z · b, then aᵀ · Y: 24,576 multiply-adds per row and stage instead of 262,144. The
  gathers are `out[i] = z[p[i]]`; `cond_v` (stored (512, 8), not transposed) is transposed once at load for the GEMM.
- **No padding.** `HadamardMLP` pads x to the next power of two; Whistle's width is 512 = n, so the port takes n from
  `d1` and requires inputs of that width.

**Permutations.** `LoadHadamardMLP` reads `hada_p1`/`hada_p2` from the `Source`: `.cact` records 226/227 or, from
safetensors, the numpy MT19937 port in `internal/cact/derived.go`. It checks that each is a permutation of n.
`TestHadaPermsFixture` pins the fixture's `p1`/`p2` (upstream's `RandomState(11 / 13).permutation(512)`) to
`cact.NumpyPermutation` without the model; `TestHadaPermsBothBackends` loads layer 0 through both backends and
requires identical permutations equal to the fixture's. `internal/weights` `TestWhistleCrossFormat` compares every
derived record across the backends as well.

**Result.** Safetensors (FP32) weights, both batch rows of each fixture, Apple M5 Pro. Every cosine is 1.000000000000
to 12 digits; "of bound" is max |Δ| / (1e-5 · max(1, max |ref|)):

| fixture             | tensors (worst first)                                                                   | max \|Δ\|                 | of bound |
| ------------------- | --------------------------------------------------------------------------------------- | ------------------------- | -------- |
| `zcrmsnorm`         | `rms_unit`, `y`, `rms`                                                                  | 4.8e-7, 8.9e-8, 1.2e-7    | ≤ 0.01   |
| `attention_decoder` | `gate` 0.39, `out` 0.21, `probs`/`attn` 0.13, `gated` 0.10, the other ten ≤ 0.09        | `out` 7.0e-4 at 329       | 0.39     |
| `attention_encoder` | `gate` 0.41, `gated` 0.15, `attn`/`v_proj` 0.11, `out`/`out_rope` 0.04, the rest ≤ 0.09 | `out_rope` 4.9e-4 at 1202 | 0.41     |
| `hadamard_mlp`      | `cond` 0.14, `z3`/`out` 0.13, `a2` 0.08, `z2` 0.06, `cond_logits` 0.04, `z1` 0.02       | `out` 2.0e-3 at 1481      | 0.14     |
| RoPE tables         | `rope_cos`, `rope_sin`                                                                  | 6.0e-8, 1.2e-7            | 0.01     |

The pure-Go fallbacks (`-tags purego`) give the same picture (worst `gate` 0.40). The only tensor above a fifth of its
bound is `gate`, whose max is 1, so its bound is the flat 1e-5. That gap is the fixture's own float32 rounding: a
probe that accumulates the gate logits in float64 lands 3.4e-6 (decoder) and 3.8e-6 (encoder) from the fixture,
against 3.9e-6 and 4.1e-6 for the float32 port, and F-009's float64 rerun measured 3.4e-6 and 3.9e-6 for the fixture.
The logits reach |78.5|; the float32 rounding of their 512-term sums passes through σ, whose slope is at most 1/4.

**Summation order depends on T.** `MatMulTransB` computes 4×4 blocks with NEON on arm64 and a 1×4 tail, so a row's
dot products take a different path when T changes. A 3-step causal run therefore equals the first three rows of a
5-step run only to about 1e-7, not bit for bit. Causality itself is exact: changing rows 3–4 at the same T leaves
rows 0–2 bit-identical (`TestAttentionCausal`).

**Not covered.** RoPE offsets and a KV cache for incremental decoding, padding masks, parallelism and speed (each
block is a single goroutine; the projections use `MatMulTransB`, the rest is scalar), and the remaining Phase 2
blocks (`MHC`, `Engram`, `Block`).

## F-012 — STFT, mel bank and the front-end hypothesis set (2026-10-10)

**Question.** Whistle's front end is closed. What is fixed, what stays open, and how is the open part represented so
that Phase 4 can settle it against `needle_embed`, given that there is no mel-level oracle?

**Sources.** The upstream snapshot (`original/needle` @ `af2654dffa`) has no front-end code: a grep for `mel`, `stft`,
`spectro`, `hann`, `log10` and `band` finds only `whistle.py`'s WAV I/O and unrelated hits. cactuscompute.com/whistle
says, verbatim: "16 kHz mono audio is framed at a 25 ms window and a 10 ms hop into 80 log-mel bins, band-limited to
250-3500 Hz and normalised per channel. Thirty seconds is 3,000 frames." The config and the audio manifest give
fft 512, win 400, hop 160 and mels 80 (F-003). Tensor 679 is librosa's Slaney bank over the full 0–8 kHz (F-006).

**Fixed.** 16 kHz input; a 400-sample window zero-padded to 512; hop 160; 257 bins; the mel bank
`audio.mel_filters` `(257, 80)` read from a `weights.Source`. `TestMelBankBackends` (model-gated, ran here) finds the
`.cact` and safetensors banks bit-identical to each other and to `cact.SlaneyMel`, as F-006/F-007 said.

**STFT on algo-dsp.** `dsp/stft` `STFT32` (v0.12.4) centres an NFFT-long frame on sample `i·hop` and multiplies it
by an NFFT-long window, with zero or reflect padding. `internal/frontend` passes it a 512-long custom window
holding the 400-sample Hann at an offset:

- centred (reflect, zero): offset 56, as torch.stft and librosa pad a short window, so frame i covers
  `[i·160 − 200, i·160 + 200)`;
- none (Kaldi `snip_edges`): offset 96 in algo-dsp frame `i + 1`, which covers `[i·160, i·160 + 400)`. algo-dsp's
  own `PadNone` needs all 512 samples inside the signal and loses the last frame (2997 instead of 2998 for 30 s).

The zeros around the window change only the phase, so the power equals that of the 400 windowed samples
zero-padded to 512. The Hann windows come from `dsp/window` (within 1e-15 of the formulas). One gap: algo-dsp's
reflect needs more than 256 samples, where only 200 are mirrored; 160–256 samples (one frame) return `ErrTooShort`.
The engine returns no encoder frame below 640 samples (rule below), so nothing usable is lost.

**Verified.** Power spectra against two independent references, for all 2 windows × 3 paddings
(`TestSTFTDirectDFT`, `TestSTFTNumpyFixture`):

| reference                                                                | max \|Δ\| / peak    | max rel., bins within 60 dB of the frame peak |
| ------------------------------------------------------------------------ | ------------------- | --------------------------------------------- |
| float64 direct DFT of explicitly framed samples (2037-sample tone+noise) | 1.0·10⁻⁷ – 3.2·10⁻⁷ | 1.5·10⁻⁵ – 4.7·10⁻⁵                           |
| numpy `rfft` fixture (`just mel-fixture`, 4321- and 300-sample inputs)   | 1.3·10⁻⁷ – 2.5·10⁻⁷ | 1.1·10⁻⁶ – 3.7·10⁻⁵                           |

Pinned at 1e-6 and 2e-4. The fixture is `testdata/frontend/stft.safetensors` (183 KB, 10 cases; the 300-sample burst
has no `none` frame); two runs are byte-identical. The mel product (float64 accumulation over each channel's
non-zero bins) equals the dense product to 1e-12.

**Frame count.** Centred framing gives `n / 160` (rounded down): 3000 for 30 s, the published count. torch.stft and
NeMo give `n/160 + 1` (3001) and algo-dsp `ceil(n/160)`. `none` gives `1 + (n − 400) / 160` (2998). The engine
decides between them through the embedding length. Three stride-2 halvings that round up (SAME, or pad 1 with
kernel 3), applied to each mel count, against the 32 corpus embeddings plus F-001's seven lengths:

| mel frames                   | encoder frames match    |
| ---------------------------- | ----------------------- |
| **`floor(n/160)`** (centred) | **39/39**               |
| `ceil(n/160)`                | 35/39                   |
| `1 + (n − 400)/160` (none)   | 36/39                   |
| `1 + (n − 512)/160`          | 33/39                   |
| F-001's `(n + 640) / 1280`   | 32/39 (25 of 32 corpus) |

Stems that round down or pad VALID, and `n/160 + 1`, score below 30/39 with every count. **Winner: `floor(n/160)`
mel frames and a stem that rounds up**, pinned in `TestFrameCountMatchesEngine`. This corrects F-001: its seven
synthetic lengths all had `n mod 1280` in {0, 1, 640}, which cannot tell `(n + 640)/1280` from
`ceil(floor(n/160)/8)`. Seven corpus clips with `n mod 1280` in [160, 640) can, and they follow the latter (e.g.
53040 samples: 42 frames, not 41). The rule supports centred framing; `none` stays in the set, since an engine could
pad its frames, but it starts behind.

**Hypothesis set (`frontend.Options`).** Each axis is a typed enum starting at 1, so a zero `Options` is invalid and
nothing is a default yet:

| axis          | values                                         | reading                                                                                                                                                                                           |
| ------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Window`      | `HannPeriodic`, `HannSymmetric`                | `cos(2πk/400)` (torch default, Whisper) or `cos(2πk/399)` (`np.hanning`, NeMo)                                                                                                                    |
| `Padding`     | `PadReflect`, `PadZero`, `PadNone`             | as above                                                                                                                                                                                          |
| `Spectrum`    | `Power`, `Magnitude`                           | \|X\|² or \|X\|                                                                                                                                                                                   |
| `Log`         | `LogLn`, `LogLog10`, `LogAddEps`, `LogWhisper` | `ln(max(x, 1e-10))`; `log10(max(x, 1e-10))`; `ln(x + 2⁻²⁴)` (NeMo's guard); Whisper: `y = log10(max(x, 1e-10))`, `max(y, max(y) − 8)` over the clip, `(y + 4)/4`                                  |
| `Norm`        | `NormNone`, `NormChannel`, `NormGlobal`        | `(x − mean)/(std + 1e-5)` with the n−1 std, per mel channel over time (NeMo `per_feature`) or over everything; one frame gives 0                                                                  |
| `Band`        | `BandFull`, `BandBins`, `BandChannels`         | ignore the claim; zero FFT bins outside [250, 3500] Hz before the bank (bins 8–112 stay; an ideal pre-filter); zero channels whose centre is outside (6–58 stay), which then sit at the log floor |
| `Preemphasis` | `PreemphNone`, `Preemph097`                    | `y[t] = x[t] − 0.97·x[t−1]`, `y[0] = x[0]`                                                                                                                                                        |

`Variants()` enumerates all 864 combinations in a stable order (last axis fastest); `Options.String()` is a stable
key (`win=… pad=… spec=… log=… norm=… band=… pre=…`) and `ParseOptions` reads it back. Unit tests show each value
doing what it says: windows against their formulas, reflect against zero (they differ exactly in the edge frames),
magnitude² = power, each log on known energies (the Whisper clamp lifts `log10(1e-5)` to `max − 8`), each norm
against a two-pass reference, `BandBins` zeroing exactly bins 0–7 and 113–256, `BandChannels` putting channels 0–5
and 59–79 at `ln(1e-10)`, and pre-emphasis equal to filtering first, with its analytic 30.8 dB tilt between
93.75 Hz and 5 kHz.

**Choices made for the set.**

- Pre-emphasis is not in PLAN.md's list. It was added because NeMo's `AudioToMelSpectrogramPreprocessor` in its
  Conformer configuration matches every published Whistle constant (n_fft 512, 25 ms / 10 ms, 80 librosa Slaney mels
  with Slaney norm, `dither` 1e-5 as in Whistle's config, `normalize="per_feature"` = "normalised per channel") and
  applies 0.97 pre-emphasis by default. `NeMo()` is that preset (symmetric Hann, zero padding, power, `ln(x + 2⁻²⁴)`,
  per-channel, 0.97); `Whisper()` is Whisper's (periodic Hann, reflect, power, Whisper log, no norm).
- The band limit is read two ways. Not implemented: a librosa bank re-derived over `fmin` 250, `fmax` 3500, because
  the shipped bank (679) is full band and is what the engine loads; and a time-domain IIR band-pass, whose order and
  shape nothing states. Either can be added if both readings fail in Phase 4.
- The `ln` and `log10` floors are Whisper's 1e-10. Normalisation statistics include every frame, since a clip has
  no padding frames here.
- Dither (1e-5 in the config) is training-only noise and is not applied.

**Smoke test.** All 32 clips under `ln`/no-norm, `Whisper()` and `NeMo()`: shapes `[n/160, 80]`, all finite. The
`ln` mel spans [−23.03, 3.01]; the floor values come from digital silence (under 5 % per clip), and the quietest
clip peaks at −4.06. The Whisper spread is at most 2 and the NeMo channels have mean 0 and std 1 (within 1e-3). All
864 variants on 1.5 s of one clip are finite. 30 s gives 3000 × 80 (2998 for `none`) in about 5 ms on an Apple
M5 Pro.

**Open, for Phase 4.** Every axis above. Resolve it with the encoder probe harness, iterating `Variants()` (greedily
per axis), by the mean per-frame cosine against `needle_embed`.

## F-013 — MHC, Engram and the decoder Block in Go (2026-10-10)

**Question.** Can float32 Go ports of `_ScanBody`'s mHC advance (with `_sinkhorn` and `Stack`'s lane expand and
collapse), of `Engram` with `engram_indices`, and of `Block` meet the F-009 contract on every tensor of the `mhc`,
`engram` and `block` fixtures? Which upstream details matter?

**API (`internal/needle`).** `MHC` (`LoadMHC(src, prefix, layer)`, `Forward(stream, T, block)`, `Lanes`, `Dim`) with
`ExpandLanes` and `CollapseLanes`; `Engram` (`LoadEngram(src, prefix, orders, tables)`, `Forward(tokens)` →
`*EngramKV`, `Indices`, `IndicesAt`, `Dim`, `Tables`, `Slots`, `MaxOrder`); `Block` (`LoadBlock(src, prefix)`,
`Forward(x, T, mask, rope, engram)`). As in F-011, the dimensions come from the shapes: the lanes from `mhc_b_pre`,
the width from φ, the slots and sub-width from `tables`, the model width from `key_proj` and the tap count from
`taps`. The orders and the table count are parameters, since the `.cact` header and the config hold them
(`weights.Config.Orders`, `Tables`). `MHC.Forward` takes the block as a function. The decoder passes a `Block`; the
encoder (`enc.mhc_*`, the same layout) will pass its Conformer block. `Block` is upstream's `Block` and so has no
cross-attention; Whistle's decoder adds one per layer (PLAN.md card), which is Phase 4's job.

**mHC details.**

- **Lane layout.** `lane = eye(n)[layer % n]`, `pre_off = 8·lane − 4`, `post_off = −4·(1 − lane)` match `Stack`
  (`TestLaneOffsets`). The layer's own lane gets +4 before `h_pre`'s sigmoid and 0 before `h_post`'s; the others get
  −4 and −4. The Source serves φ lane-major (F-006), so rows layer·n … layer·n + n − 1 of `mhc_phi_pre` are the
  layer's n columns and row 16·layer + 4i + j of `mhc_phi_res` is `h_res[i, j]`. The heads are then one
  `MatMulTransB` each.
- **Stream layout.** A row is n lanes of C values, lane-major, which is upstream's `(B, T, n, C)` and also
  `nx = rms_unit(stream.reshape(B, T, n·C))` without a copy. `rms_unit` runs over all 2048 values.
- **Rounding order.** `a · (nx @ φ) + b + off`, `x + σ(gate) · y`, `Σ_j h_res[i, j] · s_j + h_post[i] · y` and the
  engram's `x + α · v` round the product before the add (explicit `float32(…)` conversions), as upstream's
  elementwise jnp ops do. Go may otherwise fuse them to FMA on arm64. The effect is far below the contract either
  way; it keeps the port's rounding close to the reference's.
- **`y = block(u) − u`** is computed from the block's full output, as upstream does. It is not the block's residual
  branch, which would round differently.

**Sinkhorn: 20 iterations do not converge on Whistle's `h_res`.** `_sinkhorn` subtracts each row's logsumexp, then
each column's, 20 times, then takes `exp`. The port does exactly that in float32, each logsumexp shifted by its
maximum like `jax.nn.logsumexp`. It stays finite for logits of any finite size (1e30 in `TestSinkhornLargeLogits`,
where a direct `exp` overflows above 88.7). Because the last step normalises the columns, they sum to 1 within float32
rounding (≤ 1.4e-7 on the fixture). The rows do not: Whistle's `h_res` logits reach 70.8 (layer 0) and 19.9
(layer 1), Sinkhorn converges slowly on such peaked matrices, and after 20 iterations the fixture's row sums deviate
from 1 by up to **6.7 %** (layer 0) and 5.4 % (layer 1). After 200 iterations they still deviate by 0.3 %. So
`h_res` is column-stochastic, not doubly stochastic, and the iteration count is part of the model. Run on the
fixture's own logits, the port reproduces the fixture's `h_res` within 1.2e-7 at 20 iterations; at 19 or 21 it
misses by 2.9e-3 to 4.4e-3, 290–440 times the bound (`TestSinkhornFixtureIterations`, no model needed). One
consequence corrects F-009: on `Stack`'s lane expand, lane i of `Σ_j h_res[i, j] · x` is (row sum i) · x, not x, so
`h_res` does matter at layer 0 (`TestMHCExpandedStream`). The `layer0` case therefore pins the Sinkhorn too.

**Engram details.**

- **Hash.** For table c = oi·hashes + h: `acc = 0x9E3779B9 · (c + 1)`, then for j < order `acc = (acc ^ tok[t − j]) ·
0x01000193`, then `acc ^= acc >> 15`, all `uint32` with wraparound (Go's `uint32` arithmetic is exactly
  `jnp.uint32`), and finally `acc mod slots`. Tokens before the start count as 0 (`_shift_right` pads zeros). The seed
  stride between orders is the hash count, because Whistle's `engram_seed_heads` is 0 (F-003); upstream sets it only
  for its half-width sub-model. `TestNgramHashHandComputed` pins seven hand-computed cases, one also worked on paper
  (col 0, order 1, token 5 → `0xCD55385D`). `TestEngramMatchesReference` checks against a uint64 transcription masked
  to 32 bits. All 96 fixture indices match exactly, including the pad id 0, the repeated trigram and the positions
  before `order − 1`, whose hashes run over the zero padding.
- **Incremental form.** A position's indices depend only on its last `MaxOrder` (3) tokens. `IndicesAt(dst, hist)`
  takes that window, shorter at the start, and treats the missing tokens as zeros. `Indices` is built from it, and
  `TestEngramIndicesIncremental` checks that a rolling 3-token window reproduces the sequence's indices at all 40
  positions. The decoder's KV cache (Phase 5) also needs the last 3·3 = 9 `v_raw` rows for the dilated taps; that
  cache is not built yet.
- **Masks.** With the causal mask, `ngram_ok[t, c] = (t ≥ order_c − 1)`: the diagonal is all ones and `_mask_diag`
  zero-pads the first `order − 1` steps. `tap_ok[j, t] = (t ≥ 3j)`. The test checks both against the fixture. The
  first is the table-row mask. The second equals where `shift_right(v, 3j)` is defined, so the taps reuse
  `applyTaps`, which now takes a dilation (1 for the attention, F-011).
- **Table layout.** The Source serves `engram0.tables` as (4 · 18432, 128) (F-006), so table c's slot s is row
  c·18432 + s. The lookup is a copy, and `e` matches the fixture exactly (max abs 0).
- **Gate.** `α = σ(⟨rms_unit(x), rms_unit(k)⟩ / √512)` and `x += α · v`, with the division by `float32(√512)` as
  upstream. Upstream sums every site's term times its flag. The flags are one-hot per layer, so one site per `Block`
  call is exact. `Block.Forward` takes the site's `*EngramKV`, or nil away from the sites (3 and 7).

**Result.** FP32 safetensors weights, Apple M5 Pro, every tensor of all seven fixtures. "Of bound" is max |Δ| /
(1e-5 · max(1, max |ref|)). The first four rows are F-011's blocks, rerun unchanged:

| fixture             | tensors | worst (of bound)                    | max \|Δ\| at max \|ref\|           | min cosine     |
| ------------------- | ------- | ----------------------------------- | ---------------------------------- | -------------- |
| `zcrmsnorm`         | 3       | `rms_unit` 0.01                     | 4.8e-7 at 3.6                      | 1.000000000000 |
| `attention_decoder` | 15      | `gate` 0.39                         | 3.9e-6 at 1.0                      | 1.000000000000 |
| `attention_encoder` | 11      | `gate` 0.41                         | 4.1e-6 at 1.0                      | 1.000000000000 |
| `hadamard_mlp`      | 7       | `cond` 0.14                         | 7.9e-6 at 5.6                      | 1.000000000000 |
| `mhc`               | 24      | `layer1.stream_out` 0.18            | 5.8e-4 at 317                      | 0.999999999999 |
| `engram`            | 7 + I64 | `v_raw` 0.12; `indices` exact 96/96 | 5.3e-5 at 46                       | 1.000000000000 |
| `block`             | 7       | `h` 0.18                            | 9.9e-5 at 56; `out` 1.1e-3 at 1457 | 0.999999999999 |

Within `mhc`: `layer0.collapsed` 0.17, `layer0.stream_out` 0.14 (1.6e-3 at 1162), `layer0.final` 0.06, `h_res` 0.05
(layer 0) and 0.14 (layer 1), and every head ≤ 0.16. `engram`'s `alpha` is 0.01 and `x_out` 0.01, both computed from
the fixture's k and v so that the gate is checked on its own. The pure-Go fallbacks (`-tags purego`) give the same
picture: the worst value is again `gate` at 0.40, and the worst of the new tensors is `layer0.collapsed` at 0.19.

**Not covered.** The decoder's cross-attention and the full decoder stack (Phase 4). The KV cache, the rolling
`v_raw` history for the engram taps, and RoPE offsets (Phase 5). The encoder's Conformer block that `MHC` will wrap.
Parallelism and speed.

## F-014 — Encoder probe harness and the first search (2026-10-10)

**Question.** The encoder runs inside the closed engine, and only its final output (`needle_embed`) is observable.
Which reading of each open choice (front end, stem, block, positions, output) reproduces it, and how is a wrong early
guess kept from hiding a right later one?

**Instrument.** `internal/encoder` builds the encoder for one `Hypothesis` from a `weights.Source` by the F-006 names:
the `frontend.Options` (864 variants, F-012) plus 19 typed encoder axes, each value a named constant starting at 1.
That is 8 stem axes (stride placement, padding, kernel axis order, flatten order, three activations, input scale), the
`pe_gate` use, RoPE, and 9 block axes (FFN scale, GLU form, conv activation, `conv_out_norm` position, depthwise
padding, module order, mHC, Hadamard permutations, output tap). `Load` rejects with `ErrRuledOut` what the tensor
shapes exclude. The harness `internal/encoder/probe_test.go` (build tag `oracle`, `just encoder-probe`) scores a
hypothesis by the mean over clips of the mean per-frame cosine against `testdata/oracle/<clip>.embed.safetensors`. It
caches the log-mel per front end and the stem output per stem key, runs (hypothesis, clip) pairs on all cores, and
rules out without a score any hypothesis whose frame count differs from the engine's on a clip. One hypothesis on all
32 clips takes about 0.3 s (Apple M5 Pro); the whole search from the prior takes about 3 min.

**Is the score meaningful?** `TestProbeBaselines` over the 32 clips: each engine frame against the mean engine frame of
the other 31 clips gives **0.2349** (0.157–0.284); a fixed random projection of 8 stacked NeMo log-mel frames gives
**−0.0048** (−0.019 to +0.009). So about 0.23 is what a constant direction (mostly `enc.final_norm`'s gain) earns, and
anything below it carries no signal. The starting hypothesis (`prior()`, below) scores 0.2485 on the inner clips and
0.18 on all 32: no signal yet.

**Search order and beam.** Six inner-loop clips, the shortest from both LibriSpeech splits and FLEURS nl/es (28–71
frames). Stages, each the full product of its axes around every kept hypothesis, with a beam of 4:

1. stem geometry (`stride`, `stempad`, `kernel`, `flatten`), because every later score depends on the stem;
2. stem activations and input scale (`convact`, `dwact`, `pwact`, `scale`, 1000 combinations per beam entry);
3. front end: `log`, `norm`, `spec`;
4. conv module: `glu`, `blockact`, `convnorm`, `dwpad`;
5. block structure: `ffn`, `order`, `mhc`, `perms`;
6. positions and output: `rope`, `pe`, `out`;
7. front end: `win`, `pad`, `band`, `pre`.

Then coordinate descent with escalating moves around the 4 kept hypotheses. First every axis on its own. If that
leaves the leader unchanged, every pair of axes jointly (351 pairs). If that does too, every stage's joint product
again. Any move that changes the leader restarts from single axes, and the search stops when no level moves it.
Results are cached per hypothesis, so a re-scored hypothesis costs nothing. Finally, the leader's sensitivity table
(each value of each axis, all else at the leader). It certifies that no single-axis change improves, and gives the
cost of every rejected value.

**Masking.** Only the final embedding is observable, so a wrong early choice could hide a right later one. Three
things guard against it. The beam keeps 4 hypotheses per stage. The descent re-opens every axis once all others are
in place: in the run below it changed `log` and `pre` after stage 7. The pair and stage moves escape coupled axes.
The test is a search started from a hypothesis wrong on all 26 axes (`WHISTLE_PROBE_KEY`, reference score 0.017903,
below the mean-frame baseline). With single-axis descent alone it stalled at 0.753162 in a coupled optimum:
`scale=sqrt-dim` with `ffn=1`, `convnorm=after-act` and ReLU in the stem compensate for each other. Pairs lifted it
to 0.866608 (`ffn=half`, `convnorm=before-act`) and stalled again on the stem cluster `scale`/`convact`/`dwact`/
`pwact`. The stage move (the stem-activation product) then reached **0.999819**, the same leader as from the prior,
32-clip mean 0.9995, min 0.9929. That took 7.5 min against about 1–2 min from the prior. Without the escalation
the search is not robust to an adversarial start, so the escalation is part of the harness.

**Starting hypothesis** (`prior()`): NeMo front end; stride (2, 2) on all convs, SAME, kernel [time, mel], NHWC
flatten (channel fastest), SiLU after the conv and the pointwise convs; RoPE off, `pe_gate` unused; FFN ½, GLU
a·σ(b), SiLU, `conv_out_norm` before the activation, centred depthwise, macaron order, mHC lanes, shared
permutations, `final_norm` output.

**Run** (`just encoder-probe`, `.cact` weights, beam 4, 6 inner clips). Stage leaders (mean / min over the inner
clips):

| stage                            | leader's change from the prior | mean     | min      |
| -------------------------------- | ------------------------------ | -------- | -------- |
| reference (prior)                | —                              | 0.248542 | 0.204152 |
| 1 stem geometry                  | `stempad=sym flatten=ch-mel`   | 0.774874 | 0.645987 |
| 2 stem activations               | (unchanged)                    | 0.774874 | 0.645987 |
| 3 front end log/norm/spec        | `log=log10` (`ln` ties)        | 0.778428 | 0.673380 |
| 4 conv module                    | (unchanged)                    | 0.778428 | 0.673380 |
| 5 block structure                | (unchanged)                    | 0.778428 | 0.673380 |
| 6 positions and output           | `rope=1e5` (`pe` ties)         | 0.999164 | 0.997439 |
| 7 front end win/pad/band/pre     | `pre=none`                     | 0.999193 | 0.997534 |
| descent 1, single axes           | `log=whisper`                  | 0.999819 | 0.999710 |
| descent 2–4, single/pairs/stages | (unchanged: converged)         | 0.999819 | 0.999710 |

The leader with `pe=none` was `encoder.Best()` (F-019 keeps `pe=none` by the silence probes; F-020 moves the log to
the soft knee `soft-76db`, 32-clip mean 0.999828, min 0.999033):
`win=hann-symmetric pad=zero spec=power log=whisper norm=channel band=full pre=none stride=all stempad=sym
kernel=time-mel flatten=ch-mel convact=silu dwact=none pwact=silu scale=1 pe=none rope=1e5 ffn=half glu=a*sig(b)
blockact=silu convnorm=before-act dwpad=centred order=macaron mhc=lanes perms=shared out=final-norm`.

**All 32 clips** (`.cact` weights): mean per-frame cosine **0.999539**, every clip ≥ **0.992932**
(fleurs-fr_fr-10174413140250646001); the next worst is 0.9981 (ls-clean-5639-40744-0014), all others ≥ 0.9992. The
worst frames are near-silent (F-018).

**Weight sources.** The same hypothesis on the float checkpoint (`whistle.safetensors`) gives mean **0.921019**, min
0.881934, max 0.938908 over the 32 clips. The dequantised `.cact` is closer to the engine by 0.0785 in mean cosine:
the engine computes with the CQ2/CQ4 weights. Whatever its 8-bit activations cost fits in the remaining gap of 4.6e-4
in mean cosine. Around the
leader, the sensitivity table on the 8 inner clips picks the same value on every axis with either source, and only the
ties of F-017/F-018 change sign.

The full search run on the float weights (`WHISTLE_PROBE_SRC=safetensors just encoder-probe`, 5 min) finds the
same stem, block, positions, output, spectrum, normalisation and band. It differs from `Best()` only on three
front-end axes: `pad=reflect`, `log=log10` and `pre=0.97`, mean 0.923691 on its inner clips. That leader overfits
the 6 inner clips. On all 32 it scores 0.920582 with the float weights, below `Best()`'s 0.921019. With the `.cact`
weights it scores 0.998560, and one clip falls to 0.987513, below the Accept. The float checkpoint is too far from
the engine to decide axes worth less than about 2e-3. The `.cact` search's choices hold on all 32 clips, and it
converges to the same leader from the prior (3 min) and from the adversarial start (7.5 min).

**Fast regression.** `TestBestMatchesEngine` (no tag, skips without `models/`) runs `Best()` on the three shortest
clips and pins the mean per-frame cosine: `.cact` 0.999709 / 0.999842 / 0.999833 (floors 0.9996 / 0.9997 / 0.9997),
safetensors 0.9132 / 0.9389 / 0.9366 (floors 0.912 / 0.938 / 0.935).

## F-015 — `needle_embed` returns `enc.final_norm` of the lane mean (2026-10-10)

**Question.** Is the engine's embedding before or after `encoder/final_norm`, and before or after the lane mean
(PLAN.md Phase 4)? F-001 left open whether 512 is the encoder width.

**Direct test.** ZCRMSNorm makes `y / (1 + scale)` have RMS exactly 1 in every row, whatever x is. For all 2205 engine
rows of the 32 clips, `rms(row / (1 + enc.final_norm))` with the `.cact` (FP16) gain is 1 within **1.8e-7**, float32
rounding (`TestEmbedIsFinalNormOutput`, model-gated). With the safetensors (FP32) gain it is 0.9974–1.0000, so the
engine applies the FP16 gain the `.cact` stores. With the decoder's `final_norm` it is 6.0–98.7. The raw row RMS
spans 0.31–1.18, so this is not a property of any embedding.

**By score.** Around the leader, the lane mean before `final_norm` (`out=lane-mean`) costs −0.396920 (F-014). The
final norm is applied to the lane mean as in upstream's `Stack` (`jnp.mean(x, axis=2)`, then `final_norm`). mHC
itself (`mhc=lanes` against a plain stack) is worth 0.678340.

**Winner.** `needle_embed` is `enc.final_norm(mean over the 4 mHC lanes)`, `[frames, 512]`, 512 being the model width.
The direct test fixes "after `final_norm`"; the lane mean rests on the score and on upstream's `Stack`, since a single
lane is not among the hypotheses.

## F-016 — The conv stem: strides, padding, layout and activations (U2) (2026-10-10)

**Question.** U2: how do `stem/w [3,3,1,128]`, `stem/dw_{1,2}`, `stem/pw_{1,2}` and `stem/out [1280,512]` turn
`[frames, 80]` log-mel into `[frames/8, 512]`?

**Ruled out by shapes and frame count.** `stem/out` takes 1280 = 10 mel positions × 128 channels, so the mel axis
must halve three times: stride only on the depthwise stages leaves 20 positions (2560 features), VALID padding 9 (1152).
Both fail at `Load` (`TestRuledOutByShapes`), 16 of stage 1's 24 hypotheses. SAME and symmetric padding both give
ceil(n/2) per stage, so 3000 mel frames → **375** and the engine's frame rule `ceil(floor(n/160)/8)` (F-012) on every
corpus clip and on F-001's lengths (`TestStemFrames`, `TestEmbed30s`).

**Decided by score** (Δ against the leader's 0.999819, inner clips, `.cact`):

| axis              | winner                                  | alternatives                                                       |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------ |
| padding           | **symmetric (1, 1)**, torch `padding=1` | lax SAME (0 before, 1 after on even lengths) −0.243827             |
| kernel axes       | **[time, mel]** (H = time)              | [mel, time] −0.509124                                              |
| flatten           | **channel-major (NCHW): `c·10 + mel`**  | NHWC `mel·128 + c` −0.713620                                       |
| after `stem/w`    | **SiLU**                                | GELU (tanh or erf) −0.000569, ReLU −0.039795, none −0.063222       |
| after each `dw_k` | **none**                                | GELU −0.092201, SiLU −0.092369, ReLU −0.317228                     |
| after each `pw_k` | **SiLU**                                | GELU tanh −0.027121, erf −0.027152, ReLU −0.121726, none −0.815920 |
| input scale       | **1**                                   | ×√512 −0.837042                                                    |

So the stem is `conv3×3/2 → SiLU → [dw3×3/2 → pw → SiLU] × 2 → NCHW flatten → stem/out`, zero padding 1 on every side,
no biases (the checkpoint has none). The symmetric padding and the NCHW flatten suggest the engine, and maybe the
trainer, uses PyTorch-style convolutions. The closest runner-up is GELU after the first conv, at 5.7e-4.

## F-017 — The encoder block (U3, U5) (2026-10-10)

**Question.** U3 (conv module, macaron scale, `pe_gate`, positions) and U5 (do the encoder's Hadamard MLPs share the
decoder's permutations?).

**Ruled out by shapes.** `pw1` is (1024, 512) and `dw` (9, 512), so pw1's output must be halved before the depthwise
conv: "no GLU" cannot run (`ErrRuledOut`).

**Decided by score** (Δ against 0.999819, inner clips, `.cact`):

| axis                  | winner                                                  | alternatives                                                       |
| --------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| FFN residual scale    | **½ (macaron)** on both Hadamard MLPs                   | 1 −0.622692                                                        |
| GLU                   | **a·σ(b)**, a = first 512 channels                      | a·silu(b) −0.403596, σ(a)·b −0.978467, silu(a)·b −0.992501         |
| activation after `dw` | **SiLU**                                                | GELU tanh −0.005362, erf −0.005372, ReLU −0.110529, none −0.820849 |
| `conv_out_norm`       | **between `dw` and the activation** (BatchNorm's place) | after the activation −0.185564, after `pw2` −0.896173              |
| depthwise padding     | **centred**, 4 + 4 frames                               | causal −0.907688                                                   |
| module order          | **macaron**: FFN½, attention, conv, FFN½                | conv before attention −0.415660                                    |
| encoder RoPE          | **θ = 1e5**, half split, as the decoder                 | θ = 1e4 −0.029625, none −0.221335                                  |
| residual stream       | **mHC, 4 lanes, lane = layer % 4**, as the decoder      | plain stack −0.678340                                              |
| permutations (U5)     | **shared `hada_p1/p2`** (RandomState 11/13)             | `split=True` halves −0.980344, identity −0.977999                  |

The block is therefore
`x += ½·hada_0(pre_hada_0(x)); x += σ(attn_gate)·post_norm(attn(norm_in(x), no mask, RoPE 1e5));
x += pw2(silu(conv_out_norm(dw(glu(pw1(conv_norm(x))))))); x += ½·hada(pre_hada(x))`, inside the same mHC
advance as the decoder (`needle.MHC` with `enc.mhc_*`). The attention is `needle.Attention` without taps.

**`pe_gate` stays open.** σ(−6.1094) = 0.00222, so a sinusoid gated by it has RMS 0.0016 against a stem output RMS of
4.6–5.1, about −70 dB. Adding it to the stem output, as `[sin | cos]` or interleaved, moves the score by ≤ 1e-6
(−0.000000 and −0.000001). As a RoPE mix, `(1 − g)·x + g·RoPE(x)` replaces the RoPE that wins and costs −0.221535,
as much as no RoPE. `Best()` leaves `pe_gate` unused. The embeddings cannot tell whether the engine uses it; a
probe with a far larger signal (or the decoder side, Phase 5) could. **Decided by such probes in F-019: unused.** On
digital silence the stack input is exactly 0, and every placement loses at least 0.0088 there.

## F-018 — The front end against the encoder oracle (U1, partial) (2026-10-10)

**Question.** U1: which of F-012's 864 front-end hypotheses does the engine run?

**Decided by score** (Δ against 0.999819, inner clips, `.cact`):

| axis          | winner                                                              | alternatives                                                             |
| ------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| normalisation | **per mel channel** over time                                       | global −0.015016, none −0.045222                                         |
| band          | **full** (the shipped bank)                                         | FFT bins outside 250–3500 Hz zeroed −0.030047, channels zeroed −0.038980 |
| spectrum      | **power**                                                           | magnitude −0.001683                                                      |
| pre-emphasis  | **none**                                                            | 0.97 −0.001250                                                           |
| log           | **Whisper clamp**: `log10`, then `max(y, max(y) − 8)` over the clip | `ln(x + 2⁻²⁴)` −0.000324, `ln`/`log10` with floor 1e-10 −0.000625        |
| centring      | zero or reflect (tie)                                               | `none` (snip edges) −0.027794                                            |
| window        | symmetric or periodic Hann (tie)                                    | —                                                                        |

The "250–3500 Hz band-limited" sentence of cactuscompute.com/whistle does not describe a mask on the mel input in
either reading. Per-channel normalisation removes any affine map of the log, which is why `ln` and `log10` tie and
why Whisper's `(y + 4)/4` drops out. The Whisper winner differs from `ln` only by its clamp at 80 dB below the clip's
peak.

**Weight-source dependence.** `pre` and `log` (margins 1.3e-3 and 3e-4) are decided with the `.cact` weights. The
float-weight search prefers 0.97, `log10` and reflect padding on its inner clips, which loses on all 32 clips with
either source (F-014). The `log` choice is the weakest decision of this table.

**Ties.** The window (Δ −0.000001) and reflect against zero padding (−0.000040 with `.cact`, +0.000014 with
safetensors) are below what the embeddings resolve; `Best()` keeps the search's choice (symmetric, zero). **F-020:**
the DC probes decide zero centring (0.9998 against 0.8775); the window still ties.

**Open: quiet frames.** The frames below 0.995 are all near-silent: peak |x| < 3e-3 within the frame's 80 ms, and a
log10-mel maximum ≤ −3.7. The three worst are fleurs-fr_fr-10174413140250646001 frames 13–14 (0.756, 0.675, peak 3e-4)
and ls-clean-5639-40744-0014 frame 69 (0.883). Elsewhere frames reach 0.9999. The floor that the clamp approximates,
dither or the engine's 8-bit activations on low-energy frames remain candidates for the next session. They cost
0.007 on the worst clip and do not threaten the 0.99 target. **F-020:** the Whisper clamp's hard edge is the cause.
A soft knee 76 dB below the clip's peak (`log=soft-76db`) lifts these three frames from 0.756, 0.675 and 0.883 to
0.9926, 0.9953 and 0.9558, and the worst clip to 0.999713. The absolute floor never binds on speech, and the probes
rule out dither.

## F-019 — `pe_gate` is unused: the silence probes (U3) (2026-10-10)

**Question.** Where does `pe_gate` (σ(−6.1094) = 0.00222) add a position signal, if anywhere? On speech a sinusoid
gated by it sits about 70 dB below the stem output, so the corpus embeddings cannot tell (F-017).

**Probes.** `tools/oracle/probe` synthesises 30 signals, `just oracle-probes` (`oracle probes`, build tag `oracle`) runs
them through `needle_embed` with telemetry off and writes `testdata/oracle/probes/<name>.embed.safetensors` plus
`manifest.json` (engine and model pins, every generator parameter, the frame counts and the SHA-256 of each signal):
1217 encoder frames, 2.5 MB. Two runs are byte-identical. The encoder tests regenerate each signal from the manifest
and refuse it unless its digest matches (`loadProbes`). Interior frames are those whose stem input
`[1280t − 1320, 1280t + 1320)` lies inside the clip.

| family      | probes                                                                                                                            | frames (interior) | why it discriminates                                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| silence     | 0.32, 1, 2, 4, 10, 30 s                                                                                                           | 4 … 375 (1 … 372) | the stem output is exactly 0 (below), so any additive position term is the stack's only input; dither would be normalised to full scale; the lengths reach position 374  |
| DC          | 0.01, 1e-4, 2 s                                                                                                                   | 25 (22)           | reflect centring makes every mel frame equal (the silence rows), zero centring changes the frames that reach past the clip; the two levels test scale invariance         |
| tone, burst | 440 Hz at 1e-1, 1e-3, 1e-5, steady or in the middle second                                                                        | 25 (22) each      | Hann leakage spans 15 decades, so any absolute floor binds somewhere; a burst makes each channel two-level, so the floor decides which channels carry the on/off pattern |
| noise       | σ 1e-1, 1e-5                                                                                                                      | 25 (22)           | every channel at a similar level; 1e-5 is the config's dither                                                                                                            |
| decay       | noise from −20 to −140 dB over 4 s                                                                                                | 50 (47)           | every channel crosses the floor, whose shape bends its trajectory                                                                                                        |
| impulse     | one click at 16000 + δ, δ ∈ {0, 20, 40, 80, 120, 140}, 0.1; δ 0 at 1e-2, 1e-4, 1e-6; δ 80 at 1e-3, 1e-5, 1e-6; a pair 40 dB apart | 25 (22) each      | 2–3 frames see the click at known window offsets (centring, window shape); falling levels test the floor on frames with tiny energy                                      |

**Why silence decides.** On digital silence every mel frame is equal, so per-channel normalisation gives exactly 0.
The stem has no biases and SiLU(0) = 0, so the stem output and the stack input are exactly 0
(`TestPEPlacements`). A position term added before or inside the blocks is then the stack's only input, and the
ZCRMSNorms scale it up to unit RMS, so its gain of 0.0022 no longer hides it. Without one the engine's rows are still
neither zero nor constant (adjacent-frame cosine 0.3–0.5, deterministic): each Hadamard MLP adds the constant its `b2`
bias makes, the blocks' zero-padded depthwise convs set the frames near the edges apart, and the global attention
with RoPE spreads that by position. So "the interior frames are identical" is not a test here, but the model is:
the hypothesis must reproduce the engine's varying rows.

**Placements** (`encoder.PE`; each adds σ(pe_gate) times the sinusoid of its row width w, frequencies
10000^(−2i/w), as `[sin | cos]` (concat) or interleaved; `TestProbePEGate`, Best otherwise, `.cact`):

| `pe`                           | where the sinusoid is added                 | silence mean | silence min  | 32 clips |
| ------------------------------ | ------------------------------------------- | ------------ | ------------ | -------- |
| **none**                       | —                                           | **0.999946** | **0.999916** | 0.999828 |
| stem-interleaved               | stem output (after `stem/out`, w = 512)     | 0.991178     | 0.977333     | 0.999829 |
| flat-concat                    | before `stem/out` (w = 1280)                | 0.962869     | 0.909087     | 0.999834 |
| block-interleaved              | every block's input u, inside mHC (w = 512) | 0.953773     | 0.936508     | 0.999825 |
| flat-interleaved               | before `stem/out`                           | 0.945616     | 0.904476     | 0.999832 |
| rope-mix                       | q, k rotated by (1 − g)·x + g·RoPE(x)       | 0.942294     | 0.914018     | 0.714247 |
| mel-concat                     | stem input, the normalised log-mel (w = 80) | 0.941623     | 0.917200     | 0.999831 |
| stem-concat                    | stem output                                 | 0.933335     | 0.844558     | 0.999828 |
| mel-interleaved                | stem input                                  | 0.925033     | 0.845102     | 0.999824 |
| block-concat                   | every block's input                         | 0.892453     | 0.821885     | 0.999827 |
| stem-ungated (control, gain 1) | stem output                                 | 0.884709     | 0.774427     | 0.998886 |
| none, `rope=none`              | —                                           | 0.942122     | 0.913784     | 0.714435 |
| none, `rope=1e4`               | —                                           | 0.985391     | 0.973419     | 0.953470 |

Per length, `pe=none` gives 0.999972, 0.999969, 0.999952, 0.999939, 0.999916 and 0.999927 (0.32 s to 30 s). That is
the engine's own reproducibility: the same tone, burst, noise or DC at two levels 2 to 4 decades apart gives
0.999926–0.999957 against itself (F-020). Every placement loses at least 0.0088 on average and 0.0226 on 10 s of silence; on the corpus they all
tie within 6e-6, except the ungated control and the RoPE mix. The silence probes also confirm the encoder's RoPE:
none scores 0.942122, θ = 1e4 0.985391.

**Winner.** `pe_gate` is unused in the encoder: with RoPE as its only position term the model reproduces the engine's
silence rows to 5e-5 at every length up to 30 s, and no tested placement comes within 0.0088. `Best()` keeps
`pe=none`. Whether the decoder reads `pe_gate` is not tested here.

## F-020 — The log floor is a soft knee 76 dB below the peak; zero centring, no dither (U1) (2026-10-10)

**Question.** F-018 left the near-silent frames (worst clip 0.992932, frames down to 0.675) and the window and
centring ties. Which floor does the engine put under quiet mel energies: Whisper's absolute 1e-10, its clamp 8
decades below the clip's peak, an epsilon inside the log, or dither? The probes of F-019 add signals where the floor
dominates.

**No absolute floor on tones, noise or DC.** A front end with only a log, a clamp relative to the peak and per-channel
normalisation gives the same rows at any gain. The engine does, from 1e-1 to 1e-5 (`TestProbeScaleInvariance`):
tone 0.999957 and 0.999953, burst 0.999943 and 0.999949, noise 0.999934, DC 0.999926. That is the same 5e-5 as the
silence rows, the engine's own reproducibility level. Whisper's 1e-10 floor binds on the quiet tones and breaks this
(tone 1e-5 0.932919, burst 1e-5 0.947681 under the old `Best()`; 0.999706 and 0.999216 with the soft knee below). On
speech no absolute floor binds: `max(x, F)` with F from 1e-10 to 1e-20 under the 8-decade clamp gives 0.999539 every
time (a one-off sweep, like every number in this entry not attributed to a test).

**Hard clamp, soft knee or epsilon** (`TestProbeLogKnee`, 32 clips, `.cact`, every other axis Best's; the soft knee is
log10(x + 10^−d · max x), the hard clamp max(log10 x, log10 max x − d)):

| depth d  | soft knee mean | soft knee min | hard clamp mean    | hard clamp min |
| -------- | -------------- | ------------- | ------------------ | -------------- |
| 7.30     | 0.999464       | 0.995388      | 0.999235           | 0.996223       |
| 7.50     | 0.999766       | 0.998921      | 0.999593           | 0.998506       |
| **7.60** | **0.999828**   | **0.999033**  | 0.999632           | 0.997588       |
| 7.65     | 0.999826       | 0.999037      | 0.999634           | 0.996607       |
| 7.80     | 0.999762       | 0.996081      | 0.999601           | 0.994170       |
| 8.00     | 0.999651       | 0.992779      | 0.999539 (Whisper) | 0.992934       |
| 8.20     | 0.999549       | 0.990966      | 0.999418           | 0.990909       |

Every soft depth from 7.35 to 8.2 beats the hard clamp at 8 in mean, and at every depth the soft knee beats the hard
clamp of the same depth, by 1.1e-4 to 2.3e-4. Fitted on one half and scored on the other:

| fit → held out | fitted d | held-out mean / min | hard clamp 8 on the same clips |
| -------------- | -------- | ------------------- | ------------------------------ |
| even → odd     | 7.60     | 0.999811 / 0.999245 | 0.999699 / 0.998082            |
| odd → even     | 7.85     | 0.999599 / 0.995128 | 0.999380 / 0.992934            |
| ls → fleurs    | 7.80     | 0.999599 / 0.996081 | 0.999282 / 0.992934            |
| fleurs → ls    | 7.60     | 0.999821 / 0.999033 | 0.999694 / 0.998082            |

So the shape is a finding and the depth a fit. Fitted per clip, the depth ranges from 7.50 to 8.55 (σ 0.20). Taking
the reference to be the waveform's peak (σ 0.24) or its mean square (σ 0.22) instead of the mel's peak spreads it
wider. An absolute epsilon inside the log does worst of the three: log10(x + 3e-8) gives 0.999679, 0.999734 under the
8-decade clamp (min 0.998582, the quiet fleurs-nl clip), and x + 1e-8 0.999323. Ruled out on the way, all with Best's other axes: a knee referenced to the power
spectrum's peak (best 0.999559 at 9.5 decades); FP16 storage of the max-normalised mel (0.999519); int8 quantisation
of the stem input, per tensor, row or group of 32 (0.999529–0.999560); Kaldi's per-frame DC removal (0.999303); a
squared Hann window (the 0.1 clicks rise to 0.9997, tones fall to 0.985).

**The probes, front end** (`TestProbeFrontEnd`, around the new `Best()`; family means silence, DC, tone, burst,
noise, decay, impulse):

| change from `Best()`                        | silence  | probes (min)        | dc / tone / burst / noise / decay / impulse     | 32 clips (min)      |
| ------------------------------------------- | -------- | ------------------- | ----------------------------------------------- | ------------------- |
| — (`soft-76db`, symmetric, zero, no dither) | 0.999946 | 0.984252 (0.845708) | 0.99984 0.99974 0.99924 0.99993 0.99943 0.96400 | 0.999828 (0.999033) |
| `win=hann-periodic`                         | 0.999946 | 0.984247 (0.845671) | 0.99981 0.99977 0.99915 0.99992 0.99943 0.96401 | 0.999829 (0.999149) |
| `log=soft-80db`                             | 0.999946 | 0.984350 (0.845706) | 0.99968 0.99992 0.99991 0.99993 0.99899 0.96409 | 0.999651 (0.992779) |
| `log=whisper` (F-018)                       | 0.999946 | 0.990395 (0.932919) | 0.98876 0.97477 0.97788 0.99725 0.99889 0.99102 | 0.999539 (0.992932) |
| `log=ln-eps`                                | 0.999946 | 0.983386 (0.831978) | 0.98506 0.98717 0.97732 0.91594 0.99992 0.98512 | 0.999532 (0.994120) |
| `pad=reflect`                               | 0.999946 | 0.975772 (0.845708) | 0.87748 0.99827 0.99924 0.99709 0.99981 0.96400 | 0.999789 (0.998593) |
| `dither=1e-5`                               | 0.915996 | 0.943547 (0.786754) | 0.95097 0.96173 0.90586 0.98467 0.99945 0.94900 | 0.999829 (0.999028) |
| `pad=none`                                  | 0.999946 | 0.959419 (0.850491) | 0.87748 0.90808 0.97818 0.97943 0.99787 0.95480 | ruled out (frames)  |

**Decided.**

- **Log: `soft-76db`**, the new `frontend.LogSoft76`, log10(x + 10^−7.6 · max x) over the clip; an all-zero clip
  takes ε = 1e-10. The soft knee fits tones, bursts and noise better than the hard clamp (0.9992 to 0.9999 against
  0.975 to 0.997). `LogSoft80` (Whisper's 80 dB, soft) fits the tones a little better still (0.99992). But on speech
  it loses 1.8e-4 and lets the old worst clip fall back to 0.992779.
- **Centring: zero.** A constant reflected at the edges is still constant, so reflect makes every DC frame equal. The
  engine's DC rows are not (DC family 0.99984 against 0.87748). `none` loses on the probes and fails the corpus' frame
  rule.
- **Dither: none.** The new `frontend.Dither1e5` (σ = 1e-5, fixed seed, the config's training value) turns silence
  into normalised noise: 0.915996 against 0.999946. The corpus cannot tell (0.999829). With the two log values and
  this axis, `frontend.Variants()` enumerates 2592 front ends (F-012's 864 × 3).
- **Window: still a tie.** Symmetric against periodic Hann differs by 1e-6 on the corpus and 5e-6 on the probes, and
  the click probes, which sample the window at known offsets, do not separate them either. `Best()` keeps the
  symmetric window.

With these, `Best()` scores mean 0.999828 over the 32 clips, every clip ≥ 0.999033 (ls-clean-260-123440-0018, then
ls-clean-5639-40744-0014 0.999245); the old worst clip fleurs-fr_fr-10174413140250646001 rises to 0.999713. 7 frames
stay below 0.995 instead of 13, all quiet (peak |x| ≤ 3e-2), the worst 0.9558 (ls-clean-5639-40744-0014 frame 69)
instead of 0.6753. The safetensors weights give 0.921134 (min 0.881205). The search from the prior, with the new
values on its axes, reaches `log=soft-76db pad=zero` by itself (`just encoder-probe`, 244 s, 6 inner clips: soft-80
−0.000033, Whisper −0.000084). On those clips `pe` and the window tie, as above.

**Open.**

- The knee depth varies per clip (7.50–8.55) for a reason not found. The probes' tones and bursts prefer 8.0, speech
  7.6.
- Clicks below about 1e-5 behave as if under an absolute threshold that tones do not show. The engine's rows for a
  1e-6 click equal its rows for silence (cosine 1.000000). A 1e-5 click at δ 80 is neither silence nor scale
  invariant (0.845468 against the 0.1 click). The same click at 1e-3 is invariant (0.999942). Clicks at δ 0, whose
  side frames sit 2 decades down, already differ from 1e-1 to 1e-2 (0.984834). With the soft knee the impulse family
  scores 0.96400, under the hard clamp 0.99102. A per-frame energy gate fits some of these clicks, but no single
  threshold fits all of them, and none touches speech, whose quiet frames hold far more energy. So it stays out of
  `Best()`.

## F-021 — The decoder's cross-attention: form and place in the block (U6) (2026-10-10)

**Question.** U6: where does the gated cross-attention sit in a decoder layer, and in what form? Upstream's `Block`
has none. The checkpoint holds, per layer, `cross_norm`, `cross_attn/{q,k,v,gate,out}_proj`, `cross_attn/{q,k}_norm`,
`post_cross_norm` and the scalar `cross_gate` (F-006, `.cact` `crossNN.*`). The only oracle is the engine's transcript.

**Instrument.** `internal/decoder` builds the decoder for one `Hypothesis`: twelve typed axes with named values, the
zero value invalid, as `internal/encoder` (F-014). The harness `internal/decoder/probe_test.go` (build tag `oracle`,
`just decoder-probe`) runs `encoder.Best()` once per clip and memory tap. It turns each committed engine transcript
(`testdata/oracle/<clip>.json`) into target tokens: prompt, language token, `RefTokenizer` ids of the text, EOS. A
hypothesis scores the mean, over the 32 clips, of the mean log-probability of those tokens in one teacher-forced pass.
The top-1 accuracy, the language wins and the language token's rank are printed beside it. The engine decodes with 5
beams and W2A8 numerics, so a perfect port does not reach log-probability 0 or top-1 1. One hypothesis on all 32 clips
takes about 0.1 s on an Apple M5 Pro (15 cores).

**Is the score meaningful?** `TestProbeBaselines`: the leader scores −0.1302 (top-1 0.9768, language 32/32). Fed
the _next_ clip's encoder output, it scores −4.2232 (top-1 0.3283, language 25/32). That is a language model guessing
from the prompt. The cross-attention therefore carries the transcript, and the gap of 4.1 nats per token leaves room
to rank hypotheses.

**Search.** Four stages, each the full product of its axes around a beam of 4: the cross-attention (`cross`, `qknorm`,
`gate`, `residual`; 256 combinations), the embedding and the head (`scale`, `head`, `final`), the prompt and the target
format (`prompt`, `lead`), and the memory tap with the switches (`memory`, `xrope`, `engram`). Then coordinate descent
with escalating moves (single axes, then all 66 axis pairs, then whole stages; any change restarts from single axes)
until no level moves the leader, then a per-axis sensitivity table. From the prior it took 2.6 min.

- **From the prior** (upstream's `Block`/`Stack`/`SimpleAttentionNetwork` wherever they define something, and the
  self-attention's form for the cross-attention), no stage and no descent level changed the leader. The prior is
  `decoder.Best()`.
- **From a start wrong on all twelve axes** (`WHISTLE_PROBE_KEY`, score −8255.9, top-1 0.0010), the stages reached
  −5.0181. Their joint products ran around a cross-attention that was still wrong: `qknorm=unit gate=none`. The first
  single-axis descent round then reached `Best()` (−0.1302); the single-axis, pair and stage rounds after it left it unchanged. The run took 3.6 min.

**Winner** (`.cact` weights, Δ = the value's score − the leader's):

| axis                  | winner                                                                    | alternatives (Δ logp, top-1, language wins)                                                                                                     |
| --------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| place (`cross`)       | **between the self-attention and the MLP**, inside the mHC block function | after the MLP −0.1484 (0.919, 20/32); before the self-attention −1.0900 (0.748, 17/32); outside mHC, added to every lane −1.5188 (0.607, 20/32) |
| q/k norm (`qknorm`)   | **ZCRMSNorm per head, gain 1 + scale**                                    | none −0.2098 (0.933); no gain −0.7983 (0.783); gain = scale −2.7037 (0.409)                                                                     |
| gate (`gate`)         | **σ(gate_proj(cross_norm(h)))**, the attention input                      | σ(gate_proj(h)) of the residual −0.0863 (0.944); none −0.7254 (0.770); silu −1.4487 (0.629)                                                     |
| residual (`residual`) | **h + σ(cross_gate) · post_cross_norm(out_proj(c))**                      | tanh(cross_gate) ±0 (identical); no post norm −4.2525 (0.509); raw cross_gate −8.9044 (0.271)                                                   |
| rotation (`xrope`)    | **none**                                                                  | q at the token's position, k at the frame's, θ = 1e5: −0.3912 (0.868)                                                                           |
| memory (`memory`)     | **`enc.final_norm` of the lane mean** (= `needle_embed`)                  | the lane mean before the norm −3.6391 (0.311, 24/32)                                                                                            |

So a decoder layer is upstream's `Block` with one more residual between its two:

```
x += α · ev                                                         engram sites 3 and 7
x += σ(attn_gate)  · post_attn_norm(self_attn(norm_in(x)))          causal, taps, RoPE 1e5
x += σ(cross_gate) · post_cross_norm(cross_attn(cross_norm(x), m))  m = needle_embed, no mask, no RoPE
x += hadamard_mlp(pre_hada_norm(x))
```

all inside the layer's mHC advance (`needle.Block.Extend` with `Hooks.AfterAttn`). The cross-attention is
`MultiHeadAttention`'s shape without taps or GQA: 8 query and 8 KV heads, q/k 48 and v 64, per-head q/k ZCRMSNorm,
softmax(q̂k̂ᵀ/√48)·v, the output times σ(gate_proj(x)), then `out_proj`. The keys and values of the memory are
projected once per clip (`Decoder.Memory`). This also settles what F-015 left as "very likely": the decoder reads the
encoder output after `enc.final_norm`, i.e. exactly `needle_embed`.

**Ties.** σ(cross_gate) and tanh(cross_gate) are both 1 in float32, because every `cross_gate` is 22.0–22.8 (like the
`attn_gate` values, 19.5–22.8). The two forms give identical scores, and no transcript can tell them apart. `Best()`
keeps σ, the self-attention's form. Every other alternative costs at least 0.086 nats per token.

**Weight sources.** The same leader on the float checkpoint scores −0.1970 (top-1 0.9465, language 32/32), against
−0.1302 with the dequantised `.cact`. As for the encoder (F-014), the engine runs the quantised weights. The whole
search on the float weights (`WHISTLE_PROBE_SRC=safetensors`, 2.7 min) ends at the same leader, and every alternative
still loses. The closest are the gate from the residual (−0.0514), the cross-attention after the MLP (−0.1098,
language 20/32) and no q/k norm (−0.1277).

## F-022 — The decoder's KV cache: incremental decoding equals a full re-run (2026-10-10)

**Question.** A token-by-token decoder must reproduce the whole-sequence forward pass. Three parts of the Needle 3
block see earlier positions: the self-attention, the causal Q/K/V taps, and the engram (n-gram hashes over the last 3
tokens, values tapped at dilation 3 over 4 taps). What state does each need?

**State** (`internal/needle` `cache.go`):

- `KVCache`, per layer: the normed and rotated keys and the values of every position, plus the raw Q/K/V projections
  of the last `taps − 1 = 2` positions, which the next position's taps read. `Attention.Extend` projects the new
  rows, applies the taps over history + new rows, norms and rotates them at their absolute positions, appends them,
  and attends causally with the query offset.
- `EngramCache`, per site: the last `MaxOrder − 1 = 2` tokens for the hashes, and the raw values (`e @ value_proj`)
  of the last `(taps − 1) · dilation = 9` positions for the dilated taps. `Engram.Extend` uses the absolute position
  for `ngram_ok` (t ≥ order − 1).
- `Block.Extend` is `Block.Forward` with the cached attention and the `Hooks` (`BeforeAttn`, `AfterAttn`, `AfterMLP`)
  that insert the cross-attention (F-021). `Forward` and `Extend` share one body, so every F-009 fixture still holds.
  mHC and the MLP are per position and need no state.

**Measured.**

- `internal/needle` `TestAttentionExtendMatchesForward`, `TestEngramExtendMatchesForward` and
  `TestBlockExtendMatchesForward` (random weights, with and without taps and engram) feed the same sequence in one
  call, row by row, and in 2 + 4 + 3 pieces. Each must match `Forward` within the F-009 contract. One call matches
  exactly (max |Δ| 0). Row by row, the attention is at most 5.96e-7 off and the engram 4.8e-7, which is the T-dependent
  summation order of `MatMulTransB` (F-011).
- `TestWhistleDecoderIncremental` runs the real layer 3 with engram site 0 on 12 positions: max |Δ| 6.1e-5 at
  max |ref| 379 (2 % of the bound), cosine 1.000000000000.
- `internal/decoder` `TestIncrementalMatchesFullRun` decodes a FLEURS-de clip greedily through the caches. It then
  feeds the 38 tokens again in one pass with empty caches. Every position's 8199 logits agree within the contract
  (worst max |Δ| 6.1e-5 at max |logit| 101), and so does every argmax: each greedy choice is the full re-run's.

## F-023 — Decoder prompt, input scale and head (U7, part 1) (2026-10-10)

**Question.** What token sequence starts a transcript, and how do the embedding and the head scale? PLAN.md listed
`[bos]`, `[bos, <speech>]`, `[bos, <|lang|>]` and more as prompt candidates.

**Embedding rows.** A trained row and an untrained one differ visibly in the float checkpoint. Text pieces, `<s>` (2),
`</s>` (1) and the seven language tokens have RMS 2.5–3.5 per row. `<pad>`, `<unk>`, all 24 chat markers (ids 4–27,
including `<speech>` and `</speech>`), 153 of the 256 byte pieces and a few JSON pieces sit at 8.3, the same level as
pieces no transcript can contain. Whistle never trained `<speech>`, `</speech>` or `<pad>` as inputs. Of the specials,
only `<s>`, the language tokens and `</s>` take part.

**By score** (Δ against the leader's −0.1302, `.cact`, all 32 clips):

| axis        | winner                                                            | alternatives (Δ logp, top-1, language wins)                                                                                                                                                            |
| ----------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| prompt      | **`[<s>]`**, then `<\|lang\|>`, the text, `</s>`                  | `[<s>, <speech>]` −0.4236 (3/32); `[<s>, <speech>, </speech>]` −0.5716 (3/32); `[<pad>]` −0.6570 (4/32); `[<speech>]` −0.7252 (5/32); `[</s>]` −0.8315 (21/32); `[<speech>, </speech>]` −1.1358 (4/32) |
| first word  | **no leading `▁`** (`Jeder …`, the tokenizer has no dummy prefix) | `▁Jeder …` −0.3202 (0.943)                                                                                                                                                                             |
| input scale | **× √512** (`embed_scale`)                                        | × 1 −4.7279 (0.284)                                                                                                                                                                                    |
| head        | **tied, x · Eᵀ** over all 8199 pieces                             | x · Eᵀ / √512 −7.8698 (same top-1; the logits are 22.6× too flat)                                                                                                                                      |
| final norm  | **`stack/final_norm` of the lane mean**                           | the lane mean −37885 (0.214)                                                                                                                                                                           |
| engram      | **on** (sites 3 and 7)                                            | off −0.1471 (0.913)                                                                                                                                                                                    |

So decoding starts from `<s>` alone. The model's first prediction is the language token, and its last is `</s>`.
That is upstream's `SimpleAttentionNetwork` input (embedding × √d_model) and output (`final_norm`, `x @ embedding.T`)
unchanged. A `<speech>` marker, the one candidate with a speech-specific meaning, costs 0.42 nats per token and
breaks language detection (3/32).

## F-024 — Language detection and greedy decoding (U7, part 2) (2026-10-10)

**Question.** How does the engine pick the language, and do greedy transcripts with F-021 to F-023 reproduce it?

**Language step.** After `[<s>]`, the language token is the argmax over **all** 8199 logits on every clip: its rank
is 0 on 32 of 32, with the `.cact` and with the safetensors weights. So an argmax over the seven tokens 8192–8198
(what `Decoder.Greedy` does when no language is given) and an unrestricted argmax agree. A softmax restricted to the
seven tokens has the same argmax. A fixed language replaces the token (`Greedy(enc, frames, "de")`); decoding then
continues from it (`TestForcedLanguage`). The engine's own `language` field matched the corpus language on 32 of 32
clips (F-002).

**Greedy decoding** (`decoder.Best()`, `encoder.Best()`, language detected, stop at `</s>` or after 320 tokens;
`TestProbeGreedy`). WER is over lower-cased words without punctuation and CER over the raw characters, both against
the engine's text:

| weights     | language | exact text | stopped at `</s>` | mean WER | max WER | mean CER | max CER |
| ----------- | -------- | ---------- | ----------------- | -------- | ------- | -------- | ------- |
| `.cact`     | 32/32    | 21/32      | 32/32             | 0.0496   | 0.3636  | 0.0188   | 0.1295  |
| safetensors | 32/32    | 10/32      | 32/32             | 0.1114   | 0.6667  | 0.0543   | 0.2235  |

With `.cact` weights, 18 of 20 LibriSpeech clips have WER 0 and 16 are character-exact. The differences are
punctuation and quotes, one dropped "a", and "no chance for his body" → "no transfer of his body". The FLEURS
differences are where the engine's own text is shaky. Polish CER is 0.13: the engine writes "wyzptów" and "ekologiczowe",
for instance, which are not Polish words, and greedy decoding differs from such a beam choice. No clip loops or runs
to the 320-token limit. Teacher forcing puts the engine's token at our top-1 for 97.7 % of positions (94.6 % with
safetensors). The rest is the 5-beam search and the engine's W2A8 arithmetic, which Phase 6 and Phase 9 address.

**Fast regression.** `TestGreedyMatchesEngine` (no tag, skips without `models/`) decodes three short clips
(LibriSpeech test-other and test-clean, FLEURS de). It requires the engine's language and text exactly, and the
teacher-forced mean log-probability above floors just below the measured −0.0880 / −0.0548 / −0.0652.

## F-025 — The decoding oracle corpus: depth, keywords, silence, limits, LibriSpeech (2026-10-10)

**Purpose.** Phase 6 needs engine output beyond the 32 default transcripts: truncated ladders (6.4), keyword biasing
(6.3), where the engine returns the empty JSON (6.6), the length limits (6.5) and a LibriSpeech subset for WER (6.6).
Five `tools/oracle` subcommands record them, one `just` recipe each.

**CLI, not the library.** Every recipe runs the `needle` CLI with `--audio-word-timestamps --threads 4`, the flags of
the existing fixtures plus the thread pin. Only the CLI has `--audio-depth`: `needle_transcribe` takes language,
keywords and word timestamps, but no depth. The CLI also gives every set the same normalised JSON as
`testdata/oracle/<clip>.json` (`normaliseJSON`: timing fields dropped, keys sorted, values verbatim). Synthetic signals
are written as 32-bit float WAVs, so the CLI reads exactly the float32 samples that `pcm_sha256` names. The silence
and limits recipes also run `needle_transcribe` (libneedle3 via purego, `word_timestamps` 1) on the same samples. Its
JSON equals the CLI's on all 320 sweep signals (the run fails otherwise) and on all 19 limit signals the CLI
accepts. The two differ only at 0 samples (below). The engine prefixes CLI errors with `argv[0]`, so the tool runs
the CLI as `needle` from the temp directory that holds the WAV, which keeps the messages free of paths.

**Determinism.** 1, 2, 3, 4 and 8 threads give byte-identical JSON on all 32 clips. The default is "the device's fast
cores, at most 4" (4 here); the pin only fixes the configuration across machines. Every recipe was run twice, and the
second run left `git status` clean. The same holds for `oracle-run` and `oracle-probes` after the shared changes
(thread pin, WAV writer, `probe.Spec` fields). The LibriSpeech audio was also fetched a second time, with the first
copy moved aside: identical WAVs, identical `subset.json`.

**Layout and size.** Each set has a `manifest.json` and one `transcripts.json`. The latter is an object keyed by fixture
name whose values are the objects a per-clip `<name>.json` would hold. One file per transcript was the first layout:
843 files, 1.47 MB of content, but 3.8 MB on disk, since the file system rounds every file up to a 4 KiB block. Now:

| set (`testdata/oracle/`) | recipe                                                   | content                                                                   | files | bytes     | `du -sk` |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------- | ----- | --------- | -------- |
| `depth/`                 | `oracle-depth`                                           | 32 clips × depth 2–7                                                      | 2     | 353,800   | 348      |
| `keywords/`              | `oracle-keywords`                                        | 53 cases over 19 clips, 7 languages: keyword file + transcript with it    | 55    | 142,170   | 356      |
| `silence/`               | `oracle-silence`                                         | 320 signals: parameters, digest, empty or not, text (manifest only)       | 1     | 197,254   | 196      |
| `limits/`                | `oracle-limits`                                          | 22 signals: CLI exit and stderr, `needle_transcribe` code and error, JSON | 2     | 22,503    | 24       |
| `librispeech/`           | `oracle-librispeech` (audio: `oracle-librispeech-audio`) | 200 utterances: `subset.json` (ids, SHA-256, reference) + engine JSON     | 3     | 652,567   | 644      |
| total                    |                                                          |                                                                           | 63    | 1,368,294 | 1,568    |

**Depth.** `--audio-depth 8` is byte-identical to the default on 32/32 clips (`oracle depth` re-checks it on every run),
so 8 is the full ladder. The CLI refuses 1 ("--audio-depth takes a layer count of 2 or more, not "1"") and fails to
load the model at 9. Clips whose text equals the full-depth text: depth 2 7/32, 3 11/32, 4 11/32, 5 16/32, 6 15/32, 7
19/32. Not monotone, so 6.4 compares per depth.

**Keywords.** Cases are derived from the reference transcripts and the engine's own text. `oracle keywords` checks
each kind against both (`checkKind`) and re-checks the run without keywords byte for byte against
`testdata/oracle/<clip>.json`, which the manifest names as `without`. A keyword "hits" a text when its words occur
consecutively in it after lower-casing and splitting at non-letters (apostrophes kept).

| kind       | definition                                      | cases | text changed | all keywords hit with the file |
| ---------- | ----------------------------------------------- | ----- | ------------ | ------------------------------ |
| present    | a reference word the engine already writes      | 15    | 1            | 15                             |
| missed     | a reference word the engine's text lacks        | 16    | 11           | 9                              |
| distractor | a plausible word in neither text                | 8     | 2            | 2                              |
| phrase     | a multi-word reference phrase                   | 12    | 8            | 8 (1 already before)           |
| list       | several reference words and phrases in one file | 2     | 2            | 1                              |

Biasing is strong but bounded. "Cowley's" (lower case too), "Paare", "Adoptionsplan", "aliviar", "accomunano", "the
door" and "did it" appear. Two distractors win against the audio ("transfer" for "chance", "luchadoras"). Names far
from the acoustics do not appear ("Fisichella", "Sanct", "gubernatora", "continuümmodel"). A file changes words
outside its keywords too: "de-paare" keeps "Adduktionsplan", while the two-word list fixes both. 6.3 tunes against
this mix.

**Silence sweep.** `probe.SilenceSet` (new kinds `pink`, `step`, `speech`; `quantize16`) over two clips,
`ls-clean-1320-122617-0003` (en) and `fleurs-de_de-10308608197319017935` (de). The manifest records every parameter
exactly (linear gains and σ as float64, clip WAV digests, seeds), so the Go side regenerates each signal with
`Spec.SignalWith` and checks `pcm_sha256`. Sums of products in the generator carry explicit `float64` conversions,
so no architecture fuses them into an FMA. The Gaussian stream uses `math.Log`, `Sin` and `Cos`, as the F-019 probes
do; the signals were generated on arm64, and a different last bit elsewhere would show as a digest mismatch.
159 of 320 signals return exactly the empty JSON. Transitions (last non-empty, first empty):

| family                                                     | result                                                                                                      |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| digital silence, 160 … 480,000 samples                     | always empty                                                                                                |
| white noise −120 … 0 dBFS (10 dB steps), pink −120 … 0     | always empty: no absolute level opens the gate                                                              |
| −40 dBFS white noise, middle second (25 %) +Δ dB louder    | Δ ≤ 2.65 empty, Δ ≥ 2.70 "Thank you." (0.05 dB steps)                                                       |
| the same at −80 and −10 dBFS                               | Δ ≤ 2.6 empty, Δ ≥ 2.7: the threshold is a level ratio, not a level                                         |
| middle second Δ dB quieter                                 | Δ ≥ −3.25 empty, Δ ≤ −3.5 not                                                                               |
| +10 dB section of w samples in 64,000                      | w ≤ 480 empty, 560 … 57,600 not, ≥ 58,240 (91 %) empty: a spread between a high and a low percentile        |
| speech × gain, float, 0 … −120 dB                          | never empty; text unchanged to −100 (en) / −90 (de), hallucinated below: an absolute floor in the front end |
| speech × gain on the 16-bit grid                           | en: −87 dB "Thank you.", −87.5 empty; de: −82.5 not, −83 empty (0.5 dB steps)                               |
| speech + white noise, σ = clip RMS · 10^(−SNR/20)          | en: −9.75 dB not, −10 empty; de: −15 not, −15.25 empty (0.25 dB steps)                                      |
| speech padded with 1, 5, 10 s each side, 23 s before/after | never empty                                                                                                 |

So the gate compares levels within the clip: steady signals at any level are silent, a 2.7 dB step of a quarter of
the clip is not, and a 10 dB louder section opens it only while it covers between about 0.8 % (560 samples) and 90 %
of the clip. 6.6 fits the measure and the threshold to these rows.

**Limits** (`probe.LimitSet`; speech excerpts start 2 s into the en clip):

| input                              | CLI                                             | `needle_transcribe`                                |
| ---------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| 0 samples                          | exit 1, `needle: silence-0.wav holds no audio`  | rc 0, exactly the empty JSON                       |
| 1 sample (silence or speech) … 161 | the empty JSON                                  | the same                                           |
| 1279, 1280, 1281 samples of speech | "you"                                           | the same                                           |
| 480,000 samples                    | silence: the empty JSON; speech: its transcript | the same                                           |
| 480,001 samples (silence, speech)  | exit 1, `needle: audio limit is 30 s`           | rc −1, `needle_last_error` = `audio limit is 30 s` |

The limit is inclusive, and the length check runs before the silence gate. The CLI refuses an empty WAV; the
library treats 0 samples as silence.

**LibriSpeech subset.** `oracle librispeech-audio` streams all of `test-clean.tar.gz` (346,663,984 bytes, SHA-256
`39fde525…6c23`, pinned in `lsArchiveSHA256`; about 45 s) and keeps, per speaker, the 5 utterances of at most 480,000
samples with the smallest SHA-256 of `"go-whistle 6.1/<utterance id>"`. The rule does not depend on the tar order
(`TestSelectSubset` reverses it). 40 speakers × 5 = 200 utterances, 1.49–28.41 s (mean 7.75 s), 4167 reference words.
The WAVs go to `models/corpus/librispeech/` (git-ignored); `subset.json` holds ids, members, sample counts, WAV SHA-256
and references. `oracle librispeech` fetches the audio when a WAV is missing or differs, and a re-fetch must reproduce
the committed `subset.json`. The engine's WER on the subset is 221/4167 = 5.30 % (103 utterances without a word error),
normalised by upper-casing and punctuation removal only. Upstream's 4.31 % is over all of test-clean with a fuller
normaliser; 6.6 computes both sides with one.

## F-026 — Beam search: Whisper's rules at 5 beams (U8) (2026-10-10)

**Question.** U8: the engine decodes with 5 beams (PLAN.md reference card, step 6), but how does it rank hypotheses
of different lengths, when does it stop, and is the language token part of the beam? Greedy decoding with F-021 to
F-024 reproduces 22 of the 32 committed engine transcripts (PLAN.md Phase 5).

**Forkable state.** A beam continues its parent's prefix, so `decoder.State.Fork` shares it instead of copying.
`needle.KVCache.Fork` keeps the keys and values of the prefix in the same arrays and caps the fork's slices at their
length (`slices.Clip`). `Extend` only appends, so the fork's first append moves its rows to a new array, while the
parent may keep appending in place behind the fork's cap. The tap history of the KV cache and both histories of
`needle.EngramCache` are replaced, never written, by every `Extend`, so a fork shares them as is. The encoder memory
(`decoder.Memory`, the cross-attention keys and values) is one pointer shared by every state. `internal/search` forks
a state only when two live hypotheses still hold it; the last holder extends it in place.

- `internal/needle` `TestAttentionFork`: a prefix fed row by row (the parent has spare capacity), then the parent, a
  fork and a fork of the fork each take their own 4 rows, interleaved. Each equals `Forward` on its whole sequence
  within the F-009 contract (max |Δ| ≤ 2.4e-7). With the cap removed, the test fails. `TestEngramFork`: two branches,
  max |Δ| ≤ 2.4e-7.
- `internal/decoder` `TestForkMatchesFullRun`: the real decoder after the prompt, the language token and 5 text
  tokens; three branches take 8 different tokens each, interleaved. Every branch's logits equal a fresh state fed its
  whole sequence in one pass within the contract (worst max |Δ| 4.6e-5 at max |logit| 102), with the same argmax.

**Search.** `internal/search.Beam` keeps the 5 best live hypotheses by summed log-probability. Per step it takes each
live hypothesis' 6 best next tokens; that is enough for every admission rule below. The axes of `Options`:

- the length normalisation: the summed log p over 1, over L, over L^α, or over GNMT's ((5 + L) / 6)^α;
- what L counts: the text tokens, the text tokens with EOS, or the whole sequence with the prompt;
- the stopping rule: StopAll (finished hypotheses stay in the beam, ranked by the normalised score, until all 5 are
  finished), StopBest (the best finished one reaches the best live one), StopK (5 finished, Whisper with patience 1)
  or StopHF (the worst of the best 5 finished reaches the best live one, Hugging Face `early_stopping=False`);
- which EOS candidates finish a hypothesis: those among the step's best 5 (Hugging Face) or every one ranked above the
  last live hypothesis kept (Whisper);
- the language step: argmax, argmax with its log p in the score, or one beam per language token.

At the end the best finished hypothesis by the normalised score wins; with none finished, the best live one wins.
When the token limit ends the search before StopK has its 5, the best live hypotheses by summed log-probability join
the finished ones up to 5 candidates, as Whisper's `finalize` fills its finished sequences (PR #20 review; before, the
live ones were left out whenever one hypothesis had finished). A forced language's log-probability is not scored
under `LangArgmaxScored`/`LangBeam`, which only rank a detected language.
`Options.Adjust` is a per-step hook on each live hypothesis' log-probabilities for keyword biasing (PLAN.md 6.3); it
runs on one goroutine at a time.

**Harness.** `internal/search/probe_test.go` (build tag `oracle`, `just search-probe`) decodes the 32 clips
(`encoder.Best()`, `decoder.Best()`, `.cact` weights) with every variant of the grid: 20 normalisations (none, L, and
L^α and GNMT for α ∈ {0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 3}) × 3 lengths (NormNone ignores L) × 7 stop/admission
pairs × 3 language steps = 1218 variants. It ranks them by exact transcript equality with the engine, then by mean CER
and WER (`internal/eval`). The search runs on a memo of each prefix's 16 best next tokens, shared by all variants of a
clip: the whole grid costs 10,569 decoder positions and 3.6 s on an Apple M5 Pro. The leader is then rerun with the
real `Beam`, which gives the memo's transcript on 32 of 32 clips.

**Winner** (exact of 32, mean WER / CER against the engine's text):

| axis          | winner                                         | best alternative per value                                                                         |
| ------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| normalisation | **Σ log p / L** (= L^α at α = 1)               | GNMT α = 1.2: 31 (CER 0.0096); L^α or GNMT at α 0.6–2: 30; none and α ≤ 0.4: 29; L^3: 28           |
| L             | **text tokens**, ties with text tokens + EOS   | the whole sequence: 30                                                                             |
| stop          | **5 finished** (StopK)                         | StopAll, StopBest, StopHF: 29 each                                                                 |
| admission     | **every EOS above the last live kept**         | among the step's best 5: 30                                                                        |
| language step | **argmax**, ties with argmax-scored and beamed | (the language token is the argmax with rank 0 on 32 of 32 clips, F-024, and no variant changes it) |
| result        | **31/32 exact**, WER 0.0208, CER 0.0085        | greedy: 22/32, WER 0.0583, CER 0.0228                                                              |

That is Whisper's beam search (`BeamSearchDecoder` with patience 1 and `MaximumLikelihoodRanker` without a length
penalty) at 5 beams, with Whisper's language step. Twelve variants tie with the leader and give its transcript on all
32 clips: L with or without EOS, the three language steps, and `pow α=1`, which is the same formula. The clips cannot
separate them. `search.Default()` takes Whisper's: L without EOS and the argmax language token, which is also
`decoder.Greedy`'s.

**Beam width** (otherwise the winner): 1 beam 22/32 (= greedy), 2 27, 3 29, 4 29, **5 31**, 6 29, 8 29, 10 27. Exact
matches peak at the published width of 5. Wider beams lower the mean WER (0.0113 at 8) but match the engine less
often.

**The remaining mismatch.** `fleurs-pl_pl-10047713455444904767`: the engine writes "… wyreguber na towarów śródranych
płodzi w ją nas obolicjantów.", ours "… Burego Bernatorowi Śródranych płaci wiążąc na supolicjantów."; neither is
Polish, and greedy has a third reading. Teacher forcing the engine's text through `RefTokenizer` gives Σ log p
−22.5562, but the engine emits whatever pieces its model picks, not the canonical spelling. A beam search of width 16
over the token sequences that spell the engine's text exactly finds a 50-token spelling with Σ log p −14.2704 (score
−0.2854 with EOS over 50 text tokens), above our transcript's −15.6215 (−0.2947 over 53). So under our model the
engine's transcript ranks higher; it is a search miss, not a model gap. Each of its tokens is within our top 5 at its
step (worst rank 3). Its prefix stays in our beam until text token 43 (`▁ob` of "obolicjantów"). There it falls out
with Σ log p −13.5889, 0.1350 nats behind the fifth live hypothesis (−13.4539). A gap that small at one pruning step is
within the engine's W2A8 noise: teacher forcing puts the engine's token at our top-1 on only 97.7 % of positions
(F-024).

**Lesson for teacher forcing.** On this clip the canonical `RefTokenizer` spelling of a transcript scores 8.3 nats
below its best spelling. F-021's teacher-forced scores use the canonical spelling. That is fine for ranking decoder
hypotheses against each other, but it understates an engine transcript's likelihood wherever the engine produced a
non-canonical spelling.

**Speed** (the leader, clips one after another, a step's beams on all cores, M5 Pro): beam 61 ms per clip against
46 ms for greedy decoding, mean over the 32 clips.

**Fast regression.** `TestBeamMatchesEngine` (no tag, skips without `models/`) beam-decodes three short clips on which
greedy decoding fails: LibriSpeech test-other and test-clean, and FLEURS de. It requires the engine's language and
text exactly, and ranking scores above floors just below the measured −0.0993 / −0.1492 / −0.0878.
`TestOneBeamIsGreedy` pins one beam without normalisation to `decoder.GreedyMemory`, token for token.

**Not covered.** The language step and the EOS term of L are unresolved; the clips cannot separate them. The keyword
hook is exercised by a ban test (`TestAdjust`) only; its biasing rules are PLAN.md 6.3.

## F-027 — The public pipeline: the engine's JSON, the 30 s limit and its timing (2026-10-10)

**Question.** What exactly does the engine print for a transcript, so that `whistle.Result` can match it byte for
byte? What does it do at the limits? And how does it measure `ttft_ms` and `decode_tps`? `testdata/oracle/*.json` cannot
answer this, because `tools/oracle` normalises them and drops the timing (F-002).

**Raw output.** `needle --model whistle.cact --audio <wav> [--audio-word-timestamps]` (engine 3.2.0, `NEEDLE_TELEMETRY=0
DO_NOT_TRACK=1`) on two speech clips, a clip whose text has quotes, a Polish clip, 2 s of digital silence and the
first clip tiled to exactly 30 s. Ten outputs are kept verbatim in `whistle/testdata/needle-cli/`. They are named
`.stdout` rather than `.json`, so prettier (`treefmt.toml` formats `*.json`) leaves the bytes alone.

- One JSON object on one line, then `\n`, on stdout. No spaces.
- Field order: `text`, `language`, then `words` if present, then `ttft_ms`, `decode_tps`.
- `words` is printed **only with `--audio-word-timestamps`**, and then it is printed **for silence too**:

  | input       | without timestamps                                         | with timestamps                                                       |
  | ----------- | ---------------------------------------------------------- | --------------------------------------------------------------------- |
  | 2 s silence | `{"text":"","language":"","ttft_ms":0.0,"decode_tps":0.0}` | `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}` |

  So the "exact silence JSON" in PLAN.md and `tests/test_whistle.py` is the timestamps-on form. The upstream test
  calls `transcribe(..., word_timestamps=True)`.

- Numbers are fixed-point: `ttft_ms` and `decode_tps` with one decimal (`7.4`, `1433.6`, `0.0`), `start` and `end`
  with two (`0.40`, `29.92`), `probability` with three (`0.225`, `1.000`).
- Strings: `"` is written as `\"`. Non-ASCII is raw UTF-8 (`ü`, `ł`, `é`), not `\u` escapes. The engine never
  printed a backslash or a control character, so their form is not observed. `Result` writes `\\`, `\n`, `\r`,
  `\t`, `\b`, `\f` and `\u00XX`, and does not escape `<`, `>`, `&` or `/`.

**Limits.**

- 480,001 samples: exit 1, nothing on stdout, and `needle: audio limit is 30 s` on stderr (kept as
  `limit-480001.stderr`). 480,000 samples transcribe normally.
- A WAV without samples: exit 1, `needle: <path> holds no audio`.
- 10 samples: the silence JSON.

**In Go.**

- `Result.MarshalJSON` writes that form by hand: fixed field order, `strconv.AppendFloat(v, 'f', 1|2|3, 64)`, the
  string escaper above. `UnmarshalJSON` reads it. `words` is written iff `Words != nil`. So `Result{}` is the silence
  JSON without timestamps, and `whistle.Silence()` (`Words: []Word{}`) is the exact
  `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}`.
- `TestResultRoundTripsEngineOutput`: all ten recorded outputs unmarshal and marshal back byte-identical. That
  includes the quote, Polish, word times up to 29.92 s and probabilities of 1.000. `TestSilenceJSON` checks both
  silence forms against the recorded ones.
- `json.Marshal` on a `Result` re-escapes `<>&` in the marshaller's output. The CLI therefore calls `MarshalJSON`
  directly, and the type's doc says so.
- `ErrTooLong` is the engine's text, `audio limit is 30 s` (`TestTooLongIsTheEngineText`). `Transcribe` and `Embed`
  wrap it as `whistle: audio limit is 30 s: 480001 samples, at most 480000`.
- `TranscribeWAV` on an empty file returns `ErrNoAudio` (`whistle: <path> holds no audio`).
- A clip under 160 samples has no encoder frame (F-012). It returns `Result{}`, which matches the engine on 10
  samples.
- Until the 6.6 silence gate, real silence goes through the decoder: 2 s of zeros give `[en] "Thank you."`.

**Timing.** The C API takes PCM, so the clock starts in `Transcribe`, once the samples are in hand. WAV decoding and
resampling are not counted. Two quantities:

- `ttft_ms` runs until the decoder has chosen the first token after the language token, text or EOS. It covers the
  front end, the encoder, the cross-attention K/V projection, the prompt step that yields the language and the first
  text step.
- `decode_tps` is the number of tokens chosen after that one, EOS included, divided by the time from that token to
  the last one.

Both are measured through the `whistle` strategy seam (`clock` in `strategy.go`, unit-tested with a fake clock). The
engine's numbers, median of 3 runs per clip, all 32 clips, Apple M5 Pro, default threads (at most 4):

| fit over the 32 clips           | engine (5 beams, W2A8)       | Go (greedy, float32, 1 thread) |
| ------------------------------- | ---------------------------- | ------------------------------ |
| `ttft_ms` against clip duration | 0.66 + 1.084 ms/s, r = 0.980 | 3.30 + 13.04 ms/s, r = 0.991   |
| median `ttft_ms` / `decode_tps` | 7.5 / 1401.3                 | 85.5 / 676.9                   |
| 30 s (the tiled clip)           | 41.2 / 1175.5                | —                              |

The engine's TTFT grows linearly with the clip's length and correlates with the duration (r 0.980) more than with
the length of the text (r 0.889, which is itself correlated with duration). So the engine's TTFT includes the front
end and the encoder, as ours does. Its `decode_tps` does not depend on the text length (r −0.147), so it is a rate
over the decode steps, not total time. That matches ours.

The fits do not settle whether the engine stops the TTFT clock at the language token or at the first text token.
Its intercept, 0.66 ms, is about one decode step at its rate (0.71 ms at 1400 tok/s). Ours, 3.3 ms, is about two
steps (1.48 ms each), but each fit's intercept is only good to about ±1 ms. We keep the first text token, as PLAN.md
asks. The Go numbers are the same order of magnitude in tokens per second and 11× slower in TTFT. The float32
encoder (13 ms per second of audio against 1.1) is that gap; Phase 9 addresses it.

**Pipeline equality** (`TestTranscribeMatchesGreedy`, all 32 clips, `.cact`): `Transcribe`'s text and `language`
equal `decoder.Greedy` on the same encoder output on 32 of 32 clips. Its text equals the engine's on 22 of 32, the
Phase 5 count on the F-020 encoder. `Embed` (PLAN.md 8.2) is `encoder.Embed` behind the 30 s check. Against
`needle_embed` (`TestEmbedMatchesEngine`) its mean per-frame cosine is 0.999828, and the worst clip is 0.999033
(ls-clean-260-123440-0018), exactly F-020's numbers.

**Safetensors.** The checkpoint has no tokenizer (F-007). `Load` takes it from `whistle.cact` next to the file, or
from `WithVocabulary(path)`. If neither exists, it fails with a message that names `WithVocabulary`.

## F-028 — CQ GEMV in the rotated domain, straight from the packed `.cact` (2026-10-10)

**Question.** Can a linear layer multiply by a CQ matrix without dequantising it, and how fast is that in pure Go?
Which Whistle tensors can use it as stored?

**Maths.** `cact.CQ` dequantises row r, group g as w = H · c · n / √G, with c the addressed codebook levels, n the FP16
norm and H the ±1 Sylvester matrix (F-004). H is symmetric, so w · x_g = n · (c · x̂_g) with x̂_g = H · x_g / √G. A
product therefore rotates each 128-wide group of the activation once (FWHT-128 and the 1/√G scale) and is then a
gather-and-accumulate over the packed indices, scaled by one norm per group. Padding columns past `In` get zeros in
x, which drops them as `CQ` does.

**Code.** `cact.File.CQPacked(i)` returns a CQ record as stored: the index bitstream and the FP16 norms (both alias
the file data), the record and index widths, the group, the shape and the codebook `CQ` uses. `cq.New` builds a
`Matrix` from it for 2 and 4 bits; it keeps the index bytes and widens the norms to float32 once. `GEMV(dst, x)` and
`GEMM(dst, x, rows)` compute y = W · x. `Rows(lo, hi)` is a view of a row range, and `Row(dst, r)` dequantises one row
with `CQ`'s arithmetic, bit for bit. `internal/cq` has its own copy of the FWHT, because 9.4 will vectorise the
activation side.

**Kernels** (pure Go, no assembly):

| kernel             | per index byte                                                                                    | role                             |
| ------------------ | ------------------------------------------------------------------------------------------------- | -------------------------------- |
| `gather`           | decode each index, `cb[idx] · x̂[k]`                                                               | reference                        |
| `nibble`           | two lookups in 16-entry tables built per activation, one per index nibble (2 bits: an index pair) | GEMV, and GEMM rows left over    |
| `nibble4`          | the same tables for four activations interleaved: one index decode feeds four adjacent entries    | GEMM, four activations at a time |
| byte LUT, rejected | one lookup in a 256-entry table per index byte, built from the two nibble tables per activation   | removed                          |

The nibble kernels read eight index bytes as one little-endian word and the tables through fixed-size array pointers, so
a word costs one bounds check. On arm64 a lookup is then about three integer instructions, one load and a float add. The
byte table saves half the lookups but costs 256 entries per index byte to build and leaves L1 (128 KB for a 512-wide CQ2
row). In one 0.2 s run before its removal, it won only on `pw1` (40.4 vs 47.4 µs) and lost everywhere else, e.g. 24.2 vs
19.4 µs on `q_proj`, 151 vs 58 µs on `phi_res` and 728 vs 658 µs on the head. So `best()` is `nibble4`, which runs
`nibble` for fewer than four rows.

**Which tensors qualify.** All 152 CQ records of `whistle.cact` (145 CQ2, 7 CQ4, group 128) are stored `[out, in]`.
That is the orientation the float path multiplies by (`internal/needle` `linear` → `MatMulTransB(dst, out, x, w, …)`),
so the kernel needs no transform at run time. `Entry.Transform` maps safetensors to `.cact`; the stored bytes already
include it:

- `transpose` (143 Flax Dense kernels: q/k/v/gate/out of the self- and cross-attentions, `pw1`, `pw2`, the engram
  `key_proj`/`value_proj`, `stem.pw_1`, `stem.pw_2`, `stem.out`): `GEMV`/`GEMM` as they are.
- `lane-major` (the six mHC φ): the stored `(L·lanes, n·C)` stacks all layers. Layer l is `Rows(4l, 4l+4)` for
  φ_pre/φ_post and `Rows(16l, 16l+16)` for φ_res, the slices `mhc.go` takes (`TestWhistlePhiRows`).
- `reshape` (`embedding`, the two engram tables): the tied head is `GEMV`/`GEMM` over the 8199 embedding rows. The
  input embedding and the engram tables are row gathers, which `Row` serves without dequantising the rest.

1-bit, 3-bit and ternary records do not occur in Whistle, and `New` rejects them. The safetensors backend has no packed
form, so it keeps the float path.

**Accuracy** (`TestWhistleGEMM`): each kernel against `cact.CQ(i)` followed by `tensor.MatMulTransB`, on five
activation rows (four Gaussian, one Gaussian cubed for heavy tails), so `nibble4` runs one block of four plus one
leftover row. The worst of the five rows:

| tensor          | CQ  | shape        | `gather` 1 − cos | `nibble` 1 − cos | `nibble4` 1 − cos | max \|Δ\| `gather` / `nibble4` | max \|y\| |
| --------------- | --- | ------------ | ---------------- | ---------------- | ----------------- | ------------------------------ | --------- |
| layer00.q_proj  | CQ2 | (384, 512)   | 5.97e-14         | 3.94e-14         | 3.96e-14          | 5.91e-05 / 3.62e-05            | 153.9     |
| layer03.k_proj  | CQ2 | (96, 512)    | 7.76e-14         | 5.04e-14         | 6.22e-14          | 4.01e-05 / 4.2e-05             | 121       |
| cross05.v_proj  | CQ2 | (512, 512)   | 4.91e-14         | 2.91e-14         | 3.39e-14          | 0.000107 / 5.3e-05             | 410       |
| enc00.pw1       | CQ2 | (1024, 512)  | 5.25e-14         | 3.63e-14         | 4.07e-14          | 6.1e-05 / 4.58e-05             | 149.5     |
| enc07.out_proj  | CQ2 | (512, 512)   | 4.91e-14         | 2.96e-14         | 3.45e-14          | 0.000122 / 9.16e-05            | 247.4     |
| stem.pw_1       | CQ2 | (128, 128)   | 4.15e-14         | 1.77e-14         | 1.92e-14          | 0.000244 / 0.000122            | 708.3     |
| stem.out        | CQ2 | (512, 1280)  | 8.77e-14         | 6.95e-14         | 7.36e-14          | 6.1e-05 / 4.58e-05             | 188.7     |
| engram0.tables  | CQ2 | (73728, 128) | 1.08e-13         | 6.41e-14         | 8.19e-14          | 6.87e-05 / 4.58e-05            | 173.4     |
| embedding       | CQ4 | (8199, 512)  | 1.07e-13         | 8.08e-14         | 9.55e-14          | 0.000403 / 0.000472            | 878.8     |
| mhc_phi_pre     | CQ4 | (32, 2048)   | 1.56e-13         | 1.21e-13         | 1.72e-13          | 0.000549 / 0.000305            | 958.2     |
| enc.mhc_phi_res | CQ4 | (128, 2048)  | 1.67e-13         | 1.19e-13         | 1.24e-13          | 0.000385 / 0.000397            | 1428      |

(Re-run on 2026-10-10 for the PR #18 review: this table had held a copy of the speed table below by mistake. The test
also runs the lut kernel of F-034, whose accuracy is there.)

The Accept asks for cosine ≥ 0.99999, i.e. 1 − cos ≤ 1e-5. The worst kernel is almost eight orders of magnitude below
it: both sides compute the same product in float32, in a different order. The largest |Δ| is 5.7e-7 of max |y| (5.4e-7
for `nibble4`). `Row` equals `CQ`'s rows bit for bit. On random matrices (`TestKernels`, groups 8 to 128, every padding
case), every kernel stays within the float32 bound γ(In + log2 G + 8) · Σ_g n · Σ_k |cb| · Σ_j |x_j| / √G of a float64
reference built with the explicit Hadamard matrix.

**Speed** (`BenchmarkGEMM`, Apple M5 Pro, `-cpu 1`, mean of 3 × 0.5 s, random packed matrices of each shape). The
float baseline is `tensor.MatMulTransB` on the dequantised matrix: the NEON kernel by default, the Go loop under
`-tags purego`. The CQ kernels are pure Go in both builds.

| shape                    | rows | float NEON | float purego | `gather` | `nibble` | `nibble4` | MB float32 | MB packed |
| ------------------------ | ---- | ---------- | ------------ | -------- | -------- | --------- | ---------- | --------- |
| k_proj (96, 512) CQ2     | 1    | 1.78 µs    | 10.1 µs      | 49.8 µs  | 7.03 µs  | 7.14 µs   | 0.197      | 0.0138    |
| q_proj (384, 512) CQ2    | 1    | 7.54 µs    | 36.6 µs      | 198 µs   | 19.5 µs  | 19.9 µs   | 0.786      | 0.0553    |
| out (512, 512) CQ2       | 1    | 10.6 µs    | 46.7 µs      | 282 µs   | 26.6 µs  | 27.3 µs   | 1.049      | 0.0737    |
| pw1 (1024, 512) CQ2      | 1    | 23.6 µs    | 92.8 µs      | 562 µs   | 49.6 µs  | 49.9 µs   | 2.097      | 0.1475    |
| stem.out (512, 1280) CQ2 | 1    | 26.8 µs    | 128 µs       | 700 µs   | 64.8 µs  | 64.3 µs   | 2.621      | 0.1843    |
| phi_res (128, 2048) CQ4  | 1    | 10.3 µs    | 52.8 µs      | 283 µs   | 59.9 µs  | 60.7 µs   | 1.049      | 0.1393    |
| head (8199, 512) CQ4     | 1    | 206 µs     | 749 µs       | 4.22 ms  | 653 µs   | 656 µs    | 16.79      | 2.230     |
| q_proj (384, 512) CQ2    | 5    | 20.3 µs    | 186 µs       | 995 µs   | 96.2 µs  | 77.2 µs   | 0.786      | 0.0553    |
| head (8199, 512) CQ4     | 5    | 470 µs     | 3.72 ms      | 21.1 ms  | 3.27 ms  | 2.33 ms   | 16.79      | 2.230     |
| q_proj (384, 512) CQ2    | 375  | 1.21 ms    | 13.0 ms      | 73.9 ms  | 7.23 ms  | 6.25 ms   | 0.786      | 0.0553    |

Pure Go against pure Go, `nibble4` is 1.15× (head) to 2.4× (375 encoder frames) faster than the float loop on every
shape except the CQ4 φ_res, where it is 1.06× slower. Against the NEON float GEMM it is 2.2× (stem.out, pw1) to 5.7×
(φ_res) slower: the float kernel does four FMAs per vector instruction, while the lookups are scalar. Batching four
activations saves 20 % (q_proj, 5 rows) to 29 % (head, 5 rows) against `nibble`, because the index decode is shared.

**Memory.** "MB packed" is `Matrix.Bytes`: the index bytes, the float32 norms and the codebook. Per weight this is
2.25 bits for CQ2 and 4.25 for CQ4 (the file's 2.125 and 4.125 plus the widened norms), i.e. 14.2× and 7.5× less
than float32. Over all 152 CQ matrices (`TestWhistleAllMatrices`): 16.63 MB packed against 218.77 MB as float32
(13.2×). The index bytes alias the `.cact` data, so with an mmap-ed file only the norms (1.7 MB) are new heap.

**Consequences for 9.2.**

- `weights.Source` needs a way to hand out `cact.Packed` (for example an optional interface on the `.cact` backend), and
  the linear call sites need to call `Matrix.GEMM` when a tensor is packed. Every site multiplies by `[out, in]`
  already; φ needs `Rows`, and the embedding and engram gathers need `Row`.
- Single-threaded, the packed path is slower than today's NEON float path on every shape. It wins only on memory (13×).
  9.4 (a NEON LUT kernel, for example vector loads of the four interleaved entries of `nibble4`, or `TBL` lookups in
  8-bit tables) and 9.5 have to recover the speed.
- The sums run in a different order than the float path's. That changes outputs by about 1e-7 relative, so tests pinned
  bit for bit to the float path need tolerances or new fixtures.

## F-029 — Keyword biasing: an Aho-Corasick bonus of 2 nats per token, revoked on failure (U9) (2026-10-10)

**Question.** U9: the engine biases its decoding toward the words and phrases of a keyword file "through an
Aho-Corasick automaton" (PLAN.md reference card), but the bonus, its revocation and the spellings it matches are
closed. `testdata/oracle/keywords` holds the engine's text with and without the file for 53 cases over 19 clips (F-025).

**Automaton.** `search.NewKeywords(tok, keywords, bias)` tokenises every keyword with `RefTokenizer` in the spellings
`Bias.Forms` selects and builds an Aho-Corasick automaton over the token ids. The vocabulary has no dummy prefix, so
a word inside the text is spelled `" "+word` (`▁C ow ley ' s`) and the first word of a transcript bare (`C ow ley ' s`);
`FormSpace` and `FormBare` are these two, and `FormCapital` and `FormLower` add the first letter upper- and lower-cased.
`internal/tokenizer` needed nothing new. The state after a text is the node of its longest suffix that is a keyword
prefix, plus which of that suffix' tokens belong to a completed keyword. `Keywords.Adjust` is a `search.Adjust`: it reruns the automaton over a hypothesis' text and adds to
every next token the change of the hypothesis' bonus:

- a token that advances a match earns `Bias.Weight` (or Weight × its characters under `PerChar`), plus
  `Bias.Complete` for every keyword it completes, a keyword that ends as a suffix of the match included;
- a token that breaks a partial match falls back through the failure links to the longest suffix that is still a
  keyword prefix and loses `Bias.Revoke` × the bonus of the tokens it drops; EOS breaks every partial match;
- the tokens of a completed keyword keep their bonus;
- a bare spelling matches only at the start of the text (a `textStart` token in front of every text anchors it).

Every token outside the automaton falls back to the root and gets the same change, so `Adjust` touches the whole row
once and only the automaton's tokens individually. The bonus enters the adjusted log-probability, which ranks, prunes
and picks the hypotheses; `Result.LogProbs` stay the model's. Without keywords `search.Beam` is unchanged
(`just search-probe` still gives 31/32).

**Harness.** `internal/search/keywords_probe_test.go` (build tag `oracle`, `just keywords-probe`) decodes every case
with `search.Default()` plus the variant's `Adjust`, and ranks the variants by exact equality with the engine's text with
the keywords, then mean CER and WER. It runs on a memo per clip that keeps each prefix's 16 best tokens by the model and
the model's log p of every token any variant's automaton holds. A token outside the automaton gets the same change as
every other such token, so the best candidates after the bias are among those two sets, and the memo is exact (it fails
if fewer than 6 of the 16 lie outside the automaton). The leader is rerun with the real `Beam`: it gives the memo's
transcript on 53 of 53 cases. Three grids (`WHISTLE_PROBE_GRID`):

1. 2550 variants on `Default()`: weight 0.25 … 5 per token, or a quarter of that per character, × revocation 0, 0.5,
   1 × completion bonus 0, 1, 2 × 5 form sets × phrase split or not, and the completion bonus alone (1 … 8);
2. 576 variants around grid 1's leader: weight 1 … 8 × revocation 0.5, 1 × completion 0, 1 × 3 form sets × the bias
   as a renormalised logit bias or not × the final pick by the model's score alone or not;
3. 720 variants: weight 1.5 … 3 × revocation 0.5, 1 × completion 0, 1 × 3 form sets × final pick, on five search rules
   (Whisper's `StopK`/`AdmitAboveLive` and the F-026 runners-up), each also run without keywords.

**Winner** (exact of 53 against the engine's text with the keywords):

| axis             | winner                                                                 | best alternative per value                                                                             |
| ---------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| weight           | **2 nats per token**                                                   | 1.75, 2.25, 1.5: 44; 1, 1.25, 2.5: 42; 3, 3.5: 41; 4: 38; 5: 31; 8: 17; per character (¼ of these): 43 |
| revocation       | **all** (1)                                                            | half: 44; none: 42                                                                                     |
| completion bonus | **1**, tied by 2 (exact, WER, CER)                                     | 0: 45 at CER 0.0122 (vs 0.0120); 3 … 8 and the completion bonus alone: 34 at best                      |
| forms            | **after a space and bare, as written, capital and lower first letter** | without lower: 44; without capital: 44; space and bare only: 44; after a space only: 42                |
| phrases          | **one keyword per line**                                               | each word on its own: 43                                                                               |
| logit bias       | **no**: the bonus is a score term                                      | renormalised (the normaliser takes the bonus back where the model is sure): 29                         |
| final pick       | **by the adjusted score**                                              | by the model's score alone, the bias steering only the pruning: 41                                     |
| search rule      | **Whisper's (`Default()`)**                                            | the others: 44 with keywords and 47/53 without (Whisper's: 50/53)                                      |
| result           | **45/53**, WER 0.0366, CER 0.0120                                      | no keywords: 27/53 against the text with them                                                          |

`search.DefaultBias()` is the winner: weight 2, revocation 1, completion 1, all four forms, no split.

**Per kind** (the winner; "without" is our search without keywords against the engine's text without them):

| kind       | cases | with keywords: exact | without keywords: exact | our text changed |
| ---------- | ----- | -------------------- | ----------------------- | ---------------- |
| present    | 15    | 14                   | 14                      | 1                |
| missed     | 16    | 13                   | 15                      | 12               |
| distractor | 8     | 7                    | 8                       | 3                |
| phrase     | 12    | 10                   | 11                      | 8                |
| list       | 2     | 1                    | 2                       | 2                |
| total      | 53    | **45**               | **50**                  | 26               |

The engine's text changes on 24 cases (F-025), ours on 26. Biasing reproduces "Cowley's" from "cowley's" (the
capitalised form), "transfer" winning against the audio, both words of the German list, "aliviar", "accomunano", "the
door" and "did it", and leaves "Fisichella", "Sanct" and "continuümmodel" out, each exactly as the engine does.

**The 8 mismatches**, each teacher-forced both ways under the winner (the engine's text in its most likely spelling or
`RefTokenizer`'s, whichever ranks higher; Default's score with the bonus, per text token):

- 3 cases on `fleurs-pl_pl-10047713455444904767` (`pl-gubernatora-missed`, `pl-dziewietnastu-policjantow-phrase`,
  `pl-zgodnie-present`), the clip our search already misses without keywords (F-026). Δ −0.0001, −0.0195 (search misses)
  and a model gap of 0.2087 on a transcript that is not Polish.
- `it-montagne-distractor`: we take the distractor "montagne", the engine keeps "montagna". Model gap 0.0036 per token,
  a near-tie.
- `pl-wysp-missed`: both write "wysp"; the engine also changes "i słynące" to "micłynące" further on. Model gap 0.0125.
- `fr-cristaux-renale-declare-list`: we write "rénale" from the list, the engine keeps "régionale". Model gap 0.0247.
- `de-diese-paare-phrase`: the engine writes "Diese Paare … Adoptionsplan", although "Adoptionsplan" is not a keyword;
  ours keeps "Adduktionsplan". After "Diese Paare" our model puts "Adoptionsplan" 2.4 nats below "Adduktionsplan"
  (model gap 0.0641 per token), so no bias on the keyword reproduces it: a model gap.
- `de-paare-missed`: the mirror case. The engine writes "Paare … Adduktionsplan", ours "Paare … Adoptionsplan", although
  the engine's text ranks 0.0688 per token higher under our model and bias: a search miss. The step trace shows why:
  "Adoptionsplan" is one token shorter, so three weaker hypotheses spelling it reach EOS one step before the leader. All
  of them carry the same +7 keyword bonus, so they rank above the last live hypothesis, Whisper's rule admits them, and
  with two more EOS candidates the search has its 5 finished hypotheses before the leader ends. The HF admission rule
  (EOS only among the step's best 5) avoids it here, but loses 3 cases without keywords (grid 3), so the engine does not
  run it.

**Accept: not met.** With keywords our transcripts equal the engine's on 45 of 53 cases (84.9 %); without keywords on 50
of 53 (94.3 %), and the reference rate is 31 of 32 clips (96.9 %, F-026). Leaving out the 3 cases on the clip we
already miss without keywords: 45 of 50 (90 %). Four of the five other mismatches are near-ties or model gaps (two of
them 2.4-nat gaps on the German pair); one is a search miss our search rules cannot avoid without losing parity
elsewhere. No variant in 3846 reaches 46.

**Review fixes (2026-10-10, PR #23 review).** Three automaton bugs and a weight bug, fixed after the tuning above:

- the bonus depended on the insertion order: a node created before its prefix became a keyword kept that keyword's
  bonus in its φ (`[1 2 3]` then `[1 2]` earned 5 unit weights for `1 2 3`, the reverse order 3);
- a keyword that completed as the suffix of a longer keyword's partial match (`2 3` inside `1 2 3 …`) earned no
  completion, and a later failure took back its tokens;
- the bare spellings were not anchored: `FormBare`'s pieces could start a match anywhere in the text;
- under `PerChar` a byte piece counted the letters of `<0xNN>`, 6 characters per byte.

The bookkeeping is now per token: each node keeps the bonus of every token on its path, the state marks which of them
a completed keyword covers (outputs through the failure links included), and only the uncovered ones are revoked. On
the 53 keyword cases with `DefaultBias` the first two fixes change no text. The anchoring changes three:
`de-diese-paare-phrase` now equals the engine's text ("… Adoptionsplan …", so it was no model gap after all), and the
two Polish cases on the clip we miss without keywords (`pl-gubernatora-missed`, `pl-dziewietnastu-policjantow-phrase`)
change their text but still differ. Result: **46/53** (86.8 %) with keywords, Accept still not met. `PerChar` is not
in `DefaultBias`, so its fix changes no default result. The grids above were run before these fixes; their ranking was
not re-run. `TestKeywordOverlaps` and `TestKeywordPerCharBytes` pin the fixes.

**Accept settled (2026-10-10).** By the user's decision the 6.3 Accept now reads "at least 85 % of the keyword cases
equal, every mismatch explained by its margin, a miss shared without keywords, or a documented search miss". 46/53
meets it: 3 Polish cases share the miss without keywords, 3 are near-ties, `de-paare-missed` is the documented search
miss (a PLAN.md Phase 10 follow-up). The verdicts above are the original wording's.

**Speed.** 72 ms per case with keywords against 61 ms without (F-026), M5 Pro; the automaton is rerun over the text of
every live hypothesis at every step.

**Fast regression.** `TestKeywordsMatchEngine` (no tag, skips without `models/`): a lower-cased missed name the engine
writes capitalised, a distractor that loses and one that wins, and the German list, each with the engine's text, its
change against the run without keywords and a ranking score above floors just below the measured values.
`TestKeywordBonus` and `TestKeywordAdjust` pin the bonus rules on token ids (no model).

## F-030 — The depth ladder: upstream's sub-network, lanes by position (2026-10-10)

**Question.** The engine's `--audio-depth N` (help: "endpoint-preserving ladder depth of the speech decoder (2..full;
default full)") runs N of the 8 decoder layers. Which layers, and how do the mHC lanes, the engram sites and the final
norm behave when layers are skipped? `testdata/oracle/depth` holds the engine's transcripts at depths 2–7 for the 32
clips (F-025); depth 8 is the default.

**Upstream.** `architecture.py` defines the ladder for training nested sub-networks:

- `_ladder_layer_order(8)` = 0, 7, 3, 5, 1, 2, 4, 6: both endpoints, then the middle of the largest gap, the leftmost
  of equal gaps. `ladder_layer_indices(depth)` takes the first `depth` of it, sorted: d2 {0, 7}, d3 {0, 3, 7}, d4 {0, 3,
  5, 7}, d5 {0, 1, 3, 5, 7}, d6 {0, 1, 2, 3, 5, 7}, d7 {0, 1, 2, 3, 4, 5, 7}.
- `Stack` with `exit_depth` (`subnetwork_only`) runs a layer whose rank is below the depth with its own weights and
  passes the stream on unchanged for the others. An active layer's lane is its position among the active layers:
  `sub_positions = cumsum(active) − 1`, `sub_lane = one_hot(sub_positions mod 4)`, so `pre_off` and `post_off` follow
  the position, not the layer index. `ladder_slice`/`ladder_config` give the same network as a standalone model:
  the selected layers renumbered 0 … N−1 (hence lanes by position), the engram sites remapped to the new indices,
  `final_norm` kept.
- The engram site flags stay on the original layers 3 and 7, so a site acts only when its layer runs.

**Port.** `decoder.LadderOrder` and `decoder.LadderLayers` port the two functions. `(*Decoder).WithDepth(n)` returns a
decoder that shares the weights, runs `LadderLayers(8, n)` in order and gives the i-th of them the lane offsets of lane
i mod 4 (`needle.MHC.WithLane`, additive). Skipped layers leave the stream as it is, an engram site acts only at an
active layer, and the final norm and the head stay. The encoder always runs all 8 layers. A `Memory` serves every depth;
`WithDepth(8)` equals the decoder logit for logit (`TestWithDepth`). Greedy decoding and `search.Beam` both run on it.

**Harness.** `internal/search/depth_probe_test.go` (build tag `oracle`, `just depth-probe`) beam-decodes the 32 clips
with `search.Default()` at every depth and compares with the engine's text at that depth. Each mismatch is
teacher-forced both ways, as in F-026: the engine's text in its most likely spelling against ours, by Default's ranking
score per text token. It is a search miss when the engine's text ranks higher under our model, otherwise a model gap of
Δ.

**Variants** (exact of 32 at depths 2 / 3 / 4 / 5 / 6 / 7; measured with a temporary switch in `WithDepth`, not kept):

| mHC lane of the i-th active layer (layer l)   | d2     | d3     | d4     | d5     | d6     | d7     | d8     |
| --------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **i mod 4 (upstream's sub-network)**          | **24** | **27** | **29** | **29** | **28** | **24** | **31** |
| l mod 4 (the layer keeps its full-depth lane) | 13     | 10     | 15     | 15     | 22     | 23     | 31     |
| i mod 4, but the last layer keeps l mod 4 = 3 | 13     | 9      | 29     | 21     | 23     | 23     | 31     |
| (N − 1 − i) mod 4 (a control)                 | 5      | 6      | 8      | 13     | 3      | 5      | 4      |

Lanes by position win at every depth; at d4 the last variant coincides with it (layer 7 is the fourth active layer).
The engine's own text at depth N equals its full-depth text on only 7 / 11 / 11 / 16 / 15 / 19 of 32 clips, so the
match rates above are not inherited from the full depth.

**Result** (`search.Default()` on `WithDepth`, M5 Pro):

| depth | layers              | beam exact | WER    | CER    | greedy exact | search misses | model gaps (max Δ) | engine = its full depth | beam ms per clip |
| ----- | ------------------- | ---------- | ------ | ------ | ------------ | ------------- | ------------------ | ----------------------- | ---------------- |
| 2     | 0, 7                | 24/32      | 0.0426 | 0.0131 | 18/32        | 2             | 6 (0.0116)         | 7/32                    | 25               |
| 3     | 0, 3, 7             | 27/32      | 0.0312 | 0.0110 | 18/32        | 1             | 4 (0.0277)         | 11/32                   | 31               |
| 4     | 0, 3, 5, 7          | 29/32      | 0.0051 | 0.0015 | 19/32        | 1             | 2 (0.0022)         | 11/32                   | 37               |
| 5     | 0, 1, 3, 5, 7       | 29/32      | 0.0047 | 0.0015 | 22/32        | 0             | 3 (0.0025)         | 16/32                   | 43               |
| 6     | 0, 1, 2, 3, 5, 7    | 28/32      | 0.0065 | 0.0040 | 20/32        | 1             | 3 (0.0288)         | 15/32                   | 46               |
| 7     | 0, 1, 2, 3, 4, 5, 7 | 24/32      | 0.0254 | 0.0088 | 21/32        | 3             | 5 (0.0155)         | 19/32                   | 54               |
| 8     | all                 | 31/32      | 0.0208 | 0.0085 | 22/32        | 1             | 0                  | 32/32                   | 59               |

**The mismatches** (31 over depths 2–7) are all near-ties: 23 model gaps of at most 0.0288 nats per text token (median
0.0025), where our model ranks our transcript higher, and 8 search misses of at most 0.0196 per token. No search rule
can fix a model gap. The differences are a comma, a capital ("Free State" / "free state"), a quote, one word ("cannot"
/ "could not", "Teil und" / "Teilen"), or a syllable in the FLEURS clips the engine already garbles. A truncated decoder
is less sure of itself, so more of its choices are near-ties, and the engine's W2A8 numerics flip them; at full depth
the one mismatch has Δ 0.0093.

**Accept: not met.** The rate at full depth is 31/32 (96.9 %). At depths 2–7 our transcripts equal the engine's on 24,
27, 29, 29, 28 and 24 of 32 (75 %–90.6 %, 161 of 192 = 83.9 % overall). Every mismatch is a near-tie of at most 0.029
nats per token. The ladder itself is settled by the variant table; the remaining gap is numeric, so the W2A8 path
(Phase 9) is the place to close it.

**Accept settled (2026-10-10).** By the user's decision the 6.4 Accept now reads "every mismatch at depths 2–7 is a
near-tie of at most 0.03 nats per text token". The 31 mismatches (at most 0.0288) meet it; "not met" above is the
original wording's verdict.

**Fast regression.** `TestDepthMatchesEngine` (no tag, skips without `models/`): three short clips at depths 2 and 4
whose engine text differs from the full depth's, each reproduced exactly with a ranking score above floors just below
the measured values. `TestLadderOrder` pins the port against upstream's output for 1, 2, 3, 5, 8 and 12 layers.

## F-031 — The silence gate, beam search in `Transcribe`, and the LibriSpeech check (U10) (2026-10-10)

**Question.** U10: by which measure and threshold does the engine decide that a clip is silence and return the empty
JSON without decoding? F-025's sweep says that it compares levels within the clip. Steady noise at any level is
empty. A middle second 2.70 dB louder opens the gate at every base level. A 10 dB louder section opens it only while
it covers between 560 samples and 90 % of the clip. Then: does the pipeline with the gate and F-026's beam search
reproduce the engine (PLAN.md 6.6 **Accept**)?

### The gate

**Evidence.** The 320 sweep signals and the 22 limit signals, regenerated from their manifests by `tools/oracle/probe`
and checked against `pcm_sha256` (on darwin/arm64, and on linux/amd64 in a container, so the digests also hold on the
CI's architecture). Two of the 342 are over 30 s, which is an error before the gate (F-025). That leaves 340 verdicts,
170 empty and 170 transcribed. A limit signal takes `needle_transcribe`'s verdict, because the CLI refuses an empty
WAV that the library treats as silence.

**Harness.** `internal/silence/fit_test.go` (tag `oracle`;
`go test -tags oracle -count=1 -v -run TestFitGate ./internal/silence/`, 5 s) scores 340 measures. Each measure is a
band × a percentile pair × a percentile definition, and every one turns a clip into a spread in dB.

- Bands: the time-domain mean square of 400 samples every 160; the power spectrogram's flat FFT bins (all, or 8–112 =
  250–3500 Hz); its mel energy summed over channels 0–79, 2–79 and every range from 5, 6 or 7 to 56, 57, 58 or 59.
- Pairs: p99−p10, p99−p9, p99−p11, p98−p10, p99.5−p10, p95−p10, p99−p5, p99−p20, max−min and p90−p10.
- Definitions: nearest rank, or numpy's linear interpolation.

For each measure the harness finds the threshold with the fewest misclassified signals. When none is wrong, it
reports the interval of thresholds that separate the two classes. An earlier, wider exploration had also tried
non-overlapping frames of 160–1280 samples, frame peaks instead of energies, absolute floors of 10⁻²⁰ to 10⁻¹⁰, the
geometric mean of the mel channels, and per-channel log-mel spreads averaged (or taken as the median, max or min) over
the channels. All of them stayed at 8 or more errors.

**Leaderboard** (excerpt; errors at the best threshold, and the separating interval of the spread in dB):

| measure                                   | wrong | interval / misclassified                                       |
| ----------------------------------------- | ----- | -------------------------------------------------------------- |
| **mel 6–57, p99−p10, nearest rank**       | **0** | **(4.985, 5.011], contains 5 dB**                              |
| mel 6–56, p99−p9 rank                     | 0     | (5.039, 5.089]                                                 |
| mel 5–58, p99−p9 rank                     | 0     | (4.980, 5.030]                                                 |
| mel 6–57, p99−p9 rank                     | 0     | (5.006, 5.056]                                                 |
| mel 6–56, p99−p10 rank                    | 0     | (5.024, 5.069]                                                 |
| mel 5–57 / 5–58 / 5–59, p99−p10 rank      | 0     | (4.945, 4.979] / (4.949, 4.990] / (4.907, 4.942]               |
| mel 6–58 (centre in 250–3500 Hz), p99−p10 | 1     | step-m40-2.65                                                  |
| mel 7–57 / 7–58 / 7–59, p99−p10           | 2–3   | en-snr-m9.5, en-snr-m9.75 (and width-57600 for 7–57)           |
| any band, p99−p11                         | ≥ 1   | width-57600: the 10 % quiet share falls below p11              |
| mel 2–79, p99−p10                         | 8     | the step and SNR transitions, speech-1279                      |
| bins 8–112 (flat 250–3500 Hz), p99−p9     | 12    | the step transitions                                           |
| mel 0–79, p99−p10                         | 15    | all 7 pink noise levels, the steps                             |
| bins 0–256 (flat full band), p99−p10      | 27    | pink noise: its low frequencies make the frame level fluctuate |
| time domain 400/160, p99−p9               | 29    | pink noise, the steps                                          |

**Winner** (`internal/silence`, `whistle.Load` builds it). The measure starts from the front end's power spectrogram:
400-sample symmetric Hann windows centred every 160 samples, zero padded, `len(pcm) / 160` frames (F-012, F-020).
Per frame it sums the mel energy over channels 6 to 57 (centres 261–3307 Hz, Slaney bank = `audio.mel_filters`,
F-006) and takes 10·log10 of the sum. A frame without energy is −∞. The spread is the nearest-rank 99th percentile of
these levels minus the 10th. It is 0 without frames or when the 99th is −∞, and +∞ when only the 10th is. A clip
whose spread is at most **5 dB** is silence.

The mel-weighted band, not the flat FFT band, is what passes the steps. Flat bins weight 2–3.5 kHz far more than the
mel channels do.

Per family, at the winner (largest spread of an empty signal, smallest of a transcribed one):

| family                          | empty: largest spread | transcribed: smallest spread |
| ------------------------------- | --------------------- | ---------------------------- |
| digital silence (7 lengths)     | 0                     | —                            |
| white noise −120 … 0 dBFS       | 2.331 (white-m110)    | —                            |
| pink noise −120 … 0 dBFS        | 3.461 (pink-m60)      | —                            |
| step at −40 dBFS (+Δ and −Δ)    | 4.985 (+2.65 dB)      | 5.035 (+2.70 dB)             |
| step at −80 and −10 dBFS        | 4.935 (+2.6 dB)       | 5.035 (+2.7 dB)              |
| +10 dB section, width w         | 3.624 (w = 58,240)    | 5.051 (w = 57,600)           |
| speech + noise, en              | 4.893 (SNR −10 dB)    | 5.011 (SNR −9.75 dB)         |
| speech + noise, de              | 4.943 (SNR −15.25 dB) | 5.071 (SNR −15 dB)           |
| speech × gain (float), padded   | —                     | 43.9; +∞ with padding        |
| speech on the 16-bit grid       | 0 (en −87.5, de −83)  | +∞ (en −87, de −82.5)        |
| limits: 1–161 samples / 1279 on | 0                     | 9.455 (speech-1279)          |

Some rows are decided by rank alone, not by level:

- On the 16-bit grid, speech at −87.5 dB (en) keeps 3 non-zero samples of 100,560, and at −87 dB 13. Below about
  1 % of frames with any energy, the 99th percentile is −∞ and the clip is silence; above it, the 10th percentile is
  −∞ and the spread infinite. The de clip turns at 1 sample against 11.
- 160 and 161 samples are one frame, a spread of 0. The engine returns the empty JSON for them too; before the gate,
  `Transcribe` decoded that one encoder frame.
- With 1279 samples there are 7 frames, and the spread is their max − min, 9.455 dB.

**What the sweep does not decide.**

- The band edges, to a channel: 5 or 6 to 56 or 57 all separate the classes with p99−p9 or p99−p10. Channel 58
  (centre 3437 Hz, inside the band) costs one step signal.
- p9 or p10.
- The threshold beyond the separating interval.
- The window (periodic Hann: (4.989, 5.007]) and reflect against zero padding (reflect: (4.973, 4.999], 0 wrong).
  Frames that stay inside the clip lose 4 signals, pre-emphasis 36.

The winner is the natural pair, p99−p10 with nearest rank, on the band whose interval holds a round 5 dB. Real speech
is far from the threshold: the 32 corpus clips have spreads of at least 41.7 dB, the 200 LibriSpeech utterances at
least 36.4 dB.

**Reading.** cactuscompute.com/whistle says the front end is "band-limited to 250-3500 Hz". F-018 found no such limit
on the encoder's input. The gate is where a 250–3500 Hz band fits the engine: without it, 8 or more of the 340 verdicts
are wrong (mel 2–79), pink noise opens the gate (full band), and so do the steps (flat bins).

**Tests.**

- `TestGateMatchesEngine` (no tag, committed fixtures only, under 1 s): 340 of 340 verdicts equal to the engine, plus
  the 2 rejected by length.
- Unit tests for `Percentile`, `Spread` and `Levels`, and synthetic signals for `Silent`.
- `TestFramingIsTheEncoders`: the gate's framing is `encoder.Best()`'s.

### The pipeline

- `Transcribe` checks the length (`ErrTooLong` over 480,000 samples), then runs the gate. A silent clip returns
  `Result{}` without decoding: the engine's JSON without word timestamps,
  `{"text":"","language":"","ttft_ms":0.0,"decode_tps":0.0}`. With timestamps (Phase 7) it will return `Silence()`.
  Fewer than 160 samples are always silence.
- Decoding is `search.Beam(d, mem, search.Default())` (F-026) behind the `strategy` seam. Greedy decoding stays as a
  strategy for tests (`TestGreedyStrategy`).
- The clock keeps F-027's semantics. The search runs as a whole, so it is read through `Options.Adjust`. The search
  calls the hook for every live hypothesis of a step after the decoder ran it, and all of them hold the same number of
  text tokens. So the first call stops the TTFT clock at the first step after the language token, and the first call
  of every later step counts one token. `decode_tps` is thus a rate over the search's steps, which continue after the
  winner's EOS until 5 hypotheses have finished. An `Adjust` already in the options (keywords, 6.3) runs after the
  clock (`TestBeamClock`).
- Keywords and depths 2–7 were wired in after 6.3 and 6.4 merged: `TranscribeOptions.Keywords` runs F-029's bias
  after the clock, `Depth` runs F-030's ladder rung on the same `Memory`.

**32 fixture clips through `Transcribe`** (`TestTranscribeMatchesEngine`, all 32 without `-short`):

- 31 of 32 (96.9 %) equal the engine's `text` and `language` exactly. Each transcript also equals `search.Beam` on the
  same encoder output.
- The one miss, `fleurs-pl_pl-10047713455444904767`, is F-026's search miss. Under our model the engine's text scores
  higher, but its prefix falls out of the beam at text token 43, 0.135 nats behind the fifth live hypothesis.
- Median TTFT 91.8 ms, median 478 tok/s (Apple M5 Pro, the beams on all cores, machine shared).

**Sweep and limits through `Transcribe`** (`TestTranscribeSilenceSweep`, `TestTranscribeLimits`):

- All 159 empty sweep signals give exactly the empty JSON.
- The transcribed signals nearest the threshold in every family give text, mostly the engine's "Thank you.". The
  exception is en-snr-m9.75, where both texts are hallucinations.
- All 22 limit signals behave like the engine: 2 errors, 12 empty, 8 transcribed. The text equals the engine's on 6 of
  the 8. speech-4000 gives "Nerdy." against "I don't know.". speech-480000 (the en clip padded to 30 s) gives "and the
  manner" against "in the manner".

### LibriSpeech test-clean subset

`just eval-librispeech` (`TestEvalLibriSpeech`, run only with `WHISTLE_EVAL_LIBRISPEECH` set; it needs the model and
`just oracle-librispeech-audio`) transcribes the 200 utterances of F-025's subset with `TranscribeWAV` and checks each
WAV's SHA-256 against `subset.json`. It applies the manifest's `wer_rule` to both sides through `internal/eval`: word
edit distance over the reference's words, split at every rune other than a letter, digit or apostrophe, case folded.
`eval.Words` lower-cases where the rule upper-cases, which is the same on this ASCII text. On the engine's transcripts
the rule reproduces `manifest.json` exactly (221 errors in 4167 words, and every utterance's count).

| quantity                                 | value                                                                        |
| ---------------------------------------- | ---------------------------------------------------------------------------- |
| WER, Go pipeline                         | 223/4167 = **5.35 %**                                                        |
| WER, engine                              | 221/4167 = **5.30 %**                                                        |
| difference                               | **+0.05 percentage points** (bound ±0.3)                                     |
| text equal to the engine's               | 181/200 (90.5 %); words after the rule 189/200 (94.5 %)                      |
| language equal                           | 200/200                                                                      |
| utterances with fewer / more errors      | 3 / 4                                                                        |
| speed (M5 Pro, shared machine, two runs) | 42.5–66.0 s for 1549.4 s of audio, RTF 0.027–0.043, 212–330 ms per utterance |

**The 19 differing texts.**

- 8 differ only in punctuation or case: a comma, "?" after "girls" instead of after "roof", "who, as a gentlewoman".
- 11 differ in words, mostly names and rare words: "Margalot"/"Margulot", "cendiaries"/"Sunday Aries",
  "Mokdomots"/"Mukdonnuts", "hit"/"hid".
- To classify them, the eval teacher-forces both texts under our model. Ours uses the search's own tokens. The
  engine's uses the tokenizer's spelling, a lower bound on its best spelling (F-026). The score is Σ log p of the text
  and EOS over the text tokens.
- Our model prefers our text on 17 of the 19, by 0.0003 to 0.0251 nats per token (median 0.0034). On the other 2 the
  engine's text scores higher, by 0.0005 and 0.0007 per token: search misses, as on the Polish clip.
- Every margin is far below the W2A8 noise that keeps teacher forcing at 97.7 % top-1 agreement with the engine
  (F-024). The differences are near-ties that the two numerics resolve differently, not a missing rule.

### Result (PLAN.md 6.6 **Accept**)

- Fixtures: 31 of 32 transcripts (96.9 %) equal the engine's, at least 95 %, and the miss is explained in F-026.
- LibriSpeech test-clean subset: our WER is 5.35 % against the engine's 5.30 %. That is +0.05 percentage points, within
  0.3.
- Both parts hold for beam search without keywords at full depth. Keywords (6.3) and the depth ladder (6.4) are not
  merged yet, and `Transcribe` rejects them.

## F-032 — The stream oracle: LocalAgreement-2 with a 15 s sentence trim (U12, part 1) (2026-10-10)

**Purpose.** 8.3 (the commit rule) and 8.4 (buffers beyond 30 s) are fitted against the engine's live API. `oracle
stream` (`just oracle-stream`) records every `needle_stream_transcribe_process` and `needle_stream_transcribe_stop`
call into `testdata/oracle/stream/`. Both symbols are now bound in `tools/oracle/engine.go` (purego), so the chunks
are exact. Keywords are always off. `NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1` are set as in every recipe.

**Sets.** One stream per run; the stop of each run ends it, so the next run starts a new one.

| set        | runs | content                                                                                                   | calls |
| ---------- | ---- | --------------------------------------------------------------------------------------------------------- | ----- |
| `clips`    | 32   | every corpus clip in 1 s chunks, language detected                                                        | 192   |
| `chunks`   | 64   | every clip in 0.5 s and in 2 s chunks                                                                     | 471   |
| `language` | 4    | 1 s chunks, forced: the de and fr clips whose first pass detects English, an en clip, the de clip as en   | 28    |
| `long`     | 3    | `long-clean`: 9 test-clean clips back to back, 44.91 s                                                    | 125   |
|            |      | `long-other-gaps`: 9 test-other clips with 0.5 s of zeros between them, 42.16 s                           |       |
|            |      | `long-silence-lead`: 32.5 s of zeros, then `ls-clean-6930-75918-0000`, 36.01 s                            |       |
| `edge`     | 5    | a stop without a call, a 0-sample call, 0-sample calls after 3 s of speech, one call of 30 s, of 30 s + 1 | 6     |

The manifest names each run's clips, `lead_samples`, `gap_samples`, `chunk_samples` (or `call_samples`) and language,
plus `pcm_sha256` of the float32 samples, so the Go side rebuilds every stream from `testdata/audio` and checks it.
`calls.json` holds, per run, every call's index, sample range `[start, end)`, return code, error and normalised JSON,
then the stop's. Normalised: `pass_ms` dropped (a timing), keys sorted, values verbatim. Together 541,815 + 86,158
bytes, 620 KB on disk.

**Checks in the recipe.** Outside the edge set a run fails unless every call returns rc ≥ 0, rc equals the number of
`words`, `text` is the words joined by spaces, `received` is the samples fed so far / 16000 to a hundredth, `pass_ms`
is > 0 on every process call and 0.0 on every stop, and the CLI agrees (below).

**What a call does.** Each rule below is checked against every call of the fixtures.

1. **One pass per call.** Every process call re-transcribes the buffer, whatever the chunk length: 0.25 s, 0.5 s, 1 s
   and 2 s chunks and 0-sample calls all report `pass_ms` > 0. The header's "about one second at a time" is advice,
   not a gate.
2. **A pass is `needle_transcribe` of the buffer.** For each call that commits words, the tool searches the buffer
   start t0 such that `needle_transcribe` on `[t0, end)`, in the stream's language, ends with the call's committed
   words followed by `pending`. The committed words must match in text, in `start` and `end` (shifted by t0) and in
   `probability`. Candidates: the previous t0, then the ends of the committed words. Such a t0 exists for 470 of 470
   calls (`buffer_start`). There is no prompt from the committed text, unlike whisper_streaming's 200-character
   `init_prompt`. `language` is the pass's own: without a forced language, 10 of 32 clips at 1 s change language
   between calls (the de clip's first second gives `[en] "Eden."`), and 0 of 32 at 2 s. Forcing the clip's language
   changes only the first call (de: "Hier dann...", fr: "Merci.").
3. **Words already committed are dropped** (`dropped`: how many leading words of the pass precede the call's words).
   The pass keeps the words starting after the end of the last committed word minus 0.1 s. If the first kept word
   starts within 1 s of that end, it also drops the longest n-gram (n ≤ 5) that repeats the committed tail. This
   predicts `dropped` on 470 of 470 calls. Without the n-gram step: 438. With a margin of 0, 0.05 or 0.2 s instead of
   0.1 s: 148, 459, 469.
4. **Commit rule: LocalAgreement-2.** A call commits the longest common prefix of the previous call's `pending` words
   and this pass's remaining words, compared as exact strings. Case and punctuation count: "Transportsysteme." does
   not agree with "Transportsysteme,". This holds on 816 of 816 process calls. rc is the number of committed words,
   `text` the committed words joined by spaces, `pending` the rest of the pass.
5. **Trim: completed sentences at 15 s.** After a call that committed words, if `received` − t0 > 15 s, t0 moves to
   the end of the second-to-last sentence of all committed words. A sentence ends at a word ending in `.`, `?` or `!`
   (closing quotes allowed) when the next word starts with a capital letter or a digit. So "broke up. and all" is one
   sentence, as in a Moses splitter. This reproduces all 470 inferred buffer starts, with 6 trims, all in the long
   runs: 12.72, 23.12 and 34.32 s in `long-clean`, 8.32 and 21.52 s in `long-other-gaps`. A threshold of 14.5 s gives
   458 of 470, 15.5 s 469 of 470. Items 2 to 5 are ufal whisper_streaming's `OnlineASRProcessor` (LocalAgreement-2,
   `buffer_trimming="sentence"`, 15 s) with `needle_transcribe` as the ASR and no prompt.
6. **Stop.** No pass (`pass_ms` 0.0). It commits the last call's `pending` verbatim, with the times and probabilities
   of that pass: 103 of 103 runs, 42 of them non-empty. Then the stream is gone. A stop without a stream returns
   `{"text":"","words":[],"pending":"","language":"","received":0.00,"pass_ms":0.0}`, rc 0.
7. **`received`** is the audio fed so far, in seconds. Until the first trim it is n · (1/16000) printed to two
   decimals: 50,160 samples print as "3.14", where n / 16000 prints "3.13" (`receivedText`). After a trim it can be one
   hundredth lower: in 2 calls of `long-other-gaps`, 674,480 samples (42.155 s) print as "42.15". The engine therefore
   adds a buffer offset and the buffer length, not a sample count. The exact form is open; t0 + (n − t0) / 16000 in
   float64 gives "42.16".
8. **Beyond 30 s.** `received` keeps counting (stops at 44.91, 42.16 and 36.01). While sentences get committed, the
   trim keeps every pass under 30 s. In `long-silence-lead`, 32 s of zeros give empty passes (language "", `pass_ms`
   about 1 ms, no error). Once the speech arrives, the passes at 33 to 36 s are not `needle_transcribe` of any single
   window `[t0, received)` on a 0.04 s grid. A brute-force search finds windows that reproduce each call's text, but
   no t0 and no window length serves 33, 34, 35 and 36 s together. The stop commits "Current Court returned to its
   place amidst the tents." (offline: "Concord …"). How the engine bounds a buffer that has nothing to trim at is
   open for 8.4.
9. **Edges.** A 0-sample first call returns the stop's empty JSON. 0-sample calls after 3 s of speech run a pass on
   the unchanged buffer. It agrees with the previous one and commits all of `pending` (9 words), which flushes without
   ending the stream. A 480,000-sample call is accepted. A 480,001-sample call returns rc −1 with `needle_last_error`
   `audio limit is 30 s` and is not counted: the following stop reports `received` 0.00.

`needle.h` describes only the commit ("the words two consecutive passes agree on"), `pending`, the 30 s per call and
the stop. Items 1 to 3, 5, 7, 8 and 9, the exact-string agreement and the "no pass at stop" are not in the header.

**CLI cross-check.** `needle --audio-stream --audio-chunk S [--audio-language L] --audio-word-timestamps --threads 4`
on each run's samples (a float WAV) prints one line per call with rc > 0, then the stop's line. Nothing is printed for
the other calls. After dropping `pass_ms`, these lines equal the library's outputs on all 103 non-edge runs; the
recipe fails otherwise. In stream mode the words are printed with or without `--audio-word-timestamps`.
`--audio-chunk` refuses 31 ("takes seconds between 0 and 30").

**Final text against offline (the Phase 8 Accept).** Text: all committed words of a run joined by spaces. Offline:
`testdata/oracle/<clip>.json`, or for the long runs the clips' texts joined by spaces. Word errors are the edit
distance over space-split words. Normalised: lower case, punctuation removed.

| chunks             | runs equal | word errors / offline words | runs equal normalised | median commit delay |
| ------------------ | ---------- | --------------------------- | --------------------- | ------------------- |
| 0.5 s              | 14 / 32    | 31 / 459 (6.8 %)            | 21 / 32               | 0.76 s              |
| 1 s                | 18 / 32    | 19 / 459 (4.1 %)            | 24 / 32               | 1.48 s              |
| 2 s                | 27 / 32    | 9 / 459 (2.0 %)             | 27 / 32               | 2.64 s              |
| 1 s, forced (4)    | 2 / 4      | 10 / 59                     |                       |                     |
| long runs, 1 s (3) | 0 / 3      | 23 / 254 (9.1 %)            |                       |                     |

The commit delay is `received` at the committing call minus the word's `end`. At 1 s the engine's own stream misses
the offline text on 14 of 32 clips, for four reasons:

- case and punctuation at commit boundaries: "broke up. and all", where offline has "broke up, and all";
  "Free State" for "free state";
- a lost opening quote: `You` for `"You`;
- a word committed from a pass over less audio: "transfer" for "chance", "Billy" for "belly", "Bahre" for "Bare";
- a word dropped between two agreements: "Vanke bij zich" for "Vanke strafreziekte bij zich".

So the Accept as worded ("the final committed text equals the offline transcript") does not hold for the engine at
1 s chunks. 8.3 should compare its stream with these fixtures call by call, and with the offline text only on the 18
clips where the engine's own stream equals it.

**Determinism.** The library has no thread setting. It runs on its default, the fast cores (at most 4, here 4), and
F-025 found the transcripts identical at 1 to 8 threads. Every run also checks the library against the CLI at
`--threads 4`. `just oracle-stream` was run twice, and the second run left `git status` clean.

## F-033 — Word timestamps: Whisper's alignment on the last half of the decoder layers (U11) (2026-10-10)

**Question.** U11: with `--audio-word-timestamps` the engine prints every word with `start`, `end` (two decimals) and
`probability` (three decimals). Which words, which probability, and which attention and alignment give the times?
The site says only that the times come "from the decoder's own attention".

**Evidence.** Every engine transcript recorded with words: the 32 fixtures (459 words), the LibriSpeech subset (200,
4174 words), the 53 keyword cases (773 words) and the 192 depth runs (2750 words), 8156 words in all (F-025). All 918
times in the fixtures and every time in the other sets are multiples of 0.08 s, so the engine aligns on the 80 ms
encoder frames. Words are contiguous (an end equals the next start) except after punctuation: "Transportsysteme,"
ends at 3.68 and "was" starts at 4.24. One fixture word has zero length. That is the shape of Whisper's
`find_alignment` (jump times, `merge_punctuations`), and the fits below confirm it.

**Instrument.**

- **Capture (PLAN.md 7.1).** `decoder.State.SetCapture(fn)` reports, per `Extend` call and decoder layer that runs,
  the cross-attention probabilities `[tokens, heads, frames]` and the output gate. Off, `Extend` pays one nil check per
  layer; on, the logits are bit-identical (`TestCapture`, which also checks rows that sum to 1, a row-by-row feed
  within 1.1e-6 of a one-pass feed, a fork that does not inherit the hook, and a ladder rung that reports only its
  layers).
- **Teacher forcing.** `internal/align/probe_test.go` (build tag `oracle`, `just align-probe`) forces each engine
  text in its most likely spelling under our model (F-026's constrained beam of width 16; with the keyword bias for
  the keyword cases; on the ladder rung for the depth runs). Every spelling decodes to the engine's text. One pass
  over `<s>`, the language token, the text tokens and EOS captures the attention and the token probabilities. The
  477 items take 1.4 min on an M5 Pro and are cached in a gob file.
- **Splits.** The LibriSpeech subset is the training set. The fixtures, the keyword cases and the depth runs are held
  out.
- **Scores.** Per word: the start and the end within 80 ms (|Δ| ≤ one frame), both within 80 ms (the Accept), exact
  frames, and the mean |Δ| in frames. For the probability: equal to three decimals, within 0.01, mean |Δ|.

### Word grouping (7.2)

A token whose piece starts with `▁` begins a word; every other token continues it, byte pieces included (no piece
has an inner `▁`, and one is a lone `▁`). So the words are the transcript split at its spaces, punctuation attached.
The engine's word strings joined by spaces are its text on all 477 transcripts, and `align.Words` reproduces them on
477 of 477 (8156 words), in the most likely spelling and in `RefTokenizer`'s.

Inside a word, the time and the probability come from its tokens without Whisper's punctuation words
(`Word.Core`). Whisper's `split_tokens_on_spaces` starts a word at a token with a leading space or at a punctuation
token, and tests punctuation with Python's substring operator: `subword.strip() in string.punctuation`. So `,`, `.`,
`!`, `?`, `;` and `!"` are punctuation words, while `."`, `?"`, `,"` and `.,` are not and stay part of their word.
`merge_punctuations` then glues a mark to its neighbour but keeps the neighbour's time and probability. The engine
follows this to the quirk (`TestProbePunctuation`, keyword cases reuse fixture clips, so counts repeat):

| trailing piece        | engine's end = the end without it | = the end with it | either |
| --------------------- | --------------------------------- | ----------------- | ------ |
| `,`                   | 321                               | 11                | 12     |
| `.` (last word / not) | 251 / 38                          | 0 / 6             | 0      |
| `?`, `!`, `;`         | 15, 4, 5                          | 0                 | 0      |
| `!"`                  | 6                                 | 0                 | 0      |
| `."`, `?"`, `,"`      | 0                                 | 12, 5, 15         | 0      |

A leading `▁"` forms one Whisper word with the token after it and is timed with it (13 of 13 where it matters). The
`,` and `:` exceptions are pauses before a quote that our DTW gives to the mark and the engine to the word, numerics
rather than a rule (two `:` cases, both before `▁"`).

### Probability (7.2)

`TestProbeProbability`, all 8156 words; the source of the token probability × the aggregate × the tokens:

| candidate                                                    | exact (3 decimals) | within 0.01 | mean abs Δ |
| ------------------------------------------------------------ | ------------------ | ----------- | ---------- |
| **mean p · core tokens · softmax over the whole vocabulary** | **50.6 %**         | **91.1 %**  | **0.0036** |
| mean p · core tokens · renormalised over the text pieces     | 50.6 %             | 91.2 %      | 0.0035     |
| mean p · core tokens · keyword-adjusted, clipped at 1        | 50.3 %             | 90.5 %      | 0.0053     |
| mean p · core tokens · keyword-adjusted, renormalised        | 50.2 %             | 90.5 %      | 0.0049     |
| geometric mean (= length-normalised product) · core tokens   | 49.7 %             | 85.5 %      | 0.0069     |
| mean p · all tokens of the word                              | 45.8 %             | 83.6 %      | 0.0102     |
| product, min, max, first or last token                       | lower              |             |            |

The winner is Whisper's `word_probabilities`: the mean of the token probabilities of the word's core tokens, from
the model's softmax over the whole vocabulary. Renormalising over the text pieces changes nothing measurable (the
specials hold almost no mass). The keyword cases separate the raw model from the beam's biased score: raw 44.5 %
exact / 87.2 % within 0.01, keyword-adjusted 42.2 % / 80.7 %. Per set: LibriSpeech 59.5 % / 95.2 %, fixtures
49.5 % / 92.2 %, keywords 44.5 % / 87.1 %, depth 38.9 % / 86.0 %. The rest is the engine's W2A8 arithmetic in the
token probabilities themselves (F-024: our top-1 agrees with the engine's token on 97.7 % of positions).

### Times (7.3)

**Grid** (`TestProbeTimes`, 69,120 variants in 1.6 min, ranked on LibriSpeech by exact frames): 192 matrix configs
(gate weighting, query rows, lead rows, normalisation, median filter) × 12 head sets (all 64 heads, layers 0–3, 4–7,
6–7, each single layer) × the cost, the step pattern, the end rule, the core tokens and an offset of −1, 0 or +1
frame. Without the onset rule below. The best variant per axis value:

| axis          | winner                         | best alternatives (LibriSpeech exact start / end)                                                            |
| ------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| heads         | **layers 4–7** 94.6 % / 98.8 % | layer 7 77.7 / 82.3; layer 5 75.3 / 77.6; layers 6–7 72.4 / 77.4; all 64 71.9 / 78.8; layers 0–3 61.8 / 64.0 |
| output gate   | **unweighted**                 | weighted by the head's mean gate on the row 87.3 / 91.5                                                      |
| rows          | **the predicting queries**     | the token's own query 45.5 / 43.5                                                                            |
| lead rows     | **1 (the `<s>` row)**          | none 94.5 / 98.8 (the first word then starts at frame 0: fixtures within 80 ms 92.6 %, against 95.6 %)       |
| normalisation | **z per frame, all positions** | z per frame over the DTW rows 94.3 / 98.4; none 69.0 / 72.2; z per row 67.9 / 71.8                           |
| median filter | **none (width 1)**             | 3: 87.8 / 91.3; 5: 67.8 / 72.2; 7 (Whisper's default on 20 ms frames): 43.2 / 48.2                           |
| cost          | **−matrix**                    | −log of the probabilities 68.9 / 72.1                                                                        |
| step pattern  | **Whisper's**                  | no shared columns 71.9 / 78.8                                                                                |
| end           | **the next row's first frame** | one past the last frame of the word's row 71.9 / 78.8                                                        |
| core tokens   | **on**                         | off 94.6 / 86.8                                                                                              |
| offset        | **0**                          | +1 frame 63.7 / 69.6; −1 frame 57.0 / 59.2                                                                   |

The leader's fixtures (held out) are 92.6 % / 99.1 % exact, 95.6 % within 80 ms; every remaining start miss but 50
of 5121 is the first word's (below).

**Whole layers** (`TestProbeLayers`, all 255 subsets of whole layers × normalisation × filter × gate): layers 4–7
(every head of the last half) give 94.6 % / 98.8 % exact; the next subsets, {5, 7}, {4, 6, 7}, {0, 5, 6, 7}, reach
85 % / 89–90 %. **Fitted heads** (`TestProbeHeads`, greedy forward selection on LibriSpeech, up to 16 heads) stall at
85.3 % / 88.5 % exact (held out 83.2 % / 86.9 %): a head set fitted to the data loses to the plain last half by 9
points. That is Whisper's default when a model has no `alignment_heads`: `all_heads[n_text_layer // 2:] = True`.

**Ladder rungs** (`TestProbeDepth`, 2750 words): on depth N the engine uses the last half of the layers that run
(`LastHalf` of the capture: depth 2 → {7}, 3 → {3, 7}, 4 → {5, 7}, 5 and 6 → {3, 5, 7}, 7 → {3, 4, 5, 7}). Exact
start / end 95.0 % / 98.0 % over all depths, at least 93.8 % / 96.7 % per depth; the layers 4–7 that run give
84.7 % / 86.9 %, all layers that run 72.1 % / 76.0 %.

**The algorithm** (`align.Default`, everything but the first word identified to the frame):

1. One teacher-forced pass over `<s>`, the language token, the text tokens and EOS, the cross-attention captured.
2. Every head of the last half of the layers that ran, each z-scored per frame over all positions of the pass
   (Whisper's `std_mean` over the token axis, before the rows are cut out); no median filter; the mean over the
   heads.
3. The DTW rows are the queries that predict the language token (the `<s>` row, which absorbs leading frames), each
   text token and EOS; cost −matrix; Whisper's `dtw` (steps from the diagonal, the row above or the column before,
   ties to the column move) and its backtrace.
4. A word starts at the first frame of its first core token's row and ends at the first frame of the row after its
   last core token (Whisper's jump times); times are frames × 80 ms.

**The first word's start is not identified.** With the algorithm above every start but the first is exact on
99.0 % of words (50 misses in 5121, the fixtures, LibriSpeech and keyword sets), every end on 98.8 %, but the first
word's start only on 12 of 285 transcripts (4 %): ours is too early, mostly frame 1,
where the engine's is at the speech onset (first words mostly last 2–3 frames). Neither the attention nor a rule on
it explains the engine (`probefirst_test.go`, first-start exact of 285): the input rows as lead (11), the
normalisation over the DTW rows (14), all heads (12), heads averaged before the z-score (≤ 22), per-row transforms
(row z-score, log, ÷ max, sqrt, square, frame 0 dropped: at most 27 %), other lead rows (EOS, the mean row,
constants −0.5 … 1.5: ≤ 56), Whisper's `max_duration` truncation (end − 2 × median: 20), end − 3 frames (115). The
best fit is an energy onset (`TestProbeOnset`): the first 10 ms frame whose speech-band level (the silence gate's,
F-031) comes within X dB of the clip's 99th percentile, rounded to the 80 ms grid, as a floor under the DTW start.
X = 14 dB, fitted on LibriSpeech (`TestProbeOnsetFit`), gives first starts exact on 65.5 % / 56.2 % / 56.6 %
(LibriSpeech / fixtures / keywords) and within 80 ms on 92.5 % / 100 % / 96.2 %. `Options.Onset` carries it; it is
a fit, not the engine's rule.

**Result, teacher forced** (`TestProbeLeader`, `align.Default`):

| set                     | words | start and end within 80 ms | exact start / end | probability within 0.01 |
| ----------------------- | ----- | -------------------------- | ----------------- | ----------------------- |
| LibriSpeech (train)     | 4174  | 99.1 %                     | 97.5 % / 98.8 %   | 95.2 %                  |
| fixtures (held out)     | 459   | 99.3 %                     | 95.9 % / 99.1 %   | 92.2 %                  |
| keywords (held out)     | 773   | 99.2 %                     | 95.9 % / 99.0 %   | 87.1 %                  |
| depth (held out)        | 2750  | 99.3 %                     | 95.0 % / 98.0 %   | 86.0 %                  |
| held out together       | 3982  | 99.3 %                     | 95.3 % / 98.3 %   |                         |
| without the onset rule  | 3982  | 94.6 % (held out)          | 92.0 % / 98.3 %   |                         |
| without the core tokens | 3982  | 90.1 % (held out)          | 95.3 % / 88.2 %   |                         |

### Pipeline (7.4)

- `TranscribeOptions.WordTimestamps` fills `Result.Words` after the search: one more decoder pass over the chosen
  transcript on the same ladder rung and `Memory`, with the capture on (`timeWords` in `whistle/transcribe.go`), then
  `align.Align`. The probabilities are the pass's own, without keyword bias. A silent clip returns `Silence()`, the
  engine's `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}`. `Transcribe` computes the gate's
  levels once and uses them for the silence verdict and the onset.
- Times stay on the 80 ms grid and within the clip: the DTW's EOS row holds at least the last frame, so an end is at
  most the last frame's start, and the code caps at the last grid point within the samples.
- **The extra pass instead of capturing inside every beam.** `BenchmarkWordTimestamps` on a 6.3 s clip (M5 Pro):
  `Transcribe` 143.9 ms without timestamps, 170.4 ms with (+18 %); the pass alone 26.0 ms. The decoder's `Extend`
  over the ~30 positions is 93 % of it; the alignment the rest. Over the 32 fixtures: 125.7 against 152.6 ms per clip
  (+21 %, `TestTranscribeWordsMatchEngine`).
- `whistle transcribe --word-timestamps` prints the text and a line per word (upstream's playground format), or with
  `--json` the engine's JSON line.

**End to end, the 32 fixtures through `Transcribe`** (`TestTranscribeWordsMatchEngine`): 31 of 32 texts equal the
engine's (F-026); our words are paired with the engine's by their strings, so the Polish search miss costs its
differing words. Of the engine's 459 words, 451 pair up; start and end within 80 ms on 97.4 % of all 459 (start
97.8 %, end 97.6 %), exact 93.9 % / 97.2 %, probability within 0.01 on 90.6 % (mean |Δ| 0.0040). Every result's
words spell its text, are monotone (start ≤ end ≤ next start), lie on the grid and within [0, duration].

**JSON** (`TestTranscribeWordsJSON`): the clips behind the raw engine outputs with timestamps in
`whistle/testdata/needle-cli/` (F-027) give the engine's bytes with every number masked: `fleurs-de…`,
`ls-clean-1320-122617-0003` and `silence-2s` exactly; `tiled-30s` (the first clip tiled to 480,000 samples), whose
transcript differs from the engine's ("air" for "heir", and the engine drops a repetition), with the strings masked
and the word objects collapsed too. F-027's marshaller needed no change.

**Not covered.** The first word's start (the onset rule is a fit). The probabilities' last decimal (W2A8). Depth runs
end to end through `Transcribe` (teacher forced only). Streaming passes (F-032) reuse the same words, untested here.

## F-034 — CQ LUT kernel in NEON and AVX2: byte-plane tables and a 32-row index (9.4, `internal/cq`) (2026-10-10)

**Question.** F-028's pure-Go `nibble`/`nibble4` kernels are 2.2–5.7× slower than the NEON float GEMM, because every
table lookup is scalar. Can assembly make the packed path at least as fast as the float path, single-threaded, while
the 9.1 Accept (cosine ≥ 0.99999 against `cact.CQ` + float GEMM on the real tensors) still holds?

**Design** (`internal/cq/lut.go`, the `lut` kernel). A byte shuffle (NEON `TBL`, AVX2 `VPSHUFB`) looks up 16 bytes in a
16-entry byte table with one instruction. The F-028 nibble tables are float32 and differ per index position, so three
changes make them fit:

- **Integer tables in two byte planes.** Per activation group, each nibble table entry e (2 bits: cb[lo]·x̂[2q] +
  cb[hi]·x̂[2q+1]; 4 bits: cb[c]·x̂[q]) is rounded to v = round(e · lutQ/M) with lutQ = 8191. M = max |cb| times the
  group's largest |x̂[2q]| + |x̂[2q+1]| (2 bits) or |x̂[q]| (4 bits), so |v| ≤ 8191 (14 bits). v is stored as lo = v &
  127 and hi = v>>7 + 64, both in [0, 127]: a shuffle reads each plane, and two lookups add in one byte without
  overflow.
- **A repacked index.** `cq.New` copies the index once into blocks of 32 rows. Per block and index byte j, the 32 rows'
  bytes are adjacent (`lutLayout`), so one vector load holds byte j of 16 (NEON) or 32 (AVX2) rows, and one shuffle
  looks up a table for all of them. The norms are blocked the same way. Rows past `Out` are zero; a `Rows` view computes
  the blocks it overlaps and copies its rows out, so the mHC φ views need no repack of their own.
- **Exact integer sums.** Per row and group, each plane's lookups are summed in uint16: at most 2·groupBytes · 127, so
  groupBytes ≤ 256 (`lutMaxGroupBytes`; Whistle has 32 and 64). At the end of a group, 128·Σhi + Σlo − 16384·groupBytes
  (the hi offset) is an exact int32. It is converted to float32 and added to y times norm · step_g, with step_g =
  M/(lutQ·√G).

The rotation is the plain FWHT; its 1/√G is folded into step_g. The assembly does exactly the float32 operations of the
Go reference (`lutTables2Generic`, `lutTables4Generic`, `lutLookupGeneric`, `fwht`), in the same order. There is no FMA
(the Go side writes `float32(a*b)` to prevent fusion), round-to-nearest-even both in `FCVTNS`/`VCVTPS2DQ` and in
`math.RoundToEven`, and the FWHT stages run in `fwht`'s order. So assembly and Go reference agree bit for bit.
`asm_test.go` checks each routine and the whole GEMM, including views that start and end inside a block.

**Kernels.** Per block of 32 rows and index byte j:

| step        | NEON (`lut_arm64.s`)                                         | AVX2 (`lut_amd64.s`)                                          |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------- |
| loads       | 2 × 16 index bytes, 64 table bytes (`LD1` ×4)                | 32 index bytes, 4 × 16 table bytes (`VBROADCASTI128`)         |
| nibbles     | `AND`, `USHR` per vector: 4                                  | `VPAND`, `VPSRLW`, `VPAND`: 3                                 |
| lookups     | 8 `TBL` (2 planes × 2 nibbles × 2 vectors)                   | 4 `VPSHUFB`                                                   |
| sums        | 4 byte `ADD`, 8 `UADDW`/`UADDW2` into 8 uint16 accumulators  | 2 `VPADDB`, 4 `VPUNPCK*BW` with zero, 4 `VPADDW`              |
| group end   | `USHLL #7`, `UADDW`, `SUB`, `SCVTF`, `FMUL`, `FADD`          | the same; in-lane unpacks permute rows, `VPERM2I128` restores |
| table build | `FMUL` by element, `FCVTNS`, `UZP1`, `SHRN #7`               | `VMULPS`, `VCVTPS2DQ`, `VPACKSSDW`, `VPSRAW`, `VPACKUSWB`     |
| FWHT        | h = 1, 2, 4 in registers (`UZP1`/`UZP2`), then 8 floats/pass | h = 1, 2, 4 in registers (`VPERMILPS`, `VPERM2F128`, blends)  |

On NEON that is 24 vector instructions per 32 rows and index byte, i.e. per 128 weights of CQ2 or 64 of CQ4; on AVX2,
17 and 5 loads. Go 1.25 (`go.mod` then; 1.27 since F-035's `exp_arm64.s`) lacks the NEON float instructions and `SHRN`, so they are `WORD`-encoded, as in
`internal/tensor`. Every word was checked against Go 1.27's mnemonics with `go tool objdump`. On amd64 the kernel
needs AVX2 only (`cpu.X86.HasAVX2` at run time); without it `lutAsm` is false, and the Go reference runs.

**Accuracy** (`TestWhistleGEMM`, the 11 real tensors of F-028, five activation rows, worst row). In each build `lut`
equals its Go reference bit for bit; the float reference (`tensor.MatMulTransB`) is NEON, purego Go or AVX2+FMA, which
moves the last digit in two places:

| tensor          | CQ  | shape        | 1 − cos arm64 / purego / amd64 | max \|Δ\| / max \|y\| |
| --------------- | --- | ------------ | ------------------------------ | --------------------- |
| layer00.q_proj  | 2   | (384, 512)   | 2.05e-8 / 2.05e-8 / 2.05e-8    | 1.3e-4                |
| layer03.k_proj  | 2   | (96, 512)    | 2.28e-8 / 2.28e-8 / 2.28e-8    | 2.2e-4                |
| cross05.v_proj  | 2   | (512, 512)   | 1.98e-8 / 1.98e-8 / 1.98e-8    | 8.9e-5                |
| enc00.pw1       | 2   | (1024, 512)  | 1.48e-8 / 1.48e-8 / 1.48e-8    | 1.5e-4                |
| enc07.out_proj  | 2   | (512, 512)   | 1.24e-8 / 1.24e-8 / 1.24e-8    | 1.7e-4                |
| stem.pw_1       | 2   | (128, 128)   | 1.90e-8 / 1.90e-8 / 1.90e-8    | 1.9e-4                |
| stem.out        | 2   | (512, 1280)  | 1.51e-8 / 1.51e-8 / 1.51e-8    | 1.3e-4                |
| engram0.tables  | 2   | (73728, 128) | 1.44e-8 / 1.44e-8 / 1.44e-8    | 1.4e-4                |
| embedding       | 4   | (8199, 512)  | 7.83e-8 / 7.83e-8 / 7.83e-8    | 4.3e-4                |
| mhc_phi_pre     | 4   | (32, 2048)   | 5.89e-8 / 5.89e-8 / 5.90e-8    | 3.5e-4                |
| enc.mhc_phi_res | 4   | (128, 2048)  | 3.63e-8 / 3.63e-8 / 3.64e-8    | 1.1e-4                |

The worst 1 − cos, 7.8e-8, is 128× inside the Accept's 1e-5. That is the table rounding: the F-028 kernels stay near
1e-13. The table width was measured with the Go reference on the same tensors. With int8 tables (lutQ = 127, one plane)
the worst 1 − cos is 5.2e-4, 52× outside the Accept. With 12 bits (lutQ = 2047, which would let a byte hold four
lookups) it is 2.0e-6, a margin of 5×. 14 bits is the widest that adds two lookups per byte. `TestKernels` bounds every
element on random matrices by the float32 bound of F-028 plus the rounding bound 2·groupBytes · n · M/(2·lutQ·√G) per
group.

**Speed** (`BenchmarkGEMM`, Apple M5 Pro, `-cpu 1`, mean of 3 × 0.5 s, random packed matrices; float is
`tensor.MatMulTransB` on the dequantised matrix). The default build runs the NEON `lut`; "purego" is the same benchmark
under `-tags purego`, where `lut` is the Go reference. amd64 ran under emulation only, so it has no timings here.

| shape                    | rows | float NEON | `lut` NEON | `nibble4` (9.1) | `lut` vs float   | `lut` vs `nibble4` | MB float32 | MB `lut` |
| ------------------------ | ---- | ---------- | ---------- | --------------- | ---------------- | ------------------ | ---------- | -------- |
| k_proj (96, 512) CQ2     | 1    | 1.79 µs    | 1.15 µs    | 6.50 µs         | 1.56× faster     | 5.7×               | 0.197      | 0.0277   |
| q_proj (384, 512) CQ2    | 1    | 7.52 µs    | 2.91 µs    | 18.5 µs         | 2.58× faster     | 6.4×               | 0.786      | 0.111    |
| out (512, 512) CQ2       | 1    | 10.1 µs    | 3.68 µs    | 24.0 µs         | 2.75× faster     | 6.5×               | 1.049      | 0.148    |
| pw1 (1024, 512) CQ2      | 1    | 22.0 µs    | 6.84 µs    | 45.8 µs         | 3.21× faster     | 6.7×               | 2.097      | 0.295    |
| stem.out (512, 1280) CQ2 | 1    | 25.3 µs    | 9.18 µs    | 59.6 µs         | 2.76× faster     | 6.5×               | 2.621      | 0.369    |
| phi_res (128, 2048) CQ4  | 1    | 9.37 µs    | 8.89 µs    | 52.2 µs         | 1.05× faster     | 5.9×               | 1.049      | 0.279    |
| head (8199, 512) CQ4     | 1    | 192 µs     | 97.2 µs    | 652 µs          | 1.98× faster     | 6.7×               | 16.79      | 4.467    |
| q_proj (384, 512) CQ2    | 5    | 20.3 µs    | 14.5 µs    | 74.7 µs         | 1.40× faster     | 5.2×               | 0.786      | 0.111    |
| head (8199, 512) CQ4     | 5    | 461 µs     | 487 µs     | 2.34 ms         | **1.06× slower** | 4.8×               | 16.79      | 4.467    |
| q_proj (384, 512) CQ2    | 375  | 1.21 ms    | 1.09 ms    | 5.24 ms         | 1.11× faster     | 4.8×               | 0.786      | 0.111    |

Under purego the Go reference of `lut` is 2.2–4.9× slower than `nibble4` (q_proj 57.4 vs 19.5 µs, head 2.29 vs
0.654 ms), so `best()` is `lut` only where `lutAsm` is true, and `nibble4` elsewhere.

**Where the time goes** (one GEMV, NEON). q_proj: FWHT 0.16 µs, tables 0.36 µs, lookup 2.33 µs. k_proj: 0.16, 0.37 and
0.59 µs. phi_res: 0.62, 2.1 and 6.0 µs. head: 0.16, 0.53 and 96 µs. The lookup takes 1.5 ns per block and index
byte; at an assumed 4.5 GHz that is about 3.5 of its 24 vector instructions per cycle, close to four vector pipes. The
rest is table building, which costs O(In · 16) per activation whatever Out is. That is why the small k_proj gains least
among the CQ2 shapes, and the 2048-wide φ_res least of all.

**Batches.** The kernel runs one activation at a time (five GEMVs cost 487 µs on the head, the same as one 5-row
GEMM). Only the index load and the nibble split (4 of 24 instructions) could be shared between activations, while the
float GEMM's 4×4 blocking reuses every weight four times. So the float GEMM catches up with batches: `lut` is still
1.40× faster at 5 rows of q_proj and 1.11× at 375 rows, but 1.06× slower on the 5-beam head.

**Tried and rejected.** Unrolling the NEON loop over two index bytes in separate registers: within 1 % (head 97.1 vs
97.2 µs). The loop is limited by instruction throughput, not latency.

**Counted, not built.**

- Pairing two index bytes of one row in adjacent lanes, with a 2-register `TBL` (index + 16 for the odd lanes) and
  `UADALP` as the widening add: also 24 instructions per 32 rows and byte, because the + 16 costs an `ORR` per vector.
- 6+6-bit planes, so that a byte holds four lookups: 22 instead of 24 instructions, for 12-bit tables (5× margin,
  above).
- Not tried: a 2-activation lookup would save the shared 4 of 24 instructions per extra activation (about 7 % at 5
  rows); it needs 31 of the 32 NEON registers.

**Memory.** The blocked copy of index and norms comes on top of the packed form: 33.27 MB for all 152 CQ matrices with
the assembly against 16.63 MB in the purego build (`TestWhistleAllMatrices`), still 6.6× less than float32
(218.77 MB). `cact.Open` reads the whole 16.9 MB `.cact` into memory, so today the copy is extra heap. With an mmap-ed
file the `lut` kernel never touches the original index pages (`Row` does, for gathers), so they would not count toward
RSS. If 9.2 needs the memory back, the kernel could run on the original layout and transpose 16 rows × 16 bytes in
registers: 64 `TRN1`/`TRN2`, about 8 more instructions per 32 rows and byte (+⅓).

**Consequences for 9.2.**

- Use `Matrix.GEMV`/`GEMM` as they are. `best()` picks `lut` on arm64 and on amd64 with AVX2. The packed path is now
  faster than the float path on all seven benchmarked GEMV shapes and on the batches of q_proj. The exception is the
  5-beam head, which is 6 % slower; for it, 9.5's parallelism over row blocks applies directly, since blocks are
  independent.
- Outputs differ from the float path by up to 4.3e-4 of max |y| (1 − cos ≤ 7.8e-8), much more than F-028's 5.7e-7. Tests
  pinned to float fixtures need cosine checks, as AGENTS.md asks.
- The 64 MB RSS target: about 35 MB of CQ weights with the file read into memory, about 18 MB new heap with an mmap.

## F-035 — `internal/needle` hot loops: GEMMs, a float64 Sinkhorn and a vector exp (2026-10-10)

**Question.** Where does `whistle.Transcribe` spend its time in `internal/needle`, and which of its loops are worth
vectorising (PLAN.md 9.4, the `internal/needle` half; the CQ GEMV and the FWHT are `internal/cq`'s)?

**Profile.** An untracked driver loads `whistle.cact` and runs `Transcribe` with default options over the 32 corpus
clips, three passes under one CPU profile (Apple M5 Pro, Go 1.27.1, default GOMAXPROCS). Before: 24.3 s of samples in
12.4 s. Of these, 13.6 s are OS wait and wake-up (`pthread_cond_signal`/`_wait`, `usleep`), the goroutine hand-offs of
`search`'s parallel beam expansion (`sync.WaitGroup.Go` in `(*stepper).expand`). That is not `needle`'s code, and it
stays at 14.0 s after. The `internal/needle` rows, in seconds over the three passes:

| function                  | flat before | cum before | flat after | cum after | change                                     |
| ------------------------- | ----------- | ---------- | ---------- | --------- | ------------------------------------------ |
| `(*HadamardMLP).forward`  | 0.22        | 2.33       | 0.27       | 0.78      | through `mix`                              |
| `kronApply`               | 0.40        | 1.78       | 0          | 0.21      | 1.37 s were `tensor.Axpy`; now two GEMMs   |
| `linear`                  | 0           | 1.69       | 0          | 1.36      | `tensor.MatMulTransB`, not changed         |
| `(*Attention).output`     | 0.03        | 1.11       | 0.02       | 0.78      | gate and out projections (GEMM)            |
| `(*Attention).attend`     | 0.13        | 0.65       | 0          | 0.14      | per-key dot/axpy; now GEMMs from 8 rows    |
| `ZCRMSNorm.Apply`         | 0.16        | 0.39       | 0.07       | 0.12      | `meanSquare`                               |
| `sinkhorn`                | 0.02        | 0.36       | 0          | 0.04      | 0.34 s were `logSumExp`; now `sinkhornExp` |
| `meanSquare`              | 0.34        | 0.34       | 0.06       | 0.06      | four partial sums                          |
| `sigmoid`                 | 0           | 0.17       | 0          | 0.20      | not changed                                |
| `softmaxInPlace`          | 0.05        | 0.15       | 0          | 0.04      | `expSubSum` kernel                         |
| `RMSUnit`                 | 0.04        | 0.15       | 0.03       | 0.05      | `meanSquare`                               |
| `applyTaps`, `(*MHC).mix` | 0.09        | 0.16       | 0.07       | 0.13      | not changed                                |

`(*MHC).forward` (8.09 → 4.57 s cum) and `(*Block).run` (3.13 → 1.80 s) contain everything above, the decoder's
cross-attention hook (`runHook`, `internal/decoder`, 1.30 → 0.81 s) and the encoder's own parts. A 30 s clip (the first
clip tiled) ranks the same, except that `softmaxInPlace` grows with the square of the frames. After the first two
changes below, it was 40 ms of the encoder's 312 ms there, a third of the attention at t = 375.

**What changed, in profile order.** The order of preference was (1) `internal/tensor`'s NEON/AVX2 kernels, (2) plain Go,
(3) new assembly with a measured reason. Micro-benchmarks (`internal/needle/bench_test.go`, random weights at Whistle's
shapes, single thread, median of three):

| benchmark                         | before               | after                |          |
| --------------------------------- | -------------------- | -------------------- | -------- |
| `KronApply` (one 16×32 mix)       | 5.07 µs              | 0.68 µs              | 7.5×     |
| `HadamardMLP`, t = 1              | 19.0 µs              | 5.75 µs              | 3.3×     |
| `HadamardMLP`, t = 128            | 2.37 ms              | 0.66 ms              | 3.6×     |
| `AttentionEncoder`, t = 128       | 4.32 ms              | 2.48 ms              | 1.7×     |
| `AttentionEncoder`, t = 375       | 27.0 ms              | 10.2 ms              | 2.6×     |
| `AttentionDecodeStep` (64 cached) | 57.3 µs              | 54.4 µs              | —        |
| `Sinkhorn` (4×4, 20 iterations)   | 4.46 µs              | 0.52 µs              | 8.6×     |
| `Softmax`, n = 8 / 64 / 375       | 29.9 / 258 / 1466 ns | 11.3 / 73.1 / 435 ns | 2.6–3.5× |
| `Norms/ZCRMSNorm` (128 × 512)     | 69.8 µs              | 33.2 µs              | 2.1×     |
| `Norms/RMSUnit` (128 × 2048)      | 271 µs               | 121 µs               | 2.2×     |

1. **Kronecker products: the existing GEMM.** `kronApply` computed Y = Z · b and aᵀ · Y with one `tensor.Axpy` of 32
   per entry, 768 calls per mix. It now makes two `tensor.MatMulTransB` calls on factors transposed once at load:
   Yᵀ = bᵀ · Zᵀ (m 32, n 16, k 32), then out = aᵀ · Y (m 16, n 32, k 16). Both contract over contiguous rows, so no
   transpose is needed per row, and both fill the NEON 4×4 blocks exactly. Batching all rows of the first stage into
   one GEMM would leave the second stage contracting over a strided axis, so it would need a transpose per row
   anyway. With the GEMM, a dedicated Kronecker kernel has no target left: `kronApply` is 1 % of the samples after.
2. **Attention: the existing GEMM.** From `attendGEMMRows` = 8 query rows on, `attend` gathers each KV head's keys
   (keys × 48) and transposed values (64 × keys) once, and computes each query head's logits as Q · Kᵀ and its output
   as P · V. The output is written straight into the head's columns, with `ldc` = heads · v. A masked key's weight is
   set to 0, so it adds an exact zero, and a causal row stays independent of later rows bit for bit
   (`TestAttentionCausalBatched`). Decoder steps (t = 1) keep the per-key loop: there the gather would copy the whole
   cache every step, as much work as the attention itself. `TestAttendGEMMMatchesRows` compares both paths at
   Whistle's geometry for 1–130 rows, unmasked and causal with offsets: outputs within 1e-5 and weights within 1e-6,
   with the same zeros.
3. **Sinkhorn: plain Go.** Subtracting a row's logsumexp equals dividing the row of exp(logits) by its sum.
   `sinkhornExp` therefore runs upstream's 20 iterations on exp(logits − max) in float64. That is 16 exps instead of 640
   exps and 160 logs, and more accurate than the float32 log domain. Range guard: every entry starts in [e^−spread, 1],
   never exceeds 1, and a normalisation divides it by at most n. So if spread + 2 · iters · ln n ≤ 700, no entry leaves
   float64's normal range and no sum is 0. Outside that bound (spread above about 644 for n = 4) and for non-finite
   logits, `sinkhornLog`, the previous code, runs instead. `TestSinkhornExpMatchesLog`: within 1e-6 of the log domain
   and 2e-7 of the float64 transcription for n = 1–5 and 8, also at a spread of 640. The fixture's h_res is reproduced
   as before, 1.19e-7 max abs.
4. **Norms: plain Go.** `meanSquare`'s float64 sum was a single chain of dependent adds. Four partial sums give the
   same float32 result on all 600 rows of `TestMeanSquareOrder` (widths 1 … 2048).
5. **Softmax: new assembly.** The measured reason: after 1 and 2, the float64 `math.Exp` in `softmaxInPlace` was a
   third of the encoder attention at t = 375 and 40 ms of a 30 s clip's TTFT. `expSubSum(s, m)` sets s[i] =
   exp(s[i] − m) and returns the sum. `exp_arm64.s` (NEON, 4 lanes) and `exp_amd64.s` (AVX2 + FMA, 8 lanes, chosen at
   run time via `golang.org/x/sys/cpu`) compute Cephes' expf:
   - range reduction k = round(x · log2e), r = x − k · ln2hi − k · ln2lo;
   - the degree-6 polynomial for e^r;
   - 2^k added to the exponent bits.

   Inputs below −87.33 are clamped, and their results flushed to 0. There, `math.Exp` gives at most 1.2e-38, and
   the integer exponent add would leave the normal range. NaN propagates. The float64 loop is the `purego` fallback
   and handles the tails. `TestExpSubSumMatchesGeneric` (lengths 0–69, 127–129, 375, 1000, 4099 at offsets 0–3,
   logits spread up to 100, ±Inf, NaN) and a 65,536-point sweep of [−90, 0] give at most 0.99 ulp from
   `math.Exp`. The kernel takes 150 ns for 375 values against 1186 ns for the fallback.

   Unlike the other NEON kernels (F-034, F-041), `exp_arm64.s` uses Go 1.27's vector float mnemonics (`VFSUB`,
   `VFCMGT`, `VFRINTN`, …) rather than `WORD` encodings, which Go 1.25 and 1.26 do not assemble. CI ran on amd64 only,
   so this showed first when the `v0.1.0` release workflow cross-compiled darwin/arm64 and linux/arm64 (2026-10-12).
   `go.mod` now requires Go 1.27, and `just check-cross` (in `just ci` and CI's `test-unit`) builds every package for
   linux/arm64, darwin/arm64 and windows/amd64.

**Not changed, and why.**

- `linear` (`tensor.MatMulTransB`) is the largest remaining item at 1.36 s. It is `internal/tensor`'s kernel, and a
  faster one would replace it rather than add to it. At t = 1 the decoder's projections also stream float32 weights, so
  9.2 (packed CQ weights) is the lever there.
- `sigmoid` (0.20 s, float64 `math.Exp` per value in HadamardMLP and the attention gate) is 1 % of the samples. A
  vector sigmoid could reuse the expf core, but the gain is too small for another kernel now. The encoder's own
  sigmoid (0.35 s) is `internal/encoder`'s.
- The softmax's max and division loops (about 0.3 ns per value) and `applyTaps`/`(*MHC).mix` (0.13 s together) were
  left as is.
- The OS wait time from `search`'s goroutines, which is half of all samples, belongs to 9.5.

**Parity.** Every `internal/needle` fixture stays inside the F-009 contract. The largest share of a bound moves from
0.21 to 0.20 (`attention_decoder.out`); others move by up to 0.02 in either direction (`block.h` 0.18 → 0.20,
`hadamard_mlp.out` 0.13 → 0.15, `mhc` layer 0 0.14 → 0.11), and `probs` stays at 1.3e-6. The encoder corpus cosines
(F-020) change in the sixth decimal (worst frame 0.992554 → 0.992552). The decoder's incremental-vs-full test stays at
5.3e-5 worst. `TestTranscribeMatchesEngine` gives 31 / 32, and all 32 transcripts are byte-identical to the old code's
in six alternating driver runs. No tolerance moved.

**End to end.** The same machine, alternating old and new driver binaries (three corpus passes each) while the
process list showed no other test binary:

| `Transcribe`, 32 clips                | before              | after               |
| ------------------------------------- | ------------------- | ------------------- |
| median TTFT                           | 81.3–82.4 ms        | 55.5–57.1 ms        |
| median decode                         | 554–570 tok/s       | 556–621 tok/s       |
| wall time, 32 clips                   | 4.05–4.12 s         | 3.19–3.49 s         |
| 30 s tiled clip, TTFT (6 runs)        | 511–524 ms          | 279–283 ms          |
| `TestTranscribeMatchesEngine` (1 run) | 82.5 ms / 561 tok/s | 57.8 ms / 609 tok/s |

The engine does 7.5 ms and about 1400 tok/s on this machine (F-027). Decode barely moves because a step is one row:
its time is in the float32 GEMVs and in `search`'s goroutine hand-offs, not in `needle`'s loops.

**amd64.** This Mac has no Rosetta. The linux/amd64 test binaries ran under Docker Desktop's emulation, which reports
`avx2` and `fma`, with the checkout mounted at its own path so the fixtures and models resolve. All 53 `internal/needle`
tests pass there, the model-gated ones included, so the AVX2 kernel ran (worst 0.99 ulp in the sweep). `internal/tensor`
passes too, and `GOARCH=amd64 go vet` (asmdecl) is clean. Emulated timings mean nothing and are not reported.

## F-036 — `Stream`: the engine's stream in Go, call by call (U12, part 2; PLAN.md 8.3, 8.4) (2026-10-10)

**Question.** F-032 read the engine's stream off `testdata/oracle/stream/`: whisper_streaming's `OnlineASRProcessor`
with `needle_transcribe` as the ASR. Does a Go port of exactly that, on our `Transcribe`, reproduce the 930 recorded
calls? What is left open in F-032 (the arithmetic behind `received`, the repeat rule's n-gram order, buffers over
30 s with nothing to trim), and does the Phase 8 Accept hold as written?

**API.** `(*Model).NewStream(StreamOptions{Language, Keywords}) (*Stream, error)`, `(*Stream).Process(pcm)
(StreamResult, error)`, `(*Stream).Stop() StreamResult`. `StreamResult` holds `Text`, `Words` (never nil), `Pending`,
`Language`, `Received` (seconds, float64) and `Pass`; its `MarshalJSON` writes the engine's object in the engine's
order, `{"text","words","pending","language","received","pass_ms"}`, with `received` to two decimals, `pass_ms` to one
and the words through `Result`'s marshaller (now `appendWords`, F-027). A `Stream` belongs to one goroutine; the `Model`
stays safe for concurrent use, so several streams and `Transcribe` calls can share it.

### What a call does (`whistle/stream.go`)

Each rule is F-032's item, implemented as whisper_streaming writes it:

1. **One pass per call** over the whole buffer `[t0, received)`: `transcribe` with word timestamps always on, the
   stream's forced language or detection per pass, its keywords, the silence gate. Word times are shifted by t0.
2. **Dropping committed words** (`fresh`, `HypothesisBuffer.insert`): keep the words that start after the last
   committed end − 0.1 s; when the first kept word starts within 1 s of that end, drop the shortest n-gram (n ≤ 5)
   that repeats the tail of the committed words still in the buffer. F-032 says "longest". Both orders give the same
   result on all 108 runs (the full replay was run once with each), so the code keeps the source's shortest-first.
3. **LocalAgreement-2** (`agree`): commit the longest common prefix of the previous call's pending words and the
   kept words, exact strings. The commit carries this pass's times and probabilities; the rest becomes `Pending`.
4. **Sentence trim** (`trimSentence`): after a call that committed words, if the buffer holds more than 15 s, cut
   at the end of the second-to-last sentence of the committed words. A sentence ends at a word ending in `.`, `?` or
   `!` (closing quotes allowed) before a word starting with an upper-case letter or a digit (opening quotes allowed).
   Only words still in the buffer are scanned; a boundary at or before t0 would not move it, so this equals scanning
   all committed words. Committed words ending at the cut leave the buffer (`chunk_at`).
5. **Stop** commits `Pending` verbatim without a pass (`pass_ms` 0.0) and ends the stream; the next `Process` starts
   a new one. A `Stop` without a stream, and a 0-sample first call, return
   `{"text":"","words":[],"pending":"","language":"","received":0.00,"pass_ms":0.0}`. A 0-sample later call runs a
   pass on the unchanged buffer. A call over 480,000 samples fails with `ErrTooLong` ("audio limit is 30 s") and
   feeds nothing.

### `received` (F-032 item 7)

Before any trim the engine prints n · (1/16000); after a trim in `long-other-gaps` it prints 674,480 samples as
"42.15" where every closed form in t0 and n prints "42.16". Plain sums of the chunks in float32 or float64 miss 12
to 56 of the 929 values. A grid of 64 variants of the trim's arithmetic (word time in float32 or float64, offset in
float32 or float64, the cut rounded or truncated, the length divided or multiplied, the offset from the absolute word
time or summed) has one variant that prints all 929 `received` values. The engine holds a word time as
`float32(frames) · 0.08f` after the pass' buffer offset, the offset as a float64 sum of those, the cut rounded to the
sample. It prints `received` as offset + (buffer samples) · (1/16000) in float64. `streamWord.end` carries that float
next to the exact `Word`, and the trim moves the offset to it. Only one recorded value (two calls) separates this
variant from the next best, so it is the simplest fit, not proven.

### Parity: 108 runs, 930 calls (`just stream-parity`)

`TestStreamMatchesEngineAll` (build tag `oracle`) rebuilds every run from `manifest.json` and `testdata/audio`
(checking `pcm_sha256`), feeds the recorded chunks into a fresh `Stream` and compares every call with `calls.json`.
A call's **decision** is rc, the error, `text`, `pending`, `language` and `received`: everything but the word times
and probabilities. **JSON equal** is the whole object byte for byte, `pass_ms` dropped. The words are compared where
`text` is equal.

| set      | runs, every decision equal | calls, decision equal  | JSON equal | final text = engine stream | buffer starts |
| -------- | -------------------------- | ---------------------- | ---------- | -------------------------- | ------------- |
| clips    | 22 / 32                    | 193 / 224              | 92         | 28 / 32                    | 125 / 125     |
| chunks   | 40 / 64                    | 454 / 535              | 257        | 60 / 64                    | 267 / 267     |
| language | 2 / 4                      | 21 / 32                | 12         | 3 / 4                      | 15 / 15       |
| long     | 0 / 3                      | 115 / 128              | 62         | 2 / 3                      | 58 / 63       |
| edge     | 5 / 5                      | 11 / 11                | 10         | 5 / 5                      | 1 / 1         |
| **all**  | **69 / 108**               | **794 / 930 (85.4 %)** | **433**    | **98 / 108**               | **466 / 471** |

By field, over all 930 calls: rc 850, `text` 849, `pending` 826, `language` 930, `received` 930. "Buffer starts" is
F-032's inferred t0 against ours on the calls that have one. The five that differ are `long-clean` calls 40 to 44,
after that run's first divergence (call 17), where other commits move the trim. Committed words compared: 1465.
Start exact 1403 (95.8 %), end exact 1447 (98.8 %), both within 80 ms 1448 (98.8 %), probability within 0.01 1369
(93.4 %), all three exact 716 (48.9 %), which is F-033's timestamp and probability parity. So byte-equal JSON is
mostly a matter of the third decimal of a probability.

**On the packed path (9.2, F-037).** The numbers above were measured on the float path. Replayed on `main` after
9.2 made the packed path `Load`'s default: decision equal on 797 of 930 calls (clips 194/224, chunks 456/535, the
other sets unchanged), every decision of 69 of 108 runs, JSON equal 433, final text equal to the engine's stream 98
of 108; `TestStreamAccept` unchanged (18/32, normalised 24/32).

**Why calls differ** (`TestStreamDivergences`). Each run's first call whose decision differs, replayed to that call:
39 runs. In 38 of them our pass spells other words than the engine's pass. In one (`ls-other-4198-12281-0004.c0.5`
call 5) the pass spells the same words, but our time for `James."` puts it before the last committed end − 0.1 s, so
it is dropped as committed. No first divergence comes from the bookkeeping: with the same pass, every call decides as
the engine does. For the 38 pass differences, the margin is our beam's score (Σ log p over the text tokens / their
number) minus the engine's pass text teacher-forced in its canonical spelling, on our window:

- 32 of 38 within ±0.07 nats per token, the median +0.0058: near ties. The same handful of short passes recurs in
  the 0.5 s, 1 s and 2 s sets: "Jan Carlo fitait là." / "fit ses là." (+0.0018), "gli" / "vi accumulano" (+0.0060),
  "Vanke strafrecht ziet erbij." / "strafreziekte bij zich." (+0.0074), "Diese Paris." / "Pare." (+0.0227).
- Four below −0.1 (−0.32, −0.23 twice, −0.12): passes over the first second or two that the engine ends without the
  final mark or with "..." ("Here are some" against our "Here are some.", "These are"). The score leaves EOS out, so
  it does not measure where a hypothesis stops; these are ties on the end, not on the words.
- +0.159: `fleurs-pl_pl-10047713455444904767.c2`, the clip whose offline transcript our search misses (F-026).
- +0.343: `long-silence-lead` at 34 s (below).

No rule was tuned on these. The fast regression `TestStreamMatchesEngine` (no tag) replays 11 runs (three clips at
1 s, one at 2 s, a forced language, `long-silence-lead` and the five edge runs) in about 6 s. It requires the edge
runs exactly, every `received` and `language`, and floors just below the measured 7 / 11 runs, 71 / 83 decisions,
56 / 56 words within 80 ms and 11 / 11 final texts.

### The Accept as written (Phase 8)

`TestStreamAccept`: the 32 clips in 1 s chunks, the final committed text (every call's text and the stop's, joined)
against the offline transcript; normalised = `eval.Words` (lower case, punctuation dropped).

| stream       | offline text         | equal   | normalised | word errors      |
| ------------ | -------------------- | ------- | ---------- | ---------------- |
| ours         | the engine's fixture | 18 / 32 | 24 / 32    | 25 / 459 (5.4 %) |
| ours         | our `Transcribe`     | 18 / 32 | 24 / 32    | 22 / 457 (4.8 %) |
| the engine's | the engine's fixture | 18 / 32 | 24 / 32    | 19 / 459 (4.1 %) |

The Accept's text half does not hold, for us or for the engine: a stream commits from passes over less audio than
the offline pass, F-032's four reasons. Each stream misses 14 clips; 13 are the same, plus `fleurs-it_it-10210638754804006465`
for ours (the "gli" / "vi" tie above) and one for the engine's. The
embedding half holds since 8.2 (F-027: mean per-frame cosine 0.999828, every clip ≥ 0.999033).

**Accept settled (2026-10-10).** By the user's decision the Phase 8 Accept now asks for the offline text at least as
often as the engine's own stream reaches it, with the call parity reported. 18/32 against the engine's 18/32 meets
it, with 797 of 930 calls (85.7 %) equal to the engine's.

### Buffers over 30 s with nothing to trim (8.4)

`long-silence-lead` is 32.5 s of zeros, then `ls-clean-6930-75918-0000`, fed in 1 s chunks. Nothing is committed in
the silence, so the sentence trim never fires and the buffer passes 30 s at the 31st call. The engine's passes at 33,
34, 35, 36 and 36.01 s give "Vielen Dank." [de], "Coming forward, we'll try.", "Current Court returned to its
place.", "Concord returned to its place amidst the tents." and "Current Court returned to its place amidst the
tents."; the stop commits the last with word times 33.12, 33.44, … 35.76.

Evidence on what the engine does:

- **Times.** Every stop time is a multiple of 0.08 s from the stream start. A pass over the last 30 s exactly starts
  at 6.01 s and would put the words off that grid (our model: 33.05, 33.45, …). So the window starts on the frame
  grid, at 0 or at a multiple of 1280 samples.
- **`received`** at 36.01 s prints "36.01", which is n · (1/16000) with the offset at 0. An offset moved by
  float32 frame times would print "36.00" (our first attempt did).
- **Windows ≤ 30 s.** F-032's brute force on the engine found no single t0 and no single window length within 30 s
  that serves the 33 to 36 s passes together.
- **Our model, every window start on the grid** (`[s, end)`, s = 0, 0.08, … up to end − 1 s, 2116 passes without
  the 30 s check; `TestStreamWindowScan`, `just stream-parity -run TestStreamWindowScan`, 7 min):

  | call    | the engine's pass                                 | starts that reproduce it  | s = 0 (whole buffer) | last ≤ 30 s on the grid |
  | ------- | ------------------------------------------------- | ------------------------- | -------------------- | ----------------------- |
  | 33 s    | [de] "Vielen Dank."                               | 272 of 401                | yes                  | yes (3.04 s)            |
  | 34 s    | "Coming forward, we'll try."                      | 0 of 413                  | no                   | no                      |
  | 35 s    | "Current Court returned to its place."            | 251 of 426                | yes                  | yes (5.04 s)            |
  | 36 s    | "Concord returned to its place amidst the tents." | 212 of 438, all ≥ 16.40 s | no                   | no (6.00 s)             |
  | 36.01 s | "Current Court returned to its place amidst …"    | 205 of 438                | yes                  | yes (6.08 s)            |

So the engine most likely passes the whole buffer beyond 30 s: `needle_transcribe`'s 30 s check guards the public
call, not the stream's internal pass, and the window, the times and `received` all fit an offset of 0. The 34 s and
36 s passes are out of our model's reach from any start, which is W2A8 noise on a 0.5 s fragment after 32 s of
zeros.

**Decision.** The port never feeds more than 30 s to a pass (PLAN.md 8.4). `limit` drops the oldest audio in whole
80 ms frames once the buffer exceeds 30 s, so a pass sees the last 30 s, or up to one frame less, on the frame grid.
It moves the offset by the exact sample time, which keeps `received` at the engine's (36.01). The committed words in
the dropped audio go with it, as in the trim. This also bounds memory in a long silence; the engine's buffer appears
to grow without bound there. On the evidence above it gives the same passes as the whole buffer, 3 of 5. Its stream
then differs from the engine's at 34 and 36 s and commits "Current Court returned to its" at 36 s, where the engine
waits and commits the whole sentence at the stop. What still differs: any stream whose speech needs more than 30 s
of context from before the window. None is recorded, and recording more needs new engine runs.

**Not covered.** The stream fixtures have keywords off, so `StreamOptions.Keywords` is checked only as
`Transcribe`'s. `pass_ms` is the call's wall time. Nothing pins its exact span in the engine (the stop's 0.0 aside).
The sentence splitter is F-032's rule, not a Moses port: abbreviations such as "Mr." before a name would end a
sentence. None occurs in the fixtures.

## F-037 — The packed path: CQ matrices straight from `whistle.cact` in every linear layer (9.2) (2026-10-10)

**Question.** Can the whole model run on F-028's packed CQ matrices (F-034's `lut` kernel) instead of the 221 MB of
dequantised float32, with the same transcripts and WER, and does it fit PLAN.md 9.2's memory target of 64 MB?

### Design

**Source side** (`internal/weights`). `Options.Packed` opts a `.cact` source into the packed path; the zero `Options`
keeps today's float behaviour, so every existing caller and fixture test is unchanged. A packed source is a
`weights.Packer`: `Matrix(name)` returns the CQ record as a `*cq.Matrix` built from `cact.File.CQPacked`, and `false`
for any other record, for a source in float mode and for safetensors (`weights.PackedMatrix` does the type assertion).
The matrix is built once per name and cached like `Tensor`'s values (a per-name `sync.Once`; `Release` drops it, `Close`
fails later calls with `ErrClosed`). No float32 copy is made. The matrix's index bytes alias the file data, so this is
the one exception to F-007's aliasing rule: a packed matrix, and a model built on it, is valid until `Close` only.
`whistle.Model.Close` already forbade later use; its doc now also says that `Close` must not overlap a call, because
a read of an unmapped page faults instead of failing. A CQ width without a `cq` kernel (Whistle has only 2 and 4 bits)
is served as not packed, so the model falls back to `Tensor` for it. `Cached()` lists the names the float32 cache holds, for the tests below.

**Deferred lut copy** (`internal/cq`). `cq.New` builds F-034's blocked copy of index and norms at once. The new
`cq.NewGather` leaves it to `Prepare` or the first `GEMM`, and `Bytes` counts it only once built. The source builds
matrices with `NewGather`; the model calls `Prepare` for every matrix it multiplies by and not for the engram tables,
which are only gathered. That saves the tables' blocked copy, 2 × 2.65 MB.

**Model side** (`internal/needle`). `needle.Linear` holds an `[out, in]` matrix as float32 values or as a `*cq.Matrix`:
`Apply(dst, x, rows)` is `tensor.MatMulTransB` or `Matrix.GEMM`, `Row` copies a row or dequantises it with `CQ`'s
arithmetic (bit for bit), and `Rows(lo, hi)` is a row view. `needle.LoadLinear(src, name, out, in)` takes the packed
matrix where the source serves one and the float32 tensor otherwise. Every CQ site uses it:

| site                                                     | tensors                                               | call            |
| -------------------------------------------------------- | ----------------------------------------------------- | --------------- |
| self-attention, encoder and decoder (`needle.Attention`) | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `out_proj` | `Apply`         |
| mHC (`needle.MHC`)                                       | `mhc_phi_pre`, `mhc_phi_post`, `mhc_phi_res` (CQ4)    | `Rows`, `Apply` |
| engram (`needle.Engram`)                                 | `tables`                                              | `Row`           |
| engram                                                   | `key_proj`, `value_proj`                              | `Apply`         |
| encoder stem (`encoder.stem`)                            | `stem.pw_1`, `stem.pw_2`, `stem.out`                  | `Apply`         |
| encoder conv module (`encoder.block`)                    | `pw1`, `pw2`                                          | `Apply`         |
| cross-attention (`decoder.cross`)                        | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `out_proj` | `Apply`         |
| token embedding and tied head (`decoder.Decoder`, CQ4)   | `embedding`                                           | `Row`, `Apply`  |

That is all 152 CQ records. The FP16 records (norms, gates, taps, the Hadamard MLP's factors, `cond_u`/`cond_v`, the
depthwise kernels) stay float32. The encoder's hypothesis wrapper (`permSource`) forwards `Matrix`, so the packed path
survives it. `LoadMHC` still loads the stacked φ per layer, but the source's cache makes that one matrix per tensor,
viewed per layer.

**Load** (`whistle/model.go`). `Load` opens a `.cact` packed by default; `WithFloatWeights()` restores the float path.
A safetensors checkpoint is float either way. The evidence below is why packed is the default.

**Float caches.** Model code never calls `Tensor` on a packed name, so no float32 copy of a CQ tensor is made, let alone
kept. `TestPackedMatchesFloat` (`internal/decoder`) loads the encoder and the decoder from one packed source and finds
527 float32 tensors in its cache, none of them CQ.

### Parity

All numbers come from the Apple M5 Pro (arm64, `lut` NEON kernel). amd64 (AVX2, Docker emulation) and `purego` are
noted where they differ.

| check                                                              | packed                                      | float                 |
| ------------------------------------------------------------------ | ------------------------------------------- | --------------------- |
| `Linear`, 7 real tensors, 5 random rows, worst 1 − cos             | 1.02e-7 (embedding); purego 2.3e-13         | —                     |
| `Attention` / mHC layer / engram on random input, 1 − cos vs float | 6.1e-8 / 6.0e-9 / 9.2e-9                    | —                     |
| `Embed` vs `needle_embed`, mean frame cosine, 32 clips             | 0.999827 (amd64 0.999828)                   | 0.999828              |
| `Embed` packed vs float, mean frame 1 − cos / worst frame cosine   | 4.8e-8 / 0.999945 (amd64 7.0e-8 / 0.999893) | —                     |
| greedy decoding, 32 clips: transcripts equal packed vs float       | 32/32 (amd64 32/32)                         | —                     |
| teacher-forced mean log p of the engine's texts, max \|Δ\|         | 7.2e-5 (amd64 1.9e-4, purego 1.9e-7)        | —                     |
| `Transcribe` (beam search) equal to the engine, 32 clips           | 31/32                                       | 31/32                 |
| `Transcribe` packed equal to float, 32 clips                       | 32/32                                       | —                     |
| depth ladder 2–7 (`testdata/oracle/depth`), equal to the engine    | 161/192                                     | 161/192               |
| depth ladder, packed equal to float                                | 192/192                                     | —                     |
| keyword cases (`testdata/oracle/keywords`), equal to the engine    | 45/53                                       | 45/53                 |
| keyword cases, packed equal to float                               | 53/53                                       | —                     |
| LibriSpeech subset (`just eval-librispeech`), WER                  | 223/4167 = **5.35 %**                       | 223/4167 = **5.35 %** |
| LibriSpeech, text equal to the engine's                            | 181/200                                     | 181/200               |
| LibriSpeech, packed equal to float                                 | 200/200                                     | —                     |

The miss on the 32 clips is F-026's Polish search miss in both modes. On every transcript compared (32 + 192 + 53 + 200),
packed and float give the same text and language, so the WER is unchanged (0.00 points; the Phase 9 Accept allows 0.1).
The depth and keyword runs used a throwaway driver over `Transcribe` with `Depth` and `Keywords`; the committed tests
cover their fast subsets (`TestTranscribeDepthMatchesEngine`, `TestTranscribeKeywordsMatchEngine`), which now run packed.
The engine's own W2A8 numerics are much further away (F-024) than the lut kernel's table rounding (F-034).

### Memory

`just mem-probe` (`TestMemProbe`, `internal/weights`) runs its own test binary twice as a child, once per mode. Each
child reads the first corpus clip tiled to 30 s, then runs `Load` and one `Transcribe` and reports:

- `ru_maxrss` before `Load` (the test binary and the clip), after it and after the `Transcribe`;
- the live Go heap after a GC right after `Load`, and again at the end;
- the resident pages of the `.cact` mapping at the end (`vmmap` on darwin, `/proc/self/smaps` on linux).

The model's memory is defined as the live Go heap plus the resident `.cact` pages. That leaves out the binary, the
runtime's free heap and the GC's not yet collected garbage. M5 Pro, `GOGC=100`, MiB:

| MiB                                        | packed    | float |
| ------------------------------------------ | --------- | ----- |
| process before `Load` (`ru_maxrss`)        | 12.0      | 12.1  |
| process after `Load` (`ru_maxrss`)         | **41.7**  | 242.8 |
| live Go heap after `Load` (clip 1.8 incl.) | 20.1      | 216.3 |
| process peak over the 30 s `Transcribe`    | **127.6** | 488.3 |
| live Go heap at the end                    | 19.4      | 214.5 |
| resident `.cact` pages at the end          | 14.7      | 16.1  |
| **model: live heap + `.cact` pages**       | **34.1**  | 230.6 |

The linux/amd64 child (emulated, so its baseline is 50 MiB) agrees: model 36.5 against 230.9 MiB, with `RssFile` 22.1
MiB (binary and mapping) and `RssAnon` 122.9 MiB at the end.

Where the packed heap goes after `Load` (allocation profile): the lut blocked copies 10.6 MiB, the widened FP16 norms of
the matrices 1.7 MiB, the FP16 records as float32 2.3 MiB, the tokenizer 0.9 MiB. The `.cact` pages are resident because
`Prepare` and the FP16 decode read them once. Afterwards the lut kernel reads only its blocked copy, and `Row` reads the
embedding's and the engram tables' pages.

So the weights take 34 MiB against 231, 6.8× less, and the process stays under 64 MiB after `Load` (41.7). A 30 s
`Transcribe` still peaks at 127.6 MiB. That is not weights but activations and GC headroom, and the float path has
them too:

- The stem's first conv outputs 1500 × 40 × 128 floats, 29.3 MiB, live at once: the 50 MiB live heap at the sixth GC,
  19 ms into the run (`GODEBUG=gctrace=1`).
- The cross-attention memory of 8 layers × 375 frames × (384 + 512) floats is 10.3 MiB for the whole decode.
- `GOGC=100` lets the heap grow to twice the live heap before a collection: 70–100 MiB. Under `GOMEMLIMIT=40MiB` the peak
  is still 91.5 MiB, because the stem's spike alone needs a 64 MiB heap.

### Speed

`TestSpeedProbe` (`WHISTLE_SPEED_PROBE=1`, `internal/weights`) loads both modes in one process and alternates them over
the 32 clips, three timed passes after a warm-up, and over the 30 s clip (six runs). Four runs on the M5 Pro while other
sessions kept the load average at 3.5–5.7, so the absolute numbers sit above F-035's; the ratios come from the same
process and the same minutes:

| median                      | packed        | float         | packed vs float         |
| --------------------------- | ------------- | ------------- | ----------------------- |
| `Load`                      | 9.9–10.9 ms   | 160–170 ms    | 16× faster              |
| TTFT, 32 clips              | 69.9–73.1 ms  | 65.2–71.0 ms  | 2–7 % slower            |
| decode rate, 32 clips       | 765–792 tok/s | 487–497 tok/s | 1.56–1.59×              |
| TTFT, 30 s clip             | 320–373 ms    | 302–357 ms    | 4–8 % slower            |
| LibriSpeech, 200 utterances | RTF 0.0188    | RTF 0.0210    | 1.12× (sequential runs) |

Decoding runs one row per beam, where F-034 measured the lut GEMV 1.05–3.2× faster than the NEON float GEMV: the decode
rate gains 56–59 %. The time to the first token is dominated by the encoder and the cross-attention memory, which multiply
375 rows at once. There the lut kernel runs one activation per pass over the index, while the float GEMM reuses each
weight across four rows (F-034, "Batches"). The CQ4 φ add a 2048-wide table build per row for only 4 or 16 outputs, three
times per layer on the same input. A throwaway 30 s `Embed` benchmark under the same load took 328 ms packed against
320 ms float. In its profile,
table building and FWHT take 9 %, and `runtime.madvise` (the GC returning memory) 13 %.

### Tests and tolerances

No existing tolerance moved, and no fixture changed. The `internal/needle`, `internal/encoder` and `internal/decoder`
tests and fixtures still open their sources with the zero `Options`, so they still run the float path bit for bit: the
float `Linear` calls `tensor.MatMulTransB` with the arguments `linear` used, and a throwaway check hashed the float
encoder output and the decoder logits of the prompt plus three tokens on the three fast clips, equal to `origin/main`'s
bit for bit. The `whistle` tests now run the packed
default and pass with their thresholds as they were (`TestEmbedMatchesEngine` 0.999827 against its 0.999 floor). New
regressions, model-gated and skipped without `models/`, with bounds just outside the worst measured architecture:

- `TestWhistlePacked`, `TestWhistlePackedLifecycle` (`internal/weights`): 152 packed records, shapes, cache, rows bit
  for bit against `Tensor`, no float decode.
- `TestLinear`, `TestWhistleLinearPacked`, `TestWhistleModulesPacked` (`internal/needle`): 1 − cos ≤ 2e-7.
- `TestNewGather` (`internal/cq`): the deferred copy equals `New`'s, concurrently and through an early `Rows` view.
- `TestBestPackedMatchesFloat` (`internal/encoder`): mean cosine to `needle_embed` ≥ 0.9998 and within 2e-5 of float;
  packed vs float mean frame 1 − cos ≤ 6e-7, worst frame ≥ 0.99985.
- `TestPackedMatchesFloat` (`internal/decoder`): greedy transcripts equal, |Δ mean log p| ≤ 3e-4, no CQ tensor in the
  float32 cache.

### Open

- **The 30 s `Transcribe` peak.** By the user's decision (2026-10-10) PLAN.md 9.2's 64 MB target counts the weights
  (live Go heap plus resident `.cact` pages, 34 MiB, met); the process peak (127.6 MiB) is a separate goal.
  The levers are outside 9.2's wiring: run the stem in time chunks (removes the 29 MiB spike), keep the cross-attention
  memory in int8 (9.3), let the application set `GOMEMLIMIT`, and move the lut copy out of the Go heap or into the file
  pages it duplicates (F-034's in-register transpose), so that the GC does not double it.
- **TTFT.** Sharing one FWHT of an activation between the matrices that read it (q/k/v, the three φ, k/v of the memory),
  and 9.5's parallelism over row blocks for the batched products.

## F-038 — Parallelism: the encoder over time blocks and heads, shared rotations, the stem in chunks (9.5) (2026-10-10)

**Question.** Since 9.2 every linear layer runs `internal/cq`'s single-threaded lut kernel. `tensor.Workers()` defaulted
to 1 and nothing called `SetWorkers`, and `tensor.MatMulTransB`, which `internal/needle` uses for every float product,
ignored it anyway. So the whole encoder ran on one core; only `search`'s beams used more (one goroutine each, F-026).
Where does parallelism pay, can every output stay the same bit for bit, and do F-037's two open levers (one rotation per
activation for all matrices that read it, the stem in time chunks) hold?

**Starting point.** A CPU profile of `Embed` on the 30 s clip (the first corpus clip tiled, M5 Pro, packed): 279 ms,
the stem 25 % (its SiLUs, a float64 `math.Exp` per value, 56 % of that; the convs 32 %), the 8 layers 75 %.
`TestSpeedProbe` before: median TTFT 58.8 ms packed, 56.1 ms float; the 30 s clip 309 / 288 ms (load average 2.0–3.2).

### Design

**One worker pool for all kernels** (`internal/tensor`). `tensor.Parallel(n, grain, fn)` runs `fn` on pieces of
`[0, n)` on the calling goroutine and helpers. Three rules make it safe to nest and to call from many goroutines:

- **One helper budget for the process.** A helper starts only while fewer than `Workers() − 1` run, counted over all
  calls at once; a caller that finds the budget used runs its pieces itself. So a GEMM inside a loop over heads, or the
  five beams each running their own GEMVs, never add more than `Workers() − 1` goroutines to their callers.
  `TestParallelBoundsHelpers` runs Parallel inside Parallel from five goroutines on 4 workers: never more than 5 + 3
  run `fn` at once, and the budget is back at 0 after.
- **Pieces fixed by n, grain and Workers() only**, not by how many helpers are free. Each starts at a multiple of grain.
  Up to 4 · `Workers()` pieces whose sizes differ by at most one grain, taken in turn by whoever is free: a core that
  another process holds up takes fewer. The first version cut `ceil(units / w)`-sized pieces, which left the stem's 47
  chunks in 12 pieces on 15 workers.
- **The default is GOMAXPROCS, read once at start-up.** `runtime.GOMAXPROCS(0)` takes the scheduler's lock; called on
  every `Parallel`, it showed up as 2.5 s of lock contention in a 8 s profile. `SetWorkers(1)` restores single-threaded
  kernels.

`MatMulTransB` splits a product of at least 2²⁰ multiply-adds: over the rows of a from 8 rows on, else over the rows
of b (the decoder's GEMVs), in pieces that start at multiples of 4. The NEON kernel computes 4×4 blocks of rows and
1×4 blocks for the leftover rows, and a 4×4 block sums in another order than a 1×4 one, so a split that moved the block
boundaries would change the last bits. With pieces aligned to 4, every block stays as it is in one call
(`TestMatMulTransBWorkers`, nine shapes with row and column splits, k tails and leftover rows: equal bit for bit on 1,
2, 3, 7 and 16 workers).

**cq** (`internal/cq`). The lut kernel splits a product over its activations when there are at least `Workers()` of
them, or when one activation's work is below 2 · 2¹⁸ index bytes; else, for a single large product (the tied head,
8199 rows), it builds the tables once and splits the lookup over the blocks of 32 rows. Both splits leave every output
the same sum in the same order: an activation's tables do not depend on the split, and `lutLookup` computes every row of
a block on its own. The scalar kernels split over activations in multiples of four, so `nibble4` (purego, amd64 without
AVX2) sees the same groups of four. `TestGEMMWorkers`: every kernel on 1, 2, 5 and 16 workers, through `Rows` views that
start and end inside a block, equal bit for bit, and the lut kernel also equal to its unsplit Go reference.

**One rotation and one table build per activation** (`cq.GEMMs`, `needle.ApplyShared`). `.cact` stores one codebook
per width, so every CQ2 matrix of input width 512 builds the same tables from the same activation. `GEMMs` rotates and
tabulates each activation once and looks it up in every matrix: q, k, v and the gate's projection in `needle.Attention`
(the encoder's, and the decoder's per step), the three mHC φ (CQ4, 2048 wide, the costliest tables per output, F-037),
and the engram's key and value projections. The result is the same bits as a GEMM per matrix (`TestGEMMsMatchesGEMM`,
`TestApplySharedMatchesApply`), so this is exact. `BenchmarkGEMMs` (`-cpu 1`, median of three): q, k, v and
gate 6.47 against 8.20 µs per decode step and 2.43 against 3.04 ms over 375 frames (1.25×); the φ 7.43 against 13.2 µs
and 2.80 against 4.96 ms (1.77×). On 15 cores over 375 frames: 0.35 against 0.70 ms and 0.34 against 1.02 ms.

**needle** (`internal/needle`). The attention runs its (head, query row) pairs on `Parallel`, in blocks of rows that
start at multiples of 4 within a head, so that its Q·Kᵀ and P·V GEMMs keep their 4-row blocks. The norms, the sigmoid
gate, `HadamardMLP`'s rows and the mHC's per-row heads and mix run over row ranges; each row is computed on its own.
In a decoder step (t = 1) the row loops are one piece and stay on the calling goroutine; of
its products only the tied head (8199 rows, packed or float) is large enough to split.

**encoder** (`internal/encoder`). The conv module's GLU, depthwise conv and activation run over row ranges. The stem
runs in chunks of 8 output frames (`stemChunk`), on `Parallel`. For its output frames [o0, o1), a chunk works back
through the three stride-2 convs to the rows each needs, and pads its slice of rows only where the whole axis is padded,
so each conv pixel reads the same taps in the same order as in one call. The activations work value by value; the
pointwise products (`stem.pw_*`, `stem.out`) sum some rows in blocks of 4 (the NEON float GEMM, `nibble4`), so a chunk
must start on such a block: true for Best's mel widths of 20 and 10, not for VALID padding's 19 and 9, which the
weights rule out anyway (F-016). A chunk recomputes one shared row per stride-2 stage: 35 rows of the first conv's 1500
for 8 frames (32 without the overlap), 9 % more for that conv. Instead of 1500 × 40 × 128 floats (29.3 MiB, F-037) at
once, each worker holds one chunk's 35 × 40 × 128 (0.7 MiB).

**The decoder over beams** stays as it was: `search` runs each hypothesis' `Extend` on its own goroutine (F-026). With
the kernels now parallel too, the five beams and their kernels share the one helper budget. Nothing in
`internal/decoder` or `internal/search` changed.

### Bit for bit

Every split keeps each output's sums in the same order, so outputs do not depend on the worker count, and with one
worker they equal the code before. The tests, all on random data without `models/` unless marked, pass on arm64, under
`purego` and on linux/amd64 (AVX2, Docker emulation), also under `-race`:

| test                                                         | compares                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------- |
| `TestParallelPieces`, `TestParallelBoundsHelpers` (`tensor`) | pieces cover once, aligned, even; nested callers stay in budget   |
| `TestMatMulTransBWorkers` (`tensor`)                         | 1 vs 2, 3, 7, 16 workers, row and column splits                   |
| `TestGEMMWorkers`, `TestGEMMsMatchesGEMM` (`cq`)             | every kernel 1 vs N; lut vs its unsplit reference; shared vs not  |
| `TestApplySharedMatchesApply`, `Test*Workers` (`needle`)     | packed and float attention (encoder, causal with cache), MLP, mHC |
| `TestStemChunks` (`encoder`)                                 | chunks vs `whole` for 9 clip lengths, 6 stem geometries, 1 vs 16  |
| `TestWhistleStemChunks`, `TestWhistleEmbedWorkers` (model)   | real weights, packed and float, a short clip and 30 s             |

Against `origin/main` (62a4458), with a throwaway test in both trees: the `Embed` rows of all 32 corpus clips (one tiled
to 30 s) hash the same, packed and float, and so do the text, language, word times and word probabilities (float64
bits) of `Transcribe` on the 32 clips and the 30 s clip with word timestamps, with `Depth` 4 and with two keywords.

### Speed

`TestSpeedProbe` (packed and float in one process, alternating, F-037) on the M5 Pro (5 + 10 cores). Other sessions kept
the load average at 4–7, so the old (`origin/main`) and the new test binaries ran alternately, two rounds each, in the
same minutes; the ranges are the two rounds. `WHISTLE_SPEED_WORKERS=1` (new) runs the kernels on one worker,
`GOMAXPROCS=1` runs everything, the beams too, on one core.

| median                 | old           | new           | new, 1 worker  | old, 1 core   | new, 1 core   |
| ---------------------- | ------------- | ------------- | -------------- | ------------- | ------------- |
| TTFT, 32 clips, packed | 65.3–67.1 ms  | 21.2–21.3 ms  | 59.0–59.9 ms   | 59.4–62.4 ms  | 54.9–61.2 ms  |
| TTFT, 32 clips, float  | 60.7–62.9 ms  | 23.4–24.0 ms  | 60.7–60.9 ms   | 56.0–59.6 ms  | 55.9–62.4 ms  |
| TTFT, 30 s, packed     | 332–337 ms    | 66–67 ms      | 300–327 ms     | 307–314 ms    | 290–324 ms    |
| TTFT, 30 s, float      | 311–314 ms    | 72–75 ms      | 303–333 ms     | 284–291 ms    | 292–312 ms    |
| decode, packed         | 819–947 tok/s | 960–961 tok/s | 991–1012 tok/s | 243–252 tok/s | 233–270 tok/s |
| decode, float          | 539–570 tok/s | 543–545 tok/s | 564–566 tok/s  | 143–149 tok/s | 128–149 tok/s |

So TTFT is 3.1× (32 clips) and 5.0× (30 s) shorter packed, 2.6× and 4.2× float; packed is now faster than float on
both. A first run at a load average of 2–3 gave 58.8 → 19.8 ms and 309 → 67 ms packed, 56.1 → 22.3 ms and
288 → 73 ms float. On one core the new code is within the noise of the old: the shared rotations and the chunked stem
save some 5–10 % there, and the split costs nothing.

`just eval-librispeech` (200 utterances, sequential, load 4): RTF 0.0086, 67 ms per utterance; F-037 had 0.0188.

**The decoder over beams.** One core decodes 233–270 tok/s packed, all cores 960–1060: the beams' goroutines give the
4×, as before. The only decode product big enough to split is the tied head (lut 97 → 39 µs on 15 cores, float 193 →
75 µs, `BenchmarkGEMM`). Three more alternating rounds (load 3–5) compared the new build with a build that does not
split single activations: 1028–1060 against 1000–1044 tok/s packed, so the head split stays. With one kernel worker the
decode rate is 1070–1081 tok/s, and the no-split build gives 1055–1078 on one worker against 1000–1044 on all, with the
same decode work: the 3–5 % that the parallel encoder costs the following decode steps is not in the steps' own
kernels (whose splits are the head only). It may be the cores' clocks or the GC after the burst; it is not resolved.

**Nested parallelism.** Beams × kernel workers stays bounded by the helper budget: 5 beams plus at most 14 helpers
(`TestParallelBoundsHelpers`). Below the piece sizes (2²⁰ multiply-adds for the float GEMM, 2 · 2¹⁸ index bytes for the
lut kernel, 16 to 128 rows for the row loops) nothing splits, so a decode step's projections, norms and MLP run on its
beam's goroutine.

**Where the 30 s TTFT goes now** (a throwaway phase timer, packed, load 6): log-mel 6 ms (single-threaded,
`internal/frontend`), stem 11 ms, the 8 blocks 39 ms (attention 17, conv module 12, the two MLPs 10), the mHC around them
9 ms, then the cross-attention memory and the first decode step. The attention at t = 375 takes 4.4 ms on one worker and 1.2 ms on 8 or 15
(load 6). In a CPU profile of 40 `Embed` calls, about half of the samples are the scheduler (idle cores waking and
looking for work) and 8.5 % are `madvise`: the GC hands freed pages back, and the next large `make` asks for them
again (about 17 collections per `Embed`).

### Memory

`just mem-probe` (F-037), alternating the old and new binaries, two rounds, load 3.5–4:

| MiB                                   | old           | new           | old, 1 core   | new, 1 core   |
| ------------------------------------- | ------------- | ------------- | ------------- | ------------- |
| packed: process after `Load`          | 41.5 / 41.4   | 41.5 / 41.7   |               |               |
| packed: peak over a 30 s `Transcribe` | 127.6 / 126.8 | 108.1 / 106.4 | 126.4 / 123.0 | 97.5 / 97.7   |
| packed: model (live heap + `.cact`)   | 34.3 / 34.2   | 34.9 / 34.7   | 32.9 / 32.9   | 32.9 / 32.9   |
| float: peak                           | 489.3 / 489.0 | 494.1 / 495.2 | 488.8 / 489.1 | 489.6 / 489.5 |

On one core the chunked stem alone takes 26–29 MiB off the packed peak (the 29.3 MiB of F-037). On 15 cores the workers'
buffers live at the same time and give some 10 MiB back, and the scratch pools keep about 0.6 MiB more. What is left
of the 106–108 MiB is the GC's headroom: during the decode the live heap is 29–40 MiB (20 MiB weights, 10 MiB
cross-attention memory, the beams), and `GOGC=100` lets the heap grow to twice that before a collection
(`GODEBUG=gctrace=1`: 70–84 MiB heaps).

### Parity

All outputs are the same bits as before, so every parity number is unchanged, packed and float:
`TestTranscribeMatchesEngine` 31/32 (the same 32 transcripts), `TestEmbedMatchesEngine` 0.999827 (floor 0.9998), the
depth and keyword regressions pass, `just eval-librispeech` WER 223/4167 = 5.35 % (engine 5.30 %), `just stream-parity`
797/930 calls with an equal decision.

### Open

- **The decoder's cross-attention memory** (`internal/decoder/cross.go`, k and v of 375 frames) still rotates the
  encoder output twice. `needle.ApplyShared` takes both projections as they are; it is a two-line change in a package
  outside 9.5's scope.
- **Allocation.** Each block allocates its intermediates afresh, 0.8–3 MB each for 30 s, so the GC runs about 17 times
  per `Embed` and `madvise` shows in the profile; reusing buffers per layer would cut both the time and the GC headroom
  of the peak.
- **The log-mel front end** (6 ms of the 30 s TTFT) is single-threaded.
- **The 3–5 % decode cost after a parallel encoder** (above).

### The stem's conv through algo-tensor v0.3.1 (2026-10-12)

algo-tensor v0.3.0 (its PLAN 1.3) runs `ops.Conv2D` with groups = 1 as im2col plus `MatMulTransBBias` on arm64 and
js/wasm, not on amd64; v0.3.1 adds a bit-exact `Linear` that go-whistle does not call. Of the stem's convs only the
first (3 × 3, 1 → 128) has groups = 1; the depthwise stages keep their code. The CLI on the 32 fixture clips, `main`
(3a747a2, v0.1.0) against the bump (afd44c8):

| output (32 clips; float on 3)                          | arm64 NEON                                                       | arm64 purego, linux/amd64 |
| ------------------------------------------------------ | ---------------------------------------------------------------- | ------------------------- |
| `embed`, packed                                        | all differ: max abs diff 0.022, cosine ≥ 0.9999996               | byte-identical            |
| `embed`, float                                         | all differ: max abs diff 2.6e-6                                  | byte-identical            |
| `transcribe --json --word-timestamps`, packed, depth 4 | text, words, times equal; 15 clips move one probability by 0.001 | byte-identical            |
| `transcribe`, float                                    | byte-identical                                                   | byte-identical            |

The packed difference is larger, most likely because the lut kernel quantises its activations (F-034), so a last-bit
change sometimes moves a value across a rounding step. Parity on arm64 (the Phase 9 numbers):
`TestTranscribeMatchesEngine` 31/32, `TestEmbedMatchesEngine` 0.999827 → 0.999828 (worst clip 0.999034 → 0.999038),
keywords 47/53 (28/32 sets, 28/31 variants), depth ladder 161/192, `just stream-parity` 797/930, WER 5.35 %. Packed
against float: mean 1 − cos 4.8e-8 → 2.3e-8, worst frame 0.999945 → 0.999984. Only the int8 mode moves the other
way, within its floors: 30/32 as before, `Embed` 0.999719 → 0.999716, worst clip 0.994891 → 0.994790.
`TestWhistleStemChunks` (chunked stem = whole stem) and `TestWhistleEmbedWorkers` still hold bit for bit, and
`just wasm-test ""` passes with the model.

Speed: none at the default 15 workers, where the stem already runs in parallel chunks. Three interleaved `whistle-bench`
runs gave packed TTFT medians of 18.9, 19.3, 19.6 ms (main) against 19.4, 19.1, 19.2 ms, and about 61 ms for both on
the 30 s clip. On one worker (`GOMAXPROCS=1`), `whistle embed` of the longest fixture clip takes 79.1 ms against
81.2 ms (median of 20).

## F-040 — Benchmarks against the engine on the same machine, and the packed WER (9.6, Phase 9 Accept) (2026-10-10)

**Question.** How fast is go-whistle against the Needle 3 engine on the same machine: TTFT, decode rate, real-time
factor, model load and peak RSS (PLAN.md 9.6)? Does the packed default keep the float path's WER within 0.1 (the
Phase 9 **Accept**)? And what exactly do the engine's `ttft_ms` and `decode_tps` time, so that the comparison is like
with like?

### Harness

- **Go side**, `just bench` (`bench/cmd/whistle-bench run`), public `whistle` API only:
  - Both modes are loaded in one process: packed (`whistle.Load` on `whistle.cact`, the default since F-037) and float
    (`WithFloatWeights()`). Five `Load`s per mode, alternating, give the load time.
  - The 32 fixture clips of `TestTranscribeMatchesEngine`, each checked against its SHA-256: one untimed warm-up pass,
    then five timed passes. Per clip the mode that goes first alternates with clip and pass, so both modes see the same
    machine load. The warm-up pass gives the transcripts, compared with the committed engine fixtures.
  - The 30 s clip, the first corpus clip tiled to 480,000 samples (F-027, F-035, F-037): one warm-up and ten timed runs
    per mode, alternating.
  - Peak RSS: a fresh child process per mode (`whistle-bench child`) runs `Load` and one 30 s `Transcribe` and reports
    `ru_maxrss` after each, as `just mem-probe` does. `ru_maxrss` is a high-water mark, so the child loads the model
    before it reads any audio and builds the 30 s clip from the first clip alone (PR #33 review: decoding all 32 clips
    first had put 40.6 MiB after `Load`). The committed memory values are the median of three fresh children per mode
    after that fix: after `Load` 37.9 MiB packed and 238.9 float, the 30 s peak 125.7 and 489.3 (the full run's
    children had given 125.3 and 489.4). A platform without `ru_maxrss` reports 0, which the report shows as missing.
  - WER: the 200 LibriSpeech utterances of F-025, digests checked, both modes per utterance in alternating order. The
    manifest's `wer_rule` goes through `internal/eval` as in `TestEvalLibriSpeech`. The run fails unless the rule
    reproduces `manifest.json`'s count for every engine transcript and its totals.
- **Engine side**, `just oracle-bench` (`tools/oracle bench`, tag `oracle`, `NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1`):
  - The CLI: one `needle` process per transcription, `--threads 4` (`cliThreads`), no word timestamps, the same passes
    and runs. It records `ttft_ms`, `decode_tps`, the process's wall time and its `ru_maxrss`, and checks the warm-up
    pass's text and language against the committed fixtures.
  - The library: `libneedle3` through purego in the oracle process, on its default threads (the fast cores, at most 4,
    so 4 here). Five more `needle_load` calls on the file's bytes in memory, then `needle_transcribe` without word
    timestamps on the same PCM, in the same passes.
  - The 30 s clip is written as a 32-bit float WAV of the tiled samples, so the CLI reads exactly the samples that the
    Go side and the library transcribe.
- **Statistics.** All transcriptions of a set are pooled over the passes. The median of an even count is the mean of
  the middle two; the p90 is the nearest rank. The real-time factor is Σ wall / Σ audio per pass, median over the
  passes.
- **Report.** Both sides write JSON (`bench/results/go.json`, `engine.json`); `whistle-bench report` renders
  `bench/RESULTS.md` from them. Its tables are padded the way prettier pads them, so the generated file and the JSON
  pass `treefmt --fail-on-change` unchanged. Each run records its machine (CPU, cores, core types, OS, architecture)
  and its inputs (the model's SHA-256 and a SHA-256 over the clip manifest's ids and WAV digests). The report divides
  the Go side by the engine's only when both match; otherwise it names each side's CPU, leaves the ratios empty and
  says that the results are not comparable (PR #33 review).

### What the engine's clock measures

| first run, 32 clips, median                                      | ms              |
| ---------------------------------------------------------------- | --------------- |
| CLI `ttft_ms`                                                    | 7.6             |
| library `ttft_ms`                                                | 7.5             |
| CLI process, wall                                                | 39.4            |
| library `needle_transcribe` call, wall                           | 29.8            |
| `needle_load` on the bytes in memory                             | 2.7             |
| call − `ttft_ms` − (n − 1) / `decode_tps`, n = the call's return | 1.35 (p90 2.20) |

- The CLI's and the library's `ttft_ms` agree within 0.1 ms, while the CLI process takes 9.6 ms longer than the
  library call. So `ttft_ms` leaves out the process start, the model load and the WAV reading; it is the same clock in
  both.
- `needle_transcribe` returns the number of tokens generated, n. The call's wall time is `ttft_ms` + (n − 1) /
  `decode_tps` + 1.35 ms at the median. At about 1400 tok/s one decode step takes 0.7 ms, so the engine's two numbers
  account for the whole call but about two steps.
- A throwaway per-call log (one pass) shows that remainder in whole steps: near 0.7 or 1.4 ms (0.64–0.89 and
  1.35–2.07 ms) on 30 of the 32 clips, and 5.0 and 9.2 ms on the two Polish clips, the second F-026's search miss.
  That fits steps that the search runs beyond the n returned tokens: after the winner's EOS until 5 hypotheses have
  finished, as ours does (F-031). It does not fit set-up work outside the clock.
- So the engine's clock starts when the call has the PCM and stops at the first token after the language token (TTFT)
  and at the last search step (decode). That is F-027's reading, and `Result.TTFT` and `Result.DecodeTPS` are defined
  the same way: the clock starts in `Transcribe` after the length check, before the silence gate.
- **Like with like** is therefore `Transcribe` against the library call: both in-process, the model loaded, PCM in
  memory, no word timestamps, after a warm-up. The CLI's process time is reported for completeness only.

### Results before 9.5

Apple M5 Pro (5 Super + 10 Performance cores, 48 GiB), macOS 27.0.1, Go 1.27.1, engine 3.2.0, commit `fe5884e4e6`.
Go ran with GOMAXPROCS 15, the engine with 4 threads. Other sessions kept the load average at 5.2–8.4 during the Go
run and 5.2–5.5 during the engine's (`bench/RESULTS.md` has every row).

| median                        | Go packed | Go float | engine library | packed / engine |
| ----------------------------- | --------- | -------- | -------------- | --------------- |
| TTFT, 32 clips (ms)           | 60.4      | 57.5     | 7.5            | 8.1×            |
| TTFT p90, 32 clips (ms)       | 76.1      | 71.1     | 10.3           | 7.4×            |
| decode, 32 clips (tok/s)      | 1013      | 604      | 1383           | 0.73×           |
| RTF, 32 clips                 | 0.0156    | 0.0188   | 0.0056         | 2.8×            |
| TTFT, 30 s clip (ms)          | 310.4     | 290.1    | 36.3           | 8.6×            |
| decode, 30 s clip (tok/s)     | 620       | 408      | 1082           | 0.57×           |
| RTF, 30 s clip                | 0.0165    | 0.0192   | 0.0048         | 3.4×            |
| model load (ms)               | 7.0       | 138.7    | 2.7            | 2.6×            |
| peak RSS, load + 30 s (MiB)   | 125.7     | 489.3    | 70.0 (CLI)     | 1.8×            |
| equal to the engine, 32 clips | 31/32     | 31/32    | 32/32 (CLI)    | —               |

- **TTFT** is 8× the engine's on the clips and 8.6× on the 30 s clip. Both grow with the audio: the 30 s clip takes
  5.1× the clips' median for us and 4.8× for the engine, so the gap scales with the audio and is not a fixed overhead
  (F-027, F-035). 9.5's parallelism over time blocks and heads is the lever. Packed is 5–7 % slower than float
  there: the batched products (F-037).
- **Decode**: packed runs at 0.73× the engine's rate, float at 0.44×. Packed is 1.7× float, more than F-037's 1.56–1.59×
  under a different load.
- **Real-time factor**: 0.016–0.017, about 60× faster than real time, against the engine's 0.005 (2.8–3.4×).
- **Load**: 7.0 ms packed against `needle_load`'s 2.7 ms, which gets the bytes already in memory; float takes 139 ms.
- **Memory**: a 30 s run peaks at 125.7 MiB packed against the engine CLI's 70.0 MiB (model load included in both;
  ours includes the Go runtime), and 37.9 MiB after `Load`. F-037's open levers (the stem's activation spike, `GOGC`
  headroom) are the difference.
- **Against the published figures**: the engine here gives 7.5 ms and 1383 tok/s, the published M4 Pro figures 11.1 ms
  and 1319 tok/s on another clip set. This machine is faster in TTFT and about equal in decode, so the engine on the
  same machine, not the published figure, is the yardstick.

**Spread.** A first full run on the same code before it was committed (load average 4.1–6.9) gave packed 69.2 ms /
796 tok/s, float 66.3 ms / 528 tok/s, and engine library 7.6 ms / 1339 tok/s. The Go numbers move by up to 27 %
between runs under the shared load, the engine's by a few percent. Packed / engine: TTFT 8.1–9.1×, decode 0.59–0.73×.

**Workers.** A throwaway run with `-procs 4` (the engine's thread count, no WER, load average 7.9–8.0) gave packed
68.1 ms / 504 tok/s and float 65.4 ms / 295 tok/s, and a 30 s TTFT of 336 / 317 ms. TTFT barely depends on the workers,
because the encoder runs on one goroutine today. Decode does: the search fans the beam expansion out over goroutines
(F-035). At the engine's 4 threads, packed decodes at 0.36× its rate.

### Results after 9.5 (2026-10-11)

`just bench` on the merge of `main` with 9.5 (F-038, commit `cd6111f0e3`) and `just oracle-bench` right after it
(`513d7f19bf`, the same code), on a quieter machine: load average 2.2–5.7 during the Go run, 3.7–3.8 during the
engine's. These are the numbers in `bench/RESULTS.md`.

| median                        | Go packed | Go float | engine library | packed / engine |
| ----------------------------- | --------- | -------- | -------------- | --------------- |
| TTFT, 32 clips (ms)           | 19.3      | 21.6     | 6.9            | 2.8×            |
| TTFT p90, 32 clips (ms)       | 21.9      | 24.8     | 8.5            | 2.6×            |
| decode, 32 clips (tok/s)      | 1071      | 597      | 1499           | 0.71×           |
| RTF, 32 clips                 | 0.0084    | 0.0130   | 0.0048         | 1.75×           |
| TTFT, 30 s clip (ms)          | 59.9      | 66.9     | 34.7           | 1.73×           |
| decode, 30 s clip (tok/s)     | 639       | 414      | 1164           | 0.55×           |
| RTF, 30 s clip                | 0.0081    | 0.0116   | 0.0045         | 1.8×            |
| model load (ms)               | 6.7       | 136.2    | 2.4            | 2.8×            |
| peak RSS, load + 30 s (MiB)   | 105.7     | 493.0    | 69.9 (CLI)     | 1.51×           |
| equal to the engine, 32 clips | 31/32     | 31/32    | 32/32 (CLI)    | —               |

- **TTFT** fell from 60.4 to 19.3 ms on the clips (3.1×) and from 310 to 60 ms on the 30 s clip (5.2×). The gap to the
  engine is now 2.8× on the clips and 1.7× on the 30 s clip, so what remains is mostly a fixed cost per call, not one
  that grows with the audio. Packed is now 11 % faster than float in TTFT, where it was 5 % slower: the shared rotations
  and the parallel lut GEMM (F-038).
- **Decode** barely moved (1013 → 1071 tok/s packed): the beams were parallel before 9.5. The engine's rate also came
  out higher on the quieter machine (1383 → 1499), so the ratio stays at 0.71–0.73×.
- **Real-time factor**: 0.008, about 120× faster than real time, against the engine's 0.0045–0.0048.
- **Memory**: the 30 s peak fell from 125.7 to 105.7 MiB with the stem in chunks, 1.5× the engine CLI's. One fresh
  child per mode this time, not the median of three.
- The **Workers** paragraph above describes the code before 9.5, whose encoder ran on one goroutine.

### WER (Phase 9 Accept)

| LibriSpeech test-clean subset, 200 utterances | packed (default) | float (`WithFloatWeights()`) | engine   |
| --------------------------------------------- | ---------------- | ---------------------------- | -------- |
| errors / reference words                      | 223/4167         | 223/4167                     | 221/4167 |
| WER                                           | 5.35 %           | 5.35 %                       | 5.30 %   |
| text equal to the engine's                    | 181/200          | 181/200                      | —        |
| real-time factor, before 9.5                  | 0.0163           | 0.0187                       | —        |
| real-time factor, after 9.5                   | 0.0077           | 0.0113                       | —        |

The error counts are the same before and after 9.5, whose outputs are bit for bit the same (F-038). Packed minus float
is **0.00 percentage points**, and all 200 transcripts are equal in text and language, as F-037
found. The Accept's 0.1 holds. Packed minus the engine is +0.05 points, F-031's figure.

### Tests

Without `models/` and without the engine:

- `bench`: `Summarise` (median and nearest-rank p90), `Samples.Set` (the per-pass real-time factor), the table layout,
  `Render` with and without results, including the Accept's verdict at +0.00 and at +0.12 points, and the JSON round
  trip.
- `bench/cmd/whistle-bench`: the 32 committed clips with their digests and transcripts, the 30 s tiling, and the
  LibriSpeech fixtures under the rule, every utterance reproducing `manifest.json`'s count. Also the command's errors
  and a report without results.
- `BenchmarkTranscribe` (packed and float, median TTFT and tok/s as metrics) skips without the model.

### Re-running

`just bench` re-measures the Go side and re-renders `bench/RESULTS.md` with the committed `engine.json`; that is the
one command after a change to the Go side. `just oracle-bench` is only needed when the engine, the machine or the clips
change; it was re-run after 9.5 only because the machine was quieter.

### linux/amd64 (12.3, 2026-10-12)

[`bench/RESULTS-linux-amd64.md`](../bench/RESULTS-linux-amd64.md), from `bench/results/linux-amd64/`. The machine is
2× Intel Xeon Gold 5520+, 80 vCPUs, in a KVM guest without cpufreq, so there is no governor to record. It is a shared
server; the load average was 6–14 during the runs. The Go side ran with `GOMAXPROCS` 16 (`-procs 16`), and the engine
was 3.2.0's `linux-x86_64` build. `whistle doctor` there reports "AVX2: CQ LUT GEMV; AVX2+FMA: dot, axpy, softmax
exp; GEMM in Go".

| median over the 32 clips | Go packed | Go float | engine library | Go packed, `GOMAXPROCS` 4 |
| ------------------------ | --------: | -------: | -------------: | ------------------------: |
| TTFT (ms)                |      47.3 |     70.7 |          160.9 |                      51.4 |
| decode (tok/s)           |       547 |      130 |            523 |                       363 |
| wall (ms)                |      96.5 |    282.6 |          217.1 |                     124.2 |

The last column is a separate `-procs 4 -wer=false` run, not committed, with a load average of 46 → 11. It answers
whether Go wins only on threads, since the engine uses at most 4.

- **Here Go is ahead to the first token.**
  - The engine library's median TTFT is 160.9 ms against Go's 47.3 ms (0.29×), and 51.4 ms with 4 workers (0.32×).
  - On the 30 s clip, it is 727.2 against 159.6 ms.
  - On the M5 Pro the engine leads, 6.9 against 18.8 ms. Its TTFT here is 23× its M5 value; Go's is 2.5×.
  - Why the engine's x86 build is that much slower is not visible from outside, since the engine is closed. Plausible
    causes are less tuned x86 kernels than NEON, or its "fast cores, at most 4" choice on an 80-vCPU guest. Neither
    was investigated.
- **Decode** is even at 16 workers (547 against 523 tok/s, 1.05×). At 4 workers the engine leads (363 tok/s, 0.69×).
- **The float path shows 12.3's missing GEMM.**
  - On amd64 the float matrices run through the pure-Go `tensor.MatMulTransB`: float decodes at 0.24× of packed, on
    the M5 at 0.56×.
  - The packed path is the CQ LUT GEMV, which has AVX2.
- **Memory:** a 30 s run peaks at 99.8 MiB RSS, against 52.2 MiB for the engine CLI.
- **The engine's own transcripts differ by platform.**
  - Its `linux-x86_64` CLI gives the committed (macos-arm64) fixtures' text on 29/32 clips.
  - Go gives 31/32 on both platforms.
- **Parity on amd64 is the same as on arm64.** These are the numbers 12.3's GEMM change has to hold:
  - `TestTranscribeMatchesEngine`: 31/32.
  - `TestEmbedMatchesEngine`: mean cosine 0.999828.
  - Keywords at the default bias (`TestProbeFold`): 47/53 cases, 18/32 sets and 29/31 variants.
  - The depth ladder 2–7 (`just depth-probe`): 24 + 27 + 29 + 29 + 28 + 24 = 161/192.
  - The stream decisions (`just stream-parity`): 797/930.
  - The LibriSpeech WER (`just eval-librispeech`): 5.35 %, against the engine's 5.30 %.

### Open

- **TTFT**, 2.8× the engine's on the clips after 9.5: F-038's open items (the decoder's cross-attention memory, the
  allocations per block, the single-threaded log-mel front end).
- **Decode**, 0.71× the engine's rate.
- **Peak RSS** of a 30 s run, 1.5× the engine CLI's after 9.5: F-037's remaining levers (the int8 cross-attention
  memory of 9.3, `GOMEMLIMIT`, the lut copy).
- **Noise.** The harness alternates the modes within a run, but the Go and the engine runs follow each other, so their
  ratio carries the load difference between those minutes. The run after 9.5 was on a quieter machine (load 2–6) than
  the first (5–8).

## F-041 — The engine's int8 numerics: where it quantises, an int8 KV cache, and what that costs (9.3) (2026-10-11)

**Question.** `whistle.cact` was trained for `act_bits 8` and `kv_bits 8` (PLAN.md's reference card), and the engine runs
W2A8: 2-bit weights, which the packed path already multiplies (F-037), 8-bit activations and an 8-bit KV cache. We run
float32 activations and a float32 cache. Do int8 activations and an int8 cache move us closer to the engine, where does
the engine apply them, and what do they cost or save?

### Upstream's rule

`needle/model/quantize.py` (pin `af2654dffa`) has one quantiser, `fake_quant(x, group_size = x.shape[-1], bits = 8)`:
s = max|x| / 127 per row (1 for a zero row), q = clip(round(x / s), −128, 127) with `jnp.round` (half to even), x ≈
s · q, all in float32. `architecture.py` applies it with `quant` set:

- `_aq` (`fake_quant_act`): the attention's input before q/k/v/gate, the gated attention output before `out_proj`, the
  engram rows before their projections, the mHC heads' input `rms_unit(stream)` and the head's input.
- `maybe_quant_query`, per head after the norm and RoPE, "matching the native attention kernel".
- `maybe_quant_kv`, keys and values per head after the norm and RoPE, "Per-head symmetric A8 KV quantization matching
  the native cache".

The HadamardMLP has none. Whistle's stem, conv module and cross-attention are not in the open code.
`needle.QuantizeRow` is this rule; `TestQuantizeRow` pins the scale, the ties and the zero row.

### Measuring "closer to the engine"

A transcript is a coarse instrument: on the 200 LibriSpeech utterances, every site we tried moved the count of equal
texts within 177–184 and the WER within 220–224 errors, noise either way. Two engine outputs are continuous:

- the encoder: `needle_embed`'s rows for the 32 corpus clips, 2205 frames;
- the decoder: the engine's word probabilities (three decimals) for 7383 words, the 32 clips, the 200 LibriSpeech
  utterances and the 192 depth-ladder cases (F-025). Our side teacher-forces the engine's own tokens through the
  timestamp pass (`timeWords`), so every word pairs.

Let a be our output with int8 at one site, b ours in float32 and e the engine's. Then exactly |a − e|² = |b − e|² +
|a − b|² − 2⟨a − b, e − b⟩. With S = |a − b|², the share ρ = ⟨a − b, e − b⟩ / S is the part of our int8 step that the
engine takes too: 1 if it quantises the same way at that site (the distance falls by S), 0 for an unrelated
perturbation (the distance grows by S), and below 1 where the engine quantises the same way but its values differ from
ours by a sizeable part of the int8 step, so some roundings go the other way. The encoder uses unit-norm frame rows, the
decoder the probabilities and their logits (clipped to [0.0005, 0.9995]). The standard error is the ratio estimator's
over clips or utterances, so a site with a small S cannot be read. Throwaway tests computed all of this; none is
committed.

### Where the engine quantises

Encoder, each site alone (S as the median per-frame 1 − cos against our float32; float32 itself is at a median 1 − cos
of 7.93e-5 to the engine):

| site, per row of                              | S       | ρ               | verdict           |
| --------------------------------------------- | ------- | --------------- | ----------------- |
| mHC input `rms_unit(stream)`, 2048            | 1.14e-5 | **1.08 ± 0.13** | the engine's rule |
| keys per head, 48                             | 6.2e-6  | **0.27 ± 0.03** | partial           |
| values per head, 64                           | 2.8e-6  | **1.15 ± 0.41** | the engine's rule |
| queries per head, 48                          | 1.5e-6  | 1.5 ± 1.4       | too small to read |
| attention input (q/k/v/gate projections), 512 | 8.3e-6  | −0.72 ± 0.27    | not the engine's  |
| gated output before `out_proj`, 512           | 3.3e-5  | 0.04 ± 0.12     | not the engine's  |
| conv module `pw1`/`pw2` inputs, 512           | 2.2e-4  | −0.01 ± 0.10    | not the engine's  |
| stem pointwise inputs, 128                    | 1.9e-6  | 1.2 ± 0.9       | too small to read |
| `stem.out` input, 1280                        | 2.9e-6  | −1.3 ± 0.6      | not the engine's  |

Controls: the keys on a grid 1 % coarser (s = max|x| / 125.7) give ρ −0.09 ± 0.04 and per token instead of per head
0.41 ± 0.10, so the engine quantises the keys on exactly this grid, per head; before RoPE gives 0.47 ± 0.19,
indistinguishable from after, so we keep upstream's place. The partial ρ fits a key that differs from the engine's by
about half an int8 step (F-020's encoder gap); the mHC input, whose step is wide against its outliers, keeps nearly every
rounding the same. At the three CQ inputs that the engine leaves alone, other schemes do not help either: per group of
128, per group in the rotated (FWHT) domain, per row in the rotated domain and asymmetric per row all give ρ between
−0.1 and 0.15 there, and bfloat16 rounding at all three −0.09. Whatever the engine's GEMMs read, it is not one of these
int8 forms.

Decoder, each site alone with the encoder in float32, on the 7383 words (float32 itself at a mean squared probability
gap of 3.63e-4):

| site                                | S      | ρ (probability) | ρ (logit)       | verdict           |
| ----------------------------------- | ------ | --------------- | --------------- | ----------------- |
| mHC input                           | 2.2e-6 | 1.13 ± 0.22     | **0.87 ± 0.13** | the engine's rule |
| self-attention keys per head        | 2.2e-5 | 0.83 ± 0.08     | **0.82 ± 0.06** | the engine's rule |
| self-attention queries per head     | 7.2e-6 | 0.82 ± 0.17     | **0.78 ± 0.11** | the engine's rule |
| self-attention values per head      | 1.1e-6 | 0.97 ± 0.52     | **1.09 ± 0.26** | the engine's rule |
| attention input                     | 7.4e-7 | 0.59 ± 0.54     | 0.08 ± 0.27     | not read          |
| gated output before `out_proj`      | 1.8e-6 | −0.13 ± 0.42    | −0.03 ± 0.19    | not the engine's  |
| cross-attention projections' inputs | 3.4e-6 | 0.14 ± 0.20     | 0.12 ± 0.12     | not the engine's  |
| cross-attention memory k/v per head | 6.8e-7 | 0.70 ± 0.45     | 0.00 ± 0.26     | not read          |
| cross-attention queries per head    | 1.2e-7 | 0.3 ± 1.2       | −0.2 ± 0.8      | not read          |
| engram rows                         | 9.3e-8 | −0.7 ± 1.7      | 0.3 ± 0.8       | not read          |
| head input                          | 1.1e-6 | −0.31 ± 0.36    | 0.18 ± 0.22     | not the engine's  |

So the engine follows upstream's rule at the mHC heads and in the attention kernel (queries, keys, values per head) and
not at the other CQ products' inputs, where upstream trained with `_aq` too. Applying upstream's rule everywhere moves us
away from the engine: decoder gap 3.69e-4 against 3.63e-4 (ρ 0.44 ± 0.06 for the mix), 28/32 corpus clips equal,
LibriSpeech 178/200.

**One fragile frame.** With the keys and another site on, frame 14 of `fleurs-fr_fr-10174413140250646001` (norm 8, not
silence) falls from cosine 0.9947 (float32) to 0.63–0.95 against the engine, depending on the combination; the bulk
moves closer at the same time (median 1 − cos 7.93e-5 → 7.29e-5 with everything below). Our pre-quant values differ
from the engine's by part of a step, and here one flipped rounding is amplified. The means that F-020's tests use
(clip means, then their mean) feel such a frame, the medians do not.

### What the flag does

`whistle.WithInt8()` loads the model with `needle.Int8{Act: true, KV: true}`; the default is unchanged. The sites are the
engine's, as measured above:

- `Act`: the mHC φ products quantise each row of `rms_unit(stream)` (`Linear.WithInt8Input`, which `ApplyShared` turns
  into one quantised copy for the three φ), and every attention, the encoder's, the decoder's and the cross-attention,
  quantises its queries per head.
- `KV`: every attention quantises its keys and values per head and position. The decoder's self-attention cache
  (`KVCache`) and the cross-attention memory keep them as `needle.Int8Rows`: int8 codes plus a float32 scale per head and
  row, appended and forked like the float32 slices.
- A decoding step attends with `needle.AttendInt8`: per head, the int8 query against the int8 keys is an exact int32 dot
  product times the two scales, then the softmax, then the weights times the value scales against the value codes, key
  after key. A block of 8 or more rows (the timestamp pass) runs the existing GEMMs on the dequantised cache. The
  encoder, which has no cache, uses the values the codes stand for in float32.
- Kept float32, since the engine does not quantise there: the stem, the attention's and the conv module's projections,
  the cross-attention's projections, the engram and the head.

The encoder and the decoder get it through `encoder.LoadInt8` and `decoder.LoadInt8`; `Load` stays the float32 path.
Nothing else changes with the flag off. Against `main` (`f26e3d4`), a throwaway test hashed the `Embed` rows of the 32
clips and the 30 s clip and the text, language, word times and word probabilities of `Transcribe` on the same 33
inputs with default options, word timestamps, `Depth` 4 and two keywords, packed and float: the four hashes are equal.

**Shared rotation** (F-038's open item, independent of the flag). `cross.project` now computes the memory's keys and
values with one `needle.ApplyShared`, so the packed k and v projections share the encoder output's rotation and tables.
`TestProjectSharesRotation` (`internal/decoder`, random packed and float32 weights, 1, 13 and 375 frames) finds the same
bits as one `Apply` each, and the hashes above did not move.

### Kernels

`dotsI8` (the int32 logits) and `axpysI8` (the weighted value codes) have NEON (`int8kernel_arm64.s`) and AVX2
(`int8kernel_amd64.s`, chosen at run time) assembly for widths that are multiples of 16, and Go references for the rest
and for `purego`:

- NEON dots: `SMULL`/`SMULL2` to int16 and `SADALP` into int32 lanes, four keys at once, reduced by `ADDP`. AVX2:
  `VPMOVSXBW` and `VPMADDWD`. The sums are exact, so every kernel gives the same int32.
- Axpys: sign-extend, convert, `FMUL` by the weight, `FADD` into 16 outputs kept in registers, key after key. No FMA, so
  they equal the Go reference bit for bit.
- `TestInt8KernelsMatchGeneric` covers widths 8 to 128, 1 to 375 keys and padded strides, on arm64, under `purego` and
  on linux/amd64 (AVX2, Docker emulation). Go 1.25 lacks the NEON integer and vector float instructions, so they are
  `WORD`-encoded, each checked against `go tool objdump`.
- `fakeQuant` and `Int8Rows.Append` run their rows on `tensor.Parallel`; each row is its own, so the result does not
  depend on the worker count.

One decoding step's cross-attention for one layer, 8 heads over 375 frames (`BenchmarkCrossAttendStep`, M5 Pro, one
core): float32 (`cross.go`'s dot product and axpy per frame) 55.0 µs, int8 in Go 124 µs, int8 with NEON **23.9 µs**
(2.3× faster than float32). `QuantizeRow` takes 0.83 ns per value (47 ns for a head of 48, 1.69 µs for the mHC's 2048).

### Parity

All on the M5 Pro, packed weights; "int8" is `WithInt8()`. The model-gated tests ran with `WHISTLE_TEST_INT8=1`, which
makes their `loadModel` pass `WithInt8()`.

| check                                                     | float32 (default)   | int8                  | engine |
| --------------------------------------------------------- | ------------------- | --------------------- | ------ |
| `TestTranscribeMatchesEngine`, 32 clips equal             | 31/32               | 30/32                 | —      |
| depth ladder 2–7, 192 cases equal (throwaway driver)      | 161/192             | **174/192**           | —      |
| keyword cases, 53 equal (throwaway driver)                | 46/53               | 46/53                 | —      |
| `TestEmbedMatchesEngine`, mean / worst clip               | 0.999827 / 0.999034 | 0.999719 / 0.994891   | 1      |
| `Embed`, median frame 1 − cos                             | 7.93e-5             | 7.29e-5               | 0      |
| teacher-forced word probabilities, mean squared gap, 7383 | 3.63e-4             | **3.33e-4** (ρ 0.91)  | 0      |
| teacher-forced word times equal to the engine's, 7383     | 6984                | **7044**              | —      |
| `just stream-parity`, calls with an equal decision        | 797/930             | **812/930**           | —      |
| stream, final text equal to the engine's stream           | 98/108              | 100/108               | —      |
| `just eval-librispeech`, WER (200 utterances)             | 223/4167 = 5.35 %   | 222/4167 = **5.33 %** | 5.30 % |
| LibriSpeech, text equal to the engine's                   | 181/200             | 182/200               | —      |
| `TestTranscribeWordsMatchEngine`, both times within 80 ms | 97.4 %              | 96.7 %                | —      |

The new miss on the 32 clips is `fleurs-pl_pl-10013271501230525766`, the corpus' second Polish clip: "W większości
mniejszych wyzwań" against the engine's "W większość mniejszych wyzptów". Its depth-ladder transcripts differ from the
engine's at depths 2–5 and 7 in float32 already. Under int8 the 30/32 falls below `TestTranscribeMatchesEngine`'s 95 %
floor (PLAN.md 6.6), and the word times miss their 97 % floor by 0.3 points; both floors were set on float32. So under
`WHISTLE_TEST_INT8=1` the two tests hold int8 expectations instead (PR #36 review): the misses `int8Misses` (the two
Polish clips) and at least 93 % of the clips, and the word times at least 96.5 %. The default run keeps the float32
floors. The keyword cases that differ are not the same: int8 gains
`fr-cristaux-renale-declare-list` and `pl-wysp-missed` and loses `pl-luksusowe-missed` and `pl-francja-present`. On
linux/amd64 (AVX2, emulated) `TestInt8MatchesEngine` gives the same 30/32 and `Embed` 0.999706 / 0.994307.

So int8 mirrors the engine in the bulk: the word probabilities' gap falls 8 %, 13 more depth cases, 15 more stream
decisions and one more LibriSpeech text equal the engine's, and the WER moves to 0.03 points from the engine's. It also
moves a few fragile outputs the other way (one corpus clip, one encoder frame, two keyword cases).

### Speed

`TestSpeedProbe` (now packed, float32 and int8 rotating in one process), M5 Pro, load average 5–8, two runs:

| median           | packed            | float32         | int8              | int8 vs packed |
| ---------------- | ----------------- | --------------- | ----------------- | -------------- |
| TTFT, 32 clips   | 19.2 / 19.6 ms    | 22.2 / 22.3 ms  | 20.9 / 21.0 ms    | 7–9 % slower   |
| decode, 32 clips | 1057 / 1010 tok/s | 601 / 589 tok/s | 1121 / 1100 tok/s | 6–9 % faster   |
| TTFT, 30 s clip  | 61 / 62 ms        | 68 / 68 ms      | 65 / 65 ms        | 5–7 % slower   |
| `Load`           | 13.3 / 10.8 ms    | 157 / 151 ms    | 8.4 / 8.5 ms      | —              |

On one kernel worker (`WHISTLE_SPEED_WORKERS=1`): TTFT 54.2 / 56.0 / 56.8 ms, decode 1081 / 610 / 1179 tok/s, 30 s clip
284 / 288 / 295 ms. Decoding gains because the int8 cross-attention is faster than the float32 one, and the cache reads
a quarter of the bytes. The time to the first token pays for the quantisation: the encoder's queries, keys, values
and mHC inputs, 8 M values for 30 s, and the memory's 2.7 M. The first version, with the Go kernels, decoded at 902
against 1029 tok/s.

### Memory

`just mem-probe` (a fresh child per mode, now with int8), two runs, MiB:

| MiB                               | packed        | float32       | int8             |
| --------------------------------- | ------------- | ------------- | ---------------- |
| process after `Load`              | 41.6 / 41.5   | 242.7 / 242.6 | 41.6 / 41.5      |
| peak over a 30 s `Transcribe`     | 107.9 / 107.2 | 495.2 / 494.4 | **103.6 / 99.7** |
| model (live heap + `.cact` pages) | 35.0 / 34.9   | 230.8 / 231.6 | 34.7 / 34.7      |
| Go mapped at the end              | 90.4 / 89.9   | 476.1 / 474.8 | 69.1 / 71.8      |

The 30 s cross-attention memory shrinks from 10.25 MiB (8 layers × 375 frames × 896 float32) to 2.75 MiB (codes and
scales); the 5 beams' self-attention caches by the same factor. The peak falls by 4–8 MiB, less than the memory's
7.5 MiB, because the encoder's transients and `GOGC` headroom (F-038) set it before the decoder runs.

### Recommendation

Keep the flag off by default for now; turning it on is a separate decision. For it: it mirrors the engine's numerics
where the engine applies them (ρ near 1 at the mHC and in the attention kernel); the word probabilities, the word times,
the median encoder frame and the depth, stream and LibriSpeech counts move toward the engine; decoding is 6–9 % faster
and the 30 s peak 4–8 MiB lower. Against it: TTFT is 5–9 % slower, `Embed`'s clip means fall (one fragile frame), one
more corpus clip differs, and two float32-calibrated floors (`TestTranscribeMatchesEngine` at 95 %, the word times at
97 %) fail by one clip and 0.3 points. If the default changes, those floors need int8 values, and `whistle-bench` a
column for it.

### Open

- **The engine's other activations.** It does not quantise the CQ products' inputs per row, per group or in the rotated
  domain, nor round them to bfloat16; float16 (a lever of 5.6e-8) is too small to read with these instruments.
- **The cross-attention, the engram and the head** are below the instruments' resolution (except the head and the
  cross-attention projections, which the engine does not quantise). The flag quantises the cross-attention's queries,
  keys and values by analogy with the self-attention.
- **Faster quantisation.** `QuantizeRow` is Go (0.83 ns per value); a NEON/AVX2 version would recover most of the TTFT
  cost. An `SDOT` dot kernel (FEAT_DotProd) would replace 4 instructions per 16 bytes by one.

## F-042 — `de-paare-missed`: no finished-slot rule, the keyword bonus is too strong (PLAN.md Phase 10) (2026-10-11)

**Question.** F-029's one keyword search miss: with the keyword `Paare` on `fleurs-de_de-10308608197319017935` the
engine writes "Diese Paare können sich für einen **Adduktionsplan** für ihr Baby entscheiden.", ours "… Adoptionsplan
…", although the engine's text ranks 0.0688 nats per text token higher under our model and `DefaultBias`. Which rule of
the engine's finished-hypothesis bookkeeping keeps the right hypothesis, without losing anything elsewhere?

### The step trace

`Default()` with `DefaultBias()`: every live hypothesis carries the +7 of `Paare` (3 tokens × 2 + 1) from text token 6
on. At step 35 the five finished slots fill in one step, one step before the leader ends:

| finished at step 35                                          | L   | Σ log p  | ranking score |
| ------------------------------------------------------------ | --- | -------- | ------------- |
| "… Adoptionsplan … entscheiden." (`▁Ad op tions plan`)       | 35  | −9.1950  | **−0.0627**   |
| "… Adoptionsplan … entscheiden." (`▁A do ptions plan`)       | 35  | −9.1960  | −0.0627       |
| "… Adduptionsplan … entscheiden."                            | 35  | −9.2659  | −0.0647       |
| "… Adduktionsplan … entscheiden" (EOS without the full stop) | 35  | −15.1254 | −0.2322       |
| "… Aduktionsplan … entscheiden" (EOS without the full stop)  | 35  | −16.1916 | −0.2626       |

The live leader, "… Adduktionsplan … entscheiden." (Σ log p −6.7803, score +0.0061 before its EOS), never finishes. Two
of the five are the same text in two spellings, and two are weak endings without the full stop that Whisper's rule
admits because they rank above the last live hypothesis.

### Finished-slot rules (`finRule`, a test-only switch in `Options`)

`TestProbeFinish` (`internal/search/finish_probe_test.go`, tag `oracle`) runs the real search with each rule on every
oracle set the search answers for: the 53 keyword cases and the 32 keyword sets below under `DefaultBias`, the 32
fixture clips, and the 192 clips × depths 2–7. Won and lost are against Whisper's rule.

| rule                                          | `de-paare-missed` | keyword cases | keyword sets | clips | depths 2–7 | lost (besides sets)                      |
| --------------------------------------------- | ----------------- | ------------- | ------------ | ----- | ---------- | ---------------------------------------- |
| **Whisper: 5 slots** (`Default`)              | differs           | **46/53**     | 17/32        | 31/32 | 161/192    | —                                        |
| a finished text takes one slot                | equal             | 46/53         | 18/32        | 31/32 | 161/192    | `de-diese-paare-phrase`                  |
| one live hypothesis per text                  | differs           | 45/53         | 9/32         | 31/32 | 161/192    | `de-diese-paare-phrase`                  |
| both                                          | differs           | 45/53         | 9/32         | 31/32 | 161/192    | `de-diese-paare-phrase`                  |
| 6 slots (patience 1.2)                        | equal             | 44/53         | 18/32        | 30/32 | 158/192    | 3 keyword cases, 1 clip, 6 depth cases   |
| 7 slots (patience 1.4)                        | equal             | 44/53         | 18/32        | 30/32 | 153/192    | 3 keyword cases, 1 clip, 11 depth cases  |
| 10 slots (patience 2)                         | equal             | 44/53         | 18/32        | 29/32 | 153/192    | 3 keyword cases, 2 clips, 11 depth cases |
| the final pick weighs the live ones too       | equal             | 44/53         | 18/32        | 29/32 | 153/192    | 3 keyword cases, 2 clips, 11 depth cases |
| only EOS among the step's best 5 count        | equal             | 44/53         | 18/32        | 30/32 | 158/192    | 3 keyword cases, 1 clip, 6 depth cases   |
| 5 slots and the best finished ≥ the best live | equal             | 44/53         | 18/32        | 29/32 | 153/192    | 3 keyword cases, 2 clips, 11 depth cases |

_One live hypothesis per text_ draws 4 × (5 + 1) candidates per parent, EOS still only among a parent's best 6.
Whisper's 6 hold 5 live texts unless two tokens of one parent spell the same text (such as byte pieces that decode to
U+FFFD). Such pairs occur, 1842 over the 10 036 steps of the runs, but the beam never fell short of 5. The first version
drew 6 (PR #39 review), and its rows are the same.

Every rule that makes `de-paare-missed` equal loses its mirror, `de-diese-paare-phrase`, where the engine writes the
early "Adoptionsplan". Under our search the two cases have the same beam from the plan word on: the same hypotheses
with the same Σ log p, the bonus +7 against +13 for all of them. A rule that ranks within a step cannot tell a bonus
that every hypothesis shares; the final pick by Σ / L could, but "Adoptionsplan" (L 35) beats the leader (L 36) only
from a shared bonus of 93 nats on. So no finished-slot rule separates the two cases; the engine's beam differs from
ours earlier.

### More engine evidence: 32 keyword sets on the clip

`oracle kwsets` (`tools/oracle/kwsets.go`, `testdata/oracle/kwsets/manifest.json`) records the engine's text on the
clip with 32 keyword files that vary the shared bonus. The engine's choice follows the strength of the bonus on `Paare`
against `Bare`, not its size:

| keywords                                                                                                            | engine                     |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| none, `Baby`, `können`, `Diese`, `einen`, `Paar`, `Adduktionsplan`, `für ihr Baby`, `Baby entscheiden`              | Bare … **Adduktionsplan**  |
| `Paare`, `paare`, `PAARE`, `Paare`+`Paare`, `Paare` with `Baby`, `entscheiden`, `können`, `Diese` or `für ihr Baby` | Paare … **Adduktionsplan** |
| `Diese Paare`, `diese Paare`, `Paare können`, `Paare können sich`, `Diese Paare können`, `Diese Paare`+`Baby`       | Paare … **Adoptionsplan**  |
| `Bare`, `Diese Bare`, `Paare`+`Bare`, `Bare`+`Paare`                                                                | Bare … **Adoptionsplan**   |
| `Pare`                                                                                                              | Pare … Adoptionsplan       |

Our trace says why: "Diese Paare" sits 4.47 nats below "Diese Bare" under the model. Our +7 puts the `Paare`
hypotheses 2.5 nats ahead, so by the plan word (step 20) they fill the beam, and the "Adoptionsplan" spellings live in
the room the pruned `Bare` hypotheses leave. In the engine, the single word `Paare` wins the text but leaves the `Bare`
hypotheses in the beam ("Adduktionsplan"), while a phrase, whose extra words only the `Paare` hypotheses earn, or a bonus
on `Bare` itself prunes the other spelling ("Adoptionsplan"). The engine's per-word bonus on `Paare` is weaker than ours
relative to the 4.47-nat gap. `DefaultBias` reproduces only 17 of the 32 sets.

### Bias variants (`TestProbeKeywordSets`)

273 biases on the memo of F-029's probe (exact for biased searches), scored on the 53 cases and the 32 sets: per-token
weight 1.5 … 2.5 × completion 0 … 2 × revocation ½ or 1, per-character weight 0.375 … 1.5, fewer spellings, and three
completion structures added for this probe as test-only `Bias` fields:

- _transient_: the completion bonus lasts one token, the next token (EOS included) takes it back;
- _transient + echo_: the completion also earns the keyword's own token bonus for that token, then loses it, as
  icefall's / sherpa-onnx's `ContextGraph` adds `output_score`;
- _once_: a keyword earns its bonus once per text, in any of its forms (space or bare, either case); a repeat's tokens
  are taken back when it completes. The first version counted each spelling on its own (PR #39 review); counting
  per keyword changes none of the 273 rows below.

| family (variants)            | make `de-paare-missed` equal | best that does (cases, sets, lost)                                                                     | best overall (cases, sets, lost)          |
| ---------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| per token, revocation 1 (71) | 31                           | w 2.1, completion 0.25: **47/53**, 26/32, `pl-francja-present`                                         | w 1.8, completion 1.5: 47/53, 18/32, none |
| per token, revocation ½ (63) | 28                           | w 1.5, completion 2: 44/53, 29/32, 4 cases                                                             | 45/53, 16/32, 2 cases                     |
| per character (21)           | 0                            | —                                                                                                      | w 1.25: 42/53, 17/32, 4 cases             |
| fewer spellings (18)         | 9                            | without lower case, w 2.1, completion 0.25: 46/53, 27/32, `de-fast-jeder-phrase`, `pl-francja-present` | 45/53, 18/32, 1 case                      |
| transient completion (40)    | 32                           | w 2, completion 0: 46/53, 26/32, `it-accomunano-missed`, `pl-francja-present`                          | the same                                  |
| transient + echo (40)        | 32                           | w 2: 46/53, 26/32, the same two                                                                        | the same                                  |
| once per keyword (20)        | 10                           | w 2, completion 0: 46/53, 26/32, the same two                                                          | w 2, completion 1: 46/53, 17/32, none     |

142 of the 273 biases make `de-paare-missed` equal; none loses no keyword case. The best,
`w=2.1 revoke=1 complete=0.25`, wins `de-paare-missed` and `it-montagne-distractor` (F-029's 0.0036 near-tie) and loses
`pl-francja-present`, where the engine changes "wspomnieżone" to "wspierzone" before the keyword: under that bias our
text ranks above the engine's by 0.0033 nats per token, a near-tie. `de-paare-missed` itself is a search miss of 0.0688
per token under `DefaultBias`. It also reproduces 26 of the 32 sets (`DefaultBias` 17). By the rule of this item (a
winner must lose nothing) it is not kept; whether a 0.0033 near-tie may be traded for a 0.0688 search miss and nine
engine sets is the user's call.

The sets fit best at **w 1.5 per token** (sherpa-onnx's default hotword score): completion 0.75 … 2 reproduces 31 of
32 sets, all but `PAARE`, and makes `de-paare-missed` equal, but loses `en-cowleys-missed`, `en-cowleys-lower-missed`,
`es-luchadoras-distractor` and `it-accomunano-missed` (44/53). The English, Spanish and Italian cases want a stronger
bonus than the German clip, so a single per-token weight does not describe the engine; the per-character weights fit
worse still.

### Verdict

Negative. The miss is not in the finished-hypothesis bookkeeping: no rule there separates `de-paare-missed` from
`de-diese-paare-phrase`, and Whisper's 5 slots stay. It is in the bias: the engine's bonus for a single keyword word is
weaker relative to the competing spelling than our 2 nats per token + 1, while its phrases and its other clips need
ours. `Default()` and `DefaultBias()` are unchanged, so every guard number is unchanged (PLAN.md Phase 10, the PR).

**Open.**

- **Case-insensitive matching.** `Paare`, `paare` and `PAARE` give the engine the same text; we match the first
  letter's two cases only (`FormCapital`, `FormLower`), so `PAARE` matches nothing. A matcher over case-folded
  characters would also decide how a bonus is shared across the tokens of a word, which may be what the English and
  Romance cases want differently from the German one.
- **Loops.** With `Diese Paare können` our search repeats the phrase nine times; the engine does not. _once_ does not
  stop it at w 2 (the partial match of each repeat still earns its tokens' bonus until it completes); w 1.5 does.
- **The capitalised bare spelling.** With `für ihr Baby` our search skips the first half of the clip to start with
  "Für ihr Baby" at w 2; the engine does not, nor do we at w 1.5.
- **A gain without `de-paare-missed`.** `w=1.8 revoke=1 complete=1.5` wins `it-montagne-distractor` and loses nothing
  (47/53, 18/32 sets). It is outside this item and was not run on the guard set.

**Kept for the probes.** `Options.fin` (`finRule`) and the `Bias` fields `transient`, `echo` and `once` are
unexported; their zero values are today's rules, and `TestFinRules` and `TestKeywordProbeAxes` pin them.

## F-043 — js/wasm: the same transcripts without FMA or threads, and where the time goes (Phase 10) (2026-10-11)

**Question.** The web demo (`cmd/whistle-wasm`, `web/`) runs the library as `GOOS=js GOARCH=wasm`: the pure-Go kernels
(`*_noasm.go`), one thread, no SIMD, and no fused multiply-add, since WebAssembly 1.0 has none and Go's wasm back end
emits none. On arm64, Go fuses `x*y + z` even in pure Go, so the wasm float32 sums round differently from every native
build. Do the transcripts survive that, and how fast is it?

### Parity

- The whole `whistle` package passes under `go_js_wasm_exec` (Node 24.12) without `-short`, in 657 s: the 32-clip
  engine parity (`TestTranscribeMatchesEngine`, the same single miss as native, F-026), the word-time floors (F-033),
  the stream parity floors (F-036), `Embed`'s cosines and the depth and keyword tests. So does every other package but
  `cmd/whistle` and `internal/server`, whose tests need sockets (wasm's `net` is a stub) and run out of wasm32's 4 GB
  after many model loads; neither ships to the browser.
- On 4 clips (en ×2, de, fr) the page's path (the worker, `decodeWAV`, `transcribe` with word timestamps) gives
  `whistle transcribe --json --word-timestamps`'s JSON byte for byte once `ttft_ms` and `decode_tps` are dropped,
  probabilities included; a 1 s file stream gives `whistle stream --json`'s calls byte for byte without `pass_ms`
  (`just wasm-test`, `cmd/whistle-wasm/node/headless.mjs`). In Chromium the page gave the same JSON for the German clip.
- The one visible effect of the missing FMA: in `TestInt8Memory` a projected value fell within float32 rounding of a
  quantisation tie (x / s = −20.4999…), rounded the other way than natively and landed a hair over the test's
  "half a step" bound; the bound now allows the float32 rounding of x / s and s · q. Nothing else moved.

### Speed

`TestEngineTranscribe` (the 4 clips, beam search at full depth, word timestamps on), Apple M5 Pro, idle machine:

| build                           | TTFT ms   | decode tok/s |
| ------------------------------- | --------- | ------------ |
| arm64, NEON, all cores          | 13 – 21   | 885 – 1121   |
| arm64, NEON, `GOMAXPROCS=1`     | 24 – 64   | 251 – 292    |
| arm64, `purego`, all cores      | 48 – 79   | 290 – 307    |
| arm64, `purego`, `GOMAXPROCS=1` | 101 – 265 | 64 – 68      |
| js/wasm, Node 24.12 (V8)        | 284 – 770 | 24 – 27      |
| js/wasm, Chromium 153, the page | 913 (de)  | 20.5 (de)    |

So wasm is about 2.6× slower than pure Go on one core, which is itself 4× slower than NEON on one core; the
cores are worth another 3.5–4.5×. A 6.9 s clip takes 2.4 s under Node and 3.0 s in the page, faster than real time.
The model loads from bytes in 15–45 ms.

A V8 profile of two clips (unstripped build): the packed CQ kernels take two thirds of the time
(`cq.(*Matrix).lookupNibble` 28.5 %, `nibbleTables` 15.1 %, `lookupNibble4` 11.6 %, `nibbleTables4` 4.7 %, `rotate`
5.8 %), then `tensor.dotF32Generic` 6.0 % and `math.exp` with its helpers 6 %. `WithFloatWeights` is no way
out: dense float32 decodes at 22–24 against 24–25 tok/s packed and needs 221 MB instead of 17 MB, so the demo stays
packed.

### Open

- **Threads.** Go's wasm port runs one thread; several workers, each with its own model, could run beams or encoder
  blocks in parallel, at 17 MB a copy.
- **SIMD.** Go's wasm back end has no SIMD (`GOWASM` knows only `satconv` and `signext`) and no assembly; the LUT
  kernels would need a hand-written wasm module through JS, outside pure Go.

## F-044 — `de-paare-missed`, second round: the engine folds case on tokens, a bonus no token count explains (PLAN.md Phase 10) (2026-10-11)

**Question.** F-042's open list for the one keyword search miss: does the engine match case-insensitively, and how is a
word's bonus shared across its tokens; which rule stops our phrase loops; why does the engine not jump to the
capitalised bare spelling ("Für ihr Baby"); and does F-042's gain `w=1.8 revoke=1 complete=1.5` hold on the guard
set. The goal is unchanged: `de-paare-missed` equal to the engine without losing anything.

### More engine evidence: 31 keyword variants on 8 clips

`oracle kwvariants` (`tools/oracle/kwvariants.go`, `testdata/oracle/kwvariants/manifest.json`) records the engine
on clips whose keyword cases it already answers, with the keyword written differently:

| variant                                                                                                                                                             | engine                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| capitals: `COWLEY'S`, `cOWLEY'S`, `ADOPTIONSPLAN`, `adoptionsplan`, `VARKENS`, `varkens`, `MUGGEN`, `ALIVIAR`, `Aliviar`, `ACCOMUNANO`, `CRISTAUX`, `WYSP`, `UNCAS` | 13 of 13 the text of the keyword as the case has it           |
| a piece from inside a word: `aare`, `optionsplan`, `doptionsplan`, `owley's`, `arkens`, `comunano`                                                                  | 6 of 6 the text without keywords                              |
| a word start that is a token prefix: `Adop` (`▁Ad op`), `Adoptions` (`▁A do ptions`), `Cowley`, `Varken`, `accomun` (`▁ac com un`)                                  | 5 of 5 change the text, 4 to the full keyword's text          |
| a word start that is no token prefix: `Paa` (`▁P a a` against `▁P a are`), `alivi` (`▁al iv i` against `▁al iv iar`)                                                | 2 of 2 the text without keywords                              |
| short phrases: `können sich`, `für einen`, `some verses`, `el dolor`; `Paare können sich für`                                                                       | the text without keywords; the phrase sets' "… Adoptionsplan" |

So the engine ignores the keyword's case, needs a word start, needs no word end, and matches **token ids**: `Adop`
matches the model's own spelling `▁Ad op tions plan` of "Adoptionsplan" (not `RefTokenizer`'s `▁A do ptions plan`),
while `Paa` and `alivi`, whose characters start the word but whose tokens do not, do nothing. A matcher over characters
would reward them. The open-source sibling engine (`cactus-compute/cactus`, `utils.h`) biases vocabularies by a
context-free logit bias of 5 on every keyword token and a case-insensitive Levenshtein spelling correction; that
would write the keyword's own spelling (`PAARE`, `paare`) into the text, where the engine writes "Paare", so it is
not needle3's mechanism.

### Hypotheses (`TestProbeFold`)

`internal/search/fold_probe_test.go` (tag `oracle`) runs the real search with `Default()` and each bias on the 53
keyword cases, the 32 keyword sets of F-042 and the 31 variants (four grids, 308 biases, `WHISTLE_PROBE_GRID`); won
and lost are against F-029's bias (`DefaultBias` until now). Every axis is an unexported `Bias` field whose zero value
is the old rule:

- `fold=cases`: the spellings also include the keyword lower-cased, and lower-cased with its first letter
  upper-cased, still on token ids (`caseSpellings`);
- `fold=bytes`: an Aho-Corasick automaton over the case-folded bytes of the text, so every tokenisation matches
  (`fold.go`), the bonus of the keyword's own tokens on their last byte (`share=canon`, a token that is only the space
  on the next one) or `Weight` per token of the text (`share=token`) or per character (`share=char`), and with
  `word-end` a completion only before a non-letter;
- `spent`: once a keyword completed, its tokens and its completion earn nothing more;
- `bare-as-written`: the bare spellings only as written.

The summary (grid 4):

| bias                                                    | keyword cases | sets      | variants  | `de-paare-missed` | won / lost                                                                       |
| ------------------------------------------------------- | ------------- | --------- | --------- | ----------------- | -------------------------------------------------------------------------------- |
| F-029: `w=2 revoke=1 complete=1`, all forms             | 46/53         | 17/32     | 20/31     | differs           | —                                                                                |
| + `fold=cases`                                          | 46/53         | 17/32     | 29/31     | differs           | none                                                                             |
| + `spent`                                               | 46/53         | 18/32     | 19/31     | differs           | none (the loop set won, the variant `Cowley` lost)                               |
| + `fold=cases spent`                                    | 46/53         | 18/32     | 28/31     | differs           | none                                                                             |
| `w=1.8 complete=1.5` (F-042's gain)                     | 47/53         | 18/32     | 20/31     | differs           | won `it-montagne-distractor`                                                     |
| **`w=1.8 complete=1.5 fold=cases`** (`DefaultBias` now) | **47/53**     | **18/32** | **29/31** | differs           | won `it-montagne-distractor`                                                     |
| `w=1.8 complete=1.5 fold=cases spent`                   | 47/53         | 19/32     | 28/31     | differs           | won `it-montagne-distractor`                                                     |
| `w=2.15 complete=0 fold=cases spent`                    | 47/53         | 28/32     | 28/31     | **equal**         | won `de-paare-missed`, `it-montagne-distractor`; lost `pl-francja-present`       |
| `w=1.5 complete=2 fold=cases spent`                     | 44/53         | **32/32** | 25/31     | **equal**         | won the same two; lost both `en-cowleys-*`, `es-luchadoras-…`, `it-accomunano-…` |
| `fold=bytes share=canon w=1.8 complete=0`               | 42/53         | 22/32     | 23/31     | equal             | won 3, lost 7                                                                    |
| `fold=bytes share=token w=2.2 complete=0`               | 42/53         | 26/32     | 23/31     | equal             | won 2, lost 6                                                                    |

Per open item:

1. **Case.** `fold=cases` reproduces 9 more variants (all 13 capitals but `WYSP`, which is `pl-wysp-missed`'s text,
   a 0.0125 near-tie of F-029) and makes `paare` and `PAARE` behave like `Paare` (the engine's three are one text;
   ours now too, though all three still differ from the engine with `de-paare-missed`), at no cost anywhere. The
   byte matcher is the wrong unit: 112 rows over three shares, word end or not, 0.375 … 2.5 per token or character,
   best 42/53, and it misses `Paa` and `alivi` exactly as the engine's token matching predicts. (Grid 1's 80 byte
   rows, best 40/53, predate two fixes: the bonus sat on a token's first byte, so a lone `▁` piece earned it anywhere,
   and under `share=token` the start of the text earned a weight; grids 2 and 4 rerun the fixed matcher.) A bonus per
   character on token ids, with `fold=cases spent` (0.75 … 1.25 × completion 0.5 … 2), is no better: 40–42/53. How
   a word's bonus is shared across its tokens is thus as before: `Weight` per token of the keyword's spelling.
2. **Loops.** `spent` stops them: with `Diese Paare können` our text is the engine's ("… Adoptionsplan …") instead of
   nine repeats, and no keyword case changes. But on the variant `Cowley` it makes the engine's text a search miss
   (0.0839 per token): without the repeats' bonus the beam holds other hypotheses, and five weaker endings fill
   Whisper's finished slots one step before "Cowley's. Tell …" ends, the mechanism of F-042. One set won, one
   variant lost: a trade, so `spent` stays out of `DefaultBias`. F-042's `once` did not stop the loops at w 2.
3. **The capitalised bare spelling.** It is the weight, not the form: `w=1.5 complete=2` keeps "Für ihr Baby" off
   the start with every form, and dropping the form costs elsewhere: without `FormCapital` (with `fold=cases`)
   "für ihr Baby" is equal but `diese Paare` is not, the engine's "Diese Paare" at the start needing that same
   spelling; `bare-as-written` loses `VARKENS` and `varkens` (the engine's "Varkens…" at the start); without bare
   forms `de-diese-paare-phrase` and `nl-varkens-missed` are lost (grids 1 and 3).
4. **F-042's gain.** `w=1.8 complete=1.5` wins `it-montagne-distractor` and loses nothing; with `fold=cases` it is
   the new `DefaultBias`: 47/53 cases, 18/32 sets, 29/31 variants, at least F-029's bias on every count. The guard
   set is unchanged: 31/32 clips, 161/192 depth cases, 797/930 stream calls, WER 5.35 %, the word floors (the guard
   tests other than the keyword fixtures run without keywords). `TestKeywordsMatchEngine`'s score floors are
   re-measured (0.1270, −0.1575, 0.0711, 0.1830; the same four texts).

### Why no bias reaches `de-paare-missed` without a loss

The step traces split the conflicting cases by mechanism. `de-paare-missed` is decided by the pruning: the `Paare`
hypotheses carry their whole bonus from text token 6 on, and once it is above about 6.6 nats (3 tokens × 2.2) they
push the "Bare" spellings out of the beam before the plan word (F-042), so the "Adoptionsplan" spellings fill the
finished slots. `it-accomunano-missed` and `pl-francja-present` are decided by the final pick: our text and the
engine's both hold the keyword and differ elsewhere, "gli" against "vi" (2 tokens against 1) and "wspomnieżone"
against "wspierzone" (7 against 4). Under the ranking (Σ log p + bonus) / L a shared bonus B favours the shorter
text by B (1/L₁ − 1/L₂), and the engine writes the shorter one, so they want B large: `accomunano` (4 tokens)
between 8.3 and 8.6, `Francją` (5 tokens) about 11 and more with a completion bonus (kept at w 2/complete 1 = 11,
lost at w 2.2/complete 0 = 11 and w 2.4/complete 0 = 12, kept at w 2.4/complete 0.1), the chaos of its 0.0033
near-tie. With B = n·w + c that leaves w ≈ 2.15–2.2 and c ≈ 0: the window above, which gets `accomunano` but not
`Francją`. The variant `accomun` says the engine's bonus is no
function of the token count at all: its 3 tokens give the engine the shorter "vi" as `accomunano` does, while the 3
tokens of `Paare` must stay below 6.6; ours gives `accomun` 3·1.8 + 1.5 = 6.9 and writes "gli". A bonus per
character would separate them (7 against 5 characters) but fits the cases worse (above).

### Verdict

Negative for `de-paare-missed`, a partial gain elsewhere. The engine folds case and matches token ids; `fold=cases`
adds that at no cost and joins `DefaultBias` with F-042's `w=1.8 complete=1.5` (47/53, guard set unchanged). No
bias of the four grids makes `de-paare-missed` equal without losing a keyword case. Two trades are left to the user:

- `w=2.15 complete=0` (with `fold=cases`, and `spent` or not): `de-paare-missed` equal and 28/32 sets, for
  `pl-francja-present`, F-042's 0.0033 near-tie;
- `spent`: no phrase loops (the set `Diese Paare können`), for the variant `Cowley`, a finished-slot search miss.

**Open.** What the engine's bonus is a function of, given that `accomun` and `Paare` have 3 tokens each and want
bonuses on either side of the `Bare` threshold: the model's own probability of the tokens, the characters, or a
pruning that differs from Whisper's only under a bias. More variants per clip (a keyword's length against its token
count at fixed text) would narrow it.

**Kept for the probes.** `Bias.fold` (`foldCases` is now on in `DefaultBias`), `share`, `wordEnd`, `spent` and
`bareWritten` are unexported, their zero values the old rules; `TestCaseSpellings`, `TestKeywordSpent`,
`TestByteKeywords` and `TestCaseKeywords` pin them. The probes of F-042 and F-044 vary F-029's bias (`f029Bias`), so
their tables stay reproducible.

### Decision (2026-10-12)

The user took both trades: `DefaultBias` is now `w=2.15 complete=0 fold=cases spent` (revoke 1, all forms), the
summary table's row with `de-paare-missed` equal. `TestProbeFold` (grid 4) on it: 47/53 keyword cases (won
`de-paare-missed`, lost `pl-francja-present` against the old default), 28/32 keyword sets (was 18/32), 28/31
variants (was 29/31: the variant `Cowley` is now a finished-slot search miss), and no phrase loops. The engine's rule
is still unknown; this is the best trade the grids found, not the rule. The four fast keyword cases of
`TestKeywordsMatchEngine` keep the engine's text, two with lower scores (`en-transfer-distractor` 0.0711 → 0.0072,
`de-paare-adoptionsplan-list` 0.1830 → 0.1673), whose floors moved below them. Nothing without keywords changes.

## F-045 — The Needle 3 text model: its files against Whistle's, the `.cact` map, and a float forward pass equal to JAX (PLAN.md 11.1, 11.2) (2026-10-11, 2026-10-12)

**Question.** Do `needle3.safetensors` and `needle3.cact` (HF `Cactus-Compute/needle3` @ `93e88ede`) load through the
readers built for Whistle, what differs from Whistle's files, and does a forward pass on the shared blocks of
`internal/needle` reproduce upstream's `SimpleAttentionNetwork`?

### The files against Whistle's

- **safetensors.** 64 tensors, all **F16** (Whistle: 116, F32), 121,019,861 parameters; `config.json`'s
  `total_parameters` 121,021,910 is 2049 more, exactly the derived `.cact` records `hada_p1`, `hada_p2` (1024 each)
  and `heads.manifest` (1). The reader handles F16 unchanged. `__metadata__["config"]` uses upstream's
  `TransformerConfig` names (`vocab_size`, `d_model`, `num_layers`, …), not Whistle's (`text`, `dim`, `dec_layers`,
  …), so Whistle's strict `Config()` rejects it; `TextConfig()`/`TextRun()` decode it strictly instead. The config
  omits what `from_saved` defaults: `engram_orders` (2, 3), `engram_heads` (0 → 768 / (2 · 128) = 3 hashes per order,
  6 tables), `rope_theta` (1e5); `out_vocab` is 8192 = the vocab. New keys: `sliding_window` 1024, `global_layers`
  [4, 9, 14, 19], `kv_window` 256, `kv_bits`/`act_bits`/`weight_bits` (in the run for Whistle), the confidence head's
  `confidence_probes`/`confidence_queries` 4/4, and seven `image_*` keys at 0. The run's `tokenizer_hash` is the MD5 of
  the pinned `tokenizer.model`; `merged_heads` names the confidence head merged at step 10,000.
- **Tensors that Whistle's checkpoint does not have:** `engrams_2…4` (five sites instead of two), the
  `confidence_head/*` (probes (21, 4, 768), gain, query, `row_bias`, proj (3072, 1) with bias), and no
  `encoder/*`, `stem/*`, `cross*` or `pe_gate`. Shapes that change with the width: `q_proj` 768 → 576 (12 heads),
  `gate_proj` and `out_proj` 768 × 768, the HadamardMLP diagonals and `cond_u` at 1024, `cond_v` (768, 8), the Kronecker
  factors 32 × 32 for both a and b (Whistle 16 × 16 and 32 × 32), the engram tables (6, 18432, 128).
- **.cact.** `cact.Open` reads it unchanged: same tag, 196-byte header, the same 28 codebook floats. Header fields
  Whistle leaves at 0 are set: `kv_window` 256, `out_vocab` 8192, `max_seq_len` 8192, `sliding_window` 1024, the
  global mask 0x84210 (layers 4, 9, 14, 19). 581 records: FP16 456, FP32 2, CQ2 115, CQ4 7, RAW 1. The first blob sits
  at 25,920, the directory's end (25,872) rounded up to 64 bytes (Whistle's directory ends on a boundary). After
  `final_norm` comes no audio manifest but `heads.manifest` FP16 `[2]` and the confidence head's six records, laid out
  as `export.py` `_head_tensors` writes them: the matrices CQ4 although `weight_bits` says `default=2`, because
  `quantize.py` `HEAD_BITS` = 4 overrides the plan for heads. The upstream Python finds the head the same way
  (`needle/__init__.py` `_confidence_head_present`: index `num_tensors − 2 − 6·heads`).

### The map

`cact.TextNames` derives the map from the header and the manifest (`TextNamesFor` from a header and head shapes, which
the safetensors side uses). With the F-006 checks all 581 records pass: 456 FP16 records equal their F16 source exactly
(cosine 1), 2 FP32 records are numpy's `RandomState(11/13).permutation(1024)`, and each of the 122 CQ records is what
the CQ quantiser makes of its source, except 29 indices at float32 near-ties (margin ≤ 3.2e-8) and 31 FP16 norms 1 ulp
apart. Minimum cosines: CQ2 0.933338362129 (`layer17.out_proj`; 115 CQ2 between 0.9333 and 0.9402), CQ4
0.994432977749 (`confidence_head.proj`). Whistle's decoder trunk gets the same names from `TextNamesFor`, entry for
entry, so `internal/needle`'s loaders read both models.

### The forward pass

`text.Model` is `SimpleAttentionNetwork.__call__` with `quant=False` on `needle.Block`, `MHC`, `Engram` and the RoPE of
Phase 2. One change in `internal/needle` was needed: the HadamardMLP of a 768-wide model mixes 1024 values. Upstream
pads the input with zeros to 1024 and cuts the output back (`HadamardMLP.__call__`, `pad = n − d_model`); Whistle's
width is a power of two, so F-011's port had taken the model width from `d1`. `LoadHadamardMLP` now takes n from `d1`
and the width from `cond_v` and runs a narrower row zero-padded; at width = n the old path runs unchanged. The SHA-256
of Whistle's `Transcribe` JSON (packed and float weights, word timestamps, depth 4) and `Embed` rows on all 32 clips
equal the base commit's, natively and with `-tags purego`.

The JAX reference (`scripts/text_forward_fixtures.py`) runs upstream's module in float32 on three `render_example`
prompts and their 32-token greedy completions, and keeps every position's top-1, final state and margin. Against it,
from `needle3.safetensors`:

| case               | positions | final cos (worst row) | logits / hidden cos | top-1 equal | smallest JAX margin |
| ------------------ | --------- | --------------------- | ------------------- | ----------- | ------------------- |
| `timer_en`         | 120       | 0.999999974           | 1.000000000         | 120/120     | 0.0146              |
| `message_de`       | 188       | 0.999999975           | 1.000000000         | 188/188     | 0.0826              |
| `lights_two_calls` | 188       | 0.999999975           | 1.000000000         | 188/188     | 0.0016              |
| `long_tools`       | 1230      | 0.999999974           | 1.000000000         | 1230/1230   | 0.0051              |

The 6-token plain text agrees at every layer and position at cosine 1.000000000, though the pre-norm stream grows to
1.7e7 at the last block (the per-layer lane means reach 1e4 at layer 0). The F16 rounding of the stored final states
caps their cosine near 1 − 3e-8.

Two findings on the way: the float model writes `"label":"pasta"` where the engine writes `"Pasta"` for the same
request (11.4: engine numerics, or post-processing); and the fixture script first stored `hidden[:, -1]`, a strided
numpy view, which `safetensors.numpy.save_file` writes as the underlying buffer rather than the view's elements, so
every layer past 0 compared at cosine 0.19–0.58. The script now makes every tensor contiguous.

**The dequantised weights alone.** The same prompts on the dequantised `needle3.cact` with float32 activations: final
cosine 0.939–0.946 (worst row 0.79) and the reference's top-1 on 440/496 positions, 93/99 on the completions. The CQ
weights are no stand-in for the float model; the engine's numerics are 11.5's business.

`long_tools` is the band's case (2026-10-12, below); its final rows are stored and compared from position 992 on.

### The band past 1024 positions (2026-10-12)

Upstream's stack builds `local_mask = mask & band` with `band = (pos_q − pos_k) < sliding_window` and gives it to every
layer outside `global_layers` (`jnp.where(gflag > 0, mask, local_mask)`); `__call__` always passes a mask (the causal
one when given none), so the band is always on. The attention reads the mask twice: in the softmax, and in the causal
taps, whose `tap_ok` is the mask's j-th diagonal. A band of w keeps the diagonals below w, so for w = 1024 and three
taps the taps are unchanged. The engram values are computed once from the full causal mask (`_engram_kv`), not the band.

`needle.Attention.SetWindow(w)` (and `Block.SetWindow`) narrows the causal mask to the keys max(0, p − w + 1) … p in
both attend paths and in `Extend`; 0 is the old mask, and Whistle never sets it. It rejects a window narrower than the
taps (where `tap_ok` would cut them, which is not implemented), a window on the unmasked attention, and `Extend` on the
int8 KV cache (the text model's engine numerics are 11.5's). A float64 reference of upstream's attention with the band
agrees on windows 1–12, with and without taps, on the per-row and the GEMM path; a window at least as long as the
sequence gives the causal output bit for bit. Whistle's `Transcribe` and `Embed` hashes on the 32 clips (as above)
equal `main`'s natively and with `-tags purego`.

`text.Load` sets the window on the 16 local layers, and `Forward` takes up to `max_seq_len` (8192) positions (the error
is now `ErrTooLong`). The new fixture case `long_tools` is a `render_example` prompt with 16 tools (1198 tokens; the
`set_timer` tool the query needs comes first, some 1160 positions before the answer) and its greedy completion, which
calls `set_timer` with 25 minutes and the label "bread". The script also runs the network without the band
(`sliding_window` 0) as a control: past position 1024 the final rows then fall to cosine 0.8876 (max |Δ| 0.0101), the
logits to 0.9947, and 15 of 206 top-1 tokens change; before 1024 the two agree exactly. Go reproduces both sides: with
the band, the row above; without it (`TestForwardBandMatters`), the same 0.887637 and 15 changed top-1 tokens.

### Incremental decoding (2026-10-12)

`text.State` holds a `KVCache` per layer and an `EngramCache` per site, and `Model.Extend(s, tokens)` runs the new
tokens through `Engram.Extend` and `Block.Extend` inside the mHC, as Whistle's decoder does (F-022), with RoPE tables
over `max_seq_len` positions. The local layers' caches keep every position and attend to their band. `Fork` shares the
prefix's keys and values (F-026). On all four forward cases, one call, the prompt then one token per call, and uneven
chunks (on `long_tools` 1000 + 30 + 200, across the window's edge) give `Forward`'s logits at every position at cosine
1.000000000000, and so JAX's top-1 everywhere. A parent and two forks continuing from one prompt, interleaved token by
token, each equal `Forward` on their own sequence; rejected calls (a token outside the vocabulary, more than
`max_seq_len` positions) leave the state as it was.

### Open

The engine's side of the text model: its int8 activations and KV with the KV window of 256 (11.5), and the prompt
rendering, decoding and envelope (11.3, 11.4).

## F-046 — The Needle 3 text tokenizer is sentencepiece, and `RefTokenizer` agrees on valid text (PLAN.md 11.1) (2026-10-11)

**Hypothesis.** The text model tokenizes with sentencepiece over `tokenizer.model` (sha256 `97dfd566…`), as upstream
`SANTokenizer.encode` does: `[BOS] + encode(prompt)` (`run.py`), prompt and target encoded separately
(`finetune._encode`). **Alternatives:** `export.py` `RefTokenizer` over the `.cact` RAW blob (record 580), which the
closed engine might implement; a normaliser other than identity.

**Measured.** The model is BPE with byte fallback and the identity normaliser: no charsmap, no dummy prefix, no
whitespace squeezing, only U+0020 escaped to `▁`; 8192 pieces (7908 NORMAL scored 0, −1, … in id order, 256 BYTE at
ids 28–283, 24 USER_DEFINED at 4–27, 3 CONTROL, 1 UNKNOWN). The 24 markers are upstream's 16 `CHAT_MARKERS` plus
`<image>`, `<speech>`, `<tts>`, `<imagen>` and their closing tags. The `.cact` blob holds the same pieces, bit-equal
scores (−0 included) and types. `internal/spm`, a port of sentencepiece 0.2.2 (whole-piece user-defined symbols,
score-ordered merges leftmost on ties, CONTROL/UNKNOWN/BYTE never merged, UNUSED re-split, byte fallback, one U+FFFD
per invalid byte, `" ⁇ "` for unk; 0.2.2 orders equal scores by bit pattern, so +0 ranks above −0), equals Python
sentencepiece on 117/117 fixture texts and 80/80 decode sequences (6,668 ids) and on the decode of every single id.
`RefTokenizer`, upstream's Python and the `internal/tokenizer` port, equals sentencepiece on all 109 valid fixture
texts, on 30,000 random texts (Python, not committed) and 20,000 (Go): no CONTROL, UNKNOWN or BYTE piece is the join
of two pieces and there are no UNUSED ones, so its merge-into-any-piece rule never fires. It differs only on invalid
UTF-8 (sentencepiece encodes U+FFFD, the port the raw bytes; 8/8) and on decoding `<unk>` (`" ⁇ "` against nothing;
5/80).

**Winner.** `internal/spm` with sentencepiece's semantics. For valid UTF-8 both candidates give the same ids, so the
engine's choice cannot change its input; callers pass valid UTF-8, and `<unk>` never comes out of `Encode` with byte
fallback on.

## F-047 — The engine on the text model: the envelope, determinism, threads, depth and audio (PLAN.md 11.4) (2026-10-11)

**Question.** Does the local engine 3.2.0 (`engine/macos-arm64`, HF revision `2ae11323`) load the pinned
`needle3.cact`, and what does it answer? `tools/oracle/text` records it into `testdata/text/engine/` (`just oracle-text`).

- **Loading.** It loads `needle3.cact` alone and together with `whistle.cact` (`needle_models` 3); the `.cact` tag is
  generation 3, and `ENGINE_VERSIONS[3]` is 3.2.0. `config.json` names engine 3.1.0; nothing newer was needed.
- **The envelope.** Every answer has `type, success, error, error_code, reason, function_calls, suppressed_calls,
reasoning, confidence`, `validation{ungrounded, negation}` when a call was produced, `audio_text, audio_language`
  for speech, and the timings `prefill_tps, decode_tps, peak_ram_mb, audio_ttft_ms, audio_decode_tps`, which the
  fixtures drop. Over 19 cases and 20 turns: `call` 19, `respond` 1 (only after a tool result); a refusal is `call`
  with no calls; 2 calls suppressed below confidence 0.1 (0.092, 0.0315, the second with `negation: true` and
  "unlock" turned into `lock_door`); "don't turn off the bedroom lights" still dispatches `on:false` with
  `negation: true`; `--max 8` gives `success:false, error_code:truncated, reason:runtime_failure`. Confidence ranges
  0.0315–0.9595. The engine reports `invoice.total` ungrounded for `$1,200.00` → 1200.0, with the tool name lowercased.
- **Determinism and threads.** Without the timings, two full runs are byte-identical, `needle_complete` through
  purego equals the CLI, a second pass in the same process too, and `--serve` equals `--prompt`. Unlike Whistle
  (F-025), the text answers depend on `--threads`: at 1, 2, 3 or 8 threads 18–19 of 19 answers differ from 4 threads
  in `confidence` (2nd–4th decimal) and in 3 cases in `reasoning`; the calls never changed. No flag equals 4 on this
  machine. The fixtures pin `--threads 4`.
- **Dates.** `--system` takes a file path; a literal string is silently ignored, as if absent. Neither the CLI nor the
  library adds a date: upstream's `_with_date_fact` runs in the Python client. Without a date fact both relative-date
  cases end `truncated`; every case passes `date: 2026-10-11 Sun 09:30`. A trailing newline in the system file keeps
  the call and moves confidence (0.3399 → 0.297).
- **Depth.** `--depth 2…20` loads (20 equals the default), 0 and 1 are refused (`--depth takes a layer count of 2 or
more`), 21 fails to load. Depth 19 already moves the confidence; low depths change the calls.
- **Counts.** `needle_init` returns the static prefix's token count (314, 243 and 91 for the three tool sets; a
  `; locale: xx-XX` suffix adds 6), and `needle_complete` returns the number of generated tokens (8 under `--max 8`).
- **Text embeddings.** `needle_embed` of text returns 3072 L2-normalised floats, the same after `needle_init`. 3072 =
  4 × 768 is the shape of a probe head's pooled output (`probe_pool`, q · d), although `needle3.cact` carries only the
  confidence head (PLAN.md 11.9).
- **Audio in, tool calls out.** On three English clips the fused answer equals the `--prompt` answer to its own
  `audio_text`, plus `audio_text` and `audio_language`; none dispatched a call (two suppressed at 0.0059 and 0.0146,
  one empty at 0.5247).

**Consequences for the port.** Decoding parity (11.4) is measured at 4 threads, on calls and arguments first; the
confidence and the reasoning get tolerances, since the engine itself moves them with the thread count. The static
prefix counts pin the prompt rendering (11.3) without seeing the engine's tokens.

## F-048 — The text model's prompt: `render_example` in Go, the static prefix `needle_init` counts, and a float decode against the engine (PLAN.md 11.3) (2026-10-12)

**Question.** What does the engine feed the text model for a request, and how far does the float model, decoding
greedily on that prompt, get to the engine's calls before the envelope (11.4)?

**The rendering.** `toolcall.Render` translates `finetune.render_example`. An optional
`<|im_start|>system\n{facts}<|im_end|>\n`, then `<|im_start|>user\n<tools>{tools}</tools>\n{query}<|im_end|>\n` and
`<|im_start|>assistant\n`; the facts are stripped as Python's `str.strip`, the input is `[BOS]` and the sentencepiece
ids (`toolcall.Encode`). `toolcall.ToolsJSON` writes the tool schemas as Python's
`json.dumps(json.loads(raw), separators=(",", ":"), ensure_ascii=False)`:

- **Order:** the caller's key order, with a repeated key keeping its first place and its last value, as a dict does.
- **Text:** non-ASCII and U+2028 raw; only `"`, `\`, the five named escapes and other control characters (`\u0007`,
  lower case) are escaped. Go's `encoding/json` would also escape `<`, `>`, `&` and U+2028.
- **Numbers:** integers in full (`-0` is 0); every literal with a `.`, `e` or `E` is a float in Python's repr
  (`1e2` → `100.0`, `2E-5` → `2e-05`, `1e16` → `1e+16`, `1e400` → `Infinity`); like `json.loads` it reads the
  literals `NaN`, `Infinity` and `-Infinity`, which `encoding/json` rejects, so `ToolsJSON` has its own parser.

`toolcall.WithDateFact` is the Python client's `_with_date_fact` with the clock passed in (the engine adds no date,
F-047). `scripts/text_prompt_fixtures.py` runs upstream's functions on the 19 engine cases, the forward fixture's 4
prompts and 10 edge cases (escapes, floats, duplicates, overflow, non-finite literals, blank and padded facts, no tools, markers in the
query), plus 10 date cases at a fixed clock (`testdata/text/prompt/render.json`). Go gives every prompt byte for
byte, every id sequence and every date fact.

**The static prefix.** `needle_init` returns 314, 243 and 91 for the three tool sets of the engine corpus, 6 more
with a `; locale: xx-XX` suffix (F-047). `[BOS]` and the rendering up to `</tools>` (`toolcall.StaticPrefix`) gives
exactly that on all 19 cases (`TestStaticPrefixMatchesEngine`); its ids are the start of the whole prompt's ids,
since `</tools>` is a user-defined piece. One other reading fits as well: the rendering up to the query without
`[BOS]`, because the `\n` after `</tools>` is a token of its own. The counts cannot tell the two apart; the first is
the training format (`[BOS] + encode(prompt)`), so it is the one kept. Both give the same input for a whole prompt.

**The float decode.** `TestFloatGreedyAgainstEngine` renders each case's first turn, decodes greedily with the float
model (`text.Model.Extend`, F-045) until `</tool_call>`, `<|im_end|>` or EOS within the case's token budget, and
sets the calls against the engine's. 7 of 19 are equal: the thermostat, both refusals, `negation_only`, the German
and French cases, and the truncation under `--max 8`. The other 12 differ where the engine works on the decode:

| what the engine does                     | cases                                                                  | float model against engine                                                   |
| ---------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| capitalises the first letter of a string | `timer_pasta`, `timer_then_result`, `event_tomorrow`, `event_explicit` | `pasta` → `Pasta`, `dentist`, `lunch with Marco`                             |
| re-resolves dates against the date fact  | `weather_friday`, `event_no_year` (also capitalised)                   | Friday `2026-10-13` → `2026-10-16`; `2026-03-04` → `2027-03-04`              |
| rewrites a message                       | `message_and_timer`                                                    | `I'm running late` → `Running late`                                          |
| holds the arguments to the schema        | `lights_and_lock`, `spanish_lights_music`, `extract_invoice`           | extra `action`; `off` for `on`; `invoice_number` for the required `due_date` |
| suppresses calls below confidence 0.1    | `alarm_morning`, `negation_door`                                       | an extra `label`; one call for the engine's two                              |

The engine's own `reasoning` still carries the float model's values in several of these (`'pasta' -> label
'pasta'`, `date '2026-10-13' from 'Friday'`, `'March 4th' -> date '2026-03-04'`), so they are rewritten after the
decode, not decoded differently. The schema cases fit "a byte-level grammar compiled from your schemas" (F-047),
which would forbid `action` and `off` and require `due_date`. The smallest top-1 margin along each decode is
0.002–0.48; none of the 12 differences needs a near-tie to explain it.

**Consequences for 11.4.** The decode half of 11.3's Accept waits for the envelope: the schema grammar, the
confidence head and its suppression, and the post-decode rewrites (capitalisation, date resolution against the
date fact and the source years, message trimming). These cases are the test set for each.

## F-049 — Tool retrieval is off for the pinned model: the engine renders every tool, up to the 8192-token context (PLAN.md 11.3) (2026-10-12)

**Question.** Upstream's `llms.txt` says that "with more than 5 declared tools, a built-in retrieval head renders only
the top-5 per turn and constrains the grammar to that subset", and `--tool-index` keeps "this tool set's embeddings".
What ranks the tools, when `needle3.cact` has no embedding head?

**Upstream.** The Python docs (cactuscompute.com/blog/needle-python-docs) say "every schema is embedded once at init
by a built-in contrastive head" and "each turn embeds the query". The porting notes
(cactuscompute.com/blog/porting-needle) say when: "That path switches on only when the archive carries an embedding
head, and this release does not ship one", so "every declared tool goes into the prefix"; "If your port sees no head
with code 1 in the manifest, retrieval is off." The engine binary's only trace of retrieval is the budget error below.

**Measured.** `just oracle-text retrieval` writes `testdata/text/engine/retrieval.json`:

- **Heads.** `needle3.cact`'s `heads.manifest` holds one code, 2 (confidence). There is no 1 (embedding) and no 3
  (router).
- **Every tool is rendered.** `needle_init`, given a tool index path, runs on every prefix of a 17-tool `catalogue`
  (the corpus's 10 assistant and home tools and 7 more) and on the catalogue plus synthetic tools up to 200 and 239
  tools. Its count grows with every tool (72, 123, …, 1025 for 17 tools, 1488, 2524, 4605, 6923 and 8191) and equals
  `toolcall.StaticPrefix`, `[BOS]` and the rendering of all the tools, on all 22 accepted sets
  (`TestStaticPrefixPastFiveTools`), and on the 5 new corpus cases with 1025 tokens (`TestStaticPrefixMatchesEngine`,
  24 cases). Earlier scratch runs agree to 400 tools.
- **No index.** No run writes the index file, neither through the library nor with the CLI's `--tool-index`.
- **The budget.** At 8191 tokens `needle_init` accepts (about 4 s of prefill). At 8192 it refuses at once:
  "needle_init context budget exceeded: static prefix is 8192 tokens, model context is 8192 tokens; reduce
  system/tools JSON or declare more than 5 tools to enable per-turn retrieval". The message suggests retrieval
  whatever the model carries. 8192 is `max_position_embeddings`. Scratch runs with 250, 300 and 400 tools were refused
  at 8470, 10109 and 13378 tokens, each the Go count.
- **The calls.** Five new corpus cases (`catalogue_*`, 24 cases now) on the 17 tools ask for the 8th, 13th, 14th,
  16th and 17th tool, and the engine calls each. The float model, greedy over the same prompt with all 17 tools, picks
  the same tool in all five. Three are equal. In `catalogue_lock` it adds a `required` argument (the schema grammar of
  11.4). In `catalogue_email` it writes `review` for `Review` and `["tom"]` for `["Tom","Sara"]`. Over the 24 cases
  `TestFloatGreedyAgainstEngine` now gives 10 equal and 14 differing (F-048's 7 of 19 plus 3).

**Port.**

- `toolcall.StaticPrefix` stays as it is: it already renders every tool.
- `toolcall.CheckPrefix(prefix, context)` is the budget check: `ErrContextBudget` at `context` tokens or more, with
  the engine's numbers.
- `toolcall.Retrieves(embeddingHead, tools)` is upstream's rule: an embedding head and more than `RetrievalTools`
  (5) tools.
- `text.Source.HasHead(code)` reads the heads.
- The pinned model never retrieves, so the branch with an embedding head is not measured.

**Open.**

- **A future model with an embedding head:** retrieval comes back, and with it the question of what the engine
  embeds per schema.
- **A caller's own shortlist:** the porting notes give a recipe with `needle_embed`, the 5 tools with the highest
  `index @ q`. They call `needle_embed` "not a contrastively trained embedding", a "unit-norm vector of 3072 floats
  (four probe queries over width 768)" (PLAN.md 11.9).
- **Normalisation before rendering:** the same notes say the engine normalises the tools first. It unwraps
  OpenAI-style `{"type":"function","function":{…}}` entries, turns tool and property names into snake_case, drops keys
  the model was not trained on (`triggers` among them), and maps the names back in the calls. None of the corpus's
  tools needs this; it is a new 11.3 item.

## F-050 — Tool normalisation: the engine unwraps, snake-cases and drops `triggers` before it renders, and maps the names back (PLAN.md 11.3) (2026-10-12)

**Question.** The porting notes say the engine normalises the tools before rendering: it unwraps OpenAI-style
entries, converts tool and property names to snake_case, drops keys the model was not trained on (`triggers` among
them), and keeps an alias table so the calls carry the caller's names. Upstream's Python client passes dict schemas
to `needle_init` unchanged, so the rules exist only inside the engine. Which are they exactly?

**Measured.** `just oracle-text normalise` writes `testdata/text/engine/normalise.json` (byte-identical on a re-run):
`needle_init`'s count on 58 tool sets and the CLI's answers to two queries over three of them.

- **Unwrapping.** An entry with a `function` object is that object: `{"type":"function","function":{…}}`, the same
  with `"type":"tool"`, and with other keys beside it (dropped) all count 314, the plain `assistant` set's count. An
  entry whose `function` is a string stays as it is.
- **Dropped keys.** Only a tool's top-level `triggers`. `examples`, `x-foo`, `strict`, `title`, `returns`,
  `metadata`, `$schema`, … at the top, in `parameters` or in a property, are all counted as written; so is a
  `triggers` key inside `parameters`.
- **Snake case.** 26 tool names and 6 property names pin the rule `toolcall.Snake` implements: a capital that starts
  a lower-case word, and one after a lower-case letter or a digit, get an underscore before them; runs of other
  characters than ASCII letters and digits become one underscore; underscores at the ends go; the name is lower-cased
  (`getHTTPResponseCode` → `get_http_response_code`, `get2FACode` → `get2_fa_code`, `set timer` → `set_timer`,
  `__zoom_in__` → `zoom_in`). A name with non-ASCII characters is left as it is (`naïveBayes`, `ÄpfelZählen`,
  `größeWert`). Spaces and leading or trailing underscores are pinned by the counts; `-`, `.`, `/`, `:`, `+`, `$`
  and `~` tokenise exactly as `_` does, so the counts cannot tell whether they become `_`; the port follows the notes'
  "snake*case" and turns them into `*`.
- **Where.** Tool names; the property names of `parameters` and of every property schema, of `items` when it is an
  object, of the `anyOf`, `oneOf` and `allOf` branches and of an `additionalProperties` schema; `required` entries
  that name a renamed property. Not `$defs` (nor the `$ref` names), not an `items` list, not `required` names that
  are no property, not enum values.
- **Collisions.** Tools whose different names share a snake name all get a suffix: `setTimer` and `set_timer` count
  as `set_timer_1` and `set_timer_2`, three such as `_1`, `_2`, `_3`; two tools of the same name stay as they are.
  Within one object the first property keeps the snake name and the next get one: `maxCount`, `max_count` count as
  `max_count`, `max_count_1`, and the `required` order shows which is which. All digits tokenise alike, so which
  number goes to which tool, and that the suffixes are numbers at all, is the port's choice (declaration order).
- **The alias table.** Over the camelCase tools the engine calls `setTimer` and `createEvent` with `durationMinutes`;
  over the OpenAI-style ones `set_timer` and `create_event` with `duration_minutes`; over the colliding pair it calls
  `setTimer` for "set a timer" and `set_timer` (the "kitchen" timer) for the meeting. So the calls carry the caller's
  tool and argument names.

**Port.** `toolcall.Normalise(raw)` returns the tools JSON to render (`Normalised.JSON`) and the alias table:
`CallerName` and `CallerArgs` map the model's names back, through nested objects, arrays, branches and
`additionalProperties`. Without anything to normalise it is `ToolsJSON`. `TestNormaliseMatchesEngine`: the static
prefix of the normalised tools has `needle_init`'s count on all 58 sets (33 of them differ without normalising).
`TestNormaliseAliasesMatchEngine`: the alias table gives back the engine's names in all 6 answers.

**Port-side, not measured: mapping arguments through composed schemas.** The engine's answers show that calls carry
the caller's names; how it maps them back under `anyOf`, `oneOf` and `allOf` is not observable from the recorded
answers, so `CallerArgs` follows a rule of the port's own (the reviews of PRs #61 and #62):

- every `allOf` branch names keys, together with the schema's own properties;
- of each `anyOf` and `oneOf` list one branch does: the one whose names (with its own nested branches) cover most of
  the object's keys, then whose `additionalProperties` takes most of the rest, the first on a tie;
- a key gets the caller's name from the first applicable node that declares it (the schema's own properties first),
  else falls under the first applicable `additionalProperties` schema, else stays as it is;
- the alternatives considered were the first branch with the key (keys of sibling branches mixed in one object) and
  the branch naming most keys of the whole schema (the common properties decided). Values are not checked against the
  branches' types, so two branches that name the same keys differently resolve to the first;
- an array's items follow the same rule per element; generated name suffixes never repeat a name already given.

**Open.** The separators that tokenise like `_`, the collision suffixes' digits (above), and the engine's own rule for
composed schemas (above). 11.4 applies `CallerName` and `CallerArgs` to the parsed calls.

## F-051 — Multi-turn: the wire format of later turns and tool results, the per-turn budget, and what a refused turn does (PLAN.md 11.3) (2026-10-12)

**Question.** How does the engine render a conversation after the first turn, when is a turn a tool result, what is
its turn limit, and what happens to a turn that does not fit?

**The oracle.** The CLI's `--fail-input-overflow` refuses "a turn that would not fit the context window instead of
trimming it" with `input exceeds the context window (N tokens, M available)`. A large `--max` shrinks the room (8192 −
314 − 7000 = 878 tokens), so probes of about a thousand tokens are refused at once with their exact length.
`just oracle-text turns` writes `testdata/text/engine/turns.json` (byte-identical on a re-run): 53 conversations over
`needle --serve`, each on a fresh server, every turn's answer, and a last probe. The probes are stored as head, a
repeated unit and tail, with each text's SHA-256.

**Measured.**

- **The first turn** continues the user message the static prefix opened: `\n{input}<|im_end|>\n<|im_start|>assistant\n`
  (N = the input's tokens + 6), as `Render` writes it.
- **A later user turn** opens its own: `\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n` (+ 9).
- **A tool result**: `\n<|im_start|>tool\n<tool_result>{json}</tool_result><|im_end|>\n<|im_start|>assistant\n`
  (+ 11). An input is one when the answer before it made at least one call (two calls too) and it is JSON: arrays,
  objects, strings, numbers and literals alike. After a suppressed call, a refusal (no call), a `respond` answer or a
  refused turn, JSON is a user turn; so is invalid JSON after a call, and JSON as the first turn.
- **The tool result's JSON** loses the white space outside its strings and keeps everything else as written: `1.50`,
  `1e2`, `-0`, large integers, `\/`, the `\uXXXX` escapes (non-ASCII, ASCII, control, upper-case hex) and repeated
  keys all count as written; a pretty-printed or padded array counts as its compact form. Python's
  `json.dumps(json.loads(…))` misses 5 of these probes. `<`, `&` and `>` inside its strings are the exception: each
  costs 2 tokens more alone, and the three adjacent cost 8 more, which no per-character spelling we tried (`<`,
  entities, full-width forms, any 3-byte character) explains. Their spelling is not identified.
- **A refused turn still counts:** after a refused first turn the next one is a later turn (+ 9), and after a refused
  turn that followed a call, JSON is a user turn.
- **Reset** starts over: the first turn's form again.
- **The budget** is per turn: M = context − static prefix − the answer's token limit (7366 at `--max 512`, 7778 at
  100, 878 at 7000), whatever came before. Earlier turns do not count; the engine makes room for a turn that fits M.
- **Without the flag** an oversized turn is cut, not refused, and keeps its end: with a timer request at the end of
  about 1500 tokens the engine calls `set_timer` (25 minutes, "Eggs"); with the request at the start it does not.

**Port.** `toolcall.Conversation` (`NewConversation`, `Turn`, `Answered`, `Reset`, `Available`) renders the three
forms, decides tool results as above, refuses with `*OverflowError` (`ErrContextOverflow` with the engine's two
numbers), and otherwise drops the input's first tokens until the turn fits. `TestConversationMatchesEngine` replays
all 53 conversations: 41 refusals equal the engine's numbers; the 12 with `<`, `&` or `>` in a tool result's strings
are listed as unidentified, where the engine counts more. `TestConversationTrim` checks the two cut turns.

**Open.**

- **The history.** Neither the error nor the answer reports the conversation's length, so how the engine drops
  earlier turns to make room, and how it renders the assistant's turns in the history (the porting notes: the forced
  `<think>`, one line, `</think>`, a newline, `<tool_call>[…]</tool_call>`), are not measured. `Conversation` renders
  the turns the engine adds, not the history.
- **The cut's size.** An oversized turn keeps its end; how many tokens it keeps is not measured. `Turn` keeps as many
  as fit.
- **`<`, `&` and `>`** in a tool result's strings (above).
