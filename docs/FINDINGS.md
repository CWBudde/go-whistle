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
