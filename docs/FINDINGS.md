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

| tensor                   | CQ  | shape   | rows    | float NEON | float purego | `gather` | `nibble` | `nibble4` | MB float32 | MB packed |
| ------------------------ | --- | ------- | ------- | ---------- | ------------ | -------- | -------- | --------- | ---------- | --------- |
| k_proj (96, 512) CQ2     | 1   | 1.84 µs | 8.88 µs | 50.8 µs    | 7.25 µs      | 7.29 µs  | 0.197    | 0.0138    |
| q_proj (384, 512) CQ2    | 1   | 7.70 µs | 34.4 µs | 204 µs     | 20.2 µs      | 20.1 µs  | 0.786    | 0.0553    |
| out (512, 512) CQ2       | 1   | 10.6 µs | 45.9 µs | 274 µs     | 25.6 µs      | 25.8 µs  | 1.049    | 0.0737    |
| pw1 (1024, 512) CQ2      | 1   | 22.8 µs | 91.7 µs | 601 µs     | 49.7 µs      | 50.2 µs  | 2.097    | 0.1475    |
| stem.out (512, 1280) CQ2 | 1   | 28.1 µs | 127 µs  | 651 µs     | 61.2 µs      | 61.5 µs  | 2.621    | 0.1843    |
| phi_res (128, 2048) CQ4  | 1   | 9.69 µs | 52.3 µs | 264 µs     | 55.8 µs      | 55.6 µs  | 1.049    | 0.1393    |
| head (8199, 512) CQ4     | 1   | 192 µs  | 748 µs  | 4.19 ms    | 653 µs       | 652 µs   | 16.79    | 2.230     |
| q_proj (384, 512) CQ2    | 5   | 20.4 µs | 173 µs  | 982 µs     | 96.1 µs      | 76.9 µs  | 0.786    | 0.0553    |
| head (8199, 512) CQ4     | 5   | 464 µs  | 3.71 ms | 20.9 ms    | 3.27 ms      | 2.33 ms  | 16.79    | 2.230     |
| q_proj (384, 512) CQ2    | 375 | 1.21 ms | 12.9 ms | 73.4 ms    | 7.22 ms      | 5.43 ms  | 0.786    | 0.0553    |

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
- Keywords and depths 2–7 still return `ErrNotSupported`; they follow 6.3 and 6.4.

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
