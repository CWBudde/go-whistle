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
the same test). The first two values stay open for the tensor-name map (Phase 1).

Also measured: the header is 196 bytes, not the 120 the `export.py` docstring states; the codebooks are stored already
scaled by 1/√128 (cb2 = ±0.0399, ±0.1333); blobs are 64-byte aligned and contiguous, and the last one ends exactly at
EOF. Whistle uses only CQ2 (145 tensors) and CQ4 (7), both with group 128; there is no CQ3, ternary or binary tensor.
