# Needle 3 text model forward-pass fixtures

Reference outputs of upstream's `SimpleAttentionNetwork` (`needle/model/architecture.py` @ `af2654dffa`) for the Needle
3 text model, for `internal/text`. `scripts/text_forward_fixtures.py` writes them from the pinned
`needle3.safetensors` @ `93e88ede` (Hugging Face `Cactus-Compute/needle3`) and the pinned sentencepiece
`tokenizer.model`. This file is the contract the Go forward pass is held to; `docs/FINDINGS.md` F-045 records how it was
built and what it measured.

## Regenerate

```bash
just fetch-needle3          # models/needle3/: needle3.safetensors, tokenizer.model, …
just text-forward-fixtures  # writes forward.safetensors
```

This also needs `original/needle` (`just fetch-upstream`). The script refuses inputs that are not the pinned ones. The
run is deterministic: two runs give byte-identical files, and the `__metadata__` keys are written sorted.

## Numerics

- float32 everywhere: the F16 checkpoint is widened to float32, `TransformerConfig.dtype` is overridden from
  `bfloat16` to `float32`, `jax_default_matmul_precision` is `"highest"`, and the platform is CPU.
- `flash=False` (the explicit softmax path) and `quant=False`: no fake quantisation of weights, activations or K/V. The
  engine's W2A8 numerics are not modelled here.
- Upstream's own masks: the causal mask, banded to the sliding window (1024) on the layers outside `global_layers`.
  Only `long_tools` is longer than the window. As a control the script also runs it without the band (`sliding_window` 0) and writes how far that lands from the banded run into the metadata (`band_control`).

## Contents

`forward.safetensors`, no weights. The cases are four tool-calling prompts rendered with upstream's
`finetune.render_example` (system fact `date: 2026-10-11 Sun 09:30`, the tools JSON, the query), plus one short plain
text. Each prompt is `[BOS] + sentencepiece.encode(prompt)`, as `run.py` does it, followed by the reference's own
greedy completion: `run.py generate` at temperature 0 for at most 32 tokens, stopping at EOS. One forward pass over
prompt and completion then gives every position's outputs, so each position's next token is teacher-forced on the
reference's text.

| Case               | Prompt | Completion | Content                                                  |
| ------------------ | ------ | ---------- | -------------------------------------------------------- |
| `timer_en`         | 88     | 32         | one tool, "set a timer for 10 minutes for the pasta"     |
| `message_de`       | 156    | 32         | two tools, a German message request                      |
| `lights_two_calls` | 156    | 32         | two tools, a request for two calls                       |
| `long_tools`       | 1198   | 32         | 16 tools, the needed one first, past the 1024 window     |
| `tiny`             | 6      | 0          | `Hello, world` without the chat format, every layer kept |

Per case `<case>`:

| Tensor                  | Type | Shape        | Content                                                      |
| ----------------------- | ---- | ------------ | ------------------------------------------------------------ |
| `<case>.tokens`         | I64  | (T)          | `[BOS]`, the prompt ids, the greedy completion               |
| `<case>.prompt_len`     | I64  | (1)          | positions of `[BOS]` and the prompt                          |
| `<case>.argmax`         | I64  | (T)          | top-1 next token at every position                           |
| `<case>.margin`         | F32  | (T)          | top-1 minus top-2 logit at every position                    |
| `<case>.final`          | F16  | (T, 768)     | stack output after `final_norm`; `long_tools` from 992 on    |
| `long_tools.final_from` | I64  | (1)          | position of `long_tools.final`'s first row (992)             |
| `<case>.hidden_last`    | F32  | (20, 768)    | lane mean of the stream after each block, last position      |
| `<case>.logits_last`    | F32  | (2, 8192)    | logits at the last two positions                             |
| `tiny.tokens`           | I64  | (6)          | `[BOS]` and the ids of `Hello, world`                        |
| `tiny.embed`            | F32  | (6, 768)     | scaled input embedding, `embedding[tokens] · √768`           |
| `tiny.hidden`           | F32  | (20, 6, 768) | lane mean after each block, every position (above F16 range) |
| `tiny.logits`           | F32  | (6, 8192)    | logits at every position                                     |

The metadata holds the pins, the SHA-256 of both inputs, the numerics and, under `prompts`, every rendered prompt and
its decoded completion.

## Tolerance

Cosine against the reference and top-1 equality, never relative error: `final` per row, `logits_last` per row,
`hidden_last` per layer and every layer and position of `tiny` at cosine ≥ 0.9999, and the top-1 token equal at every
position. `final` is stored as F16, which costs at most about 1e-7 of cosine; the pre-norm stream reaches 1.7e7 at the
last block, so `tiny.hidden` stays F32.
