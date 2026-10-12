# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

Pure-Go port of [Whistle](https://huggingface.co/Cactus-Compute/whistle), the 55M-parameter speech-to-text model from
[cactus-compute/needle](https://github.com/cactus-compute/needle). Upstream runs Whistle in a closed-source C++ engine
(`libneedle3`), so this port is a **re-implementation from the published weights** and the open JAX reference blocks.
The engine itself is only used as a black-box test oracle. See `PLAN.md` for the roadmap and the model reference card.

**Module:** `github.com/cwbudde/go-whistle`
**Go version:** 1.27+ (the arm64 assembly uses Go 1.27 mnemonics)

## Ground Rules

- **Pure Go in the main module.** No cgo and no Python at runtime. Plan 9 assembly for hot loops is fine, as long as
  there is always a pure-Go fallback (`*_noasm.go`) and a test that compares both.
- **Oracle code is separated.** Anything that loads `libneedle3`, runs the `needle` binary or Python lives under
  `tools/oracle/` or `scripts/`, behind the build tag `oracle`, and never ships with the library or CLI.
- **Never commit weights or engine binaries.** `models/`, `engine/` and `original/` are git-ignored. Tests that need
  them skip with a clear message when they are missing.
- **Every reverse-engineered decision goes into `docs/FINDINGS.md`**: the hypothesis, the alternatives tried, the
  metric against the oracle and the winner. Code comments reference the finding.
- **Parity tests compare by cosine similarity and top-1 / transcript equality**, not by relative error. The engine runs
  W2A8 numerics, so bit-exactness is not a goal.
- When a numerics change invalidates fixtures, regenerate them in the same change.

## Commands

```bash
just test              # go test -v ./...
just test-race
just lint              # golangci-lint (config in .golangci.yml)
just fmt               # treefmt (gofumpt, gci, prettier, shfmt)
just ci                # check-formatted + test + lint + check-tidy + check-cross

go test -v -run TestName ./internal/<pkg>/...
```

## Upstream Pins

| What          | Where                                 | Pin                        |
| ------------- | ------------------------------------- | -------------------------- |
| Python + JAX  | github.com/cactus-compute/needle      | `af2654dffa` (2026-10-07)  |
| Weights       | huggingface.co/Cactus-Compute/whistle | `b358ddad` (weights 2.0.0) |
| Engine oracle | huggingface.co/Cactus-Compute/needle3 | engine 3.2.0               |

Read upstream files at the pinned commit (`gh api repos/cactus-compute/needle/contents/<path>?ref=af2654dffa`) or from
the local snapshot in `original/needle`.
