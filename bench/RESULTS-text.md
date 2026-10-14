# Benchmarks: the Needle 3 text model against the engine

PLAN.md 11.5's **Accept**: speed and memory against the engine, and the packed weights. This file is
generated from `bench/results/text-go.json` (`just bench-text`) and `bench/results/text-engine.json`
(`just oracle-text bench`); do not edit it by hand. Method and reading: `docs/FINDINGS.md` F-061.

## Runs

| run                         | Go (`just bench-text`)        | engine (`just oracle-text bench`)                          |
| --------------------------- | ----------------------------- | ---------------------------------------------------------- |
| date                        | 2026-10-14T18:07:39+02:00     | 2026-10-14T17:51:40+02:00                                  |
| commit                      | 924f8366ed                    | 267eea9065                                                 |
| CPU                         | Apple M5 Pro                  | Apple M5 Pro                                               |
| cores                       | 15 (5 Super + 10 Performance) | 15 (5 Super + 10 Performance)                              |
| memory                      | 48 GiB                        | 48 GiB                                                     |
| OS                          | macOS 27.0.1 (26A434)         | macOS 27.0.1 (26A434)                                      |
| Go                          | go1.27.1                      | go1.27.1                                                   |
| needle3.cact SHA-256        | c9d915eca2                    | c9d915eca2                                                 |
| needle3.safetensors SHA-256 | c234c70dcc                    | —                                                          |
| cases.json SHA-256          | 698aee1811                    | 698aee1811                                                 |
| engine                      | —                             | 3.2.0 (2ae11323dc)                                         |
| workers                     | GOMAXPROCS 15                 | CLI `--threads 4`, library default (fast cores, at most 4) |
| passes over the cases       | 1 warm-up + 5 timed           | 1 warm-up + 5 timed                                        |
| load average, start         | 2.43 2.83 4.40                | 3.88 6.06 6.47                                             |
| load average, end           | 7.83 9.50 9.06                | 4.44 5.81 6.35                                             |
| run time                    | 1232 s                        | 67 s                                                       |

**The Go and the engine results are not comparable**: they come from different machines (CPU, cores, OS or architecture) or answered different inputs (model, cases or recorded answers), so this report does not compare them. Re-run `just bench-text` and `just oracle-text bench` on the same machine and inputs for the ratios.

## Speed: the 24 first turns

| the 24 first turns                        | Go safetensors float32 | Go packed exact | Go packed + int8 + window | Go packed LUT + int8 + window | engine library | engine library, prefix cached | engine CLI (process) |
| ----------------------------------------- | ---------------------: | --------------: | ------------------------: | ----------------------------: | -------------: | ----------------------------: | -------------------: |
| prefill tokens, median                    |                    332 |             332 |                       332 |                           332 |            332 |                           332 |                  332 |
| generated tokens, median                  |                     58 |              60 |                        59 |                            59 |             61 |                            61 |                   61 |
| TTFT, median (ms)                         |                  109.6 |           345.6 |                     349.5 |                         110.7 |           73.0 |                           5.7 |                    — |
| TTFT, p90 (ms)                            |                  329.3 |          1041.3 |                    1054.6 |                         347.9 |          286.3 |                           8.7 |                    — |
| prefill, median (tok/s)                   |                   3010 |             947 |                       937 |                          2961 |           4526 |                             — |                    — |
| engine's `prefill_tps` (the turn), median |                      — |               — |                         — |                             — |           3839 |                          3864 |                 3948 |
| decode, median (tok/s)                    |                    190 |             104 |                       116 |                           411 |           1739 |                          1746 |                 1804 |
| decode, p90 (tok/s)                       |                    210 |             113 |                       125 |                           448 |           1874 |                          1884 |                 1911 |
| answer, median (ms)                       |                  591.1 |          1339.0 |                    1286.2 |                         424.9 |          109.2 |                          40.2 |                122.0 |
| answer, p90 (ms)                          |                 1249.2 |          2869.0 |                    2743.0 |                        1076.1 |          329.3 |                          60.8 |                335.3 |
| answers timed (cases × passes)            |                 24 × 5 |          24 × 5 |                    24 × 5 |                        24 × 5 |         24 × 5 |                        24 × 5 |               24 × 5 |
| `function_calls` equal to the recording   |                  20/24 |           24/24 |                     23/24 |                         23/24 |          24/24 |                         24/24 |                24/24 |

The Go and the engine columns are not comparable (machine or inputs); compare within a side only.

## Speed: model load

| model load (ms) | Go safetensors float32 | Go packed exact | Go packed + int8 + window | Go packed LUT + int8 + window | engine library |
| --------------- | ---------------------: | --------------: | ------------------------: | ----------------------------: | -------------: |
| median          |                 1051.4 |             8.1 |                       8.0 |                          13.1 |            4.3 |
| p90             |                 1153.9 |             8.3 |                       8.4 |                          13.5 |            6.5 |

The Go load reads the file and builds the tokenizer, the model and the decoder; `needle_load` takes the file's bytes already read into memory. The CLI's load is inside its process time.

## Memory

| memory (MiB)                                 | Go safetensors float32 | Go packed exact | Go packed + int8 + window | Go packed LUT + int8 + window | engine CLI |
| -------------------------------------------- | ---------------------: | --------------: | ------------------------: | ----------------------------: | ---------: |
| peak RSS after the load                      |                  982.3 |            52.2 |                      52.3 |                          67.7 |          — |
| peak RSS after the 24 answers, load included |                 1367.4 |           478.5 |                     500.2 |                         492.3 |      107.4 |
| Go heap after the load                       |                  464.9 |            42.9 |                      42.9 |                          58.2 |          — |
| engine's `peak_ram_mb`, median               |                      — |               — |                         — |                             — |       94.5 |
| engine's `peak_ram_mb`, largest              |                      — |               — |                         — |                             — |      107.4 |

The Go figures come from a fresh `needle-bench` child per mode (the load, then one pass over the cases); peak RSS is `ru_maxrss`, a high-water mark. The engine's come from the CLI processes of the speed runs: the largest `ru_maxrss` over them, and the `peak_ram_mb` it reports.

**Packed weights:** 38.252 MiB (40,109,936 bytes), as `TestPackedMemory` (`internal/text`) counts the packed `needle3.cact` on exact kernels (F-059): under the Accept's 40 MiB, its "40 MB" read as MiB.

## What is measured

- **The cases** are the first turns of `testdata/text/engine/cases.json`: their tools, the corpus's system
  text with its date fact, and the `--max` a case sets (512 otherwise), as the corpus was recorded.
- **TTFT** runs from the turn's tokens in hand to the first generated token, the static prefix's prefill
  included. Go times the first `Extend` over the static prefix, the turn and the forced `<think>\n`, and
  the pick of the next token. The engine reports no TTFT for text: the library's column is
  `needle_init`, which prefills the static prefix, and the `needle_complete` call less (n − 1) /
  `decode_tps`, an upper bound that also holds the confidence and the envelope. The CLI's process time
  cannot be split.
- **Prefix cached**: libneedle3 keeps the static prefix's prefill while `needle_init` gets the same system
  text and tools. The library column answers every case cold (another prefix answered before it), as the
  CLI and Go do; the cached column answers it again at once.
- **Prefill** is the prefill tokens over the TTFT on both sides. The engine's own `prefill_tps` counts the
  turn's tokens only, the static prefix being prefilled by `needle_init`.
- **Decode** is the tokens after the first over their steps: Go replays its answer one `Extend` per
  token; the engine reports `decode_tps`.
- **Answer** is one turn. For Go: the tools' normalisation, the prompt, the decode under the grammar, the
  confidence and the envelope, the model loaded. For the engine library: `needle_init` and
  `needle_complete`. For the engine CLI: the whole process, start-up and model load included.
- **Like with like** is Go against the engine library: in-process, the model loaded. Go runs its four
  modes in one process, alternating which goes first per case and pass, so all see the same machine load.
- **Modes**: safetensors float32 is `needle3.safetensors`; packed exact is `needle3.cact`'s CQ records
  packed on exact kernels (F-059); packed + int8 + window adds the engine's numerics (F-060); packed LUT
  runs the lut kernel (`OpenOptions.LUT`). The bench changes no numerics.

## Reproduce

```bash
just bench-text          # the Go side: bench/results/text-go.json, then this file
just oracle-text bench   # the engine side: bench/results/text-engine.json, then this file
go run ./bench/cmd/needle-bench report   # re-render this file from both JSON files
```

`just bench-text` needs `models/needle3` (`just fetch-needle3`); `just oracle-text bench` needs
`engine/<platform>` (`just fetch-engine`).
