# Benchmarks: go-whistle against the Needle 3 engine

PLAN.md 9.6 and the Phase 9 **Accept**. This file is generated from `bench/results/go.json` (`just bench`)
and `bench/results/engine.json` (`just oracle-bench`); do not edit it by hand. Method and reading:
`docs/FINDINGS.md` F-040.

## Runs

| run                 | Go (`just bench`)                 | engine (`just oracle-bench`)                               |
| ------------------- | --------------------------------- | ---------------------------------------------------------- |
| date                | 2026-10-11T03:15:54+02:00         | 2026-10-11T03:17:10+02:00                                  |
| commit              | cd6111f0e3                        | cd6111f0e3-dirty                                           |
| CPU                 | Apple M5 Pro                      | Apple M5 Pro                                               |
| cores               | 15 (5 Super + 10 Performance)     | 15 (5 Super + 10 Performance)                              |
| memory              | 48 GiB                            | 48 GiB                                                     |
| OS                  | macOS 27.0.1 (26A434)             | macOS 27.0.1 (26A434)                                      |
| Go                  | go1.27.1                          | go1.27.1                                                   |
| model SHA-256       | b6e02f0485                        | b6e02f0485                                                 |
| clips SHA-256       | 3f723030a1                        | 3f723030a1                                                 |
| engine              | —                                 | 3.2.0 (2ae11323dc)                                         |
| workers             | GOMAXPROCS 15                     | CLI `--threads 4`, library default (fast cores, at most 4) |
| passes per clip     | 1 warm-up + 5 timed; 30 s: 1 + 10 | 1 warm-up + 5 timed; 30 s: 1 + 10                          |
| load average, start | 2.19 1.75 1.41                    | 5.01 2.78 1.83                                             |
| load average, end   | 5.69 2.79 1.82                    | 4.94 2.88 1.88                                             |
| run time            | 61 s                              | 17 s                                                       |

## WER: packed against float (Phase 9 Accept)

| LibriSpeech test-clean subset             | Go packed | Go float |   engine |
| ----------------------------------------- | --------: | -------: | -------: |
| errors / reference words                  |  223/4167 | 223/4167 | 221/4167 |
| WER                                       |    5.35 % |   5.35 % |   5.30 % |
| text equal to the engine's                |   181/200 |  181/200 |        — |
| real-time factor (Σ Transcribe / Σ audio) |    0.0077 |   0.0113 |        — |

"The float path" is `whistle.Load` with `WithFloatWeights()`; packed is `whistle.Load`'s default on `whistle.cact` (F-037). 200 utterances, 4167 reference words, 1549.4 s of audio, the manifest's `wer_rule` on all three. The engine's transcripts are the committed `testdata/oracle/librispeech`.

- Packed − float: **+0.00** percentage points; the Accept allows 0.1, so it **holds**.
- Packed and float transcripts equal (text and language): 200/200.
- Packed − engine: +0.05 percentage points.

## Speed: the 32 fixture clips

| the 32 fixture clips                             | Go packed | Go float | engine library | engine CLI (process) |
| ------------------------------------------------ | --------: | -------: | -------------: | -------------------: |
| TTFT, median (ms)                                |      19.3 |     21.6 |            7.0 |                  7.6 |
| TTFT, p90 (ms)                                   |      21.9 |     24.8 |            9.0 |                  9.0 |
| decode, median (tok/s)                           |      1071 |      597 |           1471 |                 1402 |
| decode, p90 (tok/s)                              |      1229 |      642 |           1547 |                 1432 |
| wall per transcription, median (ms)              |      43.9 |     68.5 |           26.3 |                 38.9 |
| wall per transcription, p90 (ms)                 |      68.3 |    106.6 |           48.7 |                 62.4 |
| real-time factor, median over passes             |    0.0084 |   0.0130 |         0.0050 |               0.0072 |
| transcriptions timed (clips × passes)            |    32 × 5 |   32 × 5 |         32 × 5 |               32 × 5 |
| text and language equal to the engine's fixtures |     31/32 |    31/32 |              — |                32/32 |

A `needle_transcribe` call takes its `ttft_ms` plus (n − 1) / `decode_tps` for the n tokens it returns, plus a median 1.19 ms (p90 1.90 ms) that its own clock does not explain.

## Speed: the 30 s clip (the first clip tiled)

| the 30 s clip (the first clip tiled)  | Go packed | Go float | engine library | engine CLI (process) |
| ------------------------------------- | --------: | -------: | -------------: | -------------------: |
| TTFT, median (ms)                     |      59.9 |     66.9 |           35.0 |                 34.0 |
| TTFT, p90 (ms)                        |      61.5 |     69.9 |           36.6 |                 36.2 |
| decode, median (tok/s)                |       639 |      414 |           1161 |                 1170 |
| decode, p90 (tok/s)                   |       644 |      423 |           1182 |                 1182 |
| wall per transcription, median (ms)   |     243.4 |    349.3 |          136.6 |                146.9 |
| wall per transcription, p90 (ms)      |     248.3 |    358.2 |          145.1 |                152.7 |
| real-time factor, median over passes  |    0.0081 |   0.0116 |         0.0046 |               0.0049 |
| transcriptions timed (clips × passes) |    1 × 10 |   1 × 10 |         1 × 10 |               1 × 10 |

## Speed: model load

| model load (ms) | Go packed | Go float | engine library |
| --------------- | --------: | -------: | -------------: |
| median          |       6.7 |    136.2 |            2.7 |
| p90             |       8.7 |    144.3 |            2.7 |

`whistle.Load` maps `whistle.cact` and builds the model; `needle_load` takes the file's bytes already read into memory. The CLI's load is inside its process wall time.

## Against the published figures

| median of the 32 clips                    | TTFT (ms) | decode (tok/s) | TTFT / engine here | decode / engine here |
| ----------------------------------------- | --------: | -------------: | -----------------: | -------------------: |
| engine, Apple M4 Pro, 5 beams (published) |      11.1 |           1319 |                  — |                    — |
| engine library, this machine              |       7.0 |           1471 |              1.00× |                1.00× |
| Go packed, this machine                   |      19.3 |           1071 |              2.75× |                0.73× |
| Go float, this machine                    |      21.6 |            597 |              3.09× |                0.41× |

The published figures come from another machine and clip set; the engine on the same machine is the yardstick. A TTFT ratio above 1 is slower than the engine, a decode ratio below 1 too.

## Memory

| peak RSS (MiB, ru_maxrss of a fresh process) | Go packed | Go float | engine CLI |
| -------------------------------------------- | --------: | -------: | ---------: |
| after the model load                         |      37.8 |    238.8 |          — |
| one 30 s transcription, load included        |     105.7 |    493.0 |       70.2 |
| largest over the 32 clips, load included     |         — |        — |       53.9 |

The Go figures come from a fresh `whistle-bench` child per mode (`Load`, then one `Transcribe` of the 30 s clip, as `just mem-probe`); the engine's from the CLI processes of the speed runs.

## What is measured

- **TTFT** is the engine's `ttft_ms` and `Result.TTFT`: from the PCM in hand to the first token after the
  language token. It covers the front end, the encoder, the cross-attention memory and the first decoder
  steps, not the process start, the model load or the WAV decoding (F-027, F-040).
- **Decode** is the engine's `decode_tps` and `Result.DecodeTPS`: the tokens after that one over their time,
  the beam search's steps at 5 beams.
- **Wall** is one transcription from PCM in memory to the result: `Transcribe` for Go and
  `needle_transcribe` for the engine library, both without word timestamps, the model loaded, after a
  warm-up. For the engine CLI it is the whole process: start-up, model load, WAV reading and the
  transcription.
- **Real-time factor** is the wall time over the audio duration, per pass over a clip set, median over the
  passes.
- **Like with like** is Go against the engine library: the same PCM, in-process, the model loaded. Go runs
  packed and float alternately in one process, clip by clip, so both see the same machine load.

## Reproduce

```bash
just bench          # the Go side: bench/results/go.json, then this file
just oracle-bench   # the engine side: bench/results/engine.json, then this file
go run ./bench/cmd/whistle-bench report   # re-render this file from both JSON files
```

`just bench` needs `models/whistle/whistle.cact` (`whistle download`) and, for the WER,
`models/corpus/librispeech` (`just oracle-librispeech-audio`); `just oracle-bench` needs `engine/<platform>`
(`just fetch-engine`).
