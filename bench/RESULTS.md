# Benchmarks: go-whistle against the Needle 3 engine

PLAN.md 9.6 and the Phase 9 **Accept**. This file is generated from `bench/results/go.json` (`just bench`)
and `bench/results/engine.json` (`just oracle-bench`); do not edit it by hand. Method and reading:
`docs/FINDINGS.md` F-040.

## Runs

| run                 | Go (`just bench`)                 | engine (`just oracle-bench`)                               |
| ------------------- | --------------------------------- | ---------------------------------------------------------- |
| date                | 2026-10-10T20:01:45+02:00         | 2026-10-10T20:01:27+02:00                                  |
| commit              | fe5884e4e6                        | fe5884e4e6                                                 |
| CPU                 | Apple M5 Pro                      | Apple M5 Pro                                               |
| cores               | 15 (5 Super + 10 Performance)     | 15 (5 Super + 10 Performance)                              |
| memory              | 48 GiB                            | 48 GiB                                                     |
| OS                  | macOS 27.0.1 (26A434)             | macOS 27.0.1 (26A434)                                      |
| Go                  | go1.27.1                          | go1.27.1                                                   |
| engine              | —                                 | 3.2.0 (2ae11323dc)                                         |
| workers             | GOMAXPROCS 15                     | CLI `--threads 4`, library default (fast cores, at most 4) |
| passes per clip     | 1 warm-up + 5 timed; 30 s: 1 + 10 | 1 warm-up + 5 timed; 30 s: 1 + 10                          |
| load average, start | 5.24 5.02 3.89                    | 5.45 5.03 3.86                                             |
| load average, end   | 8.43 6.12 4.44                    | 5.24 5.02 3.89                                             |
| run time            | 106 s                             | 19 s                                                       |

## WER: packed against float (Phase 9 Accept)

| LibriSpeech test-clean subset             | Go packed | Go float |   engine |
| ----------------------------------------- | --------: | -------: | -------: |
| errors / reference words                  |  223/4167 | 223/4167 | 221/4167 |
| WER                                       |    5.35 % |   5.35 % |   5.30 % |
| text equal to the engine's                |   181/200 |  181/200 |        — |
| real-time factor (Σ Transcribe / Σ audio) |    0.0163 |   0.0187 |        — |

"The float path" is `whistle.Load` with `WithFloatWeights()`; packed is `whistle.Load`'s default on `whistle.cact` (F-037). 200 utterances, 4167 reference words, 1549.4 s of audio, the manifest's `wer_rule` on all three. The engine's transcripts are the committed `testdata/oracle/librispeech`.

- Packed − float: **+0.00** percentage points; the Accept allows 0.1, so it **holds**.
- Packed and float transcripts equal (text and language): 200/200.
- Packed − engine: +0.05 percentage points.

## Speed: the 32 fixture clips

| the 32 fixture clips                             | Go packed | Go float | engine library | engine CLI (process) |
| ------------------------------------------------ | --------: | -------: | -------------: | -------------------: |
| TTFT, median (ms)                                |      60.4 |     57.5 |            7.5 |                  7.6 |
| TTFT, p90 (ms)                                   |      76.1 |     71.1 |           10.3 |                 10.0 |
| decode, median (tok/s)                           |      1013 |      604 |           1383 |                 1362 |
| decode, p90 (tok/s)                              |      1150 |      660 |           1457 |                 1420 |
| wall per transcription, median (ms)              |      86.9 |    102.4 |           29.8 |                 39.4 |
| wall per transcription, p90 (ms)                 |     125.3 |    156.0 |           56.4 |                 65.9 |
| real-time factor, median over passes             |    0.0156 |   0.0188 |         0.0056 |               0.0073 |
| transcriptions timed (clips × passes)            |    32 × 5 |   32 × 5 |         32 × 5 |               32 × 5 |
| text and language equal to the engine's fixtures |     31/32 |    31/32 |              — |                32/32 |

A `needle_transcribe` call takes its `ttft_ms` plus (n − 1) / `decode_tps` for the n tokens it returns, plus a median 1.35 ms (p90 2.20 ms) that its own clock does not explain.

## Speed: the 30 s clip (the first clip tiled)

| the 30 s clip (the first clip tiled)  | Go packed | Go float | engine library | engine CLI (process) |
| ------------------------------------- | --------: | -------: | -------------: | -------------------: |
| TTFT, median (ms)                     |     310.4 |    290.1 |           36.3 |                 37.1 |
| TTFT, p90 (ms)                        |     331.1 |    311.9 |           44.2 |                 39.7 |
| decode, median (tok/s)                |       620 |      408 |           1082 |                 1113 |
| decode, p90 (tok/s)                   |       633 |      420 |           1110 |                 1126 |
| wall per transcription, median (ms)   |     495.5 |    575.0 |          144.7 |                154.5 |
| wall per transcription, p90 (ms)      |     558.7 |    617.3 |          177.2 |                158.7 |
| real-time factor, median over passes  |    0.0165 |   0.0192 |         0.0048 |               0.0052 |
| transcriptions timed (clips × passes) |    1 × 10 |   1 × 10 |         1 × 10 |               1 × 10 |

## Speed: model load

| model load (ms) | Go packed | Go float | engine library |
| --------------- | --------: | -------: | -------------: |
| median          |       7.0 |    138.7 |            2.7 |
| p90             |       8.8 |    152.1 |            2.8 |

`whistle.Load` maps `whistle.cact` and builds the model; `needle_load` takes the file's bytes already read into memory. The CLI's load is inside its process wall time.

## Against the published figures

| median of the 32 clips                    | TTFT (ms) | decode (tok/s) | TTFT / engine here | decode / engine here |
| ----------------------------------------- | --------: | -------------: | -----------------: | -------------------: |
| engine, Apple M4 Pro, 5 beams (published) |      11.1 |           1319 |                  — |                    — |
| engine library, this machine              |       7.5 |           1383 |              1.00× |                1.00× |
| Go packed, this machine                   |      60.4 |           1013 |              8.06× |                0.73× |
| Go float, this machine                    |      57.5 |            604 |              7.67× |                0.44× |

The published figures come from another machine and clip set; the engine on this machine is the yardstick. A TTFT ratio above 1 is slower than the engine, a decode ratio below 1 too.

## Memory

| peak RSS (MiB, ru_maxrss of a fresh process) | Go packed | Go float | engine CLI |
| -------------------------------------------- | --------: | -------: | ---------: |
| after the model load                         |      40.6 |    241.1 |          — |
| one 30 s transcription, load included        |     125.3 |    489.4 |       70.0 |
| largest over the 32 clips, load included     |         — |        — |       54.5 |

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
