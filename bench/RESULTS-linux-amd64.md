# Benchmarks: go-whistle against the Needle 3 engine

PLAN.md 9.6 and the Phase 9 **Accept**. This file is generated from `bench/results/linux-amd64/go.json` (`just bench`)
and `bench/results/linux-amd64/engine.json` (`just oracle-bench`); do not edit it by hand. Method and reading:
`docs/FINDINGS.md` F-040.

## Runs

| run                 | Go (`just bench`)                            | engine (`just oracle-bench`)                               |
| ------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| date                | 2026-10-12T05:18:23+02:00                    | 2026-10-12T05:21:45+02:00                                  |
| commit              | 2c65d474a9                                   | 2c65d474a9                                                 |
| CPU                 | INTEL(R) XEON(R) GOLD 5520+                  | INTEL(R) XEON(R) GOLD 5520+                                |
| cores               | 80                                           | 80                                                         |
| governor            | none (no cpufreq; kvm guest)                 | none (no cpufreq; kvm guest)                               |
| memory              | 155 GiB                                      | 155 GiB                                                    |
| OS                  | Ubuntu 24.04.3 LTS, kernel 6.8.0-139-generic | Ubuntu 24.04.3 LTS, kernel 6.8.0-139-generic               |
| Go                  | go1.27.1                                     | go1.27.1                                                   |
| model SHA-256       | b6e02f0485                                   | b6e02f0485                                                 |
| clips SHA-256       | 3f723030a1                                   | 3f723030a1                                                 |
| engine              | —                                            | 3.2.0 (2ae11323dc)                                         |
| workers             | GOMAXPROCS 16                                | CLI `--threads 4`, library default (fast cores, at most 4) |
| passes per clip     | 1 warm-up + 5 timed; 30 s: 1 + 10            | 1 warm-up + 5 timed; 30 s: 1 + 10                          |
| load average, start | 5.95 6.72 7.50                               | 10.72 8.64 8.11                                            |
| load average, end   | 10.72 8.64 8.11                              | 13.89 9.90 8.61                                            |
| run time            | 199 s                                        | 150 s                                                      |

## WER: packed against float (Phase 9 Accept)

| LibriSpeech test-clean subset             | Go packed | Go float |   engine |
| ----------------------------------------- | --------: | -------: | -------: |
| errors / reference words                  |  223/4167 | 223/4167 | 221/4167 |
| WER                                       |    5.35 % |   5.35 % |   5.30 % |
| text equal to the engine's                |   181/200 |  181/200 |        — |
| real-time factor (Σ Transcribe / Σ audio) |    0.0159 |   0.0459 |        — |

"The float path" is `whistle.Load` with `WithFloatWeights()`; packed is `whistle.Load`'s default on `whistle.cact` (F-037). 200 utterances, 4167 reference words, 1549.4 s of audio, the manifest's `wer_rule` on all three. The engine's transcripts come from the fixtures' `manifest.json`.

- Packed − float: **+0.00** percentage points; the Accept allows 0.1, so it **holds**.
- Packed and float transcripts equal (text and language): 200/200.
- Packed − engine: +0.05 percentage points.

## Speed: the 32 fixture clips

| the 32 fixture clips                             | Go packed | Go float | engine library | engine CLI (process) |
| ------------------------------------------------ | --------: | -------: | -------------: | -------------------: |
| TTFT, median (ms)                                |      47.3 |     70.7 |          160.9 |                410.8 |
| TTFT, p90 (ms)                                   |      55.4 |     80.0 |          204.2 |                505.6 |
| decode, median (tok/s)                           |       547 |      130 |            523 |                  360 |
| decode, p90 (tok/s)                              |       616 |      139 |            565 |                  382 |
| wall per transcription, median (ms)              |      96.5 |    282.6 |          217.1 |                520.2 |
| wall per transcription, p90 (ms)                 |     147.7 |    479.7 |          294.0 |                674.2 |
| real-time factor, median over passes             |    0.0184 |   0.0548 |         0.0319 |               0.0748 |
| transcriptions timed (clips × passes)            |    32 × 5 |   32 × 5 |         32 × 5 |               32 × 5 |
| text and language equal to the engine's fixtures |     31/32 |    31/32 |              — |                29/32 |

A `needle_transcribe` call takes its `ttft_ms` plus (n − 1) / `decode_tps` for the n tokens it returns, plus a median 2.17 ms (p90 4.16 ms) that its own clock does not explain.

## Speed: the 30 s clip (the first clip tiled)

| the 30 s clip (the first clip tiled)  | Go packed | Go float | engine library | engine CLI (process) |
| ------------------------------------- | --------: | -------: | -------------: | -------------------: |
| TTFT, median (ms)                     |     159.6 |    195.6 |          727.2 |               1969.5 |
| TTFT, p90 (ms)                        |     170.8 |    210.7 |          761.4 |               2029.5 |
| decode, median (tok/s)                |       366 |       99 |            407 |                  280 |
| decode, p90 (tok/s)                   |       397 |      101 |            415 |                  283 |
| wall per transcription, median (ms)   |     480.0 |   1398.0 |         1012.6 |               2434.3 |
| wall per transcription, p90 (ms)      |     538.7 |   1441.4 |         1049.8 |               2491.9 |
| real-time factor, median over passes  |    0.0160 |   0.0466 |         0.0338 |               0.0811 |
| transcriptions timed (clips × passes) |    1 × 10 |   1 × 10 |         1 × 10 |               1 × 10 |

## Speed: model load

| model load (ms) | Go packed | Go float | engine library |
| --------------- | --------: | -------: | -------------: |
| median          |      15.3 |    311.2 |            5.7 |
| p90             |      20.0 |    340.8 |           16.1 |

`whistle.Load` maps `whistle.cact` and builds the model; `needle_load` takes the file's bytes already read into memory. The CLI's load is inside its process wall time.

## Against the published figures

| median of the 32 clips                    | TTFT (ms) | decode (tok/s) | TTFT / engine here | decode / engine here |
| ----------------------------------------- | --------: | -------------: | -----------------: | -------------------: |
| engine, Apple M4 Pro, 5 beams (published) |      11.1 |           1319 |                  — |                    — |
| engine library, this machine              |     160.9 |            523 |              1.00× |                1.00× |
| Go packed, this machine                   |      47.3 |            547 |              0.29× |                1.05× |
| Go float, this machine                    |      70.7 |            130 |              0.44× |                0.25× |

The published figures come from another machine and clip set; the engine on the same machine is the yardstick. A TTFT ratio above 1 is slower than the engine, a decode ratio below 1 too.

## Memory

| peak RSS (MiB, ru_maxrss of a fresh process) | Go packed | Go float | engine CLI |
| -------------------------------------------- | --------: | -------: | ---------: |
| after the model load                         |      35.6 |    235.0 |          — |
| one 30 s transcription, load included        |      99.8 |    477.5 |       52.2 |
| largest over the 32 clips, load included     |         — |        — |       43.1 |

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
# the Go side: bench/results/linux-amd64/go.json, then this file
just bench -go bench/results/linux-amd64/go.json -engine bench/results/linux-amd64/engine.json -report bench/RESULTS-linux-amd64.md
# the engine side: bench/results/linux-amd64/engine.json, then this file
just oracle-bench -out bench/results/linux-amd64/engine.json -go bench/results/linux-amd64/go.json -report bench/RESULTS-linux-amd64.md
# re-render this file from both JSON files
go run ./bench/cmd/whistle-bench report -go bench/results/linux-amd64/go.json -engine bench/results/linux-amd64/engine.json -report bench/RESULTS-linux-amd64.md
```

`just bench` needs `models/whistle/whistle.cact` (`whistle download`) and, for the WER,
`models/corpus/librispeech` (`just oracle-librispeech-audio`); `just oracle-bench` needs `engine/<platform>`
(`just fetch-engine`).
