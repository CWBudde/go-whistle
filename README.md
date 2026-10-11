# go-whistle

A pure-Go port of **Whistle**, Cactus Compute's 16.9 MB on-device speech-to-text model (55M parameters, 7 languages:
en, de, fr, es, it, nl, pl).

> **Status: working, not yet tuned.** Transcription with language detection, keyword biasing, the depth ladder and
> word timestamps, live streaming and embeddings all run in pure Go and match the reference engine closely (see
> [Parity](#parity-with-the-engine)). The model runs straight from the packed 2/4-bit `whistle.cact`. Performance work
> (Phase 9) is under way; the engine is still faster, several times so to the first token. See [PLAN.md](PLAN.md)
> for the roadmap.

## Quickstart

Install the CLI from a clone (Go 1.25 or later; pure Go, no cgo; the repository is not public yet) and fetch the
pinned model files:

```bash
git clone https://github.com/CWBudde/go-whistle && cd go-whistle
go install ./cmd/whistle
whistle download
```

```text
ok       whistle.cact (sha256 verified)
ok       whistle.safetensors (sha256 verified)
ok       config.json (sha256 verified)
whistle b358ddad ready in models/whistle
```

`download` verifies each file by size and SHA-256 and resumes interrupted downloads. The commands below use
`models/whistle/whistle.cact` by default (`--model` picks another, also `whistle.safetensors`) and the fixture clips
in `testdata/audio`.

### Check the install

```bash
whistle doctor
```

```text
model      models/whistle/whistle.cact
size       16919407 bytes (16.9 MB)
format     .cact (2/4-bit CQ weights with the tokenizer)
sha256     b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb = pinned Cactus-Compute/whistle@b358ddad whistle.cact
files      whistle.safetensors ok, config.json ok
weights    packed: CQ matrices straight from the file (whistle.Load's default; WithFloatWeights dequantises)
go         go1.27.1 darwin/arm64
kernels    NEON: GEMM, dot, axpy, CQ LUT GEMV, softmax exp
workers    GOMAXPROCS 15, 15 CPUs
load       ok in 9ms; languages en de fr es it nl pl
self-test  2 s tone burst: embedding 25×512, transcript [en] "Thank you." in 50ms
clip       testdata/audio/ls-other-3080-5040-0018.wav: [en] "Directed for your master." in 37ms
all checks passed
```

It exits with 1 when a check fails: a missing or modified model file, a failed load or self-test.

### Transcribe

```bash
whistle transcribe testdata/audio/ls-other-3080-5040-0018.wav
```

```text
Directed for your master.
```

`--json` prints the engine's JSON object byte for byte; `--word-timestamps` adds each word's start, end and
probability:

```bash
whistle transcribe --json --word-timestamps testdata/audio/ls-other-3080-5040-0018.wav
```

```json
{
  "text": "Directed for your master.",
  "language": "en",
  "words": [
    { "word": "Directed", "start": 0.64, "end": 1.12, "probability": 0.894 },
    { "word": "for", "start": 1.12, "end": 1.52, "probability": 0.739 },
    { "word": "your", "start": 1.52, "end": 1.68, "probability": 0.98 },
    { "word": "master.", "start": 1.68, "end": 2.08, "probability": 0.962 }
  ],
  "ttft_ms": 27.3,
  "decode_tps": 860.3
}
```

`--keywords` names a file of words and phrases to favour, one per line:

```bash
printf "Cowley's\n" > keywords.txt
whistle transcribe testdata/audio/ls-other-3080-5040-0011.wav
whistle transcribe --keywords keywords.txt testdata/audio/ls-other-3080-5040-0011.wav
```

```text
Here are some verses of colonies. Tell me how you like them.
Here are some verses of Cowley's. Tell me how you like them.
```

`--language` forces the language, and `--depth 2..8` runs fewer decoder layers (faster, less accurate). Several files
load the model once and print each text after its path, or a JSON line each:

```bash
whistle transcribe testdata/audio/ls-other-3080-5040-0011.wav testdata/audio/fleurs-de_de-10229344228128634115.wav
```

```text
testdata/audio/ls-other-3080-5040-0011.wav: Here are some verses of colonies. Tell me how you like them.
testdata/audio/fleurs-de_de-10229344228128634115.wav: Jeder nimmt an der Gesellschaft Teil und benutzt Transportsysteme, was jeder beklagt sich über die Transportsysteme.
```

One call takes at most 30 s, as in the engine. A longer file (here the nine `ls-clean` clips of the stream fixtures
joined, 44.91 s) fails with exit code 1:

```text
$ whistle transcribe long.wav
whistle: long.wav: audio limit is 30 s: 718560 samples, at most 480000 (transcribe takes one pass of at most 30 s; whistle stream takes longer audio)
```

### Stream

`whistle stream` transcribes audio of any length live, as the engine's stream API does: each chunk (`--chunk`
seconds, default 1) is one call that re-transcribes the buffer and commits the words two consecutive passes agree
on; the end of the input commits the rest.

```bash
whistle stream long.wav
```

```text
   1.00 s  commit ""  pending "Come and call."
   2.00 s  commit ""  pending "Concord returned to its"
   3.00 s  commit "Concord returned to its"  pending "place amidst the 10."
   4.00 s  commit "place amidst the"  pending "tents."
   5.00 s  commit "tents."  pending "\"You will be frank with me.\""
…
  44.00 s  commit "help protect"  pending "yourself if worse come.\""
  44.91 s  commit "yourself if worse"  pending "comes to worst."
  44.91 s  stop   "comes to worst."
Concord returned to its place amidst the tents. "You will be frank with me?" "I always am." There was something in his air and manner that betrayed to the scout the utter confusion of the state of his mind. Four or five of the latter only lingered about the door of the prison of Uncas, wary but close observers of the manner of their captive. She found the door, but it was locked outside. Among other things on which she cast her eyes was a small crucifix of solid silver, standing on a cabinet near the window. "I am very tired of swimming about here. Oh, Mouse!" "I almost think I can remember feeling little different," Captain Martin said. "I shall give you a pistol to help protect yourself if worse comes to worst.
```

Without a file (or with `-`) it reads raw 16 kHz mono PCM from stdin: `--format s16le` (signed 16-bit little-endian,
the default) or `f32le` (float32 little-endian in [-1, 1]). `--json` prints each call as the engine's stream JSON, and
`--commits-only` only the calls that commit words plus the stop, as the engine's `--audio-stream` does:

```bash
ffmpeg -loglevel error -i testdata/audio/ls-clean-6930-75918-0000.wav -f s16le -ac 1 -ar 16000 - | whistle stream --json --commits-only
```

```json
{"text":"Concord returned to its","words":[{"word":"Concord","start":0.56,"end":1.20,"probability":0.909},{"word":"returned","start":1.20,"end":1.60,"probability":0.843},{"word":"to","start":1.60,"end":1.84,"probability":0.990},{"word":"its","start":1.84,"end":2.00,"probability":0.998}],"pending":"place amidst the 10.","language":"en","received":3.00,"pass_ms":56.0}
{"text":"place amidst the","words":[{"word":"place","start":2.00,"end":2.24,"probability":0.998},{"word":"amidst","start":2.24,"end":2.80,"probability":0.977},{"word":"the","start":2.80,"end":2.96,"probability":0.995}],"pending":"tents.","language":"en","received":3.50,"pass_ms":63.0}
{"text":"tents.","words":[{"word":"tents.","start":2.96,"end":3.28,"probability":0.949}],"pending":"","language":"en","received":3.50,"pass_ms":0.0}
```

A microphone works the same way, for example `arecord -q -f S16_LE -r 16000 -c 1 -t raw | whistle stream`; Ctrl-C
ends the stream and commits the pending words. `--realtime` plays a file no faster than real time.

### Embed

`whistle embed` writes the encoder output, what the engine's `needle_embed` returns: one row of 512 values per 80 ms
frame, at most 30 s of audio. JSON by default, or a NumPy `.npy` file:

```bash
whistle embed testdata/audio/ls-other-3080-5040-0018.wav | head -c 120
whistle embed --format npy -o clip.npy testdata/audio/ls-other-3080-5040-0018.wav
```

```text
{"frames":33,"dim":512,"frame_ms":80,"embedding":[[-1.322341,-3.7772563,-1.3063984,-0.8534372,-1.0002542,-1.1180489,0.65
```

`numpy.load("clip.npy")` gives a `float32` array of shape `(33, 512)`.

### Server

`whistle serve` loads the model once and serves it over HTTP on `127.0.0.1:8080` (`--addr`), with a log line per
request on stderr. `POST /transcribe` takes a WAV file as the body, or raw 16 kHz mono PCM with `?format=s16le` or
`f32le`. Its options are query parameters: `language`, `keywords` (repeatable), `depth` and `word_timestamps`. It
answers with the JSON `whistle transcribe --json` prints:

```bash
whistle serve
curl -s --data-binary @testdata/audio/ls-other-3080-5040-0018.wav 'http://127.0.0.1:8080/transcribe?word_timestamps=true'
ffmpeg -loglevel error -i testdata/audio/ls-clean-6930-75918-0000.wav -f s16le -ac 1 -ar 16000 talk.raw
curl -s --data-binary @talk.raw 'http://127.0.0.1:8080/transcribe?format=s16le&language=en'
```

```json
{"text":"Directed for your master.","language":"en","words":[{"word":"Directed","start":0.64,"end":1.12,"probability":0.894},{"word":"for","start":1.12,"end":1.52,"probability":0.739},{"word":"your","start":1.52,"end":1.68,"probability":0.980},{"word":"master.","start":1.68,"end":2.08,"probability":0.962}],"ttft_ms":16.1,"decode_tps":947.3}
{"text":"Concord returned to its place amidst the tents.","language":"en","ttft_ms":14.0,"decode_tps":975.3}
```

Errors are JSON `{"error":"..."}`: 400 for bad options or audio, 408 for a body slower than `--read-timeout`, 413
for a body over `--max-body-mib` (16 MiB), 422 for audio over 30 s, 503 when `--concurrency` passes run and `--queue`
requests wait. `GET /healthz` and `GET /v1/info` (the model's SHA-256, weights mode, languages and the limits) are
there too.

`GET /stream` is a WebSocket (`?format=s16le|f32le`, `language`, `keywords`). Each binary message is a chunk of
16 kHz mono PCM for one stream call and gets one text message back, the call's stream JSON as `whistle stream --json`
prints it. The text message `{"type":"stop"}` commits the rest, sends that result and closes the connection. A Go
client with [`github.com/coder/websocket`](https://github.com/coder/websocket) (error checks shortened to `check`):

```go
c, _, err := websocket.Dial(ctx, "ws://127.0.0.1:8080/stream?format=s16le", nil)
check(err)

for i := 0; i < len(pcm); i += 32000 { // 1 s of s16le per call
	check(c.Write(ctx, websocket.MessageBinary, pcm[i:min(i+32000, len(pcm))]))

	_, msg, err := c.Read(ctx)
	check(err)
	fmt.Println(string(msg))
}

check(c.Write(ctx, websocket.MessageText, []byte(`{"type":"stop"}`)))

_, msg, err := c.Read(ctx) // the stop's result
check(err)
fmt.Println(string(msg))

_, _, err = c.Read(ctx) // then the server closes
fmt.Println("close:", websocket.CloseStatus(err))
```

With `pcm` read from `talk.raw`:

```text
{"text":"","words":[],"pending":"Come and call.","language":"en","received":1.00,"pass_ms":18.5}
{"text":"","words":[],"pending":"Concord returned to its","language":"en","received":2.00,"pass_ms":21.3}
{"text":"Concord returned to its","words":[{"word":"Concord","start":0.56,"end":1.20,"probability":0.909},{"word":"returned","start":1.20,"end":1.60,"probability":0.843},{"word":"to","start":1.60,"end":1.84,"probability":0.990},{"word":"its","start":1.84,"end":2.00,"probability":0.998}],"pending":"place amidst the 10.","language":"en","received":3.00,"pass_ms":34.8}
{"text":"place amidst the","words":[{"word":"place","start":2.00,"end":2.24,"probability":0.998},{"word":"amidst","start":2.24,"end":2.80,"probability":0.977},{"word":"the","start":2.80,"end":2.96,"probability":0.995}],"pending":"tents.","language":"en","received":3.50,"pass_ms":34.0}
{"text":"tents.","words":[{"word":"tents.","start":2.96,"end":3.28,"probability":0.949}],"pending":"","language":"en","received":3.50,"pass_ms":0.0}
close: StatusNormalClosure
```

Ctrl-C (or SIGTERM) shuts the server down: running requests finish, and open streams get their stop result and a
1001 close. `whistle serve --help` lists the limits; the `internal/server` package doc has the whole protocol.

All commands exit with 0 on success, 1 when they fail and 2 for a wrong command line.

## Go API

The library is the package `github.com/cwbudde/go-whistle/whistle`. The repository is not public yet, so `go get`
cannot fetch it; point your module at a clone instead:

```bash
go mod edit -replace github.com/cwbudde/go-whistle=../go-whistle
go get github.com/cwbudde/go-whistle
go mod tidy
```

```go
package main

import (
	"fmt"
	"log"

	"github.com/cwbudde/go-whistle/whistle"
)

func main() {
	m, err := whistle.Load("models/whistle/whistle.cact")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	// One call: a WAV file (or Transcribe on 16 kHz mono float32 samples), at most 30 s.
	res, err := m.TranscribeWAV("testdata/audio/ls-other-3080-5040-0011.wav", whistle.TranscribeOptions{
		Keywords:       []string{"Cowley's"},
		WordTimestamps: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("[%s] %s\n", res.Language, res.Text)
	for _, w := range res.Words[:3] {
		fmt.Printf("%s %v-%v p=%.3f\n", w.Text, w.Start, w.End, w.Probability)
	}

	// The engine's JSON, byte for byte.
	js, err := res.MarshalJSON()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(js[:60]))

	// Live audio of any length: about a second per Process call, Stop at the end.
	s, err := m.NewStream(whistle.StreamOptions{Language: "en"})
	if err != nil {
		log.Fatal(err)
	}
	second := make([]float32, whistle.SampleRate) // here: a second of silence
	for range 2 {
		r, err := s.Process(second)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%.2f s: committed %q, pending %q\n", r.Received, r.Text, r.Pending)
	}
	fmt.Printf("stop: committed %q\n", s.Stop().Text)

	// The encoder output, one row of 512 values per 80 ms frame.
	e, err := m.Embed(second)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(e.Frames, e.Dim)
}
```

Built that way and run from the clone's root after `whistle download`, it prints:

```text
[en] Here are some verses of Cowley's. Tell me how you like them.
Here 640ms-800ms p=0.380
are 800ms-880ms p=0.998
some 880ms-1.04s p=1.000
{"text":"Here are some verses of Cowley's. Tell me how you l
1.00 s: committed "", pending ""
2.00 s: committed "", pending ""
stop: committed ""
13 512
```

The package documentation (`go doc github.com/cwbudde/go-whistle/whistle`) covers the options, the JSON forms and the
streaming rules; [`whistle/example_test.go`](whistle/example_test.go) has runnable examples.

## Parity with the engine

Whistle's engine (`libneedle3`) is closed source, so this port is rebuilt from the published weights and checked
against the engine as a black-box oracle. The engine runs W2A8 numerics, so the goal is equal transcripts, not equal
bits. Measured with the default packed weights:

| What                                                    | Go port                                     | Engine | Details                                                                                                                                                                                                                                                 |
| ------------------------------------------------------- | ------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transcript and language equal to the engine's, 32 clips | 31/32                                       | —      | [F-031](docs/FINDINGS.md#f-031--the-silence-gate-beam-search-in-transcribe-and-the-librispeech-check-u10-2026-10-10)                                                                                                                                    |
| LibriSpeech test-clean subset, 200 utterances, WER      | 5.35 %                                      | 5.30 % | [F-031](docs/FINDINGS.md#f-031--the-silence-gate-beam-search-in-transcribe-and-the-librispeech-check-u10-2026-10-10), [F-037](docs/FINDINGS.md#f-037--the-packed-path-cq-matrices-straight-from-whistlecact-in-every-linear-layer-92-2026-10-10)        |
| Keyword cases equal to the engine's                     | 45/53                                       | —      | [F-029](docs/FINDINGS.md#f-029--keyword-biasing-an-aho-corasick-bonus-of-2-nats-per-token-revoked-on-failure-u9-2026-10-10), [F-037](docs/FINDINGS.md#f-037--the-packed-path-cq-matrices-straight-from-whistlecact-in-every-linear-layer-92-2026-10-10) |
| Depth ladder 2–7 equal to the engine's                  | 161/192                                     | —      | [F-030](docs/FINDINGS.md#f-030--the-depth-ladder-upstreams-sub-network-lanes-by-position-2026-10-10)                                                                                                                                                    |
| Word start and end within 80 ms of the engine's         | 97.4 % of 459 words                         | —      | [F-033](docs/FINDINGS.md#f-033--word-timestamps-whispers-alignment-on-the-last-half-of-the-decoder-layers-u11-2026-10-10)                                                                                                                               |
| Embedding vs `needle_embed`, mean per-frame cosine      | 0.999827, every clip ≥ 0.999                | —      | [F-027](docs/FINDINGS.md#f-027--the-public-pipeline-the-engines-json-the-30-s-limit-and-its-timing-2026-10-10), [F-037](docs/FINDINGS.md#f-037--the-packed-path-cq-matrices-straight-from-whistlecact-in-every-linear-layer-92-2026-10-10)              |
| Stream: calls with the engine's decision                | 797/930; final text equal on 98 of 108 runs | —      | [F-036](docs/FINDINGS.md#f-036--stream-the-engines-stream-in-go-call-by-call-u12-part-2-planmd-83-84-2026-10-10)                                                                                                                                        |
| Stream at 1 s chunks: final text equal to offline       | 18/32                                       | 18/32  | [F-036](docs/FINDINGS.md#f-036--stream-the-engines-stream-in-go-call-by-call-u12-part-2-planmd-83-84-2026-10-10)                                                                                                                                        |
| Result and stream JSON                                  | byte-identical in fields, order and numbers | —      | [F-027](docs/FINDINGS.md#f-027--the-public-pipeline-the-engines-json-the-30-s-limit-and-its-timing-2026-10-10), [F-036](docs/FINDINGS.md#f-036--stream-the-engines-stream-in-go-call-by-call-u12-part-2-planmd-83-84-2026-10-10)                        |

Every remaining difference is a near tie under our model, explained in the linked findings.

## Limits

- **30 s per call** for `transcribe`, `Transcribe` and `Embed`, the engine's limit (`ErrTooLong`, "audio limit is
  30 s"). A stream has no limit; it trims committed audio once its buffer holds more than 15 s.
- **Languages:** en, de, fr, es, it, nl, pl, detected per call or forced.
- **WAV input** is integer PCM (8/16/24/32-bit, any channels and rate), as upstream's `whistle.py` reads it. IEEE
  float WAVs are rejected; convert them, or pipe raw `f32le` into `whistle stream`.
- **Speed:** slower than the engine, which is hand-tuned C++ with W2A8 kernels; see
  [F-035](docs/FINDINGS.md#f-035--internalneedle-hot-loops-gemms-a-float64-sinkhorn-and-a-vector-exp-2026-10-10) and
  [F-037](docs/FINDINGS.md#f-037--the-packed-path-cq-matrices-straight-from-whistlecact-in-every-linear-layer-92-2026-10-10)
  for the current numbers and PLAN.md Phase 9 for the work under way.

## Why this is a re-implementation, not a translation

In [cactus-compute/needle](https://github.com/cactus-compute/needle), Whistle is a thin Python wrapper around a
closed-source C++ engine. So this port rebuilds the model from three sources:

- the published weights on [Hugging Face](https://huggingface.co/Cactus-Compute/whistle): `whistle.safetensors` (FP32)
  and `whistle.cact` (2/4-bit quantised)
- the open JAX reference for the shared Needle 3 blocks
- the prebuilt engine, which is used only as a test oracle for encoder embeddings and transcripts

## Goals

- No cgo, no Python and no ONNX at runtime
- Load both `whistle.safetensors` and the compact `whistle.cact`
- An API and JSON output compatible with upstream `needle.transcribe`, including language detection, keyword biasing,
  word timestamps, streaming and embeddings
- Transcripts that match the reference engine on a fixture set, and competitive CPU speed

## License

Apache-2.0, the same as the upstream code and weights. Whistle and Needle are by
[Cactus Compute](https://cactuscompute.com); this project is not affiliated with them.
