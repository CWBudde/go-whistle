// Package whistle is a pure-Go port of Whistle, the 55M-parameter speech-to-text model of Cactus Compute: no cgo, no
// Python and no ONNX at run time. It is rebuilt from the published weights and checked against the closed-source
// engine (libneedle3) as a test oracle; docs/FINDINGS.md records every decision.
//
// # Loading
//
// Load opens the deployed whistle.cact (2/4-bit CQ weights with the tokenizer, 16.9 MB) or the float32
// whistle.safetensors checkpoint, which takes its tokenizer from the whistle.cact next to it. `whistle download`
// fetches both at the pinned revision. A .cact model multiplies straight from its packed weights, mapped from the
// file; WithFloatWeights dequantises them to float32 instead (about 221 MB), and WithoutMmap reads the file into
// memory.
//
//	m, err := whistle.Load("models/whistle/whistle.cact")
//	if err != nil {
//		return err
//	}
//	defer m.Close()
//
// A Model is safe for concurrent use. Close it when done, and not while a call runs.
//
// # Transcribing
//
// Transcribe takes 16 kHz mono float32 samples in [-1, 1], at most MaxSamples (30 s); TranscribeWAV reads a RIFF
// WAV file with integer PCM of any width, channel count and rate. A longer clip fails with ErrTooLong, whose text
// is the engine's own, "audio limit is 30 s"; use a Stream for longer audio.
//
//	res, err := m.TranscribeWAV("clip.wav", whistle.TranscribeOptions{})
//	fmt.Println(res.Language, res.Text) // en Directed for your master.
//
// Decoding is the engine's: a silence gate first, so silence and steady noise give the empty Result without
// decoding, then a 5-beam search that detects the language among Languages (en, de, fr, es, it, nl, pl).
// TranscribeOptions forces the Language, favours Keywords (the engine's keyword biasing), runs fewer decoder layers
// (Depth 2..8, the engine's ladder) and adds WordTimestamps: every word with its start, end and probability on the
// 80 ms frame grid.
//
// Result marshals to the engine's JSON byte for byte, field order and number formatting included:
//
//	{"text":"Directed for your master.","language":"en","ttft_ms":34.9,"decode_tps":852.4}
//
// with "words" between language and ttft_ms when word timestamps are on. Call MarshalJSON directly for those
// bytes; json.Marshal re-escapes '<', '>' and '&'.
//
// # Streaming
//
// NewStream is live transcription without a length limit, the engine's needle_stream_transcribe_process and _stop.
// Each Process call appends a chunk, about a second is the engine's advice, re-transcribes the buffer and commits
// the words two consecutive passes agree on; Stop commits the rest and ends the stream. Each StreamResult marshals
// to the engine's stream JSON.
//
//	s, err := m.NewStream(whistle.StreamOptions{})
//	for chunk := range chunks {
//		r, err := s.Process(chunk)
//		fmt.Print(r.Text, " ") // the words committed by this call
//	}
//	fmt.Println(s.Stop().Text) // the rest
//
// A Stream belongs to one goroutine; several Streams and Transcribe calls can share a Model.
//
// # Embeddings
//
// Embed returns what the engine's needle_embed returns: the encoder output, one row of 512 values per 80 ms frame
// (FrameDuration), at most 30 s of audio.
//
// # Parity with the engine
//
// On the 32 fixture clips the transcript and language equal the engine's on 31; on a 200-utterance LibriSpeech
// test-clean subset the word error rate is 5.35 % against the engine's 5.30 % (FINDINGS F-031). Embeddings reach a
// mean per-frame cosine of 0.9998 against needle_embed (F-027, F-037). Word times are within 80 ms of the engine's
// for 97.4 % of the words (F-033). Streams make the engine's decision on 797 of 930 recorded calls, and their final
// text equals the engine's stream on 98 of 108 runs (F-036). The engine runs W2A8 numerics, so the port aims for
// equal transcripts, not equal bits.
package whistle
