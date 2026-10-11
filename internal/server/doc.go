// Package server is the HTTP interface of `whistle serve`: one loaded model
// behind four endpoints.
//
// POST /transcribe takes the audio as the request body and answers with the
// engine's JSON, the line `whistle transcribe --json` prints for the same
// file and options (docs/FINDINGS.md F-027). The body is a RIFF WAV file
// (integer PCM, any channels and rate) or, with ?format=s16le or
// ?format=f32le, raw 16 kHz mono PCM; the Content-Type is not read. The
// options are query parameters: language (en, de, fr, es, it, nl, pl),
// keywords (repeatable, one word or phrase each; newlines split an entry),
// depth (2..8) and word_timestamps (true/false; empty means true). Status
// codes: 200 with the result; 400 for a bad option, an unknown parameter or
// audio that does not decode or holds no samples; 408 for a body that takes
// longer than Config.ReadTimeout; 413 for a body over Config.MaxBody; 422
// for audio over 30 s, the engine's "audio limit is 30 s"; 503 when
// Config.Concurrency passes run and Config.Queue requests wait already; 405
// for a method other than POST. Errors are JSON: {"error":"..."}.
//
// GET /stream upgrades to a WebSocket and runs one whistle.Stream, the
// engine's needle_stream_transcribe_process and _stop (F-032, F-036). The
// query takes language, keywords and format: s16le (the default) or f32le,
// 16 kHz mono. Each binary message is one chunk of PCM, at most 30 s, fed
// to one Process call and answered with one text message, the call's stream
// JSON as `whistle stream --json` prints it. The text message
// {"type":"stop"} runs Stop, sends its result and closes the connection
// with 1000. A close from the client ends the stream without Stop. When the
// server ends a session itself, it sends the Stop result first and closes
// with 1001 on shutdown or 1008 after Config.StreamIdle without a message.
// A chunk that ends inside a sample closes with 1007, any other text with
// 1003, a chunk over 30 s with 1009. Before the upgrade the endpoint
// answers 400 for bad options and 503 when Config.MaxStreams sessions are
// open.
//
// GET /healthz answers 200 "ok" (503 while shutting down); GET /v1/info
// the model file, its SHA-256, format, weights mode, languages and the
// server's limits.
//
// A *whistle.Model is safe for concurrent use (its scratch buffers are
// pooled, the decoder ladder is built at Load), but each pass already uses
// every core, so the server bounds the passes that run at once: a
// /transcribe request and every Process call of a stream take one of
// Config.Concurrency slots. Transcribe requests wait in a queue of
// Config.Queue and get 503 beyond it, before their body is read, so at most
// Concurrency + Queue bodies are in memory; a stream's chunk waits for its
// slot, as the WebSocket's flow control holds the client back meanwhile.
package server
