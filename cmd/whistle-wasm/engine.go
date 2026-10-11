// Command whistle-wasm is the js/wasm build of the library for the web demo
// in web/. Built with GOOS=js GOARCH=wasm, it installs globalThis.whistle
// (main_wasm.go), which a Web Worker (web/worker.js) calls. Elsewhere it only
// prints how to build it; the engine below is plain Go so that it is tested
// natively too.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
)

// Errors of the engine's calls; the JS API rejects with their text.
var (
	errNoModel  = errors.New("no model loaded; call load first")
	errNoStream = errors.New("no stream open; call streamStart first")
	errStreamOn = errors.New("a stream is open; call streamStop first")
)

// engine is the state behind the JS API: one model and at most one stream.
// The JS calls of one page are serialised by its worker; the mutex keeps the
// goroutines of overlapping promises apart all the same.
type engine struct {
	mu     sync.Mutex
	model  *whistle.Model
	stream *whistle.Stream
}

// options are transcribe's options as the page passes them, a JS object
// with these keys, each optional; any other key is an error, so a typo does
// not go unnoticed.
type options struct {
	Language       string   `json:"language"`
	Keywords       []string `json:"keywords"`
	WordTimestamps bool     `json:"word_timestamps"`
	Depth          int      `json:"depth"`
}

// streamOptions are streamStart's options; a stream always has word
// timestamps and full depth (whistle.StreamOptions).
type streamOptions struct {
	Language string   `json:"language"`
	Keywords []string `json:"keywords"`
}

// modelInfo is what load resolves to.
type modelInfo struct {
	Languages []string `json:"languages"`
	Bytes     int      `json:"bytes"`
}

// decodeOptions reads the JSON of an options object into v; "", "null" and
// "undefined" (no object passed) leave v zero.
func decodeOptions(js string, v any) error {
	switch strings.TrimSpace(js) {
	case "", "null", "undefined":
		return nil
	}

	dec := json.NewDecoder(strings.NewReader(js))
	dec.DisallowUnknownFields()

	err := dec.Decode(v)
	if err != nil {
		return fmt.Errorf("options: %w", err)
	}

	return nil
}

// load builds a model on the bytes of whistle.cact and replaces the loaded
// one. A safetensors checkpoint has no tokenizer and fails. An open stream
// belongs to the old model and is dropped.
func (e *engine) load(data []byte) (modelInfo, error) {
	m, err := whistle.LoadBytes(data)
	if err != nil {
		return modelInfo{}, err
	}

	e.mu.Lock()
	old := e.model
	e.model, e.stream = m, nil
	e.mu.Unlock()

	if old != nil {
		_ = old.Close()
	}

	return modelInfo{Languages: m.Languages(), Bytes: len(data)}, nil
}

// transcribe is whistle.Model.Transcribe; it returns the engine's JSON, the
// line `whistle transcribe --json` prints.
func (e *engine) transcribe(pcm []float32, js string) ([]byte, error) {
	var o options

	err := decodeOptions(js, &o)
	if err != nil {
		return nil, err
	}

	m, err := e.loaded()
	if err != nil {
		return nil, err
	}

	res, err := m.Transcribe(pcm, whistle.TranscribeOptions{
		Language: o.Language, Keywords: o.Keywords, WordTimestamps: o.WordTimestamps, Depth: o.Depth,
	})
	if err != nil {
		return nil, err
	}

	return res.MarshalJSON()
}

// decodeWAV is the CLI's WAV reader on file contents: integer PCM of any
// rate and channel count, mixed to mono and resampled to 16 kHz with the
// same soxr-HQ filter, so a WAV the page decodes this way transcribes as
// `whistle transcribe` does.
func decodeWAV(data []byte) ([]float32, error) {
	pcm, err := frontend.DecodeWAV(data)
	if err != nil {
		return nil, fmt.Errorf("whistle: %w", err)
	}

	if len(pcm) == 0 {
		return nil, fmt.Errorf("whistle: WAV %w", whistle.ErrNoAudio)
	}

	return pcm, nil
}

// streamStart opens a stream on the loaded model.
func (e *engine) streamStart(js string) error {
	var o streamOptions

	err := decodeOptions(js, &o)
	if err != nil {
		return err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	switch {
	case e.model == nil:
		return errNoModel
	case e.stream != nil:
		return errStreamOn
	}

	s, err := e.model.NewStream(whistle.StreamOptions{Language: o.Language, Keywords: o.Keywords})
	if err != nil {
		return err
	}

	e.stream = s

	return nil
}

// streamProcess feeds a chunk to the open stream and returns the call's
// stream JSON.
func (e *engine) streamProcess(pcm []float32) ([]byte, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stream == nil {
		return nil, errNoStream
	}

	res, err := e.stream.Process(pcm)
	if err != nil {
		return nil, err
	}

	return res.MarshalJSON()
}

// streamStop commits the pending words, closes the stream and returns the
// stop's stream JSON.
func (e *engine) streamStop() ([]byte, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stream == nil {
		return nil, errNoStream
	}

	res := e.stream.Stop()
	e.stream = nil

	return res.MarshalJSON()
}

func (e *engine) loaded() (*whistle.Model, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.model == nil {
		return nil, errNoModel
	}

	return e.model, nil
}
