//go:build js && wasm

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"syscall/js"

	"github.com/cwbudde/go-whistle/whistle"
)

// version is reported as whistle.version.
const version = "0.1.0-wasm"

func main() {
	register(js.Global(), &engine{})

	select {}
}

// work is the part of a call that runs on a goroutine, after the arguments
// are in Go memory.
type work func() (any, error)

// register installs the JS API as target.whistle. Every function returns a
// promise and rejects it with an Error carrying the Go error's text.
//
//	load(bytes: Uint8Array | ArrayBuffer) → {languages, bytes}
//	sha256(bytes) → the SHA-256 as 64 lowercase hex digits, for pages
//	  without Web Crypto (crypto.subtle exists only in secure contexts)
//	decodeWAV(bytes) → Float32Array, 16 kHz mono, as `whistle transcribe` reads the file
//	transcribe(pcm: Float32Array, opts?) → the engine's JSON string
//	  opts: {language, keywords: string[], word_timestamps, depth}
//	streamStart(opts?) → undefined; opts: {language, keywords}
//	streamProcess(pcm: Float32Array) → the call's stream JSON string
//	streamStop() → the stop's stream JSON string
func register(target js.Value, e *engine) {
	api := map[string]any{
		"version":    version,
		"sampleRate": whistle.SampleRate,
		"maxSamples": whistle.MaxSamples,
		"load": async(func(args []js.Value) (work, error) {
			data, err := bytesArg(args, 0)
			if err != nil {
				return nil, err
			}

			return func() (any, error) {
				info, err := e.load(data)
				if err != nil {
					return nil, err
				}

				return jsonValue(info)
			}, nil
		}),
		"sha256": async(func(args []js.Value) (work, error) {
			sum, err := sha256Arg(args, 0)
			if err != nil {
				return nil, err
			}

			return func() (any, error) { return sum, nil }, nil
		}),
		"decodeWAV": async(func(args []js.Value) (work, error) {
			data, err := bytesArg(args, 0)
			if err != nil {
				return nil, err
			}

			return func() (any, error) {
				pcm, err := decodeWAV(data)
				if err != nil {
					return nil, err
				}

				return float32sToJS(pcm), nil
			}, nil
		}),
		"transcribe": async(func(args []js.Value) (work, error) {
			pcm, err := float32sArg(args, 0)
			if err != nil {
				return nil, err
			}

			opts := optionsArg(args, 1)

			return func() (any, error) { return jsonString(e.transcribe(pcm, opts)) }, nil
		}),
		"streamStart": async(func(args []js.Value) (work, error) {
			opts := optionsArg(args, 0)

			return func() (any, error) { return js.Undefined(), e.streamStart(opts) }, nil
		}),
		"streamProcess": async(func(args []js.Value) (work, error) {
			pcm, err := float32sArg(args, 0)
			if err != nil {
				return nil, err
			}

			return func() (any, error) { return jsonString(e.streamProcess(pcm)) }, nil
		}),
		"streamStop": async(func([]js.Value) (work, error) {
			return func() (any, error) { return jsonString(e.streamStop()) }, nil
		}),
	}

	target.Set("whistle", js.ValueOf(api))
}

// async wraps f as a JS function that returns a promise. f reads the
// arguments before the function returns, since JS may reuse or transfer
// their buffers afterwards; the work it returns then runs on a goroutine,
// as a wrapped function must not block the event loop (syscall/js.FuncOf).
// The functions live as long as the program, so they are never released.
func async(f func(args []js.Value) (work, error)) js.Func {
	return js.FuncOf(func(_ js.Value, args []js.Value) any {
		w, err := guard(func() (work, error) { return f(args) })

		var handler js.Func

		handler = js.FuncOf(func(_ js.Value, p []js.Value) any {
			handler.Release()

			resolve, reject := p[0], p[1]

			if err != nil {
				reject.Invoke(js.Global().Get("Error").New(err.Error()))

				return nil
			}

			go func() {
				v, err := guard(w)
				if err != nil {
					reject.Invoke(js.Global().Get("Error").New(err.Error()))

					return
				}

				resolve.Invoke(v)
			}()

			return nil
		})

		return js.Global().Get("Promise").New(handler)
	})
}

// guard runs f and turns a panic, which would end the program and every
// later call with it, into an error.
func guard[T any](f func() (T, error)) (v T, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("whistle-wasm: panic: %v", r)
		}
	}()

	return f()
}

// bytesArg copies args[i], a Uint8Array or an ArrayBuffer, into Go memory.
func bytesArg(args []js.Value, i int) ([]byte, error) {
	v, err := uint8ArrayArg(args, i)
	if err != nil {
		return nil, err
	}

	data := make([]byte, v.Get("byteLength").Int())
	js.CopyBytesToGo(data, v)

	return data, nil
}

// uint8ArrayArg is args[i], a Uint8Array or an ArrayBuffer, as a Uint8Array.
func uint8ArrayArg(args []js.Value, i int) (js.Value, error) {
	if i >= len(args) {
		return js.Undefined(), fmt.Errorf("argument %d: want a Uint8Array or an ArrayBuffer", i+1)
	}

	v := args[i]
	if v.InstanceOf(js.Global().Get("ArrayBuffer")) {
		v = js.Global().Get("Uint8Array").New(v)
	}

	if !v.InstanceOf(js.Global().Get("Uint8Array")) {
		return js.Undefined(), fmt.Errorf("argument %d: want a Uint8Array or an ArrayBuffer", i+1)
	}

	return v, nil
}

// sha256Block is how much of the bytes sha256Arg copies into Go at a time.
const sha256Block = 1 << 20

// sha256Arg hashes args[i], a Uint8Array or an ArrayBuffer, in blocks, so
// that a model (17 MB) is never copied into Go memory whole, and returns
// the hex digest.
func sha256Arg(args []js.Value, i int) (string, error) {
	v, err := uint8ArrayArg(args, i)
	if err != nil {
		return "", err
	}

	h := sha256.New()
	buf := make([]byte, sha256Block)

	for off, n := 0, v.Get("byteLength").Int(); off < n; off += sha256Block {
		k := js.CopyBytesToGo(buf, v.Call("subarray", off, min(off+sha256Block, n)))
		h.Write(buf[:k])
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// float32sArg copies args[i], a Float32Array, into Go memory.
func float32sArg(args []js.Value, i int) ([]float32, error) {
	if i >= len(args) || !args[i].InstanceOf(js.Global().Get("Float32Array")) {
		return nil, fmt.Errorf("argument %d: want a Float32Array of 16 kHz samples", i+1)
	}

	v := args[i]
	raw := make([]byte, v.Get("byteLength").Int())
	js.CopyBytesToGo(raw, js.Global().Get("Uint8Array").New(v.Get("buffer"), v.Get("byteOffset"), len(raw)))

	pcm := make([]float32, len(raw)/4)
	for k := range pcm {
		pcm[k] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*k:]))
	}

	return pcm, nil
}

// float32sToJS copies pcm into a new Float32Array.
func float32sToJS(pcm []float32) js.Value {
	raw := make([]byte, 4*len(pcm))
	for k, x := range pcm {
		binary.LittleEndian.PutUint32(raw[4*k:], math.Float32bits(x))
	}

	u8 := js.Global().Get("Uint8Array").New(len(raw))
	js.CopyBytesToJS(u8, raw)

	return js.Global().Get("Float32Array").New(u8.Get("buffer"))
}

// optionsArg is args[i] as JSON, "" when it is missing; the engine decodes
// it strictly.
func optionsArg(args []js.Value, i int) string {
	if i >= len(args) || args[i].IsUndefined() || args[i].IsNull() {
		return ""
	}

	return js.Global().Get("JSON").Call("stringify", args[i]).String()
}

// jsonString passes the engine's JSON on as a JS string.
func jsonString(b []byte, err error) (any, error) {
	if err != nil {
		return nil, err
	}

	return string(b), nil
}

// jsonValue turns v into a plain JS object through JSON.
func jsonValue(v any) (js.Value, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return js.Undefined(), errors.New("whistle-wasm: " + err.Error())
	}

	return js.Global().Get("JSON").Call("parse", string(b)), nil
}
