//go:build oracle

package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/ebitengine/purego"
)

// engine binds the libneedle3 C API (engine/<platform>/needle.h) through purego. The library keeps one
// process-global speech model, so an engine is not safe for concurrent use.
type engine struct {
	load      func(cact *byte, n uint64) int32
	lastError func() string
	embed     func(input *byte, pcm *float32, samples int32, out *float32, capacity int32) int32
	transcr   func(pcm *float32, samples int32, language, keywords *byte, words int32, out *byte, capacity int32) int32
}

// platformDir names the engine/<platform> directory that scripts/fetch-engine.sh writes for this machine.
func platformDir() (string, error) {
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return "macos-arm64", nil
	case "linux/amd64":
		return "linux-x86_64", nil
	case "linux/arm64":
		return "linux-arm64", nil
	default:
		return "", fmt.Errorf("no Needle 3 engine build for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

func libName() string {
	if runtime.GOOS == "darwin" {
		return "libneedle3.dylib"
	}

	return "libneedle3.so"
}

// openEngine loads libneedle3 from dir and the speech model from cact.
func openEngine(dir string, cact []byte) (*engine, error) {
	if len(cact) == 0 {
		return nil, errors.New("empty .cact")
	}

	path := filepath.Join(dir, libName())

	_, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	lib, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", path, err)
	}

	e := &engine{}
	purego.RegisterLibFunc(&e.load, lib, "needle_load")
	purego.RegisterLibFunc(&e.lastError, lib, "needle_last_error")
	purego.RegisterLibFunc(&e.embed, lib, "needle_embed")
	purego.RegisterLibFunc(&e.transcr, lib, "needle_transcribe")

	rc := e.load(&cact[0], uint64(len(cact)))
	runtime.KeepAlive(cact)

	if rc < 0 {
		return nil, fmt.Errorf("needle_load: %s (%d)", e.lastError(), rc)
	}

	return e, nil
}

// embedSpeech returns needle_embed's flattened per-frame speech embedding of 16 kHz mono pcm.
func (e *engine) embedSpeech(pcm []float32) ([]float32, error) {
	if len(pcm) == 0 {
		return nil, errors.New("empty pcm")
	}

	n := e.embed(nil, &pcm[0], int32(len(pcm)), nil, 0) // #nosec G115 -- clips are at most 30 s.
	if n <= 0 {
		return nil, fmt.Errorf("needle_embed size: %s (%d)", e.lastError(), n)
	}

	out := make([]float32, n)

	got := e.embed(nil, &pcm[0], int32(len(pcm)), &out[0], n) // #nosec G115 -- clips are at most 30 s.
	runtime.KeepAlive(pcm)

	if got != n {
		return nil, fmt.Errorf("needle_embed: %s (returned %d, want %d)", e.lastError(), got, n)
	}

	return out, nil
}

// transcribeOutCap is the JSON buffer needle_transcribe writes into; 30 s of speech with word times needs ~40 KB.
const transcribeOutCap = 1 << 20

// libResult is needle_transcribe's outcome: its return code, its JSON and, on failure, needle_last_error. Empty and
// SameAsCLI are filled in by the limits run.
type libResult struct {
	RC        int32  `json:"rc"`
	Error     string `json:"error,omitempty"`
	Empty     bool   `json:"empty"`
	SameAsCLI bool   `json:"same_as_cli"`
	out       []byte
}

// transcribePCM runs needle_transcribe on 16 kHz mono pcm with language detection, no keywords and word times on.
// An empty pcm is passed as a valid pointer with zero samples.
func (e *engine) transcribePCM(pcm []float32) libResult {
	buf := make([]byte, transcribeOutCap)
	ptr := new(float32)

	if len(pcm) > 0 {
		ptr = &pcm[0]
	}

	rc := e.transcr(ptr, int32(len(pcm)), nil, nil, 1, &buf[0], int32(len(buf))) // #nosec G115 -- at most 30 s.
	runtime.KeepAlive(pcm)

	if rc < 0 {
		return libResult{RC: rc, Error: e.lastError()}
	}

	n := bytes.IndexByte(buf, 0)
	if n < 0 {
		n = len(buf)
	}

	return libResult{RC: rc, out: buf[:n]}
}
