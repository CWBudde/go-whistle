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

// Model kinds needle_models reports (needle.h NEEDLE_TEXT, NEEDLE_SPEECH).
const (
	needleText   = 1
	needleSpeech = 2
)

// completeOutCap is the JSON buffer needle_complete writes into; a 512-token answer with reasoning and word times
// stays far below it.
const completeOutCap = 1 << 20

// library binds the libneedle3 C API (engine/<platform>/needle.h) through purego. The library keeps one process-global
// model per kind and one session, so a library is not safe for concurrent use.
type library struct {
	load      func(cact *byte, n uint64) int32
	models    func() int32
	lastError func() string
	init      func(system, tools, toolIndex *byte) int32
	complete  func(input *byte, pcm *float32, samples, maxNewTokens int32, out *byte, capacity int32) int32
	setAudio  func(language, keywords *byte, wordTimestamps int32)
	reset     func()
	embed     func(input *byte, pcm *float32, samples int32, out *float32, capacity int32) int32
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

// openLibrary sets the telemetry opt-outs for this process (the CLI gets them per run), loads libneedle3 from dir and
// then every .cact in order, each with needle_load.
func openLibrary(dir string, cacts ...string) (*library, error) {
	for k, v := range telemetryEnv {
		err := os.Setenv(k, v)
		if err != nil {
			return nil, fmt.Errorf("setenv %s: %w", k, err)
		}
	}

	path := filepath.Join(dir, libName())

	_, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", path, err)
	}

	l := &library{}
	purego.RegisterLibFunc(&l.load, h, "needle_load")
	purego.RegisterLibFunc(&l.models, h, "needle_models")
	purego.RegisterLibFunc(&l.lastError, h, "needle_last_error")
	purego.RegisterLibFunc(&l.init, h, "needle_init")
	purego.RegisterLibFunc(&l.complete, h, "needle_complete")
	purego.RegisterLibFunc(&l.setAudio, h, "needle_set_audio")
	purego.RegisterLibFunc(&l.reset, h, "needle_reset")
	purego.RegisterLibFunc(&l.embed, h, "needle_embed")

	for _, p := range cacts {
		err := l.loadFile(p)
		if err != nil {
			return nil, err
		}
	}

	return l, nil
}

// loadFile runs needle_load on the .cact at p.
func (l *library) loadFile(p string) error {
	cact, err := os.ReadFile(p) // #nosec G304 -- the model dirs are flags.
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}

	if len(cact) == 0 {
		return fmt.Errorf("%s is empty", p)
	}

	rc := l.load(&cact[0], uint64(len(cact)))
	runtime.KeepAlive(cact)

	if rc < 0 {
		return fmt.Errorf("needle_load %s: %s (%d)", p, l.lastError(), rc)
	}

	return nil
}

// session runs needle_init with the system facts and the tools JSON (no tool index) and returns its result, the
// tokenized static-prefix length.
func (l *library) session(system, tools string) (int32, error) {
	return l.sessionIndex(system, tools, "")
}

// sessionIndex is session with a tool index path ("" for none), the file in which needle_init keeps the tool set's
// embeddings when it retrieves.
func (l *library) sessionIndex(system, tools, index string) (int32, error) {
	sys, tls, idx := cString(system), cString(tools), cString(index)

	rc := l.init(sys, tls, idx)
	runtime.KeepAlive(sys)
	runtime.KeepAlive(tls)
	runtime.KeepAlive(idx)

	if rc < 0 {
		return rc, fmt.Errorf("needle_init: %s (%d)", l.lastError(), rc)
	}

	return rc, nil
}

// completion is one needle_complete call: its return code and the JSON it wrote.
type completion struct {
	rc  int32
	out []byte
}

// completeText runs needle_complete on a text turn.
func (l *library) completeText(input string, maxNewTokens int) (completion, error) {
	in := cString(input)
	if in == nil {
		in = &[]byte{0}[0] // an empty turn is still a text turn, not a speech one
	}

	return l.run(func(out *byte, capacity int32) int32 {
		rc := l.complete(in, nil, 0, int32(maxNewTokens), out, capacity) // #nosec G115 -- a small token budget.
		runtime.KeepAlive(in)

		return rc
	})
}

// completeSpeech runs needle_complete on 16 kHz mono pcm: the engine transcribes it with the speech model and
// answers the transcript.
func (l *library) completeSpeech(pcm []float32, maxNewTokens int) (completion, error) {
	if len(pcm) == 0 {
		return completion{}, errors.New("empty pcm")
	}

	return l.run(func(out *byte, capacity int32) int32 {
		rc := l.complete(nil, &pcm[0], int32(len(pcm)), int32(maxNewTokens), out, capacity) // #nosec G115 -- at most 30 s.
		runtime.KeepAlive(pcm)

		return rc
	})
}

func (l *library) run(call func(out *byte, capacity int32) int32) (completion, error) {
	buf := make([]byte, completeOutCap)

	rc := call(&buf[0], int32(len(buf)))
	if rc < 0 {
		return completion{rc: rc}, fmt.Errorf("needle_complete: %s (%d): %q", l.lastError(), rc, cOut(buf))
	}

	return completion{rc: rc, out: cOut(buf)}, nil
}

// embedText returns needle_embed's embedding of a text.
func (l *library) embedText(text string) ([]float32, error) {
	in := cString(text)
	if in == nil {
		return nil, errors.New("empty text")
	}

	n := l.embed(in, nil, 0, nil, 0)
	if n <= 0 {
		return nil, fmt.Errorf("needle_embed size: %s (%d)", l.lastError(), n)
	}

	out := make([]float32, n)

	got := l.embed(in, nil, 0, &out[0], n)
	runtime.KeepAlive(in)

	if got != n {
		return nil, fmt.Errorf("needle_embed: %s (returned %d, want %d)", l.lastError(), got, n)
	}

	return out, nil
}

// cString is s as a NUL-terminated C string, or nil when s is empty.
func cString(s string) *byte {
	if s == "" {
		return nil
	}

	return &append([]byte(s), 0)[0]
}

// cOut is the NUL-terminated string in buf.
func cOut(buf []byte) []byte {
	s, _, _ := bytes.Cut(buf, []byte{0})

	return s
}
