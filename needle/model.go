package needle

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync/atomic"

	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/text"
	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// The files Load reads from its directory, as `just fetch-needle3` lays
// them out.
const (
	cactFile        = "needle3.cact"
	safetensorsFile = "needle3.safetensors"
	tokenizerFile   = "tokenizer.model"
)

var (
	// ErrContextOverflow is the engine's refusal of a turn that does not
	// fit the context window, with AgentOptions.FailInputOverflow: the
	// error is an *OverflowError.
	ErrContextOverflow = errors.New("needle: input exceeds the context window")
	// ErrContextBudget is the engine's refusal of tools and system facts
	// whose rendering alone fills the context window.
	ErrContextBudget = errors.New("needle: context budget exceeded")
	// ErrClosed is a call on a closed Model.
	ErrClosed = errors.New("needle: model is closed")
)

// OverflowError is ErrContextOverflow with the engine's two numbers: the
// turn's length in tokens and the room a turn has, the context less the
// static prefix and the answer's token limit.
type OverflowError struct {
	Tokens, Available int
}

func (e *OverflowError) Error() string {
	return fmt.Sprintf("%v (%d tokens, %d available)", ErrContextOverflow, e.Tokens, e.Available)
}

func (e *OverflowError) Unwrap() error { return ErrContextOverflow }

// Model is a loaded Needle 3 text model: the weights, the tokenizer and
// the decoder. Several agents may use one Model at the same time; Close it
// when done, and not while a call runs.
type Model struct {
	src    *text.Source
	tm     *text.Model
	p      *spm.Processor
	d      *toolcall.Decoder
	closed atomic.Bool
}

// Option configures Load.
type Option func(*loadOptions)

type loadOptions struct {
	float, noInt8, lut bool
}

// WithFloatWeights runs the float32 checkpoint needle3.safetensors in
// float32 activations instead of needle3.cact in the engine's numerics.
// Its answers are the float weights', not the engine's (docs/FINDINGS.md
// F-061).
func WithFloatWeights() Option {
	return func(o *loadOptions) { o.float = true }
}

// WithoutInt8 runs needle3.cact on its packed weights in float32
// activations, without the engine's int8 activations, int8 KV cache and KV
// window: the dequantised weights' answers (docs/FINDINGS.md F-059).
func WithoutInt8() Option {
	return func(o *loadOptions) { o.noInt8 = true }
}

// WithLUT multiplies by the packed weights on the lookup-table kernel
// instead of the exact one: faster where its assembly runs, but its tables
// round to 14 bits, which moves the confidence head's logit by up to 3.5e-4
// (docs/FINDINGS.md F-059). WithFloatWeights ignores it.
func WithLUT() Option {
	return func(o *loadOptions) { o.lut = true }
}

// Load opens the Needle 3 text model in dir: needle3.cact and
// tokenizer.model, as `just fetch-needle3` fetches them. By default it
// runs the engine's numerics (docs/FINDINGS.md F-060): the packed CQ
// weights on the exact kernel, int8 activations and KV cache, and the KV
// window of 256 positions with the static prefix kept in view. The packed
// weights alias the file's bytes, which the Model holds until Close.
func Load(dir string, opts ...Option) (*Model, error) {
	var o loadOptions
	for _, opt := range opts {
		opt(&o)
	}

	p, err := spm.Load(filepath.Join(dir, tokenizerFile))
	if err != nil {
		return nil, wrap(err)
	}

	path, oo := filepath.Join(dir, cactFile), text.OpenOptions{Packed: true, LUT: o.lut}
	if o.float {
		path, oo = filepath.Join(dir, safetensorsFile), text.OpenOptions{}
	}

	src, err := text.OpenWith(path, oo)
	if err != nil {
		return nil, wrap(err)
	}

	m, err := build(src, p, o)
	if err != nil {
		_ = src.Close()

		return nil, err
	}

	return m, nil
}

// toolsEnd ends the static prefix, the KV window's sink.
const toolsEnd = "</tools>"

// build loads the text model from src with the numerics o names, and its
// decoder.
func build(src *text.Source, p *spm.Processor, o loadOptions) (*Model, error) {
	cfg := src.TextConfig()

	var lo text.LoadOptions

	if !o.float && !o.noInt8 {
		sink, ok := p.PieceID(toolsEnd)
		if !ok {
			return nil, fmt.Errorf("needle: the tokenizer has no %s", toolsEnd)
		}

		lo = text.LoadOptions{Int8: true, KVWindow: cfg.KVWindow, SinkToken: sink}
	}

	tm, err := text.LoadWith(src, cfg, lo)
	if err != nil {
		return nil, wrap(err)
	}

	if !tm.HasConfidence() {
		return nil, errors.New("needle: the weights carry no confidence head")
	}

	d, err := toolcall.NewDecoder(tm, p)
	if err != nil {
		return nil, wrap(err)
	}

	return &Model{src: src, tm: tm, p: p, d: d}, nil
}

// Close releases the weights. Closing twice is a no-op.
func (m *Model) Close() error {
	if m.closed.Swap(true) {
		return nil
	}

	return wrap(m.src.Close())
}

// context is the model's context length, 8192 positions for needle3.
func (m *Model) context() int { return m.tm.Config().MaxSeqLen }

// wrap prefixes an error of the packages below with the package name.
func wrap(err error) error {
	if err == nil {
		return nil
	}

	return fmt.Errorf("needle: %w", err)
}
