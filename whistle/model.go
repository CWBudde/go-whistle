package whistle

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/encoder"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/search"
	"github.com/cwbudde/go-whistle/internal/silence"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
	"github.com/cwbudde/go-whistle/internal/weights"
)

const (
	// SampleRate is the only rate Transcribe and Embed take: 16 kHz mono.
	SampleRate = frontend.SampleRate
	// MaxSamples is the longest clip one pass takes, 30 s at SampleRate.
	MaxSamples = 30 * SampleRate
	// FrameDuration is the hop of one encoder frame, one Embed row.
	FrameDuration = 80 * time.Millisecond
)

// Model is a loaded Whistle model: the encoder, the decoder and the
// vocabulary on one weight file. It is safe for concurrent use; Close it
// when done.
type Model struct {
	src      weights.Source
	enc      *encoder.Encoder
	dec      *decoder.Decoder
	ladder   []*decoder.Decoder // ladder[n] runs depth n (docs/FINDINGS.md F-030); nil below 2
	gate     *silence.Gate
	strategy strategy
	dim      int // encoder width, Embedding.Dim
	layers   int // decoder blocks, the full Depth
}

// Option configures Load.
type Option func(*loadOptions)

type loadOptions struct {
	vocabulary string
	noMmap     bool
}

// WithVocabulary names the .cact file whose tokenizer the model uses. Only
// the .cact format holds one, so a safetensors checkpoint needs it; the
// default is whistle.cact next to the checkpoint.
func WithVocabulary(cactPath string) Option {
	return func(o *loadOptions) { o.vocabulary = cactPath }
}

// WithoutMmap reads the weight file into memory instead of mapping it.
func WithoutMmap() Option {
	return func(o *loadOptions) { o.noMmap = true }
}

// vocabularyFile is the .cact Load looks for next to a safetensors file.
const vocabularyFile = "whistle.cact"

// Load opens a Whistle weight file, the deployed whistle.cact or the float32
// whistle.safetensors checkpoint, and builds the model on it with the
// encoder and decoder the oracle settled (docs/FINDINGS.md F-014 to F-024),
// the engine's beam search (F-026) and its silence gate (F-031).
// The .cact weights are the ones the engine runs; they match it best.
func Load(path string, opts ...Option) (*Model, error) {
	var o loadOptions
	for _, opt := range opts {
		opt(&o)
	}

	src, err := weights.Open(path, weights.Options{NoMmap: o.noMmap})
	if err != nil {
		return nil, wrap(err)
	}

	m, err := build(src, path, o)
	if err != nil {
		_ = src.Close()

		return nil, err
	}

	return m, nil
}

// build loads the vocabulary, the encoder and the decoder from src and sets
// up the silence gate and the beam search.
func build(src weights.Source, path string, o loadOptions) (*Model, error) {
	tok, err := vocabulary(src, path, o)
	if err != nil {
		return nil, err
	}

	enc, err := encoder.Load(src, encoder.Best())
	if err != nil {
		return nil, wrap(err)
	}

	dec, err := decoder.Load(src, tok, decoder.Best())
	if err != nil {
		return nil, wrap(err)
	}

	gate, err := silence.New()
	if err != nil {
		return nil, wrap(err)
	}

	cfg := src.Config()

	// Every ladder rung shares the weights; building them up front keeps
	// Transcribe free of locks.
	ladder := make([]*decoder.Decoder, cfg.DecLayers+1)
	ladder[cfg.DecLayers] = dec

	for n := 2; n < cfg.DecLayers; n++ {
		ladder[n], err = dec.WithDepth(n)
		if err != nil {
			return nil, wrap(err)
		}
	}

	return &Model{
		src: src, enc: enc, dec: dec, ladder: ladder, gate: gate, strategy: beam{opt: search.Default()},
		dim: cfg.Dim, layers: cfg.DecLayers,
	}, nil
}

// vocabulary reads the tokenizer from src, or from the WithVocabulary file
// or the .cact next to path when src has none.
func vocabulary(src weights.Source, path string, o loadOptions) (*tokenizer.Tokenizer, error) {
	if o.vocabulary == "" {
		tok, err := decoder.LoadTokenizer(src)
		if !errors.Is(err, weights.ErrUnavailable) {
			return tok, wrap(err)
		}

		o.vocabulary = filepath.Join(filepath.Dir(path), vocabularyFile)

		_, err = os.Stat(o.vocabulary)
		if err != nil {
			return nil, fmt.Errorf("whistle: %s holds no tokenizer and %s is missing; pass WithVocabulary", path, o.vocabulary)
		}
	}

	v, err := weights.OpenCact(o.vocabulary, weights.Options{})
	if err != nil {
		return nil, fmt.Errorf("whistle: vocabulary: %w", err)
	}

	defer func() { _ = v.Close() }()

	tok, err := decoder.LoadTokenizer(v)
	if err != nil {
		return nil, fmt.Errorf("whistle: vocabulary %s: %w", o.vocabulary, err)
	}

	return tok, nil
}

// Close releases the weight file. The model must not be used afterwards.
func (m *Model) Close() error {
	return wrap(m.src.Close())
}

// Languages lists the language codes Transcribe detects and accepts, in
// token order: en, de, fr, es, it, nl, pl.
func (m *Model) Languages() []string {
	langs := m.dec.Languages()

	out := make([]string, len(langs))
	for i, l := range langs {
		out[i] = l.Code
	}

	return out
}

// Embedding is the encoder output of one clip in needle_embed's layout:
// Frames rows of Dim values, one row per 80 ms frame, row-major in Values.
type Embedding struct {
	Values []float32
	Frames int
	Dim    int
}

// Frame returns row i, a slice of Values.
func (e Embedding) Frame(i int) []float32 {
	return e.Values[i*e.Dim : (i+1)*e.Dim]
}

// Embed runs the front end and the encoder on 16 kHz mono samples in
// [-1, 1], at most MaxSamples, and returns what needle_embed returns:
// enc.final_norm of the lane mean, 512 values per 80 ms frame
// (docs/FINDINGS.md F-015), ceil(floor(n / 160) / 8) frames for n samples
// (F-012). Fewer than 160 samples give no frame.
func (m *Model) Embed(pcm []float32) (Embedding, error) {
	err := checkLength(len(pcm))
	if err != nil {
		return Embedding{}, err
	}

	if m.enc.Frames(len(pcm)) == 0 {
		return Embedding{Dim: m.dim}, nil
	}

	values, frames, err := m.enc.Embed(pcm)
	if err != nil {
		return Embedding{}, wrap(err)
	}

	return Embedding{Values: values, Frames: frames, Dim: m.dim}, nil
}

// wrap prefixes a lower-level error with the package name.
func wrap(err error) error {
	if err == nil {
		return nil
	}

	return fmt.Errorf("whistle: %w", err)
}
