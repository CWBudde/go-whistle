package whistle

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/search"
)

var (
	// ErrTooLong reports a clip over MaxSamples. Its text is the engine's
	// own error, "audio limit is 30 s".
	ErrTooLong = errors.New("audio limit is 30 s")
	// ErrNoAudio reports a WAV file without samples, as the engine's CLI.
	ErrNoAudio = errors.New("holds no audio")
	// ErrNotSupported reports a TranscribeOptions field this version of
	// the port does not implement yet; it is never silently ignored.
	ErrNotSupported = errors.New("not yet supported")
)

// TranscribeOptions are the per-call settings of Transcribe, the options of
// needle_transcribe and the engine's --audio-* flags. The zero value detects
// the language and decodes at full depth without keywords or timestamps.
type TranscribeOptions struct {
	// Language forces the language ("en", "de", "fr", "es", "it", "nl",
	// "pl"); empty detects it.
	Language string
	// Keywords are words or phrases to favour, the engine's keyword
	// biasing (docs/FINDINGS.md F-029). Each entry is trimmed and its inner
	// white space collapsed; empty entries are dropped. An entry holding
	// newlines is split like the engine's --audio-keywords file.
	Keywords []string
	// WordTimestamps adds Result.Words. Not yet supported (PLAN.md
	// Phase 7).
	WordTimestamps bool
	// Depth is the number of decoder layers to run, 2 up to the full 8; 0
	// is full. It is the engine's endpoint-preserving ladder
	// (--audio-depth, docs/FINDINGS.md F-030): fewer layers decode faster
	// and less accurately.
	Depth int
}

// Transcribe turns 16 kHz mono samples in [-1, 1], at most MaxSamples, into
// a transcript by the engine's beam search (docs/FINDINGS.md F-026). A clip
// the engine treats as silence (F-031: digital silence, steady noise, too
// little level contrast in the speech band, fewer than 160 samples) gives
// the empty result without decoding: Result{}, the engine's JSON without
// word timestamps.
func (m *Model) Transcribe(pcm []float32, opts TranscribeOptions) (Result, error) {
	return m.transcribe(pcm, opts, time.Now)
}

// TranscribeWAV is Transcribe on a RIFF WAV file: 8/16/24/32-bit PCM or
// float, any channels and rate, mixed to mono and resampled to 16 kHz.
func (m *Model) TranscribeWAV(path string, opts TranscribeOptions) (Result, error) {
	pcm, err := frontend.ReadWAV(path)
	if err != nil {
		return Result{}, wrap(err)
	}

	if len(pcm) == 0 {
		return Result{}, fmt.Errorf("whistle: %s %w", path, ErrNoAudio)
	}

	return m.Transcribe(pcm, opts)
}

func (m *Model) transcribe(pcm []float32, opts TranscribeOptions, now func() time.Time) (Result, error) {
	err := m.check(opts)
	if err != nil {
		return Result{}, err
	}

	req, err := m.request(opts)
	if err != nil {
		return Result{}, err
	}

	err = checkLength(len(pcm))
	if err != nil {
		return Result{}, err
	}

	// The length check comes first, then the gate (F-025).
	clk := newClock(now)

	silent, err := m.gate.Silent(pcm)
	if err != nil {
		return Result{}, wrap(err)
	}

	if silent || m.enc.Frames(len(pcm)) == 0 {
		return silentResult(opts), nil
	}

	enc, frames, err := m.enc.Embed(pcm)
	if err != nil {
		return Result{}, wrap(err)
	}

	// One Memory serves every ladder depth (F-030).
	mem, err := m.dec.Memory(enc, frames)
	if err != nil {
		return Result{}, wrap(err)
	}

	out, err := m.strategy.decode(m.decoder(opts.Depth), mem, req, clk)
	if err != nil {
		return Result{}, wrap(err)
	}

	text, err := m.dec.Decode(out.tokens)
	if err != nil {
		return Result{}, wrap(err)
	}

	return Result{Text: text, Language: out.lang, TTFT: clk.ttft(), DecodeTPS: clk.tps()}, nil
}

// silentResult is the engine's result for a silent clip: Silence() with word
// timestamps, Result{} without (F-027).
func silentResult(opts TranscribeOptions) Result {
	if opts.WordTimestamps {
		return Silence()
	}

	return Result{}
}

// check rejects options the port cannot honour.
func (m *Model) check(opts TranscribeOptions) error {
	if opts.Language != "" && !slices.Contains(m.Languages(), opts.Language) {
		return fmt.Errorf("whistle: unknown language %q, want one of %s", opts.Language, strings.Join(m.Languages(), ", "))
	}

	if opts.WordTimestamps {
		return fmt.Errorf("whistle: word timestamps: %w", ErrNotSupported)
	}

	if opts.Depth != 0 && (opts.Depth < 2 || opts.Depth > m.layers) {
		return fmt.Errorf("whistle: depth %d outside 2..%d", opts.Depth, m.layers)
	}

	return nil
}

// request builds the decoding settings of one call: the forced language
// and the keyword automaton (docs/FINDINGS.md F-029), nil without keywords.
func (m *Model) request(opts TranscribeOptions) (request, error) {
	req := request{lang: opts.Language}

	keywords := search.ParseKeywords(strings.Join(opts.Keywords, "\n"))
	if len(keywords) == 0 {
		return req, nil
	}

	kw, err := search.NewKeywords(m.dec.Tokenizer(), keywords, search.DefaultBias())
	if err != nil {
		return request{}, fmt.Errorf("whistle: keywords: %w", err)
	}

	req.adjust = kw.Adjust

	return req, nil
}

// decoder is the decoder at ladder depth n, 0 for the full depth; check has
// validated n.
func (m *Model) decoder(n int) *decoder.Decoder {
	if n == 0 {
		return m.dec
	}

	return m.ladder[n]
}

// checkLength enforces the 30 s limit of one pass.
func checkLength(n int) error {
	if n > MaxSamples {
		return fmt.Errorf("whistle: %w: %d samples, at most %d", ErrTooLong, n, MaxSamples)
	}

	return nil
}
