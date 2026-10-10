package whistle

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/cwbudde/go-whistle/internal/align"
	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/search"
	"github.com/cwbudde/go-whistle/internal/silence"
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
	// WordTimestamps adds Result.Words: every word with its start, end and
	// probability, as the engine's --audio-word-timestamps
	// (docs/FINDINGS.md F-033). It costs one more decoder pass over the
	// transcript.
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
// word timestamps, or Silence() with them.
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

	// The gate's levels also place the first word (F-033).
	levels, err := m.gate.Levels(pcm)
	if err != nil {
		return Result{}, wrap(err)
	}

	if silence.Spread(levels) <= silence.ThresholdDB || m.enc.Frames(len(pcm)) == 0 {
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

	d := m.decoder(opts.Depth)

	out, err := m.strategy.decode(d, mem, req, clk)
	if err != nil {
		return Result{}, wrap(err)
	}

	text, err := m.dec.Decode(out.tokens)
	if err != nil {
		return Result{}, wrap(err)
	}

	res := Result{Text: text, Language: out.lang, TTFT: clk.ttft(), DecodeTPS: clk.tps()}

	if opts.WordTimestamps {
		res.Words, err = timeWords(d, mem, out, levels, len(pcm))
		if err != nil {
			return Result{}, wrap(err)
		}
	}

	return res, nil
}

// timeWords times the transcript out (docs/FINDINGS.md F-033): one
// teacher-forced pass of d over the prompt, the language token, the text
// tokens and EOS with the cross-attention captured, then Whisper's
// alignment on it (align.Default). The probabilities are the pass' own,
// without keyword bias. Times are on the 80 ms frame grid, at most the last
// grid point within the clip's samples.
func timeWords(d *decoder.Decoder, mem *decoder.Memory, out decoded, levels []float64, samples int) ([]Word, error) {
	lang := -1

	for _, l := range d.Languages() {
		if l.Code == out.lang {
			lang = l.ID
		}
	}

	if lang < 0 {
		return nil, fmt.Errorf("word timestamps: language %q", out.lang)
	}

	prompt := d.Prompt()
	seq := append(append(prompt, lang), out.tokens...)
	seq = append(seq, d.Tokenizer().EOSID())

	var attn []decoder.Attention

	s := d.NewState(mem)
	s.SetCapture(func(a decoder.Attention) { attn = append(attn, a) })

	logits, err := d.Extend(s, seq)
	if err != nil {
		return nil, fmt.Errorf("word timestamps: %w", err)
	}

	first, v := len(prompt)+1, d.Vocab()
	logp := make([]float64, len(out.tokens))

	for k, tok := range out.tokens {
		logp[k] = logSoftmaxAt(logits[(first+k-1)*v:(first+k)*v], tok)
	}

	in := align.Input{Attn: attn, First: first, Tokens: len(out.tokens), Frames: mem.Frames(), Levels: levels}

	timed, err := align.Align(d.Tokenizer(), out.tokens, logp, in, align.Default())
	if err != nil {
		return nil, fmt.Errorf("word timestamps: %w", err)
	}

	last := samples / int(SampleRate*FrameDuration/time.Second) // the last grid point within the clip
	words := make([]Word, len(timed))

	for i, w := range timed {
		words[i] = Word{
			Text:        w.Text,
			Start:       time.Duration(min(w.Start, last)) * FrameDuration,
			End:         time.Duration(min(w.End, last)) * FrameDuration,
			Probability: w.Probability,
		}
	}

	return words, nil
}

// logSoftmaxAt is log softmax(v)[i] in float64.
func logSoftmaxAt(v []float32, i int) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = math.Max(m, float64(x))
	}

	var s float64
	for _, x := range v {
		s += math.Exp(float64(x) - m)
	}

	return float64(v[i]) - m - math.Log(s)
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
