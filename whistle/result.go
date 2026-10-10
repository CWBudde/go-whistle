package whistle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"
	"unicode/utf8"
)

// Result is one transcript, the needle_transcribe JSON object.
//
// Its JSON form is the engine's byte for byte (docs/FINDINGS.md F-027): the
// fields text, language, words, ttft_ms and decode_tps in that order, times
// in seconds with two decimals, probabilities with three, ttft_ms and
// decode_tps with one. words is written only when Words is non-nil, as the
// engine writes it only with word timestamps on: Result{} is a silent clip
// without timestamps, Silence() one with them.
//
// MarshalJSON writes that form. json.Marshal re-escapes '<', '>' and '&' in
// it as <, > and &; for the engine's bytes call MarshalJSON
// directly or use a json.Encoder with SetEscapeHTML(false).
type Result struct {
	Text     string // the transcript
	Language string // the language code ("en", "de", …), detected or forced; "" for silence
	Words    []Word // the timed words; nil without word timestamps

	// TTFT is the time from the PCM in hand to the first text token: the
	// front end, the encoder and the decoder up to that token.
	TTFT time.Duration
	// DecodeTPS is the decode rate after the first text token, in tokens
	// per second.
	DecodeTPS float64
}

// Word is one timed word of a transcript.
type Word struct {
	Text        string        // the word, with punctuation attached ("mind.")
	Start, End  time.Duration // from the start of the clip, on the 80 ms frame grid
	Probability float64       // in [0, 1]
}

// Silence is the result the engine returns for a silent clip with word
// timestamps on, the exact JSON
// {"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}.
// Without timestamps the engine omits words, which is Result{}.
func Silence() Result {
	return Result{Words: []Word{}}
}

// MarshalJSON writes r in the engine's form.
func (r Result) MarshalJSON() ([]byte, error) {
	if math.IsNaN(r.DecodeTPS) || math.IsInf(r.DecodeTPS, 0) {
		return nil, fmt.Errorf("whistle: decode_tps %v is not a JSON number", r.DecodeTPS)
	}

	err := validWords(r.Words)
	if err != nil {
		return nil, err
	}

	b := make([]byte, 0, 96+len(r.Text)+64*len(r.Words))

	b = append(b, `{"text":`...)
	b = appendString(b, r.Text)
	b = append(b, `,"language":`...)
	b = appendString(b, r.Language)

	if r.Words != nil {
		b = append(b, `,"words":`...)
		b = appendWords(b, r.Words)
	}

	b = append(b, `,"ttft_ms":`...)
	b = strconv.AppendFloat(b, float64(r.TTFT)/float64(time.Millisecond), 'f', 1, 64)
	b = append(b, `,"decode_tps":`...)
	b = strconv.AppendFloat(b, r.DecodeTPS, 'f', 1, 64)

	return append(b, '}'), nil
}

// validWords rejects a word probability that has no JSON number.
func validWords(words []Word) error {
	for _, w := range words {
		if math.IsNaN(w.Probability) || math.IsInf(w.Probability, 0) {
			return fmt.Errorf("whistle: probability %v of %q is not a JSON number", w.Probability, w.Text)
		}
	}

	return nil
}

// appendWords appends words as the engine's JSON array: word, start, end and
// probability per word, times in seconds with two decimals, probabilities
// with three.
func appendWords(b []byte, words []Word) []byte {
	b = append(b, '[')

	for i, w := range words {
		if i > 0 {
			b = append(b, ',')
		}

		b = append(b, `{"word":`...)
		b = appendString(b, w.Text)
		b = append(b, `,"start":`...)
		b = strconv.AppendFloat(b, w.Start.Seconds(), 'f', 2, 64)
		b = append(b, `,"end":`...)
		b = strconv.AppendFloat(b, w.End.Seconds(), 'f', 2, 64)
		b = append(b, `,"probability":`...)
		b = strconv.AppendFloat(b, w.Probability, 'f', 3, 64)
		b = append(b, '}')
	}

	return append(b, ']')
}

// resultJSON and wordJSON are the wire form UnmarshalJSON reads.
type (
	resultJSON struct {
		Text      string     `json:"text"`
		Language  string     `json:"language"`
		Words     []wordJSON `json:"words"`
		TTFTMs    float64    `json:"ttft_ms"`
		DecodeTPS float64    `json:"decode_tps"`
	}
	wordJSON struct {
		Word        string  `json:"word"`
		Start       float64 `json:"start"`
		End         float64 `json:"end"`
		Probability float64 `json:"probability"`
	}
)

// UnmarshalJSON reads the engine's form. A missing or null words field
// leaves Words nil and "words":[] makes it empty, so the engine's output
// marshals back to the same bytes.
func (r *Result) UnmarshalJSON(data []byte) error {
	var v resultJSON

	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()

	err := dec.Decode(&v)
	if err != nil {
		return fmt.Errorf("whistle: result JSON: %w", err)
	}

	*r = Result{
		Text:      v.Text,
		Language:  v.Language,
		TTFT:      durationOf(v.TTFTMs, time.Millisecond),
		DecodeTPS: v.DecodeTPS,
	}

	if v.Words != nil {
		r.Words = make([]Word, len(v.Words))
		for i, w := range v.Words {
			r.Words[i] = Word{
				Text:        w.Word,
				Start:       durationOf(w.Start, time.Second),
				End:         durationOf(w.End, time.Second),
				Probability: w.Probability,
			}
		}
	}

	return nil
}

// durationOf converts v units to a Duration, rounded to the nanosecond.
func durationOf(v float64, unit time.Duration) time.Duration {
	return time.Duration(math.Round(v * float64(unit)))
}

// appendString appends s as a JSON string the way the engine writes one
// (docs/FINDINGS.md F-027): '"' and '\' escaped, control characters as \n,
// \r, \t, \b, \f or \u00XX, everything else, non-ASCII included, as raw
// UTF-8. Invalid UTF-8 becomes U+FFFD, as in encoding/json.
func appendString(b []byte, s string) []byte {
	const hex = "0123456789abcdef"

	b = append(b, '"')

	for i := 0; i < len(s); {
		c := s[i]
		if c >= utf8.RuneSelf {
			r, n := utf8.DecodeRuneInString(s[i:])
			if r == utf8.RuneError && n == 1 {
				b = utf8.AppendRune(b, utf8.RuneError)
			} else {
				b = append(b, s[i:i+n]...)
			}

			i += n

			continue
		}

		switch {
		case c == '"' || c == '\\':
			b = append(b, '\\', c)
		case c == '\n':
			b = append(b, '\\', 'n')
		case c == '\r':
			b = append(b, '\\', 'r')
		case c == '\t':
			b = append(b, '\\', 't')
		case c == '\b':
			b = append(b, '\\', 'b')
		case c == '\f':
			b = append(b, '\\', 'f')
		case c < 0x20:
			b = append(b, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xf])
		default:
			b = append(b, c)
		}

		i++
	}

	return append(b, '"')
}
