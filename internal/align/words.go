// Package align turns a decoded transcript into timed words (PLAN.md
// Phase 7): it groups the text tokens into the engine's words, scores each
// word's probability from the token probabilities, and times the words by
// a monotonic DTW over the decoder's cross-attention on the 80 ms encoder
// frames. The engine's algorithm is closed; every choice is fitted against
// its recorded words, and the probe harness (probe_test.go, build tag
// oracle) holds the evidence (docs/FINDINGS.md F-033).
package align

import (
	"fmt"
	"strings"

	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// Word is one word of a transcript and its text tokens.
type Word struct {
	Text       string // the word as the engine prints it, punctuation attached ("mind.")
	Start, End int    // the word's text tokens [Start, End)
	// Core is the tokens [Core[0], Core[1]) that carry the word's time
	// and probability: Start … End without its leading and trailing
	// punctuation in Whisper's sense (split_tokens_on_spaces, then
	// merge_punctuations keeps the time and probability of the word a mark
	// is merged into; docs/FINDINGS.md F-033). A word of punctuation only
	// keeps all its tokens.
	Core [2]int
}

// Words groups text tokens into words (docs/FINDINGS.md F-033): a token
// whose piece starts with "▁" begins a word and every other token, byte
// pieces included, continues it, so the engine's words are the transcript
// split at its spaces, punctuation attached. Tokens that decode to nothing
// (control pieces) join the current word.
func Words(tok *tokenizer.Tokenizer, tokens []int) ([]Word, error) {
	var (
		out   []Word
		start int
	)

	flush := func(end int) error {
		if end == start {
			return nil
		}

		text, err := tok.Decode(tokens[start:end])
		if err != nil {
			return fmt.Errorf("align: %w", err)
		}

		w := Word{Text: strings.TrimPrefix(text, " "), Start: start, End: end}
		w.Core = core(tok, tokens, start, end)
		out = append(out, w)
		start = end

		return nil
	}

	for i, id := range tokens {
		if id < 0 || id >= tok.Len() {
			return nil, fmt.Errorf("align: token %d out of range [0, %d)", id, tok.Len())
		}

		if i > start && tok.Type(id) == tokenizer.TypeNormal && strings.HasPrefix(tok.Piece(id), "▁") {
			err := flush(i)
			if err != nil {
				return nil, err
			}
		}
	}

	err := flush(len(tokens))
	if err != nil {
		return nil, err
	}

	return out, nil
}

// stringPunctuation is Python's string.punctuation.
const stringPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

// core is the tokens of [start, end) from the first to the last of its
// Whisper words that are not punctuation. Whisper's split_tokens_on_spaces
// starts a word at a token with a leading space or a punctuation token,
// and a token is punctuation when its text without white space is a
// substring of string.punctuation, as Python's `in` tests it: "," and "!\""
// are, ".\"" and "?\"" are not, so "am.\"" keeps its closing piece and
// "mouse!\"" does not. A word of punctuation only keeps all its tokens.
func core(tok *tokenizer.Tokenizer, tokens []int, start, end int) [2]int {
	a, b := -1, -1

	for i := start; i < end; {
		j := i + 1
		for j < end && !leadSpace(tok, tokens[j]) && !punctuation(tok, tokens[j]) {
			j++
		}

		text, err := tok.Decode(tokens[i:j])
		if err == nil && !strings.Contains(stringPunctuation, strings.TrimSpace(text)) {
			if a < 0 {
				a = i
			}

			b = j
		}

		i = j
	}

	if a < 0 {
		return [2]int{start, end}
	}

	return [2]int{a, b}
}

// leadSpace reports whether a token's piece starts with "▁".
func leadSpace(tok *tokenizer.Tokenizer, id int) bool {
	return tok.Type(id) == tokenizer.TypeNormal && strings.HasPrefix(tok.Piece(id), "▁")
}

// punctuation reports whether a token is punctuation in Whisper's sense:
// a NORMAL piece whose text without white space is a substring of
// string.punctuation (so a lone "▁" is too). A byte piece spells part of a
// non-ASCII letter or symbol, never ASCII punctuation, and is not.
func punctuation(tok *tokenizer.Tokenizer, id int) bool {
	if tok.Type(id) != tokenizer.TypeNormal {
		return false
	}

	return strings.Contains(stringPunctuation, strings.TrimSpace(strings.ReplaceAll(tok.Piece(id), "▁", " ")))
}
