package align

import (
	"fmt"
	"math"

	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// Probability is a word's probability (docs/FINDINGS.md F-033): the mean
// of its core tokens' probabilities, Whisper's word_probabilities, with
// logp the model's log-probability of every text token (the softmax over
// the whole vocabulary, without keyword bias).
func Probability(logp []float64, w Word) float64 {
	var s float64
	for _, lp := range logp[w.Core[0]:w.Core[1]] {
		s += math.Exp(lp)
	}

	return s / float64(w.Core[1]-w.Core[0])
}

// Timed is one timed word of a transcript.
type Timed struct {
	Text        string
	Start, End  int // frames of 80 ms from the start of the clip
	Probability float64
}

// Align times the words of a transcript: tokens are its text tokens, logp
// their log-probabilities and in the teacher-forced pass over the prompt,
// the language token, tokens and EOS with the cross-attention captured
// (docs/FINDINGS.md F-033).
func Align(tok *tokenizer.Tokenizer, tokens []int, logp []float64, in Input, opt Options) ([]Timed, error) {
	if len(logp) < len(tokens) || in.Tokens != len(tokens) {
		return nil, fmt.Errorf("align: %d tokens, %d log-probabilities, input of %d", len(tokens), len(logp), in.Tokens)
	}

	words, err := Words(tok, tokens)
	if err != nil || len(words) == 0 {
		return []Timed{}, err
	}

	spans, err := Times(in, words, opt)
	if err != nil {
		return nil, err
	}

	out := make([]Timed, len(words))
	for i, w := range words {
		out[i] = Timed{Text: w.Text, Start: spans[i].Start, End: spans[i].End, Probability: Probability(logp, w)}
	}

	return out, nil
}
