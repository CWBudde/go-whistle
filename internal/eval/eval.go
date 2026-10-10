// Package eval holds the error rates the parity tests and the evaluation
// harnesses score transcripts with: the edit distance, the word error rate
// over normalised words and the character error rate over the raw text.
package eval

import (
	"strings"
	"unicode"
)

// EditDistance is the Levenshtein distance between a and b: the fewest
// insertions, deletions and substitutions that turn a into b.
func EditDistance[T comparable](a, b []T) int {
	prev, cur := make([]int, len(b)+1), make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}

	for i := range a {
		cur[0] = i + 1

		for j := range b {
			cost := 1
			if a[i] == b[j] {
				cost = 0
			}

			cur[j+1] = min(prev[j+1]+1, cur[j]+1, prev[j]+cost)
		}

		prev, cur = cur, prev
	}

	return prev[len(b)]
}

// Words is the WER normalisation of text: lower case, split at every rune
// that is not a letter, a digit or an apostrophe, so punctuation drops out.
func Words(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
}

// WER is the word error rate of hyp against ref over Words, case and
// punctuation ignored. An empty ref gives the number of hypothesis words.
func WER(ref, hyp string) float64 {
	r := Words(ref)
	if len(r) == 0 {
		return float64(len(Words(hyp)))
	}

	return float64(EditDistance(r, Words(hyp))) / float64(len(r))
}

// CER is the character error rate of hyp against ref over the raw runes,
// case and punctuation included. An empty ref gives the number of
// hypothesis runes.
func CER(ref, hyp string) float64 {
	r := []rune(ref)
	if len(r) == 0 {
		return float64(len([]rune(hyp)))
	}

	return float64(EditDistance(r, []rune(hyp))) / float64(len(r))
}
