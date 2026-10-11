package search

import (
	"math"
	"slices"
	"testing"
)

// f029Bias is DefaultBias as F-029 picked it (docs/FINDINGS.md): 2 nats
// per token, completion 1, every form, token ids as written. The probes of
// F-042 and F-044 vary it, so that their tables stay reproducible.
func f029Bias() Bias {
	return Bias{Weight: 2, Revoke: 1, Complete: 1, Forms: FormSpace | FormBare | FormCapital | FormLower}
}

// TestCaseSpellings pins the spellings foldCases adds: the keyword
// lower-cased and with only its first letter upper-cased, under the
// forms, without those spellings already has.
func TestCaseSpellings(t *testing.T) {
	for _, tc := range []struct {
		kw    string
		forms Form
		want  []string
	}{
		{"PAARE", FormSpace | FormBare | FormCapital | FormLower, []string{" paare", "paare", " Paare", "Paare"}},
		{"PAARE", FormSpace, []string{" paare", " Paare"}},
		{"Paare", FormSpace | FormBare | FormCapital | FormLower, []string{" PAARE", "PAARE"}},
		{"cOWLEY'S", FormSpace | FormBare, []string{
			" cowley's", "cowley's", " Cowley's", "Cowley's", " COWLEY'S", "COWLEY'S",
		}},
		{"ÉCRIT", FormSpace, []string{" écrit", " Écrit"}},
		// PR #45 review: casings beyond the first letter.
		{"NEW YORK", FormSpace, []string{" new york", " New york", " New York"}},
		{"foia", FormSpace, []string{" Foia", " FOIA"}},
	} {
		if got := caseSpellings(tc.kw, tc.forms); !slices.Equal(got, tc.want) {
			t.Errorf("caseSpellings(%q, %s) = %q, want %q", tc.kw, tc.forms, got, tc.want)
		}
	}
}

// TestKeywordSpent pins Bias.spent on "1 2" with unit weights and a
// completion bonus of 0.5: once the keyword completed, a repeat's tokens
// earn nothing, so there is nothing to take back, and its completion
// earns nothing either.
func TestKeywordSpent(t *testing.T) {
	ids := [][]int{{1, 2}}

	for _, tc := range []struct {
		name  string
		spent bool
		text  []int
		want  float64
	}{
		{"repeated", false, []int{1, 2, 1, 2}, 5},
		{"spent, repeated", true, []int{1, 2, 1, 2}, 2.5},
		{"spent, a repeat in progress", true, []int{1, 2, 1}, 2.5},
		{"spent, a repeat broken", true, []int{1, 2, 1, 7}, 2.5},
		{"spent, a partial match first", true, []int{1, 7, 1, 2}, 2.5},
	} {
		k := build(ids, unit, Bias{Weight: 1, Revoke: 1, Complete: 0.5, Forms: FormSpace, spent: tc.spent})
		k.identify(ids, []int{0})

		if got := k.Bonus(tc.text, -1); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("%s: Bonus(%v) = %g, want %g", tc.name, tc.text, got, tc.want)
		}
	}
}

// TestByteKeywords pins the byte matcher (foldBytes) on the real
// vocabulary: it matches a keyword whatever its case and its tokens, the
// keyword's own tokens earn what the token matcher gives them, shareToken
// pays per token of the text, the word end and the spent rule.
func TestByteKeywords(t *testing.T) {
	d, _ := loadSetup(t)
	tok := d.Tokenizer()

	ids := func(pieces ...string) []int {
		out := make([]int, len(pieces))

		for i, p := range pieces {
			id, ok := tok.PieceID(p)
			if !ok {
				t.Fatalf("piece %q is not in the vocabulary", p)
			}

			out[i] = id
		}

		return out
	}

	diese := ids("D", "ies", "e")
	paare := ids("▁P", "a", "are")
	eos := tok.EOSID()

	for _, tc := range []struct {
		name string
		kws  []string
		b    func(*Bias)
		text []int
		want float64
	}{
		{"the keyword's own tokens", []string{"Paare"}, nil, slices.Concat(diese, paare), 7},
		{"other tokens", []string{"Paare"}, nil, slices.Concat(diese, ids("▁paar", "e")), 7},
		{"capitals, as written in 4 tokens", []string{"PAARE"}, nil, slices.Concat(diese, paare), 9},
		{"a piece inside a word", []string{"aare"}, nil, slices.Concat(diese, paare), 0},
		{
			"per token of the text",
			[]string{"Paare"},
			func(b *Bias) { b.share = shareToken },
			slices.Concat(diese, ids("▁paar", "e")), 5,
		},
		{"a word start", []string{"Paar"}, nil, slices.Concat(diese, paare), 5},
		{"a word start under wordEnd", []string{"Paar"}, func(b *Bias) { b.wordEnd = true }, slices.Concat(diese, paare), 0},
		{"a word end under wordEnd", []string{"Paare"}, func(b *Bias) { b.wordEnd = true }, slices.Concat(diese, paare,
			ids("▁k")), 7},
		{"a repeat", []string{"Paare"}, nil, slices.Concat(diese, paare, paare), 14},
		{"a repeat, spent", []string{"Paare"}, func(b *Bias) { b.spent = true }, slices.Concat(diese, paare, paare), 7},
	} {
		b := f029Bias()
		b.fold = foldBytes

		if tc.b != nil {
			tc.b(&b)
		}

		k, err := NewKeywords(tok, tc.kws, b)
		if err != nil {
			t.Fatal(err)
		}

		if got := k.Bonus(tc.text, eos); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("%s: Bonus = %g, want %g", tc.name, got, tc.want)
		}

		// Adjust adds the same changes token by token.
		sum := 0.0

		for i := range len(tc.text) + 1 {
			want := eos
			if i < len(tc.text) {
				want = tc.text[i]
			}

			logp := make([]float64, d.Vocab())
			k.Adjust(tc.text[:i], logp)
			sum += logp[want]
		}

		if math.Abs(sum-tc.want) > 1e-9 {
			t.Errorf("%s: Adjust sums to %g, want %g", tc.name, sum, tc.want)
		}
	}
}

// TestCaseKeywords pins foldCases on the real vocabulary: a keyword in
// capitals earns the bonus of its first-letter capital spelling.
func TestCaseKeywords(t *testing.T) {
	d, _ := loadSetup(t)
	tok := d.Tokenizer()
	text := tok.Encode("Diese Paare können")

	for _, tc := range []struct {
		fold foldMode
		want float64
	}{{foldNone, 0}, {foldCases, 7}} {
		b := f029Bias()
		b.fold = tc.fold

		k, err := NewKeywords(tok, []string{"PAARE"}, b)
		if err != nil {
			t.Fatal(err)
		}

		if got := k.Bonus(text, -1); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("fold %d: Bonus = %g, want %g", tc.fold, got, tc.want)
		}
	}
}
