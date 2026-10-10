package search

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestParseKeywords(t *testing.T) {
	got := ParseKeywords("  Cowley's \n\n verses   of\tCowley's\r\nPaare\n")
	want := []string{"Cowley's", "verses of Cowley's", "Paare"}

	if !slices.Equal(got, want) {
		t.Errorf("ParseKeywords = %q, want %q", got, want)
	}

	if ParseKeywords(" \n\n") != nil {
		t.Error("blank lines give keywords")
	}
}

func TestSpellings(t *testing.T) {
	got := spellings("écrit", FormSpace|FormBare|FormCapital|FormLower)
	want := []string{" écrit", "écrit", " Écrit", "Écrit", " écrit", "écrit"}

	if !slices.Equal(got, want) {
		t.Errorf("spellings = %q, want %q", got, want)
	}

	if got := spellings("Paare", FormSpace); !slices.Equal(got, []string{" Paare"}) {
		t.Errorf("space only: %q", got)
	}
}

// unit gives every token a bonus of 1.
func unit(int) float64 { return 1 }

// TestKeywordBonus pins the bonus rules on token ids: an advancing token
// earns its weight, a completing one the completion bonus on top, a
// failing one gives back Revoke of the partial match's bonus beyond the
// suffix it falls back to, a completed keyword keeps its bonus, and EOS
// breaks a partial match.
func TestKeywordBonus(t *testing.T) {
	const eos = 99

	b := Bias{Weight: 1, Revoke: 1, Complete: 0.5, Forms: FormSpace}
	k := build([][]int{{1, 2, 3}, {2, 4}}, unit, b)

	for _, tc := range []struct {
		text []int
		eos  bool
		want float64
	}{
		{[]int{1}, false, 1},
		{[]int{1, 2}, false, 2},
		{[]int{1, 2, 3}, false, 3.5},     // completed
		{[]int{1, 2, 3}, true, 3.5},      // kept after EOS
		{[]int{1, 2, 3, 7}, false, 3.5},  // kept after a failure
		{[]int{1, 2}, true, 0},           // revoked by EOS
		{[]int{1, 2, 7}, false, 0},       // revoked
		{[]int{1, 2, 4}, false, 1 + 1.5}, // falls back to "2", which then completes "2 4"
		{[]int{7, 1, 1, 2, 3}, false, 3.5},
	} {
		e := -1
		if tc.eos {
			e = eos
		}

		if got := k.Bonus(tc.text, e); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("Bonus(%v, eos %v) = %g, want %g", tc.text, tc.eos, got, tc.want)
		}
	}

	b.Revoke = 0.5
	half := build([][]int{{1, 2, 3}}, unit, b)

	if got := half.Bonus([]int{1, 2, 7}, -1); got != 1 {
		t.Errorf("half revocation: %g, want 1", got)
	}
}

// TestKeywordAdjust: Adjust adds to every token exactly the change of
// Bonus that token makes, for random texts over a small vocabulary and an
// automaton with shared prefixes, suffix overlaps and nested keywords.
func TestKeywordAdjust(t *testing.T) {
	const vocab = 8

	rng := rand.New(rand.NewPCG(1, 2))
	weight := func(id int) float64 { return 0.5 + 0.25*float64(id) }

	for _, b := range []Bias{
		{Weight: 1, Revoke: 1, Forms: FormSpace},
		{Weight: 1, Revoke: 0.3, Complete: 2, Forms: FormSpace},
		{Weight: 1, Revoke: 0, Complete: 1, Forms: FormSpace},
	} {
		k := build([][]int{{1, 2, 3}, {2, 3}, {1, 2}, {3, 1, 4}, {5}, {1, 2, 3, 6}}, weight, b)

		for range 300 {
			text := make([]int, rng.IntN(9))
			for i := range text {
				text[i] = rng.IntN(vocab)
			}

			row := make([]float64, vocab)
			k.Adjust(text, row)

			before := k.Bonus(text, -1)

			for id := range vocab {
				want := k.Bonus(append(slices.Clone(text), id), -1) - before
				if math.Abs(row[id]-want) > 1e-9 {
					t.Fatalf("%s: text %v, token %d: Adjust adds %g, Bonus changes by %g", b, text, id, row[id], want)
				}
			}
		}
	}
}

func TestNewKeywordsRejects(t *testing.T) {
	for _, b := range []Bias{
		{Weight: math.NaN(), Revoke: 1, Forms: FormSpace},
		{Weight: 1, Revoke: 1.5, Forms: FormSpace},
		{Weight: 1, Revoke: math.NaN(), Forms: FormSpace},
		{Weight: 1, Complete: math.Inf(1), Forms: FormSpace},
		{Weight: 1, Revoke: 1, Forms: FormCapital},
	} {
		_, err := NewKeywords(nil, nil, b)
		if !errors.Is(err, ErrKeywords) {
			t.Errorf("%s: %v, want ErrKeywords", b, err)
		}
	}
}
