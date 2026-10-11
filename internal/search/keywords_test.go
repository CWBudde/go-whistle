package search

import (
	"errors"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/cwbudde/go-whistle/internal/tokenizer"
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
		{Weight: 1, Revoke: 1, Complete: 0.5, Forms: FormSpace, transient: true, echo: true},
		{Weight: 1, Revoke: 1, Complete: 1, Forms: FormSpace, once: true},
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

// TestKeywordOverlaps pins the review cases of the automaton: the bonus
// does not depend on the order the keywords are inserted, a keyword that
// completes as the suffix of a longer keyword's partial match earns its
// completion and keeps its tokens when that match fails, and a bare
// spelling matches only at the start of the text.
func TestKeywordOverlaps(t *testing.T) {
	b := Bias{Weight: 1, Revoke: 1, Complete: 0.5, Forms: FormSpace}

	for _, order := range [][][]int{{{1, 2, 3}, {1, 2}}, {{1, 2}, {1, 2, 3}}} {
		k := build(order, unit, b)
		if got := k.Bonus([]int{1, 2, 3}, -1); got != 3+2*0.5 {
			t.Errorf("keywords %v: Bonus(1 2 3) = %g, want 4", order, got)
		}
	}

	k := build([][]int{{1, 2, 3, 4}, {2, 3}}, unit, b)

	for _, tc := range []struct {
		text []int
		want float64
	}{
		{[]int{1, 2, 3}, 3 + 0.5},       // "2 3" completes inside "1 2 3 …"
		{[]int{1, 2, 3, 7}, 2 + 0.5},    // the failure takes back "1" only
		{[]int{1, 2, 3, 4}, 4 + 2*0.5},  // both complete
		{[]int{1, 2, 3, 4, 7}, 4 + 1.0}, // and stay
	} {
		if got := k.Bonus(tc.text, -1); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("Bonus(%v) = %g, want %g", tc.text, got, tc.want)
		}
	}

	bare := build([][]int{{textStart, 5, 6}, {7}}, unit, b)

	for _, tc := range []struct {
		text []int
		want float64
	}{
		{[]int{5, 6}, 2 + 0.5},
		{[]int{7, 5, 6}, 1 + 0.5}, // "5 6" inside the text is no bare match
		{[]int{5, 7}, 1 + 0.5},    // the failed bare start is taken back
	} {
		if got := bare.Bonus(tc.text, -1); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("bare: Bonus(%v) = %g, want %g", tc.text, got, tc.want)
		}

		row := make([]float64, 8)
		bare.Adjust(tc.text[:len(tc.text)-1], row)

		want := tc.want - bare.Bonus(tc.text[:len(tc.text)-1], -1)
		if last := tc.text[len(tc.text)-1]; math.Abs(row[last]-want) > 1e-12 {
			t.Errorf("bare: Adjust(%v)[%d] = %g, want %g", tc.text[:len(tc.text)-1], last, row[last], want)
		}
	}
}

// TestKeywordPerCharBytes: under PerChar a keyword character spelled in
// byte pieces earns one character's weight, not one per "<0xNN>" letter.
func TestKeywordPerCharBytes(t *testing.T) {
	d, _ := loadSetup(t)
	tok := d.Tokenizer()

	const kw = "🎵🎶" // two characters in byte pieces

	ids := tok.Encode(kw)
	if !slices.ContainsFunc(ids, func(id int) bool { return tok.Type(id) == tokenizer.TypeByte }) {
		t.Fatalf("%q is no byte-fallback spelling: %v", kw, ids)
	}

	k, err := NewKeywords(tok, []string{kw}, Bias{Weight: 1, PerChar: true, Revoke: 1, Forms: FormBare})
	if err != nil {
		t.Fatal(err)
	}

	if got := k.Bonus(ids, -1); got != 2 {
		t.Errorf("Bonus(%q) = %g, want 2 characters", kw, got)
	}

	// Byte runs that are not valid UTF-8 count as the tokenizer decodes
	// them: one U+FFFD per byte that starts no valid sequence.
	for _, kw := range []string{"\x80", "a\x80b", "\xe6\x97", "\xe6\x97\xa5x", "\xc3\xa9\x80"} {
		ids := tok.Encode(kw)

		text, err := tok.Decode(ids)
		if err != nil {
			t.Fatal(err)
		}

		want := float64(utf8.RuneCountInString(text))

		k, err := NewKeywords(tok, []string{kw}, Bias{Weight: 1, PerChar: true, Revoke: 1, Forms: FormBare})
		if err != nil {
			t.Fatal(err)
		}

		if got := k.Bonus(ids, -1); got != want {
			t.Errorf("Bonus(%q) = %g, want %g characters (%q)", kw, got, want, text)
		}
	}

	// A failed longer match keeps a completed prefix keyword's bonus even
	// when the tail's shares are negative: "\xf0\x9f" (2 characters) inside
	// "🎵x", whose fourth byte turns 3 characters into 1.
	k, err = NewKeywords(tok, []string{"\xf0\x9f", "🎵x"}, Bias{Weight: 1, PerChar: true, Revoke: 1, Forms: FormBare})
	if err != nil {
		t.Fatal(err)
	}

	if got := k.Bonus(tok.Encode("🎵"), tok.EOSID()); got != 2 {
		t.Errorf("Bonus(🎵 + EOS) = %g, want the completed prefix's 2", got)
	}
}

// TestKeywordLong: a spelling longer than 64 tokens works like a short one:
// it completes, a failure takes back only the tokens no completed keyword
// covers, and a keyword completed inside it past token 64 keeps its tokens.
func TestKeywordLong(t *testing.T) {
	const n = 100

	long := make([]int, n)
	for i := range long {
		long[i] = 10 + i
	}

	inner := long[70:75] // also a keyword, completed at position 75
	k := build([][]int{long, inner}, unit, Bias{Weight: 1, Revoke: 1, Complete: 0.5, Forms: FormSpace})

	if got := k.Bonus(long, -1); got != n+2*0.5 {
		t.Errorf("Bonus(long) = %g, want %g", got, n+2*0.5)
	}

	// The first 80 tokens, then a token outside the automaton: all but the
	// five of the inner keyword are taken back.
	if got := k.Bonus(append(slices.Clone(long[:80]), 99), -1); got != 5+0.5 {
		t.Errorf("Bonus(long[:80] + miss) = %g, want 5.5", got)
	}

	row := make([]float64, 200)
	k.Adjust(long[:n-1], row)

	if want := 1 + 0.5; row[long[n-1]] != want {
		t.Errorf("Adjust before the last token: %g, want %g", row[long[n-1]], want)
	}
}

// TestKeywordLongMemory: the automaton of one long spelling stays linear in
// its length (a full path per node would hold n(n+1)/2 weights, 1.6 GB here).
func TestKeywordLongMemory(t *testing.T) {
	const n = 20000

	long := make([]int, n)
	for i := range long {
		long[i] = i
	}

	var before, after runtime.MemStats

	runtime.GC()
	runtime.ReadMemStats(&before)

	k := build([][]int{long}, unit, Bias{Weight: 1, Revoke: 1, Forms: FormSpace})

	runtime.ReadMemStats(&after)

	if grown := after.TotalAlloc - before.TotalAlloc; grown > 64<<20 {
		t.Errorf("building a %d-token spelling allocated %d MB", n, grown>>20)
	}

	if got := k.Bonus(long, -1); got != n {
		t.Errorf("Bonus = %g, want %d", got, n)
	}
}

// TestKeywordLongPrefixLinear: following a long spelling token by token
// costs time linear in its length (no ancestor walk on direct steps).
func TestKeywordLongPrefixLinear(t *testing.T) {
	const n = 20000

	long := make([]int, n)
	for i := range long {
		long[i] = i
	}

	k := build([][]int{long}, unit, Bias{Weight: 1, Revoke: 1, Forms: FormSpace})

	start := time.Now()

	if got := k.Bonus(long[:n-1], -1); got != n-1 {
		t.Errorf("Bonus = %g, want %d", got, n-1)
	}

	if d := time.Since(start); d > 50*time.Millisecond {
		t.Errorf("following %d tokens took %v", n-1, d)
	}
}

// TestKeywordProbeAxes pins the completion variants of the F-042 probe on
// "1 2" with unit weights and a completion bonus of 0.5: a transient
// completion lasts one token, the icefall echo adds the keyword's own
// bonus for that token, and once a repeat earns nothing.
func TestKeywordProbeAxes(t *testing.T) {
	const eos = 99

	for _, tc := range []struct {
		name string
		b    Bias
		text []int
		eos  bool
		want float64
	}{
		{"lasting", Bias{}, []int{1, 2, 7}, false, 2.5},
		{"transient, at the completion", Bias{transient: true}, []int{1, 2}, false, 2.5},
		{"transient, after it", Bias{transient: true}, []int{1, 2, 7}, false, 2},
		{"transient, at EOS", Bias{transient: true}, []int{1, 2}, true, 2},
		{"echo, at the completion", Bias{transient: true, echo: true}, []int{1, 2}, false, 4.5},
		{"echo, after it", Bias{transient: true, echo: true}, []int{1, 2, 7}, false, 2},
		{"repeated", Bias{}, []int{1, 2, 1, 2}, false, 5},
		{"once, repeated", Bias{once: true}, []int{1, 2, 1, 2}, false, 2.5},
		{"once, a repeat in progress", Bias{once: true}, []int{1, 2, 1}, false, 3.5},
		{"once, a repeat broken", Bias{once: true}, []int{1, 2, 1, 7}, false, 2.5},
	} {
		b := tc.b
		b.Weight, b.Revoke, b.Complete, b.Forms = 1, 1, 0.5, FormSpace
		k := build([][]int{{1, 2}}, unit, b)

		e := -1
		if tc.eos {
			e = eos
		}

		if got := k.Bonus(tc.text, e); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("%s: Bonus(%v, eos %v) = %g, want %g", tc.name, tc.text, tc.eos, got, tc.want)
		}
	}
}

// TestKeywordOnceAcrossForms pins that Bias.once counts a keyword once
// whatever its form: "1 2" and "3 2" spell one keyword, "5 6" another,
// under unit weights and a completion bonus of 0.5.
func TestKeywordOnceAcrossForms(t *testing.T) {
	ids := [][]int{{1, 2}, {3, 2}, {5, 6}}

	for _, tc := range []struct {
		name     string
		identify bool
		text     []int
		want     float64
	}{
		{"one form, then the other", true, []int{1, 2, 3, 2}, 2.5},
		{"the same form twice", true, []int{3, 2, 3, 2}, 2.5},
		{"two keywords", true, []int{1, 2, 5, 6}, 5},
		{"forms of their own without identify", false, []int{1, 2, 3, 2}, 5},
	} {
		b := Bias{Weight: 1, Revoke: 1, Complete: 0.5, Forms: FormSpace, once: true}
		k := build(ids, unit, b)

		if tc.identify {
			k.identify(ids, []int{0, 0, 1})
		}

		if got := k.Bonus(tc.text, -1); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("%s: Bonus(%v) = %g, want %g", tc.name, tc.text, got, tc.want)
		}
	}
}
