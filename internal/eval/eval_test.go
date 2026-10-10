package eval

import (
	"math"
	"slices"
	"testing"
)

func TestEditDistance(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"abc", "", 3},
		{"", "abc", 3},
		{"kitten", "sitting", 3},
		{"flaw", "lawn", 2},
		{"über", "uber", 1},
	}

	for _, c := range cases {
		if got := EditDistance([]rune(c.a), []rune(c.b)); got != c.want {
			t.Errorf("EditDistance(%q, %q) = %d, want %d", c.a, c.b, got, c.want)
		}

		if got := EditDistance([]rune(c.b), []rune(c.a)); got != c.want {
			t.Errorf("EditDistance(%q, %q) = %d, want %d (symmetry)", c.b, c.a, got, c.want)
		}
	}

	if got := EditDistance([]int{1, 2, 3, 4}, []int{2, 3, 5}); got != 2 {
		t.Errorf("EditDistance over ints = %d, want 2", got)
	}
}

func TestWords(t *testing.T) {
	cases := []struct {
		text string
		want []string
	}{
		{"She found a door.", []string{"she", "found", "a", "door"}},
		{"\"Don't,\" he said — twice!", []string{"don't", "he", "said", "twice"}},
		{"Über 3 Brücken", []string{"über", "3", "brücken"}},
		{"  ...  ", nil},
	}

	for _, c := range cases {
		if got := Words(c.text); !slices.Equal(got, c.want) {
			t.Errorf("Words(%q) = %q, want %q", c.text, got, c.want)
		}
	}
}

func TestErrorRates(t *testing.T) {
	cases := []struct {
		ref, hyp string
		wer, cer float64
	}{
		{"She found a door.", "She found a door.", 0, 0},
		{"She found a door.", "she found the door", 0.25, 5.0 / 17},
		{"a b c d", "a c d e", 0.5, 3.0 / 7},
		{"", "", 0, 0},
		{"", "two words", 2, 9},
	}

	for _, c := range cases {
		if w := WER(c.ref, c.hyp); math.Abs(w-c.wer) > 1e-12 {
			t.Errorf("WER(%q, %q) = %v, want %v", c.ref, c.hyp, w, c.wer)
		}

		if r := CER(c.ref, c.hyp); math.Abs(r-c.cer) > 1e-12 {
			t.Errorf("CER(%q, %q) = %v, want %v", c.ref, c.hyp, r, c.cer)
		}
	}
}
