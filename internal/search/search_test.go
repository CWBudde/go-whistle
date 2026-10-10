package search

import (
	"math"
	"slices"
	"testing"
)

// TestOneBeamIsGreedy: one beam without length normalisation keeps the
// argmax token at every step, so it reproduces decoder.GreedyMemory token
// for token, with the forced and the detected language.
func TestOneBeamIsGreedy(t *testing.T) {
	d, clips := loadSetup(t, "ls-other-3080-5040-0018", "fleurs-de_de-10308608197319017935")

	for _, c := range clips {
		for _, lang := range []string{"", "fr"} {
			want, err := d.GreedyMemory(c.Mem, lang)
			if err != nil {
				t.Fatal(err)
			}

			opt := Default()
			opt.Beams, opt.Norm, opt.Stop, opt.Language = 1, NormNone, StopK, lang

			got, err := Beam(d, c.Mem, opt)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(got.Tokens, want.Tokens) || got.Text != want.Text || got.Language != want.Language ||
				got.EOS != want.EOS {
				t.Errorf("%s [%s]: beam 1 %q (%s, EOS %v), greedy %q (%s, EOS %v)", c.ID, lang, got.Text,
					got.Language, got.EOS, want.Text, want.Language, want.EOS)
			}

			if len(got.LogProbs) != len(got.Tokens)+1 {
				t.Errorf("%s: %d log-probs for %d tokens and EOS", c.ID, len(got.LogProbs), len(got.Tokens))
			}
		}
	}
}

// fastClips are short clips (LibriSpeech test-other and test-clean, FLEURS
// de) on which greedy decoding misses the engine's transcript and Default
// reproduces it exactly (docs/FINDINGS.md F-026).
var fastClips = []string{
	"ls-other-7902-96591-0023",
	"ls-clean-260-123440-0007",
	"fleurs-de_de-10229344228128634115",
}

// fastFloors are Default's ranking scores on fastClips, measured with the
// .cact weights (−0.0993, −0.1492, −0.0878) and floored just below.
var fastFloors = []float64{-0.11, -0.16, -0.10}

// TestBeamMatchesEngine beam-decodes fastClips with Default on the .cact
// weights, detecting the language, and requires the engine's language and
// text, EOS, and a ranking score above fastFloors.
func TestBeamMatchesEngine(t *testing.T) {
	d, clips := loadSetup(t, fastClips...)

	for i, c := range clips {
		r, err := Beam(d, c.Mem, Default())
		if err != nil {
			t.Fatal(err)
		}

		g, err := d.GreedyMemory(c.Mem, "")
		if err != nil {
			t.Fatal(err)
		}

		t.Logf("%s: [%s] %q, score %.4f; greedy %q", c.ID, r.Language, r.Text, r.Score, g.Text)

		if r.Language != c.Language || r.Text != c.Text || !r.EOS {
			t.Errorf("%s: got [%s] %q (EOS %v), engine [%s] %q", c.ID, r.Language, r.Text, r.EOS, c.Language, c.Text)
		}

		if r.Score < fastFloors[i] {
			t.Errorf("%s: score %.4f below %.2f", c.ID, r.Score, fastFloors[i])
		}
	}
}

// TestAdjust: a hook that bans the first text token Default picks makes
// the search start differently, the hook sees each live hypothesis' text,
// and Result.LogProbs stay the model's (the teacher-forced log p of the
// result).
func TestAdjust(t *testing.T) {
	d, clips := loadSetup(t, fastClips[1])
	c := clips[0]

	plain, err := Beam(d, c.Mem, Default())
	if err != nil {
		t.Fatal(err)
	}

	banned := plain.Tokens[0]
	maxText := 0

	opt := Default()
	opt.Adjust = func(text []int, logp []float64) {
		maxText = max(maxText, len(text))
		logp[banned] = math.Inf(-1)
	}

	r, err := Beam(d, c.Mem, opt)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("banned %q: %q, score %.4f", d.Tokenizer().Piece(banned), r.Text, r.Score)

	if slices.Contains(r.Tokens, banned) || maxText < len(r.Tokens) || !r.EOS {
		t.Fatalf("banned token %d: tokens %v, longest text seen %d", banned, r.Tokens, maxText)
	}

	var lang int

	for _, l := range d.Languages() {
		if l.Code == r.Language {
			lang = l.ID
		}
	}

	seq := append(append(append(d.Prompt(), lang), r.Tokens...), d.Tokenizer().EOSID())

	f, err := d.TeacherForce(c.Mem, seq)
	if err != nil {
		t.Fatal(err)
	}

	if len(f.LogProbs) != len(r.LogProbs)+1 {
		t.Fatalf("%d log-probs, teacher forcing has %d after the language token", len(r.LogProbs), len(f.LogProbs)-1)
	}

	for i, lp := range r.LogProbs {
		if math.Abs(lp-f.LogProbs[i+1]) > 1e-4 {
			t.Errorf("token %d: log p %.6f, teacher-forced %.6f", i, lp, f.LogProbs[i+1])
		}
	}

	if math.Abs(r.LangLogProb-f.LogProbs[0]) > 1e-4 {
		t.Errorf("language log p %.6f, teacher-forced %.6f", r.LangLogProb, f.LogProbs[0])
	}
}
