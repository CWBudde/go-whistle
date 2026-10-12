package search

import (
	"fmt"
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

// TestForcedLanguageUnscored: a forced language token is not a choice, so
// LangArgmaxScored and LangBeam leave its log-probability out of the score
// and decode like LangArgmax.
func TestForcedLanguageUnscored(t *testing.T) {
	d, clips := loadSetup(t, "ls-other-3080-5040-0018")

	opt := Default()
	opt.Language = "fr"

	want, err := Beam(d, clips[0].Mem, opt)
	if err != nil {
		t.Fatal(err)
	}

	for _, l := range []LangStep{LangArgmaxScored, LangBeam} {
		opt.Lang = l

		got, err := Beam(d, clips[0].Mem, opt)
		if err != nil {
			t.Fatal(err)
		}

		if got.Text != want.Text || got.Score != want.Score || got.LangLogProb != want.LangLogProb {
			t.Errorf("language step %d: %q score %.6f, LangArgmax %q score %.6f", l, got.Text, got.Score, want.Text,
				want.Score)
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

// depthClips are short clips whose engine transcript at --audio-depth 2
// and 4 differs from the full depth's, and which Default on
// decoder.WithDepth reproduces at both (docs/FINDINGS.md F-030).
var depthClips = []string{
	"ls-other-8280-266249-0032",
	"ls-clean-260-123440-0007",
	"ls-clean-7729-102255-0012",
}

// depthFloors are Default's ranking scores on depthClips at depth 2 and 4,
// measured with the .cact weights (d2 −0.3097, −0.1498, −0.1133; d4
// −0.1675, −0.1146, −0.0719) and floored just below.
var depthFloors = map[int][]float64{2: {-0.32, -0.16, -0.12}, 4: {-0.18, -0.12, -0.08}}

// TestDepthMatchesEngine beam-decodes depthClips with Default at ladder
// depths 2 and 4 and requires the engine's text at that depth
// (testdata/oracle/depth), which is not its full-depth text, and a
// ranking score above depthFloors.
func TestDepthMatchesEngine(t *testing.T) {
	d, clips := loadSetup(t, depthClips...)
	texts := depthTexts(t)

	for _, depth := range []int{2, 4} {
		dd, err := d.WithDepth(depth)
		if err != nil {
			t.Fatal(err)
		}

		for i, c := range clips {
			want := texts[fmt.Sprintf("%s.d%d", c.ID, depth)]

			r, err := Beam(dd, c.Mem, Default())
			if err != nil {
				t.Fatal(err)
			}

			t.Logf("d%d %s: %q, score %.4f; full depth %q", depth, c.ID, r.Text, r.Score, c.Text)

			if want.Text == c.Text {
				t.Errorf("d%d %s: the fixture equals the full depth's text", depth, c.ID)
			}

			if r.Language != want.Language || r.Text != want.Text || !r.EOS {
				t.Errorf("d%d %s: got [%s] %q (EOS %v), engine [%s] %q", depth, c.ID, r.Language, r.Text, r.EOS,
					want.Language, want.Text)
			}

			if r.Score < depthFloors[depth][i] {
				t.Errorf("d%d %s: score %.4f below %.2f", depth, c.ID, r.Score, depthFloors[depth][i])
			}
		}
	}
}

// keywordFast are keyword fixture cases on short clips that Default with
// DefaultBias reproduces (docs/FINDINGS.md F-029), in manifest order: a
// lower-cased missed name the engine writes capitalised, a distractor that
// loses against the audio, one that wins, and a list of two keywords.
var keywordFast = []string{
	"en-cowleys-lower-missed",
	"en-curses-distractor",
	"en-transfer-distractor",
	"de-paare-adoptionsplan-list",
}

// keywordFloors are their ranking scores with the bonus, measured with the
// .cact weights (0.1395, −0.1575, 0.0072, 0.1673) and floored just below.
var keywordFloors = []float64{0.13, -0.17, 0, 0.16}

// TestKeywordsMatchEngine beam-decodes keywordFast with Default and the
// Adjust of their keyword files under DefaultBias, and requires the
// engine's text with the keywords, a ranking score above keywordFloors,
// and, where the engine's text changes, a change of ours.
func TestKeywordsMatchEngine(t *testing.T) {
	cases := keywordCases(t, keywordFast...)

	var ids []string
	for _, c := range cases {
		if !slices.Contains(ids, c.Clip) {
			ids = append(ids, c.Clip)
		}
	}

	d, clips := loadSetup(t, ids...)

	for i, c := range cases {
		cl := clips[slices.Index(ids, c.Clip)]

		kw, err := NewKeywords(d.Tokenizer(), c.Keywords, DefaultBias())
		if err != nil {
			t.Fatal(err)
		}

		opt := Default()
		opt.Adjust = kw.Adjust

		r, err := Beam(d, cl.Mem, opt)
		if err != nil {
			t.Fatal(err)
		}

		t.Logf("%s %q: %q, score %.4f", c.Name, c.Keywords, r.Text, r.Score)

		if r.Text != c.TextWith || r.Language != c.Language || !r.EOS {
			t.Errorf("%s: got [%s] %q (EOS %v), engine [%s] %q", c.Name, r.Language, r.Text, r.EOS, c.Language, c.TextWith)
		}

		if c.Changed == (cl.Text == r.Text) {
			t.Errorf("%s: the engine's text changed %v, ours %v", c.Name, c.Changed, cl.Text != r.Text)
		}

		if r.Score < keywordFloors[i] {
			t.Errorf("%s: score %.4f below %.2f", c.Name, r.Score, keywordFloors[i])
		}
	}
}
