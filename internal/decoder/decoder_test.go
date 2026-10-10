package decoder

import (
	"errors"
	"math"
	"slices"
	"testing"
)

func TestHypothesisRoundTrip(t *testing.T) {
	for _, a := range Axes() {
		for v := range a.Names {
			h := Best()
			a.Set(&h, v+1)

			got, err := ParseHypothesis(h.String())
			if err != nil {
				t.Fatalf("%s: %v", h, err)
			}

			if got != h {
				t.Errorf("round trip: %s → %s", h, got)
			}
		}
	}

	for _, key := range []string{"", "cross=after-attn", Best().String() + " extra=1"} {
		_, err := ParseHypothesis(key)
		if !errors.Is(err, ErrHypothesis) {
			t.Errorf("ParseHypothesis(%q) = %v, want ErrHypothesis", key, err)
		}
	}

	err := (Hypothesis{}).Validate()
	if !errors.Is(err, ErrHypothesis) {
		t.Errorf("zero hypothesis: %v", err)
	}
}

// fastClips are short clips from LibriSpeech test-other, test-clean and
// FLEURS de on which greedy decoding with Best() reproduces the engine's
// transcript exactly (docs/FINDINGS.md F-024).
var fastClips = []string{
	"ls-other-3080-5040-0018",
	"ls-clean-5639-40744-0011",
	"fleurs-de_de-10308608197319017935",
}

// fastFloors are the teacher-forced mean log-probabilities of fastClips'
// engine transcripts under Best(), measured with the .cact weights (−0.0880,
// −0.0548, −0.0652) and floored just below.
var fastFloors = []float64{-0.09, -0.06, -0.07}

// TestGreedyMatchesEngine decodes fastClips greedily with Best() on the
// .cact weights, detecting the language, and requires the engine's language
// and text; it also holds their teacher-forced scores to fastFloors.
func TestGreedyMatchesEngine(t *testing.T) {
	src := openModel(t, whistleCactPath)
	clips := loadClips(t, fastClips...)
	encs, frames := encode(t, src, clips, MemFinalNorm)

	d, err := Load(src, nil, Best())
	if err != nil {
		t.Fatal(err)
	}

	for i, c := range clips {
		mem, err := d.Memory(encs[i], frames[i])
		if err != nil {
			t.Fatal(err)
		}

		r, err := d.GreedyMemory(mem, "")
		if err != nil {
			t.Fatal(err)
		}

		seq, err := d.Targets(c.Language, c.Text)
		if err != nil {
			t.Fatal(err)
		}

		f, err := d.TeacherForce(mem, seq)
		if err != nil {
			t.Fatal(err)
		}

		lp := mean64(f.LogProbs)
		t.Logf("%s: [%s] %q, teacher-forced mean logp %.4f, language rank %d", c.ID, r.Language, r.Text, lp,
			f.LangRank)

		if r.Language != c.Language || r.Text != c.Text || !r.EOS {
			t.Errorf("%s: got [%s] %q (EOS %v), engine [%s] %q", c.ID, r.Language, r.Text, r.EOS, c.Language, c.Text)
		}

		if lp < fastFloors[i] || f.LangRank != 0 {
			t.Errorf("%s: mean logp %.4f below %.2f or language rank %d", c.ID, lp, fastFloors[i], f.LangRank)
		}
	}
}

// TestForcedLanguage forces each language on an English clip: the language
// token is the forced one, and decoding still ends at EOS.
func TestForcedLanguage(t *testing.T) {
	src := openModel(t, whistleCactPath)
	clips := loadClips(t, fastClips[0])
	encs, frames := encode(t, src, clips, MemFinalNorm)

	d, err := Load(src, nil, Best())
	if err != nil {
		t.Fatal(err)
	}

	for _, l := range d.Languages() {
		r, err := d.Greedy(encs[0], frames[0], l.Code)
		if err != nil {
			t.Fatal(err)
		}

		if r.Language != l.Code || len(r.Tokens) == 0 || len(r.Tokens) >= d.MaxTokens() {
			t.Errorf("forced %s: [%s] %d tokens %q", l.Code, r.Language, len(r.Tokens), r.Text)
		}
	}

	_, err = d.Greedy(encs[0], frames[0], "xx")
	if err == nil {
		t.Error("unknown language: no error")
	}
}

// TestIncrementalMatchesFullRun pins the KV cache (docs/FINDINGS.md F-022):
// a greedy transcript decoded token by token through the caches, fed again
// in one pass with empty caches, gives the same logits at every position
// within the F-009 contract, and the same argmax, so every greedy choice is
// the full re-run's.
func TestIncrementalMatchesFullRun(t *testing.T) {
	src := openModel(t, whistleCactPath)
	clips := loadClips(t, fastClips[2])
	encs, frames := encode(t, src, clips, MemFinalNorm)

	d, err := Load(src, nil, Best())
	if err != nil {
		t.Fatal(err)
	}

	mem, err := d.Memory(encs[0], frames[0])
	if err != nil {
		t.Fatal(err)
	}

	r, err := d.GreedyMemory(mem, "")
	if err != nil {
		t.Fatal(err)
	}

	seq, err := d.Targets(r.Language, r.Text)
	if err != nil {
		t.Fatal(err)
	}

	full, err := d.Extend(d.NewState(mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	s := d.NewState(mem)
	v := d.Vocab()

	var worst, worstRef float64

	for i, tok := range seq {
		step, err := d.Extend(s, []int{tok})
		if err != nil {
			t.Fatal(err)
		}

		ref := full[i*v : (i+1)*v]
		maxAbs, refMax, cos := compare(step, ref)

		if maxAbs > 1e-5*math.Max(1, refMax) || cos < 0.99999 || argmax(step) != argmax(ref) {
			t.Errorf("position %d: max |Δ| %.3g at max |ref| %.3g, cosine %.9f, argmax %d vs %d", i, maxAbs, refMax,
				cos, argmax(step), argmax(ref))
		}

		if i+1 < len(seq) && i >= len(d.Prompt()) && argmax(ref) != seq[i+1] {
			t.Errorf("position %d: full run's argmax %d, greedy chose %d", i, argmax(ref), seq[i+1])
		}

		if maxAbs > worst {
			worst, worstRef = maxAbs, refMax
		}
	}

	if !slices.Equal(s.Tokens(), seq) {
		t.Errorf("state tokens %v, want %v", s.Tokens(), seq)
	}

	t.Logf("%d positions: worst max |Δ| %.3g at max |logit| %.3g", len(seq), worst, worstRef)
}

// compare returns max |got − want|, max |want| and the cosine.
func compare(got, want []float32) (maxAbs, refMax, cos float64) {
	var dot, gg, ww float64

	for i, w := range want {
		g, wf := float64(got[i]), float64(w)
		maxAbs = math.Max(maxAbs, math.Abs(g-wf))
		refMax = math.Max(refMax, math.Abs(wf))
		dot += g * wf
		gg += g * g
		ww += wf * wf
	}

	return maxAbs, refMax, dot / math.Sqrt(gg*ww)
}

func mean64(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x
	}

	return s / float64(len(v))
}
