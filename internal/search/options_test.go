package search

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
)

func TestOptionsValidate(t *testing.T) {
	err := Default().Validate()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}

	for name, mod := range map[string]func(*Options){
		"zero beams":    func(o *Options) { o.Beams = 0 },
		"negative max":  func(o *Options) { o.MaxTokens = -1 },
		"unset norm":    func(o *Options) { o.Norm = 0 },
		"unset length":  func(o *Options) { o.Length = 0 },
		"unknown stop":  func(o *Options) { o.Stop = StopHF + 1 },
		"unset admit":   func(o *Options) { o.Admit = 0 },
		"unset lang":    func(o *Options) { o.Lang = 0 },
		"NaN alpha":     func(o *Options) { o.Alpha = math.NaN() },
		"infinite α":    func(o *Options) { o.Alpha = math.Inf(1) },
		"zero value":    func(o *Options) { *o = Options{} },
		"unknown norm":  func(o *Options) { o.Norm = NormGNMT + 1 },
		"unknown admit": func(o *Options) { o.Admit = AdmitAboveLive + 1 },
	} {
		o := Default()
		mod(&o)

		err := o.Validate()
		if !errors.Is(err, ErrOptions) {
			t.Errorf("%s: %v, want ErrOptions", name, err)
		}
	}

	want := "norm=len len=text stop=k admit=above-live lang=argmax beams=5"
	if got := Default().String(); got != want {
		t.Errorf("Default().String() = %q, want %q", got, want)
	}

	o := Options{
		Beams: 3, Norm: NormGNMT, Alpha: 0.6, Length: LenSeq, Stop: StopAll, Admit: AdmitTopK, Lang: LangBeam,
		MaxTokens: 50,
	}
	if got := o.String(); got != "norm=gnmt α=0.6 len=seq stop=all lang=beam beams=3 max=50" {
		t.Errorf("String() = %q", got)
	}
}

func TestPenalty(t *testing.T) {
	cases := []struct {
		norm  Norm
		alpha float64
		l     int
		want  float64
	}{
		{NormNone, 0, 7, 1},
		{NormLen, 0, 7, 7},
		{NormLen, 0, 0, 1},
		{NormPow, 0.5, 9, 3},
		{NormPow, 2, 3, 9},
		{NormGNMT, 1, 7, 2},
		{NormGNMT, 2, 1, 1},
	}

	for _, c := range cases {
		o := Default()
		o.Norm, o.Alpha = c.norm, c.alpha

		if got := o.penalty(c.l); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("%s penalty(%d) = %v, want %v", o, c.l, got, c.want)
		}
	}
}

// TestCandidates: the n best tokens by adjusted log softmax, best first,
// ties to the lower id; allowed restricts them and skips the hook; the
// hook sees the text and changes the ranking but not the model's log p.
func TestCandidates(t *testing.T) {
	lp := logSoftmax([]float32{1, 3, 2, 3, 0, 2.5})

	var sum float64
	for _, v := range lp {
		sum += math.Exp(v)
	}

	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("softmax sums to %v", sum)
	}

	toks := func(cs []cand) []int {
		out := make([]int, len(cs))
		for i, c := range cs {
			out[i] = c.tok
		}

		return out
	}

	if got := toks(candidates(lp, nil, 4, nil, nil)); !slices.Equal(got, []int{1, 3, 5, 2}) {
		t.Errorf("top 4 = %v, want [1 3 5 2]", got)
	}

	if got := toks(candidates(lp, nil, 10, nil, nil)); !slices.Equal(got, []int{1, 3, 5, 2, 0, 4}) {
		t.Errorf("top 10 = %v, want all six", got)
	}

	if got := toks(candidates(lp, nil, 2, []int{0, 4, 2}, nil)); !slices.Equal(got, []int{2, 0}) {
		t.Errorf("allowed top 2 = %v, want [2 0]", got)
	}

	var seen []int

	ban := func(text []int, logp []float64) {
		seen = text
		logp[1] = math.Inf(-1)
		logp[4] += 10
	}

	cs := candidates(lp, []int{7, 8}, 3, nil, ban)
	if got := toks(cs); !slices.Equal(got, []int{4, 3, 5}) || !slices.Equal(seen, []int{7, 8}) {
		t.Errorf("adjusted top 3 = %v (text %v), want [4 3 5] ([7 8])", got, seen)
	}

	if cs[0].lp != lp[4] || cs[0].adj != lp[4]+10 {
		t.Errorf("adjusted candidate lp %v adj %v, want %v and %v", cs[0].lp, cs[0].adj, lp[4], lp[4]+10)
	}

	seen = nil

	candidates(lp, []int{7}, 1, []int{2}, ban)

	if seen != nil {
		t.Error("the hook ran on a restricted step")
	}
}

// fixed is an expander that serves the same candidates to every
// hypothesis with the given text length.
type fixed map[int][]cand

func (f fixed) expand(hs []*hyp, n int, _ []int, _ Adjust) ([][]cand, error) {
	out := make([][]cand, len(hs))
	for i, h := range hs {
		out[i] = f[h.textLen()][:min(n, len(f[h.textLen()]))]
	}

	return out, nil
}

// TestStepRules pins the admission and stopping rules on one step from one
// live hypothesis with 2 beams. In high the candidates are a (−0.1), EOS
// (−0.5), b (−1), c (−2): EOS ranks second, inside the top 2 and above b,
// the last live hypothesis kept, so both admission rules finish it. In low
// EOS ranks third, below b, so neither does. StopAll keeps the finished
// hypothesis in the beam, which leaves one live slot.
func TestStepRules(t *testing.T) {
	const eos = 1

	start := func() []*hyp { return []*hyp{{seq: []int{2, 9}, text: 2}} }
	c := func(tok int, lp float64) cand { return cand{tok: tok, lp: lp, adj: lp} }

	high := fixed{0: {c(10, -0.1), c(eos, -0.5), c(11, -1), c(12, -2)}}
	low := fixed{0: {c(10, -0.1), c(11, -0.3), c(eos, -0.5), c(12, -2)}}

	for _, tc := range []struct {
		name     string
		ex       fixed
		stop     Stop
		admit    Admit
		alive    []int // last token of each live hypothesis
		finished int
		done     bool
	}{
		{"above-live, EOS second", high, StopK, AdmitAboveLive, []int{10, 11}, 1, false},
		{"top-k, EOS second", high, StopK, AdmitTopK, []int{10, 11}, 1, false},
		{"above-live, EOS third", low, StopK, AdmitAboveLive, []int{10, 11}, 0, false},
		{"top-k, EOS third", low, StopK, AdmitTopK, []int{10, 11}, 0, false},
		{"best, EOS second", high, StopBest, AdmitAboveLive, []int{10, 11}, 1, false},
		{"all, EOS second", high, StopAll, AdmitTopK, []int{10}, 1, false},
	} {
		o := Default()
		o.Beams, o.Stop, o.Admit, o.Norm = 2, tc.stop, tc.admit, NormNone
		s := &beam{opt: o, eos: eos}

		next, done, err := s.step(tc.ex, start())
		if err != nil {
			t.Fatal(err)
		}

		var last []int
		for _, h := range next {
			last = append(last, h.seq[len(h.seq)-1])
		}

		if !slices.Equal(last, tc.alive) || len(s.fin) != tc.finished || done != tc.done {
			t.Errorf("%s: alive %v, finished %d, done %v; want %v, %d, %v", tc.name, last, len(s.fin), done, tc.alive,
				tc.finished, tc.done)
		}
	}

	// StopBest ends once the finished score reaches the best live one:
	// EOS at −0.05 beats a at −0.1.
	o := Default()
	o.Beams, o.Stop, o.Norm = 2, StopBest, NormNone
	s := &beam{opt: o, eos: eos}

	_, done, err := s.step(fixed{0: {c(eos, -0.05), c(10, -0.1), c(11, -1)}}, start())
	if err != nil || !done {
		t.Errorf("StopBest with the finished hypothesis on top: done %v, err %v", done, err)
	}

	// StopK ends at Beams finished hypotheses: two EOS candidates from two
	// live parents.
	o.Stop = StopK
	s = &beam{opt: o, eos: eos}
	two := []*hyp{{seq: []int{2, 9, 10}, text: 2}, {seq: []int{2, 9, 11}, text: 2, adj: -0.2}}

	_, done, err = s.step(fixed{1: {c(eos, -0.1), c(12, -3)}}, two)
	if err != nil || !done || len(s.fin) != 2 {
		t.Errorf("StopK with two finished: done %v, finished %d, err %v", done, len(s.fin), err)
	}
}

// TestPickAtTokenLimit: a search that reaches the token limit before
// StopK's Beams hypotheses have finished lets the best live ones compete
// with the finished, as Whisper's finalize does, up to Beams of them; a
// search StopK ended picks among the finished only.
func TestPickAtTokenLimit(t *testing.T) {
	o := Default()
	o.Beams, o.Norm = 3, NormNone

	short := &hyp{seq: []int{2, 9}, text: 2, adj: -5, sum: -5, eos: true}
	good := &hyp{seq: []int{2, 9, 10}, text: 2, adj: -1, sum: -1}
	worse := &hyp{seq: []int{2, 9, 11}, text: 2, adj: -2, sum: -2}
	third := &hyp{seq: []int{2, 9, 12}, text: 2, adj: -0.5, sum: -0.5}

	s := &beam{opt: o, fin: []*hyp{short}}

	if got := s.pick([]*hyp{good, worse}, false); got != good {
		t.Errorf("at the limit: picked adj %v, want the live −1", got.adj)
	}

	// Beams 3 with one finished admits the two best live by adj: third
	// (−0.5) and good (−1), not worse.
	if got := s.pick([]*hyp{worse, good, third}, false); got != third {
		t.Errorf("at the limit, three live: picked adj %v, want −0.5", got.adj)
	}

	s.opt.Beams = 2
	if got := s.pick([]*hyp{worse, third}, false); got != third {
		t.Errorf("Beams 2: picked adj %v, want −0.5", got.adj)
	}

	if got := s.pick([]*hyp{good, worse}, true); got != short {
		t.Errorf("stopped by StopK: picked adj %v, want the finished −5", got.adj)
	}

	if got := (&beam{opt: o}).pick([]*hyp{worse, good}, false); got != good {
		t.Errorf("none finished: picked adj %v, want −1", got.adj)
	}
}

// byLast is an expander that serves candidates by a hypothesis' last
// token.
type byLast map[int][]cand

func (e byLast) expand(hs []*hyp, n int, _ []int, _ Adjust) ([][]cand, error) {
	out := make([][]cand, len(hs))
	for i, h := range hs {
		cs := e[h.seq[len(h.seq)-1]]
		out[i] = cs[:min(n, len(cs))]
	}

	return out, nil
}

// TestFinRules pins the finished-hypothesis variants of the F-042 probe
// against Whisper's rule (the zero finRule) with 2 beams. Tokens 10 and 11
// spell the same text "a", 12 spells "b".
func TestFinRules(t *testing.T) {
	const eos = 1

	c := func(tok int, lp float64) cand { return cand{tok: tok, lp: lp, adj: lp} }
	spell := func(ids []int) (string, error) {
		var b strings.Builder
		for _, id := range ids {
			b.WriteString(map[int]string{10: "a", 11: "a", 12: "b", 13: "c"}[id])
		}

		return b.String(), nil
	}

	twin := func() []*hyp {
		return []*hyp{{seq: []int{2, 9, 10}, text: 2}, {seq: []int{2, 9, 11}, text: 2, adj: -0.2}}
	}
	ends := byLast{10: {c(eos, -0.1), c(12, -3)}, 11: {c(eos, -0.1), c(13, -3)}}

	for _, tc := range []struct {
		name     string
		rule     finRule
		finished int
		done     bool
	}{
		{"Whisper: two finished fill 2 slots", finRule{}, 2, true},
		{"one text takes one slot", finRule{dedupFin: true}, 1, false},
		{"3 slots", finRule{slots: 3}, 2, false},
	} {
		o := Default()
		o.Beams, o.Norm, o.fin = 2, NormNone, tc.rule
		s := &beam{opt: o, eos: eos, spell: spell}

		_, done, err := s.step(ends, twin())
		if err != nil || len(s.fin) != tc.finished || done != tc.done {
			t.Errorf("%s: finished %d, done %v, err %v; want %d, %v", tc.name, len(s.fin), done, err, tc.finished,
				tc.done)
		}
	}

	// Live hypotheses with the same text: only the best is kept.
	o := Default()
	o.Beams, o.Norm, o.fin = 2, NormNone, finRule{dedupLive: true}
	s := &beam{opt: o, eos: eos, spell: spell}

	alive, _, err := s.step(byLast{9: {c(10, -0.1), c(11, -0.2), c(12, -0.3)}}, []*hyp{{seq: []int{2, 9}, text: 2}})
	if err != nil || len(alive) != 2 || alive[1].seq[2] != 12 {
		t.Errorf("dedupLive: %d alive, err %v; want a and b", len(alive), err)
	}

	// Only EOS among the step's best Beams count: a (−0.1) and EOS (−0.5)
	// are the best two, the second EOS (−0.6) is admitted above b but
	// does not count.
	o.fin = finRule{countTopK: true}
	s = &beam{opt: o, eos: eos, spell: spell}
	two := []*hyp{{seq: []int{2, 9, 12}, text: 2}, {seq: []int{2, 9, 13}, text: 2}}

	_, done, err := s.step(byLast{12: {c(10, -0.1), c(eos, -0.5)}, 13: {c(eos, -0.6), c(11, -2)}}, two)
	if err != nil || len(s.fin) != 2 || s.counted != 1 || done {
		t.Errorf("countTopK: finished %d, counted %d, done %v, err %v; want 2, 1, false", len(s.fin), s.counted, done,
			err)
	}

	// Slots full, but a live hypothesis outscores the best finished one.
	o.fin = finRule{stopLive: true}
	s = &beam{opt: o, eos: eos, spell: spell}

	_, done, err = s.step(byLast{12: {c(10, -0.1), c(eos, -0.5)}, 13: {c(eos, -0.6), c(11, -2)}}, two)
	if err != nil || len(s.fin) != 2 || done {
		t.Errorf("stopLive: finished %d, done %v, err %v; want 2, false", len(s.fin), done, err)
	}

	// The final pick weighs the live hypotheses even after the stop.
	short := &hyp{seq: []int{2, 9}, text: 2, adj: -5, sum: -5, eos: true}
	good := &hyp{seq: []int{2, 9, 10}, text: 2, adj: -1, sum: -1}

	o.fin = finRule{pickLive: true}
	if got := (&beam{opt: o, fin: []*hyp{short, short}}).pick([]*hyp{good}, true); got != good {
		t.Errorf("pickLive: picked adj %v, want the live −1", got.adj)
	}
}
