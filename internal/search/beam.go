// Package search decodes Whistle transcripts by beam search over the
// decoder (PLAN.md 6.2). Beams share their prefix through
// decoder.State.Fork, so a step costs one decoder position per live beam,
// and all of them read the same encoder memory. The engine's search is
// closed; the length normalisation, the stopping rule and the language
// step are axes of Options, and the probe harness (probe_test.go, build
// tag oracle) picks them by transcript parity with the engine
// (docs/FINDINGS.md F-026).
package search

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/cwbudde/go-whistle/internal/decoder"
)

// Result is the best hypothesis of a search.
type Result struct {
	decoder.Result

	LogProbs    []float64 // the model's log p of every text token, then of EOS when Result.EOS
	LangLogProb float64   // the model's log p of the language token
	Score       float64   // the ranking score: the (adjusted) summed log-probability over the length penalty
}

// cand is one next-token candidate of a hypothesis.
type cand struct {
	tok int
	lp  float64 // the model's log p
	adj float64 // log p after Adjust
}

// hyp is one hypothesis: its token sequence and score, and the decoder
// state that has consumed the first fed tokens of seq. Children share
// their parent's state until they extend it.
type hyp struct {
	seq    []int     // prompt, language token, text tokens
	text   int       // index of the first text token in seq
	lps    []float64 // model log p of the text tokens, then EOS
	langLP float64
	sum    float64 // summed model log p in the score
	adj    float64 // summed adjusted log p in the score: the ranking key
	eos    bool
	st     *decoder.State
	fed    int
}

// textLen is the number of text tokens.
func (h *hyp) textLen() int { return max(0, len(h.seq)-h.text) }

// textTokens is the text so far.
func (h *hyp) textTokens() []int {
	if len(h.seq) < h.text {
		return nil
	}

	return h.seq[h.text:]
}

// expander yields the next-token candidates of hypotheses.
type expander interface {
	// expand returns the n best next tokens of every hypothesis by
	// adjusted log-probability, best first; allowed restricts them (nil
	// for every token) and skips adjust.
	expand(hs []*hyp, n int, allowed []int, adjust Adjust) ([][]cand, error)
}

// Beam decodes the memory mem with d by beam search under opt.
func Beam(d *decoder.Decoder, mem *decoder.Memory, opt Options) (Result, error) {
	return run(d, &stepper{d: d}, d.NewState(mem), opt)
}

// run is the search over any expander, from the empty state root.
func run(d *decoder.Decoder, ex expander, root *decoder.State, opt Options) (Result, error) {
	err := opt.Validate()
	if err != nil {
		return Result{}, err
	}

	maxTok := opt.MaxTokens
	if maxTok == 0 {
		maxTok = d.MaxTokens()
	}

	alive, err := languageStep(d, ex, root, opt)
	if err != nil {
		return Result{}, err
	}

	s := &beam{opt: opt, eos: d.Tokenizer().EOSID(), spell: d.Decode}

	done := false
	for step := 0; step < maxTok-1 && !done; step++ {
		alive, done, err = s.step(ex, alive)
		if err != nil {
			return Result{}, err
		}
	}

	best := s.pick(alive, done)

	return result(d, best, s.final(best))
}

// pick is the hypothesis with the best final score among the finished
// ones. Unless the stopping rule ended the search, that is, at the token
// limit or with none finished, the live hypotheses compete too: under StopK
// the best of them by adjusted log-probability until there are Beams
// candidates, as Whisper's finalize fills its finished sequences
// (docs/FINDINGS.md F-026), under the other rules all of them.
func (s *beam) pick(alive []*hyp, done bool) *hyp {
	pool := slices.Clone(s.fin)

	if !done || len(pool) == 0 || s.opt.fin.pickLive {
		live := alive
		if s.opt.Stop == StopK && !s.opt.fin.pickLive {
			live = slices.Clone(alive)
			slices.SortStableFunc(live, func(a, b *hyp) int { return cmp.Compare(b.adj, a.adj) })
			live = live[:min(len(live), max(s.opt.Beams-len(pool), 0))]
		}

		pool = append(pool, live...)
	}

	return slices.MaxFunc(pool, func(a, b *hyp) int { return cmp.Compare(s.final(a), s.final(b)) })
}

// languageStep starts the hypotheses after the prompt: the forced
// language, the most likely language token, or one per language token up
// to opt.Beams (LangBeam).
func languageStep(d *decoder.Decoder, ex expander, root *decoder.State, opt Options) ([]*hyp, error) {
	prompt := d.Prompt()
	langs := d.Languages()
	ids := make([]int, 0, len(langs))

	for _, l := range langs {
		if opt.Language == "" || l.Code == opt.Language {
			ids = append(ids, l.ID)
		}
	}

	if len(ids) == 0 {
		return nil, fmt.Errorf("search: unknown language %q", opt.Language)
	}

	n := 1
	if opt.Lang == LangBeam {
		n = min(opt.Beams, len(ids))
	}

	start := &hyp{seq: prompt, text: len(prompt) + 1, st: root}

	cs, err := ex.expand([]*hyp{start}, n, ids, nil)
	if err != nil {
		return nil, err
	}

	out := make([]*hyp, len(cs[0]))

	for i, c := range cs[0] {
		h := &hyp{
			seq: append(slices.Clone(prompt), c.tok), text: start.text, langLP: c.lp, st: start.st, fed: start.fed,
		}
		// A forced language is no choice: only a detected one is scored.
		if opt.Lang != LangArgmax && opt.Language == "" {
			h.sum, h.adj = c.lp, c.lp
		}

		out[i] = h
	}

	return out, nil
}

// beam is the search's bookkeeping between steps.
type beam struct {
	opt Options
	eos int
	fin []*hyp // finished hypotheses (StopAll: those in the beam)

	spell   func([]int) (string, error) // the text of tokens, for the fin rules' dedup
	counted int                         // finished hypotheses that count toward fin.countTopK's slots
}

// slots is the number of finished hypotheses that ends StopK.
func (s *beam) slots() int {
	if s.opt.fin.slots > 0 {
		return s.opt.fin.slots
	}

	return s.opt.Beams
}

// textOf is h's text, or its token ids where they do not decode.
func (s *beam) textOf(h *hyp) string {
	txt, err := s.spell(h.textTokens())
	if err != nil {
		return fmt.Sprint(h.textTokens())
	}

	return txt
}

// length is L of h under opt.Length.
func (s *beam) length(h *hyp) int {
	l := h.textLen()
	if s.opt.Length == LenSeq {
		l = len(h.seq)
	}

	if h.eos && s.opt.Length != LenText {
		l++
	}

	return l
}

// score is h's normalised ranking score.
func (s *beam) score(h *hyp) float64 { return h.adj / s.opt.penalty(s.length(h)) }

// final is h's score for the final pick.
func (s *beam) final(h *hyp) float64 {
	if s.opt.rankModel {
		return h.sum / s.opt.penalty(s.length(h))
	}

	return s.score(h)
}

// child is h extended by c.
func (s *beam) child(h *hyp, c cand) *hyp {
	k := &hyp{
		seq: h.seq, text: h.text, lps: append(slices.Clone(h.lps), c.lp), langLP: h.langLP,
		sum: h.sum + c.lp, adj: h.adj + c.adj, st: h.st, fed: h.fed,
	}

	if c.tok == s.eos {
		k.eos = true
	} else {
		k.seq = append(slices.Clone(h.seq), c.tok)
	}

	return k
}

// step expands the live hypotheses by one token and returns the next live
// ones and whether the search is done.
func (s *beam) step(ex expander, alive []*hyp) ([]*hyp, bool, error) {
	k := s.opt.Beams

	kids, err := s.kids(ex, alive)
	if err != nil {
		return nil, false, err
	}

	if s.opt.Stop == StopAll {
		pool := append(slices.Clone(s.fin), kids...)
		slices.SortStableFunc(pool, func(a, b *hyp) int { return cmp.Compare(s.score(b), s.score(a)) })

		s.fin, alive = nil, nil

		for _, h := range pool[:min(k, len(pool))] {
			if h.eos {
				s.fin = append(s.fin, h)
			} else {
				alive = append(alive, h)
			}
		}

		return alive, len(alive) == 0, nil
	}

	alive = nil
	seen := map[string]bool{}

	for rank, h := range kids {
		if !h.eos {
			if s.opt.fin.dedupLive {
				txt := s.textOf(h)
				if seen[txt] {
					continue
				}

				seen[txt] = true
			}

			alive = append(alive, h)
			if len(alive) == k {
				break
			}

			continue
		}

		if s.opt.Admit == AdmitTopK && rank >= k {
			continue
		}

		s.finish(h, rank)
	}

	return alive, s.done(alive), nil
}

// kids is the step's candidates of the live hypotheses, best first.
// Whisper's k+1 per parent hold k live ones. Under fin.dedupLive two
// tokens of one parent can spell the same text (such as byte pieces that
// decode to U+FFFD), so the step draws a wider pool to keep the best of k
// distinct texts; EOS still counts only among a parent's best k+1
// (docs/FINDINGS.md F-042).
func (s *beam) kids(ex expander, alive []*hyp) ([]*hyp, error) {
	k := s.opt.Beams

	n := k + 1
	if s.opt.fin.dedupLive {
		n = dedupPool * (k + 1)
	}

	cs, err := ex.expand(alive, n, nil, s.opt.Adjust)
	if err != nil {
		return nil, err
	}

	var kids []*hyp

	for i, h := range alive {
		for j, c := range cs[i] {
			if j > k && c.tok == s.eos {
				continue
			}

			kids = append(kids, s.child(h, c))
		}
	}

	// Stable: ties keep the parent's rank, then the candidate's.
	slices.SortStableFunc(kids, func(a, b *hyp) int { return cmp.Compare(b.adj, a.adj) })

	return kids, nil
}

// dedupPool is how many times Whisper's k+1 candidates per parent a step
// draws under fin.dedupLive.
const dedupPool = 4

// finish records a finished hypothesis of the step's rank under the
// stopping rule's limits.
func (s *beam) finish(h *hyp, rank int) {
	k := s.opt.Beams

	switch s.opt.Stop {
	case StopK:
		s.finishK(h, rank)
	case StopHF:
		s.fin = append(s.fin, h)
		if len(s.fin) > k {
			w := s.worst()
			s.fin = slices.Delete(s.fin, w, w+1)
		}
	default:
		s.fin = append(s.fin, h)
	}
}

// finishK records a finished hypothesis under StopK: Whisper's slots, or
// a variant of opt.fin.
func (s *beam) finishK(h *hyp, rank int) {
	if s.opt.fin.dedupFin {
		txt := s.textOf(h)

		i := slices.IndexFunc(s.fin, func(f *hyp) bool { return s.textOf(f) == txt })
		if i >= 0 {
			if s.score(h) > s.score(s.fin[i]) {
				s.fin[i] = h
			}

			return
		}
	}

	switch {
	case s.opt.fin.countTopK:
		s.fin = append(s.fin, h)
		if rank < s.opt.Beams {
			s.counted++
		}
	case len(s.fin) < s.slots():
		s.fin = append(s.fin, h)
	case s.opt.fin.stopLive:
		// The slots are full but the search goes on: a better hypothesis
		// replaces the worst.
		if w := s.worst(); s.score(h) > s.score(s.fin[w]) {
			s.fin[w] = h
		}
	}
}

// worst is the index of the finished hypothesis with the lowest score.
func (s *beam) worst() int {
	worst := 0
	for i, f := range s.fin {
		if s.score(f) < s.score(s.fin[worst]) {
			worst = i
		}
	}

	return worst
}

// done applies the stopping rule after a step.
func (s *beam) done(alive []*hyp) bool {
	if len(alive) == 0 {
		return true
	}

	if len(s.fin) == 0 {
		return false
	}

	live := math.Inf(-1)
	for _, h := range alive {
		live = math.Max(live, s.score(h))
	}

	switch s.opt.Stop {
	case StopK:
		n := len(s.fin)
		if s.opt.fin.countTopK {
			n = s.counted
		}

		if n < s.slots() {
			return false
		}

		if s.opt.fin.stopLive {
			best := math.Inf(-1)
			for _, h := range s.fin {
				best = math.Max(best, s.score(h))
			}

			return best >= live
		}

		return true
	case StopBest:
		best := math.Inf(-1)
		for _, h := range s.fin {
			best = math.Max(best, s.score(h))
		}

		return best >= live
	case StopHF:
		if len(s.fin) < s.opt.Beams {
			return false
		}

		worst := math.Inf(1)
		for _, h := range s.fin {
			worst = math.Min(worst, s.score(h))
		}

		return worst >= live
	}

	return false
}

// result turns the best hypothesis into a Result.
func result(d *decoder.Decoder, h *hyp, score float64) (Result, error) {
	r := Result{LogProbs: h.lps, LangLogProb: h.langLP, Score: score}
	r.Tokens = slices.Clone(h.textTokens())
	r.EOS = h.eos

	lang := h.seq[h.text-1]
	for _, l := range d.Languages() {
		if l.ID == lang {
			r.Language = l.Code
		}
	}

	var err error

	r.Text, err = d.Decode(r.Tokens)

	return r, err
}

// stepper is the expander over the decoder: it brings every hypothesis'
// state up to its sequence, forking a state that another hypothesis still
// holds, and runs the hypotheses on all cores. Adjust runs on the calling
// goroutine, one hypothesis after the other.
type stepper struct {
	d *decoder.Decoder
}

func (e *stepper) expand(hs []*hyp, n int, allowed []int, adjust Adjust) ([][]cand, error) {
	own(hs, func(*hyp) bool { return true })

	lps := make([][]float64, len(hs))
	errs := make([]error, len(hs))

	var wg sync.WaitGroup

	for i, h := range hs {
		wg.Go(func() {
			logits, err := e.d.Extend(h.st, h.seq[h.fed:])
			if err != nil {
				errs[i] = err

				return
			}

			h.fed = len(h.seq)
			lps[i] = logSoftmax(logits[len(logits)-e.d.Vocab():])
		})
	}

	wg.Wait()

	out := make([][]cand, len(hs))

	for i, h := range hs {
		if errs[i] != nil {
			return nil, errs[i]
		}

		out[i] = candidates(lps[i], h.textTokens(), n, allowed, adjust)
	}

	return out, nil
}

// own gives every hypothesis that feed selects a state of its own: a
// state another hypothesis in hs still holds is forked, so the last holder
// that feeds it extends it in place. Hypotheses outside hs are dead and
// never feed their state again.
func own(hs []*hyp, feed func(*hyp) bool) {
	refs := map[*decoder.State]int{}
	for _, h := range hs {
		refs[h.st]++
	}

	for _, h := range hs {
		if !feed(h) {
			continue
		}

		if refs[h.st] > 1 {
			refs[h.st]--
			h.st = h.st.Fork()
			refs[h.st] = 1
		}
	}
}

// candidates is the n best tokens of one row of log-probabilities lp after
// adjust, best first, ties to the lower id; allowed restricts the tokens
// and skips adjust.
func candidates(lp []float64, text []int, n int, allowed []int, adjust Adjust) []cand {
	adj := lp

	if adjust != nil && allowed == nil {
		adj = slices.Clone(lp)
		adjust(slices.Clone(text), adj)
	}

	top := make([]cand, 0, n+1)
	pick := func(tok int) {
		c := cand{tok: tok, lp: lp[tok], adj: adj[tok]}
		if len(top) == n && c.adj <= top[n-1].adj {
			return
		}

		i := len(top)
		for i > 0 && top[i-1].adj < c.adj {
			i--
		}

		top = slices.Insert(top, i, c)
		top = top[:min(len(top), n)]
	}

	if allowed != nil {
		for _, tok := range allowed {
			pick(tok)
		}

		return top
	}

	for tok := range lp {
		pick(tok)
	}

	return top
}

// logSoftmax is the log softmax of v in float64.
func logSoftmax(v []float32) []float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = math.Max(m, float64(x))
	}

	var sum float64
	for _, x := range v {
		sum += math.Exp(float64(x) - m)
	}

	lse := m + math.Log(sum)
	out := make([]float64, len(v))

	for i, x := range v {
		out[i] = float64(x) - lse
	}

	return out
}
