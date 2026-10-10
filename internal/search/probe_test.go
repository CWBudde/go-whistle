//go:build oracle

// The beam search probe harness (PLAN.md 6.2, docs/FINDINGS.md F-026). The
// engine decodes with 5 beams, but its length normalisation, stopping rule
// and language step are closed (U8). TestProbeBeam decodes the 32 fixture
// clips with every variant of a grid over those axes and ranks them by
// exact transcript equality with the engine (testdata/oracle/<clip>.json),
// then by mean CER and WER. The search runs on a memo of next-token
// candidates per prefix, shared by all variants of a clip, so the grid
// costs little more than the distinct prefixes it visits. For the leader
// it prints the per-clip table (engine, beam, greedy) and explains every
// remaining mismatch by teacher forcing both transcripts: their summed
// log-probabilities and ranking scores under our model, and the first
// token where they part. It then reruns the leader with the real search
// (Beam), checks it gives the memo's transcripts, and times it against
// greedy decoding. Run it with `just search-probe`; the knobs are
// environment variables:
//
//	WHISTLE_PROBE_TOP    leaderboard rows (default 25)
//	WHISTLE_PROBE_KEY    an Options.String to report instead of the leader

package search

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/eval"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// memoTop is how many candidates the memo keeps per prefix: enough for up
// to memoTop − 1 beams.
const memoTop = 16

// memo is an expander that remembers every prefix's best candidates, so
// variants that visit the same prefix share one decoder position. A miss
// feeds all of a hypothesis' pending tokens in one call and records every
// prefix on the way; it is not safe for concurrent use.
type memo struct {
	d     *decoder.Decoder
	rows  map[string][]cand
	steps int // decoder positions run
}

func newMemo(d *decoder.Decoder) *memo { return &memo{d: d, rows: map[string][]cand{}} }

// key is the memo key of a prefix and whether the candidates are the
// language step's.
func key(seq []int, lang bool) string {
	b := make([]byte, 0, 2*len(seq)+1)
	for _, t := range seq {
		b = binary.AppendUvarint(b, uint64(t))
	}

	if lang {
		b = append(b, 0xff)
	}

	return string(b)
}

func (m *memo) expand(hs []*hyp, n int, allowed []int, adjust Adjust) ([][]cand, error) {
	if adjust != nil || n > memoTop {
		return nil, fmt.Errorf("search: the memo serves at most %d candidates without Adjust", memoTop)
	}

	out := make([][]cand, len(hs))
	miss := map[*hyp]bool{}

	for i, h := range hs {
		top, ok := m.rows[key(h.seq, allowed != nil)]
		if ok {
			out[i] = top[:min(n, len(top))]
		} else {
			miss[h] = true
		}
	}

	own(hs, func(h *hyp) bool { return miss[h] })

	for i, h := range hs {
		if !miss[h] {
			continue
		}

		logits, err := m.d.Extend(h.st, h.seq[h.fed:])
		if err != nil {
			return nil, err
		}

		v := m.d.Vocab()
		rows := len(logits) / v
		m.steps += rows

		for r := range rows {
			row := logSoftmax(logits[r*v : (r+1)*v])
			prefix := h.seq[:h.fed+r+1]
			m.rows[key(prefix, false)] = candidates(row, nil, memoTop, nil, nil)

			if allowed != nil && r == rows-1 {
				m.rows[key(prefix, true)] = candidates(row, nil, memoTop, allowed, nil)
			}
		}

		h.fed = len(h.seq)
		top := m.rows[key(h.seq, allowed != nil)]
		out[i] = top[:min(n, len(top))]
	}

	return out, nil
}

// outcome is one clip's transcript under one variant.
type outcome struct {
	r        Result
	exact    bool
	wer, cer float64
}

// variantScore is one variant's outcome on all clips.
type variantScore struct {
	opt      Options
	clips    []outcome
	exact    int
	wer, cer float64 // means
	langOK   int
	eos      int
}

// grid is every variant the probe scores: each length normalisation (α
// over alphas for the powered ones) × what L counts × stopping rule ×
// admission rule × language step, at 5 beams; NormNone ignores L.
func grid() []Options {
	alphas := []float64{0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0}

	type norm struct {
		n Norm
		a float64
	}

	norms := make([]norm, 0, 2+2*len(alphas))
	norms = append(norms, norm{NormNone, 0}, norm{NormLen, 1})

	for _, a := range alphas {
		norms = append(norms, norm{NormPow, a}, norm{NormGNMT, a})
	}

	var out []Options

	for _, nm := range norms {
		for _, l := range []Length{LenText, LenTextEOS, LenSeq} {
			if nm.n == NormNone && l != LenText {
				continue
			}

			for _, st := range []Stop{StopAll, StopBest, StopK, StopHF} {
				for _, ad := range []Admit{AdmitTopK, AdmitAboveLive} {
					if st == StopAll && ad != AdmitTopK {
						continue
					}

					for _, lang := range []LangStep{LangArgmax, LangArgmaxScored, LangBeam} {
						out = append(out, Options{
							Beams: 5, Norm: nm.n, Alpha: nm.a, Length: l, Stop: st, Admit: ad, Lang: lang,
						})
					}
				}
			}
		}
	}

	return out
}

// clipIDs lists the corpus clips in manifest order.
func clipIDs(t testing.TB) []string {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(clipDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Clips []struct {
			ID string `json:"id"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	ids := make([]string, len(manifest.Clips))
	for i, c := range manifest.Clips {
		ids[i] = c.ID
	}

	return ids
}

// probeEnv reads an integer knob.
func probeEnv(name string, def int) int {
	v, err := strconv.Atoi(os.Getenv(name))
	if err == nil && v > 0 {
		return v
	}

	return def
}

// scoreGrid decodes every clip with every variant, the clips on all cores
// with one memo each.
func scoreGrid(t *testing.T, d *decoder.Decoder, clips []clip, opts []Options) []variantScore {
	t.Helper()

	res := make([][]outcome, len(opts))
	for i := range res {
		res[i] = make([]outcome, len(clips))
	}

	jobs := make(chan int)
	errs := make([]error, len(clips))
	steps := make([]int, len(clips))

	var wg sync.WaitGroup

	for range runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			for c := range jobs {
				m := newMemo(d)

				for v, opt := range opts {
					r, err := run(d, m, d.NewState(clips[c].Mem), opt)
					if err != nil {
						errs[c] = fmt.Errorf("%s %s: %w", clips[c].ID, opt, err)

						break
					}

					res[v][c] = outcome{
						r: r, exact: r.Text == clips[c].Text,
						wer: eval.WER(clips[c].Text, r.Text), cer: eval.CER(clips[c].Text, r.Text),
					}
				}

				steps[c] = m.steps
			}
		})
	}

	for c := range clips {
		jobs <- c
	}

	close(jobs)
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}

	var total int
	for _, s := range steps {
		total += s
	}

	t.Logf("grid: %d variants × %d clips, %d decoder positions in the memos", len(opts), len(clips), total)

	out := make([]variantScore, len(opts))

	for v, opt := range opts {
		vs := variantScore{opt: opt, clips: res[v]}

		for c, o := range res[v] {
			if o.exact {
				vs.exact++
			}

			if o.r.Language == clips[c].Language {
				vs.langOK++
			}

			if o.r.EOS {
				vs.eos++
			}

			vs.wer += o.wer / float64(len(clips))
			vs.cer += o.cer / float64(len(clips))
		}

		out[v] = vs
	}

	return out
}

// rankVariants orders by exact matches, then mean CER, then mean WER, then
// the grid order.
func rankVariants(vs []variantScore) {
	slices.SortStableFunc(vs, func(a, b variantScore) int {
		return cmp.Or(cmp.Compare(b.exact, a.exact), cmp.Compare(a.cer, b.cer), cmp.Compare(a.wer, b.wer))
	})
}

// TestProbeBeam scores the grid, prints the leaderboard and the axis
// marginals, then reports the leader (or WHISTLE_PROBE_KEY) in detail.
func TestProbeBeam(t *testing.T) {
	d, clips := loadSetup(t, clipIDs(t)...)

	start := time.Now()
	vs := scoreGrid(t, d, clips, grid())
	t.Logf("grid scored in %s", time.Since(start).Round(time.Millisecond))

	rankVariants(vs)

	top := probeEnv("WHISTLE_PROBE_TOP", 25)

	t.Logf("── leaderboard (%d variants; exact of %d, mean WER/CER against the engine)", len(vs), len(clips))
	t.Logf("| # | variant | exact | WER | CER | language | EOS |")
	t.Logf("| - | ------- | ----- | --- | --- | -------- | --- |")

	for i, v := range vs[:min(top, len(vs))] {
		t.Logf("| %d | `%s` | %d/%d | %.4f | %.4f | %d | %d |", i+1, v.opt, v.exact, len(clips), v.wer, v.cer,
			v.langOK, v.eos)
	}

	ties := 0

	for _, v := range vs {
		if v.exact == vs[0].exact && v.cer == vs[0].cer && v.wer == vs[0].wer {
			ties++
		}
	}

	t.Logf("%d variants tie with the leader; they give the leader's transcript on %d/%d clips", ties,
		sameTranscripts(vs[:ties]), len(clips))
	marginals(t, vs, len(clips))

	pick := vs[0]

	if k := os.Getenv("WHISTLE_PROBE_KEY"); k != "" {
		i := slices.IndexFunc(vs, func(v variantScore) bool { return v.opt.String() == k })
		if i < 0 {
			t.Fatalf("WHISTLE_PROBE_KEY %q is not in the grid", k)
		}

		pick = vs[i]
	}

	widths(t, d, clips, pick.opt)
	report(t, d, clips, pick)
}

// sameTranscripts counts the clips on which every variant of vs gives the
// first one's transcript.
func sameTranscripts(vs []variantScore) int {
	n := 0

	for c := range vs[0].clips {
		same := true
		for _, v := range vs[1:] {
			same = same && v.clips[c].r.Text == vs[0].clips[c].r.Text && v.clips[c].r.Language == vs[0].clips[c].r.Language
		}

		if same {
			n++
		}
	}

	return n
}

// widths scores opt at other beam widths.
func widths(t *testing.T, d *decoder.Decoder, clips []clip, opt Options) {
	t.Helper()

	ks := []int{1, 2, 3, 4, 5, 6, 8, 10}
	opts := make([]Options, 0, len(ks))

	for _, k := range ks {
		o := opt
		o.Beams = k
		opts = append(opts, o)
	}

	t.Logf("── beam width, otherwise %s", opt)

	for _, v := range scoreGrid(t, d, clips, opts) {
		t.Logf("beams %2d: exact %d/%d, WER %.4f, CER %.4f, EOS %d", v.opt.Beams, v.exact, len(clips), v.wer, v.cer,
			v.eos)
	}
}

// marginals prints, per axis value, the best and the mean exact count over
// the variants with that value.
func marginals(t *testing.T, vs []variantScore, n int) {
	t.Helper()

	axes := []struct {
		name string
		val  func(Options) string
	}{
		{"norm", func(o Options) string {
			if o.Norm == NormPow || o.Norm == NormGNMT {
				return fmt.Sprintf("%s α=%g", name(normNames, int(o.Norm)), o.Alpha)
			}

			return name(normNames, int(o.Norm))
		}},
		{"len", func(o Options) string { return name(lenNames, int(o.Length)) }},
		{"stop", func(o Options) string { return name(stopNames, int(o.Stop)) }},
		{"admit", func(o Options) string {
			if o.Stop == StopAll {
				return "-"
			}

			return name(admitNames, int(o.Admit))
		}},
		{"lang", func(o Options) string { return name(langNames, int(o.Lang)) }},
	}

	for _, ax := range axes {
		type agg struct {
			best, sum, cnt int
			bestCER        float64
		}

		m := map[string]*agg{}

		var order []string

		for _, v := range vs {
			k := ax.val(v.opt)

			a, ok := m[k]
			if !ok {
				a = &agg{bestCER: math.Inf(1)}
				m[k] = a
				order = append(order, k)
			}

			if v.exact > a.best || (v.exact == a.best && v.cer < a.bestCER) {
				a.best, a.bestCER = v.exact, v.cer
			}

			a.sum += v.exact
			a.cnt++
		}

		parts := make([]string, 0, len(order))
		for _, k := range order {
			a := m[k]
			parts = append(parts, fmt.Sprintf("%s: best %d/%d (CER %.4f), mean %.1f", k, a.best, n, a.bestCER,
				float64(a.sum)/float64(a.cnt)))
		}

		t.Logf("axis %s — %s", ax.name, strings.Join(parts, "; "))
	}
}

// report prints the per-clip table of one variant against the engine and
// greedy decoding, explains its mismatches, and checks the real search
// against the memo and times it.
func report(t *testing.T, d *decoder.Decoder, clips []clip, v variantScore) {
	t.Helper()

	greedy := make([]decoder.Result, len(clips))
	beam := make([]Result, len(clips))

	var greedyDur, beamDur time.Duration

	for i, c := range clips {
		start := time.Now()

		g, err := d.GreedyMemory(c.Mem, "")
		if err != nil {
			t.Fatal(err)
		}

		greedyDur += time.Since(start)
		start = time.Now()

		b, err := Beam(d, c.Mem, v.opt)
		if err != nil {
			t.Fatal(err)
		}

		beamDur += time.Since(start)
		greedy[i], beam[i] = g, b
	}

	var same, gExact int

	t.Logf("── %s", v.opt)
	t.Logf("| clip | language | beam exact | beam WER | beam CER | greedy exact | greedy WER | greedy CER |")
	t.Logf("| ---- | -------- | ---------- | -------- | -------- | ------------ | ---------- | ---------- |")

	for i, c := range clips {
		b, g := beam[i], greedy[i]
		if b.Text == v.clips[i].r.Text && b.Language == v.clips[i].r.Language {
			same++
		}

		if g.Text == c.Text {
			gExact++
		}

		t.Logf("| %s | %s/%s | %v | %.3f | %.3f | %v | %.3f | %.3f |", c.ID, c.Language, b.Language, b.Text == c.Text,
			eval.WER(c.Text, b.Text), eval.CER(c.Text, b.Text), g.Text == c.Text, eval.WER(c.Text, g.Text),
			eval.CER(c.Text, g.Text))
	}

	t.Logf("beam (real search) %d/%d exact, greedy %d/%d; the real search equals the memo on %d/%d clips",
		countExact(clips, beam), len(clips), gExact, len(clips), same, len(clips))
	t.Logf("speed: beam %.0f ms per clip, greedy %.0f ms per clip (serial over clips, beams on all cores)",
		float64(beamDur.Milliseconds())/float64(len(clips)), float64(greedyDur.Milliseconds())/float64(len(clips)))

	for i, c := range clips {
		if beam[i].Text != c.Text {
			explain(t, d, c, beam[i], greedy[i], v.opt)
		}
	}
}

func countExact(clips []clip, rs []Result) int {
	n := 0

	for i, c := range clips {
		if rs[i].Text == c.Text {
			n++
		}
	}

	return n
}

// forced is a transcript teacher-forced under our model.
type forced struct {
	toks  []int     // text tokens
	lps   []float64 // text tokens, then EOS
	sum   float64
	score float64 // the variant's ranking score
}

// force teacher-forces the text tokens toks in language lang.
func force(t *testing.T, d *decoder.Decoder, c clip, lang string, toks []int, opt Options) forced {
	t.Helper()

	var langID int

	for _, l := range d.Languages() {
		if l.Code == lang {
			langID = l.ID
		}
	}

	seq := append(append(d.Prompt(), langID), toks...)
	seq = append(seq, d.Tokenizer().EOSID())

	f, err := d.TeacherForce(c.Mem, seq)
	if err != nil {
		t.Fatal(err)
	}

	h := &hyp{seq: seq[:len(seq)-1], text: len(d.Prompt()) + 1, eos: true}
	fo := forced{toks: toks, lps: f.LogProbs[1:]}

	for _, lp := range fo.lps {
		fo.sum += lp
	}

	h.adj = fo.sum
	if opt.Lang != LangArgmax {
		h.adj += f.LogProbs[0]
	}

	s := &beam{opt: opt}
	fo.score = s.score(h)

	return fo
}

// explain prints why the engine's transcript and ours differ: both
// teacher-forced (summed log p, length, ranking score), and the first
// token where they part with both tokens' log p there.
func explain(t *testing.T, d *decoder.Decoder, c clip, b Result, g decoder.Result, opt Options) {
	t.Helper()

	ref := d.Encode(c.Text)
	best := bestTokens(t, d, c, 16)
	eng := force(t, d, c, c.Language, best, opt)
	refF := force(t, d, c, c.Language, ref, opt)
	ours := force(t, d, c, b.Language, b.Tokens, opt)

	div := 0
	for div < len(eng.toks) && div < len(ours.toks) && eng.toks[div] == ours.toks[div] {
		div++
	}

	piece := func(toks []int, i int) string {
		if i >= len(toks) {
			return "</s>"
		}

		return d.Tokenizer().Piece(toks[i])
	}

	outside, worst, worstAt := ranks(t, d, c, eng.toks, opt.Beams)
	drop := trace(t, d, c, best, opt)

	verdict := "model gap: the engine's transcript ranks lower under our model"
	if eng.score > ours.score {
		verdict = "search miss: the engine's transcript ranks higher under our model but left our beam"
	}

	t.Logf("%s (%s)\n    engine: %s\n    beam:   %s\n    greedy: %s\n"+
		"    engine: Σlogp %.4f over %d tokens with EOS, score %.4f (its most likely spelling)\n"+
		"    beam:   Σlogp %.4f over %d tokens with EOS, score %.4f\n"+
		"    Δscore (beam − engine) %.4f, ΔΣlogp %.4f; part at text token %d: engine %q logp %.4f, beam %q logp %.4f "+
		"(Δ %.4f)\n    the engine's tokens rank outside our top %d at %d of %d steps; worst rank %d (%q at token %d)"+
		"\n    engine text by RefTokenizer: Σlogp %.4f over %d tokens (the best spelling above is %d tokens)\n    %s",
		c.ID, verdict, c.Text, b.Text, g.Text, eng.sum, len(eng.lps), eng.score, ours.sum, len(ours.lps), ours.score,
		ours.score-eng.score, ours.sum-eng.sum, div, piece(eng.toks, div), at(eng.lps, div), piece(ours.toks, div),
		at(ours.lps, div), at(ours.lps, div)-at(eng.lps, div), opt.Beams, outside, len(eng.lps), worst,
		piece(eng.toks, worstAt), worstAt, refF.sum, len(refF.lps), len(eng.lps), drop)
}

// trace reruns the search under opt step by step and reports where the
// engine's tokens (its best spelling) leave the beam: the step, the
// engine prefix's summed log p there and the last kept live hypothesis'.
func trace(t *testing.T, d *decoder.Decoder, c clip, toks []int, opt Options) string {
	t.Helper()

	m := newMemo(d)

	alive, err := languageStep(d, m, d.NewState(c.Mem), opt)
	if err != nil {
		t.Fatal(err)
	}

	s := &beam{opt: opt, eos: d.Tokenizer().EOSID()}
	first := len(d.Prompt()) + 1

	for step := range toks {
		want := toks[:step+1]

		cs, err := m.expand(alive, memoTop, nil, nil)
		if err != nil {
			t.Fatal(err)
		}

		// The engine prefix's candidate, if its parent is alive.
		eng := math.NaN()

		for i, h := range alive {
			if !slices.Equal(h.seq[first:], toks[:step]) {
				continue
			}

			for _, cd := range cs[i] {
				if cd.tok == toks[step] {
					eng = h.adj + cd.lp
				}
			}
		}

		next, done, err := s.step(m, alive)
		if err != nil {
			t.Fatal(err)
		}

		kept := slices.ContainsFunc(next, func(h *hyp) bool { return slices.Equal(h.seq[first:], want) })
		if !kept {
			last := next[len(next)-1]

			return fmt.Sprintf("the engine's prefix leaves the beam at text token %d (%q): Σlogp %.4f, the last kept "+
				"of %d live hypotheses %.4f (margin %.4f nats)", step, d.Tokenizer().Piece(toks[step]), eng, len(next),
				last.adj, last.adj-eng)
		}

		if done {
			return fmt.Sprintf("the search ends at text token %d with the engine's prefix still alive", step)
		}

		alive = next
	}

	return "the engine's tokens stay in the beam to the end"
}

// tokenBytes is the bytes every piece spells: NORMAL pieces their surface
// with "▁" as a space, BYTE pieces their byte, every other piece nil.
func tokenBytes(d *decoder.Decoder) [][]byte {
	tk := d.Tokenizer()
	out := make([][]byte, tk.Len())

	for id := range out {
		switch tk.Type(id) {
		case tokenizer.TypeNormal:
			out[id] = []byte(strings.ReplaceAll(tk.Piece(id), "▁", " "))
		case tokenizer.TypeByte:
			v, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(tk.Piece(id), "<0x"), ">"), 16, 8)
			if err == nil {
				out[id] = []byte{byte(v)}
			}
		}
	}

	return out
}

// bestTokens is the most likely spelling of the engine's text under our
// model: a beam search of width w over the token sequences whose bytes
// spell c.Text exactly, ranked by summed log p with EOS, so a mismatch is
// not blamed on RefTokenizer picking a spelling the engine did not.
func bestTokens(t *testing.T, d *decoder.Decoder, c clip, w int) []int {
	t.Helper()

	type path struct {
		st   *decoder.State
		toks []int
		off  int
		sum  float64
	}

	var langID int

	for _, l := range d.Languages() {
		if l.Code == c.Language {
			langID = l.ID
		}
	}

	target, spell := []byte(c.Text), tokenBytes(d)
	root := d.NewState(c.Mem)

	logits, err := d.Extend(root, append(d.Prompt(), langID))
	if err != nil {
		t.Fatal(err)
	}

	v := d.Vocab()
	rows := map[*path][]float64{}
	live := []*path{{st: root}}
	rows[live[0]] = logSoftmax(logits[len(logits)-v:])

	var best *path

	bestSum := math.Inf(-1)

	for len(live) > 0 {
		var next []*path

		for _, p := range live {
			lp := rows[p]

			if p.off == len(target) {
				if s := p.sum + lp[d.Tokenizer().EOSID()]; s > bestSum {
					best, bestSum = p, s
				}

				continue
			}

			for id, b := range spell {
				if len(b) > 0 && len(b) <= len(target)-p.off && string(target[p.off:p.off+len(b)]) == string(b) {
					next = append(next, &path{
						st: p.st, toks: append(slices.Clone(p.toks), id), off: p.off + len(b),
						sum: p.sum + lp[id],
					})
				}
			}
		}

		slices.SortStableFunc(next, func(a, b *path) int { return cmp.Compare(b.sum, a.sum) })
		live = next[:min(w, len(next))]

		if len(live) == 0 || live[0].sum <= bestSum {
			break
		}

		for _, p := range live {
			p.st = p.st.Fork()

			l, err := d.Extend(p.st, p.toks[len(p.toks)-1:])
			if err != nil {
				t.Fatal(err)
			}

			rows[p] = logSoftmax(l)
		}
	}

	return best.toks
}

// ranks teacher-forces the engine's text tokens toks (then EOS) and
// returns how many rank outside our top k, the worst rank (0 = top) and
// its token index.
func ranks(t *testing.T, d *decoder.Decoder, c clip, toks []int, k int) (outside, worst, worstAt int) {
	t.Helper()

	var langID int

	for _, l := range d.Languages() {
		if l.Code == c.Language {
			langID = l.ID
		}
	}

	seq := append(append(d.Prompt(), langID), toks...)

	logits, err := d.Extend(d.NewState(c.Mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	v, first := d.Vocab(), len(d.Prompt())

	for i := range len(toks) + 1 {
		row := logits[(first+i)*v : (first+i+1)*v]

		want := d.Tokenizer().EOSID()
		if i < len(toks) {
			want = toks[i]
		}

		r := 0

		for _, x := range row {
			if x > row[want] {
				r++
			}
		}

		if r >= k {
			outside++
		}

		if r > worst {
			worst, worstAt = r, i
		}
	}

	return outside, worst, worstAt
}

func at(v []float64, i int) float64 {
	if i < len(v) {
		return v[i]
	}

	return math.NaN()
}
