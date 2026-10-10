//go:build oracle

// The keyword biasing probe (PLAN.md 6.3, docs/FINDINGS.md F-029). The
// engine biases its beam search toward the keywords of a file through an
// Aho-Corasick automaton (U9); testdata/oracle/keywords holds its text
// with and without the file for 53 cases over 19 clips (F-025).
// TestProbeKeywords decodes every case with Default plus the Adjust of
// Keywords under every Bias of a grid (weight, per token or per character,
// revocation, completion bonus, spellings, phrase splitting) and ranks the
// variants by exact equality with the engine's text with the keywords,
// then by mean CER and WER; each search rule also runs without keywords,
// against the engine's text without them. The search runs on a memo per clip that keeps
// each prefix's memoTop best next tokens by the model and the model's log
// p of every token any variant's automaton holds: a token outside the
// automaton gets the same bonus change as every other such token, so the
// best candidates after the bias are among those. The leader is then rerun
// with the real search, its per-kind table printed against the run
// without keywords, and every remaining mismatch teacher-forced both ways
// under the leader's bias. Run it with `just keywords-probe`; the knobs
// are environment variables:
//
//	WHISTLE_PROBE_GRID   1 (wide, about 10 min), 2 (renormalised bias, final
//	                     pick by the model, about 8 min) or 3 (search rules,
//	                     default, under a minute); see kwGrid1 … kwGrid3
//	WHISTLE_PROBE_TOP    leaderboard rows (default 25)
//	WHISTLE_PROBE_KEY    a variant (Bias.String, " rank=model") to report instead of the leader

package search

import (
	"cmp"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/eval"
)

// kwRow is one prefix's memo entry: the model's best next tokens and its
// log p of the extra tokens.
type kwRow struct {
	top   []cand
	extra []float64
}

// kwMemo is memo for biased searches. kw is the automaton of the variant
// being run (nil for none); the search's Adjust must be kw.Adjust.
type kwMemo struct {
	d       *decoder.Decoder
	extras  []int       // every token of every variant's automata, ascending
	extraAt map[int]int // token → index in extras
	rows    map[string]kwRow
	kw      *Keywords
	steps   int
}

func (m *kwMemo) expand(hs []*hyp, n int, allowed []int, adjust Adjust) ([][]cand, error) {
	biased := adjust != nil && allowed == nil
	if n > memoTop || biased != (m.kw != nil && allowed == nil) {
		return nil, fmt.Errorf("search: kwMemo: %d candidates, adjust %v, automaton %v", n, adjust != nil, m.kw != nil)
	}

	miss := map[*hyp]bool{}

	for _, h := range hs {
		if _, ok := m.rows[key(h.seq, allowed != nil)]; !ok {
			miss[h] = true
		}
	}

	own(hs, func(h *hyp) bool { return miss[h] })

	for _, h := range hs {
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
			lp := logSoftmax(logits[r*v : (r+1)*v])
			prefix := h.seq[:h.fed+r+1]
			row := kwRow{top: candidates(lp, nil, memoTop, nil, nil), extra: make([]float64, len(m.extras))}

			for i, id := range m.extras {
				row.extra[i] = lp[id]
			}

			m.rows[key(prefix, false)] = row

			if allowed != nil && r == rows-1 {
				m.rows[key(prefix, true)] = kwRow{top: candidates(lp, nil, memoTop, allowed, nil)}
			}
		}

		h.fed = len(h.seq)
	}

	out := make([][]cand, len(hs))

	for i, h := range hs {
		row := m.rows[key(h.seq, allowed != nil)]
		if !biased {
			out[i] = row.top[:min(n, len(row.top))]

			continue
		}

		var err error

		out[i], err = m.biased(row, h.textTokens(), n)
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}

// biased is candidates on a memo row under m.kw. A token outside the
// automaton gets the same bonus change as every other such token, so only
// the model's best tokens and the automaton's tokens after text can win;
// they are taken in id order, so that ties go to the lower id as in
// candidates.
func (m *kwMemo) biased(row kwRow, text []int, n int) ([]cand, error) {
	lp := func(id int) float64 {
		if i, ok := m.extraAt[id]; ok {
			return row.extra[i]
		}

		return math.Inf(-1) // not an automaton token: unused
	}

	base, special := m.kw.changes(text, lp)
	pool := make([]cand, 0, len(row.top)+len(special))
	plain := 0

	for _, c := range row.top {
		d, ok := special[c.tok]
		if !ok {
			d = base
			plain++
		}

		pool = append(pool, cand{tok: c.tok, lp: c.lp, adj: c.lp + d})
	}

	if plain < n {
		return nil, fmt.Errorf("search: kwMemo: %d of the memo's %d tokens outside the automaton, %d needed", plain,
			len(row.top), n)
	}

	for id, d := range special {
		if slices.ContainsFunc(row.top, func(c cand) bool { return c.tok == id }) {
			continue
		}

		i, ok := m.extraAt[id]
		if !ok {
			return nil, fmt.Errorf("search: kwMemo: automaton token %d is not an extra", id)
		}

		pool = append(pool, cand{tok: id, lp: row.extra[i], adj: row.extra[i] + d})
	}

	slices.SortFunc(pool, func(a, b cand) int { return cmp.Compare(a.tok, b.tok) })

	top := make([]cand, 0, n+1)

	for _, c := range pool {
		if len(top) == n && c.adj <= top[n-1].adj {
			continue
		}

		i := len(top)
		for i > 0 && top[i-1].adj < c.adj {
			i--
		}

		top = slices.Insert(top, i, c)
		top = top[:min(len(top), n)]
	}

	return top, nil
}

// kwOutcome is one case's transcript under one variant.
type kwOutcome struct {
	text     string
	exact    bool // equals the engine's text with the keywords
	without  bool // equals the engine's text without them
	wer, cer float64
}

// kwVariant is one search under one bias: Default with its stopping and
// admission rule, and whether the final pick ignores the bias; none runs
// it without keywords.
type kwVariant struct {
	bias      Bias
	none      bool
	rankModel bool
	stop      Stop
	admit     Admit
}

func (v kwVariant) String() string {
	name := v.bias.String()
	if v.none {
		name = "no keywords"
	}

	if v.rankModel {
		name += " rank=model"
	}

	if v.stop != StopK || v.admit != AdmitAboveLive {
		name += " " + v.search()
	}

	return name
}

// search names the variant's search rules.
func (v kwVariant) search() string {
	return fmt.Sprintf("stop=%s admit=%s", stopNames[v.stop-1], admitNames[v.admit-1])
}

// options is Default with the variant's rules and kw's Adjust (nil for
// none).
func (v kwVariant) options(kw *Keywords) Options {
	o := Default()
	o.Stop, o.Admit, o.rankModel = v.stop, v.admit, v.rankModel

	if kw != nil {
		o.Adjust = kw.Adjust
	}

	return o
}

// kwScore is one variant's outcome on all cases.
type kwScore struct {
	kwVariant

	cases    []kwOutcome
	exact    int
	without  int
	wer, cer float64
}

var kwForms = []Form{
	FormSpace, FormSpace | FormBare, FormSpace | FormBare | FormCapital, FormSpace | FormBare | FormLower,
	FormSpace | FormBare | FormCapital | FormLower,
}

// searches are the stopping and admission rules the probe crosses with
// the bias: Default's (Whisper's) first, then the runners-up of F-026.
var searches = []struct {
	stop  Stop
	admit Admit
}{{StopK, AdmitAboveLive}, {StopK, AdmitTopK}, {StopHF, AdmitAboveLive}, {StopHF, AdmitTopK}, {StopBest, AdmitAboveLive}}

// kwGrid is the variants of grid WHISTLE_PROBE_GRID (default 3), each
// search rule without keywords first.
func kwGrid() []kwVariant {
	var out []kwVariant

	for _, sr := range searches {
		out = append(out, kwVariant{none: true, stop: sr.stop, admit: sr.admit})
	}

	switch probeEnv("WHISTLE_PROBE_GRID", 3) {
	case 1:
		return append(out, kwGrid1()...)
	case 2:
		return append(out, kwGrid2()...)
	default:
		return append(out, kwGrid3()...)
	}
}

// kwGrid1 is the wide grid on Default (2550 variants): weight 0.25 … 5
// per token, or a quarter of it per character, × revocation 0, 0.5, 1 ×
// completion bonus 0, 1, 2 × 5 form sets × phrase splitting, and the
// completion bonus alone.
func kwGrid1() []kwVariant {
	weights := []float64{0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 5}
	completions := []float64{1, 2, 3, 4, 6, 8}
	out := make([]kwVariant, 0, 2*len(weights)*3*3*len(kwForms)*2+len(completions)*len(kwForms))

	for _, per := range []bool{false, true} {
		for _, w := range weights {
			if per {
				w /= 4
			}

			for _, rv := range []float64{0, 0.5, 1} {
				for _, cp := range []float64{0, 1, 2} {
					for _, f := range kwForms {
						for _, split := range []bool{false, true} {
							out = append(out, kwVariant{
								bias: Bias{Weight: w, PerChar: per, Revoke: rv, Complete: cp, Forms: f, Split: split},
								stop: StopK, admit: AdmitAboveLive,
							})
						}
					}
				}
			}
		}
	}

	for _, cp := range completions {
		for _, f := range kwForms {
			out = append(out, kwVariant{bias: Bias{Complete: cp, Revoke: 1, Forms: f}, stop: StopK, admit: AdmitAboveLive})
		}
	}

	return out
}

// kwGrid2 crosses grid 1's strong region (a per-token weight near 2, full
// revocation, every form) with the renormalised bias and the final pick
// by the model, on Default (576 variants).
func kwGrid2() []kwVariant {
	weights := []float64{1, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 5, 6, 8}
	out := make([]kwVariant, 0, len(weights)*2*2*len(kwForms[2:])*2*2)

	for _, w := range weights {
		for _, rv := range []float64{0.5, 1} {
			for _, cp := range []float64{0, 1} {
				for _, f := range kwForms[2:] {
					for _, renorm := range []bool{false, true} {
						for _, rank := range []bool{false, true} {
							out = append(out, kwVariant{
								bias:      Bias{Weight: w, Revoke: rv, Complete: cp, Forms: f, Renorm: renorm},
								rankModel: rank, stop: StopK, admit: AdmitAboveLive,
							})
						}
					}
				}
			}
		}
	}

	return out
}

// kwGrid3 crosses the strong region with the search rules (720 variants).
func kwGrid3() []kwVariant {
	weights := []float64{1.5, 1.75, 2, 2.25, 2.5, 3}
	out := make([]kwVariant, 0, len(searches)*len(weights)*2*2*len(kwForms[2:])*2)

	for _, sr := range searches {
		for _, w := range weights {
			for _, rv := range []float64{0.5, 1} {
				for _, cp := range []float64{0, 1} {
					for _, f := range kwForms[2:] {
						for _, rank := range []bool{false, true} {
							out = append(out, kwVariant{
								bias:      Bias{Weight: w, Revoke: rv, Complete: cp, Forms: f},
								rankModel: rank, stop: sr.stop, admit: sr.admit,
							})
						}
					}
				}
			}
		}
	}

	return out
}

// kwSetup loads the cases and their clips.
func kwSetup(t *testing.T) (*decoder.Decoder, []keywordCase, map[string]clip) {
	t.Helper()

	cases := keywordCases(t)

	var ids []string

	for _, c := range cases {
		if !slices.Contains(ids, c.Clip) {
			ids = append(ids, c.Clip)
		}
	}

	d, clips := loadSetup(t, ids...)
	byID := map[string]clip{}

	for _, c := range clips {
		byID[c.ID] = c
	}

	return d, cases, byID
}

// kwExtras is every token of the automata of the cases on clip id, in all
// forms, split and not.
func kwExtras(t *testing.T, d *decoder.Decoder, cases []keywordCase, id string) []int {
	t.Helper()

	var extras []int

	for _, c := range cases {
		if c.Clip != id {
			continue
		}

		for _, split := range []bool{false, true} {
			kw, err := NewKeywords(d.Tokenizer(), c.Keywords, Bias{Weight: 1, Forms: kwForms[len(kwForms)-1], Split: split})
			if err != nil {
				t.Error(err)

				return nil
			}

			for _, nx := range kw.next {
				for tok := range nx {
					if !slices.Contains(extras, tok) {
						extras = append(extras, tok)
					}
				}
			}
		}
	}

	slices.Sort(extras)

	return extras
}

// scoreKeywords decodes every case under every bias, and once without
// keywords (the first row), the clips on all cores with one memo each.
func scoreKeywords(t *testing.T, d *decoder.Decoder, cases []keywordCase, clips map[string]clip,
	variants []kwVariant,
) []kwScore {
	t.Helper()

	res := make([][]kwOutcome, len(variants))
	for i := range res {
		res[i] = make([]kwOutcome, len(cases))
	}

	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		errs  []error
		steps int
	)

	jobs := make(chan string)

	for range runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			for id := range jobs {
				m := &kwMemo{d: d, extras: kwExtras(t, d, cases, id), extraAt: map[int]int{}, rows: map[string]kwRow{}}
				for i, tok := range m.extras {
					m.extraAt[tok] = i
				}

				err := scoreClip(d, m, cases, clips[id], variants, res)

				mu.Lock()

				steps += m.steps

				if err != nil {
					errs = append(errs, err)
				}

				mu.Unlock()
			}
		})
	}

	ids := make([]string, 0, len(clips))
	for id := range clips {
		ids = append(ids, id)
	}

	slices.Sort(ids)

	for _, id := range ids {
		jobs <- id
	}

	close(jobs)
	wg.Wait()

	if len(errs) > 0 {
		t.Fatal(errs[0])
	}

	t.Logf("grid: %d variants × %d cases, %d decoder positions in the memos", len(variants), len(cases), steps)

	out := make([]kwScore, len(res))

	for v := range res {
		s := kwScore{kwVariant: variants[v], cases: res[v]}

		for _, o := range res[v] {
			if o.exact {
				s.exact++
			}

			if o.without {
				s.without++
			}

			s.wer += o.wer / float64(len(cases))
			s.cer += o.cer / float64(len(cases))
		}

		out[v] = s
	}

	return out
}

// scoreClip runs every case on clip c under every variant into
// res[v][case].
func scoreClip(d *decoder.Decoder, m *kwMemo, cases []keywordCase, c clip, variants []kwVariant,
	res [][]kwOutcome,
) error {
	plain := map[int]string{}

	for v, vr := range variants {
		if !vr.none {
			continue
		}

		m.kw = nil

		r, err := run(d, m, d.NewState(c.Mem), vr.options(nil))
		if err != nil {
			return fmt.Errorf("%s %s: %w", c.ID, vr, err)
		}

		plain[v] = r.Text
	}

	for i, kc := range cases {
		if kc.Clip != c.ID {
			continue
		}

		for v, vr := range variants {
			if vr.none {
				res[v][i] = outcomeOf(kc, plain[v])

				continue
			}

			kw, err := NewKeywords(d.Tokenizer(), kc.Keywords, vr.bias)
			if err != nil {
				return err
			}

			m.kw = kw

			r, err := run(d, m, d.NewState(c.Mem), vr.options(kw))
			if err != nil {
				return fmt.Errorf("%s %s: %w", kc.Name, vr, err)
			}

			res[v][i] = outcomeOf(kc, r.Text)
		}
	}

	return nil
}

func outcomeOf(c keywordCase, got string) kwOutcome {
	return kwOutcome{
		text: got, exact: got == c.TextWith, without: got == c.TextWithout,
		wer: eval.WER(c.TextWith, got), cer: eval.CER(c.TextWith, got),
	}
}

func TestProbeKeywords(t *testing.T) {
	d, cases, clips := kwSetup(t)

	start := time.Now()
	all := scoreKeywords(t, d, cases, clips, kwGrid())
	t.Logf("grid scored in %s", time.Since(start).Round(time.Millisecond))

	var vs, none []kwScore

	for _, v := range all {
		if v.none {
			none = append(none, v)
		} else {
			vs = append(vs, v)
		}
	}

	slices.SortStableFunc(vs, func(a, b kwScore) int {
		return cmp.Or(cmp.Compare(b.exact, a.exact), cmp.Compare(a.cer, b.cer), cmp.Compare(a.wer, b.wer))
	})

	top := probeEnv("WHISTLE_PROBE_TOP", 25)

	t.Logf("── leaderboard (%d variants; exact of %d against the engine's text with the keywords, mean WER/CER)",
		len(vs), len(cases))
	t.Logf("| # | variant | exact | WER | CER |")
	t.Logf("| - | ------- | ----- | --- | --- |")

	for i, v := range vs[:min(top, len(vs))] {
		t.Logf("| %d | `%s` | %d/%d | %.4f | %.4f |", i+1, v, v.exact, len(cases), v.wer, v.cer)
	}

	t.Logf("── per search rule: without keywords against the engine's text without them, and its best bias")
	t.Logf("| search | without: exact | best bias | with: exact |")
	t.Logf("| ------ | -------------- | --------- | ----------- |")

	for _, n := range none {
		i := slices.IndexFunc(vs, func(v kwScore) bool { return v.search() == n.search() })
		if i < 0 { // not in this grid
			t.Logf("| %s | %d/%d | - | - |", n.search(), n.without, len(cases))

			continue
		}

		t.Logf("| %s | %d/%d | `%s` | %d/%d |", n.search(), n.without, len(cases), vs[i].kwVariant, vs[i].exact,
			len(cases))
	}

	ties := 0

	for _, v := range vs {
		if v.exact == vs[0].exact && v.cer == vs[0].cer && v.wer == vs[0].wer {
			ties++
		}
	}

	t.Logf("%d variants tie with the leader", ties)
	kwMarginals(t, vs, len(cases))

	pick := vs[0]

	if k := os.Getenv("WHISTLE_PROBE_KEY"); k != "" {
		i := slices.IndexFunc(vs, func(v kwScore) bool { return v.String() == k })
		if i < 0 {
			t.Fatalf("WHISTLE_PROBE_KEY %q is not in the grid", k)
		}

		pick = vs[i]
	}

	kwReport(t, d, cases, clips, pick)
}

// kwMarginals prints, per axis value, the best and the mean exact count.
func kwMarginals(t *testing.T, vs []kwScore, n int) {
	t.Helper()

	axes := []struct {
		name string
		val  func(kwVariant) string
	}{
		{"weight", func(v kwVariant) string {
			if v.bias.PerChar {
				return fmt.Sprintf("%g/char", v.bias.Weight)
			}

			return fmt.Sprintf("%g", v.bias.Weight)
		}},
		{"revoke", func(v kwVariant) string { return fmt.Sprintf("%g", v.bias.Revoke) }},
		{"complete", func(v kwVariant) string { return fmt.Sprintf("%g", v.bias.Complete) }},
		{"forms", func(v kwVariant) string { return v.bias.Forms.String() }},
		{"split", func(v kwVariant) string { return strconv.FormatBool(v.bias.Split) }},
		{"renorm", func(v kwVariant) string { return strconv.FormatBool(v.bias.Renorm) }},
		{"rank=model", func(v kwVariant) string { return strconv.FormatBool(v.rankModel) }},
		{"search", kwVariant.search},
	}

	for _, ax := range axes {
		best, sum, cnt := map[string]int{}, map[string]int{}, map[string]int{}

		var order []string

		for _, v := range vs {
			k := ax.val(v.kwVariant)
			if _, ok := cnt[k]; !ok {
				order = append(order, k)
			}

			best[k] = max(best[k], v.exact)
			sum[k] += v.exact
			cnt[k]++
		}

		parts := make([]string, 0, len(order))
		for _, k := range order {
			parts = append(parts, fmt.Sprintf("%s: best %d/%d, mean %.1f", k, best[k], n,
				float64(sum[k])/float64(cnt[k])))
		}

		t.Logf("axis %s — %s", ax.name, strings.Join(parts, "; "))
	}
}

// kwReport reruns v with the real search, prints the per-kind table
// against the runs without keywords and explains every mismatch.
func kwReport(t *testing.T, d *decoder.Decoder, cases []keywordCase, clips map[string]clip, v kwScore) {
	t.Helper()

	type kindRow struct{ n, with, plainWithout, changed int }

	order := []string{"present", "missed", "distractor", "phrase", "list"}
	kinds := map[string]*kindRow{}

	for _, k := range order {
		kinds[k] = &kindRow{}
	}

	var (
		same, with, plainWithout int
		dur                      time.Duration
	)

	plain := map[string]Result{}

	for i, c := range cases {
		kw, err := NewKeywords(d.Tokenizer(), c.Keywords, v.bias)
		if err != nil {
			t.Fatal(err)
		}

		start := time.Now()

		r, err := Beam(d, clips[c.Clip].Mem, v.options(kw))
		if err != nil {
			t.Fatal(err)
		}

		dur += time.Since(start)

		p, ok := plain[c.Clip]
		if !ok {
			p, err = Beam(d, clips[c.Clip].Mem, v.options(nil))
			if err != nil {
				t.Fatal(err)
			}

			plain[c.Clip] = p
		}

		k := kinds[c.Kind]
		k.n++

		if r.Text == v.cases[i].text {
			same++
		}

		if r.Text == c.TextWith {
			with++
			k.with++
		}

		if r.Text != p.Text {
			k.changed++
		}

		if p.Text == c.TextWithout {
			plainWithout++
			k.plainWithout++
		}

		if r.Text != c.TextWith {
			kwExplain(t, d, c, clips[c.Clip], v.kwVariant, kw, r, p)
		}
	}

	t.Logf("── %s (the real search equals the memo on %d/%d cases, %.0f ms per case)", v, same, len(cases),
		float64(dur.Milliseconds())/float64(len(cases)))
	t.Logf("with keywords = the engine's text with them: %d/%d; without keywords = the engine's text without: %d/%d",
		with, len(cases), plainWithout, len(cases))
	t.Logf("| kind | cases | with: exact with the engine | without: exact with the engine | our text changed |")
	t.Logf("| ---- | ----- | --------------------------- | ------------------------------ | ---------------- |")

	for _, k := range order {
		r := kinds[k]
		t.Logf("| %s | %d | %d | %d | %d |", k, r.n, r.with, r.plainWithout, r.changed)
	}
}

// kwExplain teacher-forces the engine's text with the keywords (its most
// likely spelling under the model, or RefTokenizer's, whichever ranks
// higher) and ours, and prints which ranks higher by the variant's final
// score: a search miss when the engine's does, a gap of the model and the
// bias when ours does.
func kwExplain(t *testing.T, d *decoder.Decoder, c keywordCase, cl clip, v kwVariant, kw *Keywords, r, plain Result) {
	t.Helper()

	target := cl
	target.Language, target.Text = c.Language, c.TextWith

	score := func(lang string, toks []int) float64 {
		sum := adjustedSum(t, d, cl, lang, toks, kw, v.rankModel)

		return sum / float64(max(len(toks), 1))
	}

	eng := math.Max(score(c.Language, bestTokens(t, d, target, 16)), score(c.Language, d.Encode(c.TextWith)))
	ours := score(r.Language, r.Tokens)

	verdict := "model/bias gap: the engine's text ranks lower"
	if eng > ours {
		verdict = "search miss: the engine's text ranks higher"
	}

	inherited := ""
	if plain.Text != c.TextWithout {
		inherited = " (without keywords we already differ from the engine)"
	}

	t.Logf("%s [%s %q] %s%s: score ours %.4f, engine %.4f (Δ %.4f)\n    engine with:    %s\n    ours with:      %s"+
		"\n    engine without: %s", c.Name, c.Kind, c.Keywords, verdict, inherited, ours, eng, ours-eng, c.TextWith, r.Text,
		c.TextWithout)
}

// adjustedSum is the summed log p of the text tokens toks and EOS in
// language lang, each after kw.Adjust (the model's own with model).
func adjustedSum(t *testing.T, d *decoder.Decoder, c clip, lang string, toks []int, kw *Keywords, model bool) float64 {
	t.Helper()

	var langID int

	for _, l := range d.Languages() {
		if l.Code == lang {
			langID = l.ID
		}
	}

	seq := append(append(d.Prompt(), langID), toks...)

	logits, err := d.Extend(d.NewState(c.Mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	v, first := d.Vocab(), len(d.Prompt())

	var sum float64

	for i := range len(toks) + 1 {
		row := logSoftmax(logits[(first+i)*v : (first+i+1)*v])
		if !model {
			kw.Adjust(toks[:i], row)
		}

		want := d.Tokenizer().EOSID()
		if i < len(toks) {
			want = toks[i]
		}

		sum += row[want]
	}

	return sum
}
