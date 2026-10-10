//go:build oracle

// The timing leaderboards of the alignment probe (probe_test.go,
// docs/FINDINGS.md F-033): the grid, the head sets, the depth runs and the
// remaining misses.

package align

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
)

// timeScore is one timing candidate's agreement with the engine's words on
// the frame grid.
type timeScore struct {
	words, start1, end1, both1, startExact, endExact, fails int
	absStart, absEnd                                        float64 // frames
}

func (s *timeScore) add(o *timeScore) {
	s.words += o.words
	s.start1 += o.start1
	s.end1 += o.end1
	s.both1 += o.both1
	s.startExact += o.startExact
	s.endExact += o.endExact
	s.fails += o.fails
	s.absStart += o.absStart
	s.absEnd += o.absEnd
}

// share is n of the words in percent.
func (s *timeScore) share(n int) float64 { return 100 * float64(n) / float64(max(1, s.words)) }

// better ranks by both within 80 ms, then exact frames, then mean |Δ|.
func (s *timeScore) better(o *timeScore) bool {
	a, b := s.both1*o.words, o.both1*s.words
	if a != b {
		return a > b
	}

	a, b = (s.startExact+s.endExact)*o.words, (o.startExact+o.endExact)*s.words
	if a != b {
		return a > b
	}

	return (s.absStart+s.absEnd)*float64(o.words) < (o.absStart+o.absEnd)*float64(s.words)
}

// frameOf is an engine time on the 80 ms grid.
func frameOf(sec float64) int { return int(math.Round(sec / 0.08)) }

// score adds one item's spans (nil: the alignment failed) to s.
func (s *timeScore) score(it *item, spans []Span) {
	s.words += len(it.Words)
	if spans == nil {
		s.fails += len(it.Words)

		return
	}

	for i, w := range it.Words {
		ds, de := spans[i].Start-frameOf(w.Start), spans[i].End-frameOf(w.End)
		s.absStart += math.Abs(float64(ds))
		s.absEnd += math.Abs(float64(de))

		if ds == 0 {
			s.startExact++
		}

		if de == 0 {
			s.endExact++
		}

		if ds >= -1 && ds <= 1 {
			s.start1++
		}

		if de >= -1 && de <= 1 {
			s.end1++
		}

		if ds >= -1 && ds <= 1 && de >= -1 && de <= 1 {
			s.both1++
		}
	}
}

func (s *timeScore) String() string {
	return fmt.Sprintf("%d | %.1f %% | %.1f %% | %.1f %% | %.1f %% / %.1f %% | %.2f / %.2f", s.words,
		s.share(s.both1), s.share(s.start1), s.share(s.end1), s.share(s.startExact), s.share(s.endExact),
		s.absStart/float64(max(1, s.words)), s.absEnd/float64(max(1, s.words)))
}

const timeHeader = "| words | start and end within 80 ms | start within 80 ms | end within 80 ms | exact start / end " +
	"| mean abs Δ start / end (frames) |"

// input is an item's alignment input.
func input(it *item) Input {
	return Input{Attn: it.Attn, First: it.first(), Tokens: len(it.Tokens), Frames: it.Frames}
}

// matConfig is the part of Options that shapes the per-head matrices.
type matConfig struct {
	Gate   bool
	Rows   Rows
	Lead   int
	Norm   Norm
	Filter int
}

// headSet is a named head selection.
type headSet struct {
	name  string
	heads []Head
}

// headSets are the grid's head selections: all, each layer, the halves.
func headSets() []headSet {
	layer := func(ls ...int) []Head {
		var out []Head

		for _, l := range ls {
			for h := range 8 {
				out = append(out, Head{Layer: l, Head: h})
			}
		}

		return out
	}

	out := []headSet{
		{"all", layer(0, 1, 2, 3, 4, 5, 6, 7)},
		{"layers 0-3", layer(0, 1, 2, 3)},
		{"layers 4-7 (the last half)", layer(4, 5, 6, 7)},
		{"layers 6-7", layer(6, 7)},
	}

	for l := range 8 {
		out = append(out, headSet{fmt.Sprintf("layer %d", l), layer(l)})
	}

	return out
}

// headMatrices is one item's per-head matrices under c: heads × rows ×
// frames, normalised and filtered, nil when c's rows do not fit.
func headMatrices(it *item, c matConfig) ([][]float64, int) {
	opt := Options{Gate: c.Gate, Rows: c.Rows, Lead: c.Lead, Norm: c.Norm, Filter: c.Filter}

	shift := 0
	if c.Rows == RowsPredict {
		shift = -1
	}

	top := it.first() + shift - c.Lead
	rows := c.Lead + len(it.Tokens) + 1

	if top < 0 {
		return nil, 0
	}

	out := make([][]float64, 64)

	for l := range 8 {
		for h := range 8 {
			m, err := HeadMatrix(input(it), Head{Layer: l, Head: h}, top, rows, opt)
			if err == nil {
				out[l*8+h] = m
			}
		}
	}

	return out, rows
}

// pathCost is the cost of the mean of heads' matrices; nil when one is
// missing.
func pathCost(hm [][]float64, heads []Head, n int, cost Cost) []float64 {
	sum := make([]float64, n)

	for _, h := range heads {
		m := hm[h.Layer*8+h.Head]
		if m == nil {
			return nil
		}

		for i, v := range m {
			sum[i] += v
		}
	}

	for i, v := range sum {
		a := v / float64(len(heads))
		if cost == CostNegLog {
			sum[i] = -math.Log(math.Max(a, 1e-30))
		} else {
			sum[i] = -a
		}
	}

	return sum
}

// tail is the part of Options applied after the matrix.
type tail struct {
	Cost   Cost
	Step   Step
	End    End
	Core   bool
	Offset int
}

func tails(norm Norm) []tail {
	costs := []Cost{CostNeg}
	if norm == NormNone {
		costs = append(costs, CostNegLog)
	}

	var out []tail

	for _, c := range costs {
		for _, s := range []Step{StepWhisper, StepNoShare} {
			for _, e := range []End{EndNext, EndLast} {
				for _, core := range []bool{true, false} {
					for off := -1; off <= 1; off++ {
						out = append(out, tail{Cost: c, Step: s, End: e, Core: core, Offset: off})
					}
				}
			}
		}
	}

	return out
}

func (c matConfig) options(heads []Head, tl tail) Options {
	return Options{
		Heads: heads, Gate: c.Gate, Rows: c.Rows, Lead: c.Lead, Norm: c.Norm, Filter: c.Filter, Cost: tl.Cost,
		Step: tl.Step, End: tl.End, Core: tl.Core, Offset: tl.Offset,
	}
}

// itemWords is Words of every item, by item.
func itemWords(t *testing.T, ps probeSetup, items []*item) map[*item][]Word {
	t.Helper()

	out := map[*item][]Word{}

	for _, it := range items {
		ws, err := Words(ps.tok, it.Tokens)
		if err != nil {
			t.Fatal(err)
		}

		out[it] = ws
	}

	return out
}

// gridResult is one grid variant's scores per set.
type gridResult struct {
	opt  Options
	name string
	sets map[string]*timeScore
}

// runGrid scores every matrix config × head set × tail on the items, on
// all cores.
func runGrid(items []*item, words map[*item][]Word, mats []matConfig, hsets []headSet) []*gridResult {
	type key struct {
		m  int
		h  int
		tl tail
	}

	var (
		mu  sync.Mutex
		all = map[key]*gridResult{}
		wg  sync.WaitGroup
	)

	jobs := make(chan int)

	for range runtime.NumCPU() {
		wg.Go(func() {
			local := map[key]map[string]*timeScore{}

			for mi := range jobs {
				for _, it := range items {
					hm, rows := headMatrices(it, mats[mi])

					for hi, hs := range hsets {
						for _, tl := range tails(mats[mi].Norm) {
							k := key{mi, hi, tl}
							if local[k] == nil {
								local[k] = map[string]*timeScore{}
							}

							if local[k][it.Set] == nil {
								local[k][it.Set] = &timeScore{}
							}

							local[k][it.Set].score(it, gridSpans(hm, rows, it, words[it], hs.heads, tl, mats[mi]))
						}
					}
				}
			}

			mu.Lock()
			defer mu.Unlock()

			for k, sets := range local {
				r := all[k]
				if r == nil {
					r = &gridResult{
						opt: mats[k.m].options(hsets[k.h].heads, k.tl), name: hsets[k.h].name, sets: map[string]*timeScore{},
					}
					all[k] = r
				}

				for set, s := range sets {
					if r.sets[set] == nil {
						r.sets[set] = &timeScore{}
					}

					r.sets[set].add(s)
				}
			}
		})
	}

	for i := range mats {
		jobs <- i
	}

	close(jobs)
	wg.Wait()

	out := make([]*gridResult, 0, len(all))
	for _, r := range all {
		out = append(out, r)
	}

	return out
}

// gridSpans aligns one item under one variant; nil when it fails.
func gridSpans(hm [][]float64, rows int, it *item, ws []Word, heads []Head, tl tail, c matConfig) []Span {
	if hm == nil {
		return nil
	}

	cost := pathCost(hm, heads, rows*it.Frames, tl.Cost)
	if cost == nil {
		return nil
	}

	starts, ends := DTW(cost, rows, it.Frames, tl.Step)
	if starts == nil {
		return nil
	}

	return spans(starts, ends, ws, c.options(heads, tl), it.Frames)
}

// matConfigs is the grid of matrix configs.
func matConfigs() []matConfig {
	var out []matConfig

	for _, gate := range []bool{false, true} {
		for _, rows := range []Rows{RowsPredict, RowsInput} {
			for lead := range 3 {
				for _, norm := range []Norm{NormNone, NormFrame, NormFrameAll, NormRow} {
					for _, filter := range []int{1, 3, 5, 7} {
						out = append(out, matConfig{Gate: gate, Rows: rows, Lead: lead, Norm: norm, Filter: filter})
					}
				}
			}
		}
	}

	return out
}

// TestProbeTimes ranks the timing grid on the LibriSpeech subset (the
// training split) by exact frames and reports the leaders on the held-out
// fixtures and keyword cases: 192 matrix configs (gate, rows, lead rows,
// normalisation, median filter) × 12 head sets (all, halves, single
// layers) × the cost, step pattern, end rule, core tokens and an offset.
// Without the onset rule, which TestProbeOnsetFit adds.
func TestProbeTimes(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)

	start := time.Now()
	res := runGrid(items, words, matConfigs(), headSets())

	slices.SortFunc(res, func(a, b *gridResult) int {
		sa, sb := a.sets["librispeech"], b.sets["librispeech"]
		switch {
		case betterExact(sa, sb):
			return -1
		case betterExact(sb, sa):
			return 1
		}

		return strings.Compare(a.opt.String(), b.opt.String())
	})

	t.Logf("── grid: %d variants in %s, ranked on librispeech by exact frames (start + end), then within 80 ms",
		len(res), time.Since(start).Round(time.Second))
	t.Logf("| # | heads | options | set %s", timeHeader)

	for i, r := range res[:min(probeTop(), len(res))] {
		for _, set := range []string{"librispeech", "fixtures", "keywords"} {
			t.Logf("| %d | %s | `%s` | %s | %s |", i+1, r.name, r.opt, set, r.sets[set])
		}
	}

	marginals(t, res)
}

// marginals prints, per axis value, the best variant (res is ranked) on
// the training set and on the held-out fixtures.
func marginals(t *testing.T, res []*gridResult) {
	t.Helper()

	axes := []struct {
		name  string
		value func(r *gridResult) string
	}{
		{"heads", func(r *gridResult) string { return r.name }},
		{"gate", func(r *gridResult) string { return strconv.FormatBool(r.opt.Gate) }},
		{"rows", func(r *gridResult) string { return pick(r.opt.Rows, "predict", "input") }},
		{"lead", func(r *gridResult) string { return strconv.Itoa(r.opt.Lead) }},
		{"norm", func(r *gridResult) string { return pick(r.opt.Norm, "none", "frame", "row", "frameall") }},
		{"filter", func(r *gridResult) string { return strconv.Itoa(r.opt.Filter) }},
		{"cost", func(r *gridResult) string { return pick(r.opt.Cost, "neg", "neglog") }},
		{"step", func(r *gridResult) string { return pick(r.opt.Step, "whisper", "noshare") }},
		{"end", func(r *gridResult) string { return pick(r.opt.End, "next", "last") }},
		{"core", func(r *gridResult) string { return strconv.FormatBool(r.opt.Core) }},
		{"offset", func(r *gridResult) string { return strconv.Itoa(r.opt.Offset) }},
	}

	t.Logf("── marginals: the best variant per axis value")
	t.Logf("| axis | value | librispeech exact start / end | within 80 ms | fixtures exact start / end | within 80 ms |")
	t.Logf("| ---- | ----- | ----------------------------- | ------------ | -------------------------- | ------------ |")

	for _, a := range axes {
		seen := map[string]bool{}

		for _, r := range res {
			v := a.value(r)
			if seen[v] {
				continue
			}

			seen[v] = true
			l, f := r.sets["librispeech"], r.sets["fixtures"]
			t.Logf("| %s | %s | %.1f %% / %.1f %% | %.1f %% | %.1f %% / %.1f %% | %.1f %% |", a.name, v,
				l.share(l.startExact), l.share(l.endExact), l.share(l.both1), f.share(f.startExact), f.share(f.endExact),
				f.share(f.both1))
		}
	}
}

// precomputed is the per-head matrices of every item under one config.
type precomputed struct {
	it   *item
	ws   []Word
	hm   [][]float64
	rows int
}

func precompute(items []*item, words map[*item][]Word, c matConfig) []precomputed {
	out := make([]precomputed, len(items))

	var wg sync.WaitGroup

	for i, it := range items {
		wg.Go(func() {
			hm, rows := headMatrices(it, c)
			out[i] = precomputed{it: it, ws: words[it], hm: hm, rows: rows}
		})
	}

	wg.Wait()

	return out
}

// evalHeads scores one head set and tail on the precomputed items, per
// set.
func evalHeads(pre []precomputed, c matConfig, heads []Head, tl tail) map[string]*timeScore {
	parts := make([]timeScore, len(pre))

	var wg sync.WaitGroup

	for i, p := range pre {
		wg.Go(func() {
			parts[i].score(p.it, gridSpans(p.hm, p.rows, p.it, p.ws, heads, tl, c))
		})
	}

	wg.Wait()

	out := map[string]*timeScore{}

	for i, p := range pre {
		if out[p.it.Set] == nil {
			out[p.it.Set] = &timeScore{}
		}

		out[p.it.Set].add(&parts[i])
	}

	return out
}

// exactKey ranks a training score by exact frames, then within 80 ms.
func exactKey(s *timeScore) [2]int { return [2]int{s.startExact + s.endExact, s.both1} }

func betterExact(a, b *timeScore) bool {
	ka, kb := exactKey(a), exactKey(b)

	return ka[0] > kb[0] || (ka[0] == kb[0] && ka[1] > kb[1])
}

// selectHeads grows a head set greedily on the training set, best exact
// frames first, until no head improves it or max heads.
func selectHeads(pre []precomputed, c matConfig, tl tail, train string, maxHeads int) ([]Head, []string) {
	var (
		sel  []Head
		best *timeScore
		log  []string
	)

	train1 := slices.DeleteFunc(slices.Clone(pre), func(p precomputed) bool { return p.it.Set != train })

	for len(sel) < maxHeads {
		var (
			pick  Head
			pickS *timeScore
		)

		for l := range 8 {
			for h := range 8 {
				cand := Head{Layer: l, Head: h}
				if slices.Contains(sel, cand) {
					continue
				}

				s := evalHeads(train1, c, append(slices.Clone(sel), cand), tl)[train]
				if pickS == nil || betterExact(s, pickS) {
					pick, pickS = cand, s
				}
			}
		}

		if best != nil && !betterExact(pickS, best) {
			break
		}

		sel, best = append(sel, pick), pickS
		log = append(log, fmt.Sprintf("+%d.%d → exact %.1f %% / %.1f %%, both within 80 ms %.1f %%", pick.Layer,
			pick.Head, best.share(best.startExact), best.share(best.endExact), best.share(best.both1)))
	}

	return sel, log
}

func headNames(hs []Head) string {
	out := make([]string, len(hs))
	for i, h := range hs {
		out[i] = fmt.Sprintf("%d.%d", h.Layer, h.Head)
	}

	return strings.Join(out, " ")
}

// TestProbeHeads fits a head set on the LibriSpeech subset for the grid's
// leading family and reports it on the held-out sets, with every single
// head's score.
func TestProbeHeads(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	tl := tail{Cost: CostNeg, Step: StepWhisper, End: EndNext, Core: true}

	for _, norm := range []Norm{NormFrameAll, NormFrame} {
		for _, filter := range []int{1, 3} {
			for _, gate := range []bool{false, true} {
				c := matConfig{Gate: gate, Rows: RowsPredict, Lead: 1, Norm: norm, Filter: filter}
				pre := precompute(items, words, c)

				sel, log := selectHeads(pre, c, tl, "librispeech", 16)
				res := evalHeads(pre, c, sel, tl)

				t.Logf("── %s (the selection in order: %s)", c.options(sel, tl), headNames(sel))

				for _, l := range log {
					t.Logf("    %s", l)
				}

				t.Logf("| set %s", timeHeader)

				for _, set := range []string{"librispeech", "fixtures", "keywords"} {
					t.Logf("| %s | %s |", set, res[set])
				}

				if norm != NormFrameAll || filter != 1 || gate {
					continue
				}

				t.Logf("single heads on librispeech (exact start / end %%):")

				for l := range 8 {
					var row []string

					for h := range 8 {
						s := evalHeads(pre, c, []Head{{l, h}}, tl)["librispeech"]
						row = append(row, fmt.Sprintf("%4.1f/%4.1f", s.share(s.startExact), s.share(s.endExact)))
					}

					t.Logf("    layer %d: %s", l, strings.Join(row, "  "))
				}
			}
		}
	}
}

// TestProbeLayers scores every subset of whole layers (all eight heads of
// each) for the grid's leading family, ranked by exact frames on the
// training set.
func TestProbeLayers(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	tl := tail{Cost: CostNeg, Step: StepWhisper, End: EndNext, Core: true}

	type row struct {
		c    matConfig
		mask int
		res  map[string]*timeScore
	}

	var rows []row

	for _, norm := range []Norm{NormFrame, NormFrameAll, NormNone} {
		for _, filter := range []int{1, 3, 7} {
			for _, gate := range []bool{false, true} {
				c := matConfig{Gate: gate, Rows: RowsPredict, Lead: 1, Norm: norm, Filter: filter}
				pre := precompute(items, words, c)

				for mask := 1; mask < 256; mask++ {
					rows = append(rows, row{c: c, mask: mask, res: evalHeads(pre, c, layerHeads(mask), tl)})
				}
			}
		}
	}

	slices.SortStableFunc(rows, func(a, b row) int {
		switch sa, sb := a.res["librispeech"], b.res["librispeech"]; {
		case betterExact(sa, sb):
			return -1
		case betterExact(sb, sa):
			return 1
		}

		return 0
	})

	t.Logf("| # | layers | options | set %s", timeHeader)

	for i, r := range rows[:min(probeTop(), len(rows))] {
		for _, set := range []string{"librispeech", "fixtures", "keywords"} {
			t.Logf("| %d | %s | `%s` | %s | %s |", i+1, layerNames(r.mask), r.c.options(layerHeads(r.mask), tl), set,
				r.res[set])
		}
	}
}

// layerHeads is every head of the layers in mask (bit l = layer l).
func layerHeads(mask int) []Head {
	var out []Head

	for l := range 8 {
		if mask&(1<<l) != 0 {
			for h := range 8 {
				out = append(out, Head{Layer: l, Head: h})
			}
		}
	}

	return out
}

func layerNames(mask int) string {
	var out []string

	for l := range 8 {
		if mask&(1<<l) != 0 {
			out = append(out, strconv.Itoa(l))
		}
	}

	return strings.Join(out, ",")
}

// TestProbeMisses breaks Default's mismatches down by word position and
// size, and prints the first ones (WHISTLE_PROBE_TOP).
func TestProbeMisses(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	ins := withLevels(t, items)
	opt := Default()

	count := map[string]int{}
	shown := 0

	for _, it := range items {
		sp, err := Times(ins[it], words[it], opt)
		if err != nil {
			t.Fatal(err)
		}

		for i, w := range it.Words {
			where := "middle"

			switch {
			case len(it.Words) == 1:
				where = "only"
			case i == 0:
				where = "first"
			case i == len(it.Words)-1:
				where = "last"
			}

			ds, de := sp[i].Start-frameOf(w.Start), sp[i].End-frameOf(w.End)
			count["words "+where]++

			if ds != 0 {
				count[fmt.Sprintf("start %s Δ%+d", where, max(-3, min(3, ds)))]++
			}

			if de != 0 {
				count[fmt.Sprintf("end %s Δ%+d", where, max(-3, min(3, de)))]++
			}

			if (ds != 0 || de != 0) && shown < probeTop() {
				shown++

				t.Logf("%s %s word %d %q: engine %.2f–%.2f, ours %.2f–%.2f", it.Set, it.ID, i, w.Word, w.Start, w.End,
					float64(sp[i].Start)*0.08, float64(sp[i].End)*0.08)
			}
		}
	}

	for _, k := range sortedKeys(count) {
		t.Logf("%-24s %d", k, count[k])
	}
}

// TestProbePunctuation: for every word with leading or trailing
// punctuation tokens, whether the engine's start / end matches the time of
// the core or of all tokens, by the punctuation pieces.
func TestProbePunctuation(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	base := Options{
		Heads: layerHeads(0xf0), Rows: RowsPredict, Lead: 1, Norm: NormFrameAll, Filter: 1, Cost: CostNeg,
		Step: StepWhisper, End: EndNext, Core: true,
	}
	whole := base
	whole.Core = false

	count := map[string][3]int{} // core only, all only, both

	for _, it := range items {
		a, err := Times(input(it), words[it], base)
		if err != nil {
			t.Fatal(err)
		}

		b, err := Times(input(it), words[it], whole)
		if err != nil {
			t.Fatal(err)
		}

		for i, w := range words[it] {
			var pieces []string

			key := ""

			if w.Core[1] < w.End {
				for _, id := range it.Tokens[w.Core[1]:w.End] {
					pieces = append(pieces, ps.tok.Piece(id))
				}

				key = fmt.Sprintf("end %q last=%t", strings.Join(pieces, " "), i == len(words[it])-1)
				e := frameOf(it.Words[i].End)
				c := count[key]

				switch {
				case a[i].End == e && b[i].End == e:
					c[2]++
				case a[i].End == e:
					c[0]++
				case b[i].End == e:
					c[1]++
				}

				count[key] = c
			}

			if w.Core[0] > w.Start {
				pieces = pieces[:0]
				for _, id := range it.Tokens[w.Start:w.Core[0]] {
					pieces = append(pieces, ps.tok.Piece(id))
				}

				key = fmt.Sprintf("start %q first=%t", strings.Join(pieces, " "), i == 0)
				s := frameOf(it.Words[i].Start)
				c := count[key]

				switch {
				case a[i].Start == s && b[i].Start == s:
					c[2]++
				case a[i].Start == s:
					c[0]++
				case b[i].Start == s:
					c[1]++
				}

				count[key] = c
			}
		}
	}

	t.Logf("%-40s %6s %6s %6s", "punctuation tokens", "core", "all", "both")

	for _, k := range sortedKeys(count) {
		c := count[k]
		t.Logf("%-40s %6d %6d %6d", k, c[0], c[1], c[2])
	}
}

// TestProbeDepth scores the leader on the depth runs (F-025) with two
// readings of "the last half of the layers" on a ladder rung: the layers
// 4–7 that run, or the last half of the layers that run.
func TestProbeDepth(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set != "depth" })
	words := itemWords(t, ps, items)
	ins := withLevels(t, items)

	rules := map[string]func(run []int) []int{
		"layers 4-7 that run": func(run []int) []int {
			return slices.DeleteFunc(slices.Clone(run), func(l int) bool { return l < 4 })
		},
		"last half of the layers that run": func(run []int) []int { return run[len(run)/2:] },
		"all layers that run":              slices.Clone[[]int],
	}

	for _, name := range sortedKeys(rules) {
		per := map[int]*timeScore{}

		for _, it := range items {
			run, err := decoderLadder(it.Depth)
			if err != nil {
				t.Fatal(err)
			}

			mask := 0
			for _, l := range rules[name](run) {
				mask |= 1 << l
			}

			opt := Default()
			opt.Heads = layerHeads(mask)

			sp, err := Times(ins[it], words[it], opt)
			if err != nil {
				t.Fatal(err)
			}

			if per[it.Depth] == nil {
				per[it.Depth] = &timeScore{}
			}

			per[it.Depth].score(it, sp)
		}

		var all timeScore

		for d := 2; d <= 7; d++ {
			all.add(per[d])
			t.Logf("%-34s depth %d: %s", name, d, per[d])
		}

		t.Logf("%-34s all:     %s", name, &all)
	}
}

// decoderLadder is the layers depth runs (0 for all eight).
func decoderLadder(depth int) ([]int, error) {
	if depth == 0 {
		return []int{0, 1, 2, 3, 4, 5, 6, 7}, nil
	}

	return decoder.LadderLayers(8, depth)
}
