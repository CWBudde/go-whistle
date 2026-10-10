//go:build oracle

// The first word's start (docs/FINDINGS.md F-033): every word but the first
// is timed exactly on 94–99 % of words, the first one's start on 8 %. These
// tests score the candidates for it; the onset rule is fitted, not
// identified.

package align

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/silence"
)

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}

	if len(v)%2 == 1 {
		return v[len(v)/2]
	}

	return (v[len(v)/2-1] + v[len(v)/2]) / 2
}

// TestProbeFirstStart scores candidate rules for the first word's start
// alone (the leader's other times are exact on 94–98 % of words).
func TestProbeFirstStart(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	base := Options{
		Heads: layerHeads(0xf0), Rows: RowsPredict, Lead: 1, Norm: NormFrameAll, Filter: 1, Cost: CostNeg,
		Step: StepWhisper, End: EndNext, Core: true,
	}

	type cand struct {
		name string
		f    func(it *item, lead []Span) int
	}

	alt := func(o Options) func(it *item, _ []Span) int {
		return func(it *item, _ []Span) int {
			sp, err := Times(input(it), words[it], o)
			if err != nil {
				return -99
			}

			return sp[0].Start
		}
	}

	with := func(f func(*Options)) Options {
		o := base
		f(&o)

		return o
	}

	cands := []cand{
		{"leader", func(_ *item, sp []Span) int { return sp[0].Start }},
		{"input rows, lead 1", alt(with(func(o *Options) { o.Rows = RowsInput }))},
		{"input rows, lead 2", alt(with(func(o *Options) { o.Rows, o.Lead = RowsInput, 2 }))},
		{"predict rows, frame norm", alt(with(func(o *Options) { o.Norm = NormFrame }))},
		{"predict rows, no norm", alt(with(func(o *Options) { o.Norm = NormNone }))},
		{"predict rows, all heads", alt(with(func(o *Options) { o.Heads = nil }))},
		{"predict rows, core off", alt(with(func(o *Options) { o.Core = false }))},
	}

	for _, c := range cands {
		var n, exact, near int

		for _, it := range items {
			sp, err := Times(input(it), words[it], base)
			if err != nil {
				t.Fatal(err)
			}

			d := c.f(it, sp) - frameOf(it.Words[0].Start)
			n++

			if d == 0 {
				exact++
			}

			if d >= -1 && d <= 1 {
				near++
			}
		}

		t.Logf("%-28s first start exact %d/%d, within 80 ms %d", c.name, exact, n, near)
	}
}

// TestProbeOnset checks whether the first word starts at the speech onset
// by energy: the first 80 ms frame whose speech-band level (the silence
// gate's, F-031) comes within X dB of the clip's 99th percentile, alone or
// as a floor under the DTW start.
func TestProbeOnset(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	base := Options{
		Heads: layerHeads(0xf0), Rows: RowsPredict, Lead: 1, Norm: NormFrameAll, Filter: 1, Cost: CostNeg,
		Step: StepWhisper, End: EndNext, Core: true,
	}

	gate, err := silence.New()
	if err != nil {
		t.Fatal(err)
	}

	type clipLevels struct {
		frame []float64 // per 80 ms frame: the max of its eight 10 ms levels
		sub   []float64 // the 10 ms levels
		p99   float64
		dtw   int
	}

	lv := map[*item]clipLevels{}

	for _, it := range items {
		pcm, err := frontend.ReadWAV(it.Clip)
		if err != nil {
			t.Fatal(err)
		}

		l, err := gate.Levels(pcm)
		if err != nil {
			t.Fatal(err)
		}

		sorted := slices.Clone(l)
		slices.Sort(sorted)

		c := clipLevels{p99: silence.Percentile(sorted, 99), sub: l}

		for f := 0; f*8 < len(l); f++ {
			c.frame = append(c.frame, slices.Max(l[f*8:min(len(l), f*8+8)]))
		}

		sp, err := Times(input(it), words[it], base)
		if err != nil {
			t.Fatal(err)
		}

		c.dtw = sp[0].Start
		lv[it] = c
	}

	for x := 10.0; x <= 50; x += 5 {
		for _, mapping := range []string{"floor", "round", "ceil"} {
			var hit, hitMax, near, nearMax int

			for _, it := range items {
				c := lv[it]
				on := len(c.sub)

				for i, v := range c.sub {
					if v >= c.p99-x {
						on = i

						break
					}
				}

				f := on / 8
				switch mapping {
				case "round":
					f = (on + 4) / 8
				case "ceil":
					f = (on + 7) / 8
				}

				if f == frameOf(it.Words[0].Start) {
					hit++
				}

				if max(f, c.dtw) == frameOf(it.Words[0].Start) {
					hitMax++
				}

				if d := f - frameOf(it.Words[0].Start); d >= -1 && d <= 1 {
					near++
				}

				if d := max(f, c.dtw) - frameOf(it.Words[0].Start); d >= -1 && d <= 1 {
					nearMax++
				}
			}

			t.Logf("X = %2.0f dB, 10 ms onset, %s: alone %d/%d (within 80 ms %d), max(onset, DTW) %d (%d)", x, mapping,
				hit, len(items), near, hitMax, nearMax)
		}
	}

	for x := 10.0; x <= 50; x += 5 {
		var alone, floor, n int

		for _, it := range items {
			c := lv[it]
			on := 0

			for f, v := range c.frame {
				if v >= c.p99-x {
					on = f

					break
				}
			}

			want := frameOf(it.Words[0].Start)
			n++

			if on == want {
				alone++
			}

			if max(on, c.dtw) == want {
				floor++
			}
		}

		t.Logf("X = %2.0f dB: onset alone %d/%d, max(onset, DTW) %d/%d", x, alone, n, floor, n)
	}

	for i, it := range items[:min(probeTop(), len(items))] {
		c := lv[it]

		var b strings.Builder
		for f := range min(len(c.frame), 16) {
			fmt.Fprintf(&b, "%6.1f", c.frame[f]-c.p99)
		}

		t.Logf("%2d %-30s engine %2d dtw %2d | %s", i, it.ID, frameOf(it.Words[0].Start), c.dtw, b.String())
	}
}

// TestProbeTransforms scores per-head transforms of the attention before
// the per-frame z-score over all positions, by the first word's start and
// by every start and end.
func TestProbeTransforms(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	opt := Options{Rows: RowsPredict, Lead: 1, Cost: CostNeg, Step: StepWhisper, End: EndNext, Core: true}

	type transform struct {
		name string
		f    func(row []float64)
	}

	rowZ := func(r []float64) { zscore(r, len(r), 1) }
	transforms := []transform{
		{"none", func([]float64) {}},
		{"row z-score", rowZ},
		{"log", func(r []float64) {
			for i, v := range r {
				r[i] = math.Log(math.Max(v, 1e-12))
			}
		}},
		{"÷ row max", func(r []float64) {
			m := slices.Max(r)
			for i := range r {
				r[i] /= m
			}
		}},
		{"frame 0 zeroed, renormalised", func(r []float64) {
			r[0] = 0

			var s float64
			for _, v := range r {
				s += v
			}

			for i := range r {
				r[i] /= s
			}
		}},
		{"sqrt", func(r []float64) {
			for i, v := range r {
				r[i] = math.Sqrt(v)
			}
		}},
		{"square", func(r []float64) {
			for i, v := range r {
				r[i] = v * v
			}
		}},
	}

	for _, tr := range transforms {
		for _, mask := range []int{0xf0, 0xff} {
			var first, n timeScore

			for _, it := range items {
				in := input(it)
				top, rows := it.first()-2, len(it.Tokens)+2
				f := it.Frames
				sum := make([]float64, rows*f)

				for _, h := range layerHeads(mask) {
					all := positions(in, h.Layer)
					m := make([]float64, all*f)

					for p := range all {
						err := headRow(m[p*f:(p+1)*f], in, h, p, false)
						if err != nil {
							t.Fatal(err)
						}

						tr.f(m[p*f : (p+1)*f])
					}

					Normalise(m, all, f, NormFrame)

					for i := range sum {
						sum[i] += m[top*f+i]
					}
				}

				for i := range sum {
					sum[i] = -sum[i]
				}

				starts, ends := DTW(sum, rows, f, StepWhisper)
				sp := spans(starts, ends, words[it], opt, f)

				n.score(it, sp)

				one := *it
				one.Words = it.Words[:1]
				first.score(&one, sp[:1])
			}

			t.Logf("%-30s layers %s: first start exact %.1f %%; all words exact start/end %.1f %% / %.1f %%, both within 80 ms %.1f %%",
				tr.name, layerNames(mask), first.share(first.startExact), n.share(n.startExact), n.share(n.endExact),
				n.share(n.both1))
		}
	}
}

// TestProbeOrder: the heads averaged before the per-frame z-score instead
// of after, with the statistics over all positions or the DTW rows.
func TestProbeOrder(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	opt := Options{Rows: RowsPredict, Lead: 1, Cost: CostNeg, Step: StepWhisper, End: EndNext, Core: true}

	for _, mask := range []int{0xf0, 0xff, 0xc0, 0x80} {
		for _, stats := range []string{"all positions", "DTW rows", "text rows", "none"} {
			var first, n timeScore

			for _, it := range items {
				in := input(it)
				all, f := positions(in, 4), it.Frames
				top, rows := it.first()-2, len(it.Tokens)+2
				mean := make([]float64, all*f)
				hs := layerHeads(mask)
				buf := make([]float64, f)

				for _, h := range hs {
					for p := range all {
						err := headRow(buf, in, h, p, false)
						if err != nil {
							t.Fatal(err)
						}

						for i, v := range buf {
							mean[p*f+i] += v / float64(len(hs))
						}
					}
				}

				m := mean[top*f : (top+rows)*f]

				switch stats {
				case "all positions":
					Normalise(mean, all, f, NormFrame)
				case "DTW rows":
					Normalise(m, rows, f, NormFrame)
				case "text rows":
					Normalise(m[f:], rows-1, f, NormFrame)
				}

				cost := make([]float64, len(m))
				for i, v := range m {
					cost[i] = -v
				}

				starts, ends := DTW(cost, rows, f, StepWhisper)
				sp := spans(starts, ends, words[it], opt, f)
				n.score(it, sp)

				one := *it
				one.Words = it.Words[:1]
				first.score(&one, sp[:1])
			}

			t.Logf("layers %-16s z over %-14s: first start exact %.1f %%; all exact start/end %.1f %% / %.1f %%, both within 80 ms %.1f %%",
				layerNames(mask), stats, first.share(first.startExact), n.share(n.startExact), n.share(n.endExact),
				n.share(n.both1))
		}
	}
}

// TestProbeLeadSource replaces the leader's lead row by candidate rows.
func TestProbeLeadSource(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	opt := Options{Rows: RowsPredict, Lead: 1, Cost: CostNeg, Step: StepWhisper, End: EndNext, Core: true}

	type source struct {
		name string
		row  func(z []float64, all, f, top, rows int) []float64 // z: all positions × f, normalised
	}

	pos := func(p func(all int) int) func(z []float64, all, f, top, rows int) []float64 {
		return func(z []float64, all, f, _, _ int) []float64 { return z[p(all)*f : (p(all)+1)*f] }
	}

	constant := func(c float64) func(z []float64, all, f, top, rows int) []float64 {
		return func(_ []float64, _, f, _, _ int) []float64 {
			out := make([]float64, f)
			for i := range out {
				out[i] = c
			}

			return out
		}
	}

	meanOf := func(lo, hi func(all, top, rows int) int) func(z []float64, all, f, top, rows int) []float64 {
		return func(z []float64, all, f, top, rows int) []float64 {
			out := make([]float64, f)
			a, b := lo(all, top, rows), hi(all, top, rows)

			for p := a; p < b; p++ {
				for i := range f {
					out[i] += z[p*f+i] / float64(b-a)
				}
			}

			return out
		}
	}

	sources := []source{
		{"BOS input (leader)", pos(func(int) int { return 0 })},
		{"EOS input", pos(func(all int) int { return all - 1 })},
		{"mean of all positions", meanOf(func(int, int, int) int { return 0 }, func(all, _, _ int) int { return all })},
		{"mean of the text rows", meanOf(func(_, top, _ int) int { return top + 1 }, func(_, top, rows int) int { return top + rows })},
	}

	for c := -0.5; c <= 1.51; c += 0.25 {
		sources = append(sources, source{fmt.Sprintf("constant %.2f", c), constant(c)})
	}

	for _, src := range sources {
		var first, n timeScore

		for _, it := range items {
			in := input(it)
			all, f := positions(in, 4), it.Frames
			top, rows := it.first()-2, len(it.Tokens)+2
			z := make([]float64, all*f)

			for _, h := range layerHeads(0xf0) {
				m, err := HeadMatrix(in, h, 0, all, Options{Norm: NormFrame, Filter: 1})
				if err != nil {
					t.Fatal(err)
				}

				for i, v := range m {
					z[i] += v / 32
				}
			}

			m := slices.Clone(z[top*f : (top+rows)*f])
			copy(m[:f], src.row(z, all, f, top, rows))

			for i, v := range m {
				m[i] = -v
			}

			starts, ends := DTW(m, rows, f, StepWhisper)
			sp := spans(starts, ends, words[it], opt, f)
			n.score(it, sp)

			one := *it
			one.Words = it.Words[:1]
			first.score(&one, sp[:1])
		}

		t.Logf("%-24s first start exact %5.1f %% within 80 ms %5.1f %%; all exact %.1f %% / %.1f %%, both within 80 ms %.1f %%",
			src.name, first.share(first.startExact), first.share(first.start1), n.share(n.startExact),
			n.share(n.endExact), n.share(n.both1))
	}
}

// TestProbeFirstDuration: the first start as the first end minus a
// duration (a constant, or a statistic of the other words' durations),
// never before our DTW start.
func TestProbeFirstDuration(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	base := Options{
		Heads: layerHeads(0xf0), Rows: RowsPredict, Lead: 1, Norm: NormFrameAll, Filter: 1, Cost: CostNeg,
		Step: StepWhisper, End: EndNext, Core: true,
	}

	type rule struct {
		name string
		dur  func(sp []Span) float64
	}

	stat := func(f func([]float64) float64) func(sp []Span) float64 {
		return func(sp []Span) float64 {
			var d []float64

			for _, s := range sp[1:] {
				if s.End > s.Start {
					d = append(d, float64(s.End-s.Start))
				}
			}

			slices.Sort(d)

			return f(d)
		}
	}

	rules := []rule{
		{"2 frames", func([]Span) float64 { return 2 }},
		{"3 frames", func([]Span) float64 { return 3 }},
		{"median", stat(median)},
		{"2 × median", stat(func(d []float64) float64 { return 2 * median(d) })},
		{"mean", stat(func(d []float64) float64 {
			var s float64
			for _, v := range d {
				s += v
			}

			return s / float64(max(1, len(d)))
		})},
	}

	for _, r := range rules {
		for _, rnd := range []string{"floor", "round", "ceil"} {
			var exact, near int

			for _, it := range items {
				sp, err := Times(input(it), words[it], base)
				if err != nil {
					t.Fatal(err)
				}

				d := r.dur(sp)

				var k int

				switch rnd {
				case "floor":
					k = int(math.Floor(d))
				case "round":
					k = int(math.Round(d))
				default:
					k = int(math.Ceil(d))
				}

				s := max(sp[0].Start, sp[0].End-k)
				e := frameOf(it.Words[0].Start)

				if s == e {
					exact++
				}

				if s-e >= -1 && s-e <= 1 {
					near++
				}
			}

			t.Logf("end − %-12s (%s): first start exact %d/%d, within 80 ms %d", r.name, rnd, exact, len(items), near)
		}
	}
}

// TestProbeOnsetFit fits Options.Onset on the LibriSpeech subset and
// reports it on the held-out sets, beside the leader without it.
func TestProbeOnsetFit(t *testing.T) {
	ps := openSetup(t)
	items := slices.DeleteFunc(loadItems(t, ps), func(it *item) bool { return it.Set == "depth" })
	words := itemWords(t, ps, items)
	ins := withLevels(t, items)
	base := Options{
		Heads: layerHeads(0xf0), Rows: RowsPredict, Lead: 1, Norm: NormFrameAll, Filter: 1, Cost: CostNeg,
		Step: StepWhisper, End: EndNext, Core: true,
	}

	eval := func(opt Options) (map[string]*timeScore, map[string]*timeScore) {
		all, first := map[string]*timeScore{}, map[string]*timeScore{}

		for _, it := range items {
			sp, err := Times(ins[it], words[it], opt)
			if err != nil {
				t.Fatal(err)
			}

			if all[it.Set] == nil {
				all[it.Set], first[it.Set] = &timeScore{}, &timeScore{}
			}

			all[it.Set].score(it, sp)

			one := *it
			one.Words = it.Words[:1]
			first[it.Set].score(&one, sp[:1])
		}

		return all, first
	}

	bestX, best := 0.0, (*timeScore)(nil)

	for x := 5.0; x <= 30; x += 0.5 {
		opt := base
		opt.Onset = x

		_, first := eval(opt)
		if s := first["librispeech"]; best == nil || betterExact(s, best) {
			bestX, best = x, s
		}
	}

	for _, x := range []float64{0, bestX} {
		opt := base
		opt.Onset = x
		all, first := eval(opt)

		t.Logf("── `%s`", opt)
		t.Logf("| set | first start exact | first start within 80 ms %s", timeHeader)

		for _, set := range []string{"librispeech", "fixtures", "keywords"} {
			t.Logf("| %s | %.1f %% | %.1f %% | %s |", set, first[set].share(first[set].startExact),
				first[set].share(first[set].start1), all[set])
		}
	}
}
