//go:build oracle

// The decoder probe harness (PLAN.md Phase 5, docs/FINDINGS.md F-021 to
// F-024). The engine's decoder is observable only through its transcripts,
// so it scores Hypothesis values by teacher forcing the committed engine
// transcripts (testdata/oracle/<clip>.json): each transcript becomes its
// target tokens (prompt, language token, RefTokenizer ids, EOS), and a
// hypothesis earns the mean log-probability of those tokens, with the top-1
// accuracy and the language token's rank beside it. It searches the axes
// stage by stage with a beam, then by coordinate descent over single axes,
// axis pairs and whole stages, and prints the leader's per-axis
// sensitivity. TestProbeGreedy decodes every clip greedily with the
// leader. Run it with `just decoder-probe`; the knobs are environment
// variables:
//
//	WHISTLE_PROBE_SRC    cact (default) or safetensors: the weights the search runs on
//	WHISTLE_PROBE_BEAM   hypotheses kept per stage (default 4)
//	WHISTLE_PROBE_TOP    leaderboard rows per stage (default 8)
//	WHISTLE_PROBE_KEY    the Hypothesis.String to start from (TestProbeSearch) or to score (TestProbeScore, TestProbeGreedy)

package decoder

import (
	"cmp"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/tokenizer"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// result is one hypothesis' teacher-forced score.
type result struct {
	h        Hypothesis
	perClip  []float64 // mean log-probability per clip
	mean     float64   // mean over clips
	min      float64   // worst clip
	top1     float64   // top-1 accuracy over all predicted tokens
	langOK   int       // clips whose language token wins among the language tokens
	langRank float64   // mean rank of the language token among all tokens
	ruled    string    // why the hypothesis could not run
}

func (r result) ok() bool { return r.ruled == "" }

// evaluator scores hypotheses on the clips with one weight source, caching
// the encoder output per memory tap.
type evaluator struct {
	src    weights.Source
	tok    *tokenizer.Tokenizer
	clips  []clip
	frames map[MemoryTap][]int
	encs   map[MemoryTap][][]float32
	shift  int // memory of clip (i + shift) mod n: 0 is the clip's own
	done   map[string]result
}

func newEvaluator(t testing.TB, src weights.Source, tok *tokenizer.Tokenizer, clips []clip) *evaluator {
	t.Helper()

	ev := &evaluator{
		src: src, tok: tok, clips: clips,
		frames: map[MemoryTap][]int{}, encs: map[MemoryTap][][]float32{}, done: map[string]result{},
	}

	for _, tap := range []MemoryTap{MemFinalNorm, MemLaneMean} {
		ev.encs[tap], ev.frames[tap] = encode(t, src, clips, tap)
	}

	return ev
}

// scoreAll scores hs, each hypothesis once per evaluator.
func (ev *evaluator) scoreAll(hs []Hypothesis) []result {
	out := make([]result, len(hs))

	for i, h := range hs {
		k := h.String()

		r, ok := ev.done[k]
		if !ok {
			r = ev.score(h)
			ev.done[k] = r
		}

		out[i] = r
	}

	return out
}

// score teacher-forces every clip with h, the clips on all cores.
func (ev *evaluator) score(h Hypothesis) result {
	r := result{h: h, perClip: make([]float64, len(ev.clips))}

	d, err := Load(ev.src, ev.tok, h)
	if err != nil {
		r.ruled = err.Error()

		return r
	}

	forced := make([]Forced, len(ev.clips))
	errs := make([]error, len(ev.clips))
	jobs := make(chan int)

	var wg sync.WaitGroup

	for range runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			for i := range jobs {
				forced[i], errs[i] = ev.force(d, h, i)
			}
		})
	}

	for i := range ev.clips {
		jobs <- i
	}

	close(jobs)
	wg.Wait()

	var hits, tokens int

	for i, f := range forced {
		if errs[i] != nil {
			r.ruled = fmt.Sprintf("%s: %v", ev.clips[i].ID, errs[i])

			return r
		}

		r.perClip[i] = mean(f.LogProbs)
		r.langRank += float64(f.LangRank) / float64(len(ev.clips))

		if f.LangTop {
			r.langOK++
		}

		for _, ok := range f.Top1 {
			tokens++

			if ok {
				hits++
			}
		}
	}

	r.mean, r.min, r.top1 = mean(r.perClip), slices.Min(r.perClip), float64(hits)/float64(tokens)

	return r
}

// force teacher-forces clip i's transcript against its memory (or the
// shifted clip's, for the language-model baseline).
func (ev *evaluator) force(d *Decoder, h Hypothesis, i int) (Forced, error) {
	j := (i + ev.shift) % len(ev.clips)

	mem, err := d.Memory(ev.encs[h.Memory][j], ev.frames[h.Memory][j])
	if err != nil {
		return Forced{}, err
	}

	seq, err := d.Targets(ev.clips[i].Language, ev.clips[i].Text)
	if err != nil {
		return Forced{}, err
	}

	return d.TeacherForce(mem, seq)
}

// mean is the arithmetic mean, 0 for none.
func mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}

	var s float64
	for _, x := range v {
		s += x
	}

	return s / float64(len(v))
}

// probeEnv reads an integer knob.
func probeEnv(name string, def int) int {
	v, err := strconv.Atoi(os.Getenv(name))
	if err == nil && v > 0 {
		return v
	}

	return def
}

// probeSource opens the weights named by WHISTLE_PROBE_SRC and the
// vocabulary, which only the .cact holds.
func probeSource(t *testing.T) (weights.Source, *tokenizer.Tokenizer, string) {
	t.Helper()

	tok := cactTokenizer(t)

	if os.Getenv("WHISTLE_PROBE_SRC") == "safetensors" {
		return openModel(t, whistleSafetensorsPath), tok, "safetensors"
	}

	return openModel(t, whistleCactPath), tok, "cact"
}

func cactTokenizer(t testing.TB) *tokenizer.Tokenizer {
	t.Helper()

	tok, err := LoadTokenizer(openModel(t, whistleCactPath))
	if err != nil {
		t.Fatal(err)
	}

	return tok
}

// probeStart is WHISTLE_PROBE_KEY, or def.
func probeStart(t *testing.T, def Hypothesis) Hypothesis {
	t.Helper()

	key := os.Getenv("WHISTLE_PROBE_KEY")
	if key == "" {
		return def
	}

	h, err := ParseHypothesis(key)
	if err != nil {
		t.Fatal(err)
	}

	return h
}

// prior is the starting hypothesis: upstream's Block, Stack and
// SimpleAttentionNetwork for everything they define, and the
// self-attention's form for the cross-attention (docs/FINDINGS.md F-021).
func prior() Hypothesis { return Best() }

// withValue returns h with axis key set to value v.
func withValue(h Hypothesis, key string, v int) Hypothesis {
	axisByKey(key).Set(&h, v)

	return h
}

func axisByKey(key string) Axis {
	for _, a := range axes {
		if a.Key == key {
			return a
		}
	}

	panic("probe: unknown axis " + key)
}

// product returns every combination of the given axes around h.
func product(h Hypothesis, keys []string) []Hypothesis {
	out := []Hypothesis{h}

	for _, k := range keys {
		var next []Hypothesis

		for _, base := range out {
			for v := range axisByKey(k).Names {
				next = append(next, withValue(base, k, v+1))
			}
		}

		out = next
	}

	return out
}

// dedupe drops repeated hypotheses, keeping the first.
func dedupe(hs []Hypothesis) []Hypothesis {
	seen := map[string]bool{}
	out := hs[:0:0]

	for _, h := range hs {
		if k := h.String(); !seen[k] {
			seen[k] = true

			out = append(out, h)
		}
	}

	return out
}

// rank sorts results best first: scored before failed, then by mean
// log-probability. The sort is stable, so exact ties keep the order of the
// candidates, the kept hypotheses' own values first.
func rank(rs []result) {
	slices.SortStableFunc(rs, func(a, b result) int {
		if a.ok() != b.ok() {
			if a.ok() {
				return -1
			}

			return 1
		}

		return cmp.Compare(b.mean, a.mean)
	})
}

// diff names the axes where h differs from ref, as key=value.
func diff(h, ref Hypothesis) string {
	a, b := strings.Fields(h.String()), strings.Fields(ref.String())

	var out []string

	for i := range a {
		if a[i] != b[i] {
			out = append(out, a[i])
		}
	}

	if len(out) == 0 {
		return "(reference)"
	}

	return strings.Join(out, " ")
}

// leaderboard prints the top rows of ranked results relative to ref.
func leaderboard(t *testing.T, title string, rs []result, ref Hypothesis, top int) {
	t.Helper()

	ruled := 0

	for _, r := range rs {
		if !r.ok() {
			ruled++
		}
	}

	t.Logf("── %s: %d hypotheses, %d could not run", title, len(rs), ruled)

	for i, r := range rs {
		if i == top || !r.ok() {
			break
		}

		t.Logf("  #%-2d logp %.4f min %.4f top1 %.4f lang %d/%d rank %.2f | %s", i+1, r.mean, r.min, r.top1,
			r.langOK, len(r.perClip), r.langRank, diff(r.h, ref))
	}
}

// stage is a set of axes searched jointly.
type stage struct {
	name string
	keys []string
}

// searchStages is the search order (docs/FINDINGS.md F-021): the
// cross-attention first, since without it the decoder is a language model;
// then the embedding and the head, the prompt and the target format, and
// the memory tap with the remaining switches.
var searchStages = []stage{
	{"cross-attention", []string{"cross", "qknorm", "gate", "residual"}},
	{"embedding and head", []string{"scale", "head", "final"}},
	{"prompt and target format", []string{"prompt", "lead"}},
	{"memory and switches", []string{"memory", "xrope", "engram"}},
}

// TestProbeBaselines prints what no signal looks like: the hypothesis with
// the memory of another clip (a language model guessing from the prompt),
// and with its own.
func TestProbeBaselines(t *testing.T) {
	src, tok, srcName := probeSource(t)
	ev := newEvaluator(t, src, tok, loadClips(t))
	h := probeStart(t, Best())

	own := ev.scoreAll([]Hypothesis{h})[0]
	ev.shift, ev.done = 1, map[string]result{}
	other := ev.scoreAll([]Hypothesis{h})[0]

	t.Logf("weights %s: %s", srcName, h)
	t.Logf("own memory:         logp %.4f min %.4f top1 %.4f lang %d/%d rank %.2f", own.mean, own.min, own.top1,
		own.langOK, len(own.perClip), own.langRank)
	t.Logf("next clip's memory: logp %.4f min %.4f top1 %.4f lang %d/%d rank %.2f", other.mean, other.min,
		other.top1, other.langOK, len(other.perClip), other.langRank)
}

// TestProbeSearch is the staged beam search, then the escalating
// coordinate descent until the leader stops changing, then the leader's
// per-axis sensitivity and its scores with both weight sources.
func TestProbeSearch(t *testing.T) {
	src, tok, srcName := probeSource(t)
	beam := probeEnv("WHISTLE_PROBE_BEAM", 4)
	top := probeEnv("WHISTLE_PROBE_TOP", 8)
	clips := loadClips(t)
	ev := newEvaluator(t, src, tok, clips)
	ref := probeStart(t, prior())

	t.Logf("weights %s, %d clips, beam %d; reference: %s", srcName, len(clips), beam, ref)

	kept := ev.scoreAll([]Hypothesis{ref})
	leaderboard(t, "reference", kept, ref, 1)

	for _, st := range searchStages {
		kept = searchStep(t, ev, "stage "+st.name+" ("+strings.Join(st.keys, ", ")+")", kept, st.keys, ref, beam, top)
	}

	singles := make([][]string, 0, len(axes))
	pairs := make([][]string, 0, len(axes)*(len(axes)-1)/2)
	stages := make([][]string, 0, len(searchStages))

	for i := range axes {
		singles = append(singles, []string{axes[i].Key})

		for j := i + 1; j < len(axes); j++ {
			pairs = append(pairs, []string{axes[i].Key, axes[j].Key})
		}
	}

	for _, st := range searchStages {
		stages = append(stages, st.keys)
	}

	moves := [][][]string{singles, pairs, stages}
	names := []string{"single axes", "axis pairs", "stages"}

	for round, level := 1, 0; round <= 30 && level < len(moves); round++ {
		leader := kept[0].h.String()

		for _, keys := range moves[level] {
			kept = searchStep(t, ev, "", kept, keys, ref, beam, 0)
		}

		leaderboard(t, fmt.Sprintf("coordinate descent round %d, %s", round, names[level]), kept, ref, top)

		if kept[0].h.String() != leader {
			level = 0
		} else {
			level++
		}
	}

	best := kept[0].h
	t.Logf("leader: %s", best)
	t.Logf("leader vs Best(): %s", diff(best, Best()))

	sensitivity(t, ev, best)

	for _, name := range []string{"cact", "safetensors"} {
		path := whistleCactPath
		if name == "safetensors" {
			path = whistleSafetensorsPath
		}

		full := ev
		if name != srcName {
			full = newEvaluator(t, openModel(t, path), tok, clips)
		}

		r := full.scoreAll([]Hypothesis{best})[0]
		t.Logf("leader, %s weights: logp %.4f min %.4f top1 %.4f lang %d/%d rank %.2f", name, r.mean, r.min,
			r.top1, r.langOK, len(r.perClip), r.langRank)
	}
}

// searchStep scores the product of keys around every kept hypothesis and
// returns the best beam; it prints a leaderboard unless title is empty.
func searchStep(t *testing.T, ev *evaluator, title string, kept []result, keys []string, ref Hypothesis,
	beam, top int,
) []result {
	t.Helper()

	cand := make([]Hypothesis, 0, len(kept))
	for _, r := range kept {
		cand = append(cand, product(r.h, keys)...)
	}

	rs := ev.scoreAll(dedupe(cand))
	rank(rs)

	if title != "" {
		leaderboard(t, title, rs, ref, top)
	}

	n := 0
	for n < len(rs) && n < beam && rs[n].ok() {
		n++
	}

	if n == 0 {
		t.Fatalf("%s: every hypothesis failed", title)
	}

	return rs[:n]
}

// sensitivity prints, for every axis, the score of each value with all
// other axes at best: what each alternative costs.
func sensitivity(t *testing.T, ev *evaluator, best Hypothesis) {
	t.Helper()

	t.Logf("── sensitivity of the leader (Δ logp = value's − leader's, top-1, language wins)")

	base := ev.scoreAll([]Hypothesis{best})[0]

	for _, a := range axes {
		rs := ev.scoreAll(product(best, []string{a.Key}))

		var parts []string

		for i, r := range rs {
			v := a.Names[i]

			switch {
			case !r.ok():
				parts = append(parts, v+" failed")
			case i+1 == a.Get(best):
				parts = append(parts, fmt.Sprintf("[%s %.4f %.3f %d]", v, r.mean, r.top1, r.langOK))
			default:
				parts = append(parts, fmt.Sprintf("%s %+.4f %.3f %d", v, r.mean-base.mean, r.top1, r.langOK))
			}
		}

		t.Logf("  %-8s %s", a.Key, strings.Join(parts, "  "))
	}
}

// TestProbeScore scores WHISTLE_PROBE_KEY (default: Best) with both weight
// sources and prints the per-clip numbers.
func TestProbeScore(t *testing.T) {
	tok := cactTokenizer(t)
	h := probeStart(t, Best())
	clips := loadClips(t)

	for _, name := range []string{"cact", "safetensors"} {
		path := whistleCactPath
		if name == "safetensors" {
			path = whistleSafetensorsPath
		}

		r := newEvaluator(t, openModel(t, path), tok, clips).scoreAll([]Hypothesis{h})[0]
		if !r.ok() {
			t.Fatalf("%s: %s", h, r.ruled)
		}

		t.Logf("%s weights: %s", name, h)

		for i, c := range clips {
			t.Logf("  %-38s logp %.4f", c.ID, r.perClip[i])
		}

		t.Logf("  over %d clips: logp %.4f min %.4f top1 %.4f lang %d/%d rank %.2f", len(clips), r.mean, r.min,
			r.top1, r.langOK, len(clips), r.langRank)
	}
}

// greedyRow is one clip's greedy transcript against the engine's.
type greedyRow struct {
	c        clip
	r        Result
	wer, cer float64
}

// TestProbeGreedy decodes every clip greedily with WHISTLE_PROBE_KEY
// (default: Best), with language detection, on both weight sources, and
// prints the per-clip language, WER and CER against the engine's
// transcript, then the differing transcripts side by side.
func TestProbeGreedy(t *testing.T) {
	tok := cactTokenizer(t)
	h := probeStart(t, Best())
	clips := loadClips(t)

	for _, name := range []string{"cact", "safetensors"} {
		path := whistleCactPath
		if name == "safetensors" {
			path = whistleSafetensorsPath
		}

		src := openModel(t, path)

		d, err := Load(src, tok, h)
		if err != nil {
			t.Fatal(err)
		}

		encs, frames := encode(t, src, clips, h.Memory)
		rows := make([]greedyRow, len(clips))

		var wg sync.WaitGroup

		for i, c := range clips {
			wg.Go(func() {
				r, err := d.Greedy(encs[i], frames[i], "")
				if err != nil {
					t.Errorf("%s: %v", c.ID, err)

					return
				}

				rows[i] = greedyRow{c: c, r: r, wer: wer(c.Text, r.Text), cer: cer(c.Text, r.Text)}
			})
		}

		wg.Wait()
		logGreedy(t, name, h, rows)
	}
}

// logGreedy prints the per-clip table, the summary and the differing
// transcripts.
func logGreedy(t *testing.T, name string, h Hypothesis, rows []greedyRow) {
	t.Helper()

	t.Logf("── greedy, %s weights: %s", name, h)
	t.Logf("| clip | engine | ours | WER | CER | exact | EOS |")
	t.Logf("| ---- | ------ | ---- | --- | --- | ----- | --- |")

	var langOK, exact, eos int

	wers, cers := make([]float64, 0, len(rows)), make([]float64, 0, len(rows))

	for _, g := range rows {
		same := g.c.Text == g.r.Text
		if same {
			exact++
		}

		if g.c.Language == g.r.Language {
			langOK++
		}

		if g.r.EOS {
			eos++
		}

		wers, cers = append(wers, g.wer), append(cers, g.cer)
		t.Logf("| %s | %s | %s | %.3f | %.3f | %v | %v |", g.c.ID, g.c.Language, g.r.Language, g.wer, g.cer, same,
			g.r.EOS)
	}

	t.Logf("%s: language %d/%d, exact %d/%d, EOS %d/%d, mean WER %.4f (max %.4f), mean CER %.4f (max %.4f)",
		name, langOK, len(rows), exact, len(rows), eos, len(rows), mean(wers), slices.Max(wers), mean(cers),
		slices.Max(cers))

	for _, g := range rows {
		if g.c.Text != g.r.Text {
			t.Logf("%s\n    engine: %s\n    ours:   %s", g.c.ID, g.c.Text, g.r.Text)
		}
	}
}
