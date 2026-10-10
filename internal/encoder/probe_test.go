//go:build oracle

// The encoder probe harness (PLAN.md Phase 4, docs/FINDINGS.md F-014). It
// scores Hypothesis values by the mean per-frame cosine of the encoder
// output against the engine's committed needle_embed rows and searches the
// axes stage by stage with a beam, then by coordinate descent over single
// axes, axis pairs and whole stages. Run it with
// `just encoder-probe`; the knobs are environment variables:
//
//	WHISTLE_PROBE_SRC    cact (default) or safetensors: the weights the search runs on
//	WHISTLE_PROBE_BEAM   hypotheses kept per stage (default 4)
//	WHISTLE_PROBE_TOP    leaderboard rows per stage (default 8)
//	WHISTLE_PROBE_CLIPS  inner-loop clips (default 6, at most 8)
//	WHISTLE_PROBE_KEY    TestProbeScore: the Hypothesis.String to score on all clips

package encoder

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// innerClips are the inner-loop clips: short ones from both LibriSpeech
// splits and from FLEURS.
var innerClips = []string{
	"ls-other-8280-266249-0032",
	"ls-other-3080-5040-0018",
	"ls-clean-5639-40744-0011",
	"ls-clean-6930-75918-0007",
	"fleurs-nl_nl-1004717084892689357",
	"fleurs-es_419-10149016836336603986",
	"ls-clean-260-123440-0007",
	"fleurs-it_it-10210638754804006465",
}

// result is one hypothesis' score.
type result struct {
	h       Hypothesis
	perClip []float64   // mean per-frame cosine per clip
	frames  [][]float64 // per-frame cosines per clip
	mean    float64     // mean over clips
	min     float64     // worst clip
	ruled   string      // why the hypothesis was excluded without a score
}

func (r result) ok() bool { return r.ruled == "" }

// evaluator scores hypotheses on a clip set with one weight source,
// caching the log-mel features and the stem output.
type evaluator struct {
	src   weights.Source
	clips []clip

	mu    sync.Mutex
	mels  map[string]frontend.Features // front end key | clip
	stems map[string][]float32         // stem key | clip
	done  map[string]result            // Hypothesis.String
}

func newEvaluator(src weights.Source, clips []clip) *evaluator {
	return &evaluator{
		src: src, clips: clips,
		mels: map[string]frontend.Features{}, stems: map[string][]float32{}, done: map[string]result{},
	}
}

// scoreAll scores hs, each hypothesis once per evaluator.
func (ev *evaluator) scoreAll(hs []Hypothesis) []result {
	var fresh []Hypothesis

	for _, h := range hs {
		if _, ok := ev.done[h.String()]; !ok {
			fresh = append(fresh, h)
		}
	}

	for _, r := range ev.scoreFresh(dedupe(fresh)) {
		ev.done[r.h.String()] = r
	}

	out := make([]result, len(hs))
	for i, h := range hs {
		out[i] = ev.done[h.String()]
	}

	return out
}

// job is one (hypothesis, clip) pair.
type job struct{ h, c int }

// scoreFresh scores hs, running the (hypothesis, clip) pairs on all cores.
// A hypothesis whose shapes or frame count disagree with the engine is
// ruled out without a score.
func (ev *evaluator) scoreFresh(hs []Hypothesis) []result {
	out := make([]result, len(hs))
	encs := make([]*Encoder, len(hs))

	for i, h := range hs {
		out[i] = result{h: h, perClip: make([]float64, len(ev.clips)), frames: make([][]float64, len(ev.clips))}

		e, err := Load(ev.src, h)
		if err != nil {
			out[i].ruled = err.Error()

			continue
		}

		for _, c := range ev.clips {
			if got := e.Frames(len(c.Samples)); got != c.Frames {
				out[i].ruled = fmt.Sprintf("frame count: %s gives %d frames, engine %d", c.ID, got, c.Frames)

				break
			}
		}

		if out[i].ok() {
			encs[i] = e
		}
	}

	jobs := make(chan job)

	var (
		wg      sync.WaitGroup
		errOnce sync.Once
		runErr  error
	)

	for range runtime.GOMAXPROCS(0) {
		wg.Go(func() {
			for j := range jobs {
				cos, err := ev.run(encs[j.h], j.c)
				if err != nil {
					errOnce.Do(func() { runErr = fmt.Errorf("%s on %s: %w", hs[j.h], ev.clips[j.c].ID, err) })

					continue
				}

				out[j.h].frames[j.c] = cos
				out[j.h].perClip[j.c] = mean(cos)
			}
		})
	}

	for i, e := range encs {
		if e == nil {
			continue
		}

		for c := range ev.clips {
			jobs <- job{i, c}
		}
	}

	close(jobs)
	wg.Wait()

	if runErr != nil {
		panic(runErr)
	}

	for i := range out {
		if out[i].ok() {
			out[i].mean, out[i].min = mean(out[i].perClip), slices.Min(out[i].perClip)
		}
	}

	return out
}

// run encodes clip c with e and returns its per-frame cosines.
func (ev *evaluator) run(e *Encoder, c int) ([]float64, error) {
	cl := ev.clips[c]
	h := e.Hypothesis()

	melKey := h.Frontend.String() + "|" + cl.ID

	ev.mu.Lock()
	mel, ok := ev.mels[melKey]
	ev.mu.Unlock()

	if !ok {
		var err error

		mel, err = e.fe.LogMel(cl.Samples)
		if err != nil {
			return nil, err
		}

		ev.mu.Lock()
		ev.mels[melKey] = mel
		ev.mu.Unlock()
	}

	// The stem output depends on the front end, the stem axes and a
	// stem-side pe_gate signal.
	stemKey := h.StemKey() + " pe=" + strconv.Itoa(int(h.PE)) + "|" + cl.ID

	ev.mu.Lock()
	x, ok := ev.stems[stemKey]
	ev.mu.Unlock()

	if !ok {
		var err error

		x, _, err = e.Stem(mel)
		if err != nil {
			return nil, err
		}

		ev.mu.Lock()
		ev.stems[stemKey] = x
		ev.mu.Unlock()
	}

	y, err := e.Stack(x, cl.Frames)
	if err != nil {
		return nil, err
	}

	return frameCosines(y, cl.Embed, cl.Frames, e.dim), nil
}

// probeEnv reads an integer knob.
func probeEnv(name string, def int) int {
	v, err := strconv.Atoi(os.Getenv(name))
	if err == nil && v > 0 {
		return v
	}

	return def
}

// probeSource opens the weights named by WHISTLE_PROBE_SRC.
func probeSource(t *testing.T) (weights.Source, string) {
	t.Helper()

	if os.Getenv("WHISTLE_PROBE_SRC") == "safetensors" {
		return openModel(t, whistleSafetensorsPath), "safetensors"
	}

	return openModel(t, whistleCactPath), "cact"
}

// prior is the starting hypothesis: the shapes' and upstream's strongest
// readings (docs/FINDINGS.md F-014).
func prior() Hypothesis {
	return Hypothesis{
		Frontend:   frontend.NeMo(),
		StemStride: StrideAll, StemPad: PadSame, StemKernel: KernelTimeMel, StemFlatten: FlattenMelChannel,
		StemConvAct: ActSiLU, StemDWAct: ActNone, StemPWAct: ActSiLU, InputScale: ScaleOne,
		PE: PENone, Rope: RopeNone,
		FFNScale: FFNHalf, GLU: GLUValueGate, ConvAct: ActSiLU, ConvNorm: NormBeforeAct, DWPad: DWCentred,
		Order: OrderMacaron, MHC: MHCLanes, HadaPerms: PermsShared, Output: OutFinalNorm,
	}
}

// TestProbeBaselines prints what a meaningless score looks like: the
// cosine of the engine frames with the mean engine frame of the other
// clips, and with a random projection of the stacked log-mel frames.
func TestProbeBaselines(t *testing.T) {
	clips := loadClips(t)
	src := openModel(t, whistleCactPath)

	bank, err := frontend.LoadMelBank(src)
	if err != nil {
		t.Fatal(err)
	}

	fe, err := frontend.New(bank, frontend.NeMo())
	if err != nil {
		t.Fatal(err)
	}

	const (
		dim   = 512
		width = 8 * frontend.Mels
	)

	rng := rand.New(rand.NewPCG(1, 2))

	proj := make([]float32, dim*width)
	for i := range proj {
		proj[i] = float32(rng.NormFloat64())
	}

	meanBase, projBase := make([]float64, 0, len(clips)), make([]float64, 0, len(clips))

	for i, c := range clips {
		avg := make([]float32, dim)

		for j, o := range clips {
			if j != i {
				for k, v := range o.Embed {
					avg[k%dim] += v
				}
			}
		}

		rep := make([]float32, c.Frames*dim)
		for f := range c.Frames {
			copy(rep[f*dim:], avg)
		}

		mel, err := fe.LogMel(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		rp := make([]float32, c.Frames*dim)

		for f := range c.Frames {
			for k := range dim {
				var s float32

				for m := range width {
					if idx := f*width + m; idx < len(mel.Data) {
						s += proj[k*width+m] * mel.Data[idx]
					}
				}

				rp[f*dim+k] = s
			}
		}

		mb := mean(frameCosines(rep, c.Embed, c.Frames, dim))
		pb := mean(frameCosines(rp, c.Embed, c.Frames, dim))

		meanBase, projBase = append(meanBase, mb), append(projBase, pb)
		t.Logf("%-40s frames %3d  mean-frame %.4f  random-projection %+.4f", c.ID, c.Frames, mb, pb)
	}

	t.Logf("baseline mean-frame (leave one clip out): mean %.4f min %.4f max %.4f",
		mean(meanBase), slices.Min(meanBase), slices.Max(meanBase))
	t.Logf("baseline random projection of 8 stacked mel frames: mean %+.4f min %+.4f max %+.4f",
		mean(projBase), slices.Min(projBase), slices.Max(projBase))
}

// TestProbeScore scores WHISTLE_PROBE_KEY (default: the prior) on every
// clip with both weight sources and prints per-clip and edge detail.
func TestProbeScore(t *testing.T) {
	h := prior()

	if key := os.Getenv("WHISTLE_PROBE_KEY"); key != "" {
		var err error

		h, err = ParseHypothesis(key)
		if err != nil {
			t.Fatal(err)
		}
	}

	clips := loadClips(t)

	for _, name := range []string{"cact", "safetensors"} {
		path := whistleCactPath
		if name == "safetensors" {
			path = whistleSafetensorsPath
		}

		r := newEvaluator(openModel(t, path), clips).scoreAll([]Hypothesis{h})[0]
		if !r.ok() {
			t.Fatalf("%s: ruled out: %s", h, r.ruled)
		}

		t.Logf("%s weights: %s", name, h)
		logClips(t, r, clips)
	}
}

// logClips prints a result's per-clip numbers, the edge frames apart.
func logClips(t *testing.T, r result, clips []clip) {
	t.Helper()

	for i, c := range clips {
		f := r.frames[i]

		inner := f
		if len(f) > 4 {
			inner = f[2 : len(f)-2]
		}

		t.Logf("  %-40s mean %.4f min %.4f max %.4f  first %.3f last %.3f inner %.4f", c.ID, r.perClip[i],
			slices.Min(f), slices.Max(f), f[0], f[len(f)-1], mean(inner))
	}

	t.Logf("  over %d clips: mean %.4f min %.4f max %.4f", len(clips), r.mean, r.min, slices.Max(r.perClip))
}

// probeAxis is one searchable axis, front end or encoder, as its key and
// value names in Hypothesis.String.
type probeAxis struct {
	key    string
	values []string
}

// probeAxes lists every axis: the front end's from frontend.Variants, then
// Axes().
func probeAxes() []probeAxis {
	var out []probeAxis

	seen := map[string]map[string]bool{}

	for _, v := range frontend.Variants() {
		for tok := range strings.FieldsSeq(v.String()) {
			k, val, _ := strings.Cut(tok, "=")
			if seen[k] == nil {
				seen[k] = map[string]bool{}
				out = append(out, probeAxis{key: k})
			}

			if !seen[k][val] {
				seen[k][val] = true

				for i := range out {
					if out[i].key == k {
						out[i].values = append(out[i].values, val)
					}
				}
			}
		}
	}

	for _, a := range Axes() {
		out = append(out, probeAxis{key: a.Key, values: a.Names})
	}

	return out
}

// axisByKey finds an axis.
func axisByKey(key string) probeAxis {
	for _, a := range probeAxes() {
		if a.key == key {
			return a
		}
	}

	panic("probe: unknown axis " + key)
}

// withValue returns h with axis key set to value.
func withValue(h Hypothesis, key, value string) Hypothesis {
	toks := strings.Fields(h.String())
	for i, tok := range toks {
		if k, _, _ := strings.Cut(tok, "="); k == key {
			toks[i] = key + "=" + value
		}
	}

	out, err := ParseHypothesis(strings.Join(toks, " "))
	if err != nil {
		panic(err)
	}

	return out
}

// valueOf returns the value name of axis key in h.
func valueOf(h Hypothesis, key string) string {
	for tok := range strings.FieldsSeq(h.String()) {
		if k, v, _ := strings.Cut(tok, "="); k == key {
			return v
		}
	}

	return ""
}

// product returns every combination of the given axes around h.
func product(h Hypothesis, keys []string) []Hypothesis {
	out := []Hypothesis{h}

	for _, k := range keys {
		var next []Hypothesis

		for _, base := range out {
			for _, v := range axisByKey(k).values {
				next = append(next, withValue(base, k, v))
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

// rank sorts results best first: scored before ruled out, then by mean.
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

// leaderboard prints the top rows of ranked results relative to ref, and
// how many were ruled out without a score.
func leaderboard(t *testing.T, title string, rs []result, ref Hypothesis, top int) {
	t.Helper()

	ruled := 0

	for _, r := range rs {
		if !r.ok() {
			ruled++
		}
	}

	t.Logf("── %s: %d hypotheses, %d ruled out without a score", title, len(rs), ruled)

	for i, r := range rs {
		if i == top || !r.ok() {
			break
		}

		t.Logf("  #%-2d mean %.6f min %.6f max %.6f | %s", i+1, r.mean, r.min, slices.Max(r.perClip), diff(r.h, ref))
	}
}

// stage is a set of axes searched jointly.
type stage struct {
	name string
	keys []string
}

// searchStages is the documented search order (docs/FINDINGS.md F-014):
// the conv stem first, because every later score depends on what it
// feeds; then the block's conv module, the residual structure and the
// positions; the front end last but one, then the stem's activations
// again with everything else in place.
var searchStages = []stage{
	{"stem geometry", []string{"stride", "stempad", "kernel", "flatten"}},
	{"stem activations", []string{"convact", "dwact", "pwact", "scale"}},
	{"front end: log, norm, spectrum", []string{"log", "norm", "spec"}},
	{"conv module", []string{"glu", "blockact", "convnorm", "dwpad"}},
	{"block structure", []string{"ffn", "order", "mhc", "perms"}},
	{"positions and output", []string{"rope", "pe", "out"}},
	{"front end: window, padding, band, pre-emphasis, dither", []string{"win", "pad", "band", "pre", "dither"}},
}

// TestProbeSearch is the staged beam search, then the escalating
// coordinate descent until the leader stops changing, then the leader's
// per-axis sensitivity and its scores on all clips with both weight
// sources (docs/FINDINGS.md F-014).
func TestProbeSearch(t *testing.T) {
	src, srcName := probeSource(t)
	beam := probeEnv("WHISTLE_PROBE_BEAM", 4)
	top := probeEnv("WHISTLE_PROBE_TOP", 8)
	nClips := min(probeEnv("WHISTLE_PROBE_CLIPS", 6), len(innerClips))

	ev := newEvaluator(src, loadClips(t, innerClips[:nClips]...))
	ref := prior()

	if key := os.Getenv("WHISTLE_PROBE_KEY"); key != "" {
		var err error

		ref, err = ParseHypothesis(key)
		if err != nil {
			t.Fatal(err)
		}
	}

	t.Logf("weights %s, %d inner clips, beam %d; reference: %s", srcName, nClips, beam, ref)

	kept := ev.scoreAll([]Hypothesis{ref})
	leaderboard(t, "reference", kept, ref, 1)

	for _, st := range searchStages {
		kept = searchStep(t, ev, st.name, kept, st.keys, ref, beam, top)
	}

	// Coordinate descent with escalating moves around every kept
	// hypothesis: each axis on its own; when that leaves the leader
	// unchanged, every pair of axes; when that does too, every stage's joint
	// product again. Coupled axes (the stem's input scale with its
	// activations, say) can trap smaller moves. Any move that changes the
	// leader restarts from single axes; the search stops when none does.
	all := probeAxes()
	singles := make([][]string, 0, len(all))
	pairs := make([][]string, 0, len(all)*(len(all)-1)/2)
	stages := make([][]string, 0, len(searchStages))

	for i := range all {
		singles = append(singles, []string{all[i].key})

		for j := i + 1; j < len(all); j++ {
			pairs = append(pairs, []string{all[i].key, all[j].key})
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

	sensitivity(t, ev, best)

	clips := loadClips(t)

	for _, name := range []string{"cact", "safetensors"} {
		path := whistleCactPath
		if name == "safetensors" {
			path = whistleSafetensorsPath
		}

		full := newEvaluator(openModel(t, path), clips)
		rs := full.scoreAll(hypotheses(kept))
		rank(rs)
		leaderboard(t, "kept hypotheses on all clips, "+name+" weights", rs, ref, top)
		t.Logf("leader on all clips, %s weights:", name)
		logClips(t, full.scoreAll([]Hypothesis{best})[0], clips)
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
		leaderboard(t, "stage "+title+" ("+strings.Join(keys, ", ")+")", rs, ref, top)
	}

	n := 0
	for n < len(rs) && n < beam && rs[n].ok() {
		n++
	}

	if n == 0 {
		t.Fatalf("stage %s: every hypothesis ruled out", title)
	}

	return rs[:n]
}

func hypotheses(rs []result) []Hypothesis {
	out := make([]Hypothesis, len(rs))
	for i, r := range rs {
		out[i] = r.h
	}

	return out
}

// sensitivity prints, for every axis, the score of each value with all
// other axes at best: what each alternative costs, or why it is ruled out.
func sensitivity(t *testing.T, ev *evaluator, best Hypothesis) {
	t.Helper()

	t.Logf("── sensitivity of the leader (Δ = value's mean − leader's mean, inner clips)")

	base := ev.scoreAll([]Hypothesis{best})[0]

	for _, a := range probeAxes() {
		rs := ev.scoreAll(product(best, []string{a.key}))

		var parts []string

		for _, r := range rs {
			v := valueOf(r.h, a.key)

			switch {
			case !r.ok():
				parts = append(parts, v+" ruled out")
			case v == valueOf(best, a.key):
				parts = append(parts, fmt.Sprintf("[%s %.6f]", v, r.mean))
			default:
				parts = append(parts, fmt.Sprintf("%s %+.6f", v, r.mean-base.mean))
			}
		}

		t.Logf("  %-9s %s", a.key, strings.Join(parts, "  "))
	}
}

// TestProbeSweep prints the per-axis sensitivity around WHISTLE_PROBE_KEY
// (default: the prior) without searching.
func TestProbeSweep(t *testing.T) {
	src, srcName := probeSource(t)
	nClips := min(probeEnv("WHISTLE_PROBE_CLIPS", 6), len(innerClips))
	h := prior()

	if key := os.Getenv("WHISTLE_PROBE_KEY"); key != "" {
		var err error

		h, err = ParseHypothesis(key)
		if err != nil {
			t.Fatal(err)
		}
	}

	t.Logf("weights %s, %d inner clips: %s", srcName, nClips, h)
	sensitivity(t, newEvaluator(src, loadClips(t, innerClips[:nClips]...)), h)
}

// family is a probe's kind, the part of its name before the first dash.
func family(id string) string {
	f, _, _ := strings.Cut(id, "-")

	return f
}

// probeScore is one hypothesis on the probe set and on the corpus.
type probeScore struct {
	h                    Hypothesis
	silence, probes, all result
	families             map[string]float64 // mean per probe family
}

// scoreProbes scores hs on the silence probes, on all probes and on all
// corpus clips with the .cact weights.
func scoreProbes(t *testing.T, hs []Hypothesis) []probeScore {
	t.Helper()

	src := openModel(t, whistleCactPath)
	probes := loadProbes(t)

	var silence []clip

	for _, p := range probes {
		if family(p.ID) == "silence" {
			silence = append(silence, p)
		}
	}

	sil := newEvaluator(src, silence).scoreAll(hs)
	all := newEvaluator(src, probes).scoreAll(hs)
	corpus := newEvaluator(src, loadClips(t)).scoreAll(hs)

	out := make([]probeScore, len(hs))

	for i, h := range hs {
		out[i] = probeScore{h: h, silence: sil[i], probes: all[i], all: corpus[i], families: map[string]float64{}}

		if !all[i].ok() {
			continue
		}

		n := map[string]int{}

		for c, p := range probes {
			out[i].families[family(p.ID)] += all[i].perClip[c]
			n[family(p.ID)]++
		}

		for f := range n {
			out[i].families[f] /= float64(n[f])
		}
	}

	return out
}

// logProbeScores prints scores sorted by key, best first, relative to
// ref's values.
func logProbeScores(t *testing.T, title string, ps []probeScore, ref Hypothesis, key func(probeScore) float64) {
	t.Helper()

	slices.SortStableFunc(ps, func(a, b probeScore) int { return cmp.Compare(key(b), key(a)) })

	fams := []string{"silence", "dc", "tone", "burst", "noise", "decay", "impulse"}
	t.Logf("── %s (.cact; probe means per family: %s)", title, strings.Join(fams, " "))

	for i, p := range ps {
		if !p.probes.ok() {
			t.Logf("  #%-2d ruled out: %s | %s", i+1, p.probes.ruled, diff(p.h, ref))

			continue
		}

		fm := make([]string, 0, len(fams))
		for _, f := range fams {
			fm = append(fm, fmt.Sprintf("%.5f", p.families[f]))
		}

		corpus := "32 clips ruled out (frame count)"
		if p.all.ok() {
			corpus = fmt.Sprintf("32 clips %.6f (min %.6f)", p.all.mean, p.all.min)
		}

		t.Logf("  #%-2d silence %.6f (min %.6f) | probes %.6f (min %.6f) | %s | %s | %s",
			i+1, p.silence.mean, p.silence.min, p.probes.mean, p.probes.min, corpus, strings.Join(fm, " "),
			diff(p.h, ref))
	}
}

// TestProbePEGate scores every pe_gate placement, each with and without
// the encoder's RoPE, on the silence probes, where the stem output is 0
// and any additive position term is the stack's only input, on all
// probes and on the 32 clips (docs/FINDINGS.md F-019).
func TestProbePEGate(t *testing.T) {
	ref := Best()
	ps := scoreProbes(t, product(ref, []string{"pe", "rope"}))

	logProbeScores(t, "pe_gate placements × RoPE, sorted by the silence probes", ps, ref,
		func(p probeScore) float64 { return p.silence.mean })

	for _, p := range ps {
		if p.h.Rope != ref.Rope {
			continue
		}

		var parts []string
		for c, cl := range loadProbes(t)[:len(p.silence.perClip)] {
			parts = append(parts, fmt.Sprintf("%s %.6f", strings.TrimPrefix(cl.ID, "silence-"), p.silence.perClip[c]))
		}

		t.Logf("  silence per length, pe=%s: %s", valueOf(p.h, "pe"), strings.Join(parts, "  "))
	}
}

// TestProbeFrontEnd sweeps the log floor, the window, the centring and
// dither around Best on the probes and the 32 clips (docs/FINDINGS.md
// F-020).
func TestProbeFrontEnd(t *testing.T) {
	ref := Best()
	ps := scoreProbes(t, product(ref, []string{"log", "win", "pad", "dither"}))

	logProbeScores(t, "front end: log × window × centring × dither, sorted by the 32 clips", ps, ref,
		func(p probeScore) float64 { return p.all.mean })
}

// TestProbeScaleInvariance compares the engine's rows for the same probe
// at different levels: a front end without an absolute floor (log, a clamp
// relative to the peak, per-channel normalisation) gives the same rows at
// any gain (docs/FINDINGS.md F-020).
func TestProbeScaleInvariance(t *testing.T) {
	ps := map[string]clip{}
	for _, p := range loadProbes(t) {
		ps[p.ID] = p
	}

	for _, pair := range [][2]string{
		{"tone-1e-01", "tone-1e-03"},
		{"tone-1e-01", "tone-1e-05"},
		{"burst-1e-01", "burst-1e-03"},
		{"burst-1e-01", "burst-1e-05"},
		{"noise-1e-01", "noise-1e-05"},
		{"dc-1e-02", "dc-1e-04"},
		{"impulse-d0-1e-01", "impulse-d0-1e-02"},
		{"impulse-d0-1e-01", "impulse-d0-1e-04"},
		{"impulse-d0-1e-01", "impulse-d0-1e-06"},
		{"impulse-d80-1e-01", "impulse-d80-1e-03"},
		{"impulse-d80-1e-01", "impulse-d80-1e-05"},
		{"impulse-d80-1e-01", "impulse-d80-1e-06"},
		{"silence-32000", "impulse-d80-1e-06"},
		{"silence-32000", "impulse-d0-1e-06"},
	} {
		a, b := ps[pair[0]], ps[pair[1]]
		cos := frameCosines(a.Embed, b.Embed, a.Frames, 512)
		t.Logf("engine %-17s vs %-17s mean %.6f min %.6f", a.ID, b.ID, mean(cos), slices.Min(cos))
	}
}

// TestProbeLogKnee sweeps the depth of the log's floor below the clip's
// peak, as a soft knee log10(x + 10^−d·max x) and as Whisper's hard clamp
// max(log10 x, log10 max x − d), on the 32 clips, then fits the soft depth
// on half the clips and scores it on the other half (docs/FINDINGS.md
// F-020). Every other axis is Best's.
func TestProbeLogKnee(t *testing.T) {
	src := openModel(t, whistleCactPath)
	clips := loadClips(t)

	e, err := Load(src, Best())
	if err != nil {
		t.Fatal(err)
	}

	bank, _, err := src.Tensor(frontend.MelBankName)
	if err != nil {
		t.Fatal(err)
	}

	energies := make([][]float64, len(clips))

	for i, c := range clips {
		spec, frames, err := e.fe.Spectrogram(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		energies[i] = make([]float64, frames*frontend.Mels)

		for f := range frames {
			for m := range frontend.Mels {
				var s float64
				for k := range frontend.Bins {
					s += float64(spec[f*frontend.Bins+k]) * float64(bank[k*frontend.Mels+m])
				}

				energies[i][f*frontend.Mels+m] = s
			}
		}
	}

	score := func(soft bool, depth float64) []float64 {
		per := make([]float64, len(clips))

		for i, c := range clips {
			mel := slices.Clone(energies[i])
			top := math.Log10(slices.Max(mel))

			for j, x := range mel {
				if soft {
					mel[j] = math.Log10(x + math.Pow(10, top-depth))
				} else {
					mel[j] = max(math.Log10(max(x, 1e-30)), top-depth)
				}
			}

			frames := len(mel) / frontend.Mels
			for m := range frontend.Mels {
				standardiseColumn(mel, m, frames)
			}

			ft := frontend.Features{Frames: frames, Data: make([]float32, len(mel))}
			for j, v := range mel {
				ft.Data[j] = float32(v)
			}

			y, n, err := e.Forward(ft)
			if err != nil {
				t.Fatal(err)
			}

			per[i] = mean(frameCosines(y, c.Embed, n, 512))
		}

		return per
	}

	var depths []float64
	for d := 7.3; d < 8.25; d += 0.05 {
		depths = append(depths, math.Round(d*100)/100)
	}

	soft := make([][]float64, len(depths))

	for k, d := range depths {
		soft[k] = score(true, d)
		hard := score(false, d)
		t.Logf("depth %.2f: soft knee mean %.6f min %.6f | hard clamp mean %.6f min %.6f",
			d, mean(soft[k]), slices.Min(soft[k]), mean(hard), slices.Min(hard))
	}

	hard8 := score(false, 8)

	// Fit the soft depth on one half, score it on the other.
	var even, odd, ls, fleurs []int

	for i, c := range clips {
		if i%2 == 0 {
			even = append(even, i)
		} else {
			odd = append(odd, i)
		}

		if strings.HasPrefix(c.ID, "ls-") {
			ls = append(ls, i)
		} else {
			fleurs = append(fleurs, i)
		}
	}

	sub := func(v []float64, idx []int) []float64 {
		out := make([]float64, len(idx))
		for j, i := range idx {
			out[j] = v[i]
		}

		return out
	}

	for _, split := range []struct {
		name       string
		fit, check []int
	}{{"even → odd", even, odd}, {"odd → even", odd, even}, {"ls → fleurs", ls, fleurs}, {"fleurs → ls", fleurs, ls}} {
		bestK := 0
		for k := range depths {
			if mean(sub(soft[k], split.fit)) > mean(sub(soft[bestK], split.fit)) {
				bestK = k
			}
		}

		got, ref := sub(soft[bestK], split.check), sub(hard8, split.check)
		t.Logf("cross-validation %-11s fit depth %.2f on %2d clips: held out mean %.6f min %.6f, hard clamp 8 mean %.6f min %.6f",
			split.name, depths[bestK], len(split.fit), mean(got), slices.Min(got), mean(ref), slices.Min(ref))
	}
}

// standardiseColumn is NormChannel on mel channel m of frames rows:
// (x − mean) / (std + 1e-5), std with n − 1.
func standardiseColumn(mel []float64, m, frames int) {
	var s float64
	for f := range frames {
		s += mel[f*frontend.Mels+m]
	}

	mu := s / float64(frames)

	var d float64

	for f := range frames {
		v := mel[f*frontend.Mels+m] - mu
		d += v * v
	}

	sd := 0.0
	if frames > 1 {
		sd = math.Sqrt(d / float64(frames-1))
	}

	for f := range frames {
		mel[f*frontend.Mels+m] = (mel[f*frontend.Mels+m] - mu) / (sd + 1e-5)
	}
}
