//go:build oracle

package whistle

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/eval"
	"github.com/cwbudde/go-whistle/internal/silence"
)

// Every stream of testdata/oracle/stream, call by call (PLAN.md 8.3, 8.4,
// docs/FINDINGS.md F-036), and the Phase 8 Accept: the final committed text
// at 1 s chunks against the offline transcript. `just stream-parity`.
func TestStreamMatchesEngineAll(t *testing.T) {
	m := loadModel(t)
	sets, runs := replayStreams(t, m)

	var total streamTally

	for _, r := range runs {
		total.add(r)

		if r.firstDiff >= 0 {
			t.Logf("%s: %d of %d calls differ, the first:\n%s", r.run.Name, len(r.diffs), len(r.calls), r.diffs[0])
		}
	}

	names := make([]string, 0, len(sets))
	for name := range sets {
		names = append(names, name)
	}

	slices.Sort(names)

	for _, name := range names {
		t.Logf("%s: %v", name, sets[name])
	}

	t.Logf("all: %v", &total)
}

// The first call of every run whose decision differs from the engine's,
// explained (F-036): the pass under our model spells other words than the
// engine's (by how much the engine's spelling scores below ours), the same
// words but other times (so another word is dropped as committed), or the
// silence gate.
func TestStreamDivergences(t *testing.T) {
	m := loadModel(t)
	runs, calls := loadStreamFixtures(t)
	cache := map[string][]float32{}
	kinds := map[string]int{}

	var margins []float64

	for _, r := range runs {
		fx := calls[r.Name]
		x := streamSamples(t, r, cache)
		res := replayStream(t, m, r, fx, x)

		if res.firstDiff < 0 || res.firstDiff >= len(fx.Calls) {
			continue
		}

		d := explainDivergence(t, m, r, fx, x, res.firstDiff)
		kinds[d.kind]++

		if d.kind == "pass" {
			margins = append(margins, d.margin)
		}

		t.Logf("%s call %d: %s", r.Name, res.firstDiff, d)
	}

	slices.Sort(margins)
	t.Logf("first divergences: %v; pass margins (ours − engine's, nats per token, canonical spelling): %.4f", kinds, margins)
}

// divergence is why one call differs.
type divergence struct {
	kind            string // pass, times, silence, language, bookkeeping
	ours, engine    string // the words after the committed ones
	margin, sum     float64
	tokens, eTokens int
}

func (d divergence) String() string {
	s := fmt.Sprintf("%s: ours %q, engine %q", d.kind, d.ours, d.engine)
	if d.kind == "pass" {
		s += fmt.Sprintf("; score ours − engine %.4f nats/token, Σ log p %+.3f (%d vs %d tokens)", d.margin, d.sum,
			d.tokens, d.eTokens)
	}

	return s
}

// explainDivergence replays r up to call i and compares that call's pass
// with the engine's.
func explainDivergence(t *testing.T, m *Model, r streamFixtureRun, fx streamFixture, x []float32, i int) divergence {
	t.Helper()

	s, err := m.NewStream(StreamOptions{Language: r.Language})
	if err != nil {
		t.Fatal(err)
	}

	fed := 0

	for _, n := range r.callLengths(len(x))[:i] {
		_, err = s.Process(x[fed : fed+n])
		if err != nil {
			t.Fatal(err)
		}

		fed += n
	}

	end := *fx.Calls[i].End
	s.buf = append(s.buf, x[fed:end]...)
	s.received = end
	s.limit()
	win := s.buf

	words, lang, err := s.pass()
	if err != nil {
		t.Fatal(err)
	}

	ours := s.fresh(words)

	var out streamCallOut

	err = json.Unmarshal(fx.Calls[i].Out, &out)
	if err != nil {
		t.Fatal(err)
	}

	engine := slices.Concat(strings.Fields(out.Text), strings.Fields(out.Pending))
	d := divergence{ours: text(ours), engine: strings.Join(engine, " ")}
	all := strings.Fields(text(words))
	prefix := all[:len(all)-len(ours)]

	switch {
	case lang != out.Language && (lang == "" || out.Language == ""):
		d.kind = "silence"
	case lang != out.Language:
		d.kind = "language"
	case d.ours == d.engine:
		d.kind = "bookkeeping"
	case len(all) >= len(engine) && slices.Equal(all[len(all)-len(engine):], engine):
		d.kind = "times"
	default:
		d.kind = "pass"
		d.margin, d.sum, d.tokens, d.eTokens = passMargin(t, m, win, lang, strings.Join(append(prefix, engine...), " "))
	}

	return d
}

// passMargin is how much better our beam's transcript of win scores than
// text, teacher forced in its canonical spelling: the difference of the
// search's score (Σ log p over the text tokens / their number) and of Σ log p.
func passMargin(t *testing.T, m *Model, win []float32, lang, transcript string) (float64, float64, int, int) {
	t.Helper()

	best := beamDecode(t, m, win, lang)

	emb, err := m.Embed(win)
	if err != nil {
		t.Fatal(err)
	}

	mem, err := m.dec.Memory(emb.Values, emb.Frames)
	if err != nil {
		t.Fatal(err)
	}

	seq, err := m.dec.Targets(lang, transcript)
	if err != nil {
		t.Fatal(err)
	}

	f, err := m.dec.TeacherForce(mem, seq)
	if err != nil {
		t.Fatal(err)
	}

	n := len(f.LogProbs) - 2 // the language token and EOS
	sum := 0.0

	for _, lp := range f.LogProbs[1 : 1+n] {
		sum += lp
	}

	var ours float64
	for _, lp := range best.LogProbs[:len(best.Tokens)] {
		ours += lp
	}

	return best.Score - sum/float64(max(n, 1)), ours - sum, len(best.Tokens), n
}

// The Phase 8 Accept as written: on the 32 clips fed in 1 s chunks, the
// final committed text against the offline transcript (exact, and
// normalised as eval.Words: lower case, no punctuation), for our stream and
// for the engine's, against the engine's offline text and against our own
// Transcribe.
func TestStreamAccept(t *testing.T) {
	m := loadModel(t)
	runs, _ := loadStreamFixtures(t)

	var names []string

	for _, r := range runs {
		if r.Set == "clips" {
			names = append(names, r.Name)
		}
	}

	_, results := replayStreams(t, m, names...)
	cache := map[string][]float32{}

	var oursEngine, oursOwn, engineEngine acceptCount

	for _, r := range results {
		x := streamSamples(t, r.run, cache)

		own, err := m.Transcribe(x, TranscribeOptions{})
		if err != nil {
			t.Fatal(err)
		}

		oursEngine.add(r.text, r.run.Offline)
		oursOwn.add(r.text, own.Text)
		engineEngine.add(r.engine, r.run.Offline)

		if r.text != r.run.Offline {
			t.Logf("%s:\n  stream  %q\n  offline %q\n  engine stream %q", r.run.Name, r.text, r.run.Offline, r.engine)
		}
	}

	t.Logf("our stream vs the engine's offline text: %v", &oursEngine)
	t.Logf("our stream vs our Transcribe:            %v", &oursOwn)
	t.Logf("the engine's stream vs its offline text: %v", &engineEngine)
}

// acceptCount compares streamed texts with offline ones.
type acceptCount struct{ runs, equal, normal, errors, words int }

func (a *acceptCount) String() string {
	return fmt.Sprintf("equal %d/%d, normalised %d/%d, word errors %d/%d (%.1f %%)", a.equal, a.runs, a.normal, a.runs,
		a.errors, a.words, 100*float64(a.errors)/float64(max(a.words, 1)))
}

func (a *acceptCount) add(streamed, offline string) {
	a.runs++
	a.equal += b2i(streamed == offline)
	a.normal += b2i(slices.Equal(eval.Words(streamed), eval.Words(offline)))
	a.errors += eval.EditDistance(strings.Fields(offline), strings.Fields(streamed))
	a.words += len(strings.Fields(offline))
}

// The 8.4 evidence (F-036): which windows [s, end), s on the 80 ms grid from
// the stream start, reproduce the engine's passes in long-silence-lead once
// the buffer is over 30 s, including the whole buffer (s = 0), which no
// Stream feeds. About 2100 passes; run it with
// `just stream-parity -run TestStreamWindowScan`.
func TestStreamWindowScan(t *testing.T) {
	m := loadModel(t)
	runs, calls := loadStreamFixtures(t)

	i := slices.IndexFunc(runs, func(r streamFixtureRun) bool { return r.Name == "long-silence-lead" })
	if i < 0 {
		t.Fatal("no long-silence-lead run")
	}

	x := streamSamples(t, runs[i], map[string][]float32{})

	for _, c := range calls[runs[i].Name].Calls {
		end := *c.End
		if end <= MaxSamples+SampleRate {
			continue
		}

		var out streamCallOut

		err := json.Unmarshal(c.Out, &out)
		if err != nil {
			t.Fatal(err)
		}

		if out.Pending == "" {
			continue // a silent pass
		}

		var hits []string

		tried := 0
		grid := (end - MaxSamples + frameSamples - 1) / frameSamples * frameSamples

		for s := 0; s <= end-SampleRate; s += frameSamples {
			r := passAnyLength(t, m, x[s:end])
			tried++

			if r.Language == out.Language && r.Text == out.Pending {
				hits = append(hits, fmt.Sprintf("%.2f", float64(s)/SampleRate))
			}
		}

		t.Logf("%.2f s, the engine [%s] %q: %d of %d starts; whole buffer %v, last ≤ 30 s on the grid (%.2f s) %v; %v",
			float64(end)/SampleRate, out.Language, out.Pending, len(hits), tried, slices.Contains(hits, "0.00"),
			float64(grid)/SampleRate, slices.Contains(hits, fmt.Sprintf("%.2f", float64(grid)/SampleRate)), hits)
	}
}

// passAnyLength is Transcribe's pass without the 30 s check, for the window
// scan only.
func passAnyLength(t *testing.T, m *Model, pcm []float32) Result {
	t.Helper()

	levels, err := m.gate.Levels(pcm)
	if err != nil {
		t.Fatal(err)
	}

	if silence.Spread(levels) <= silence.ThresholdDB {
		return Silence()
	}

	enc, frames, err := m.enc.Embed(pcm)
	if err != nil {
		t.Fatal(err)
	}

	mem, err := m.dec.Memory(enc, frames)
	if err != nil {
		t.Fatal(err)
	}

	out, err := m.strategy.decode(m.dec, mem, request{}, newClock(time.Now))
	if err != nil {
		t.Fatal(err)
	}

	text, err := m.dec.Decode(out.tokens)
	if err != nil {
		t.Fatal(err)
	}

	return Result{Text: text, Language: out.lang}
}
