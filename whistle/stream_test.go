package whistle

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/eval"
	"github.com/cwbudde/go-whistle/internal/frontend"
)

// streamDir holds the engine's stream calls (docs/FINDINGS.md F-032).
const streamDir = "../testdata/oracle/stream"

// The engine's empty stream JSON, a Stop without a stream (F-032 item 6).
func TestStreamResultJSON(t *testing.T) {
	got, err := emptyStream().MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	const want = `{"text":"","words":[],"pending":"","language":"","received":0.00,"pass_ms":0.0}`
	if string(got) != want {
		t.Errorf("empty = %s, want %s", got, want)
	}

	r := StreamResult{
		Text:     `Hi "there".`,
		Words:    []Word{{Text: "Hi", Start: 400 * time.Millisecond, End: 720 * time.Millisecond, Probability: 0.9876}},
		Pending:  "and more",
		Language: "en",
		Received: 3.125,
		Pass:     12345 * time.Microsecond,
	}

	got, err = r.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	const want2 = `{"text":"Hi \"there\".","words":[{"word":"Hi","start":0.40,"end":0.72,"probability":0.988}],` +
		`"pending":"and more","language":"en","received":3.12,"pass_ms":12.3}`
	if string(got) != want2 {
		t.Errorf("result = %s, want %s", got, want2)
	}
}

func TestEndsSentence(t *testing.T) {
	for _, c := range []struct {
		word, next string
		want       bool
	}{
		{"up.", "And", true},
		{"up.", "and", false},
		{"up,", "And", false},
		{"why?", "Then", true},
		{"stop!", "1984", true},
		{`it."`, "Then", true},
		{"it.", `"Then`, true},
		{"Mr", "Smith", false},
		{"Straße.", "Über", true},
	} {
		if got := endsSentence(c.word, c.next); got != c.want {
			t.Errorf("endsSentence(%q, %q) = %v, want %v", c.word, c.next, got, c.want)
		}
	}
}

// The bookkeeping between passes on synthetic words, without the model:
// the repeat rule, LocalAgreement-2 and the sentence trim.
func TestStreamBookkeeping(t *testing.T) {
	w := func(text string, start, end int) streamWord {
		return streamWord{
			Word: Word{Text: text, Start: time.Duration(start) * 10 * time.Millisecond, End: time.Duration(end) * 10 * time.Millisecond},
			end:  float64(end) / 100,
		}
	}

	s := &Stream{}

	// The first pass commits nothing: there is no previous pass.
	if c := s.agree(s.fresh([]streamWord{w("a", 0, 40), w("b", 40, 80)})); len(c) != 0 {
		t.Fatalf("first commit %v", c)
	}

	// The second agrees on "a" only.
	c := s.agree(s.fresh([]streamWord{w("a", 0, 40), w("x", 40, 80), w("c", 80, 120)}))
	if joinWords(c) != "a" || text(s.pending) != "x c" || s.last != 400*time.Millisecond {
		t.Fatalf("commit %q, pending %q, last %v", joinWords(c), text(s.pending), s.last)
	}

	// "a" ends at 0.40 s: a pass word starting at 0.30 s or earlier is
	// dropped by time, one at 0.31 s is kept but repeats "a" and is
	// dropped as a 1-gram.
	got := s.fresh([]streamWord{w("a", 0, 40), w("x", 30, 80)})
	if text(got) != "" {
		t.Errorf("fresh by time = %q, want empty", text(got))
	}

	got = s.fresh([]streamWord{w("a", 31, 50), w("x", 50, 80)})
	if text(got) != "x" {
		t.Errorf("fresh by n-gram = %q, want x", text(got))
	}

	// More than a second after the last commit, no repeat is dropped.
	got = s.fresh([]streamWord{w("a", 140, 150)})
	if text(got) != "a" {
		t.Errorf("fresh after 1 s = %q, want a", text(got))
	}

	// The trim cuts at the end of the second-to-last sentence.
	s = &Stream{buf: make([]float32, 20*SampleRate)}
	s.inBuffer = []streamWord{w("One.", 0, 100), w("Two", 100, 200), w("three.", 200, 300), w("Four", 300, 400)}
	s.trimSentence()

	if s.offset != 3*SampleRate || s.at != 3 || len(s.buf) != 17*SampleRate || text(s.inBuffer) != "Four" {
		t.Errorf("trim: offset %d (%v s), %d samples, in buffer %q", s.offset, s.at, len(s.buf), text(s.inBuffer))
	}

	// One sentence: nothing to cut.
	s = &Stream{buf: make([]float32, 20*SampleRate)}
	s.inBuffer = []streamWord{w("One", 0, 100), w("two.", 100, 200)}
	s.trimSentence()

	if s.offset != 0 {
		t.Errorf("one sentence: offset %d", s.offset)
	}

	// Over 30 s with nothing to trim: the oldest audio goes in whole frames
	// (PLAN.md 8.4), and the committed words in it with it.
	s = &Stream{buf: make([]float32, MaxSamples+16080), offset: 1280, at: 0.08, received: 1280 + MaxSamples + 16080}
	s.inBuffer = []streamWord{w("old", 0, 100), w("new", 200, 300)}
	s.limit()

	if s.offset != 1280+13*1280 || len(s.buf) != MaxSamples+16080-13*1280 || s.at != 0.08+13*0.08 ||
		text(s.inBuffer) != "new" || s.receivedSeconds() != float64(s.received)/SampleRate {
		t.Errorf("limit: offset %d (%v s), %d samples, in buffer %q, received %v", s.offset, s.at, len(s.buf),
			text(s.inBuffer), s.receivedSeconds())
	}

	if s.limit(); len(s.buf) > MaxSamples || s.offset != 1280+13*1280 {
		t.Errorf("limit twice: offset %d, %d samples", s.offset, len(s.buf))
	}
}

// text is the words joined by spaces.
func text(words []streamWord) string { return joinWords(plainWords(words)) }

// streamFixtureRun is one run of manifest.json.
type streamFixtureRun struct {
	Name         string   `json:"name"`
	Set          string   `json:"set"`
	Clips        []string `json:"clips"`
	LeadSamples  int      `json:"lead_samples"`
	GapSamples   int      `json:"gap_samples"`
	ChunkSamples int      `json:"chunk_samples"`
	CallSamples  []int    `json:"call_samples"`
	Language     string   `json:"language"`
	Samples      int      `json:"samples"`
	PCMSHA256    string   `json:"pcm_sha256"`
	Offline      string   `json:"offline"`
	Text         string   `json:"text"`
}

// streamFixtureCall is one call of calls.json.
type streamFixtureCall struct {
	Start       *int            `json:"start"`
	End         *int            `json:"end"`
	RC          int             `json:"rc"`
	Error       string          `json:"error"`
	BufferStart *int            `json:"buffer_start"`
	Dropped     *int            `json:"dropped"`
	Out         json.RawMessage `json:"out"`
}

type streamFixture struct {
	Calls []streamFixtureCall `json:"calls"`
	Stop  streamFixtureCall   `json:"stop"`
}

// loadStreamFixtures reads the runs and their calls, or skips the test.
func loadStreamFixtures(t testing.TB) ([]streamFixtureRun, map[string]streamFixture) {
	t.Helper()

	requireFile(t, filepath.Join(streamDir, "calls.json"))

	raw, err := os.ReadFile(filepath.Join(streamDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Runs []streamFixtureRun `json:"runs"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	raw, err = os.ReadFile(filepath.Join(streamDir, "calls.json"))
	if err != nil {
		t.Fatal(err)
	}

	var calls map[string]streamFixture

	err = json.Unmarshal(raw, &calls)
	if err != nil {
		t.Fatal(err)
	}

	return manifest.Runs, calls
}

// streamSamples builds a run's samples from testdata/audio (manifest.json's
// audio: lead zeros, the clips with gap zeros between them) and checks them
// against pcm_sha256.
func streamSamples(t testing.TB, r streamFixtureRun, cache map[string][]float32) []float32 {
	t.Helper()

	x := make([]float32, r.LeadSamples, r.Samples)

	for i, id := range r.Clips {
		pcm, ok := cache[id]
		if !ok {
			var err error

			pcm, err = frontend.ReadWAV(filepath.Join(clipDir, id+".wav"))
			if err != nil {
				t.Fatalf("%s: %v", id, err)
			}

			cache[id] = pcm
		}

		if i > 0 {
			x = append(x, make([]float32, r.GapSamples)...)
		}

		x = append(x, pcm...)
	}

	h := sha256.New()
	_ = binary.Write(h, binary.LittleEndian, x)

	if got := hex.EncodeToString(h.Sum(nil)); got != r.PCMSHA256 || len(x) != r.Samples {
		t.Fatalf("%s: %d samples, sha256 %s; manifest %d, %s", r.Name, len(x), got, r.Samples, r.PCMSHA256)
	}

	return x
}

// callLengths cuts n samples into the run's calls.
func (r streamFixtureRun) callLengths(n int) []int {
	if r.ChunkSamples == 0 {
		return r.CallSamples
	}

	var out []int
	for i := 0; i < n; i += r.ChunkSamples {
		out = append(out, min(r.ChunkSamples, n-i))
	}

	return out
}

// streamCallOut is a call's JSON without pass_ms, numbers as printed.
type streamCallOut struct {
	Text     string        `json:"text"`
	Words    []fixtureWord `json:"words"`
	Pending  string        `json:"pending"`
	Language string        `json:"language"`
	Received json.Number   `json:"received"`
}

type fixtureWord struct {
	Word        string      `json:"word"`
	Start       json.Number `json:"start"`
	End         json.Number `json:"end"`
	Probability json.Number `json:"probability"`
}

// normalStreamJSON is a stream JSON with pass_ms dropped and the keys sorted,
// compacted: the form of calls.json's out.
func normalStreamJSON(js []byte) (string, error) {
	var fields map[string]json.RawMessage

	err := json.Unmarshal(js, &fields)
	if err != nil {
		return "", err
	}

	delete(fields, "pass_ms")

	out, err := json.Marshal(fields)
	if err != nil {
		return "", err
	}

	var b bytes.Buffer

	err = json.Compact(&b, out)

	return b.String(), err
}

// streamCallResult compares one of our calls with the engine's.
type streamCallResult struct {
	equal                           bool // rc, error and the JSON without pass_ms, byte for byte
	decision                        bool // rc, error, text, pending, language and received: everything but word times and probabilities
	rc, text, pending, lang, recv   bool
	words                           int  // committed words paired by position, when text is equal
	start, end, near, prob, wordsEq int  // of those: start, end exact; both within 80 ms; probability within 0.01; all exact
	hasBufferStart, bufferStart     bool // our pass started where the engine's did (calls with buffer_start)
}

// streamRunResult is one replayed run.
type streamRunResult struct {
	run       streamFixtureRun
	calls     []streamCallResult // the process calls, then the stop
	text      string             // every committed word, joined by spaces
	engine    string             // the engine's committed text
	firstDiff int                // the first call whose decision differs, -1 when none
	diffs     []string
}

// replayStream feeds a run through a new Stream call by call and compares
// every call with the engine's.
func replayStream(t testing.TB, m *Model, r streamFixtureRun, fx streamFixture, x []float32) streamRunResult {
	t.Helper()

	s, err := m.NewStream(StreamOptions{Language: r.Language})
	if err != nil {
		t.Fatal(err)
	}

	lengths := r.callLengths(len(x))
	if len(lengths) != len(fx.Calls) {
		t.Fatalf("%s: %d calls, the engine made %d", r.Name, len(lengths), len(fx.Calls))
	}

	res := streamRunResult{run: r, firstDiff: -1}

	committed := make([]string, 0, 64)
	engine := make([]string, 0, 64)
	fed := 0

	for i, n := range lengths {
		want := fx.Calls[i]

		if *want.Start != fed || *want.End != fed+n {
			t.Fatalf("%s call %d: samples [%d, %d), the engine's [%d, %d)", r.Name, i, fed, fed+n, *want.Start, *want.End)
		}

		t0 := s.offset

		got, err := s.Process(x[fed : fed+n])
		fed += n

		c := compareStreamCall(t, got, err, want)
		if want.BufferStart != nil {
			c.hasBufferStart = true
			c.bufferStart = t0 == *want.BufferStart
		}

		res.add(c, i, got, err, want)
		committed = append(committed, strings.Fields(got.Text)...)
		engine = append(engine, outText(t, want)...)
	}

	got := s.Stop()
	c := compareStreamCall(t, got, nil, fx.Stop)
	res.add(c, len(lengths), got, nil, fx.Stop)

	res.text = strings.Join(append(committed, strings.Fields(got.Text)...), " ")
	res.engine = strings.Join(append(engine, outText(t, fx.Stop)...), " ")

	return res
}

func (res *streamRunResult) add(c streamCallResult, i int, got StreamResult, err error, want streamFixtureCall) {
	res.calls = append(res.calls, c)

	if c.decision {
		return
	}

	if res.firstDiff < 0 {
		res.firstDiff = i
	}

	js, _ := got.MarshalJSON()
	ours, _ := normalStreamJSON(js)

	if err != nil {
		ours = err.Error()
	}

	theirs, _ := normalStreamJSON(want.Out)
	res.diffs = append(res.diffs, fmt.Sprintf("call %d:\n  ours   %s\n  engine rc %d %s%s", i, ours, want.RC, want.Error, theirs))
}

// outText is the committed words of a fixture call.
func outText(t testing.TB, c streamFixtureCall) []string {
	t.Helper()

	if c.RC < 0 {
		return nil
	}

	var out streamCallOut

	err := json.Unmarshal(c.Out, &out)
	if err != nil {
		t.Fatal(err)
	}

	return strings.Fields(out.Text)
}

func compareStreamCall(t testing.TB, got StreamResult, err error, want streamFixtureCall) streamCallResult {
	t.Helper()

	if want.RC < 0 || err != nil {
		ok := want.RC < 0 && err != nil && strings.Contains(err.Error(), want.Error)

		return streamCallResult{equal: ok, decision: ok, rc: ok, text: ok, pending: ok, lang: ok, recv: ok}
	}

	js, mErr := got.MarshalJSON()
	if mErr != nil {
		t.Fatal(mErr)
	}

	ours, nErr := normalStreamJSON(js)
	if nErr != nil {
		t.Fatal(nErr)
	}

	theirs, nErr := normalStreamJSON(want.Out)
	if nErr != nil {
		t.Fatal(nErr)
	}

	var a, b streamCallOut

	_ = json.Unmarshal([]byte(ours), &a)
	_ = json.Unmarshal([]byte(theirs), &b)

	c := streamCallResult{
		equal:   ours == theirs && len(got.Words) == want.RC,
		rc:      len(got.Words) == want.RC,
		text:    a.Text == b.Text,
		pending: a.Pending == b.Pending,
		lang:    a.Language == b.Language,
		recv:    a.Received == b.Received,
	}
	c.decision = c.rc && c.text && c.pending && c.lang && c.recv

	if !c.text || len(a.Words) != len(b.Words) {
		return c
	}

	for i, w := range a.Words {
		e := b.Words[i]
		c.words++
		c.start += b2i(w.Start == e.Start)
		c.end += b2i(w.End == e.End)
		c.near += b2i(math.Abs(num(w.Start)-num(e.Start)) <= 0.0801 && math.Abs(num(w.End)-num(e.End)) <= 0.0801)
		c.prob += b2i(math.Abs(num(w.Probability)-num(e.Probability)) <= 0.0101)
		c.wordsEq += b2i(w == e)
	}

	return c
}

// num is a JSON number's value, NaN when it is none.
func num(n json.Number) float64 {
	v, err := n.Float64()
	if err != nil {
		return math.NaN()
	}

	return v
}

// streamTally sums replayed runs.
type streamTally struct {
	runs, runsEqual, calls, callsEqual, decisions         int
	rc, text, pending, lang, recv                         int
	words, start, end, near, prob, wordsEq                int
	bufferStarts, bufferStartsEqual                       int
	finalEqual                                            int // the run's committed text equals the engine's
	offline, offlineEqual, offlineNormEqual, offlineWords int
	offlineErrors                                         int
}

func (st *streamTally) String() string {
	s := fmt.Sprintf("runs with every decision equal %d/%d; calls: decision equal %d/%d (rc %d, text %d, pending %d, "+
		"language %d, received %d), JSON equal %d; committed words %d: start %d, end %d, both within 80 ms %d, "+
		"probability within 0.01 %d, all exact %d; buffer starts %d/%d; final text = engine stream %d/%d",
		st.runsEqual, st.runs, st.decisions, st.calls, st.rc, st.text, st.pending, st.lang, st.recv, st.callsEqual,
		st.words, st.start, st.end, st.near, st.prob, st.wordsEq, st.bufferStartsEqual, st.bufferStarts,
		st.finalEqual, st.runs)

	if st.offline > 0 {
		s += fmt.Sprintf("; vs offline: equal %d/%d, normalised %d/%d, word errors %d/%d",
			st.offlineEqual, st.offline, st.offlineNormEqual, st.offline, st.offlineErrors, st.offlineWords)
	}

	return s
}

func (st *streamTally) add(r streamRunResult) {
	st.runs++
	st.runsEqual += b2i(r.firstDiff < 0)

	for _, c := range r.calls {
		st.calls++
		st.callsEqual += b2i(c.equal)
		st.decisions += b2i(c.decision)
		st.rc += b2i(c.rc)
		st.text += b2i(c.text)
		st.pending += b2i(c.pending)
		st.lang += b2i(c.lang)
		st.recv += b2i(c.recv)
		st.words += c.words
		st.start += c.start
		st.end += c.end
		st.near += c.near
		st.prob += c.prob
		st.wordsEq += c.wordsEq

		if c.hasBufferStart {
			st.bufferStarts++
			st.bufferStartsEqual += b2i(c.bufferStart)
		}
	}

	st.finalEqual += b2i(r.text == r.engine)

	if r.run.Offline != "" {
		st.offline++
		st.offlineEqual += b2i(r.text == r.run.Offline)
		st.offlineNormEqual += b2i(slices.Equal(eval.Words(r.text), eval.Words(r.run.Offline)))
		st.offlineWords += len(strings.Fields(r.run.Offline))
		st.offlineErrors += eval.EditDistance(strings.Fields(r.run.Offline), strings.Fields(r.text))
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}

	return 0
}

// replayStreams replays the named runs, all when names is empty, and tallies
// them per set.
func replayStreams(t testing.TB, m *Model, names ...string) (map[string]*streamTally, []streamRunResult) {
	t.Helper()

	runs, calls := loadStreamFixtures(t)
	cache := map[string][]float32{}
	sets := map[string]*streamTally{}

	var out []streamRunResult

	for _, r := range runs {
		if len(names) > 0 && !slices.Contains(names, r.Name) {
			continue
		}

		fx, ok := calls[r.Name]
		if !ok {
			t.Fatalf("%s: no calls", r.Name)
		}

		x := streamSamples(t, r, cache)
		res := replayStream(t, m, r, fx, x)

		if sets[r.Set] == nil {
			sets[r.Set] = &streamTally{}
		}

		sets[r.Set].add(res)
		out = append(out, res)
	}

	return sets, out
}

// fastStreamRuns are the default test's runs: three clips at 1 s (en, de,
// es), one at 2 s, a forced language, 32 s of silence before speech (8.4) and
// the edge cases.
var fastStreamRuns = []string{
	"ls-clean-6930-75918-0000.c1",
	"fleurs-de_de-10308608197319017935.c1",
	"fleurs-es_419-10236991702517253859.c1",
	"ls-other-3080-5040-0018.c2",
	"fleurs-fr_fr-10174413140250646001.c1.fr",
	"long-silence-lead",
	"edge-stop-only", "edge-empty-call", "edge-empty-calls-after-speech", "edge-30s-call", "edge-30s-plus-1-call",
}

// A subset of the engine's streams, call by call (PLAN.md 8.3, 8.4,
// docs/FINDINGS.md F-036); the whole set runs under the oracle tag
// (`just stream-parity`). The floors sit just below the measured counts:
// the calls that differ follow a pass that spells other words than the
// engine's, a near tie under our model (F-036).
func TestStreamMatchesEngine(t *testing.T) {
	m := loadModel(t)
	_, runs := replayStreams(t, m, fastStreamRuns...)

	var total streamTally

	for _, r := range runs {
		total.add(r)

		if r.firstDiff >= 0 {
			t.Logf("%s: %d calls differ, the first:\n%s", r.run.Name, len(r.diffs), r.diffs[0])
		}

		if strings.HasPrefix(r.run.Name, "edge-") && r.firstDiff >= 0 {
			t.Errorf("%s differs", r.run.Name)
		}
	}

	t.Logf("%v", &total)

	floors := []struct {
		name      string
		got, want int
	}{
		{"runs", total.runs, len(fastStreamRuns)},
		{"runs with every decision equal", total.runsEqual, 6},
		{"calls with the engine's decision", total.decisions, 69},
		{"calls with the engine's received", total.recv, total.calls},
		{"calls with the engine's language", total.lang, total.calls},
		{"committed words with both times within 80 ms", total.near, total.words - 1},
		{"final texts equal to the engine's stream", total.finalEqual, 10},
	}

	for _, f := range floors {
		if f.got < f.want {
			t.Errorf("%s: %d, want at least %d", f.name, f.got, f.want)
		}
	}
}

// The stream's own invariants on a clip: Received counts every sample,
// Pass is the clock's, a call over MaxSamples feeds nothing, Stop ends the
// stream, and the committed words are monotone and on the 80 ms grid.
func TestStreamCalls(t *testing.T) {
	m := loadModel(t)
	c := loadClips(t, shortClips[1])[0]

	var ms time.Duration

	s, err := m.newStream(StreamOptions{}, func() time.Time { ms += time.Millisecond; return time.Unix(0, 0).Add(ms) })
	if err != nil {
		t.Fatal(err)
	}

	_, err = s.Process(make([]float32, MaxSamples+1))
	if !errors.Is(err, ErrTooLong) || s.open || s.received != 0 {
		t.Fatalf("over 30 s: %v, open %v, %d received", err, s.open, s.received)
	}

	var words []Word

	for i := 0; i < len(c.Samples); i += SampleRate / 2 {
		end := min(i+SampleRate/2, len(c.Samples))

		r, err := s.Process(c.Samples[i:end])
		if err != nil {
			t.Fatal(err)
		}

		if r.Pass <= 0 || r.Words == nil || len(strings.Fields(r.Text)) != len(r.Words) {
			t.Fatalf("call at %d: pass %v, %d words for %q", i, r.Pass, len(r.Words), r.Text)
		}

		if want := float64(end) / SampleRate; math.Abs(r.Received-want) > 1e-9 {
			t.Errorf("call at %d: received %v, want %v", i, r.Received, want)
		}

		words = append(words, r.Words...)
	}

	r := s.Stop()
	words = append(words, r.Words...)

	if r.Pass != 0 || r.Pending != "" || s.open {
		t.Errorf("stop: pass %v, pending %q, open %v", r.Pass, r.Pending, s.open)
	}

	if again := s.Stop(); again.Received != 0 || len(again.Words) != 0 {
		t.Errorf("second stop = %v", again)
	}

	for i, w := range words {
		if w.Start%FrameDuration != 0 || w.End%FrameDuration != 0 || w.Start > w.End ||
			(i > 0 && w.Start < words[i-1].End) {
			t.Errorf("word %d %q [%v, %v] after %v", i, w.Text, w.Start, w.End, words[max(0, i-1)].End)
		}
	}

	t.Logf("streamed %q, offline %q", joinWords(words), c.Text)

	_, err = m.NewStream(StreamOptions{Language: "xx"})
	if err == nil {
		t.Error("NewStream accepted language xx")
	}
}
