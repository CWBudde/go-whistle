package whistle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"
)

// engineWords reads the engine's words of a corpus clip.
func engineWords(t *testing.T, id string) []Word {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(oracleDir, id+".json"))
	if err != nil {
		t.Fatal(err)
	}

	var r Result

	err = json.Unmarshal(raw, &r)
	if err != nil {
		t.Fatalf("%s: %v", id, err)
	}

	return r.Words
}

// wordScore counts our words against the engine's.
type wordScore struct {
	engine, matched, start1, end1, both1, startExact, endExact, prob01 int
	probAbs                                                            float64
}

func (s *wordScore) String() string {
	return fmt.Sprintf("%d engine words, %d paired; start and end within 80 ms %.1f %% (start %.1f %%, end %.1f %%); "+
		"exact start %.1f %%, end %.1f %%; probability within 0.01 %.1f %%, mean |Δp| %.4f", s.engine, s.matched,
		s.share(s.both1), s.share(s.start1), s.share(s.end1), s.share(s.startExact), s.share(s.endExact),
		s.share(s.prob01), s.probAbs/float64(max(1, s.matched)))
}

// add pairs ours with the engine's words by their strings in order (a
// longest common subsequence, so a transcript that differs costs only its
// differing words) and counts every engine word.
func (s *wordScore) add(ours, engine []Word) {
	s.engine += len(engine)

	near := func(d time.Duration) bool { return d.Abs() <= FrameDuration+time.Millisecond }

	for _, p := range pairWords(ours, engine) {
		o, e := ours[p[0]], engine[p[1]]
		ds, de := o.Start-e.Start, o.End-e.End
		s.matched++

		if ds == 0 {
			s.startExact++
		}

		if de == 0 {
			s.endExact++
		}

		if near(ds) {
			s.start1++
		}

		if near(de) {
			s.end1++
		}

		if near(ds) && near(de) {
			s.both1++
		}

		s.probAbs += math.Abs(o.Probability - e.Probability)
		if math.Abs(o.Probability-e.Probability) <= 0.01+1e-9 {
			s.prob01++
		}
	}
}

func (s *wordScore) share(n int) float64 { return 100 * float64(n) / float64(max(1, s.engine)) }

// pairWords is the longest common subsequence of the word strings, as
// index pairs.
func pairWords(a, b []Word) [][2]int {
	n, m := len(a), len(b)
	l := make([][]int, n+1)

	for i := range l {
		l[i] = make([]int, m+1)
	}

	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i].Text == b[j].Text {
				l[i][j] = l[i+1][j+1] + 1
			} else {
				l[i][j] = max(l[i+1][j], l[i][j+1])
			}
		}
	}

	var out [][2]int

	for i, j := 0, 0; i < n && j < m; {
		switch {
		case a[i].Text == b[j].Text:
			out = append(out, [2]int{i, j})
			i++
			j++
		case l[i+1][j] >= l[i][j+1]:
			i++
		default:
			j++
		}
	}

	return out
}

// checkWords requires our words to spell the text, to be monotone (start
// ≤ end ≤ the next start), on the 80 ms grid and within [0, the clip's
// duration], with probabilities in [0, 1].
func checkWords(t *testing.T, id string, res Result, samples int) {
	t.Helper()

	dur := time.Duration(samples) * time.Second / SampleRate
	texts := make([]string, len(res.Words))

	for i, w := range res.Words {
		texts[i] = w.Text

		if w.Start < 0 || w.End < w.Start || w.End > dur || (i > 0 && w.Start < res.Words[i-1].End) {
			t.Errorf("%s word %d %q: %v–%v after %v, clip %v", id, i, w.Text, w.Start, w.End,
				res.Words[max(0, i-1)].End, dur)
		}

		if w.Start%FrameDuration != 0 || w.End%FrameDuration != 0 || w.Probability < 0 || w.Probability > 1 {
			t.Errorf("%s word %d %q: %v–%v off the 80 ms grid or probability %v", id, i, w.Text, w.Start, w.End,
				w.Probability)
		}
	}

	if strings.Join(texts, " ") != res.Text {
		t.Errorf("%s: words %q do not spell %q", id, texts, res.Text)
	}
}

// wordFloors are the shares (%) TestTranscribeWordsMatchEngine requires on
// the corpus, just below the measured values (docs/FINDINGS.md F-033):
// start and end within 80 ms, and probability within 0.01.
var wordFloors = struct{ both, prob float64 }{both: 97, prob: 88}

// Accept (PLAN.md Phase 7): through Transcribe with word timestamps, the
// words spell the text, are monotone and within the clip, and their start
// and end lie within 80 ms of the engine's on at least 90 % of the
// engine's words (all 32 clips without -short; a word of a transcript that
// differs from the engine's counts only when its string pairs up). It also
// times the extra decoder pass.
func TestTranscribeWordsMatchEngine(t *testing.T) {
	m := loadModel(t)
	clips := corpus(t)

	var (
		all            wordScore
		plain, timed   time.Duration
		exactTexts, nw int
	)

	for _, c := range clips {
		start := time.Now()

		ref, err := m.Transcribe(c.Samples, TranscribeOptions{})
		if err != nil {
			t.Fatal(err)
		}

		mid := time.Now()

		res, err := m.Transcribe(c.Samples, TranscribeOptions{WordTimestamps: true})
		if err != nil {
			t.Fatal(err)
		}

		plain, timed = plain+mid.Sub(start), timed+time.Since(mid)

		if res.Text != ref.Text || res.Language != ref.Language || ref.Words != nil {
			t.Errorf("%s: with timestamps [%s] %q, without [%s] %q, words %v", c.ID, res.Language, res.Text,
				ref.Language, ref.Text, ref.Words)
		}

		checkWords(t, c.ID, res, len(c.Samples))

		var s wordScore

		s.add(res.Words, engineWords(t, c.ID))
		all.add(res.Words, engineWords(t, c.ID))
		nw += len(res.Words)

		if res.Text == c.Text {
			exactTexts++
		}

		t.Logf("%s: %s", c.ID, &s)
	}

	t.Logf("%d clips (%d with the engine's text), %d words: %s", len(clips), exactTexts, nw, &all)
	t.Logf("Transcribe %.1f ms per clip without timestamps, %.1f ms with (+%.1f %%)",
		float64(plain.Microseconds())/1000/float64(len(clips)), float64(timed.Microseconds())/1000/float64(len(clips)),
		100*(timed.Seconds()/plain.Seconds()-1))

	if all.share(all.both1) < 90 || all.share(all.both1) < wordFloors.both {
		t.Errorf("start and end within 80 ms on %.1f %% of the engine's words, want ≥ 90 %% and ≥ %.0f %%",
			all.share(all.both1), wordFloors.both)
	}

	if all.share(all.prob01) < wordFloors.prob {
		t.Errorf("probability within 0.01 on %.1f %%, want ≥ %.0f %%", all.share(all.prob01), wordFloors.prob)
	}
}

// numbers matches the JSON numbers of a result line, values the string
// values and words a run of word objects once both are masked.
var (
	numbers = regexp.MustCompile(`-?[0-9]+\.[0-9]+`)
	values  = regexp.MustCompile(`:"(?:[^"\\]|\\.)*"`)
	words   = regexp.MustCompile(`(\{"word":\$,"start":#,"end":#,"probability":#\},?)+`)
)

// shape is a result line with its numbers masked, and with its strings
// masked and its word objects collapsed too unless exact.
func shape(line []byte, exact bool) []byte {
	out := numbers.ReplaceAll(line, []byte("#"))
	if exact {
		return out
	}

	out = values.ReplaceAll(out, []byte(":$"))

	return words.ReplaceAll(out, []byte("W"))
}

// TestTranscribeWordsJSON transcribes the clips behind the engine's raw
// --audio-word-timestamps outputs (docs/FINDINGS.md F-027) with word
// timestamps and requires the engine's JSON with every number masked: the
// field names and order, the words, the silence form. The 30 s clip is the
// first one tiled to 480,000 samples; its transcript differs from the
// engine's (a decoding difference, not a timing one), so there the strings
// are masked and the word objects collapsed as well.
func TestTranscribeWordsJSON(t *testing.T) {
	m := loadModel(t)
	outs := engineOutputs(t)

	clipOf := map[string][]float32{}
	for _, c := range loadClips(t, "fleurs-de_de-10229344228128634115", "ls-clean-1320-122617-0003") {
		clipOf[c.ID] = c.Samples
	}

	first := clipOf["ls-clean-1320-122617-0003"]
	tiled := make([]float32, MaxSamples)

	for i := range tiled {
		tiled[i] = first[i%len(first)]
	}

	clipOf["tiled-30s"] = tiled
	clipOf["silence-2s"] = make([]float32, 2*SampleRate)

	names := make([]string, 0, len(clipOf))
	for name := range clipOf {
		names = append(names, name)
	}

	slices.Sort(names)

	for _, name := range names {
		want, ok := outs[name+".words.stdout"]
		if !ok {
			t.Fatalf("no engine output for %s", name)
		}

		res, err := m.Transcribe(clipOf[name], TranscribeOptions{WordTimestamps: true})
		if err != nil {
			t.Fatal(err)
		}

		got, err := res.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}

		checkWords(t, name, res, len(clipOf[name]))

		var eng Result

		err = json.Unmarshal(want, &eng)
		if err != nil {
			t.Fatal(err)
		}

		exact := res.Text == eng.Text
		if !bytes.Equal(shape(got, exact), shape(want, exact)) {
			t.Errorf("%s: masked JSON differs (same text %v)\n got %s\nwant %s", name, exact, got, want)
		}

		var s wordScore

		s.add(res.Words, eng.Words)
		t.Logf("%s: %d bytes, masked JSON equal (same text %v); %s", name, len(got), exact, &s)
	}
}

// BenchmarkWordTimestamps prices the timing pass (PLAN.md 7.1) on a 6.3 s
// clip: Transcribe without and with word timestamps, and the teacher-forced
// capture pass with the alignment alone.
func BenchmarkWordTimestamps(b *testing.B) {
	m := loadModel(b)
	c := loadClips(b, "ls-clean-1320-122617-0003")[0]

	for _, ts := range []bool{false, true} {
		b.Run(fmt.Sprintf("transcribe/timestamps=%v", ts), func(b *testing.B) {
			for b.Loop() {
				_, err := m.Transcribe(c.Samples, TranscribeOptions{WordTimestamps: ts})
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}

	enc, frames, err := m.enc.Embed(c.Samples)
	if err != nil {
		b.Fatal(err)
	}

	mem, err := m.dec.Memory(enc, frames)
	if err != nil {
		b.Fatal(err)
	}

	levels, err := m.gate.Levels(c.Samples)
	if err != nil {
		b.Fatal(err)
	}

	out, err := m.strategy.decode(m.dec, mem, request{}, newClock(time.Now))
	if err != nil {
		b.Fatal(err)
	}

	b.Run("pass", func(b *testing.B) {
		for b.Loop() {
			_, err := timeWords(m.dec, mem, out, levels, len(c.Samples))
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
