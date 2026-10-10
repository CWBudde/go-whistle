//go:build oracle

// The alignment probe harness (PLAN.md 7.2, 7.3, docs/FINDINGS.md F-033).
// The engine prints, with --audio-word-timestamps, every word with its
// start, end and probability; how it gets them is closed (U11). Every
// engine transcript recorded with words (the 32 fixtures, the LibriSpeech
// subset, the keyword cases and the depth runs, F-025) is teacher forced
// in its most likely spelling under our model with the cross-attention
// captured (probedata_test.go), so the fits are not confounded by
// transcript mismatches. The LibriSpeech subset is the training split; the
// fixtures, the keyword cases and the depth runs are held out.
//
//   - TestProbeWords: Words against the engine's word strings;
//   - TestProbeProbability: the word probability leaderboard;
//   - TestProbeLeader: Default on every set, times and probabilities;
//   - probetimes_test.go: the timing grid (TestProbeTimes), whole-layer
//     head sets (TestProbeLayers), fitted head sets (TestProbeHeads), the
//     ladder rungs (TestProbeDepth), punctuation (TestProbePunctuation) and
//     the remaining misses (TestProbeMisses);
//   - probefirst_test.go: the first word's start, the one time the grid
//     does not identify, and the fitted onset rule (TestProbeOnsetFit).
//
// Run it with `just align-probe`; WHISTLE_PROBE_TOP sets the leaderboard
// rows, WHISTLE_ALIGN_CACHE / WHISTLE_ALIGN_FRESH the data cache.

package align

import (
	"cmp"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/silence"
)

// sets is the order the probe reports the sets in.
var sets = []string{"librispeech", "fixtures", "keywords", "depth"}

// TestProbeWords: Words on the engine's text in its most likely spelling,
// and in RefTokenizer's, against the engine's word strings.
func TestProbeWords(t *testing.T) {
	ps := openSetup(t)
	items := loadItems(t, ps)

	for _, set := range sets {
		var n, ok, okRef, words int

		for _, it := range items {
			if it.Set != set {
				continue
			}

			n++
			words += len(it.Words)

			want := make([]string, len(it.Words))
			for i, w := range it.Words {
				want[i] = w.Word
			}

			if slices.Equal(wordTexts(t, ps, it.Tokens), want) {
				ok++
			} else {
				t.Logf("%s: %q, engine %q", it.ID, wordTexts(t, ps, it.Tokens), want)
			}

			if slices.Equal(wordTexts(t, ps, ps.dec.Encode(it.Text)), want) {
				okRef++
			}
		}

		t.Logf("%-11s %3d transcripts, %4d words: word strings equal on %d (best spelling), %d (RefTokenizer)", set, n,
			words, ok, okRef)
	}
}

// wordTexts is the strings of Words.
func wordTexts(t *testing.T, ps probeSetup, tokens []int) []string {
	t.Helper()

	ws, err := Words(ps.tok, tokens)
	if err != nil {
		t.Fatal(err)
	}

	out := make([]string, len(ws))
	for i, w := range ws {
		out[i] = w.Text
	}

	return out
}

// probSource turns item it's token k into a log-probability.
type probSource struct {
	name string
	lp   func(it *item, k int) float64
}

var probSources = []probSource{
	{"softmax over the vocabulary", func(it *item, k int) float64 { return it.LP[k] }},
	{"renormalised over the text pieces", func(it *item, k int) float64 { return it.LP[k] - it.TextLS[k] }},
	{"renormalised over the top 5", func(it *item, k int) float64 { return topRenorm(it, k, 5) }},
	{"renormalised over the top 6", func(it *item, k int) float64 { return topRenorm(it, k, 6) }},
	{"keyword-adjusted, clipped at 1", func(it *item, k int) float64 { return min(0, it.Adj[k]) }},
	{"keyword-adjusted, renormalised", func(it *item, k int) float64 { return it.Adj[k] - it.AdjLS[k] }},
}

// topRenorm is log p renormalised over the row's n best tokens (−∞ when
// the token is not among them).
func topRenorm(it *item, k, n int) float64 {
	if it.LP[k] < it.Top[k][n-1] {
		return math.Inf(-1)
	}

	return it.LP[k] - logSumExp(it.Top[k][:n])
}

// probAgg combines a word's token log-probabilities into its probability.
type probAgg struct {
	name string
	f    func(lps []float64) float64
}

var probAggs = []probAgg{
	{"mean p", func(v []float64) float64 {
		var s float64
		for _, x := range v {
			s += math.Exp(x)
		}

		return s / float64(len(v))
	}},
	{"product", func(v []float64) float64 {
		var s float64
		for _, x := range v {
			s += x
		}

		return math.Exp(s)
	}},
	{"geometric mean", func(v []float64) float64 {
		var s float64
		for _, x := range v {
			s += x
		}

		return math.Exp(s / float64(len(v)))
	}},
	{"min", func(v []float64) float64 { return math.Exp(slices.Min(v)) }},
	{"max", func(v []float64) float64 { return math.Exp(slices.Max(v)) }},
	{"first token", func(v []float64) float64 { return math.Exp(v[0]) }},
	{"last token", func(v []float64) float64 { return math.Exp(v[len(v)-1]) }},
}

// probScore is one candidate's agreement with the engine's probabilities.
type probScore struct {
	name               string
	words, exact, near int // near: within 0.01
	sumAbs             float64
}

// TestProbeProbability ranks the word probability candidates: a token
// log-probability source × an aggregate × the tokens it spans (all of the
// word's, or its core without the punctuation tokens), against the
// engine's three decimals on every word of every set.
func TestProbeProbability(t *testing.T) {
	ps := openSetup(t)
	items := loadItems(t, ps)

	var scores []probScore

	for _, src := range probSources {
		for _, agg := range probAggs {
			for _, span := range []string{"all tokens", "core tokens"} {
				for _, set := range append([]string{""}, sets...) {
					s := probScore{name: fmt.Sprintf("%s · %s · %s", agg.name, span, src.name)}
					if set != "" {
						s.name += " [" + set + "]"
					}

					for _, it := range items {
						if set != "" && it.Set != set {
							continue
						}

						ws, err := Words(ps.tok, it.Tokens)
						if err != nil {
							t.Fatal(err)
						}

						for i, w := range ws {
							a, b := w.Start, w.End
							if span == "core tokens" {
								a, b = w.Core[0], w.Core[1]
							}

							lps := make([]float64, 0, b-a)
							for k := a; k < b; k++ {
								lps = append(lps, src.lp(it, k))
							}

							p := agg.f(lps)
							want := it.Words[i].Probability
							s.words++
							s.sumAbs += math.Abs(p - want)

							if math.Round(p*1000) == math.Round(want*1000) {
								s.exact++
							}

							if math.Abs(p-want) <= 0.01 {
								s.near++
							}
						}
					}

					scores = append(scores, s)
				}
			}
		}
	}

	report := func(title string, keep func(probScore) bool) {
		var rows []probScore

		for _, s := range scores {
			if keep(s) {
				rows = append(rows, s)
			}
		}

		slices.SortStableFunc(rows, func(a, b probScore) int {
			return cmp.Or(cmp.Compare(b.exact*a.words, a.exact*b.words), cmp.Compare(a.sumAbs/float64(a.words),
				b.sumAbs/float64(b.words)))
		})

		t.Logf("── %s", title)
		t.Logf("| # | candidate | words | exact (3 decimals) | within 0.01 | mean abs Δ |")
		t.Logf("| - | --------- | ----- | ------------------ | ----------- | ---------- |")

		for i, s := range rows[:min(probeTop(), len(rows))] {
			t.Logf("| %d | %s | %d | %.1f %% | %.1f %% | %.4f |", i+1, s.name, s.words,
				100*float64(s.exact)/float64(s.words), 100*float64(s.near)/float64(s.words), s.sumAbs/float64(s.words))
		}
	}

	report("all sets", func(s probScore) bool { return !strings.Contains(s.name, "[") })

	for _, set := range sets {
		report(set, func(s probScore) bool { return strings.HasSuffix(s.name, "["+set+"]") })
	}
}

// probeTop is the number of leaderboard rows (WHISTLE_PROBE_TOP).
func probeTop() int {
	n, err := strconv.Atoi(os.Getenv("WHISTLE_PROBE_TOP"))
	if err == nil && n > 0 {
		return n
	}

	return 12
}

// withLevels returns the items' inputs with the silence gate's levels.
func withLevels(t *testing.T, items []*item) map[*item]Input {
	t.Helper()

	gate, err := silence.New()
	if err != nil {
		t.Fatal(err)
	}

	out := map[*item]Input{}

	for _, it := range items {
		pcm, err := frontend.ReadWAV(it.Clip)
		if err != nil {
			t.Fatal(err)
		}

		in := input(it)

		in.Levels, err = gate.Levels(pcm)
		if err != nil {
			t.Fatal(err)
		}

		out[it] = in
	}

	return out
}

// TestProbeLeader reports Default on every set, teacher forced: the word
// strings, the times (train: librispeech; held out: fixtures, keywords,
// depth) and the probabilities, beside Default without the onset rule and
// without the core tokens.
func TestProbeLeader(t *testing.T) {
	ps := openSetup(t)
	items := loadItems(t, ps)
	ins := withLevels(t, items)

	noOnset, noCore := Default(), Default()
	noOnset.Onset, noCore.Core = 0, false

	for _, v := range []struct {
		name string
		opt  Options
	}{{"Default", Default()}, {"without the onset rule", noOnset}, {"without the core tokens", noCore}} {
		t.Logf("── %s: `%s`", v.name, v.opt)
		t.Logf("| set %s probability exact / within 0.01 / mean abs Δ |", timeHeader)

		total := map[bool]*timeScore{false: {}, true: {}}

		for _, set := range sets {
			var (
				s                 timeScore
				pn, pExact, pNear int
				pAbs              float64
			)

			for _, it := range items {
				if it.Set != set {
					continue
				}

				timed, err := Align(ps.tok, it.Tokens, it.LP, ins[it], v.opt)
				if err != nil {
					t.Fatal(err)
				}

				sp := make([]Span, len(timed))

				for i, w := range timed {
					if w.Text != it.Words[i].Word {
						t.Fatalf("%s word %d: %q, engine %q", it.ID, i, w.Text, it.Words[i].Word)
					}

					sp[i] = Span{Start: w.Start, End: w.End}
					d := math.Abs(w.Probability - it.Words[i].Probability)
					pn++
					pAbs += d

					if math.Round(w.Probability*1000) == math.Round(it.Words[i].Probability*1000) {
						pExact++
					}

					if d <= 0.01+1e-9 {
						pNear++
					}
				}

				s.score(it, sp)
			}

			total[set == "librispeech"].add(&s)
			t.Logf("| %s | %s | %.1f %% / %.1f %% / %.4f |", set, &s, 100*float64(pExact)/float64(pn),
				100*float64(pNear)/float64(pn), pAbs/float64(pn))
		}

		t.Logf("| held out (fixtures, keywords, depth) | %s | |", total[false])
	}
}
