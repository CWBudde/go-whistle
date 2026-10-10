//go:build oracle

// The depth ladder probe (PLAN.md 6.4, docs/FINDINGS.md F-030). The engine
// decodes with --audio-depth N on the endpoint-preserving ladder rung of N
// decoder layers; testdata/oracle/depth holds its transcripts at depths
// 2–7 for the 32 fixture clips (F-025). TestProbeDepth beam-decodes every
// clip at every depth with Default on decoder.WithDepth and counts the
// transcripts equal to the engine's at that depth, beside the full-depth
// rate (31/32, F-026) and greedy decoding. Every mismatch is teacher
// forced both ways, as in F-026: the engine's text in its most likely
// spelling against ours, by Default's ranking score. A search miss is an
// engine transcript that ranks higher under our model; a model gap one
// that ranks lower, by the margin printed. Run it with `just depth-probe`;
// WHISTLE_PROBE_MISSES=1 prints every mismatch.

package search

import (
	"fmt"
	"math"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/eval"
)

func TestProbeDepth(t *testing.T) {
	d, clips := loadSetup(t, clipIDs(t)...)
	texts := depthTexts(t)
	opt := Default()

	type row struct {
		depth, exact, greedy, engFull, miss int
		layers                              []int
		wer, cer                            float64
		gaps                                []float64 // ranking score: ours − the engine's, per model gap
		ms                                  float64
	}

	var rows []row

	for depth := 2; depth <= 8; depth++ {
		dd, err := d.WithDepth(depth)
		if err != nil {
			t.Fatal(err)
		}

		rw := row{depth: depth, layers: dd.Layers()}

		var dur time.Duration

		for _, c := range clips {
			want := c
			if depth < 8 {
				f, ok := texts[fmt.Sprintf("%s.d%d", c.ID, depth)]
				if !ok {
					t.Fatalf("%s: no depth %d fixture", c.ID, depth)
				}

				want.Language, want.Text = f.Language, f.Text
			}

			start := time.Now()

			r, err := Beam(dd, c.Mem, opt)
			if err != nil {
				t.Fatal(err)
			}

			dur += time.Since(start)

			g, err := dd.GreedyMemory(c.Mem, "")
			if err != nil {
				t.Fatal(err)
			}

			if g.Text == want.Text {
				rw.greedy++
			}

			if want.Text == c.Text {
				rw.engFull++
			}

			rw.wer += eval.WER(want.Text, r.Text) / float64(len(clips))
			rw.cer += eval.CER(want.Text, r.Text) / float64(len(clips))

			if r.Text == want.Text {
				rw.exact++

				continue
			}

			eng := force(t, dd, want, want.Language, bestTokens(t, dd, want, 16), opt)
			ours := force(t, dd, want, r.Language, r.Tokens, opt)

			kind := "search miss"
			if eng.score <= ours.score {
				kind = "model gap"

				rw.gaps = append(rw.gaps, ours.score-eng.score)
			} else {
				rw.miss++
			}

			if os.Getenv("WHISTLE_PROBE_MISSES") != "" {
				t.Logf("d%d %s: %s, score ours %.4f, engine %.4f (Δ %.4f)\n    engine: %s\n    beam:   %s", depth, c.ID,
					kind, ours.score, eng.score, ours.score-eng.score, want.Text, r.Text)
			}
		}

		rw.ms = float64(dur.Milliseconds()) / float64(len(clips))
		rows = append(rows, rw)
	}

	t.Logf("── Default on decoder.WithDepth (ladder order %v), against the engine's --audio-depth transcripts",
		decoder.LadderOrder(len(rows)+1))
	t.Logf("| depth | layers | beam exact | WER | CER | greedy exact | search misses | model gaps (max Δscore) | " +
		"engine = its full depth | beam ms per clip |")
	t.Logf("| ----- | ------ | ---------- | --- | --- | ------------ | ------------- | ----------------------- | " +
		"----------------------- | ---------------- |")

	for _, rw := range rows {
		maxGap := math.NaN()
		if len(rw.gaps) > 0 {
			maxGap = slices.Max(rw.gaps)
		}

		t.Logf("| %d | %v | %d/%d | %.4f | %.4f | %d/%d | %d | %d (%.4f) | %d/%d | %.0f |", rw.depth, rw.layers, rw.exact,
			len(clips), rw.wer, rw.cer, rw.greedy, len(clips), rw.miss, len(rw.gaps), maxGap, rw.engFull, len(clips),
			rw.ms)
	}
}
