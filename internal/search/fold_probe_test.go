//go:build oracle

// The second de-paare-missed probe (PLAN.md Phase 10, docs/FINDINGS.md
// F-044), on F-042's open list: case-folded matching (Bias.fold, share,
// wordEnd), the phrase loops (Bias.spent), the capitalised bare spelling
// (Bias.Forms) and the weights. TestProbeFold runs the real search with
// Default and every bias of foldGrid on the 53 keyword cases and the 32
// keyword sets of testdata/oracle/kwsets and the 31 keyword variants of
// testdata/oracle/kwvariants (`oracle kwvariants`: capitals, pieces from
// inside a word, word starts, short phrases), and lists against F-029's
// bias (DefaultBias before F-044) the keyword cases each one wins and
// loses and the sets and variants it misses.
//
//	go test -tags oracle -count=1 -v -timeout 120m -run 'TestProbeFold$' ./internal/search/
//
// WHISTLE_PROBE_GRID picks the grid (1 … 4, default 4; see foldGrid1 …
// foldGrid4), WHISTLE_PROBE_TOP the leaderboard rows (default 40),
// WHISTLE_PROBE_TEXTS=1 prints every text that differs from the engine's.

package search

import (
	"cmp"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
)

// foldName is biasName: Bias.String names the F-044 axes itself.
func foldName(b Bias) string { return biasName(b) }

// foldGrid is the biases of grid WHISTLE_PROBE_GRID (default 2) of
// TestProbeFold, F-029's bias first.
func foldGrid() []Bias {
	switch probeEnv("WHISTLE_PROBE_GRID", 4) {
	case 1:
		return foldGrid1()
	case 2:
		return foldGrid2()
	case 3:
		return foldGrid3()
	default:
		return foldGrid4()
	}
}

// foldGrid4 is the summary of docs/FINDINGS.md F-044 (11 biases): F-029's
// bias, with the case spellings, the spent rule and both, F-042's gain
// alone, with the case spellings (DefaultBias until 2026-10-12) and with
// both, the window that makes de-paare-missed equal (DefaultBias), the
// bias that reproduces every keyword set, and the best byte matchers.
func foldGrid4() []Bias {
	out, add := foldAdder()
	ext := func(b *Bias) { b.fold, b.spent = foldCases, true }

	add(func(b *Bias) { b.fold = foldCases })
	add(func(b *Bias) { b.spent = true })
	add(ext)
	add(func(b *Bias) { b.Weight, b.Complete = 1.8, 1.5 })
	add(func(b *Bias) { b.fold, b.Weight, b.Complete = foldCases, 1.8, 1.5 })
	add(func(b *Bias) { ext(b); b.Weight, b.Complete = 1.8, 1.5 })
	add(func(b *Bias) { ext(b); b.Weight, b.Complete = 2.15, 0 })
	add(func(b *Bias) { ext(b); b.Weight, b.Complete = 1.5, 2 })
	add(func(b *Bias) { b.fold, b.Weight, b.Complete = foldBytes, 1.8, 0 })
	add(func(b *Bias) { b.fold, b.share, b.Weight, b.Complete = foldBytes, shareToken, 2.2, 0 })

	if (*out)[7] != DefaultBias() {
		panic("foldGrid4: row 7 is not DefaultBias")
	}

	return *out
}

// foldGrid3 follows grid 2 (45 biases), all with the case spellings and
// the spent rule: a bonus per character, a smaller weight with a larger
// completion bonus, and fewer spellings at the start of the text.
func foldGrid3() []Bias {
	out, add := foldAdder()
	ext := func(b *Bias) { b.fold, b.spent = foldCases, true }

	add(ext)

	for _, w := range []float64{0.75, 0.9, 1, 1.1, 1.25} {
		for _, c := range []float64{0.5, 1, 1.5, 2} {
			add(func(b *Bias) { ext(b); b.PerChar, b.Weight, b.Complete = true, w, c })
		}
	}

	for _, wc := range [][2]float64{
		{1.5, 1.5},
		{1.5, 2},
		{1.5, 2.5},
		{1.6, 1.5},
		{1.6, 2},
		{1.6, 2.5},
		{1.7, 1.5},
		{1.7, 2},
		{1.7, 2.5},
		{1.8, 2},
		{1.8, 2.5},
	} {
		add(func(b *Bias) { ext(b); b.Weight, b.Complete = wc[0], wc[1] })
	}

	for _, wc := range [][2]float64{{2, 1}, {1.8, 1.5}, {2.15, 0}} {
		for _, f := range []func(*Bias){
			func(b *Bias) { b.Forms = FormSpace | FormBare | FormLower },
			func(b *Bias) { b.Forms = FormSpace | FormBare },
			func(b *Bias) { b.bareWritten = true },
			func(b *Bias) { b.bareWritten, b.Forms = true, FormSpace|FormBare|FormLower },
		} {
			add(func(b *Bias) { ext(b); b.Weight, b.Complete = wc[0], wc[1]; f(b) })
		}
	}

	return *out
}

// foldAdder returns the grid so far, F-029's bias first, and a function
// that adds F-029's bias changed by f unless the grid has it already.
func foldAdder() (*[]Bias, func(func(*Bias))) {
	out := []Bias{f029Bias()}

	return &out, func(f func(*Bias)) {
		b := f029Bias()
		f(&b)

		if !slices.Contains(out, b) {
			out = append(out, b)
		}
	}
}

// foldGrid2 follows grid 1 and the variants (85 biases): the token
// matcher's weight near 2.2 with a small completion bonus, with and
// without the case spellings and the spent rule, and the byte matcher
// again after its bonus moved to a token's last byte.
func foldGrid2() []Bias {
	out, add := foldAdder()

	for _, ext := range []bool{false, true} {
		for _, w := range []float64{2, 2.15, 2.2, 2.25, 2.3, 2.4} {
			for _, c := range []float64{0, 0.1, 0.2, 0.3} {
				add(func(b *Bias) {
					b.Weight, b.Complete = w, c
					if ext {
						b.fold, b.spent = foldCases, true
					}
				})
			}
		}
	}

	for _, wc := range [][2]float64{{2, 1}, {1.8, 1.5}, {1.8, 1}, {2.1, 0.25}} {
		add(func(b *Bias) { b.Weight, b.Complete, b.fold, b.spent = wc[0], wc[1], foldCases, true })
	}

	for _, share := range []foldShare{shareCanon, shareToken} {
		for _, wordEnd := range []bool{false, true} {
			for _, w := range []float64{1.5, 1.8, 2, 2.2} {
				for _, c := range []float64{0, 1} {
					add(func(b *Bias) { b.fold, b.share, b.wordEnd, b.Weight, b.Complete = foldBytes, share, wordEnd, w, c })
				}
			}
		}
	}

	return *out
}

// foldGrid1 is the first grid (167 biases): the case-folded matchers
// crossed with the weight and the completion bonus, how the byte matcher
// shares the bonus, the spent rule, fewer bare forms and F-042's leaders.
// Its byte rows in F-044 predate two fixes of the byte matcher, which
// grids 2 and 4 rerun.
func foldGrid1() []Bias {
	out, add := foldAdder()
	weights := []float64{1.5, 1.8, 2, 2.1, 2.5}
	completions := []float64{0, 0.5, 1, 1.5}

	// 1. Case-folded matching, crossed with the weight and the completion.
	for _, fold := range []foldMode{foldNone, foldCases, foldBytes} {
		for _, w := range weights {
			for _, c := range completions {
				add(func(b *Bias) { b.fold, b.Weight, b.Complete = fold, w, c })
			}
		}
	}

	// 2. How folded bytes share the bonus, and the word end.
	for _, share := range []foldShare{shareCanon, shareChar, shareToken} {
		for _, wordEnd := range []bool{false, true} {
			ws := weights
			if share == shareChar {
				ws = []float64{0.375, 0.5, 0.625, 0.75, 1}
			}

			for _, w := range ws {
				for _, c := range []float64{0, 1} {
					add(func(b *Bias) { b.fold, b.share, b.wordEnd, b.Weight, b.Complete = foldBytes, share, wordEnd, w, c })
				}
			}
		}
	}

	// 3. The loops: a completed keyword earns nothing more.
	for _, fold := range []foldMode{foldNone, foldBytes} {
		for _, w := range weights {
			for _, c := range []float64{0, 1} {
				add(func(b *Bias) { b.fold, b.spent, b.Weight, b.Complete = fold, true, w, c })
			}
		}
	}

	// 4. The capitalised bare spelling: fewer bare forms.
	for _, f := range []Form{
		FormSpace | FormCapital | FormLower, FormSpace | FormBare | FormLower,
		FormSpace | FormBare, FormSpace | FormBare | FormCapital,
	} {
		for _, w := range []float64{1.8, 2, 2.1} {
			for _, c := range []float64{0.25, 1, 1.5} {
				add(func(b *Bias) { b.Forms, b.Weight, b.Complete = f, w, c })
			}
		}
	}

	// 5. F-042's leaders.
	add(func(b *Bias) { b.Weight, b.Complete = 1.8, 1.5 })
	add(func(b *Bias) { b.Weight, b.Complete = 2.1, 0.25 })

	return *out
}

// kwVariantCases reads testdata/oracle/kwvariants/manifest.json: the
// engine's text with each keyword file on its clip, as keyword cases of
// kind "variant".
func kwVariantCases(t testing.TB) []keywordCase {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(oracleDir, "kwvariants", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var m struct {
		Sets []struct {
			Clip     string   `json:"clip"`
			Keywords []string `json:"keywords"`
			Language string   `json:"language"`
			Text     string   `json:"text"`
		} `json:"sets"`
	}

	err = json.Unmarshal(raw, &m)
	if err != nil {
		t.Fatal(err)
	}

	out := make([]keywordCase, len(m.Sets))
	for i, s := range m.Sets {
		out[i] = keywordCase{
			Name: "variant " + strings.Join(s.Keywords, "|"), Clip: s.Clip, Language: s.Language, Kind: "variant",
			Keywords: s.Keywords, TextWith: s.Text,
		}
	}

	return out
}

// foldJob is one keyword search of TestProbeFold: a keyword case, a set
// or a variant.
type foldJob struct {
	kc   keywordCase
	set  bool
	vari bool
}

func TestProbeFold(t *testing.T) {
	d, cases, clips := kwSetup(t)

	jobs := make([]foldJob, 0, len(cases)+32)
	for _, c := range cases {
		jobs = append(jobs, foldJob{kc: c})
	}

	for _, c := range kwSetCases(t) {
		jobs = append(jobs, foldJob{kc: c, set: true})
	}

	variants := kwVariantCases(t)
	for _, c := range variants {
		if _, ok := clips[c.Clip]; !ok {
			t.Fatalf("%s: clip %s has no keyword case", c.Name, c.Clip)
		}

		jobs = append(jobs, foldJob{kc: c, vari: true})
	}

	grid := foldGrid()
	texts := make([][]string, len(grid))

	type task struct{ v, j int }

	idx := make(chan task)
	errs := make(chan error, 1)

	var wg sync.WaitGroup

	for range max(runtime.GOMAXPROCS(0)/2, 1) {
		wg.Go(func() {
			for tk := range idx {
				jb := jobs[tk.j]

				kw, err := NewKeywords(d.Tokenizer(), jb.kc.Keywords, grid[tk.v])
				if err == nil {
					opt := Default()
					opt.Adjust = kw.Adjust

					var r Result

					r, err = Beam(d, clips[jb.kc.Clip].Mem, opt)
					texts[tk.v][tk.j] = r.Text
				}

				if err != nil {
					select {
					case errs <- fmt.Errorf("%s %s: %w", jb.kc.Name, foldName(grid[tk.v]), err):
					default:
					}
				}
			}
		})
	}

	for v := range grid {
		texts[v] = make([]string, len(jobs))
		for j := range jobs {
			idx <- task{v, j}
		}
	}

	close(idx)
	wg.Wait()

	select {
	case err := <-errs:
		t.Fatal(err)
	default:
	}

	type row struct {
		name                          string
		fix, set, vari                int
		paare                         bool
		won, lost, setsMiss, variMiss []string
	}

	rows := make([]row, len(grid))

	for v := range grid {
		r := row{name: foldName(grid[v])}

		for j, jb := range jobs {
			ok := texts[v][j] == jb.kc.TextWith
			base := texts[0][j] == jb.kc.TextWith

			switch {
			case jb.set && ok:
				r.set++
			case jb.set:
				r.setsMiss = append(r.setsMiss, strings.Join(jb.kc.Keywords, "|"))
			case jb.vari && ok:
				r.vari++
			case jb.vari:
				r.variMiss = append(r.variMiss, strings.Join(jb.kc.Keywords, "|"))
			case ok:
				r.fix++
			}

			if jb.kc.Name == paareCase {
				r.paare = ok
			}

			if !jb.set && !jb.vari && ok != base {
				if ok {
					r.won = append(r.won, jb.kc.Name)
				} else {
					r.lost = append(r.lost, jb.kc.Name)
				}
			}
		}

		rows[v] = r
	}

	format := func(r row) string {
		return fmt.Sprintf("| `%s` | %d/%d | %d/32 | %d/%d | %v | %s | %s | %s | %s |", r.name, r.fix, len(cases), r.set,
			r.vari, len(variants), r.paare, strings.Join(r.won, ", "), strings.Join(r.lost, ", "),
			strings.Join(r.setsMiss, "; "), strings.Join(r.variMiss, "; "))
	}

	t.Logf("── %d biases × %d searches; F-029's bias:", len(grid), len(jobs))
	t.Logf("%s", format(rows[0]))

	if os.Getenv("WHISTLE_PROBE_TEXTS") == "1" {
		for v := range grid {
			for j, jb := range jobs {
				if texts[v][j] != jb.kc.TextWith {
					t.Logf("%s | %s:\n    ours:   %s\n    engine: %s", rows[v].name, jb.kc.Name, texts[v][j], jb.kc.TextWith)
				}
			}
		}
	}

	t.Logf("── in grid order")

	for _, r := range rows {
		t.Logf("%s", format(r))
	}

	sorted := slices.Clone(rows)
	slices.SortStableFunc(sorted, func(a, b row) int {
		return cmp.Or(cmp.Compare(len(a.lost), len(b.lost)), cmp.Compare(b.fix+b.set+b.vari, a.fix+a.set+a.vari))
	})

	t.Logf("── leaderboard: fewest keyword cases lost, then the most equal")
	t.Logf("| bias | keyword cases | keyword sets | variants | %s | won | lost | sets missed | variants missed |", paareCase)
	t.Logf("| ---- | ------------- | ------------ | -------- | --- | --- | ---- | ----------- | --------------- |")

	for _, r := range sorted[:min(probeEnv("WHISTLE_PROBE_TOP", 40), len(sorted))] {
		t.Logf("%s", format(r))
	}

	fixes, clean := 0, 0

	for _, r := range rows {
		if r.paare {
			fixes++

			if len(r.lost) == 0 {
				clean++
				t.Logf("fixes %s and loses nothing: %s", paareCase, format(r))
			}
		}
	}

	t.Logf("%d of %d biases make %s equal; %d of them lose no keyword case", fixes, len(rows), paareCase, clean)
}
