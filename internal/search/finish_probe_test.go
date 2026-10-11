//go:build oracle

// The de-paare-missed probe (PLAN.md Phase 10, docs/FINDINGS.md F-042).
// With the keyword "Paare" the engine writes "… Adduktionsplan …" on
// fleurs-de_de-10308608197319017935, our search "… Adoptionsplan …":
// weaker hypotheses end one token earlier and fill Whisper's 5 finished
// slots before the leader ends (F-029). Two tests:
//
// TestProbeFinish scores variants of the finished-hypothesis bookkeeping
// (finRule) with the real search on every oracle set the search answers
// for: the 53 keyword cases and the 32 keyword sets of
// testdata/oracle/kwsets (`oracle kwsets`) under F-029's bias, the 32
// fixture clips and the 192 clips × depths 2–7, and lists every case a
// variant wins or loses against Whisper's rule.
//
// TestProbeKeywordSets scores a grid of keyword biases (weight per token
// or per character, completion bonus, revocation, spellings, and the
// transient, icefall-style and once-per-keyword completions) on the memo
// of keywords_probe_test.go against the 53 cases and the 32 sets, ranked
// by the fixture cases lost against F-029's bias (DefaultBias until
// F-044), then by the cases equal.
//
//	go test -tags oracle -count=1 -v -timeout 60m -run 'TestProbe(Finish|KeywordSets)$' ./internal/search/
//
// WHISTLE_PROBE_TOP sets the leaderboard rows (default 25).

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

	"github.com/cwbudde/go-whistle/internal/decoder"
)

// paareCase is the keyword case under probe.
const paareCase = "de-paare-missed"

// kwSetCases reads testdata/oracle/kwsets/manifest.json: the engine's text
// on one clip with each keyword set, as keyword cases of kind "set".
func kwSetCases(t testing.TB) []keywordCase {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(oracleDir, "kwsets", "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var m struct {
		Clip string `json:"clip"`
		Sets []struct {
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
			Name: "set " + strings.Join(s.Keywords, "|"), Clip: m.Clip, Language: s.Language, Kind: "set",
			Keywords: s.Keywords, TextWith: s.Text,
		}
	}

	return out
}

// finVariant is one finished-hypothesis rule under probe.
type finVariant struct {
	name string
	rule finRule
}

var finVariants = []finVariant{
	{"Whisper: 5 slots (Default)", finRule{}},
	{"a finished text takes one slot", finRule{dedupFin: true}},
	{"one live hypothesis per text", finRule{dedupLive: true}},
	{"both", finRule{dedupFin: true, dedupLive: true}},
	{"6 slots (patience 1.2)", finRule{slots: 6}},
	{"7 slots (patience 1.4)", finRule{slots: 7}},
	{"10 slots (patience 2)", finRule{slots: 10}},
	{"the final pick weighs the live ones too", finRule{pickLive: true}},
	{"only EOS among the step's best 5 count", finRule{countTopK: true}},
	{"5 slots and the best finished ≥ the best live", finRule{stopLive: true}},
}

// finJob is one search of TestProbeFinish.
type finJob struct {
	set, name string
	d         *decoder.Decoder
	mem       *decoder.Memory
	kw        *Keywords
	want      string
}

func TestProbeFinish(t *testing.T) {
	d, clips := loadSetup(t, clipIDs(t)...)
	byID := map[string]clip{}

	for _, c := range clips {
		byID[c.ID] = c
	}

	var jobs []finJob

	for set, cases := range map[string][]keywordCase{"keywords": keywordCases(t), "sets": kwSetCases(t)} {
		for _, kc := range cases {
			kw, err := NewKeywords(d.Tokenizer(), kc.Keywords, f029Bias())
			if err != nil {
				t.Fatal(err)
			}

			jobs = append(jobs, finJob{set, kc.Name, d, byID[kc.Clip].Mem, kw, kc.TextWith})
		}
	}

	for _, c := range clips {
		jobs = append(jobs, finJob{"clips", c.ID, d, c.Mem, nil, c.Text})
	}

	texts := depthTexts(t)

	for depth := 2; depth <= 7; depth++ {
		dd, err := d.WithDepth(depth)
		if err != nil {
			t.Fatal(err)
		}

		for _, c := range clips {
			name := fmt.Sprintf("%s.d%d", c.ID, depth)
			jobs = append(jobs, finJob{"depth", name, dd, c.Mem, nil, texts[name].Text})
		}
	}

	slices.SortStableFunc(jobs, func(a, b finJob) int { return cmp.Compare(a.name, b.name) })

	sets := []string{"keywords", "sets", "clips", "depth"}
	results := make([][]string, len(finVariants))

	for v, fv := range finVariants {
		results[v] = finRun(t, jobs, fv.rule)
	}

	t.Logf("| rule | %s | keyword cases | keyword sets | clips | depths 2–7 | won | lost |", paareCase)
	t.Logf("| ---- | --- | ------------- | ------------ | ----- | ---------- | --- | ---- |")

	for v, fv := range finVariants {
		exact, total := map[string]int{}, map[string]int{}
		paare := "differs"

		var won, lost []string

		for j, jb := range jobs {
			total[jb.set]++

			ok := results[v][j] == jb.want
			if ok {
				exact[jb.set]++
			}

			if jb.name == paareCase && ok {
				paare = "equal"
			}

			switch base := results[0][j] == jb.want; {
			case ok && !base:
				won = append(won, jb.name)
			case !ok && base:
				lost = append(lost, jb.name)
			}
		}

		cols := make([]string, len(sets))
		for i, set := range sets {
			cols[i] = fmt.Sprintf("%d/%d", exact[set], total[set])
		}

		t.Logf("| %s | %s | %s | %s | %s |", fv.name, paare, strings.Join(cols, " | "), strings.Join(won, ", "),
			strings.Join(lost, ", "))
	}
}

// finRun runs every job under rule, the jobs on half the cores (the search
// runs a step's beams in parallel), and returns the texts.
func finRun(t *testing.T, jobs []finJob, rule finRule) []string {
	t.Helper()

	out := make([]string, len(jobs))
	errs := make([]error, len(jobs))
	idx := make(chan int)

	var wg sync.WaitGroup

	for range max(runtime.GOMAXPROCS(0)/2, 1) {
		wg.Go(func() {
			for j := range idx {
				opt := Default()
				opt.fin = rule

				if jobs[j].kw != nil {
					opt.Adjust = jobs[j].kw.Adjust
				}

				r, err := Beam(jobs[j].d, jobs[j].mem, opt)
				out[j], errs[j] = r.Text, err
			}
		})
	}

	for j := range jobs {
		idx <- j
	}

	close(idx)
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}

	return out
}

// biasName is Bias.String with the probe axes.
func biasName(b Bias) string {
	name := b.String()

	if b.transient {
		name += " transient"
	}

	if b.echo {
		name += "+echo"
	}

	if b.once {
		name += " once"
	}

	return name
}

// kwSetGrid is the biases TestProbeKeywordSets scores, F-029's bias (then
// DefaultBias) first.
func kwSetGrid() []Bias {
	out := []Bias{f029Bias()}
	add := func(f func(*Bias)) {
		b := f029Bias()
		f(&b)

		if !slices.Contains(out, b) {
			out = append(out, b)
		}
	}

	for _, rv := range []float64{0.5, 1} {
		for _, c := range []float64{0, 0.25, 0.5, 0.75, 1, 1.5, 2} {
			for _, w := range []float64{1.5, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.5} {
				add(func(b *Bias) { b.Weight, b.Complete, b.Revoke = w, c, rv })
			}
		}
	}

	for _, c := range []float64{0, 0.5, 1} {
		for _, w := range []float64{0.375, 0.5, 0.625, 0.75, 1, 1.25, 1.5} {
			add(func(b *Bias) { b.Weight, b.Complete, b.PerChar = w, c, true })
		}
	}

	for _, f := range []Form{FormSpace | FormBare | FormCapital, FormSpace | FormBare | FormLower, FormSpace | FormBare} {
		for _, c := range []float64{0.25, 1} {
			for _, w := range []float64{1.9, 2, 2.1} {
				add(func(b *Bias) { b.Weight, b.Complete, b.Forms = w, c, f })
			}
		}
	}

	for _, once := range []bool{false, true} {
		for mode := range 3 {
			for _, c := range []float64{0, 0.5, 1, 2} {
				for _, w := range []float64{1.5, 1.75, 2, 2.1, 2.25} {
					add(func(b *Bias) {
						b.Weight, b.Complete, b.once, b.transient, b.echo = w, c, once, mode > 0, mode == 2
					})
				}
			}
		}
	}

	return out
}

func TestProbeKeywordSets(t *testing.T) {
	d, cases, clips := kwSetup(t)
	cases = append(cases, kwSetCases(t)...)

	grid := kwSetGrid()
	vs := make([]kwVariant, len(grid))

	for i, b := range grid {
		vs[i] = kwVariant{bias: b, stop: StopK, admit: AdmitAboveLive}
	}

	all := scoreKeywords(t, d, cases, clips, vs)

	type row struct {
		name               string
		fix, set, nf, ns   int
		paare              bool
		won, lost, setMiss []string
	}

	rows := make([]row, len(all))

	for v, s := range all {
		r := row{name: biasName(grid[v])}

		for i, o := range s.cases {
			c := cases[i]
			isSet := c.Kind == "set"

			switch {
			case isSet:
				r.ns++
			default:
				r.nf++
			}

			switch {
			case o.exact && isSet:
				r.set++
			case o.exact:
				r.fix++
			case isSet:
				r.setMiss = append(r.setMiss, strings.TrimPrefix(c.Name, "set "))
			}

			if c.Name == paareCase {
				r.paare = o.exact
			}

			if !isSet && o.exact != all[0].cases[i].exact {
				if o.exact {
					r.won = append(r.won, c.Name)
				} else {
					r.lost = append(r.lost, c.Name)
				}
			}
		}

		rows[v] = r
	}

	format := func(r row) string {
		return fmt.Sprintf("| `%s` | %d/%d | %d/%d | %v | %s | %s | %s |", r.name, r.fix, r.nf, r.set, r.ns, r.paare,
			strings.Join(r.won, ", "), strings.Join(r.lost, ", "), strings.Join(r.setMiss, "; "))
	}

	t.Logf("── %d biases × %d cases; F-029's bias:", len(grid), len(cases))
	t.Logf("%s", format(rows[0]))

	slices.SortStableFunc(rows, func(a, b row) int {
		return cmp.Or(cmp.Compare(len(a.lost), len(b.lost)), cmp.Compare(b.fix+b.set, a.fix+a.set))
	})

	header := "| bias | keyword cases | keyword sets | " + paareCase + " | won | lost | sets missed |"
	t.Logf("%s", header)
	t.Logf("| ---- | ------------- | ------------ | --- | --- | ---- | ----------- |")

	for _, r := range rows[:min(probeEnv("WHISTLE_PROBE_TOP", 25), len(rows))] {
		t.Logf("%s", format(r))
	}

	var (
		fixes, clean int
		best         *row
	)

	for i, r := range rows {
		if !r.paare {
			continue
		}

		fixes++

		if len(r.lost) == 0 {
			clean++
		}

		if best == nil {
			best = &rows[i]
		}
	}

	t.Logf("%d of %d biases make %s equal; %d of them lose no keyword case", fixes, len(rows), paareCase, clean)

	if best != nil {
		t.Logf("the best of them: %s", format(*best))
	}
}
