//go:build oracle

// The silence gate fit (PLAN.md 6.6, U10, docs/FINDINGS.md F-031). The
// engine's verdicts on the 6.1 silence sweep and length limits
// (testdata/oracle/silence, testdata/oracle/limits) are the only evidence:
// whether each of 340 signals (the two over 30 s are errors, not verdicts)
// returned exactly the empty JSON. TestFitGate scores a grid of
// percentile-spread measures over frame levels: the band (mel channel
// ranges of the front end's power spectrogram, flat FFT bins, the time
// domain), the percentile pair and the percentile definition. For each it
// finds the threshold with the fewest misclassified signals and, when none
// is wrong, the interval of thresholds that separate the two classes. The
// framing of the winner (window, padding) is varied last. Run it with
//
//	go test -tags oracle -count=1 -v -run TestFitGate ./internal/silence/
//
// WHISTLE_FIT_TOP sets the number of leaderboard rows (default 25).

package silence

import (
	"cmp"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/frontend"
)

// band turns a power spectrogram (or the samples) into one energy per frame.
type band struct {
	name   string
	energy func(spec []float32, frames int, pcm []float32) []float64
}

// melBand sums the mel channels first … last of each frame.
func melBand(first, last int) band {
	values := cact.SlaneyMel(frontend.SampleRate, frontend.NFFT, frontend.Mels)
	w := make([]float64, frontend.Bins)

	for m := first; m <= last; m++ {
		for k := range w {
			w[k] += float64(values[k*frontend.Mels+m])
		}
	}

	return band{name: fmt.Sprintf("mel %d-%d", first, last), energy: weighted(w)}
}

// binBand sums the FFT bins first … last with weight 1.
func binBand(first, last int) band {
	w := make([]float64, frontend.Bins)
	for k := first; k <= last; k++ {
		w[k] = 1
	}

	return band{name: fmt.Sprintf("bins %d-%d", first, last), energy: weighted(w)}
}

func weighted(w []float64) func([]float32, int, []float32) []float64 {
	return func(spec []float32, frames int, _ []float32) []float64 {
		out := make([]float64, frames)

		for t := range frames {
			for k, x := range w {
				out[t] += float64(spec[t*frontend.Bins+k]) * x
			}
		}

		return out
	}
}

// rmsBand is the mean square of 400 samples every 160, no window, no
// filter: the time-domain baseline.
var rmsBand = band{name: "time 400/160", energy: func(_ []float32, _ int, pcm []float32) []float64 {
	var out []float64

	for s := 0; s+400 <= len(pcm); s += 160 {
		var e float64
		for _, v := range pcm[s : s+400] {
			e += float64(v) * float64(v)
		}

		out = append(out, e/400)
	}

	return out
}}

// pair is a percentile pair and its definition.
type pair struct {
	hi, lo float64
	linear bool // numpy's linear interpolation; nearest rank otherwise
}

func (p pair) String() string {
	def := "rank"
	if p.linear {
		def = "linear"
	}

	return fmt.Sprintf("p%g−p%g %s", p.hi, p.lo, def)
}

func (p pair) at(sorted []float64, q float64) float64 {
	n := len(sorted)
	if !p.linear {
		return sorted[max(0, min(n-1, int(math.Ceil(q/100*float64(n)))-1))]
	}

	pos := q / 100 * float64(n-1)
	i := min(int(pos), n-1)

	if i == n-1 || pos == float64(i) {
		return sorted[i]
	}

	return sorted[i] + (pos-float64(i))*(sorted[i+1]-sorted[i])
}

// sortedDB is the frame energies in dB, ascending.
func sortedDB(energy []float64) []float64 {
	db := make([]float64, len(energy))
	for i, e := range energy {
		db[i] = 10 * math.Log10(e)
	}

	slices.Sort(db)

	return db
}

// spread is p's spread of the ascending frame levels db, with Spread's
// edge cases.
func (p pair) spread(db []float64) float64 {
	if len(db) == 0 {
		return 0
	}

	hi := p.at(db, p.hi)
	if math.IsInf(hi, -1) {
		return 0
	}

	return hi - p.at(db, p.lo)
}

// row is one measure's fit.
type row struct {
	key       string
	errs      int
	maxEmpty  float64 // the largest spread of an empty signal
	minOpen   float64 // the smallest spread of a transcribed one
	threshold float64
	wrong     []string
}

// fitThreshold scores spreads against the verdicts: the threshold (silent
// iff spread ≤ threshold) with the fewest errors; with none, the middle of
// the separating interval.
func fitThreshold(sigs []signal, spreads []float64) row {
	r := row{maxEmpty: math.Inf(-1), minOpen: math.Inf(1), errs: len(sigs) + 1}

	for i, s := range sigs {
		if s.Empty {
			r.maxEmpty = max(r.maxEmpty, spreads[i])
		} else {
			r.minOpen = min(r.minOpen, spreads[i])
		}
	}

	cands := slices.Clone(spreads)
	slices.Sort(cands)

	for _, t := range slices.Compact(cands) {
		errs := 0

		for i, s := range sigs {
			if (spreads[i] <= t) != s.Empty {
				errs++
			}
		}

		if errs < r.errs {
			r.errs, r.threshold = errs, t
		}
	}

	if r.errs == 0 {
		r.threshold = (r.maxEmpty + r.minOpen) / 2
	}

	for i, s := range sigs {
		if (spreads[i] <= r.threshold) != s.Empty {
			r.wrong = append(r.wrong, fmt.Sprintf("%s %.3f", s.Name, spreads[i]))
		}
	}

	return r
}

func (r row) String() string {
	gap := "—"
	if r.errs == 0 {
		gap = fmt.Sprintf("(%.3f, %.3f]", r.maxEmpty, r.minOpen)
		if r.maxEmpty < ThresholdDB && ThresholdDB < r.minOpen {
			gap += " ∋ 5"
		}
	}

	return fmt.Sprintf("%3d wrong  T %7.3f  %-24s %s %v", r.errs, r.threshold, gap, r.key, r.wrong[:min(len(r.wrong), 6)])
}

// spectrograms computes every signal's power spectrogram under opts.
func spectrograms(t *testing.T, sigs []signal, opts frontend.Options) ([][]float32, []int) {
	t.Helper()

	bank, err := frontend.NewMelBank(cact.SlaneyMel(frontend.SampleRate, frontend.NFFT, frontend.Mels),
		[]int{frontend.Bins, frontend.Mels})
	if err != nil {
		t.Fatal(err)
	}

	fe, err := frontend.New(bank, opts)
	if err != nil {
		t.Fatal(err)
	}

	specs, frames := make([][]float32, len(sigs)), make([]int, len(sigs))

	for i, s := range sigs {
		// Reflect padding cannot frame 256 samples or fewer; leave them frameless.
		if fe.FrameCount(len(s.PCM)) == 0 || (opts.Padding == frontend.PadReflect && len(s.PCM) <= frontend.NFFT/2) {
			continue
		}

		specs[i], frames[i], err = fe.Spectrogram(s.PCM)
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}
	}

	return specs, frames
}

// levels is every signal's sorted frame levels in band b.
func levels(sigs []signal, specs [][]float32, frames []int, b band) [][]float64 {
	out := make([][]float64, len(sigs))
	for i, s := range sigs {
		out[i] = sortedDB(b.energy(specs[i], frames[i], s.PCM))
	}

	return out
}

func score(sigs []signal, db [][]float64, name string, p pair) row {
	spreads := make([]float64, len(sigs))
	for i := range sigs {
		spreads[i] = p.spread(db[i])
	}

	r := fitThreshold(sigs, spreads)
	r.key = name + ", " + p.String()

	return r
}

func TestFitGate(t *testing.T) {
	var sigs []signal

	for _, s := range loadSignals(t) {
		if !s.TooLong {
			sigs = append(sigs, s)
		}
	}

	empty := 0

	for _, s := range sigs {
		if s.Empty {
			empty++
		}
	}

	t.Logf("%d signals with a verdict: %d empty, %d transcribed", len(sigs), empty, len(sigs)-empty)

	specs, frames := spectrograms(t, sigs, Framing)

	bands := make([]band, 0, 17)
	bands = append(bands, rmsBand, binBand(0, 256), binBand(8, 112), melBand(0, 79), melBand(2, 79))

	for _, first := range []int{5, 6, 7} {
		for _, last := range []int{56, 57, 58, 59} {
			bands = append(bands, melBand(first, last))
		}
	}

	pairs := make([]pair, 0, 20)

	for _, hl := range [][2]float64{
		{99, 10}, {99, 9}, {99, 11}, {98, 10}, {99.5, 10}, {95, 10}, {99, 5}, {99, 20}, {100, 0}, {90, 10},
	} {
		for _, lin := range []bool{false, true} {
			pairs = append(pairs, pair{hi: hl[0], lo: hl[1], linear: lin})
		}
	}

	rows := make([]row, 0, len(bands)*len(pairs))

	for _, b := range bands {
		db := levels(sigs, specs, frames, b)
		for _, p := range pairs {
			rows = append(rows, score(sigs, db, b.name, p))
		}
	}

	slices.SortStableFunc(rows, func(a, b row) int {
		if c := cmp.Compare(a.errs, b.errs); c != 0 {
			return c
		}

		return cmp.Compare(b.minOpen-b.maxEmpty, a.minOpen-a.maxEmpty)
	})

	top, err := strconv.Atoi(os.Getenv("WHISTLE_FIT_TOP"))
	if err != nil {
		top = 25
	}

	t.Logf("leaderboard (%d measures; with none wrong, the interval of thresholds in dB that separates them):", len(rows))

	for _, r := range rows[:min(top, len(rows))] {
		t.Log(r)
	}

	t.Log("best per band:")

	for _, b := range bands {
		for _, r := range rows {
			if strings.HasPrefix(r.key, b.name+",") {
				t.Log(r)

				break
			}
		}
	}

	t.Log("the winner's percentiles (p99−p10, nearest rank) and band, framing varied:")

	win := melBand(FirstChannel, LastChannel)

	for _, v := range []struct {
		name string
		set  func(*frontend.Options)
	}{
		{"Hann symmetric, zero padding (the encoder's)", func(*frontend.Options) {}},
		{"Hann periodic", func(o *frontend.Options) { o.Window = frontend.HannPeriodic }},
		{"reflect padding", func(o *frontend.Options) { o.Padding = frontend.PadReflect }},
		{"no padding (frames inside the clip)", func(o *frontend.Options) { o.Padding = frontend.PadNone }},
		{"pre-emphasis 0.97", func(o *frontend.Options) { o.Preemphasis = frontend.Preemph097 }},
	} {
		opts := Framing
		v.set(&opts)
		s, f := spectrograms(t, sigs, opts)
		r := score(sigs, levels(sigs, s, f, win), win.name, pair{hi: HighPercentile, lo: LowPercentile})
		r.key = v.name
		t.Log(r)
	}
}
