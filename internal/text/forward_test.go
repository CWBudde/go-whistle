package text

import (
	"errors"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// Pinned files as laid out by `just fetch-needle3`, and the JAX fixture of
// `just text-forward-fixtures` (testdata/text/forward/README.md).
const (
	needle3Safetensors = "../../models/needle3/needle3.safetensors"
	needle3Cact        = "../../models/needle3/needle3.cact"
	forwardFixture     = "../../testdata/text/forward/forward.safetensors"
)

// The contract: cosine against the float32 JAX reference and top-1 equal
// at every position (PLAN.md 11.2).
const minCos = 0.9999

// forwardCases are the fixture's prompts with their greedy completions;
// longCase passes the 1024-position sliding window of the local layers.
var forwardCases = []string{"timer_en", "message_de", "lights_two_calls", longCase}

const longCase = "long_tools"

func requireFile(t testing.TB, path string) {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model", path)
	}
}

var (
	floatOnce    sync.Once
	floatModel   *Model
	errFloatLoad error
)

// loadFloat loads the model from needle3.safetensors once per test binary.
func loadFloat(t *testing.T) *Model {
	t.Helper()
	requireFile(t, needle3Safetensors)

	floatOnce.Do(func() {
		src, err := Open(needle3Safetensors)
		if err != nil {
			errFloatLoad = err

			return
		}

		floatModel, errFloatLoad = Load(src, src.TextConfig())
	})

	if errFloatLoad != nil {
		t.Fatal(errFloatLoad)
	}

	return floatModel
}

type fixture struct {
	store *safetensors.Store
}

func openFixture(t testing.TB) fixture {
	t.Helper()

	store, err := safetensors.OpenStore(forwardFixture, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}

	t.Cleanup(store.Close)

	return fixture{store}
}

func (f fixture) floats(t testing.TB, name string) []float32 {
	t.Helper()

	tensor, err := f.store.Tensor(name)
	if err != nil {
		t.Fatalf("fixture: %v", err)
	}

	return tensor.Data
}

// ints reads an I64 tensor; the store widens it to float32, exact for
// token ids.
func (f fixture) ints(t testing.TB, name string) []int {
	t.Helper()

	data := f.floats(t, name)

	out := make([]int, len(data))
	for i, v := range data {
		out[i] = int(v)
	}

	return out
}

// final returns a case's reference final rows and the position of the
// first: every position, or for the long case those from <case>.final_from
// on.
func (f fixture) final(t *testing.T, name string) ([]float32, int) {
	t.Helper()

	from := 0
	if f.store.Has(name + ".final_from") {
		from = f.ints(t, name+".final_from")[0]
	}

	return f.floats(t, name+".final"), from
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64

	for k := range a {
		ab += float64(a[k]) * float64(b[k])
		aa += float64(a[k]) * float64(a[k])
		bb += float64(b[k]) * float64(b[k])
	}

	if aa == 0 && bb == 0 {
		return 1
	}

	return ab / math.Sqrt(aa*bb)
}

// rowCos returns the smallest cosine of rows of width w and the row.
func rowCos(a, b []float32, w int) (float64, int) {
	worst, at := 2.0, -1

	for r := range len(a) / w {
		if c := cosine(a[r*w:(r+1)*w], b[r*w:(r+1)*w]); c < worst {
			worst, at = c, r
		}
	}

	return worst, at
}

func argmax(row []float32) int {
	best := 0

	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}

	return best
}

// TestForwardMatchesJAX runs the float32 model from needle3.safetensors on
// the fixture's prompts and greedy completions and compares every
// position's final hidden state and next-token top-1, the last positions'
// logits and the last position's per-block hidden states with upstream's
// SimpleAttentionNetwork in float32 (docs/FINDINGS.md F-045).
func TestForwardMatchesJAX(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	fx := openFixture(t)
	d, v := m.cfg.Dim, m.cfg.OutVocab

	for _, name := range forwardCases {
		t.Run(name, func(t *testing.T) {
			tokens := fx.ints(t, name+".tokens")
			wantArg := fx.ints(t, name+".argmax")
			margin := fx.floats(t, name+".margin")
			wantFinal, from := fx.final(t, name)
			wantLast := fx.floats(t, name+".logits_last")
			wantHidden := fx.floats(t, name+".hidden_last")

			out, err := m.Forward(tokens, true)
			if err != nil {
				t.Fatal(err)
			}

			n := len(tokens)
			fin, finRow := rowCos(out.Final[from*d:], wantFinal, d)
			all := cosine(out.Final[from*d:], wantFinal)

			last := len(wantLast) / v
			lc, lcRow := rowCos(out.Logits[(n-last)*v:], wantLast, v)

			hid, hidLayer := 2.0, -1

			for l, h := range out.Hidden {
				if c := cosine(h[(n-1)*d:], wantHidden[l*d:(l+1)*d]); c < hid {
					hid, hidLayer = c, l
				}
			}

			mism, minMargin := 0, math.Inf(1)

			for p := range n {
				minMargin = min(minMargin, float64(margin[p]))

				if got := argmax(out.Logits[p*v : (p+1)*v]); got != wantArg[p] {
					mism++

					t.Errorf("position %d: top-1 %d, JAX %d (JAX margin %.4g)", p, got, wantArg[p], margin[p])
				}
			}

			t.Logf("%d positions: final cos %.9f (worst row %.9f at %d); logits_last worst %.9f at %d; "+
				"hidden_last worst %.9f at layer %d; top-1 %d/%d equal, smallest JAX margin %.4g",
				n, all, fin, from+finRow, lc, n-last+lcRow, hid, hidLayer, n-mism, n, minMargin)

			for _, c := range []struct {
				what string
				cos  float64
			}{{"final row", fin}, {"logits_last row", lc}, {"hidden_last layer", hid}} {
				if c.cos < minCos {
					t.Errorf("%s cosine %.9f < %v", c.what, c.cos, minCos)
				}
			}
		})
	}
}

// TestForwardBandMatters is the long case's control: with every layer on
// the plain causal mask, as before the band, the model must miss the
// reference somewhere past the window (cosine or top-1), so the fixture
// tells the banded mask from the causal one. The positions before the
// window must not change at all.
func TestForwardBandMatters(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	fx := openFixture(t)
	d, v, w := m.cfg.Dim, m.cfg.OutVocab, m.cfg.SlidingWindow
	tokens := fx.ints(t, longCase+".tokens")
	wantArg := fx.ints(t, longCase+".argmax")
	wantFinal, from := fx.final(t, longCase)

	banded, err := m.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	setBand(t, m, 0)
	t.Cleanup(func() { setBand(t, m, w) })

	causal, err := m.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	if len(tokens) <= w || from > w {
		t.Fatalf("%d positions, final rows from %d: nothing past the window of %d", len(tokens), from, w)
	}

	if !slices.Equal(causal.Final[:w*d], banded.Final[:w*d]) {
		t.Error("the band changed positions before the window")
	}

	fin, row := rowCos(causal.Final[from*d:], wantFinal, d)
	mism := 0

	for p := range tokens {
		if argmax(causal.Logits[p*v:(p+1)*v]) != wantArg[p] {
			mism++
		}
	}

	t.Logf("without the band: final worst row cosine %.6f at %d, top-1 %d/%d differ", fin, from+row, mism,
		len(tokens))

	if fin >= minCos && mism == 0 {
		t.Error("the long case does not tell the banded mask from the causal one")
	}
}

// setBand sets the window of the local layers (Load's SlidingWindow, 0 for
// none).
func setBand(t *testing.T, m *Model, w int) {
	t.Helper()

	for l, b := range m.blocks {
		if !slices.Contains(m.cfg.GlobalLayers, l) {
			err := b.SetWindow(w)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

// TestForwardTooLong: past MaxSeqLen, the RoPE and the trained length,
// Forward fails without running.
func TestForwardTooLong(t *testing.T) {
	m := loadFloat(t)

	_, err := m.Forward(make([]int, m.cfg.MaxSeqLen+1), false)
	if !errors.Is(err, ErrTooLong) {
		t.Errorf("%d positions: %v, want ErrTooLong", m.cfg.MaxSeqLen+1, err)
	}
}

// TestForwardTinyAllLayers compares the embedding, every block's lane mean
// at every position and every position's logits on the fixture's short
// plain-text prompt.
func TestForwardTinyAllLayers(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	fx := openFixture(t)
	d, v := m.cfg.Dim, m.cfg.OutVocab

	tokens := fx.ints(t, "tiny.tokens")
	wantEmbed := fx.floats(t, "tiny.embed")
	wantHidden := fx.floats(t, "tiny.hidden")
	wantLogits := fx.floats(t, "tiny.logits")

	out, err := m.Forward(tokens, true)
	if err != nil {
		t.Fatal(err)
	}

	n := len(tokens)

	if c, _ := rowCos(out.Embed, wantEmbed, d); c < 1-1e-12 {
		t.Errorf("embed cosine %.12f", c)
	}

	for l, h := range out.Hidden {
		ref := wantHidden[l*n*d : (l+1)*n*d]
		c, row := rowCos(h, ref, d)

		var peak float64
		for _, x := range ref {
			peak = max(peak, math.Abs(float64(x)))
		}

		t.Logf("layer %2d: worst row cosine %.9f at %d, max |ref| %.4g", l, c, row, peak)

		if c < minCos {
			t.Errorf("layer %d: cosine %.9f < %v", l, c, minCos)
		}
	}

	c, row := rowCos(out.Logits, wantLogits, v)
	t.Logf("logits: worst row cosine %.9f at %d", c, row)

	if c < minCos {
		t.Errorf("logits cosine %.9f < %v", c, minCos)
	}

	for p := range n {
		if got, want := argmax(out.Logits[p*v:(p+1)*v]), argmax(wantLogits[p*v:(p+1)*v]); got != want {
			t.Errorf("position %d: top-1 %d, JAX %d", p, got, want)
		}
	}
}

// TestForwardCactDequantised runs the same prompts on the dequantised
// needle3.cact weights in float32: the CQ weights alone, without the
// engine's int8 activations and KV. It records how far they move the
// model from the float reference; the bound is loose (PLAN.md 11.5).
func TestForwardCactDequantised(t *testing.T) {
	requireFile(t, forwardFixture)
	requireFile(t, needle3Cact)

	src, err := Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	m, err := Load(src, src.TextConfig())
	if err != nil {
		t.Fatal(err)
	}

	fx := openFixture(t)
	d, v := m.cfg.Dim, m.cfg.OutVocab

	for _, name := range forwardCases {
		tokens := fx.ints(t, name+".tokens")
		wantArg := fx.ints(t, name+".argmax")
		wantFinal, from := fx.final(t, name)
		prompt := fx.ints(t, name+".prompt_len")[0]

		out, err := m.Forward(tokens, false)
		if err != nil {
			t.Fatal(err)
		}

		n, equal, equalGen := len(tokens), 0, 0

		for p := range n {
			if argmax(out.Logits[p*v:(p+1)*v]) == wantArg[p] {
				equal++

				if p >= prompt-1 {
					equalGen++
				}
			}
		}

		fin, _ := rowCos(out.Final[from*d:], wantFinal, d)
		all := cosine(out.Final[from*d:], wantFinal)
		t.Logf("%s: final cos %.6f (worst row %.6f), top-1 %d/%d equal, %d/%d on the completion",
			name, all, fin, equal, n, equalGen, n-prompt+1)

		if all < 0.9 {
			t.Errorf("%s: dequantised weights moved the final states below cosine 0.9", name)
		}
	}
}
