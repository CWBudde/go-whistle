package text

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// The contract for the head's raw logit against upstream's
// forward_confidence in float32 (testdata/text/forward/README.md).
const confidenceTol = 1e-3

// confidenceCases are the fixture's cases with a <case>.confidence pair:
// the raw logit over the whole sequence and over [BOS] + prompt (tiny,
// without a prompt length, the whole sequence twice).
var confidenceCases = append(append([]string(nil), forwardCases...), "tiny")

// confidenceSpans returns a case's tokens and the two spans of
// <case>.confidence: the whole sequence and [BOS] + prompt.
func confidenceSpans(t *testing.T, fx fixture, name string) ([]int, [2]int) {
	t.Helper()

	tokens := fx.ints(t, name+".tokens")
	prompt := len(tokens)

	if fx.store.Has(name + ".prompt_len") {
		prompt = fx.ints(t, name+".prompt_len")[0]
	}

	return tokens, [2]int{len(tokens), prompt}
}

// confidenceDiffs runs trunk's Forward and head's ConfidenceHead on every
// case and returns, per case, the difference of the raw logit from the
// fixture's for both spans.
func confidenceDiffs(t *testing.T, trunk, head *Model) map[string][2]float64 {
	t.Helper()

	fx := openFixture(t)
	diffs := map[string][2]float64{}

	for _, name := range confidenceCases {
		tokens, spans := confidenceSpans(t, fx, name)
		want := fx.floats(t, name+".confidence")

		out, err := trunk.Forward(tokens, true)
		if err != nil {
			t.Fatal(err)
		}

		var d [2]float64

		for k, keep := range spans {
			got, err := head.Confidence(out, keep)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}

			d[k] = float64(got) - float64(want[k])
			t.Logf("%s over %4d positions: %+.6f, JAX %+.6f, diff %+.2e", name, keep, got, want[k], d[k])
		}

		diffs[name] = d
	}

	return diffs
}

// TestConfidenceMatchesJAX holds the float32 ConfidenceHead on
// needle3.safetensors to upstream's forward_confidence within 1e-3 on
// every case, over the whole sequence and over the prompt only.
func TestConfidenceMatchesJAX(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	if !m.HasConfidence() {
		t.Fatal("needle3.safetensors loaded without its confidence head")
	}

	for name, d := range confidenceDiffs(t, m, m) {
		for k, span := range []string{"whole sequence", "prompt"} {
			if math.Abs(d[k]) > confidenceTol {
				t.Errorf("%s, %s: |diff| %.3g > %v", name, span, math.Abs(d[k]), confidenceTol)
			}
		}
	}
}

// TestConfidenceCactDequantised runs the head on the dequantised
// needle3.cact weights in float32 and records how far they move the raw
// logit from the float reference: first the CQ4 head (probes, query,
// projection) alone on the float model's cells, then the whole dequantised
// model, whose CQ trunk moves the cells too. Neither is held to the 1e-3
// of the float head: on needle3.cact the head alone moves the logit by up
// to 0.83 and the whole model by up to 2.9, so only a non-finite or
// grossly wrong logit fails.
func TestConfidenceCactDequantised(t *testing.T) {
	requireFile(t, forwardFixture)
	requireFile(t, needle3Cact)

	src, err := Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	if !src.HasHead(cact.HeadConfidence) {
		t.Fatal("needle3.cact has no confidence head")
	}

	m, err := Load(src, src.TextConfig())
	if err != nil {
		t.Fatal(err)
	}

	for _, run := range []struct {
		what  string
		trunk *Model
	}{{"CQ4 head on the float cells", loadFloat(t)}, {"dequantised model", m}} {
		t.Run(run.what, func(t *testing.T) {
			var worst float64

			for name, d := range confidenceDiffs(t, run.trunk, m) {
				for k := range d {
					worst = max(worst, math.Abs(d[k]))

					if math.IsNaN(d[k]) || math.Abs(d[k]) > 5 {
						t.Errorf("%s: the raw logit moved by %+.3g", name, d[k])
					}
				}
			}

			t.Logf("largest |diff| against the float reference: %.4g", worst)
		})
	}
}

// withoutHead serves src without the names of one probe head.
type withoutHead struct {
	*Source

	key string
}

func (w withoutHead) Has(name string) bool {
	return !strings.HasPrefix(name, w.key) && w.Source.Has(name)
}

// TestConfidenceNoHead: a model whose source has no confidence head loads
// and runs, and Confidence reports ErrNoHead.
func TestConfidenceNoHead(t *testing.T) {
	requireFile(t, needle3Safetensors)

	src, err := Open(needle3Safetensors)
	if err != nil {
		t.Fatal(err)
	}

	m, err := Load(withoutHead{src, "confidence_head."}, src.TextConfig())
	if err != nil {
		t.Fatal(err)
	}

	if m.HasConfidence() {
		t.Error("HasConfidence without the head")
	}

	out, err := m.Forward([]int{1, 2, 3}, true)
	if err != nil {
		t.Fatal(err)
	}

	_, err = m.Confidence(out, 3)
	if !errors.Is(err, ErrNoHead) {
		t.Errorf("Confidence without the head: %v, want ErrNoHead", err)
	}
}

// TestConfidenceArgs: Confidence needs the per-block lane means and a
// span of at least one position within the output.
func TestConfidenceArgs(t *testing.T) {
	m := loadFloat(t)

	tokens := []int{1, 2, 3}

	plain, err := m.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	_, err = m.Confidence(plain, len(tokens))
	if err == nil {
		t.Error("Confidence without hidden states: no error")
	}

	out, err := m.Forward(tokens, true)
	if err != nil {
		t.Fatal(err)
	}

	for _, keep := range []int{0, -1, len(tokens) + 1} {
		_, err = m.Confidence(out, keep)
		if err == nil {
			t.Errorf("keep %d of %d positions: no error", keep, len(tokens))
		}
	}

	// Causal: the first positions of a longer output pool as the shorter
	// sequence does.
	short, err := m.Forward(tokens[:2], true)
	if err != nil {
		t.Fatal(err)
	}

	a, err := m.Confidence(out, 2)
	if err != nil {
		t.Fatal(err)
	}

	b, err := m.Confidence(short, 2)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(a-b)) > 1e-5 {
		t.Errorf("keep 2 of 3 positions %v, of 2 positions %v", a, b)
	}
}
