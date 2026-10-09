package needle

import (
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"
)

// tinyBlock is a random Block of width 8: tinyAttention's attention,
// tinyHadamard's MLP and the three norms and the gate under prefix.
func tinyBlock(r *rand.Rand, prefix string, taps int) (tinyAttention, *mapSource) {
	ta := newTinyAttention(r, prefix, taps)
	src := tinyHadamard(r, prefix)

	for name, v := range ta.src.data {
		src.put(name, v, ta.src.shape[name]...)
	}

	for _, name := range []string{"norm_in", "post_norm", "pre_hada"} {
		src.put(prefix+name, randVec(r, 8, 0.3), 8)
	}

	src.put(prefix+"attn_gate", randVec(r, 1, 1), 1)
	ta.src = src

	return ta, src
}

// refBlock transcribes architecture.py Block in float64 (no engram).
func refBlock(ta tinyAttention, prefix string, x []float64, t int, causal bool, rope *Rope) []float64 {
	w := func(name string) []float64 { return ta.w(prefix, name) }
	norm := func(z []float64, scale string) []float64 {
		out := slices.Clone(z)
		refNormRows(out, 8, w(scale))

		return out
	}

	a := ta.refAttention(prefix, norm(x, "norm_in"), t, causal, rope)
	post := norm(a, "post_norm")
	gate := 1 / (1 + math.Exp(-w("attn_gate")[0]))
	h := make([]float64, len(x))

	for i := range x {
		h[i] = x[i] + gate*post[i]
	}

	pre := norm(h, "pre_hada")
	out := slices.Clone(h)

	for i := range t {
		mlp := refHadamard(ta.src, prefix, pre[i*8:(i+1)*8], 2, 4, 3)
		for c := range 8 {
			out[i*8+c] += mlp[c]
		}
	}

	return out
}

func TestBlockMatchesReference(t *testing.T) {
	const steps = 5

	r := rand.New(rand.NewPCG(41, 43))
	ta, src := tinyBlock(r, "layer03.", 3)

	b, err := LoadBlock(src, "layer03.")
	if err != nil {
		t.Fatal(err)
	}

	if b.Dim() != 8 {
		t.Fatalf("dim %d", b.Dim())
	}

	x := randVec(r, steps*8, 1)
	rope := NewRope(4, steps, 1e5)

	got, err := b.Forward(x, steps, MaskCausal, rope, nil)
	if err != nil {
		t.Fatal(err)
	}

	want := refBlock(ta, "layer03.", toFloat64(x), steps, true, rope)
	if d := maxAbsDiff64(got, want); d > 1e-5 {
		t.Errorf("max |Δ| %.3g", d)
	}

	// With engram: Block_3(x, k, v) = Block_3(x + α·v), as the fixture
	// script checks upstream.
	k, v := randVec(r, steps*8, 1), randVec(r, steps*8, 1)
	xIn := slices.Clone(x)

	withEngram, err := b.Forward(x, steps, MaskCausal, rope, &EngramKV{K: k, V: v})
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(x, xIn) {
		t.Error("Forward wrote to its input")
	}

	gated := toFloat64(x)
	for i := range steps {
		row, _ := refGate(gated[i*8:(i+1)*8], toFloat64(k[i*8:(i+1)*8]), toFloat64(v[i*8:(i+1)*8]))
		copy(gated[i*8:], row)
	}

	want = refBlock(ta, "layer03.", gated, steps, true, rope)
	if d := maxAbsDiff64(withEngram, want); d > 1e-5 {
		t.Errorf("with engram: max |Δ| %.3g", d)
	}
}

func TestBlockErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 7))
	_, src := tinyBlock(r, "", 0)
	call := func(_ any, err error) error { return err }

	b, err := LoadBlock(src, "")
	if err != nil {
		t.Fatal(err)
	}

	_, bad := tinyBlock(r, "", 0)
	bad.put("attn_gate", randVec(r, 2, 1), 2)

	_, wide := tinyBlock(r, "", 0)
	wide.put("pre_hada", randVec(r, 6, 1), 6)

	shortK := &EngramKV{K: nil, V: make([]float32, 8)}

	for name, err := range map[string]error{
		"attn_gate of two values": call(LoadBlock(bad, "")),
		"pre_hada of width 6":     call(LoadBlock(wide, "")),
		"missing prefix":          call(LoadBlock(src, "layer07.")),
		"ragged input":            call(b.Forward(make([]float32, 7), 1, MaskNone, nil, nil)),
		"engram k too short":      call(b.Forward(make([]float32, 8), 1, MaskNone, nil, shortK)),
	} {
		if err == nil {
			t.Errorf("accepted %s", name)
		}
	}
}

// TestBlockFixture is decoder layer 0's Block (causal, RoPE θ = 1e5, 3
// taps, no engram) against testdata/needle/block.safetensors, every
// intermediate included (docs/FINDINGS.md F-013).
func TestBlockFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "block")

	b, err := LoadBlock(src, "layer00.")
	if err != nil {
		t.Fatal(err)
	}

	if fx.metadata("mask") != "causal" || fx.metadata("taps") != strconv.Itoa(b.attn.Taps()) ||
		fx.metadata("engram") != "none" {
		t.Fatalf("fixture mask %s taps %s engram %s, block taps %d", fx.metadata("mask"), fx.metadata("taps"),
			fx.metadata("engram"), b.attn.Taps())
	}

	theta, err := strconv.ParseFloat(fx.metadata("rope_theta"), 64)
	if err != nil {
		t.Fatal(err)
	}

	x, shape := fx.get("x")
	batch, steps, dim := shape[0], shape[1], shape[2]
	rope := NewRope(b.attn.QKDim(), steps, theta)
	all := map[string][]float32{}

	for i := range batch {
		tr := &blockTrace{}

		out, err := b.forward(x[i*steps*dim:(i+1)*steps*dim], steps, MaskCausal, rope, nil, tr)
		if err != nil {
			t.Fatal(err)
		}

		for name, v := range map[string][]float32{
			"norm_in": tr.normIn, "attn": tr.attn, "post_attn": tr.postAttn, "h": tr.h,
			"pre_hada": tr.preHada, "mlp": tr.mlp, "out": out,
		} {
			all[name] = append(all[name], v...)
		}
	}

	for _, name := range []string{"norm_in", "attn", "post_attn", "h", "pre_hada", "mlp", "out"} {
		want, _ := fx.get(name)
		checkContract(t, name, all[name], want)
	}
}
