package needle

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// mapSource is a weights.Source over in-memory tensors.
type mapSource struct {
	data  map[string][]float32
	shape map[string][]int
}

func newMapSource() *mapSource {
	return &mapSource{data: map[string][]float32{}, shape: map[string][]int{}}
}

func (m *mapSource) Names() []string {
	names := make([]string, 0, len(m.data))
	for n := range m.data {
		names = append(names, n)
	}

	slices.Sort(names)

	return names
}

func (m *mapSource) Has(name string) bool { _, ok := m.data[name]; return ok }

func (m *mapSource) Shape(name string) ([]int, error) {
	s, ok := m.shape[name]
	if !ok {
		return nil, weights.ErrNotFound
	}

	return s, nil
}

func (m *mapSource) Tensor(name string) ([]float32, []int, error) {
	v, ok := m.data[name]
	if !ok {
		return nil, nil, weights.ErrNotFound
	}

	return v, m.shape[name], nil
}

func (m *mapSource) Raw(string) ([]byte, error) { return nil, weights.ErrUnavailable }
func (m *mapSource) Config() weights.Config     { return weights.Config{} }
func (m *mapSource) Release(string)             {}
func (m *mapSource) Close() error               { return nil }

func (m *mapSource) put(name string, v []float32, shape ...int) {
	m.data[name] = v
	m.shape[name] = shape
}

// tinyAttention is a random attention with 4 query heads on 2 KV heads,
// qk 4, v 3, width 8, in the .cact layout ([out, in] projections).
type tinyAttention struct {
	dim, heads, kvHeads, qk, v, taps int
	src                              *mapSource
}

func newTinyAttention(r *rand.Rand, prefix string, taps int) tinyAttention {
	ta := tinyAttention{dim: 8, heads: 4, kvHeads: 2, qk: 4, v: 3, taps: taps, src: newMapSource()}
	qW, kW, vW, oW := ta.heads*ta.qk, ta.kvHeads*ta.qk, ta.kvHeads*ta.v, ta.heads*ta.v

	ta.src.put(prefix+"q_proj", randVec(r, qW*ta.dim, 0.5), qW, ta.dim)
	ta.src.put(prefix+"k_proj", randVec(r, kW*ta.dim, 0.5), kW, ta.dim)
	ta.src.put(prefix+"v_proj", randVec(r, vW*ta.dim, 0.5), vW, ta.dim)
	ta.src.put(prefix+"q_norm", randVec(r, ta.qk, 0.3), ta.qk)
	ta.src.put(prefix+"k_norm", randVec(r, ta.qk, 0.3), ta.qk)
	ta.src.put(prefix+"gate_proj", randVec(r, oW*ta.dim, 0.5), oW, ta.dim)
	ta.src.put(prefix+"out_proj", randVec(r, ta.dim*oW, 0.5), ta.dim, oW)

	if taps > 0 {
		ta.src.put(prefix+"q_taps", randVec(r, taps*qW, 0.7), taps, qW)
		ta.src.put(prefix+"k_taps", randVec(r, taps*kW, 0.7), taps, kW)
		ta.src.put(prefix+"v_taps", randVec(r, taps*vW, 0.7), taps, vW)
	}

	return ta
}

func (ta tinyAttention) w(prefix, name string) []float64 {
	return toFloat64(ta.src.data[prefix+name])
}

// refLinear is x @ wᵀ for w [out, in].
func refLinear(x []float64, t, in int, w []float64, out int) []float64 {
	y := make([]float64, t*out)

	for i := range t {
		for o := range out {
			var s float64
			for k := range in {
				s += x[i*in+k] * w[o*in+k]
			}

			y[i*out+o] = s
		}
	}

	return y
}

// refAttention transcribes architecture.py MultiHeadAttention (flash=False)
// in float64: jnp.repeat for GQA, so query head h reads KV head h // rep.
func (ta tinyAttention) refAttention(prefix string, x []float64, t int, causal bool, rope *Rope) []float64 {
	return ta.refAttentionBand(prefix, x, t, causal, 0, rope)
}

// refAttentionBand is refAttention with the causal mask narrowed to the
// band (i − j) < window of SimpleAttentionNetwork's local layers; window 0
// is no band. The taps are not masked, which equals upstream's tap_ok on
// the band for window ≥ taps (docs/FINDINGS.md F-045).
func (ta tinyAttention) refAttentionBand(prefix string, x []float64, t int, causal bool, window int,
	rope *Rope,
) []float64 {
	qW, kW, vW, oW := ta.heads*ta.qk, ta.kvHeads*ta.qk, ta.kvHeads*ta.v, ta.heads*ta.v
	q := refLinear(x, t, ta.dim, ta.w(prefix, "q_proj"), qW)
	k := refLinear(x, t, ta.dim, ta.w(prefix, "k_proj"), kW)
	v := refLinear(x, t, ta.dim, ta.w(prefix, "v_proj"), vW)

	if ta.taps > 0 {
		q = refTaps(q, t, qW, ta.w(prefix, "q_taps"), ta.taps)
		k = refTaps(k, t, kW, ta.w(prefix, "k_taps"), ta.taps)
		v = refTaps(v, t, vW, ta.w(prefix, "v_taps"), ta.taps)
	}

	refNormRows(q, ta.qk, ta.w(prefix, "q_norm"))
	refNormRows(k, ta.qk, ta.w(prefix, "k_norm"))

	if rope != nil {
		for i := range t {
			for h := range ta.heads {
				refRope(q[i*qW+h*ta.qk:i*qW+(h+1)*ta.qk], i, rope)
			}

			for h := range ta.kvHeads {
				refRope(k[i*kW+h*ta.qk:i*kW+(h+1)*ta.qk], i, rope)
			}
		}
	}

	// Repeat K/V to one head per query head, as jnp.repeat(…, rep, axis=1).
	rep := ta.heads / ta.kvHeads
	kr := make([][]float64, ta.heads)
	vr := make([][]float64, ta.heads)

	for h := range ta.heads {
		src := h / rep
		for i := range t {
			kr[h] = append(kr[h], k[i*kW+src*ta.qk:i*kW+(src+1)*ta.qk]...)
			vr[h] = append(vr[h], v[i*vW+src*ta.v:i*vW+(src+1)*ta.v]...)
		}
	}

	out := make([]float64, t*oW)
	scale := math.Sqrt(float64(ta.qk))

	for h := range ta.heads {
		for i := range t {
			w := make([]float64, t)
			m := math.Inf(-1)

			for j := range t {
				if causal && (j > i || window > 0 && i-j >= window) {
					w[j] = math.Inf(-1)

					continue
				}

				var s float64
				for d := range ta.qk {
					s += q[i*qW+h*ta.qk+d] * kr[h][j*ta.qk+d]
				}

				w[j] = s / scale
				m = math.Max(m, w[j])
			}

			var sum float64

			for j := range w {
				w[j] = math.Exp(w[j] - m)
				sum += w[j]
			}

			for j := range w {
				for d := range ta.v {
					out[i*oW+h*ta.v+d] += w[j] / sum * vr[h][j*ta.v+d]
				}
			}
		}
	}

	gate := refLinear(x, t, ta.dim, ta.w(prefix, "gate_proj"), oW)
	for i := range out {
		out[i] /= 1 + math.Exp(-gate[i])
	}

	return refLinear(out, t, oW, ta.w(prefix, "out_proj"), ta.dim)
}

func refTaps(z []float64, t, width int, taps []float64, n int) []float64 {
	out := make([]float64, len(z))

	for j := range n {
		for i := j; i < t; i++ {
			for c := range width {
				out[i*width+c] += taps[j*width+c] * z[(i-j)*width+c]
			}
		}
	}

	return out
}

func refNormRows(z []float64, d int, scale []float64) {
	for r := 0; r < len(z); r += d {
		var ms float64
		for _, v := range z[r : r+d] {
			ms += v * v
		}

		rms := math.Sqrt(ms/float64(d) + 1e-6)
		for c := range d {
			z[r+c] = (1 + scale[c]) * z[r+c] / rms
		}
	}
}

func refRope(x []float64, pos int, r *Rope) {
	half := len(x) / 2
	for c := range half {
		cs, sn := float64(r.cos[pos*half+c]), float64(r.sin[pos*half+c])
		x1, x2 := x[c], x[half+c]
		x[c] = x1*cs - x2*sn
		x[half+c] = x2*cs + x1*sn
	}
}

func TestGQAHeadMap(t *testing.T) {
	a := &Attention{heads: 8, kvHeads: 2}

	want := []int{0, 0, 0, 0, 1, 1, 1, 1}
	for h, w := range want {
		if got := a.kvHead(h); got != w {
			t.Errorf("kvHead(%d) = %d, want %d", h, got, w)
		}
	}
}

// TestAttentionMatchesReference runs both flavours on random tiny weights
// against the float64 transcription of upstream.
func TestAttentionMatchesReference(t *testing.T) {
	cases := []struct {
		name string
		taps int
		mask Mask
		rope bool
	}{
		{"decoder", 3, MaskCausal, true},
		{"encoder", 0, MaskNone, false},
		{"encoder+rope", 0, MaskNone, true},
		{"causal no taps", 0, MaskCausal, false},
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(uint64(i), 7))
			ta := newTinyAttention(r, "l.", tc.taps)

			a, err := LoadAttention(ta.src, "l.")
			if err != nil {
				t.Fatal(err)
			}

			if a.Heads() != 4 || a.KVHeads() != 2 || a.Taps() != tc.taps {
				t.Fatalf("dims: heads %d kv %d taps %d", a.Heads(), a.KVHeads(), a.Taps())
			}

			const steps = 6

			x := randVec(r, steps*ta.dim, 1)

			var rope *Rope
			if tc.rope {
				rope = NewRope(ta.qk, steps, 1e5)
			}

			got, err := a.Forward(x, steps, tc.mask, rope)
			if err != nil {
				t.Fatal(err)
			}

			want := ta.refAttention("l.", toFloat64(x), steps, tc.mask == MaskCausal, rope)

			var refMax float64
			for _, w := range want {
				refMax = math.Max(refMax, math.Abs(w))
			}

			if d := maxAbsDiff64(got, want); d > 1e-5*math.Max(1, refMax) {
				t.Errorf("max |Δ| %.3g at max|ref| %.3g", d, refMax)
			}
		})
	}
}

// TestAttentionCausal checks that causal outputs at steps < 3 ignore the
// inputs at steps ≥ 3, taps included, bit for bit, and that a shorter run
// gives the same rows up to float32 summation order: MatMulTransB blocks
// rows 4 × 4 on arm64, so a row's order depends on T (docs/FINDINGS.md
// F-011).
func TestAttentionCausal(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 3))
	ta := newTinyAttention(r, "", 2)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	x := randVec(r, 5*ta.dim, 1)
	rope := NewRope(ta.qk, 5, 1e5)

	full, err := a.Forward(x, 5, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	changed := slices.Clone(x)
	for i := 3 * ta.dim; i < len(changed); i++ {
		changed[i] = -3 * changed[i]
	}

	other, err := a.Forward(changed, 5, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(other[:3*ta.dim], full[:3*ta.dim]) {
		t.Errorf("causal rows 0–2 depend on rows 3–4: %v vs %v", other[:3*ta.dim], full[:3*ta.dim])
	}

	if slices.Equal(other[3*ta.dim:], full[3*ta.dim:]) {
		t.Error("rows 3–4 ignore their own inputs")
	}

	prefix, err := a.Forward(x[:3*ta.dim], 3, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	if d := maxAbsDiff64(prefix, toFloat64(full[:3*ta.dim])); d > 1e-6 {
		t.Errorf("3-step run differs from the 5-step run's rows by %.3g", d)
	}
}

func TestAttentionErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 5))
	ta := newTinyAttention(r, "", 0)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	bad := newTinyAttention(r, "", 0)
	bad.src.put("k_norm", make([]float32, 5), 5)

	call := func(_ any, err error) error { return err }

	for name, err := range map[string]error{
		"7 values for one row of 8":         call(a.Forward(make([]float32, 7), 1, MaskNone, nil)),
		"RoPE table shorter than T":         call(a.Forward(make([]float32, 3*8), 3, MaskNone, NewRope(ta.qk, 2, 1e5))),
		"RoPE table of the wrong head size": call(a.Forward(make([]float32, 8), 1, MaskNone, NewRope(6, 2, 1e5))),
		"unknown mask":                      call(a.Forward(make([]float32, 8), 1, Mask(9), nil)),
		"missing prefix":                    call(LoadAttention(ta.src, "missing.")),
		"k_norm of the wrong width":         call(LoadAttention(bad.src, "")),
	} {
		if err == nil {
			t.Errorf("accepted %s", name)
		}
	}
}

func toFloat64(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}

	return out
}

// attentionFixture runs a over every batch row of a fixture and returns
// the stacked trace in row layout.
func attentionFixture(t *testing.T, fx fixture, a *Attention, mask Mask, rope *Rope) map[string][]float32 {
	t.Helper()

	x, shape := fx.get("x")
	batch, steps, dim := shape[0], shape[1], shape[2]
	all := map[string][]float32{}

	for b := range batch {
		tr := &attnTrace{}

		out, err := a.forward(x[b*steps*dim:(b+1)*steps*dim], steps, mask, rope, tr)
		if err != nil {
			t.Fatal(err)
		}

		for name, v := range map[string][]float32{
			"q_proj": tr.qProj, "k_proj": tr.kProj, "v_proj": tr.vProj,
			"q_tapped": tr.qTapped, "k_tapped": tr.kTapped, "v_tapped": tr.vTapped,
			"q_normed": tr.qNormed, "k_normed": tr.kNormed, "q_rope": tr.qRope, "k_rope": tr.kRope,
			"probs": tr.probs, "attn": tr.attn, "gate": tr.gate, "gated": tr.gated, "out": out,
		} {
			all[name] = append(all[name], v...)
		}
	}

	return all
}

// checkAttentionFixture compares the named tensors, converting the
// per-head ones from (B, heads, T, d) to row layout; probs keeps its
// (B, heads, T, T) layout, which the trace shares.
func checkAttentionFixture(t *testing.T, fx fixture, got map[string][]float32, names []string) {
	t.Helper()

	for _, name := range names {
		want, shape := fx.get(name)

		if len(shape) == 4 && name != "probs" {
			batch, heads, steps, d := shape[0], shape[1], shape[2], shape[3]
			rows := make([]float32, 0, len(want))

			for b := range batch {
				n := heads * steps * d
				rows = append(rows, headsToRows(want[b*n:(b+1)*n], heads, steps, d)...)
			}

			want = rows
		}

		checkContract(t, name, got[name], want)
	}
}

// TestAttentionDecoderFixture is decoder layer 0 (taps, causal mask, RoPE
// θ = 1e5, GQA 8:2) against testdata/needle/attention_decoder.safetensors.
func TestAttentionDecoderFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "attention_decoder")

	a, err := LoadAttention(src, "layer00.")
	if err != nil {
		t.Fatal(err)
	}

	if a.Heads() != 8 || a.KVHeads() != 2 || a.QKDim() != 48 || a.VDim() != 64 {
		t.Fatalf("layer00 dims: heads %d kv %d qk %d v %d", a.Heads(), a.KVHeads(), a.QKDim(), a.VDim())
	}

	if fx.metadata("mask") != "causal" || fx.metadata("taps") != strconv.Itoa(a.Taps()) {
		t.Fatalf("fixture mask %s taps %s, block taps %d", fx.metadata("mask"), fx.metadata("taps"), a.Taps())
	}

	theta, err := strconv.ParseFloat(fx.metadata("rope_theta"), 64)
	if err != nil {
		t.Fatal(err)
	}

	_, shape := fx.get("x")
	rope := NewRope(a.QKDim(), shape[1], theta)

	got := attentionFixture(t, fx, a, MaskCausal, rope)
	checkAttentionFixture(t, fx, got, []string{
		"q_proj", "k_proj", "v_proj", "q_tapped", "k_tapped", "v_tapped", "q_normed", "k_normed",
		"q_rope", "k_rope", "probs", "attn", "gate", "gated", "out",
	})
}

// TestAttentionEncoderFixture is encoder layer 0 (no taps, no mask) against
// testdata/needle/attention_encoder.safetensors, without RoPE and, for
// out_rope, with the decoder's θ = 1e5 tables (U3 is open).
func TestAttentionEncoderFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "attention_encoder")

	a, err := LoadAttention(src, "enc00.")
	if err != nil {
		t.Fatal(err)
	}

	if a.Taps() != 0 || fx.metadata("mask") != "none" {
		t.Fatalf("encoder taps %d, fixture mask %s", a.Taps(), fx.metadata("mask"))
	}

	got := attentionFixture(t, fx, a, MaskNone, nil)
	checkAttentionFixture(t, fx, got, []string{
		"q_proj", "k_proj", "v_proj", "q_normed", "k_normed", "probs", "attn", "gate", "gated", "out",
	})

	_, shape := fx.get("x")
	withRope := attentionFixture(t, fx, a, MaskNone, NewRope(a.QKDim(), shape[1], 1e5))
	wantRope, _ := fx.get("out_rope")
	checkContract(t, "out_rope", withRope["out"], wantRope)
}

// TestAttendGEMMMatchesRows compares attend's two paths at Whistle's head
// geometry: the GEMM-batched one against one dot product and axpy per key,
// unmasked and causal with a cache offset, for query blocks around
// attendGEMMRows and key counts off the GEMM's 4-wide blocks. The sums
// run in a different order, so they agree to float32 rounding; the
// weights must agree as well, including the exact zeros of masked keys
// (docs/FINDINGS.md F-035).
func TestAttendGEMMMatchesRows(t *testing.T) {
	a := &Attention{dim: 512, heads: 8, kvHeads: 2, qk: 48, v: 64}
	qW, kW, vW := a.heads*a.qk, a.kvHeads*a.qk, a.kvHeads*a.v

	for _, c := range []struct {
		rows, off int
		mask      Mask
		window    int
	}{
		{1, 0, MaskNone, 0},
		{7, 0, MaskNone, 0},
		{8, 0, MaskNone, 0},
		{13, 0, MaskNone, 0},
		{130, 0, MaskNone, 0},
		{1, 9, MaskCausal, 0},
		{8, 0, MaskCausal, 0},
		{9, 5, MaskCausal, 0},
		{33, 2, MaskCausal, 0},
		{1, 9, MaskCausal, 3},
		{8, 0, MaskCausal, 1},
		{9, 5, MaskCausal, 4},
		{33, 2, MaskCausal, 7},
		{40, 0, MaskCausal, 64},
	} {
		a.window = c.window
		r := rand.New(rand.NewPCG(uint64(c.rows), uint64(c.off)))
		keys := c.off + c.rows
		q := randVec(r, c.rows*qW, 1)
		k := randVec(r, keys*kW, 1)
		v := randVec(r, keys*vW, 1)

		rowTr := &attnTrace{probs: make([]float32, a.heads*c.rows*keys)}
		gemmTr := &attnTrace{probs: make([]float32, a.heads*c.rows*keys)}
		want := a.attendRows(q, k, v, c.rows, c.off, c.mask, rowTr)
		got := a.attendGEMM(q, k, v, c.rows, c.off, c.mask, gemmTr)

		name := fmt.Sprintf("%d rows at %d, %v, window %d", c.rows, c.off, c.mask, c.window)
		if d := maxAbsDiff64(got, toFloat64(want)); d > 1e-5 {
			t.Errorf("%s: outputs differ by %.3g", name, d)
		}

		if d := maxAbsDiff64(gemmTr.probs, toFloat64(rowTr.probs)); d > 1e-6 {
			t.Errorf("%s: weights differ by %.3g", name, d)
		}

		for i, p := range rowTr.probs {
			if (p == 0) != (gemmTr.probs[i] == 0) {
				t.Fatalf("%s: weight %d is %v per row, %v batched", name, i, p, gemmTr.probs[i])
			}
		}
	}
}

// TestAttentionCausalBatched: on the GEMM path, too, a causal row does not
// depend on later rows at all, not even through rounding, because masked
// keys add exact zeros.
func TestAttentionCausalBatched(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 4))
	ta := newTinyAttention(r, "", 2)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	const steps, keep = 2 * attendGEMMRows, attendGEMMRows + 3

	x := randVec(r, steps*ta.dim, 1)
	rope := NewRope(ta.qk, steps, 1e5)

	full, err := a.Forward(x, steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	changed := slices.Clone(x)
	for i := keep * ta.dim; i < len(changed); i++ {
		changed[i] = -3 * changed[i]
	}

	other, err := a.Forward(changed, steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(other[:keep*ta.dim], full[:keep*ta.dim]) {
		t.Errorf("causal rows 0–%d depend on later rows", keep-1)
	}
}
