package needle

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"
)

// refSinkhorn transcribes architecture.py _sinkhorn in float64: iters ×
// (subtract the row logsumexp, then the column logsumexp), then exp.
func refSinkhorn(logits []float64, n, iters int) []float64 {
	k := slices.Clone(logits)
	lse := func(idx func(int) int) float64 {
		m := math.Inf(-1)
		for i := range n {
			m = math.Max(m, k[idx(i)])
		}

		var s float64
		for i := range n {
			s += math.Exp(k[idx(i)] - m)
		}

		return m + math.Log(s)
	}

	for range iters {
		for r := range n {
			l := lse(func(c int) int { return r*n + c })
			for c := range n {
				k[r*n+c] -= l
			}
		}

		for c := range n {
			l := lse(func(r int) int { return r*n + c })
			for r := range n {
				k[r*n+c] -= l
			}
		}
	}

	for i := range k {
		k[i] = math.Exp(k[i])
	}

	return k
}

// sums returns the row and the column sums of a square matrix in float64.
func sums(m []float32) (rows, cols []float64) {
	n := int(math.Sqrt(float64(len(m))))
	rows, cols = make([]float64, n), make([]float64, n)

	for r := range n {
		for c := range n {
			rows[r] += float64(m[r*n+c])
			cols[c] += float64(m[r*n+c])
		}
	}

	return rows, cols
}

// TestSinkhornStochastic: after upstream's 20 log-domain iterations the
// columns sum to 1 within float32 rounding (the last step normalises
// them) and the result equals a float64 transcription of upstream; the
// rows keep whatever 20 iterations leave, which for a strong diagonal is
// up to 5e-4 here and 7 % with Whistle's weights (docs/FINDINGS.md F-013).
// Run long enough, the rows of moderate logits sum to 1 as well; for
// spread logits (scale 4, so a range of about ±12) even 400 iterations
// leave rows 1e-3 off, as Sinkhorn's contraction is slow there.
func TestSinkhornStochastic(t *testing.T) {
	for _, scale := range []float64{0.1, 1, 4} {
		r := rand.New(rand.NewPCG(uint64(scale*10), 1))

		for trial := range 20 {
			logits := randVec(r, 16, scale)
			// upstream's b_res starts at 4·eye; keep a diagonal bias in half the trials.
			if trial%2 == 1 {
				for i := range 4 {
					logits[i*5] += 4
				}
			}

			h := slices.Clone(logits)
			sinkhorn(h, 4, mhcSinkhornIters)

			_, cols := sums(h)
			for i, c := range cols {
				if math.Abs(c-1) > 4e-7 {
					t.Errorf("scale %v trial %d: column %d sums to %v", scale, trial, i, c)
				}
			}

			want := refSinkhorn(toFloat64(logits), 4, mhcSinkhornIters)
			if d := maxAbsDiff64(h, want); d > 1e-6 {
				t.Errorf("scale %v trial %d: max |Δ| %.3g to float64", scale, trial, d)
			}

			if scale > 1 {
				continue
			}

			long := slices.Clone(logits)
			sinkhorn(long, 4, 400)

			rows, cols := sums(long)
			for i := range 4 {
				if math.Abs(rows[i]-1) > 1e-6 || math.Abs(cols[i]-1) > 4e-7 {
					t.Errorf("scale %v trial %d, 400 iterations: rows %v cols %v", scale, trial, rows, cols)
				}
			}
		}
	}
}

// TestSinkhornFixtureIterations runs the Sinkhorn on the mhc fixture's
// own h_res logits: 20 iterations reproduce its h_res within 1.2e-7, while
// 19 or 21 miss it by about 3e-3, far outside the contract. The count is
// part of the model, not a convergence setting.
func TestSinkhornFixtureIterations(t *testing.T) {
	fx := openFixture(t, "mhc")

	for _, tag := range []string{"layer0", "layer1"} {
		want, _ := fx.get(tag + ".h_res")
		logits, _ := fx.get(tag + ".h_res_logits")

		for _, iters := range []int{mhcSinkhornIters - 1, mhcSinkhornIters, mhcSinkhornIters + 1} {
			got := slices.Clone(logits)
			for i := 0; i < len(got); i += 16 {
				sinkhorn(got[i:i+16], 4, iters)
			}

			s := compare(got, want)
			t.Logf("%s, %d iterations: max abs %.3e", tag, iters, s.maxAbs)

			if fits := s.maxAbs <= contractRel; fits != (iters == mhcSinkhornIters) {
				t.Errorf("%s, %d iterations: max abs %.3e", tag, iters, s.maxAbs)
			}
		}
	}
}

// TestSinkhornLargeLogits: the log domain keeps logits far beyond
// float32's exp range finite (exp(89) already overflows float32), and the
// columns still sum to 1. A direct exp-then-normalise Sinkhorn gives
// Inf/NaN here.
func TestSinkhornLargeLogits(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 9))

	for _, scale := range []float64{1e3, 1e4, 1e30} {
		logits := randVec(r, 16, scale)
		h := slices.Clone(logits)
		sinkhorn(h, 4, mhcSinkhornIters)

		for i, v := range h {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < 0 || v > 1 {
				t.Fatalf("scale %v: entry %d = %v", scale, i, v)
			}
		}

		_, cols := sums(h)
		for c, s := range cols {
			if math.Abs(s-1) > 4e-7 {
				t.Errorf("scale %v: column %d sums to %v", scale, c, s)
			}
		}

		naive := float32(math.Exp(float64(slices.Max(logits))))
		if !math.IsInf(float64(naive), 1) {
			t.Errorf("scale %v: the case does not overflow a direct exp (%v)", scale, naive)
		}
	}
}

// TestSinkhornUniform: equal logits give the uniform matrix 1/n exactly.
func TestSinkhornUniform(t *testing.T) {
	h := make([]float32, 16)
	for i := range h {
		h[i] = 3.5
	}

	sinkhorn(h, 4, mhcSinkhornIters)

	for i, v := range h {
		if v != 0.25 {
			t.Fatalf("entry %d = %v, want 0.25", i, v)
		}
	}
}

// TestLaneOffsets pins Stack's lane = eye(n)[layer % n], pre_off =
// 8·lane − 4 and post_off = −4·(1 − lane).
func TestLaneOffsets(t *testing.T) {
	for layer := range 9 {
		pre, post := laneOffsets(4, layer)

		for i := range 4 {
			wantPre, wantPost := float32(-4), float32(-4)
			if i == layer%4 {
				wantPre, wantPost = 4, 0
			}

			if pre[i] != wantPre || post[i] != wantPost {
				t.Errorf("layer %d lane %d: pre %v post %v, want %v %v", layer, i, pre[i], post[i], wantPre, wantPost)
			}
		}
	}
}

func TestExpandCollapseLanes(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 4))
	x := randVec(r, 3*5, 2)

	s := ExpandLanes(x, 4, 5)
	if len(s) != 3*4*5 {
		t.Fatalf("expanded to %d values", len(s))
	}

	for i := range 3 {
		for n := range 4 {
			if !slices.Equal(s[(i*4+n)*5:(i*4+n+1)*5], x[i*5:(i+1)*5]) {
				t.Fatalf("row %d lane %d is not x's row", i, n)
			}
		}
	}

	if d := maxAbsDiff64(CollapseLanes(s, 4, 5), toFloat64(x)); d > 1e-6 {
		t.Errorf("collapse(expand(x)) is %.3g from x", d)
	}

	// Two rows of 4 lanes of width 2; the mean over the lanes, row by row.
	lanes := []float32{1, 2, 3, 4, 10, 20, 30, 40, -1, -2, -3, -4, 0, 0, 0, 8}
	if got := CollapseLanes(lanes, 4, 2); !slices.Equal(got, []float32{11, 16.5, -1, 0.5}) {
		t.Errorf("CollapseLanes = %v, want [11 16.5 -1 0.5]", got)
	}
}

// tinyMHC is random mHC weights for 3 layers of 4 lanes of width dim, in
// the .cact layout (φ lane-major (layers·cols, lanes·dim)).
func tinyMHC(r *rand.Rand, prefix string, dim int) *mapSource {
	const layers, n = 3, 4

	src := newMapSource()
	for _, h := range []string{"pre", "post", "res"} {
		src.put(prefix+"mhc_a_"+h, randVec(r, layers, 0.5), layers)
	}

	src.put(prefix+"mhc_b_pre", randVec(r, layers*n, 1), layers, n)
	src.put(prefix+"mhc_b_post", randVec(r, layers*n, 1), layers, n)

	bRes := randVec(r, layers*n*n, 1)
	for l := range layers {
		for i := range n {
			bRes[l*n*n+i*(n+1)] += 4
		}
	}

	src.put(prefix+"mhc_b_res", bRes, layers, n, n)
	src.put(prefix+"mhc_phi_pre", randVec(r, layers*n*n*dim, 0.3), layers*n, n*dim)
	src.put(prefix+"mhc_phi_post", randVec(r, layers*n*n*dim, 0.3), layers*n, n*dim)
	src.put(prefix+"mhc_phi_res", randVec(r, layers*n*n*n*dim, 0.3), layers*n*n, n*dim)

	return src
}

// refAdvance transcribes _ScanBody's advance in float64 for one layer, with
// block applied row by row.
func refAdvance(src *mapSource, prefix string, layer int, stream []float64, t, dim int,
	block func([]float64) []float64,
) []float64 {
	const n = 4

	w := func(name string) []float64 { return toFloat64(src.data[prefix+"mhc_"+name]) }
	nC := n * dim
	lane := func(i int) float64 {
		if i == layer%n {
			return 1
		}

		return 0
	}
	sigma := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	proj := func(nx, phi []float64, rows int) []float64 {
		out := make([]float64, rows)
		for o := range rows {
			for c := range nC {
				out[o] += nx[c] * phi[(layer*rows+o)*nC+c]
			}
		}

		return out
	}

	aPre, aPost, aRes := w("a_pre")[layer], w("a_post")[layer], w("a_res")[layer]
	bPre, bPost, bRes := w("b_pre")[layer*n:], w("b_post")[layer*n:], w("b_res")[layer*n*n:]
	out := make([]float64, len(stream))

	for i := range t {
		s := stream[i*nC : (i+1)*nC]

		var ms float64
		for _, v := range s {
			ms += v * v
		}

		inv := 1 / math.Sqrt(ms/float64(nC)+1e-6)
		nx := make([]float64, nC)

		for c, v := range s {
			nx[c] = v * inv
		}

		pre, post, res := proj(nx, w("phi_pre"), n), proj(nx, w("phi_post"), n), proj(nx, w("phi_res"), n*n)
		u := make([]float64, dim)

		for k := range n {
			hp := sigma(aPre*pre[k] + bPre[k] + 8*lane(k) - 4)
			for c := range dim {
				u[c] += hp * s[k*dim+c]
			}
		}

		y := block(u)
		for c := range dim {
			y[c] -= u[c]
		}

		logits := make([]float64, n*n)
		for k := range logits {
			logits[k] = aRes*res[k] + bRes[k]
		}

		hres := refSinkhorn(logits, n, 20)

		for a := range n {
			hpost := 2 * sigma(aPost*post[a]+bPost[a]-4*(1-lane(a)))
			for c := range dim {
				v := hpost * y[c]
				for b := range n {
					v += hres[a*n+b] * s[b*dim+c]
				}

				out[i*nC+a*dim+c] = v
			}
		}
	}

	return out
}

// TestMHCMatchesReference runs a layer of each lane phase on random
// weights around a nonlinear toy block against the float64 transcription.
func TestMHCMatchesReference(t *testing.T) {
	const dim, steps = 6, 3

	r := rand.New(rand.NewPCG(21, 22))
	src := tinyMHC(r, "enc.", dim)
	toy := func(u []float32) ([]float32, error) {
		out := make([]float32, len(u))
		for i, v := range u {
			out[i] = v + float32(math.Tanh(float64(v)))
		}

		return out, nil
	}
	toy64 := func(u []float64) []float64 {
		out := make([]float64, len(u))
		for i, v := range u {
			out[i] = v + math.Tanh(v)
		}

		return out
	}

	for layer := range 3 {
		t.Run(fmt.Sprint("layer", layer), func(t *testing.T) {
			m, err := LoadMHC(src, "enc.", layer)
			if err != nil {
				t.Fatal(err)
			}

			if m.Lanes() != 4 || m.Dim() != dim {
				t.Fatalf("lanes %d dim %d", m.Lanes(), m.Dim())
			}

			stream := randVec(r, steps*4*dim, 1.5)

			got, err := m.Forward(stream, steps, toy)
			if err != nil {
				t.Fatal(err)
			}

			want := refAdvance(src, "enc.", layer, toFloat64(stream), steps, dim, toy64)
			if d := maxAbsDiff64(got, want); d > 1e-5 {
				t.Errorf("max |Δ| %.3g", d)
			}
		})
	}
}

// TestMHCExpandedStream: on Stack's lane expand every lane is x, so with
// y = 0 lane i of the output is (Σ_j h_res[i, j]) · x. That is x only as
// far as 20 Sinkhorn iterations make the rows sum to 1, which they do not
// for a peaked h_res (docs/FINDINGS.md F-013).
func TestMHCExpandedStream(t *testing.T) {
	const dim, steps = 5, 2

	r := rand.New(rand.NewPCG(8, 1))
	src := tinyMHC(r, "", dim)

	m, err := LoadMHC(src, "", 1)
	if err != nil {
		t.Fatal(err)
	}

	x := randVec(r, steps*dim, 1)
	tr := &mhcTrace{}

	out, err := m.forward(ExpandLanes(x, 4, dim), steps, func(u []float32) ([]float32, error) {
		return slices.Clone(u), nil // y = 0
	}, tr)
	if err != nil {
		t.Fatal(err)
	}

	for i := range steps {
		rows, _ := sums(tr.hRes[i*16 : (i+1)*16])

		for n := range 4 {
			want := make([]float64, dim)
			for c := range dim {
				want[c] = rows[n] * float64(x[i*dim+c])
			}

			if d := maxAbsDiff64(out[(i*4+n)*dim:(i*4+n+1)*dim], want); d > 1e-6 {
				t.Errorf("row %d lane %d: %.3g from rowsum · x", i, n, d)
			}
		}
	}
}

func TestMHCErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(2, 3))
	src := tinyMHC(r, "", 4)

	m, err := LoadMHC(src, "", 0)
	if err != nil {
		t.Fatal(err)
	}

	id := func(u []float32) ([]float32, error) { return u, nil }
	short := func(u []float32) ([]float32, error) { return u[:1], nil }
	call := func(_ any, err error) error { return err }

	bad := tinyMHC(r, "", 4)
	bad.put("mhc_phi_res", randVec(r, 3*4*16, 1), 3*4, 16)

	for name, err := range map[string]error{
		"layer out of range":       call(LoadMHC(src, "", 3)),
		"missing prefix":           call(LoadMHC(src, "enc.", 0)),
		"phi_res of the wrong row": call(LoadMHC(bad, "", 0)),
		"ragged stream":            call(m.Forward(make([]float32, 15), 1, id)),
		"block returns too little": call(m.Forward(make([]float32, 16), 1, short)),
	} {
		if err == nil {
			t.Errorf("accepted %s", name)
		}
	}
}

// mhcFixtureStep runs one layer of the mhc fixture and checks the stream
// and every head against the fixture's tag.* tensors.
func mhcFixtureStep(t *testing.T, fx fixture, m *MHC, b *Block, stream []float32, steps int, rope *Rope,
	tag string,
) []float32 {
	t.Helper()

	tr := &mhcTrace{}

	out, err := m.forward(stream, steps, func(u []float32) ([]float32, error) {
		return b.Forward(u, steps, MaskCausal, rope, nil)
	}, tr)
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name string
		got  []float32
	}{
		{"nx", tr.nx},
		{"h_pre_logits", tr.preLogits},
		{"h_pre", tr.hPre},
		{"u", tr.u},
		{"block_out", tr.blockOut},
		{"y", tr.y},
		{"h_post_logits", tr.postLogits},
		{"h_post", tr.hPost},
		{"h_res_logits", tr.resLogits},
		{"h_res", tr.hRes},
		{"stream_out", out},
	} {
		want, _ := fx.get(tag + "." + c.name)
		checkContract(t, tag+"."+c.name, c.got, want)
	}

	return out
}

// TestMHCFixture is testdata/needle/mhc.safetensors: decoder layer 0 from
// Stack's lane expand through the lane mean and final_norm, and layer 1 on
// four distinct lanes, every head included (docs/FINDINGS.md F-013).
func TestMHCFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "mhc")

	if fx.metadata("lanes") != "4" || fx.metadata("sinkhorn_iters") != strconv.Itoa(mhcSinkhornIters) ||
		fx.metadata("mask") != "causal" {
		t.Fatalf("fixture lanes %s iters %s mask %s", fx.metadata("lanes"), fx.metadata("sinkhorn_iters"),
			fx.metadata("mask"))
	}

	x, shape := fx.get("x")
	steps, dim := shape[1], shape[2]
	rope := NewRope(src.Config().QKDim, steps, src.Config().Rope)

	layer := func(l int) (*MHC, *Block) {
		m, err := LoadMHC(src, "", l)
		if err != nil {
			t.Fatal(err)
		}

		b, err := LoadBlock(src, fmt.Sprintf("layer%02d.", l))
		if err != nil {
			t.Fatal(err)
		}

		return m, b
	}

	m0, b0 := layer(0)
	if m0.Lanes() != 4 || m0.Dim() != dim {
		t.Fatalf("mHC lanes %d dim %d", m0.Lanes(), m0.Dim())
	}

	out0 := mhcFixtureStep(t, fx, m0, b0, ExpandLanes(x, m0.Lanes(), dim), steps, rope, "layer0")

	collapsed := CollapseLanes(out0, m0.Lanes(), dim)
	want, _ := fx.get("layer0.collapsed")
	checkContract(t, "layer0.collapsed", collapsed, want)

	final, err := LoadZCRMSNorm(src, "final_norm")
	if err != nil {
		t.Fatal(err)
	}

	final.Apply(collapsed, collapsed)

	want, _ = fx.get("layer0.final")
	checkContract(t, "layer0.final", collapsed, want)

	m1, b1 := layer(1)
	in1, _ := fx.get("layer1.stream_in")
	mhcFixtureStep(t, fx, m1, b1, in1, steps, rope, "layer1")
}

// TestSinkhornExpMatchesLog: wherever sinkhornExp applies, it equals the
// log-domain form and upstream's float64 transcription within float32
// rounding, for Whistle's 4 lanes and for widths that do not fit its
// stack buffer; out of range it declines and leaves the logits alone
// (docs/FINDINGS.md F-035).
func TestSinkhornExpMatchesLog(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 8} {
		for _, scale := range []float64{0.1, 1, 4, 20} {
			r := rand.New(rand.NewPCG(uint64(n), uint64(scale*10)))
			logits := randVec(r, n*n, scale)

			for i := range n {
				logits[i*(n+1)] += 4
			}

			exp, log := slices.Clone(logits), slices.Clone(logits)
			if !sinkhornExp(exp, n, mhcSinkhornIters) {
				t.Fatalf("n %d scale %v: sinkhornExp declined logits in range", n, scale)
			}

			sinkhornLog(log, n, mhcSinkhornIters)

			if d := maxAbsDiff64(exp, toFloat64(log)); d > 1e-6 {
				t.Errorf("n %d scale %v: exp and log domain differ by %.3g", n, scale, d)
			}

			want := refSinkhorn(toFloat64(logits), n, mhcSinkhornIters)
			if d := maxAbsDiff64(exp, want); d > 2e-7 {
				t.Errorf("n %d scale %v: exp domain %.3g from float64", n, scale, d)
			}
		}
	}

	// Near the edge of the range (spread 640 + 2·20·ln 4 ≈ 695) the exp
	// domain still agrees with the log domain.
	edge := []float32{0, 320, -320, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	exp, log := slices.Clone(edge), slices.Clone(edge)

	if !sinkhornExp(exp, 4, mhcSinkhornIters) {
		t.Fatal("sinkhornExp declined a spread of 640")
	}

	sinkhornLog(log, 4, mhcSinkhornIters)

	if d := maxAbsDiff64(exp, toFloat64(log)); d > 1e-6 {
		t.Errorf("spread 640: exp and log domain differ by %.3g", d)
	}

	inf := float32(math.Inf(1))
	nan := float32(math.NaN())

	for name, logits := range map[string][]float32{
		"spread": {0, 350, -350, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
		"+inf":   {0, inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		"-inf":   {0, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		"nan":    {0, nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	} {
		k := slices.Clone(logits)
		if sinkhornExp(k, 4, mhcSinkhornIters) {
			t.Errorf("%s: sinkhornExp accepted %v", name, logits)
		}

		for i := range k {
			if math.Float32bits(k[i]) != math.Float32bits(logits[i]) {
				t.Errorf("%s: sinkhornExp wrote entry %d", name, i)
			}
		}
	}
}
