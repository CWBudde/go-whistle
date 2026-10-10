package encoder

import (
	"errors"
	"maps"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/frontend"
)

func TestHypothesisRoundTrip(t *testing.T) {
	err := (Hypothesis{}).Validate()
	if !errors.Is(err, ErrHypothesis) {
		t.Fatalf("zero Hypothesis: %v, want ErrHypothesis", err)
	}

	h := Best()

	err = h.Validate()
	if err != nil {
		t.Fatal(err)
	}

	for _, a := range Axes() {
		for v := range a.Names {
			g := h
			a.Set(&g, v+1)

			back, err := ParseHypothesis(g.String())
			if err != nil {
				t.Fatalf("%s=%s: %v", a.Key, a.Names[v], err)
			}

			if back != g || a.Get(back) != v+1 {
				t.Fatalf("%s=%s: round trip gives %s", a.Key, a.Names[v], back)
			}
		}
	}

	bad := h
	bad.GLU = 0

	err = bad.Validate()
	if !errors.Is(err, ErrHypothesis) {
		t.Fatalf("glu 0: %v, want ErrHypothesis", err)
	}

	for _, key := range []string{"", h.String() + " extra=1", h.Frontend.String() + " stride=all"} {
		_, err = ParseHypothesis(key)
		if !errors.Is(err, ErrHypothesis) {
			t.Errorf("ParseHypothesis(%q): %v, want ErrHypothesis", key, err)
		}
	}
}

// TestStemFrames pins the stem's geometry: three round-up halvings, 3000
// mel frames → 375 and 80 mels → 10 for SAME and symmetric padding; VALID
// and dw-only strides do not fit stem.out's 1280 = 10 × 128 inputs
// (docs/FINDINGS.md F-016).
func TestStemFrames(t *testing.T) {
	cases := []struct {
		stride      StemStride
		pad         StemPad
		frames, mel int
	}{
		{StrideAll, PadSymmetric, 375, 10},
		{StrideAll, PadSame, 375, 10},
		{StrideAll, PadValid, 374, 9},
		{StrideDepthwise, PadSymmetric, 750, 20},
	}

	for _, c := range cases {
		h := Best()
		h.StemStride, h.StemPad = c.stride, c.pad
		s := &stem{h: h}

		if got := s.length(3000); got != c.frames {
			t.Errorf("stride %d pad %d: 3000 frames → %d, want %d", c.stride, c.pad, got, c.frames)
		}

		if got := s.length(frontend.Mels); got != c.mel {
			t.Errorf("stride %d pad %d: 80 mels → %d, want %d", c.stride, c.pad, got, c.mel)
		}
	}

	// The engine's rule ceil(floor(n/160)/8) (F-012) on F-001's lengths.
	s := &stem{h: Best()}
	for n, want := range map[int]int{1280: 1, 1281: 1, 2560: 2, 16000: 13, 53040: 42, 160000: 125, 480000: 375} {
		if got := s.length(frontend.FrameCount(n, frontend.PadZero)); got != want {
			t.Errorf("%d samples: %d frames, engine %d", n, got, want)
		}
	}
}

func TestApplyAct(t *testing.T) {
	x := []float32{-2, -0.5, 0, 1, 3}

	want := map[Act][]float64{
		ActNone: {-2, -0.5, 0, 1, 3},
		ActReLU: {0, 0, 0, 1, 3},
	}

	for _, a := range []Act{ActSiLU, ActGELUTanh, ActGELUErf} {
		for _, v := range x {
			f := float64(v)

			var y float64

			switch a {
			case ActSiLU:
				y = f / (1 + math.Exp(-f))
			case ActGELUTanh:
				y = 0.5 * f * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(f+0.044715*f*f*f)))
			case ActGELUErf:
				y = 0.5 * f * (1 + math.Erf(f/math.Sqrt2))
			}

			want[a] = append(want[a], y)
		}
	}

	for a, w := range want {
		got := slices.Clone(x)
		applyAct(got, a)

		for i := range got {
			if math.Abs(float64(got[i])-w[i]) > 1e-6 {
				t.Errorf("act %d: f(%v) = %v, want %v", a, x[i], got[i], w[i])
			}
		}
	}
}

func TestGLUAndDepthwise(t *testing.T) {
	// One row of pw1 output, half = 2: a = (1, −2), b = (0, 3).
	p := []float32{1, -2, 0, 3}
	sig := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }

	want := map[GLU][]float64{
		GLUValueGate:  {1 * sig(0), -2 * sig(3)},
		GLUGateValue:  {sig(1) * 0, sig(-2) * 3},
		GLUSwishValue: {1 * sig(1) * 0, -2 * sig(-2) * 3},
		GLUValueSwish: {1 * 0 * sig(0), -2 * 3 * sig(3)},
	}

	for g, w := range want {
		b := &block{h: Hypothesis{GLU: g}, glu: 4, half: 2}

		got := b.gated(p, 1)
		for i := range got {
			if math.Abs(float64(got[i])-w[i]) > 1e-6 {
				t.Errorf("glu %d: [%d] = %v, want %v", g, i, got[i], w[i])
			}
		}
	}

	// A 3-tap depthwise conv over 4 frames of one channel: taps (1, 10,
	// 100) weight frames i−1, i, i+1 centred and i−2, i−1, i causal.
	x := []float32{1, 2, 3, 4}

	for pad, w := range map[DWPad][]float32{
		DWCentred: {10 + 200, 1 + 20 + 300, 2 + 30 + 400, 3 + 40},
		DWCausal:  {100, 10 + 200, 1 + 20 + 300, 2 + 30 + 400},
	} {
		b := &block{h: Hypothesis{DWPad: pad}, half: 1, dwKernel: 3, dw: []float32{1, 10, 100}}
		if got := b.depthwise(x, 4); !slices.Equal(got, w) {
			t.Errorf("dwpad %d: %v, want %v", pad, got, w)
		}
	}
}

func TestAddSinusoid(t *testing.T) {
	const dim = 8

	x := make([]float32, 3*dim)
	addSinusoid(x, 3, dim, 0.5, false)

	for p := range 3 {
		for i := range dim / 2 {
			a := float64(p) * math.Pow(10000, -float64(2*i)/dim)
			s, c := x[p*dim+i], x[p*dim+dim/2+i]

			if math.Abs(float64(s)-0.5*math.Sin(a)) > 1e-6 || math.Abs(float64(c)-0.5*math.Cos(a)) > 1e-6 {
				t.Fatalf("pos %d pair %d: %v, %v", p, i, s, c)
			}
		}
	}

	y := make([]float32, 3*dim)
	addSinusoid(y, 3, dim, 0.5, true)

	for p := range 3 {
		for i := range dim / 2 {
			if y[p*dim+2*i] != x[p*dim+i] || y[p*dim+2*i+1] != x[p*dim+dim/2+i] {
				t.Fatalf("interleaved pos %d pair %d differs from concat", p, i)
			}
		}
	}
}

// fakePermSource answers only the permutations' shape.
type fakePermSource struct{ permSource }

func (fakePermSource) Shape(string) ([]int, error) { return []int{512}, nil }

// TestSplitPerms checks PermsSplit against _hada_perms(split=True): each
// half permuted on its own by RandomState(s) and RandomState(s + 977).
func TestSplitPerms(t *testing.T) {
	ps, ok := withPerms(fakePermSource{}, PermsSplit).(permSource)
	if !ok {
		t.Fatal("withPerms did not wrap the source")
	}

	for k, seed := range []uint32{11, 13} {
		lo, hi := cact.NumpyPermutation(seed, 256), cact.NumpyPermutation(seed+977, 256)

		for i := range 256 {
			if int(ps.p[k][i]) != lo[i] || int(ps.p[k][256+i]) != 256+hi[i] {
				t.Fatalf("perm %d entries %d/%d: %v/%v", k, i, 256+i, ps.p[k][i], ps.p[k][256+i])
			}
		}
	}

	id, ok := withPerms(fakePermSource{}, PermsIdentity).(permSource)
	if !ok {
		t.Fatal("withPerms did not wrap the source")
	}

	for i, v := range id.p[1] {
		if int(v) != i {
			t.Fatalf("identity entry %d = %v", i, v)
		}
	}
}

// shortClips are the fast test's clips, the corpus' three shortest, with a
// floor per weight source about 1e-4 (.cact) or 1e-3 (safetensors) under
// the mean per-frame cosine measured on Apple M5 Pro.
var shortClips = map[string]map[string]float64{
	whistleCactPath: {
		"ls-other-8280-266249-0032": 0.9996, // 0.999709
		"ls-other-3080-5040-0018":   0.9997, // 0.999842
		"ls-clean-5639-40744-0011":  0.9997, // 0.999833
	},
	whistleSafetensorsPath: {
		"ls-other-8280-266249-0032": 0.912, // 0.9132
		"ls-other-3080-5040-0018":   0.938, // 0.9389
		"ls-clean-5639-40744-0011":  0.935, // 0.9366
	},
}

// TestBestMatchesEngine runs Best on three short clips with either weight
// source and pins the mean per-frame cosine against needle_embed, a floor
// just under the measured value. The engine runs the quantised weights, so
// the dequantised .cact is far closer than the float checkpoint
// (docs/FINDINGS.md F-014).
func TestBestMatchesEngine(t *testing.T) {
	for _, path := range []string{whistleCactPath, whistleSafetensorsPath} {
		t.Run(filepath.Base(path), func(t *testing.T) { checkBest(t, path, shortClips[path]) })
	}
}

func checkBest(t *testing.T, path string, floors map[string]float64) {
	t.Helper()

	src := openModel(t, path)

	e, err := Load(src, Best())
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range loadClips(t, slices.Sorted(maps.Keys(floors))...) {
		y, n, err := e.Embed(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		if n != c.Frames || len(y) != n*512 {
			t.Fatalf("%s: %d frames (%d values), engine %d", c.ID, n, len(y), c.Frames)
		}

		cos := frameCosines(y, c.Embed, n, 512)
		t.Logf("%-28s frames %2d  mean cosine %.6f  min %.6f", c.ID, n, mean(cos), slices.Min(cos))

		if m := mean(cos); m < floors[c.ID] {
			t.Errorf("%s: mean per-frame cosine %.6f < %.4f", c.ID, m, floors[c.ID])
		}
	}
}

// TestEmbedIsFinalNormOutput shows that needle_embed returns
// enc.final_norm's output: every engine row divided by (1 + gain) has RMS
// 1, which ZCRMSNorm guarantees (docs/FINDINGS.md F-015).
func TestEmbedIsFinalNormOutput(t *testing.T) {
	src := openModel(t, whistleCactPath)

	gain, _, err := src.Tensor("enc.final_norm")
	if err != nil {
		t.Fatal(err)
	}

	worst, rows := 0.0, 0

	for _, c := range loadClips(t) {
		for f := range c.Frames {
			var s float64

			for k, v := range c.Embed[f*512 : (f+1)*512] {
				d := float64(v) / (1 + float64(gain[k]))
				s += d * d
			}

			worst = max(worst, math.Abs(math.Sqrt(s/512)-1))
			rows++
		}
	}

	t.Logf("%d engine rows: max |rms(row / (1 + enc.final_norm)) − 1| = %.2e", rows, worst)

	if worst > 1e-6 {
		t.Errorf("max |rms − 1| = %.2e > 1e-6", worst)
	}
}

// TestRuledOutByShapes loads the hypotheses the tensor shapes exclude.
func TestRuledOutByShapes(t *testing.T) {
	src := openModel(t, whistleCactPath)

	for _, mod := range []func(*Hypothesis){
		func(h *Hypothesis) { h.StemStride = StrideDepthwise },
		func(h *Hypothesis) { h.StemPad = PadValid },
		func(h *Hypothesis) { h.GLU = GLUNone },
	} {
		h := Best()
		mod(&h)

		_, err := Load(src, h)
		if !errors.Is(err, ErrRuledOut) {
			t.Errorf("%s: %v, want ErrRuledOut", h.EncoderString(), err)
		}
	}
}

// TestEmbed30s runs 30 s of a tone: 3000 mel frames give 375 rows.
func TestEmbed30s(t *testing.T) {
	src := openModel(t, whistleCactPath)

	e, err := Load(src, Best())
	if err != nil {
		t.Fatal(err)
	}

	x := make([]float32, 480000)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.05)) * 0.1
	}

	y, n, err := e.Embed(x)
	if err != nil {
		t.Fatal(err)
	}

	if n != 375 || len(y) != 375*512 || e.Frames(len(x)) != 375 {
		t.Fatalf("30 s: %d frames, %d values, Frames %d; want 375", n, len(y), e.Frames(len(x)))
	}
}
