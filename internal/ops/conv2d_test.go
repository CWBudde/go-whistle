package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/go-whistle/internal/tensor"
)

// conv2DRef is a float64 transcription of lax.conv_general_dilated in
// NHWC/HWIO with explicit padding and feature groups. It also returns
// Σ|x·k| per output for the rounding bound.
func conv2DRef(x []float32, shape [4]int, k []float32, kh, kw, o int, b []float32, p Conv2DParams,
) (out, absSum []float64, oh, ow int) {
	n, h, w, c := shape[0], shape[1], shape[2], shape[3]
	oh = (h+p.PadTop+p.PadBottom-kh)/p.StrideH + 1
	ow = (w+p.PadLeft+p.PadRight-kw)/p.StrideW + 1
	inPer, outPer := c/p.Groups, o/p.Groups
	out, absSum = make([]float64, n*oh*ow*o), make([]float64, n*oh*ow*o)

	for idx := range out {
		oc := idx % o
		z := idx / o % ow
		y := idx / o / ow % oh
		b0 := idx / o / ow / oh

		var s, a float64
		if b != nil {
			s, a = float64(b[oc]), math.Abs(float64(b[oc]))
		}

		for i := range kh {
			for j := range kw {
				iy, iz := y*p.StrideH+i-p.PadTop, z*p.StrideW+j-p.PadLeft
				if iy < 0 || iy >= h || iz < 0 || iz >= w {
					continue
				}

				for ci := range inPer {
					xv := float64(x[((b0*h+iy)*w+iz)*c+oc/outPer*inPer+ci])
					kv := float64(k[((i*kw+j)*inPer+ci)*o+oc])
					s += xv * kv
					a += math.Abs(xv * kv)
				}
			}
		}

		out[idx], absSum[idx] = s, a
	}

	return out, absSum, oh, ow
}

func TestConv2DMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 14))

	cases := []struct {
		name   string
		shape  [4]int // n, h, w, c
		o      int
		kh, kw int
		p      Conv2DParams
		bias   bool
	}{
		{"stem conv stride 2 same", [4]int{1, 12, 10, 1}, 8, 3, 3, Conv2DParams{2, 2, 0, 1, 0, 1, 1}, false},
		{"depthwise stride 2 pad 1", [4]int{1, 9, 8, 8}, 8, 3, 3, Conv2DParams{2, 2, 1, 1, 1, 1, 8}, false},
		{"depthwise multiplier 2", [4]int{2, 7, 5, 3}, 6, 3, 3, Conv2DParams{1, 1, 1, 1, 1, 1, 3}, true},
		{"groups 2", [4]int{1, 6, 6, 4}, 6, 2, 3, Conv2DParams{1, 2, 0, 1, 1, 0, 2}, true},
		{"valid stride 1", [4]int{2, 5, 7, 3}, 4, 3, 2, Conv2DParams{1, 1, 0, 0, 0, 0, 1}, true},
		{"asymmetric stride", [4]int{1, 11, 4, 2}, 5, 3, 3, Conv2DParams{3, 1, 2, 0, 1, 1, 1}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n, h, w, c := tc.shape[0], tc.shape[1], tc.shape[2], tc.shape[3]
			x := randSlice(rng, n*h*w*c)
			k := randSlice(rng, tc.kh*tc.kw*c/tc.p.Groups*tc.o)

			var (
				b  []float32
				bt *tensor.Tensor
			)

			if tc.bias {
				b = randSlice(rng, tc.o)
				bt = mustTensorT(t, b, []int64{int64(tc.o)})
			}

			want, absSum, oh, ow := conv2DRef(x, tc.shape, k, tc.kh, tc.kw, tc.o, b, tc.p)

			xt := mustTensorT(t, x, []int64{int64(n), int64(h), int64(w), int64(c)})
			kt := mustTensorT(t, k, []int64{int64(tc.kh), int64(tc.kw), int64(c / tc.p.Groups), int64(tc.o)})

			got, err := Conv2D(xt, kt, bt, tc.p)
			if err != nil {
				t.Fatal(err)
			}

			if s := got.Shape(); s[0] != int64(n) || s[1] != int64(oh) || s[2] != int64(ow) || s[3] != int64(tc.o) {
				t.Fatalf("shape %v, want [%d %d %d %d]", s, n, oh, ow, tc.o)
			}

			// Float32 accumulation of m terms: |Δ| ≤ γ(m)·Σ|x·k| with γ(m) = m·u/(1 − m·u).
			terms := float64(tc.kh*tc.kw*c/tc.p.Groups + 2)
			gamma := terms * 0x1p-24 / (1 - terms*0x1p-24)

			for i, v := range got.RawData() {
				if d := math.Abs(float64(v) - want[i]); d > gamma*absSum[i] {
					t.Fatalf("out[%d] = %v, reference %v (|Δ| %g > %g)", i, v, want[i], d, gamma*absSum[i])
				}
			}
		})
	}
}

func TestConv2DHandComputed(t *testing.T) {
	// 4×4 ramp, 3×3 box kernel, stride 2, SAME (0 before, 1 after): output
	// (i, j) sums rows 2i…2i+2 and columns 2j…2j+2 inside the image.
	x := make([]float32, 16)
	for i := range x {
		x[i] = float32(i)
	}

	k := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}

	got, err := Conv2D(mustTensorT(t, x, []int64{1, 4, 4, 1}), mustTensorT(t, k, []int64{3, 3, 1, 1}), nil,
		Conv2DParams{StrideH: 2, StrideW: 2, PadBottom: 1, PadRight: 1, Groups: 1})
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{
		0 + 1 + 2 + 4 + 5 + 6 + 8 + 9 + 10, 2 + 3 + 6 + 7 + 10 + 11,
		8 + 9 + 10 + 12 + 13 + 14, 10 + 11 + 14 + 15,
	}
	if !equalApprox(got.RawData(), want, 0) {
		t.Fatalf("got %v, want %v", got.RawData(), want)
	}
}

func TestSamePadding(t *testing.T) {
	cases := []struct{ n, k, s, lo, hi, out int }{
		{80, 3, 2, 0, 1, 40},
		{75, 3, 2, 1, 1, 38},
		{1, 3, 2, 1, 1, 1},
		{2, 3, 2, 0, 1, 1},
		{10, 3, 1, 1, 1, 10},
		{9, 9, 1, 4, 4, 9},
		{7, 1, 2, 0, 0, 4},
	}

	for _, c := range cases {
		lo, hi := SamePadding(c.n, c.k, c.s)
		if lo != c.lo || hi != c.hi {
			t.Errorf("SamePadding(%d, %d, %d) = %d, %d, want %d, %d", c.n, c.k, c.s, lo, hi, c.lo, c.hi)
		}

		if out := (c.n+lo+hi-c.k)/c.s + 1; out != c.out {
			t.Errorf("n %d k %d s %d: %d outputs, want ceil = %d", c.n, c.k, c.s, out, c.out)
		}
	}
}

func TestConv2DErrors(t *testing.T) {
	x := mustTensorT(t, seqDataT(2*3*4), []int64{1, 2, 3, 4})
	k := mustTensorT(t, seqDataT(3*3*4*2), []int64{3, 3, 4, 2})
	ok := Conv2DParams{StrideH: 1, StrideW: 1, PadTop: 1, PadBottom: 1, PadLeft: 1, PadRight: 1, Groups: 1}

	_, err := Conv2D(nil, k, nil, ok)
	assertErrContains(t, err, "non-nil")

	bad := ok
	bad.StrideH = 0
	_, err = Conv2D(x, k, nil, bad)
	assertErrContains(t, err, "> 0")

	bad = ok
	bad.PadLeft = -1
	_, err = Conv2D(x, k, nil, bad)
	assertErrContains(t, err, ">= 0")

	_, err = Conv2D(mustTensorT(t, seqDataT(12), []int64{3, 4}), k, nil, ok)
	assertErrContains(t, err, "rank 4")

	bad = ok
	bad.Groups = 2
	_, err = Conv2D(x, k, nil, bad)
	assertErrContains(t, err, "groups")

	_, err = Conv2D(x, k, mustTensorT(t, seqDataT(3), []int64{3}), ok)
	assertErrContains(t, err, "bias")

	_, err = Conv2D(x, k, nil, Conv2DParams{StrideH: 1, StrideW: 1, Groups: 1})
	assertErrContains(t, err, "smaller than kernel")
}

func randSlice(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}

	return out
}
