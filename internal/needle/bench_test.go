package needle

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

// The benchmarks run the hot loops at Whistle's shapes on random weights,
// so they need no model (docs/FINDINGS.md F-035): width 512 in Monarch
// blocks of 16 × 32 with rank-8 conditioning, attention with 8 query heads
// of qk 48 on 2 KV heads of v 64, and the 4-lane mHC. t = 128 frames is an
// 8 s clip after the stem, 375 the 30 s limit; the decoder runs one row
// per step against its cache.

const (
	benchDim, benchHeads, benchKVHeads, benchQK, benchV = 512, 8, 2, 48, 64
	benchNA, benchNB, benchRank                         = 16, 32, 8
)

// benchHadamard is a random HadamardMLP of Whistle's geometry.
func benchHadamard(b *testing.B) *HadamardMLP {
	b.Helper()

	r := rand.New(rand.NewPCG(5, 7))
	src := newMapSource()
	n := benchDim

	for _, name := range []string{"d1", "d2", "b2", "d3", "d4"} {
		src.put(name, randVec(r, n, 0.5), n)
	}

	for s := 1; s <= 3; s++ {
		src.put(fmt.Sprintf("w%da", s), randVec(r, benchNA*benchNA, 0.25), benchNA, benchNA)
		src.put(fmt.Sprintf("w%db", s), randVec(r, benchNB*benchNB, 0.18), benchNB, benchNB)
	}

	src.put("cond_v", randVec(r, n*benchRank, 0.05), n, benchRank)
	src.put("cond_u", randVec(r, benchRank*n, 0.1), benchRank, n)

	for _, name := range []string{"hada_p1", "hada_p2"} {
		p := make([]float32, n)
		for i, k := range r.Perm(n) {
			p[i] = float32(k)
		}

		src.put(name, p, n)
	}

	h, err := LoadHadamardMLP(src, "")
	if err != nil {
		b.Fatal(err)
	}

	return h
}

// benchAttention is a random attention of Whistle's geometry, with the
// decoder's causal taps if taps is set.
func benchAttention(b *testing.B, taps int) *Attention {
	b.Helper()

	r := rand.New(rand.NewPCG(3, 4))
	src := newMapSource()
	d := benchDim
	qW, kW, vW, oW := benchHeads*benchQK, benchKVHeads*benchQK, benchKVHeads*benchV, benchHeads*benchV

	src.put("q_proj", randVec(r, qW*d, 0.04), qW, d)
	src.put("k_proj", randVec(r, kW*d, 0.04), kW, d)
	src.put("v_proj", randVec(r, vW*d, 0.04), vW, d)
	src.put("q_norm", randVec(r, benchQK, 0.2), benchQK)
	src.put("k_norm", randVec(r, benchQK, 0.2), benchQK)
	src.put("gate_proj", randVec(r, oW*d, 0.04), oW, d)
	src.put("out_proj", randVec(r, d*oW, 0.04), d, oW)

	if taps > 0 {
		src.put("q_taps", randVec(r, taps*qW, 0.5), taps, qW)
		src.put("k_taps", randVec(r, taps*kW, 0.5), taps, kW)
		src.put("v_taps", randVec(r, taps*vW, 0.5), taps, vW)
	}

	a, err := LoadAttention(src, "")
	if err != nil {
		b.Fatal(err)
	}

	return a
}

func BenchmarkKronApply(b *testing.B) {
	h := benchHadamard(b)
	r := rand.New(rand.NewPCG(1, 1))
	z := randVec(r, benchDim, 1)
	dst, y, k := make([]float32, benchDim), make([]float32, benchDim), make([]float32, benchDim)

	for b.Loop() {
		h.mix(dst, z, 2, y, k)
	}
}

func BenchmarkHadamardMLP(b *testing.B) {
	h := benchHadamard(b)

	for _, t := range []int{1, 128} {
		x := randVec(rand.New(rand.NewPCG(2, 2)), t*benchDim, 1)

		b.Run(fmt.Sprintf("t=%d", t), func(b *testing.B) {
			for b.Loop() {
				_, err := h.Forward(x, t)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkAttentionEncoder is the encoder's unmasked self-attention over t
// frames, projections included.
func BenchmarkAttentionEncoder(b *testing.B) {
	a := benchAttention(b, 0)

	for _, t := range []int{128, 375} {
		x := randVec(rand.New(rand.NewPCG(6, 6)), t*benchDim, 1)

		b.Run(fmt.Sprintf("t=%d", t), func(b *testing.B) {
			for b.Loop() {
				_, err := a.Forward(x, t, MaskNone, nil)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkAttentionDecodeStep is one decoder step against a cache of 64
// positions, taps and RoPE included.
func BenchmarkAttentionDecodeStep(b *testing.B) {
	const prefix = 64

	a := benchAttention(b, 4)
	r := rand.New(rand.NewPCG(8, 8))
	rope := NewRope(benchQK, prefix+1, 1e5)

	var c KVCache

	_, err := a.Extend(randVec(r, prefix*benchDim, 1), prefix, &c, rope)
	if err != nil {
		b.Fatal(err)
	}

	x := randVec(r, benchDim, 1)

	for b.Loop() {
		f := c.Fork()

		_, err := a.Extend(x, 1, &f, rope)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSinkhorn(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 1))
	logits := randVec(r, 16, 1)

	for i := range 4 {
		logits[i*5] += 4
	}

	h := make([]float32, 16)

	for b.Loop() {
		copy(h, logits)
		sinkhorn(h, 4, mhcSinkhornIters)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	for _, n := range []int{8, 64, 375} {
		s := randVec(rand.New(rand.NewPCG(uint64(n), 3)), n, 2)
		w := slices.Clone(s)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for b.Loop() {
				copy(w, s)
				softmaxInPlace(w)
			}
		})
	}
}

// BenchmarkNorms is ZCRMSNorm over 128 rows of the model width and
// RMSUnit over 128 mHC rows of 4 lanes.
func BenchmarkNorms(b *testing.B) {
	r := rand.New(rand.NewPCG(4, 4))
	norm := NewZCRMSNorm(randVec(r, benchDim, 0.2))
	x := randVec(r, 128*4*benchDim, 1)
	dst := make([]float32, len(x))

	b.Run("ZCRMSNorm", func(b *testing.B) {
		for b.Loop() {
			norm.Apply(dst[:128*benchDim], x[:128*benchDim])
		}
	})

	b.Run("RMSUnit", func(b *testing.B) {
		for b.Loop() {
			RMSUnit(dst, x, 4*benchDim)
		}
	})
}
