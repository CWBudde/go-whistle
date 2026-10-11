package needle

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
)

// The tests of docs/FINDINGS.md F-038: the parallel paths give the same
// bits on one worker as on several, on random weights of Whistle's
// geometry (bench_test.go), float and packed.

// withWorkers runs fn with tensor.SetWorkers(n) and restores the count.
func withWorkers(n int, fn func()) {
	prev := tensor.Workers()
	defer tensor.SetWorkers(prev)

	tensor.SetWorkers(n)
	fn()
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// packedLinear is a random [out, in] CQ matrix with codebook cb, group 128,
// as a packed Linear.
func packedLinear(t *testing.T, r *rand.Rand, out, in int, cb []float32) Linear {
	t.Helper()

	bits := 2
	if len(cb) == 16 {
		bits = 4
	}

	p := cact.Packed{Out: out, In: in, Group: 128, Bits: bits, Width: bits, Codebook: cb}
	p.Index = make([]byte, out*p.RowBytes())

	for i := range p.Index {
		p.Index[i] = byte(r.Uint32())
	}

	for range out * p.Groups() {
		p.Norms = binary.LittleEndian.AppendUint16(p.Norms, uint16(13+r.IntN(5))<<10|uint16(r.IntN(1024)))
	}

	m, err := cq.New(p)
	if err != nil {
		t.Fatal(err)
	}

	return Linear{out: out, in: in, q: m}
}

func randCodebook(r *rand.Rand, n int) []float32 {
	cb := randVec(r, n, 0.1)
	slices.Sort(cb)

	return cb
}

// packAttention replaces a's projections by random packed ones that share
// one CQ2 codebook, as whistle.cact's do.
func packAttention(t *testing.T, r *rand.Rand, a *Attention) {
	t.Helper()

	cb := randCodebook(r, 4)
	for _, l := range []*Linear{&a.qProj, &a.kProj, &a.vProj, &a.gateProj, &a.outProj} {
		*l = packedLinear(t, r, l.Out(), l.In(), cb)
	}
}

// sameOnWorkers runs fn on 1, 3 and 16 workers and fails unless every run
// returns the same bits.
func sameOnWorkers(t *testing.T, name string, fn func() []float32) {
	t.Helper()

	var want []float32

	withWorkers(1, func() { want = fn() })

	for _, w := range []int{3, 16} {
		var got []float32

		withWorkers(w, func() { got = fn() })

		if !sameBits(got, want) {
			t.Errorf("%s: %d workers differ from 1", name, w)
		}
	}
}

func TestApplySharedMatchesApply(t *testing.T) {
	r := rand.New(rand.NewPCG(38, 10))
	cb := randCodebook(r, 4)

	ls := []Linear{
		packedLinear(t, r, 384, 512, cb), packedLinear(t, r, 96, 512, cb),
		NewLinear(randVec(r, 96*512, 0.04), 96, 512), packedLinear(t, r, 384, 512, randCodebook(r, 4)),
	}

	for _, rows := range []int{1, 9, 375} {
		x := randVec(r, rows*512, 1)
		got, want := make([][]float32, len(ls)), make([][]float32, len(ls))

		for k, l := range ls {
			got[k], want[k] = make([]float32, rows*l.Out()), make([]float32, rows*l.Out())
			l.Apply(want[k], x, rows)
		}

		ApplyShared(got, x, rows, ls...)

		for k := range ls {
			if !sameBits(got[k], want[k]) {
				t.Errorf("%d rows: linear %d differs from Apply", rows, k)
			}
		}
	}
}

func TestAttentionWorkers(t *testing.T) {
	r := rand.New(rand.NewPCG(38, 11))
	float, taps := benchAttention(t, 0), benchAttention(t, 4)
	packed, packedTaps := benchAttention(t, 0), benchAttention(t, 4)
	packAttention(t, r, packed)
	packAttention(t, r, packedTaps)

	const frames = 375

	x := randVec(r, frames*benchDim, 1)
	rope := NewRope(benchQK, frames, 1e5)

	for name, a := range map[string]*Attention{"float": float, "packed": packed} {
		sameOnWorkers(t, name+" encoder", func() []float32 {
			y, err := a.Forward(x, frames, MaskNone, rope)
			if err != nil {
				t.Fatal(err)
			}

			return y
		})
	}

	for name, a := range map[string]*Attention{"float": taps, "packed": packedTaps} {
		sameOnWorkers(t, name+" causal", func() []float32 {
			var c KVCache

			y, err := a.Extend(x[:130*benchDim], 130, &c, rope)
			if err != nil {
				t.Fatal(err)
			}

			z, err := a.Extend(x[130*benchDim:131*benchDim], 1, &c, rope)
			if err != nil {
				t.Fatal(err)
			}

			return append(y, z...)
		})
	}
}

func TestHadamardMLPWorkers(t *testing.T) {
	h := benchHadamard(t)
	x := randVec(rand.New(rand.NewPCG(38, 12)), 375*benchDim, 1)

	sameOnWorkers(t, "HadamardMLP", func() []float32 {
		y, err := h.Forward(x, 375)
		if err != nil {
			t.Fatal(err)
		}

		return y
	})
}

func TestMHCWorkers(t *testing.T) {
	r := rand.New(rand.NewPCG(38, 13))
	src := tinyMHC(r, "", benchDim)

	float, err := LoadMHC(src, "", 1)
	if err != nil {
		t.Fatal(err)
	}

	packed := *float
	cb := randCodebook(r, 16)
	packed.phiPre = packedLinear(t, r, 4, 4*benchDim, cb)
	packed.phiPost = packedLinear(t, r, 4, 4*benchDim, cb)
	packed.phiRes = packedLinear(t, r, 16, 4*benchDim, cb)

	const frames = 375

	stream := randVec(r, frames*4*benchDim, 1)
	block := func(u []float32) ([]float32, error) {
		out := make([]float32, len(u))
		for i, v := range u {
			out[i] = v * sigmoid(v)
		}

		return out, nil
	}

	for name, m := range map[string]*MHC{"float": float, "packed": &packed} {
		sameOnWorkers(t, name+" mHC", func() []float32 {
			y, err := m.Forward(stream, frames, block)
			if err != nil {
				t.Fatal(err)
			}

			return y
		})
	}
}

func TestNormsWorkers(t *testing.T) {
	r := rand.New(rand.NewPCG(38, 14))
	x := randVec(r, 375*2048, 1)
	n := NewZCRMSNorm(randVec(r, 512, 0.2))

	sameOnWorkers(t, "ZCRMSNorm", func() []float32 {
		y := make([]float32, len(x))
		n.Apply(y, x)

		return y
	})

	sameOnWorkers(t, "RMSUnit", func() []float32 {
		y := make([]float32, len(x))
		RMSUnit(y, x, 2048)

		return y
	})
}
