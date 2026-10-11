package decoder

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Model-free tests of the cross-attention memory (docs/FINDINGS.md
// F-038, F-041) on random weights of Whistle's geometry: 8 heads, qk 48,
// v 64, width 512.

const xHeads, xQK, xV, xDim = 8, 48, 64, 512

// packedSource serves random CQ2 matrices packed (weights.Packer); its
// Source methods are not used by needle.LoadLinear on a packed name.
type packedSource struct {
	weights.Source

	m map[string]*cq.Matrix
}

func (s packedSource) Matrix(name string) (*cq.Matrix, bool, error) {
	m, ok := s.m[name]

	return m, ok, nil
}

// randPacked is a random [out, in] CQ2 matrix with group 128 and the
// codebook cb.
func randPacked(t *testing.T, r *rand.Rand, out, in int, cb []float32) *cq.Matrix {
	t.Helper()

	p := cact.Packed{Out: out, In: in, Group: 128, Bits: 2, Width: 2, Codebook: cb}
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

	return m
}

func randFloats(r *rand.Rand, n int, scale float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64() * scale)
	}

	return v
}

// testCross is a cross-attention with random k/v projections, packed or
// float32, and a random k_norm.
func testCross(t *testing.T, r *rand.Rand, packed bool) *cross {
	t.Helper()

	c := &cross{h: Best(), dim: xDim, heads: xHeads, qk: xQK, v: xV}
	c.kNorm = needle.NewZCRMSNorm(randFloats(r, xQK, 0.2))

	if !packed {
		c.kProj = needle.NewLinear(randFloats(r, xHeads*xQK*xDim, 0.04), xHeads*xQK, xDim)
		c.vProj = needle.NewLinear(randFloats(r, xHeads*xV*xDim, 0.04), xHeads*xV, xDim)

		return c
	}

	cb := []float32{-1.51, -0.45, 0.45, 1.51}
	src := packedSource{m: map[string]*cq.Matrix{
		"k": randPacked(t, r, xHeads*xQK, xDim, cb), "v": randPacked(t, r, xHeads*xV, xDim, cb),
	}}

	var err error

	c.kProj, err = needle.LoadLinear(src, "k", xHeads*xQK, xDim)
	if err != nil {
		t.Fatal(err)
	}

	c.vProj, err = needle.LoadLinear(src, "v", xHeads*xV, xDim)
	if err != nil {
		t.Fatal(err)
	}

	if !c.kProj.Packed() || !c.vProj.Packed() {
		t.Fatal("projections not packed")
	}

	return c
}

// TestProjectSharesRotation: the memory's keys and values, computed with
// one rotation of the encoder output for both projections
// (needle.ApplyShared), equal one Apply per projection bit for bit,
// packed and float32, for a short clip and 30 s.
func TestProjectSharesRotation(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 1))

	for _, packed := range []bool{true, false} {
		c := testCross(t, r, packed)

		for _, frames := range []int{1, 13, 375} {
			enc := randFloats(r, frames*xDim, 1)
			m := c.project(enc, frames, nil)

			k, v := make([]float32, frames*xHeads*xQK), make([]float32, frames*xHeads*xV)
			c.kProj.Apply(k, enc, frames)
			c.vProj.Apply(v, enc, frames)
			c.kNorm.Apply(k, k)

			same := func(a, b []float32) bool {
				return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
			}

			if !same(m.k, k) || !same(m.v, v) || m.frames != frames {
				t.Errorf("packed %v, %d frames: the shared projection differs from two Applys", packed, frames)
			}
		}
	}
}

// TestInt8Memory: with needle.Int8.KV the memory holds int8 keys and
// values, a quarter of the float32 bytes plus a scale per head and frame,
// standing for the float32 memory within half a step of its grid; the
// attention on it stays within float32 rounding (2e-6 of the largest
// output) of the attention on the values it stands for, its
// probabilities included.
func TestInt8Memory(t *testing.T) {
	const frames, rows = 375, 2

	r := rand.New(rand.NewPCG(41, 2))
	c := testCross(t, r, true)
	enc := randFloats(r, frames*xDim, 1)
	ref := c.project(enc, frames, nil)

	c.setInt8(needle.Int8{Act: true, KV: true})
	m := c.project(enc, frames, nil)

	if m.k != nil || m.v != nil || m.k8.Len() != frames || m.v8.Len() != frames {
		t.Fatalf("int8 memory: float %d/%d, int8 rows %d/%d", len(m.k), len(m.v), m.k8.Len(), m.v8.Len())
	}

	float := 4 * frames * xHeads * (xQK + xV)
	if got, want := m.k8.Bytes()+m.v8.Bytes(), float/4+4*2*frames*xHeads; got != want {
		t.Errorf("int8 memory takes %d bytes, want %d (float32 %d)", got, want, float)
	}

	for _, c := range []struct {
		name      string
		rows      needle.Int8Rows
		want      []float32
		headWidth int
	}{{"k", m.k8, ref.k, xQK}, {"v", m.v8, ref.v, xV}} {
		got := c.rows.Floats()

		// Half a step, plus the float32 rounding of x / s and of s · q: a
		// value near a tie may round either way and land a hair over half
		// a step away (seen under js/wasm, whose sums round without FMA).
		for i, want := range c.want {
			s := float64(c.rows.S[i/c.headWidth])
			if d := math.Abs(float64(got[i]) - float64(want)); d > s*0.5+1e-6*math.Abs(float64(want)) {
				t.Fatalf("%s[%d]: %v stands for %v", c.name, i, got[i], want)
			}
		}
	}

	q := randFloats(r, rows*xHeads*xQK, 1)
	keep := &Attention{}
	got := c.attend(slices.Clone(q), rows, m, keep)

	// The same attention in float32 on the values the codes stand for,
	// with the queries quantised per head.
	fq := memory{k: m.k8.Floats(), v: m.v8.Floats(), frames: frames}
	plain := &cross{h: c.h, dim: xDim, heads: xHeads, qk: xQK, v: xV}
	qq := slices.Clone(q)
	needle.FakeQuant(qq, xQK)

	wantKeep := &Attention{}
	want := plain.attend(qq, rows, fq, wantKeep)

	var peak float64
	for _, v := range want {
		peak = max(peak, math.Abs(float64(v)))
	}

	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > 2e-6*peak {
			t.Fatalf("out[%d] %v, float32 %v (max |out| %v)", i, got[i], want[i], peak)
		}
	}

	for i := range wantKeep.Probs {
		if d := math.Abs(float64(keep.Probs[i] - wantKeep.Probs[i])); d > 1e-6 {
			t.Fatalf("probs[%d] %v, float32 %v", i, keep.Probs[i], wantKeep.Probs[i])
		}
	}
}
