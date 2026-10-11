package needle

import (
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"
)

// The model-free tests of the int8 numerics (docs/FINDINGS.md F-041).

// TestQuantizeRow pins upstream's fake_quant rule: s = max|x| / 127, the
// largest magnitude at ±127, half-way cases to even, every value within
// s/2 of s · q, and a zero row with scale 1 and zero codes.
func TestQuantizeRow(t *testing.T) {
	q := make([]int8, 6)

	// max |x| = 127 gives s = 1, so x/s is x itself: ties go to even.
	s := QuantizeRow(q, []float32{127, 2.5, -2.5, 3.5, -0.5, 1.4999})
	if s != 1 || !slices.Equal(q, []int8{127, 2, -2, 4, 0, 1}) {
		t.Errorf("ties: s %v, q %v; want 1 and [127 2 -2 4 0 1]", s, q)
	}

	// The extreme maps to ±127 whatever its sign; −128 is never used.
	s = QuantizeRow(q[:3], []float32{-6.35, 0.05, 3})
	if want := float32(6.35) / 127; s != want || q[0] != -127 || q[1] != 1 || q[2] != 60 {
		t.Errorf("negative extreme: s %v (want %v), q %v", s, want, q[:3])
	}

	s = QuantizeRow(q[:4], make([]float32, 4))
	if s != 1 || !slices.Equal(q[:4], []int8{0, 0, 0, 0}) {
		t.Errorf("zero row: s %v, q %v; want 1 and zeros", s, q[:4])
	}

	r := rand.New(rand.NewPCG(1, 2))

	for _, n := range []int{1, 7, 48, 64, 2048} {
		for _, scale := range []float64{1e-6, 1, 1e4} {
			x, q := randVec(r, n, scale), make([]int8, n)
			x[r.IntN(n)] *= 9 // an outlier

			s := QuantizeRow(q, x)

			var m float32
			for _, v := range x {
				m = max(m, abs32(v))
			}

			if s != m/127 {
				t.Errorf("n %d: scale %v, want max|x|/127 = %v", n, s, m/127)
			}

			for i, v := range x {
				if d := math.Abs(float64(v) - float64(s)*float64(q[i])); d > float64(s)*0.5000001 || q[i] < -127 {
					t.Fatalf("n %d: x[%d] = %v → %d · %v, off by %v", n, i, v, q[i], s, d)
				}
			}
		}
	}
}

// TestFakeQuant: fakeQuant is QuantizeRow per chunk, s · q in float32,
// in place too.
func TestFakeQuant(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	x := randVec(r, 5*48, 2)
	got := make([]float32, len(x))
	fakeQuant(got, x, 48)

	q := make([]int8, 48)

	for lo := 0; lo < len(x); lo += 48 {
		s := QuantizeRow(q, x[lo:lo+48])
		for i, c := range q {
			if want := float32(c) * s; got[lo+i] != want {
				t.Fatalf("value %d: %v, want %v", lo+i, got[lo+i], want)
			}
		}
	}

	in := slices.Clone(x)
	FakeQuant(in, 48)

	if !slices.Equal(in, got) {
		t.Error("FakeQuant in place differs from fakeQuant")
	}
}

// TestInt8Rows: rows, scales per chunk, Floats, Bytes, and a Clip whose
// Append leaves the original's spare capacity alone.
func TestInt8Rows(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	x := randVec(r, 3*96, 1)
	m := QuantizeRows(x, 96, 48)

	if m.Len() != 3 || len(m.S) != 6 || m.Bytes() != 3*96+4*6 {
		t.Fatalf("Len %d, %d scales, %d bytes", m.Len(), len(m.S), m.Bytes())
	}

	want := slices.Clone(x)
	fakeQuant(want, want, 48)

	if !slices.Equal(m.Floats(), want) {
		t.Error("Floats differs from fakeQuant")
	}

	grown := NewInt8Rows(96, 48)
	for i := range 3 {
		grown = grown.Append(x[i*96 : (i+1)*96])
	}

	if !slices.Equal(grown.Q, m.Q) || !slices.Equal(grown.S, m.S) {
		t.Error("row by row differs from QuantizeRows")
	}

	a := grown.Clip().Append(randVec(r, 96, 1))
	b := grown.Append(randVec(r, 96, 1))

	if slices.Equal(a.Q[3*96:], b.Q[3*96:]) || !slices.Equal(a.Q[:3*96], m.Q) {
		t.Error("an Append to a clipped copy shares the original's spare rows")
	}

	if (Int8Rows{}).Len() != 0 {
		t.Error("the zero Int8Rows is not empty")
	}

	for _, f := range []func(){
		func() { NewInt8Rows(96, 40) },
		func() { NewInt8Rows(96, 48).Append(make([]float32, 50)) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("no panic")
				}
			}()

			f()
		}()
	}
}

// TestInt8KernelsMatchGeneric: the assembly kernels (arm64, amd64 with
// AVX2) equal their Go references, dotsI8 exactly and axpysI8 bit for
// bit, at every width the kernels take (multiples of 16), widths they
// leave to Go, key counts around the four-key blocks and strides past the
// row. Under purego both sides are the Go loop.
func TestInt8KernelsMatchGeneric(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))

	codes := func(n int) []int8 {
		c := make([]int8, n)
		for i := range c {
			c[i] = int8(r.IntN(256) - 128)
		}

		return c
	}

	for _, width := range []int{16, 32, 48, 64, 128, 8, 17} {
		for _, n := range []int{1, 2, 3, 4, 5, 8, 9, 375} {
			for _, extra := range []int{0, 16, 80} {
				stride := width + extra
				name := strconv.Itoa(width) + "/" + strconv.Itoa(n) + "/" + strconv.Itoa(stride)
				k := codes((n-1)*stride + width)

				q := codes(width)
				q[0], k[0] = -128, -128 // the largest product

				got, want := make([]int32, n), make([]int32, n)
				dotsI8(got, q, k, stride)
				dotsI8Generic(want, q, k, stride)

				if !slices.Equal(got, want) {
					t.Fatalf("dotsI8 %s: %v, want %v", name, got[:min(n, 8)], want[:min(n, 8)])
				}

				w := randVec(r, n, 3)
				out := randVec(r, width, 10)
				ref := slices.Clone(out)
				axpysI8(out, w, k, stride)
				axpysI8Generic(ref, w, k, stride)

				for d := range out {
					if math.Float32bits(out[d]) != math.Float32bits(ref[d]) {
						t.Fatalf("axpysI8 %s: out[%d] %v, want %v", name, d, out[d], ref[d])
					}
				}
			}
		}
	}

	// Empty calls do nothing; short operands panic.
	dotsI8(nil, nil, nil, 0)
	axpysI8(nil, nil, nil, 0)

	defer func() {
		if recover() == nil {
			t.Error("dotsI8 over too few codes did not panic")
		}
	}()

	dotsI8(make([]int32, 2), make([]int8, 16), make([]int8, 31), 16)
}

// TestAttendInt8MatchesFloat: AttendInt8 equals the float32 attention on
// the values its codes stand for, with the queries fake-quantised when it
// quantises them: GQA, causal and full key counts, Whistle's head widths,
// within float32 rounding. The probabilities it reports sum to 1 over
// each row's keys.
func TestAttendInt8MatchesFloat(t *testing.T) {
	const heads, kvHeads, qk, vd, n, rows = 8, 2, 48, 64, 37, 3

	r := rand.New(rand.NewPCG(9, 10))
	q := randVec(r, rows*heads*qk, 1)
	k8 := QuantizeRows(randVec(r, n*kvHeads*qk, 1), kvHeads*qk, qk)
	v8 := QuantizeRows(randVec(r, n*kvHeads*vd, 1), kvHeads*vd, vd)
	kf, vf := k8.Floats(), v8.Floats()
	scale := float32(math.Sqrt(qk))

	for _, causal := range []bool{false, true} {
		keys := func(int) int { return n }
		if causal {
			keys = func(i int) int { return n - rows + i + 1 }
		}

		for _, quantQ := range []bool{true, false} {
			probs := make([]float32, rows*heads*n)
			got := AttendInt8(q, rows, heads, k8, v8, keys, scale, quantQ, probs)

			qq := slices.Clone(q)
			if quantQ {
				fakeQuant(qq, qq, qk)
			}

			for i := range rows {
				for h := range heads {
					kv := h / (heads / kvHeads)
					s := make([]float32, keys(i))

					for j := range s {
						var dot float64
						for c := range qk {
							dot += float64(qq[(i*heads+h)*qk+c]) * float64(kf[(j*kvHeads+kv)*qk+c])
						}

						s[j] = float32(dot) / scale
					}

					softmaxInPlace(s)

					var sum float64

					for j, p := range s {
						got := probs[(i*heads+h)*n+j]
						sum += float64(got)

						if d := math.Abs(float64(p - got)); d > 2e-6 {
							t.Fatalf("causal %v quantQ %v row %d head %d key %d: p %v, want %v", causal, quantQ, i, h, j,
								got, p)
						}
					}

					if math.Abs(sum-1) > 1e-5 {
						t.Errorf("row %d head %d: probabilities sum to %v", i, h, sum)
					}

					for c := range vd {
						var want float64
						for j, p := range s {
							want += float64(p) * float64(vf[(j*kvHeads+kv)*vd+c])
						}

						if d := math.Abs(float64(got[(i*heads+h)*vd+c]) - want); d > 2e-6 {
							t.Fatalf("causal %v quantQ %v row %d head %d out %d: %v, want %v", causal, quantQ, i, h, c,
								got[(i*heads+h)*vd+c], want)
						}
					}
				}
			}
		}
	}
}

// TestAttentionInt8Extend runs Whistle's attention geometry with the
// engine's int8 numerics: the int8 cache row by row (AttendInt8) and in
// one block (the GEMMs on the dequantised cache) both stay close to
// Forward with the same quantisation, and a fork extends on its own. They
// are not equal bit for bit: a row's projections differ in the last bits
// between a block and a single row, which can move a value across a
// rounding boundary of its int8 grid.
func TestAttentionInt8Extend(t *testing.T) {
	const steps = 12

	a := benchAttention(t, 4)
	a.SetInt8(Int8{Act: true, KV: true})

	r := rand.New(rand.NewPCG(11, 12))
	x := randVec(r, steps*benchDim, 1)
	rope := NewRope(benchQK, steps, 1e5)

	want, err := a.Forward(x, steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	for _, sp := range []split{{"one block", []int{steps}}, {"per row", ones(steps)}, {"9+rest", []int{9, steps - 9}}} {
		var c KVCache

		got := runSplits(t, sp.sizes, benchDim, func(pos, n int) ([]float32, error) {
			return a.Extend(x[pos*benchDim:(pos+n)*benchDim], n, &c, rope)
		})

		if c.Len() != steps || c.k != nil || c.k8.Len() != steps || c.v8.Len() != steps {
			t.Errorf("%s: Len %d, float keys %d, int8 rows %d/%d", sp.name, c.Len(), len(c.k), c.k8.Len(), c.v8.Len())
		}

		cs := cosine32(got, want)
		t.Logf("%s: 1 − cos %.3g against Forward", sp.name, 1-cs)

		if 1-cs > 1e-5 {
			t.Errorf("%s: 1 − cos %.3g against Forward", sp.name, 1-cs)
		}
	}

	// A fork's rows go to its own arrays: the parent and the fork take
	// different rows after a shared prefix fed row by row.
	var c KVCache

	for i := range steps - 2 {
		_, err = a.Extend(x[i*benchDim:(i+1)*benchDim], 1, &c, rope)
		if err != nil {
			t.Fatal(err)
		}
	}

	f := c.Fork()

	yc, err := a.Extend(x[(steps-2)*benchDim:(steps-1)*benchDim], 1, &c, rope)
	if err != nil {
		t.Fatal(err)
	}

	yf, err := a.Extend(x[(steps-1)*benchDim:], 1, &f, rope)
	if err != nil {
		t.Fatal(err)
	}

	if slices.Equal(yc, yf) || f.k8.Len() != steps-1 || c.k8.Len() != steps-1 {
		t.Errorf("fork: %d and %d rows", f.k8.Len(), c.k8.Len())
	}

	if slices.Equal(f.k8.Q[(steps-2)*benchKVHeads*benchQK:], c.k8.Q[(steps-2)*benchKVHeads*benchQK:]) {
		t.Error("the fork's last key equals the parent's")
	}
}

// TestInt8Off: the zero Int8 leaves an attention as loaded.
func TestInt8Off(t *testing.T) {
	a, b := benchAttention(t, 4), benchAttention(t, 4)
	b.SetInt8(Int8{})

	r := rand.New(rand.NewPCG(13, 14))
	x := randVec(r, 9*benchDim, 1)
	rope := NewRope(benchQK, 9, 1e5)

	var ca, cb KVCache

	ya, err := a.Extend(x, 9, &ca, rope)
	if err != nil {
		t.Fatal(err)
	}

	yb, err := b.Extend(x, 9, &cb, rope)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(ya, yb) || cb.k8.Len() != 0 {
		t.Error("SetInt8(Int8{}) changed the attention")
	}
}

// TestLinearInt8Input: a Linear WithInt8Input multiplies the per-row
// quantised input, through Apply and ApplyShared alike, and ApplyShared
// over a mix of quantising and plain linears equals Apply on each, bit
// for bit.
func TestLinearInt8Input(t *testing.T) {
	const rows, in = 5, 64

	r := rand.New(rand.NewPCG(15, 16))
	plain := []Linear{NewLinear(randVec(r, 24*in, 0.1), 24, in), NewLinear(randVec(r, 8*in, 0.1), 8, in)}
	x := randVec(r, rows*in, 1)
	xq := quantInput(x, in)

	l := plain[0].WithInt8Input(true)
	if !l.Int8Input() || plain[0].Int8Input() {
		t.Fatal("WithInt8Input changed the original or did not set the flag")
	}

	got, want := make([]float32, rows*24), make([]float32, rows*24)
	l.Apply(got, x, rows)
	plain[0].Apply(want, xq, rows)

	if !slices.Equal(got, want) {
		t.Error("Apply with int8 input differs from Apply on the quantised input")
	}

	if v := l.Rows(4, 12); !v.Int8Input() {
		t.Error("Rows dropped the int8 input")
	}

	ls := []Linear{l, plain[1]}
	dsts := [][]float32{make([]float32, rows*24), make([]float32, rows*8)}
	ApplyShared(dsts, x, rows, ls...)

	for k, lk := range ls {
		w := make([]float32, rows*lk.Out())
		lk.Apply(w, x, rows)

		if !slices.Equal(dsts[k], w) {
			t.Errorf("ApplyShared output %d differs from Apply", k)
		}
	}
}

func cosine32(a, b []float32) float64 {
	var ab, aa, bb float64

	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}

	return ab / math.Sqrt(aa*bb)
}
