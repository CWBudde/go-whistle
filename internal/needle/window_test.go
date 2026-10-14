package needle

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// TestAttentionWindowMatchesReference pins SetWindow to upstream's banded
// local mask, causal & (i − j) < window, on the per-row path (6 rows) and
// the GEMM path (12 rows), with and without taps, for windows from 1 to
// wider than the sequence.
func TestAttentionWindowMatchesReference(t *testing.T) {
	for _, taps := range []int{3, 0} {
		for _, steps := range []int{6, 12} {
			for _, window := range []int{1, 2, 3, 5, 12} {
				if window < taps {
					continue
				}

				t.Run(fmt.Sprintf("taps %d, %d rows, window %d", taps, steps, window), func(t *testing.T) {
					r := rand.New(rand.NewPCG(uint64(taps*100+steps), uint64(window)))
					ta := newTinyAttention(r, "l.", taps)

					a, err := LoadAttention(ta.src, "l.")
					if err != nil {
						t.Fatal(err)
					}

					err = a.SetWindow(window)
					if err != nil {
						t.Fatal(err)
					}

					x := randVec(r, steps*ta.dim, 1)
					rope := NewRope(ta.qk, steps, 1e5)

					got, err := a.Forward(x, steps, MaskCausal, rope)
					if err != nil {
						t.Fatal(err)
					}

					want := ta.refAttentionBand("l.", toFloat64(x), steps, true, window, rope)

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
	}
}

// TestAttentionWindowWide: a window at least as long as the sequence is
// the causal mask, bit for bit, on both attend paths; a narrower one
// changes the rows past it.
func TestAttentionWindowWide(t *testing.T) {
	for _, steps := range []int{6, 12} {
		r := rand.New(rand.NewPCG(5, uint64(steps)))
		ta := newTinyAttention(r, "", 3)

		a, err := LoadAttention(ta.src, "")
		if err != nil {
			t.Fatal(err)
		}

		x := randVec(r, steps*ta.dim, 1)
		rope := NewRope(ta.qk, steps, 1e5)

		causal, err := a.Forward(x, steps, MaskCausal, rope)
		if err != nil {
			t.Fatal(err)
		}

		for _, window := range []int{steps, steps + 1, 1 << 20} {
			err = a.SetWindow(window)
			if err != nil {
				t.Fatal(err)
			}

			got, err := a.Forward(x, steps, MaskCausal, rope)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(got, causal) {
				t.Errorf("%d rows, window %d: differs from the causal mask", steps, window)
			}
		}

		err = a.SetWindow(steps - 1)
		if err != nil {
			t.Fatal(err)
		}

		got, err := a.Forward(x, steps, MaskCausal, rope)
		if err != nil {
			t.Fatal(err)
		}

		last := (steps - 1) * ta.dim
		if !slices.Equal(got[:last], causal[:last]) {
			t.Errorf("%d rows, window %d: rows inside the window changed", steps, steps-1)
		}

		if slices.Equal(got[last:], causal[last:]) {
			t.Errorf("%d rows, window %d: the last row ignores the window", steps, steps-1)
		}
	}
}

// TestAttentionExtendWindow pins Extend with a window to Forward with the
// same window under the F-009 contract: one call, one row per call and an
// uneven split across the window's edge, and a fork that continues past
// it, so the cache's old keys stay but drop out of the band.
func TestAttentionExtendWindow(t *testing.T) {
	const steps, window, prefix = 13, 4, 6

	r := rand.New(rand.NewPCG(53, 59))
	ta := newTinyAttention(r, "", 3)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	err = a.SetWindow(window)
	if err != nil {
		t.Fatal(err)
	}

	x := randVec(r, steps*ta.dim, 1)
	rope := NewRope(ta.qk, steps, 1e5)

	want, err := a.Forward(x, steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	for _, sp := range extendSplits(steps) {
		var c KVCache

		got := runSplits(t, sp.sizes, ta.dim, func(pos, n int) ([]float32, error) {
			return a.Extend(x[pos*ta.dim:(pos+n)*ta.dim], n, &c, rope)
		})
		checkContract(t, sp.name, got, want)
	}

	var parent KVCache

	_, err = a.Extend(x[:prefix*ta.dim], prefix, &parent, rope)
	if err != nil {
		t.Fatal(err)
	}

	other := randVec(r, (steps-prefix)*ta.dim, 1)
	child := parent.Fork()

	gotChild, err := a.Extend(other, steps-prefix, &child, rope)
	if err != nil {
		t.Fatal(err)
	}

	gotParent, err := a.Extend(x[prefix*ta.dim:], steps-prefix, &parent, rope)
	if err != nil {
		t.Fatal(err)
	}

	checkContract(t, "parent after the fork", gotParent, want[prefix*ta.dim:])

	wantChild, err := a.Forward(append(slices.Clone(x[:prefix*ta.dim]), other...), steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	checkContract(t, "fork", gotChild, wantChild[prefix*ta.dim:])
}

// TestAttentionWindowErrors: a negative window, one narrower than the taps
// (upstream's tap_ok would cut them too) and a window on the unmasked
// attention are rejected.
func TestAttentionWindowErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(61, 67))
	ta := newTinyAttention(r, "", 3)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []int{-1, 1, 2} {
		if a.SetWindow(w) == nil {
			t.Errorf("SetWindow(%d) with 3 taps accepted", w)
		}
	}

	if a.Window() != 0 {
		t.Errorf("Window %d after rejected calls, want 0", a.Window())
	}

	err = a.SetWindow(3)
	if err != nil {
		t.Fatal(err)
	}

	if a.Window() != 3 {
		t.Errorf("Window %d, want 3", a.Window())
	}

	_, err = a.Forward(randVec(r, 4*ta.dim, 1), 4, MaskNone, nil)
	if err == nil {
		t.Error("a window on the unmasked attention accepted")
	}
}

// TestBlockWindow: Block.SetWindow reaches the attention, and Extend with
// it equals Forward with it, engram rows included.
func TestBlockWindow(t *testing.T) {
	const steps, dim, window = 11, 8, 4

	r := rand.New(rand.NewPCG(71, 73))
	_, src := tinyBlock(r, "layer03.", 3)

	b, err := LoadBlock(src, "layer03.")
	if err != nil {
		t.Fatal(err)
	}

	x := randVec(r, steps*dim, 1)
	rope := NewRope(4, steps, 1e5)
	kv := &EngramKV{K: randVec(r, steps*dim, 1), V: randVec(r, steps*dim, 1)}

	causal, err := b.Forward(x, steps, MaskCausal, rope, kv)
	if err != nil {
		t.Fatal(err)
	}

	if b.SetWindow(2) == nil {
		t.Error("SetWindow(2) with 3 taps accepted")
	}

	err = b.SetWindow(window)
	if err != nil {
		t.Fatal(err)
	}

	want, err := b.Forward(x, steps, MaskCausal, rope, kv)
	if err != nil {
		t.Fatal(err)
	}

	if slices.Equal(want[(steps-1)*dim:], causal[(steps-1)*dim:]) {
		t.Error("the window does not reach the attention")
	}

	for _, sp := range extendSplits(steps) {
		var c KVCache

		got := runSplits(t, sp.sizes, dim, func(pos, n int) ([]float32, error) {
			rows := &EngramKV{K: kv.K[pos*dim : (pos+n)*dim], V: kv.V[pos*dim : (pos+n)*dim]}

			return b.Extend(x[pos*dim:(pos+n)*dim], n, &c, rope, rows, Hooks{})
		})
		checkContract(t, sp.name, got, want)
	}
}

// TestAttendInt8Spans: AttendInt8Spans equals the float32 attention on
// the values its codes stand for over each row's two spans (a sink and a
// band, or a band alone), GQA and Whistle's head widths, within float32
// rounding; it writes no weight outside the spans; and with the first span
// empty and the second from 0 it is AttendInt8 bit for bit
// (docs/FINDINGS.md F-060).
func TestAttendInt8Spans(t *testing.T) {
	const heads, kvHeads, qk, vd, n, rows, window, sink = 8, 2, 48, 64, 37, 5, 9, 4

	r := rand.New(rand.NewPCG(83, 89))
	q := randVec(r, rows*heads*qk, 1)
	k8 := QuantizeRows(randVec(r, n*kvHeads*qk, 1), kvHeads*qk, qk)
	v8 := QuantizeRows(randVec(r, n*kvHeads*vd, 1), kvHeads*vd, vd)
	kf, vf := k8.Floats(), v8.Floats()
	scale := float32(math.Sqrt(qk))

	// The last rows of a causal sequence of n keys, banded to window, the
	// even rows with the first sink keys too.
	spans := func(i int) [2]Span {
		hi := n - rows + i + 1
		if i%2 == 0 {
			return [2]Span{{0, sink}, {hi - window, hi}}
		}

		return [2]Span{{hi - window, hi - window}, {hi - window, hi}}
	}

	for _, quantQ := range []bool{true, false} {
		probs := make([]float32, rows*heads*n)
		got := AttendInt8Spans(q, rows, heads, k8, v8, spans, scale, quantQ, probs)

		qq := slices.Clone(q)
		if quantQ {
			fakeQuant(qq, qq, qk)
		}

		for i := range rows {
			sp := spans(i)

			var keys []int

			for _, p := range sp {
				for j := p.Lo; j < p.Hi; j++ {
					keys = append(keys, j)
				}
			}

			for h := range heads {
				kv := h / (heads / kvHeads)
				s := make([]float32, len(keys))

				for j, key := range keys {
					var dot float64
					for c := range qk {
						dot += float64(qq[(i*heads+h)*qk+c]) * float64(kf[(key*kvHeads+kv)*qk+c])
					}

					s[j] = float32(dot) / scale
				}

				softmaxInPlace(s)

				row := slices.Clone(probs[(i*heads+h)*n : (i*heads+h+1)*n])

				for j, key := range keys {
					if d := math.Abs(float64(s[j] - row[key])); d > 2e-6 {
						t.Fatalf("quantQ %v row %d head %d key %d: p %v, want %v", quantQ, i, h, key, row[key], s[j])
					}

					row[key] = 0
				}

				if slices.ContainsFunc(row, func(p float32) bool { return p != 0 }) {
					t.Fatalf("quantQ %v row %d head %d: weights outside the spans %v", quantQ, i, h, sp)
				}

				for c := range vd {
					var want float64
					for j, key := range keys {
						want += float64(s[j]) * float64(vf[(key*kvHeads+kv)*vd+c])
					}

					if d := math.Abs(float64(got[(i*heads+h)*vd+c]) - want); d > 2e-6 {
						t.Fatalf("quantQ %v row %d head %d out %d: %v, want %v", quantQ, i, h, c,
							got[(i*heads+h)*vd+c], want)
					}
				}
			}
		}

		keys := func(i int) int { return n - rows + i + 1 }
		causal := func(i int) [2]Span { return [2]Span{{}, {0, keys(i)}} }

		if !slices.Equal(AttendInt8Spans(q, rows, heads, k8, v8, causal, scale, quantQ, nil),
			AttendInt8(q, rows, heads, k8, v8, keys, scale, quantQ, nil)) {
			t.Errorf("quantQ %v: AttendInt8Spans from 0 differs from AttendInt8", quantQ)
		}
	}
}

// TestAttentionInt8ExtendWindow pins Extend with a window on the int8 KV
// cache (docs/FINDINGS.md F-060) to Forward with the same window and the
// same int8 numerics, with GQA and the causal taps: one call, one row per
// call (AttendInt8Spans on the codes) and uneven splits across the window's
// edge, and a fork that continues past it. As TestAttentionInt8Extend, the
// rows agree within the int8 grid, not bit for bit; the window must change
// the rows past it.
func TestAttentionInt8ExtendWindow(t *testing.T) {
	const steps, window, prefix = 23, 6, 9

	a := benchAttention(t, 3)
	a.SetInt8(Int8{Act: true, KV: true})

	r := rand.New(rand.NewPCG(97, 101))
	x := randVec(r, steps*benchDim, 1)
	rope := NewRope(benchQK, steps, 1e5)

	causal, err := a.Forward(x, steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	err = a.SetWindow(window)
	if err != nil {
		t.Fatal(err)
	}

	want, err := a.Forward(x, steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	if 1-cosine32(want[window*benchDim:], causal[window*benchDim:]) < 1e-3 {
		t.Error("the window does not change the int8 rows past it")
	}

	splits := append(extendSplits(steps), split{"5+3+9+rest", []int{5, 3, 9, steps - 17}},
		split{"8+1+rest", []int{8, 1, steps - 9}})

	for _, sp := range splits {
		var c KVCache

		got := runSplits(t, sp.sizes, benchDim, func(pos, n int) ([]float32, error) {
			return a.Extend(x[pos*benchDim:(pos+n)*benchDim], n, &c, rope)
		})

		if c.Len() != steps || c.k != nil || c.k8.Len() != steps || c.v8.Len() != steps {
			t.Errorf("%s: Len %d, float keys %d, int8 rows %d/%d", sp.name, c.Len(), len(c.k), c.k8.Len(), c.v8.Len())
		}

		for i := range steps {
			cs := cosine32(got[i*benchDim:(i+1)*benchDim], want[i*benchDim:(i+1)*benchDim])
			if 1-cs > 1e-5 {
				t.Errorf("%s: row %d at 1 − cos %.3g against Forward", sp.name, i, 1-cs)
			}
		}
	}

	var parent KVCache

	_, err = a.Extend(x[:prefix*benchDim], prefix, &parent, rope)
	if err != nil {
		t.Fatal(err)
	}

	other := randVec(r, (steps-prefix)*benchDim, 1)
	child := parent.Fork()

	var gotChild []float32

	for i := range steps - prefix {
		y, err := a.Extend(other[i*benchDim:(i+1)*benchDim], 1, &child, rope)
		if err != nil {
			t.Fatal(err)
		}

		gotChild = append(gotChild, y...)
	}

	wantChild, err := a.Forward(append(slices.Clone(x[:prefix*benchDim]), other...), steps, MaskCausal, rope)
	if err != nil {
		t.Fatal(err)
	}

	if cs := cosine32(gotChild, wantChild[prefix*benchDim:]); 1-cs > 1e-5 {
		t.Errorf("fork: 1 − cos %.3g against Forward", 1-cs)
	}
}

// TestWindowSameBand: with a window, or a KV window and a sink, a decoding
// step on the float32 cache and one on the int8 cache read the same keys
// and values: corrupting every cached row outside the step's band leaves
// its row unchanged bit for bit, corrupting the band's first row or a
// sink row changes it (docs/FINDINGS.md F-060).
func TestWindowSameBand(t *testing.T) {
	const steps, window = 17, 5

	r := rand.New(rand.NewPCG(103, 107))
	x := randVec(r, (steps+1)*benchDim, 1)
	rope := NewRope(benchQK, steps+1, 1e5)
	lo := steps + 1 - window // the band of the step at position steps

	for _, sink := range []int{0, 3} {
		for _, q := range []Int8{{}, {Act: true, KV: true}} {
			a := benchAttention(t, 3)
			a.SetInt8(q)

			set := a.SetWindow
			if sink > 0 {
				set = a.SetKVWindow
			}

			err := set(window)
			if err != nil {
				t.Fatal(err)
			}

			var c KVCache

			c.SetSink(sink)

			_, err = a.Extend(x[:steps*benchDim], steps, &c, rope)
			if err != nil {
				t.Fatal(err)
			}

			// corrupt sets the cached keys and values of rows [from, to)
			// of a fork of c to garbage.
			corrupt := func(from, to int) KVCache {
				f := c.Fork()

				if q.KV {
					f.k8.Q, f.v8.Q = slices.Clone(f.k8.Q), slices.Clone(f.v8.Q)
					f.k8.S, f.v8.S = slices.Clone(f.k8.S), slices.Clone(f.v8.S)

					for i := from * f.k8.Cols; i < to*f.k8.Cols; i++ {
						f.k8.Q[i] = int8(r.IntN(255) - 127)
					}

					for i := from * f.v8.Cols; i < to*f.v8.Cols; i++ {
						f.v8.Q[i] = int8(r.IntN(255) - 127)
					}

					for i := from * benchKVHeads; i < to*benchKVHeads; i++ {
						f.k8.S[i], f.v8.S[i] = 3, 3
					}

					return f
				}

				f.k, f.v = slices.Clone(f.k), slices.Clone(f.v)
				kW, vW := benchKVHeads*benchQK, benchKVHeads*benchV

				copy(f.k[from*kW:to*kW], randVec(r, (to-from)*kW, 3))
				copy(f.v[from*vW:to*vW], randVec(r, (to-from)*vW, 3))

				return f
			}

			step := func(f KVCache) []float32 {
				y, err := a.Extend(x[steps*benchDim:], 1, &f, rope)
				if err != nil {
					t.Fatal(err)
				}

				return y
			}

			clean := step(c.Fork())

			if !slices.Equal(step(corrupt(sink, lo)), clean) {
				t.Errorf("sink %d, int8 %+v: the step reads rows outside its band", sink, q)
			}

			if slices.Equal(step(corrupt(lo, lo+1)), clean) {
				t.Errorf("sink %d, int8 %+v: the step ignores the band's first row", sink, q)
			}

			if sink > 0 && slices.Equal(step(corrupt(0, 1)), clean) {
				t.Errorf("sink %d, int8 %+v: the step ignores the sink", sink, q)
			}
		}
	}
}

// TestAttentionKVWindowMatchesReference pins SetKVWindow and the cache's
// sink to upstream's hidden_cells mask, causal & band & ((i − j) < kv
// window | j < sink) (docs/FINDINGS.md F-060), on Extend in one call (the
// GEMM path, 12 rows) and one row per call (the per-row path), with and
// without taps and the band of SetWindow; Forward, which has no sink,
// equals the reference with none.
func TestAttentionKVWindowMatchesReference(t *testing.T) {
	const steps = 12

	for _, taps := range []int{3, 0} {
		for _, band := range []int{0, 7} {
			for _, kvw := range []int{3, 5} {
				for _, sink := range []int{0, 1, 4} {
					t.Run(fmt.Sprintf("taps %d, band %d, KV window %d, sink %d", taps, band, kvw, sink), func(t *testing.T) {
						r := rand.New(rand.NewPCG(uint64(taps*100+band), uint64(kvw*10+sink)))
						ta := newTinyAttention(r, "l.", taps)

						a, err := LoadAttention(ta.src, "l.")
						if err != nil {
							t.Fatal(err)
						}

						if a.SetWindow(band) != nil || a.SetKVWindow(kvw) != nil {
							t.Fatal("window rejected")
						}

						x := randVec(r, steps*ta.dim, 1)
						rope := NewRope(ta.qk, steps, 1e5)
						visible := func(sink int) func(i, j int) bool {
							return func(i, j int) bool {
								return j <= i && (band == 0 || i-j < band) && (i-j < kvw || j < sink)
							}
						}

						want := ta.refAttentionMask("l.", toFloat64(x), steps, visible(sink), rope)

						var refMax float64
						for _, w := range want {
							refMax = math.Max(refMax, math.Abs(w))
						}

						check := func(name string, got []float32, want []float64) {
							t.Helper()

							if d := maxAbsDiff64(got, want); d > 1e-5*math.Max(1, refMax) {
								t.Errorf("%s: max |Δ| %.3g at max|ref| %.3g", name, d, refMax)
							}
						}

						for _, sp := range []split{{"one call", []int{steps}}, {"per row", ones(steps)}} {
							var c KVCache

							c.SetSink(sink)

							got := runSplits(t, sp.sizes, ta.dim, func(pos, n int) ([]float32, error) {
								return a.Extend(x[pos*ta.dim:(pos+n)*ta.dim], n, &c, rope)
							})
							check(sp.name, got, want)
						}

						got, err := a.Forward(x, steps, MaskCausal, rope)
						if err != nil {
							t.Fatal(err)
						}

						check("Forward", got, ta.refAttentionMask("l.", toFloat64(x), steps, visible(0), rope))
					})
				}
			}
		}
	}
}

// TestAttentionKVWindowErrors: a negative KV window or one narrower than
// the taps is rejected, as is one on the unmasked attention, and a
// negative sink panics; a fork keeps the sink.
func TestAttentionKVWindowErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(109, 113))
	ta := newTinyAttention(r, "", 3)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []int{-1, 1, 2} {
		if a.SetKVWindow(w) == nil {
			t.Errorf("SetKVWindow(%d) with 3 taps accepted", w)
		}
	}

	if a.KVWindow() != 0 {
		t.Errorf("KVWindow %d after rejected calls, want 0", a.KVWindow())
	}

	err = a.SetKVWindow(4)
	if err != nil || a.KVWindow() != 4 {
		t.Fatalf("SetKVWindow(4): %v, KVWindow %d", err, a.KVWindow())
	}

	_, err = a.Forward(randVec(r, 4*ta.dim, 1), 4, MaskNone, nil)
	if err == nil {
		t.Error("a KV window on the unmasked attention accepted")
	}

	var c KVCache

	c.SetSink(5)

	if f := c.Fork(); f.Sink() != 5 {
		t.Errorf("the fork's sink is %d, want 5", f.Sink())
	}

	defer func() {
		if recover() == nil {
			t.Error("a negative sink did not panic")
		}
	}()

	c.SetSink(-1)
}

// TestAttentionInt8KVWindowSink: with the int8 KV cache, a KV window, a
// sink and the band, Extend one row per call (AttendInt8Spans on the
// codes) and in uneven splits agrees with Extend in one call (the GEMMs on
// the values the codes stand for), the text model's Forward under the KV
// window, within the int8 grid; a fork continues on its own; and the sink
// changes the rows past the window (docs/FINDINGS.md F-060).
func TestAttentionInt8KVWindowSink(t *testing.T) {
	const steps, band, kvw, sink, prefix = 30, 20, 6, 4, 11

	a := benchAttention(t, 3)
	a.SetInt8(Int8{Act: true, KV: true})

	if a.SetWindow(band) != nil || a.SetKVWindow(kvw) != nil {
		t.Fatal("window rejected")
	}

	r := rand.New(rand.NewPCG(127, 131))
	x := randVec(r, steps*benchDim, 1)
	rope := NewRope(benchQK, steps, 1e5)

	extend := func(sizes []int, sink int) []float32 {
		var c KVCache

		c.SetSink(sink)

		return runSplits(t, sizes, benchDim, func(pos, n int) ([]float32, error) {
			return a.Extend(x[pos*benchDim:(pos+n)*benchDim], n, &c, rope)
		})
	}

	want := extend([]int{steps}, sink)

	var c8, cf KVCache

	_, err := a.Extend(x, steps, &c8, rope)
	if err != nil {
		t.Fatal(err)
	}

	a.SetInt8(Int8{})

	_, err = a.Extend(x, steps, &cf, rope)
	if err != nil {
		t.Fatal(err)
	}

	a.SetInt8(Int8{Act: true, KV: true})

	// Per position 2 KV heads of 48 key and 64 value codes and a scale
	// each, against 4 bytes per value.
	if c8.Bytes() != steps*(2*(48+64)+4*4) || cf.Bytes() != steps*4*2*(48+64) {
		t.Errorf("cache bytes %d int8, %d float32", c8.Bytes(), cf.Bytes())
	}

	if 1-cosine32(want[kvw*benchDim:], extend([]int{steps}, 0)[kvw*benchDim:]) < 1e-3 {
		t.Error("the sink does not change the rows past the KV window")
	}

	for _, sp := range []split{{"per row", ones(steps)}, {"5+3+9+rest", []int{5, 3, 9, steps - 17}}, {"8+1+rest", []int{8, 1, steps - 9}}} {
		got := extend(sp.sizes, sink)

		for i := range steps {
			cs := cosine32(got[i*benchDim:(i+1)*benchDim], want[i*benchDim:(i+1)*benchDim])
			if 1-cs > 1e-5 {
				t.Errorf("%s: row %d at 1 − cos %.3g against one call", sp.name, i, 1-cs)
			}
		}
	}

	var parent KVCache

	parent.SetSink(sink)

	_, err = a.Extend(x[:prefix*benchDim], prefix, &parent, rope)
	if err != nil {
		t.Fatal(err)
	}

	other := randVec(r, (steps-prefix)*benchDim, 1)
	child := parent.Fork()

	var gotChild []float32

	for i := range steps - prefix {
		y, err := a.Extend(other[i*benchDim:(i+1)*benchDim], 1, &child, rope)
		if err != nil {
			t.Fatal(err)
		}

		gotChild = append(gotChild, y...)
	}

	var whole KVCache

	whole.SetSink(sink)

	wantChild, err := a.Extend(append(slices.Clone(x[:prefix*benchDim]), other...), steps, &whole, rope)
	if err != nil {
		t.Fatal(err)
	}

	if cs := cosine32(gotChild, wantChild[prefix*benchDim:]); 1-cs > 1e-5 {
		t.Errorf("fork: 1 − cos %.3g against one call", 1-cs)
	}
}
