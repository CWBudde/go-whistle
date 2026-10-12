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
// (upstream's tap_ok would cut them too), a window on the unmasked
// attention and Extend with a window on the int8 cache are rejected, the
// last without touching the cache.
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

	a.SetInt8(Int8{KV: true})

	var c KVCache

	_, err = a.Extend(randVec(r, 2*ta.dim, 1), 2, &c, nil)
	if err == nil {
		t.Error("Extend with a window on the int8 cache accepted")
	}

	if c.Len() != 0 {
		t.Errorf("Len %d after a rejected call, want 0", c.Len())
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
