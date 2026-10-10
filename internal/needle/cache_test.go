package needle

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"
)

// split is one way to feed a sequence to Extend: the row counts of the
// successive calls.
type split struct {
	name  string
	sizes []int
}

// extendSplits are the splits of steps rows the parity tests run: one
// call, one row per call, and an uneven split (steps ≥ 6).
func extendSplits(steps int) []split {
	return []split{
		{"one call", []int{steps}},
		{"per row", ones(steps)},
		{"2+4+rest", []int{2, 4, steps - 6}},
	}
}

func ones(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = 1
	}

	return s
}

// runSplits calls step for each size with the first position and the row
// count of the call, and concatenates the outputs, n · width values each.
func runSplits(t *testing.T, sizes []int, width int, step func(pos, n int) ([]float32, error)) []float32 {
	t.Helper()

	var out []float32

	pos := 0

	for _, n := range sizes {
		y, err := step(pos, n)
		if err != nil {
			t.Fatalf("rows %d…%d: %v", pos, pos+n-1, err)
		}

		if len(y) != n*width {
			t.Fatalf("rows %d…%d: %d values, want %d", pos, pos+n-1, len(y), n*width)
		}

		out = append(out, y...)
		pos += n
	}

	return out
}

// TestAttentionExtendMatchesForward pins Extend to Forward with
// MaskCausal under the F-009 contract (docs/FINDINGS.md F-022), with and
// without taps: one call on an empty cache, one row per call, and an
// uneven split, each leaving Len() at T.
func TestAttentionExtendMatchesForward(t *testing.T) {
	const steps = 9

	for _, taps := range []int{3, 0} {
		t.Run("taps "+strconv.Itoa(taps), func(t *testing.T) {
			r := rand.New(rand.NewPCG(uint64(taps), 11))
			ta := newTinyAttention(r, "l.", taps)

			a, err := LoadAttention(ta.src, "l.")
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

				if c.Len() != steps {
					t.Errorf("%s: Len %d, want %d", sp.name, c.Len(), steps)
				}
			}
		})
	}
}

// TestAttentionFork pins KVCache.Fork (docs/FINDINGS.md F-026): after a
// shared prefix fed row by row (so the parent's arrays have spare
// capacity), the parent, a fork and a fork of the fork each continue with
// their own rows, interleaved. Every branch must equal Forward on its own
// whole sequence under the F-009 contract, so no branch writes into
// another's rows or tap history.
func TestAttentionFork(t *testing.T) {
	const prefix, tail = 5, 4

	r := rand.New(rand.NewPCG(43, 47))
	ta := newTinyAttention(r, "", 3)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	rope := NewRope(ta.qk, prefix+tail, 1e5)
	shared := randVec(r, prefix*ta.dim, 1)
	tails := [][]float32{randVec(r, tail*ta.dim, 1), randVec(r, tail*ta.dim, 1), randVec(r, tail*ta.dim, 1)}

	var parent KVCache

	for i := range prefix {
		_, err = a.Extend(shared[i*ta.dim:(i+1)*ta.dim], 1, &parent, rope)
		if err != nil {
			t.Fatal(err)
		}
	}

	child := parent.Fork()
	grand := child.Fork()
	caches := []*KVCache{&parent, &child, &grand}
	got := make([][]float32, len(caches))

	// Row i of every branch, the grandchild first and the parent last.
	for i := range tail {
		for b := len(caches) - 1; b >= 0; b-- {
			y, err := a.Extend(tails[b][i*ta.dim:(i+1)*ta.dim], 1, caches[b], rope)
			if err != nil {
				t.Fatal(err)
			}

			got[b] = append(got[b], y...)
		}
	}

	for b, name := range []string{"parent", "fork", "fork of fork"} {
		want, err := a.Forward(append(slices.Clone(shared), tails[b]...), prefix+tail, MaskCausal, rope)
		if err != nil {
			t.Fatal(err)
		}

		checkContract(t, name, got[b], want[prefix*ta.dim:])

		if caches[b].Len() != prefix+tail {
			t.Errorf("%s: Len %d, want %d", name, caches[b].Len(), prefix+tail)
		}
	}
}

// TestAttentionExtendErrors: a RoPE table shorter than Len()+t, one of
// the wrong head width and a ragged input are rejected, and a rejected
// call leaves the cache as it was.
func TestAttentionExtendErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 17))
	ta := newTinyAttention(r, "", 3)

	a, err := LoadAttention(ta.src, "")
	if err != nil {
		t.Fatal(err)
	}

	var c KVCache

	rope := NewRope(ta.qk, 4, 1e5)

	_, err = a.Extend(randVec(r, 3*ta.dim, 1), 3, &c, rope)
	if err != nil {
		t.Fatal(err)
	}

	call := func(_ any, err error) error { return err }

	for name, err := range map[string]error{
		"RoPE table shorter than Len()+t":   call(a.Extend(make([]float32, 2*ta.dim), 2, &c, rope)),
		"RoPE table of the wrong head size": call(a.Extend(make([]float32, ta.dim), 1, &c, NewRope(6, 8, 1e5))),
		"7 values for one row of 8":         call(a.Extend(make([]float32, 7), 1, &c, rope)),
	} {
		if err == nil {
			t.Errorf("accepted %s", name)
		}
	}

	if c.Len() != 3 {
		t.Errorf("Len %d after rejected calls, want 3", c.Len())
	}

	// The fourth row still fits the table.
	_, err = a.Extend(randVec(r, ta.dim, 1), 1, &c, rope)
	if err != nil {
		t.Errorf("row 3 of a 4-position table: %v", err)
	}
}

// TestEngramExtendMatchesForward pins Engram.Extend to Forward: the
// n-gram history and the dilated taps carry across calls, so one call,
// one token per call and an uneven split give the same keys and values,
// and a negative token is rejected without advancing the cache.
func TestEngramExtendMatchesForward(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 23))

	e, err := LoadEngram(tinyEngram(r, "", 7), "", []int{2, 3}, 4)
	if err != nil {
		t.Fatal(err)
	}

	tokens := []int{0, 8198, 5, 17, 5, 17, 5, 3, 1 << 20, 2, 2, 2, 9}
	steps := len(tokens)

	want, err := e.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}

	splits := []split{{"one call", []int{steps}}, {"per token", ones(steps)}, {"1+5+rest", []int{1, 5, steps - 6}}}

	for _, sp := range splits {
		var c EngramCache

		var k, v []float32

		pos := 0

		for _, n := range sp.sizes {
			kv, err := e.Extend(tokens[pos:pos+n], &c)
			if err != nil {
				t.Fatal(err)
			}

			k, v = append(k, kv.K...), append(v, kv.V...)
			pos += n
		}

		if c.Len() != steps {
			t.Errorf("%s: Len %d, want %d", sp.name, c.Len(), steps)
		}

		checkEngramKV(t, sp.name, &EngramKV{K: k, V: v}, want)
	}

	var c EngramCache

	_, err = e.Extend([]int{4, -1}, &c)
	if err == nil || c.Len() != 0 {
		t.Errorf("negative token: err %v, Len %d", err, c.Len())
	}
}

// TestEngramFork pins EngramCache.Fork (docs/FINDINGS.md F-026): two
// branches after a shared prefix, extended alternately with different
// tokens, each equal Forward on their whole sequence.
func TestEngramFork(t *testing.T) {
	r := rand.New(rand.NewPCG(53, 59))

	e, err := LoadEngram(tinyEngram(r, "", 7), "", []int{2, 3}, 4)
	if err != nil {
		t.Fatal(err)
	}

	shared := []int{2, 8192, 5, 17, 5, 17}
	tails := [][]int{{5, 3, 9, 9, 1}, {6, 3, 5, 17, 1}}

	var parent EngramCache

	for _, tok := range shared {
		_, err = e.Extend([]int{tok}, &parent)
		if err != nil {
			t.Fatal(err)
		}
	}

	child := parent.Fork()
	caches := []*EngramCache{&parent, &child}
	got := make([]EngramKV, len(caches))

	for i := range tails[0] {
		for b := len(caches) - 1; b >= 0; b-- {
			kv, err := e.Extend(tails[b][i:i+1], caches[b])
			if err != nil {
				t.Fatal(err)
			}

			got[b].K, got[b].V = append(got[b].K, kv.K...), append(got[b].V, kv.V...)
		}
	}

	for b, name := range []string{"parent", "fork"} {
		want, err := e.Forward(append(slices.Clone(shared), tails[b]...))
		if err != nil {
			t.Fatal(err)
		}

		off := len(shared) * e.Dim()
		checkEngramKV(t, name, &got[b], &EngramKV{K: want.K[off:], V: want.V[off:]})
	}
}

func checkEngramKV(t *testing.T, name string, got, want *EngramKV) {
	t.Helper()

	if len(got.K) != len(want.K) || len(got.V) != len(want.V) {
		t.Fatalf("%s: k/v %d/%d values, want %d/%d", name, len(got.K), len(got.V), len(want.K), len(want.V))
	}

	dk, dv := maxAbsDiff64(got.K, toFloat64(want.K)), maxAbsDiff64(got.V, toFloat64(want.V))
	t.Logf("%-10s k max abs %.3e  v max abs %.3e", name, dk, dv)

	if dk > 1e-5 || dv > 1e-5 {
		t.Errorf("%s: k max |Δ| %.3g, v max |Δ| %.3g", name, dk, dv)
	}
}

// TestBlockExtendMatchesForward pins Block.Extend with empty hooks to
// Forward with MaskCausal under the F-009 contract, without and with an
// engram site's keys and values, sliced per call.
func TestBlockExtendMatchesForward(t *testing.T) {
	const steps, dim = 9, 8

	r := rand.New(rand.NewPCG(29, 31))
	_, src := tinyBlock(r, "layer03.", 3)

	b, err := LoadBlock(src, "layer03.")
	if err != nil {
		t.Fatal(err)
	}

	x := randVec(r, steps*dim, 1)
	rope := NewRope(4, steps, 1e5)
	engram := &EngramKV{K: randVec(r, steps*dim, 1), V: randVec(r, steps*dim, 1)}

	for _, kv := range []*EngramKV{nil, engram} {
		want, err := b.Forward(x, steps, MaskCausal, rope, kv)
		if err != nil {
			t.Fatal(err)
		}

		for _, sp := range extendSplits(steps) {
			var c KVCache

			got := runSplits(t, sp.sizes, dim, func(pos, n int) ([]float32, error) {
				var rows *EngramKV
				if kv != nil {
					rows = &EngramKV{K: kv.K[pos*dim : (pos+n)*dim], V: kv.V[pos*dim : (pos+n)*dim]}
				}

				return b.Extend(x[pos*dim:(pos+n)*dim], n, &c, rope, rows, Hooks{})
			})
			checkContract(t, fmt.Sprintf("%s engram=%t", sp.name, kv != nil), got, want)

			if c.Len() != steps {
				t.Errorf("%s: Len %d, want %d", sp.name, c.Len(), steps)
			}
		}
	}
}

// TestBlockExtendHooks pins where the hooks sit: a shift added by
// BeforeAttn equals Extend on the shifted input, one added by AfterMLP
// equals the plain output shifted, one added by AfterAttn passes through
// the MLP and differs from both, and a hook's error is returned.
func TestBlockExtendHooks(t *testing.T) {
	const steps, dim = 5, 8

	r := rand.New(rand.NewPCG(37, 41))
	_, src := tinyBlock(r, "", 3)

	b, err := LoadBlock(src, "")
	if err != nil {
		t.Fatal(err)
	}

	x := randVec(r, steps*dim, 1)
	shift := randVec(r, dim, 0.5)
	rope := NewRope(4, steps, 1e5)
	add := func(h []float32, _ int) error {
		for i := range h {
			h[i] += shift[i%dim]
		}

		return nil
	}
	extend := func(x []float32, hooks Hooks) []float32 {
		var c KVCache

		y, err := b.Extend(x, steps, &c, rope, nil, hooks)
		if err != nil {
			t.Fatal(err)
		}

		return y
	}

	plain := extend(x, Hooks{})
	shiftedIn := slices.Clone(x)
	_ = add(shiftedIn, steps)
	shiftedOut := slices.Clone(plain)
	_ = add(shiftedOut, steps)

	if !slices.Equal(extend(x, Hooks{BeforeAttn: add}), extend(shiftedIn, Hooks{})) {
		t.Error("BeforeAttn shift differs from shifting the input")
	}

	if !slices.Equal(extend(x, Hooks{AfterMLP: add}), shiftedOut) {
		t.Error("AfterMLP shift differs from shifting the output")
	}

	afterAttn := extend(x, Hooks{AfterAttn: add})
	if slices.Equal(afterAttn, plain) || slices.Equal(afterAttn, shiftedOut) {
		t.Error("AfterAttn shift does not pass through the MLP")
	}

	errHook := errors.New("hook failed")
	fail := func([]float32, int) error { return errHook }

	// The rows after a two-row prefix, extended without a failure: what a
	// retry after a failed hook must reproduce.
	const pre = 2

	var ref KVCache

	_, err = b.Extend(x[:pre*dim], pre, &ref, rope, nil, Hooks{})
	if err != nil {
		t.Fatal(err)
	}

	want, err := b.Extend(x[pre*dim:], steps-pre, &ref, rope, nil, Hooks{})
	if err != nil {
		t.Fatal(err)
	}

	for name, hooks := range map[string]Hooks{
		"BeforeAttn": {BeforeAttn: fail}, "AfterAttn": {AfterAttn: fail}, "AfterMLP": {AfterMLP: fail},
	} {
		var c KVCache

		_, err := b.Extend(x[:pre*dim], pre, &c, rope, nil, Hooks{})
		if err != nil {
			t.Fatal(err)
		}

		_, err = b.Extend(x[pre*dim:], steps-pre, &c, rope, nil, hooks)
		if !errors.Is(err, errHook) {
			t.Errorf("%s: err %v, want %v", name, err, errHook)
		}

		if c.Len() != pre {
			t.Errorf("%s: the failed Extend left %d positions, want %d", name, c.Len(), pre)
		}

		got, err := b.Extend(x[pre*dim:], steps-pre, &c, rope, nil, Hooks{})
		if err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(got, want) {
			t.Errorf("%s: a retry after the failed hook differs from an uninterrupted Extend", name)
		}
	}
}

// TestBlockExtendHookCalls: every hook runs once per Extend call, in the
// order BeforeAttn, AfterAttn, AfterMLP, on the call's t rows.
func TestBlockExtendHookCalls(t *testing.T) {
	const dim = 8

	r := rand.New(rand.NewPCG(43, 47))
	_, src := tinyBlock(r, "", 3)

	b, err := LoadBlock(src, "")
	if err != nil {
		t.Fatal(err)
	}

	var calls []string

	record := func(name string) Hook {
		return func(h []float32, n int) error {
			if len(h) != n*dim {
				t.Errorf("%s: %d values for %d rows", name, len(h), n)
			}

			calls = append(calls, fmt.Sprintf("%s %d", name, n))

			return nil
		}
	}
	hooks := Hooks{BeforeAttn: record("before"), AfterAttn: record("attn"), AfterMLP: record("mlp")}
	rope := NewRope(4, 5, 1e5)

	var c KVCache

	for _, n := range []int{3, 2} {
		_, err := b.Extend(randVec(r, n*dim, 1), n, &c, rope, nil, hooks)
		if err != nil {
			t.Fatal(err)
		}
	}

	want := []string{"before 3", "attn 3", "mlp 3", "before 2", "attn 2", "mlp 2"}
	if !slices.Equal(calls, want) {
		t.Errorf("hook calls %q, want %q", calls, want)
	}
}

// TestWhistleDecoderIncremental runs Whistle's engram site 0 and decoder
// layer 3 token by token through Engram.Extend and Block.Extend and holds
// the rows to the whole-sequence Forward path under the F-009 contract.
func TestWhistleDecoderIncremental(t *testing.T) {
	const steps = 12

	src := openModel(t, whistleCactPath)
	cfg := src.Config()

	b, err := LoadBlock(src, "layer03.")
	if err != nil {
		t.Fatal(err)
	}

	e, err := LoadEngram(src, "engram0.", cfg.Orders, cfg.Tables)
	if err != nil {
		t.Fatal(err)
	}

	dim := b.Dim()
	r := rand.New(rand.NewPCG(53, 59))
	x := randVec(r, steps*dim, 1)

	tokens := make([]int, steps)
	for i := range tokens {
		tokens[i] = r.IntN(8199)
	}

	rope := NewRope(48, steps, 1e5)

	kv, err := e.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}

	want, err := b.Forward(x, steps, MaskCausal, rope, kv)
	if err != nil {
		t.Fatal(err)
	}

	var (
		ec EngramCache
		kc KVCache
	)

	got := runSplits(t, ones(steps), dim, func(pos, n int) ([]float32, error) {
		rows, err := e.Extend(tokens[pos:pos+n], &ec)
		if err != nil {
			return nil, err
		}

		return b.Extend(x[pos*dim:(pos+n)*dim], n, &kc, rope, rows, Hooks{})
	})
	checkContract(t, "layer03", got, want)
}
