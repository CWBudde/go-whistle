package text

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

// minCosExtend bounds the incremental logits against Forward's: the same
// float32 model, only the summation order differs (docs/FINDINGS.md
// F-022).
const minCosExtend = 0.99999

// extendRows feeds tokens to a new state in calls of the given sizes and
// returns the logits of every position.
func extendRows(t *testing.T, m *Model, s *State, tokens []int, sizes []int) []float32 {
	t.Helper()

	var out []float32

	pos := 0

	for _, n := range sizes {
		logits, err := m.Extend(s, tokens[pos:pos+n])
		if err != nil {
			t.Fatalf("tokens %d…%d: %v", pos, pos+n-1, err)
		}

		if len(logits) != n*m.cfg.OutVocab {
			t.Fatalf("tokens %d…%d: %d logits, want %d rows of %d", pos, pos+n-1, len(logits), n, m.cfg.OutVocab)
		}

		out = append(out, logits...)
		pos += n
	}

	if s.Len() != len(tokens) {
		t.Fatalf("Len %d after %d tokens", s.Len(), len(tokens))
	}

	return out
}

// checkLogits compares two runs' logits row by row: cosine at least
// minCosExtend and the same top-1 at every position.
func checkLogits(t *testing.T, name string, got, want []float32, v int) {
	t.Helper()

	c, row := rowCos(got, want, v)
	if c < minCosExtend {
		t.Errorf("%s: logits cosine %.9f at %d < %v", name, c, row, minCosExtend)
	}

	for p := range len(want) / v {
		if g, w := argmax(got[p*v:(p+1)*v]), argmax(want[p*v:(p+1)*v]); g != w {
			t.Errorf("%s: position %d: top-1 %d, Forward %d", name, p, g, w)
		}
	}

	t.Logf("%s: worst row cosine %.12f at %d", name, c, row)
}

// TestExtendMatchesForward pins Extend to Forward on every fixture case:
// the whole sequence in one call, the prompt in one call and the
// completion a token at a time (decoding), and uneven chunks, which on the
// long case straddle the 1024-position window. Every position's logits
// must match Forward's, and so its top-1 the JAX reference's.
func TestExtendMatchesForward(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	fx := openFixture(t)
	v := m.cfg.OutVocab

	for _, name := range forwardCases {
		t.Run(name, func(t *testing.T) {
			tokens := fx.ints(t, name+".tokens")
			prompt := fx.ints(t, name+".prompt_len")[0]
			wantArg := fx.ints(t, name+".argmax")
			n := len(tokens)

			full, err := m.Forward(tokens, false)
			if err != nil {
				t.Fatal(err)
			}

			first, second := 7, 50
			if name == longCase {
				first, second = 1000, 30
			}

			uneven := []int{first, second, n - first - second}

			for _, sp := range []struct {
				name  string
				sizes []int
			}{
				{"one call", []int{n}},
				{"prompt, then per token", append([]int{prompt}, ones(n-prompt)...)},
				{fmt.Sprint(uneven), uneven},
			} {
				got := extendRows(t, m, m.NewState(), tokens, sp.sizes)
				checkLogits(t, sp.name, got, full.Logits, v)

				for p := range n {
					if g := argmax(got[p*v : (p+1)*v]); g != wantArg[p] {
						t.Errorf("%s: position %d: top-1 %d, JAX %d", sp.name, p, g, wantArg[p])
					}
				}
			}
		})
	}
}

func ones(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = 1
	}

	return s
}

// TestExtendFork: from one prompt, a fork and the parent continue with
// different tokens, the fork first, a token at a time; each must equal
// Forward on its own whole sequence, so neither writes into the other's
// keys, values or engram history. A fork of the fork continues too.
func TestExtendFork(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	fx := openFixture(t)
	v := m.cfg.OutVocab

	tokens := fx.ints(t, "timer_en.tokens")
	prompt := fx.ints(t, "timer_en.prompt_len")[0]
	tail := tokens[prompt:]

	other := slices.Clone(tail)
	slices.Reverse(other)

	third := fx.ints(t, "message_de.tokens")[:len(tail)]

	parent := m.NewState()

	_, err := m.Extend(parent, tokens[:prompt])
	if err != nil {
		t.Fatal(err)
	}

	child := parent.Fork()
	grand := child.Fork()
	states := []*State{parent, child, grand}
	tails := [][]int{tail, other, third}
	got := make([][]float32, len(states))

	// Token i of every branch, the grandchild first and the parent last.
	for i := range tail {
		for b := len(states) - 1; b >= 0; b-- {
			logits, err := m.Extend(states[b], tails[b][i:i+1])
			if err != nil {
				t.Fatal(err)
			}

			got[b] = append(got[b], logits...)
		}
	}

	for b, name := range []string{"parent", "fork", "fork of fork"} {
		seq := append(slices.Clone(tokens[:prompt]), tails[b]...)

		want, err := m.Forward(seq, false)
		if err != nil {
			t.Fatal(err)
		}

		checkLogits(t, name, got[b], want.Logits[prompt*v:], v)

		if !slices.Equal(states[b].Tokens(), seq) {
			t.Errorf("%s: Tokens differ from the sequence fed", name)
		}
	}
}

// TestExtendErrors: a token outside the vocabulary, a sequence past
// MaxSeqLen and another model, even of the same geometry, are rejected
// without changing the state, which then extends as if they had not
// happened; no tokens is a no-op.
func TestExtendErrors(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadFloat(t)
	fx := openFixture(t)
	v := m.cfg.OutVocab
	tokens := fx.ints(t, "timer_en.tokens")
	half := len(tokens) / 2

	want, err := m.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	s := m.NewState()

	_, err = m.Extend(s, tokens[:half])
	if err != nil {
		t.Fatal(err)
	}

	for name, bad := range map[string][]int{
		"a negative token":                {tokens[half], -1},
		"a token past the vocabulary":     {tokens[half], m.cfg.Vocab},
		"more positions than max_seq_len": make([]int, m.cfg.MaxSeqLen),
	} {
		_, err := m.Extend(s, bad)
		if err == nil {
			t.Errorf("%s accepted", name)
		}

		if s.Len() != half {
			t.Errorf("%s: Len %d, want %d", name, s.Len(), half)
		}
	}

	_, err = m.Extend(s, make([]int, m.cfg.MaxSeqLen))
	if !errors.Is(err, ErrTooLong) {
		t.Errorf("past max_seq_len: %v, want ErrTooLong", err)
	}

	other := &Model{cfg: m.cfg, blocks: m.blocks, engrams: m.engrams}

	_, err = other.Extend(s, tokens[half:half+1])
	if err == nil {
		t.Error("a state extended by another model of the same geometry")
	}

	if s.Len() != half {
		t.Errorf("another model: Len %d, want %d", s.Len(), half)
	}

	logits, err := m.Extend(s, nil)
	if logits != nil || err != nil {
		t.Errorf("no tokens: %d logits, %v", len(logits), err)
	}

	got, err := m.Extend(s, tokens[half:])
	if err != nil {
		t.Fatal(err)
	}

	checkLogits(t, "after the rejected calls", got, want.Logits[half*v:], v)
}
