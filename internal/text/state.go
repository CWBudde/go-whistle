package text

import (
	"errors"
	"fmt"
	"slices"

	"github.com/cwbudde/go-whistle/internal/needle"
)

// State is one sequence's decoding state for Extend (docs/FINDINGS.md
// F-022, F-026): the self-attention's KVCache per layer, the EngramCache
// per engram site and the tokens so far. The local layers' caches keep
// every position and attend to the last SlidingWindow of them; with
// LoadOptions.KVWindow each layer attends to the last KVWindow and the
// sink, which the state sets on its caches: every position until the sink
// token is fed, then the positions up to it (docs/FINDINGS.md F-060). A
// state belongs to the model that made it.
type State struct {
	model     *Model
	kv        []needle.KVCache
	engram    []needle.EngramCache
	tokens    []int
	sink      int  // the sink's positions (Model.sinkIn)
	sinkFound bool // the sink token was fed, so sink is final
}

// NewState starts an empty sequence.
func (m *Model) NewState() *State {
	return &State{
		model: m, kv: make([]needle.KVCache, len(m.blocks)), engram: make([]needle.EngramCache, len(m.engrams)),
	}
}

// Len is the number of tokens fed so far.
func (s *State) Len() int { return len(s.tokens) }

// Tokens is the token history.
func (s *State) Tokens() []int { return slices.Clone(s.tokens) }

// Fork returns a state with s's history that extends independently of s,
// as one beam's continuation: the cached keys and values of the prefix are
// shared until either state appends to them (needle.KVCache.Fork).
func (s *State) Fork() *State {
	f := &State{
		model:     s.model,
		kv:        make([]needle.KVCache, len(s.kv)),
		engram:    make([]needle.EngramCache, len(s.engram)),
		tokens:    slices.Clip(s.tokens),
		sink:      s.sink,
		sinkFound: s.sinkFound,
	}

	for i := range s.kv {
		f.kv[i] = s.kv[i].Fork()
	}

	for i := range s.engram {
		f.engram[i] = s.engram[i].Fork()
	}

	return f
}

// Extend feeds tokens to the sequence s and returns their next-token
// logits, one row of OutVocab values per token: row i scores the token
// after tokens[i]. Feeding a sequence in one call or in pieces gives
// Forward's logits up to float32 summation order. On an error s is left
// as it was. No tokens return nil.
func (m *Model) Extend(s *State, tokens []int) ([]float32, error) {
	t, off := len(tokens), s.Len()
	if t == 0 {
		return nil, nil
	}

	// Another model's caches hold its weights' keys, values and engram
	// rows, even where the geometry matches (PR #51 review).
	if s.model != m {
		return nil, errors.New("text: the state belongs to another model")
	}

	if m.cfg.MaxSeqLen <= 0 {
		return nil, errors.New("text: Extend needs max_seq_len for the RoPE tables")
	}

	err := m.checkLen(off + t)
	if err != nil {
		return nil, err
	}

	x, err := m.embed(tokens, off)
	if err != nil {
		return nil, err
	}

	m.ropeOnce.Do(func() { m.rope = needle.NewRope(m.cfg.QKDim, m.cfg.MaxSeqLen, m.cfg.Rope) })

	// The caches only append or replace their slices, so copies of the
	// headers are the old state.
	kv, engram := slices.Clone(s.kv), slices.Clone(s.engram)

	sink, found := s.sink, s.sinkFound
	if !found && m.opts.KVWindow > 0 {
		sink, found = m.sinkIn(tokens, off)
		for l := range s.kv {
			s.kv[l].SetSink(sink)
		}
	}

	logits, err := m.extend(s, x, tokens)
	if err != nil {
		s.kv, s.engram = kv, engram

		return nil, err
	}

	s.tokens, s.sink, s.sinkFound = append(s.tokens, tokens...), sink, found

	return logits, nil
}

// extend runs the stack on the embedded rows x of tokens, advancing s's
// caches.
func (m *Model) extend(s *State, x []float32, tokens []int) ([]float32, error) {
	t := len(tokens)
	ekv := make([]*needle.EngramKV, len(m.engrams))

	for i, e := range m.engrams {
		var err error

		ekv[i], err = e.Extend(tokens, &s.engram[i])
		if err != nil {
			return nil, fmt.Errorf("text: %w", err)
		}
	}

	stream := needle.ExpandLanes(x, m.cfg.Lanes, m.cfg.Dim)

	for l, b := range m.blocks {
		var site *needle.EngramKV
		if i, ok := m.site[l]; ok {
			site = ekv[i]
		}

		block := func(u []float32) ([]float32, error) {
			return b.Extend(u, t, &s.kv[l], m.rope, site, needle.Hooks{})
		}

		var err error

		stream, err = m.mhc[l].Forward(stream, t, block)
		if err != nil {
			return nil, fmt.Errorf("text: layer %d: %w", l, err)
		}
	}

	_, logits := m.output(stream, t)

	return logits, nil
}
