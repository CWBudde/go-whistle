// Package text is the Needle 3 text model (tool calling, HF
// Cactus-Compute/needle3 @ 93e88ede; PLAN.md Phase 11): upstream's
// SimpleAttentionNetwork (needle/model/architecture.py @ af2654dffa) for
// the text config, in float32, on the shared blocks of internal/needle.
//
//	x      = embedding[tokens] · √d
//	stream = 20 × mHC(Block)          engram at layers 3, 7, 11, 15, 19
//	h      = final_norm(lane mean of stream)
//	logits = h · embedding[:out_vocab]ᵀ
//
// The blocks are Whistle's decoder blocks at width 768 (12 query heads,
// GQA 12:2, a HadamardMLP of 1024 that zero-pads the width), with five
// engram sites of six tables and no cross-attention. Layers outside
// global_layers attend within a sliding window of 1024 positions, the
// band causal & (pos_q − pos_k) < sliding_window of upstream's local mask;
// the global layers see every earlier position (docs/FINDINGS.md F-045).
package text

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// ErrTooLong reports a sequence longer than MaxSeqLen, the longest the
// model was trained for.
var ErrTooLong = errors.New("text: sequence longer than max_seq_len")

// Model is the float32 text model.
type Model struct {
	cfg   Config
	emb   needle.Linear // [vocab, dim], the input table and the tied head
	head  needle.Linear // its first OutVocab rows
	scale float32       // √dim, SimpleAttentionNetwork's embed_scale

	blocks  []*needle.Block
	mhc     []*needle.MHC
	engrams []*needle.Engram // by site
	site    map[int]int      // layer → site
	final   needle.ZCRMSNorm
}

// checkConfig requires OutVocab in [1, Vocab] and cfg.GlobalLayers to be
// distinct layers in [0, Layers): Forward takes "as many global layers as
// layers" to mean that no layer is windowed, which duplicates would fake
// (PR #44 review).
func checkConfig(cfg Config) error {
	if cfg.OutVocab < 1 || cfg.OutVocab > cfg.Vocab {
		return fmt.Errorf("text: out_vocab %d for %d embedding rows", cfg.OutVocab, cfg.Vocab)
	}

	seen := make(map[int]bool, len(cfg.GlobalLayers))

	for _, l := range cfg.GlobalLayers {
		if l < 0 || l >= cfg.Layers || seen[l] {
			return fmt.Errorf("text: global_layers %v: want distinct layers in [0, %d)", cfg.GlobalLayers, cfg.Layers)
		}

		seen[l] = true
	}

	return nil
}

// Load builds the model from src, which serves the cact.TextNames names
// (Source does), with the geometry cfg. The blocks outside
// cfg.GlobalLayers get the window cfg.SlidingWindow (0 for none).
func Load(src weights.Source, cfg Config) (*Model, error) {
	err := checkConfig(cfg)
	if err != nil {
		return nil, err
	}

	m := &Model{cfg: cfg, scale: float32(math.Sqrt(float64(cfg.Dim))), site: map[int]int{}}

	m.emb, err = needle.LoadLinear(src, "embedding", cfg.Vocab, cfg.Dim)
	if err != nil {
		return nil, fmt.Errorf("text: %w", err)
	}

	m.head = m.emb.Rows(0, cfg.OutVocab)

	for l := range cfg.Layers {
		b, h, err := loadLayer(src, cfg, l)
		if err != nil {
			return nil, err
		}

		m.blocks, m.mhc = append(m.blocks, b), append(m.mhc, h)
	}

	for s, l := range cfg.Sites {
		if l < 0 || l >= cfg.Layers {
			return nil, fmt.Errorf("text: engram site %d at layer %d of %d", s, l, cfg.Layers)
		}

		e, err := needle.LoadEngram(src, fmt.Sprintf("engram%d.", s), cfg.Orders, cfg.Tables)
		if err != nil {
			return nil, fmt.Errorf("text: %w", err)
		}

		m.engrams = append(m.engrams, e)
		m.site[l] = s
	}

	m.final, err = needle.LoadZCRMSNorm(src, "final_norm")
	if err != nil {
		return nil, fmt.Errorf("text: %w", err)
	}

	return m, nil
}

// loadLayer loads layer l's block and mHC and gives a local layer the
// sliding window.
func loadLayer(src weights.Source, cfg Config, l int) (*needle.Block, *needle.MHC, error) {
	b, err := needle.LoadBlock(src, fmt.Sprintf("layer%02d.", l))
	if err != nil {
		return nil, nil, fmt.Errorf("text: %w", err)
	}

	h, err := needle.LoadMHC(src, "", l)
	if err != nil {
		return nil, nil, fmt.Errorf("text: %w", err)
	}

	if b.Dim() != cfg.Dim || h.Dim() != cfg.Dim || h.Lanes() != cfg.Lanes {
		return nil, nil, fmt.Errorf("text: layer %d widths %d/%d, %d lanes; config %d, %d lanes",
			l, b.Dim(), h.Dim(), h.Lanes(), cfg.Dim, cfg.Lanes)
	}

	if !slices.Contains(cfg.GlobalLayers, l) {
		err = b.SetWindow(cfg.SlidingWindow)
		if err != nil {
			return nil, nil, fmt.Errorf("text: layer %d: %w", l, err)
		}
	}

	return b, h, nil
}

// Config is the model's geometry.
func (m *Model) Config() Config { return m.cfg }

// Output is one forward pass over a token sequence of t positions.
type Output struct {
	// Embed is the scaled input embedding, t rows of Dim values.
	Embed []float32
	// Hidden is the lane mean of the stream after each block, t rows of
	// Dim values per block (upstream's collect_hidden); nil unless asked
	// for.
	Hidden [][]float32
	// Final is the stack output after final_norm, t rows of Dim values.
	Final []float32
	// Logits are the next-token logits, t rows of OutVocab values.
	Logits []float32
}

// Forward runs the model on tokens with the causal mask, as
// SimpleAttentionNetwork.__call__ with mask=None and quant=False (the
// causal mask, banded on the local layers), and returns every position's
// output. hidden collects the per-block lane means. Sequences longer than
// MaxSeqLen fail with ErrTooLong.
func (m *Model) Forward(tokens []int, hidden bool) (*Output, error) {
	t, d := len(tokens), m.cfg.Dim
	if t == 0 {
		return nil, errors.New("text: no tokens")
	}

	err := m.checkLen(t)
	if err != nil {
		return nil, err
	}

	out := &Output{}

	out.Embed, err = m.embed(tokens, 0)
	if err != nil {
		return nil, err
	}

	kv := make([]*needle.EngramKV, len(m.engrams))
	for s, e := range m.engrams {
		kv[s], err = e.Forward(tokens)
		if err != nil {
			return nil, fmt.Errorf("text: %w", err)
		}
	}

	rope := needle.NewRope(m.cfg.QKDim, t, m.cfg.Rope)
	stream := needle.ExpandLanes(out.Embed, m.cfg.Lanes, d)

	for l, b := range m.blocks {
		var ekv *needle.EngramKV
		if s, ok := m.site[l]; ok {
			ekv = kv[s]
		}

		block := func(u []float32) ([]float32, error) { return b.Forward(u, t, needle.MaskCausal, rope, ekv) }

		stream, err = m.mhc[l].Forward(stream, t, block)
		if err != nil {
			return nil, fmt.Errorf("text: layer %d: %w", l, err)
		}

		if hidden {
			out.Hidden = append(out.Hidden, needle.CollapseLanes(stream, m.cfg.Lanes, d))
		}
	}

	out.Final, out.Logits = m.output(stream, t)

	return out, nil
}

// checkLen rejects sequences past MaxSeqLen with ErrTooLong.
func (m *Model) checkLen(t int) error {
	if m.cfg.MaxSeqLen > 0 && t > m.cfg.MaxSeqLen {
		return fmt.Errorf("%w: %d positions, max_seq_len %d", ErrTooLong, t, m.cfg.MaxSeqLen)
	}

	return nil
}

// embed returns the scaled input embedding of tokens, t rows of Dim
// values; off is the first token's position, for the error message.
func (m *Model) embed(tokens []int, off int) ([]float32, error) {
	d := m.cfg.Dim
	x := make([]float32, len(tokens)*d)

	for i, tok := range tokens {
		if tok < 0 || tok >= m.cfg.Vocab {
			return nil, fmt.Errorf("text: token %d at %d outside the vocabulary of %d", tok, off+i, m.cfg.Vocab)
		}

		row := x[i*d : (i+1)*d]
		m.emb.Row(row, tok)

		for k := range row {
			row[k] *= m.scale
		}
	}

	return x, nil
}

// output is the stack's end on t rows of the stream: the lane mean after
// final_norm and the tied head's logits.
func (m *Model) output(stream []float32, t int) (final, logits []float32) {
	final = needle.CollapseLanes(stream, m.cfg.Lanes, m.cfg.Dim)
	m.final.Apply(final, final)

	logits = make([]float32, t*m.cfg.OutVocab)
	m.head.Apply(logits, final, t)

	return final, logits
}
