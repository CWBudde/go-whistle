// Package decoder is Whistle's text decoder: the token embedding scaled by
// √512, eight Needle 3 blocks (engram memory at layers 3 and 7) inside mHC,
// each with a gated cross-attention over the encoder output, the final norm
// and the head tied to the embedding.
//
// Upstream's Block has no cross-attention and the engine that runs Whistle
// is closed, so the cross-attention, the prompt and the head are reverse
// engineered: every open choice is an axis of Hypothesis, and the probe
// harness (probe_test.go, build tag oracle) scores them by teacher forcing
// the engine's transcripts (docs/FINDINGS.md F-021 to F-024).
package decoder

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Decoder runs one Hypothesis on one weight source. It is immutable after
// Load and safe for concurrent use; the decoding state lives in State.
type Decoder struct {
	h      Hypothesis
	tok    *tokenizer.Tokenizer
	dim    int
	vocab  int
	lanes  int
	qk     int
	emb    []float32 // [vocab, dim], also the head
	scale  float32   // input scale
	head   float32   // logit scale
	blocks []*needle.Block
	cross  []*cross
	mhc    []*needle.MHC
	run    []int         // the layers that run, ascending (WithDepth)
	runMHC []*needle.MHC // their mHC, with the lane of their position
	sites  map[int]int   // layer → engram site
	engram []*needle.Engram
	final  needle.ZCRMSNorm
	rope   *needle.Rope // self-attention and xrope tables
	prompt []int
	langs  []Language
	maxTok int
}

// Language is one language token.
type Language struct {
	Code string // "en", "de", …
	ID   int    // the token id of "<|code|>"
}

// languageCodes are whistle.py LANGUAGES, the order of the language tokens
// 8192–8198 (docs/FINDINGS.md F-005).
var languageCodes = []string{"en", "de", "fr", "es", "it", "nl", "pl"}

// ropeLen covers the positions of the longest decode (prompt, language
// token, 320 tokens) and the 375 frames of 30 s.
const ropeLen = 512

// Load builds the decoder for h from src. tok is the vocabulary; nil reads
// it from src, which only the .cact format holds.
func Load(src weights.Source, tok *tokenizer.Tokenizer, h Hypothesis) (*Decoder, error) {
	err := h.Validate()
	if err != nil {
		return nil, err
	}

	if tok == nil {
		tok, err = LoadTokenizer(src)
		if err != nil {
			return nil, err
		}
	}

	cfg := src.Config()
	d := &Decoder{h: h, tok: tok, maxTok: cfg.MaxTokens, sites: map[int]int{}}

	err = d.loadEmbedding(src)
	if err != nil {
		return nil, err
	}

	err = d.loadStack(src, cfg)
	if err != nil {
		return nil, err
	}

	d.final, err = needle.LoadZCRMSNorm(src, "final_norm")
	if err != nil {
		return nil, err
	}

	err = d.setTokens()
	if err != nil {
		return nil, err
	}

	d.rope = needle.NewRope(d.qk, ropeLen, cfg.Rope)

	return d, nil
}

// LoadTokenizer parses the vocabulary src holds (the .cact RAW record).
func LoadTokenizer(src weights.Source) (*tokenizer.Tokenizer, error) {
	blob, err := src.Raw("tokenizer")
	if err != nil {
		return nil, fmt.Errorf("decoder: tokenizer: %w", err)
	}

	return tokenizer.Parse(blob)
}

// Hypothesis is the hypothesis the decoder runs.
func (d *Decoder) Hypothesis() Hypothesis { return d.h }

// Tokenizer is the decoder's vocabulary.
func (d *Decoder) Tokenizer() *tokenizer.Tokenizer { return d.tok }

// Vocab is the number of logits per position.
func (d *Decoder) Vocab() int { return d.vocab }

// Prompt is the token sequence before the language token.
func (d *Decoder) Prompt() []int { return append([]int(nil), d.prompt...) }

// Languages lists the language tokens in id order.
func (d *Decoder) Languages() []Language { return append([]Language(nil), d.langs...) }

// MaxTokens is the decode limit, the config's max_tokens (320).
func (d *Decoder) MaxTokens() int { return d.maxTok }

// Memory is the encoder output projected to every layer's cross-attention
// keys and values, once per clip; every State on it shares it.
type Memory struct {
	layers []memory
	frames int
}

// Frames is the number of encoder frames.
func (m *Memory) Frames() int { return m.frames }

// Memory projects enc, frames rows of the model width (the encoder output
// the Memory axis names), for every layer.
func (d *Decoder) Memory(enc []float32, frames int) (*Memory, error) {
	if frames <= 0 || len(enc) != frames*d.dim {
		return nil, fmt.Errorf("decoder: encoder output has %d values, want %d rows of %d", len(enc), frames, d.dim)
	}

	if frames > d.rope.Len() && d.h.CrossRope == CrossRopeBoth {
		return nil, fmt.Errorf("decoder: %d frames exceed the RoPE tables' %d", frames, d.rope.Len())
	}

	m := &Memory{frames: frames, layers: make([]memory, len(d.cross))}
	for l, c := range d.cross {
		m.layers[l] = c.project(enc, frames, d.crossRope())
	}

	return m, nil
}

// State is one sequence's decoding state: the token history, the
// self-attention KV caches with their tap history and the engram caches
// (docs/FINDINGS.md F-022).
type State struct {
	mem     *Memory
	kv      []needle.KVCache
	engram  []needle.EngramCache
	tokens  []int
	capture Capture
}

// Attention is one decoder layer's cross-attention for the tokens of one
// Extend call (PLAN.md 7.1, docs/FINDINGS.md F-033): row i is the query of
// the token fed at position Pos+i, the one that predicts the next token.
type Attention struct {
	Layer  int // the decoder layer, 0 … 7
	Pos    int // the position of the first fed token
	Tokens int // the fed tokens, the rows
	Heads  int
	Frames int
	// Probs is softmax(q̂k̂ᵀ/√48) per token and head over the encoder
	// frames, [Tokens, Heads, Frames].
	Probs []float32
	// Gate is the output gate σ(gate_proj(x)) per token and value channel,
	// [Tokens, Heads·64]; nil when the Hypothesis has no gate.
	Gate []float32
}

// Capture receives every Attention of the layers that run, in layer order,
// once per Extend call. The slices are the receiver's to keep.
type Capture func(Attention)

// SetCapture makes the state report its cross-attention to fn on every
// later Extend; nil turns it off. Off, Extend pays one nil check per
// layer. A Fork does not inherit it.
func (s *State) SetCapture(fn Capture) { s.capture = fn }

// NewState starts a sequence against the memory m.
func (d *Decoder) NewState(m *Memory) *State {
	return &State{mem: m, kv: make([]needle.KVCache, len(d.blocks)), engram: make([]needle.EngramCache, len(d.engram))}
}

// Len is the number of tokens fed so far.
func (s *State) Len() int { return len(s.tokens) }

// Fork returns a state with s's history that extends independently of s,
// as one beam's continuation (docs/FINDINGS.md F-026). The memory is
// shared, never copied, and so are the cached keys and values of the
// prefix until either state appends to them (needle.KVCache.Fork).
func (s *State) Fork() *State {
	f := &State{
		mem:    s.mem,
		kv:     make([]needle.KVCache, len(s.kv)),
		engram: make([]needle.EngramCache, len(s.engram)),
		tokens: slices.Clip(s.tokens),
	}

	for i := range s.kv {
		f.kv[i] = s.kv[i].Fork()
	}

	for i := range s.engram {
		f.engram[i] = s.engram[i].Fork()
	}

	return f
}

// Memory is the encoder memory the state attends to.
func (s *State) Memory() *Memory { return s.mem }

// Tokens is the token history.
func (s *State) Tokens() []int { return append([]int(nil), s.tokens...) }

// Extend feeds tokens to the sequence and returns their logits, one row of
// Vocab values per token: row i scores the token after tokens[i]. Feeding
// a sequence in one call or in pieces gives the same logits up to float32
// summation order (docs/FINDINGS.md F-022).
func (d *Decoder) Extend(s *State, tokens []int) ([]float32, error) {
	t, off := len(tokens), len(s.tokens)
	if t == 0 {
		return nil, nil
	}

	if off+t > d.rope.Len() {
		return nil, fmt.Errorf("decoder: %d positions exceed the RoPE tables' %d", off+t, d.rope.Len())
	}

	x := make([]float32, t*d.dim)

	for i, tok := range tokens {
		if tok < 0 || tok >= d.vocab {
			return nil, fmt.Errorf("decoder: token %d out of range [0, %d)", tok, d.vocab)
		}

		row := x[i*d.dim : (i+1)*d.dim]
		for c, v := range d.emb[tok*d.dim : (tok+1)*d.dim] {
			row[c] = v * d.scale
		}
	}

	ekv := make([]*needle.EngramKV, len(d.engram))

	for i, e := range d.engram {
		var err error

		ekv[i], err = e.Extend(tokens, &s.engram[i])
		if err != nil {
			return nil, err
		}
	}

	stream := needle.ExpandLanes(x, d.lanes, d.dim)

	for i, l := range d.run {
		var err error

		stream, err = d.layer(l, d.runMHC[i], s, stream, t, off, ekv)
		if err != nil {
			return nil, err
		}
	}

	s.tokens = append(s.tokens, tokens...)

	return d.logits(needle.CollapseLanes(stream, d.lanes, d.dim), t), nil
}

func (d *Decoder) loadEmbedding(src weights.Source) error {
	emb, shape, err := src.Tensor("embedding")
	if err != nil {
		return fmt.Errorf("decoder: embedding: %w", err)
	}

	if len(shape) != 2 || shape[0] != d.tok.Len() {
		return fmt.Errorf("decoder: embedding has shape %v for %d pieces", shape, d.tok.Len())
	}

	d.emb, d.vocab, d.dim = emb, shape[0], shape[1]
	d.scale, d.head = 1, 1

	if d.h.InputScale == ScaleSqrtDim {
		d.scale = float32(math.Sqrt(float64(d.dim)))
	}

	if d.h.Head == HeadScaled {
		d.head = 1 / float32(math.Sqrt(float64(d.dim)))
	}

	return nil
}

// loadStack loads the blocks, their cross-attentions and mHC layers, and
// the engram sites.
func (d *Decoder) loadStack(src weights.Source, cfg weights.Config) error {
	for l := range cfg.DecLayers {
		b, err := needle.LoadBlock(src, fmt.Sprintf("layer%02d.", l))
		if err != nil {
			return err
		}

		c, err := loadCross(src, fmt.Sprintf("cross%02d.", l), d.h)
		if err != nil {
			return err
		}

		m, err := needle.LoadMHC(src, "", l)
		if err != nil {
			return err
		}

		if b.Dim() != d.dim || c.dim != d.dim || m.Dim() != d.dim {
			return fmt.Errorf("decoder: layer %d widths %d/%d/%d, embedding %d", l, b.Dim(), c.dim, m.Dim(), d.dim)
		}

		d.blocks, d.cross, d.mhc = append(d.blocks, b), append(d.cross, c), append(d.mhc, m)
		d.run, d.runMHC = append(d.run, l), append(d.runMHC, m)
		d.lanes, d.qk = m.Lanes(), c.qk
	}

	if len(d.blocks) == 0 {
		return fmt.Errorf("decoder: config has %d decoder layers", cfg.DecLayers)
	}

	if d.h.Engram == EngramOff {
		return nil
	}

	for s, layer := range cfg.Sites {
		e, err := needle.LoadEngram(src, fmt.Sprintf("engram%d.", s), cfg.Orders, cfg.Tables)
		if err != nil {
			return err
		}

		if layer < 0 || layer >= len(d.blocks) || e.Dim() != d.dim {
			return fmt.Errorf("decoder: engram site %d at layer %d, width %d", s, layer, e.Dim())
		}

		d.sites[layer] = len(d.engram)
		d.engram = append(d.engram, e)
	}

	return nil
}

// setTokens resolves the prompt and the language tokens by their pieces.
func (d *Decoder) setTokens() error {
	speech, ok1 := d.tok.PieceID("<speech>")
	speechEnd, ok2 := d.tok.PieceID("</speech>")

	if !ok1 || !ok2 {
		return errors.New("decoder: the vocabulary has no <speech> or </speech>")
	}

	d.prompt = d.h.Prompt.tokens(d.tok.BOSID(), d.tok.EOSID(), d.tok.PadID(), speech, speechEnd)

	for _, code := range languageCodes {
		id, ok := d.tok.PieceID("<|" + code + "|>")
		if !ok {
			return fmt.Errorf("decoder: the vocabulary has no <|%s|>", code)
		}

		d.langs = append(d.langs, Language{Code: code, ID: id})
	}

	return nil
}

// tokens is the prompt's token ids for the tokenizer's specials
// (docs/FINDINGS.md F-023).
func (p Prompt) tokens(bos, eos, pad, speech, speechEnd int) []int {
	switch p {
	case PromptBOS:
		return []int{bos}
	case PromptBOSSpeech:
		return []int{bos, speech}
	case PromptSpeech:
		return []int{speech}
	case PromptSpeechEnd:
		return []int{speech, speechEnd}
	case PromptBOSSpeechEnd:
		return []int{bos, speech, speechEnd}
	case PromptEOS:
		return []int{eos}
	default:
		return []int{pad}
	}
}

func (d *Decoder) crossRope() *needle.Rope {
	if d.h.CrossRope == CrossRopeBoth {
		return d.rope
	}

	return nil
}

// layer advances the stream through layer l's block and cross-attention
// inside the mHC mhc.
func (d *Decoder) layer(l int, mhc *needle.MHC, s *State, stream []float32, t, off int,
	ekv []*needle.EngramKV,
) ([]float32, error) {
	var site *needle.EngramKV
	if i, ok := d.sites[l]; ok {
		site = ekv[i]
	}

	c, mem := d.cross[l], s.mem.layers[l]

	var keep *Attention
	if s.capture != nil {
		keep = &Attention{Layer: l, Pos: off, Tokens: t, Heads: c.heads, Frames: mem.frames}
	}

	hook := func(h []float32, t int) error {
		for i, v := range c.branch(h, t, off, mem, d.crossRope(), keep) {
			h[i] += v
		}

		return nil
	}

	var hooks needle.Hooks

	switch d.h.CrossPos {
	case CrossBeforeAttn:
		hooks.BeforeAttn = hook
	case CrossAfterMLP:
		hooks.AfterMLP = hook
	case CrossAfterAttn:
		hooks.AfterAttn = hook
	}

	out, err := mhc.Forward(stream, t, func(u []float32) ([]float32, error) {
		return d.blocks[l].Extend(u, t, &s.kv[l], d.rope, site, hooks)
	})
	if err != nil {
		return nil, err
	}

	if d.h.CrossPos == CrossOutside {
		br := c.branch(needle.CollapseLanes(out, d.lanes, d.dim), t, off, mem, d.crossRope(), keep)

		for i := range t {
			for k := range d.lanes {
				row := out[(i*d.lanes+k)*d.dim : (i*d.lanes+k+1)*d.dim]
				for j, v := range br[i*d.dim : (i+1)*d.dim] {
					row[j] += v
				}
			}
		}
	}

	if keep != nil {
		s.capture(*keep)
	}

	return out, nil
}

// logits is the head on the lane mean x, t rows: final_norm, then x · Eᵀ.
func (d *Decoder) logits(x []float32, t int) []float32 {
	if d.h.FinalNorm == FinalNormOn {
		d.final.Apply(x, x)
	}

	out := make([]float32, t*d.vocab)
	tensor.MatMulTransB(out, d.vocab, x, d.emb, t, d.vocab, d.dim)

	if d.head != 1 {
		for i := range out {
			out[i] *= d.head
		}
	}

	return out
}
