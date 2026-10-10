package decoder

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ErrHypothesis reports an unset or unknown value in a Hypothesis.
var ErrHypothesis = errors.New("decoder: invalid hypothesis")

// CrossPos is where the gated cross-attention sits in a decoder layer.
type CrossPos int

const (
	// CrossAfterAttn is between the self-attention and the MLP, inside the
	// block function mHC wraps (the Transformer decoder's place).
	CrossAfterAttn CrossPos = iota + 1
	// CrossBeforeAttn is after the engram gate, before the self-attention.
	CrossBeforeAttn
	// CrossAfterMLP is after the MLP's residual add, still inside the block
	// function.
	CrossAfterMLP
	// CrossOutside is outside the block function: after the layer's mHC
	// advance, the cross-attention of the lane mean is added to every lane.
	CrossOutside
)

// QKNorm is the cross-attention's per-head q/k norm.
type QKNorm int

const (
	// QKNormZC is ZCRMSNorm, gain 1 + scale, as the self-attention's.
	QKNormZC QKNorm = iota + 1
	// QKNormPlain is an RMS norm with gain scale (not zero-centred).
	QKNormPlain
	// QKNormUnit is an RMS norm without a gain: q_norm and k_norm unused.
	QKNormUnit
	// QKNormNone leaves q and k unnormalised.
	QKNormNone
)

// Gate is the cross-attention's output gate before out_proj.
type Gate int

const (
	// GateSigmoid is σ(gate_proj(x)) of the attention input x =
	// cross_norm(h), as MultiHeadAttention's.
	GateSigmoid Gate = iota + 1
	// GateSigmoidResidual is σ(gate_proj(h)) of the un-normed residual h.
	GateSigmoidResidual
	// GateSiLU is silu(gate_proj(x)).
	GateSiLU
	// GateNone skips gate_proj.
	GateNone
)

// Residual is how the cross-attention's output joins the residual.
type Residual int

const (
	// ResSigmoid is h + σ(cross_gate) · post_cross_norm(c), as the
	// self-attention's σ(attn_gate) residual.
	ResSigmoid Residual = iota + 1
	// ResTanh is h + tanh(cross_gate) · post_cross_norm(c) (Flamingo).
	ResTanh
	// ResRaw is h + cross_gate · post_cross_norm(c).
	ResRaw
	// ResNoPostNorm is h + σ(cross_gate) · c, post_cross_norm unused.
	ResNoPostNorm
)

// InputScale multiplies the token embeddings.
type InputScale int

const (
	// ScaleSqrtDim is √512, SimpleAttentionNetwork's embed_scale.
	ScaleSqrtDim InputScale = iota + 1
	// ScaleOne leaves the embeddings unscaled.
	ScaleOne
)

// Head is the output head over the vocabulary.
type Head int

const (
	// HeadTied is x · Eᵀ, SimpleAttentionNetwork's tied head.
	HeadTied Head = iota + 1
	// HeadScaled is x · Eᵀ / √512.
	HeadScaled
)

// FinalNorm is what the head reads.
type FinalNorm int

const (
	// FinalNormOn is stack/final_norm of the lane mean, as Stack.
	FinalNormOn FinalNorm = iota + 1
	// FinalNormOff is the lane mean.
	FinalNormOff
)

// Prompt is the token sequence before the language token.
type Prompt int

const (
	// PromptBOS is [bos].
	PromptBOS Prompt = iota + 1
	// PromptBOSSpeech is [bos, <speech>].
	PromptBOSSpeech
	// PromptSpeech is [<speech>].
	PromptSpeech
	// PromptSpeechEnd is [<speech>, </speech>].
	PromptSpeechEnd
	// PromptBOSSpeechEnd is [bos, <speech>, </speech>].
	PromptBOSSpeechEnd
	// PromptEOS is [eos], a T5-style start token.
	PromptEOS
	// PromptPad is [pad].
	PromptPad
)

// MemoryTap is which encoder output the cross-attention reads.
type MemoryTap int

const (
	// MemFinalNorm is enc.final_norm of the lane mean, needle_embed
	// (encoder.OutFinalNorm).
	MemFinalNorm MemoryTap = iota + 1
	// MemLaneMean is the encoder's lane mean before enc.final_norm
	// (encoder.OutLaneMean).
	MemLaneMean
)

// CrossRope is the cross-attention's rotary embedding.
type CrossRope int

const (
	// CrossRopeNone rotates neither q nor k.
	CrossRopeNone CrossRope = iota + 1
	// CrossRopeBoth rotates q at the token's position and k at the frame's,
	// θ = 1e5.
	CrossRopeBoth
)

// Lead is how the transcript's first word is tokenised.
type Lead int

const (
	// LeadNone encodes the text as it stands ("Jeder …"); the tokenizer has
	// no dummy prefix.
	LeadNone Lead = iota + 1
	// LeadSpace encodes " " + text ("▁Jeder …") and drops the leading space
	// when decoding.
	LeadSpace
)

// EngramUse switches the engram memory at the sites.
type EngramUse int

const (
	// EngramOn gates the engram values in at layers 3 and 7, as upstream.
	EngramOn EngramUse = iota + 1
	// EngramOff ignores the engram tensors.
	EngramOff
)

// Hypothesis picks one value per open axis of the decoder. Every value of
// every axis is a typed constant starting at 1, so the zero Hypothesis is
// invalid (docs/FINDINGS.md F-021).
type Hypothesis struct {
	CrossPos   CrossPos
	QKNorm     QKNorm
	Gate       Gate
	Residual   Residual
	InputScale InputScale
	Head       Head
	FinalNorm  FinalNorm
	Prompt     Prompt
	Memory     MemoryTap
	CrossRope  CrossRope
	Lead       Lead
	Engram     EngramUse
}

// Best is the leading hypothesis of the teacher-forced probe search
// (docs/FINDINGS.md F-021, F-023).
func Best() Hypothesis {
	return Hypothesis{
		CrossPos: CrossAfterAttn, QKNorm: QKNormZC, Gate: GateSigmoid, Residual: ResSigmoid,
		InputScale: ScaleSqrtDim, Head: HeadTied, FinalNorm: FinalNormOn, Prompt: PromptBOS,
		Memory: MemFinalNorm, CrossRope: CrossRopeNone, Lead: LeadNone, Engram: EngramOn,
	}
}

// Axis describes one field of Hypothesis: its key and its value names,
// Names[v−1] for value v.
type Axis struct {
	Key   string
	Names []string
	field func(*Hypothesis) *int
}

// Get returns the axis' value in h.
func (a Axis) Get(h Hypothesis) int { return *a.field(&h) }

// Set sets the axis' value in h.
func (a Axis) Set(h *Hypothesis, v int) { *a.field(h) = v }

const noneName = "none"

var axes = []Axis{
	{"cross", []string{"after-attn", "before-attn", "after-mlp", "outside"}, func(h *Hypothesis) *int { return (*int)(&h.CrossPos) }},
	{"qknorm", []string{"zc", "plain", "unit", noneName}, func(h *Hypothesis) *int { return (*int)(&h.QKNorm) }},
	{"gate", []string{"sigmoid", "sigmoid-residual", "silu", noneName}, func(h *Hypothesis) *int { return (*int)(&h.Gate) }},
	{"residual", []string{"sigmoid", "tanh", "raw", "no-post-norm"}, func(h *Hypothesis) *int { return (*int)(&h.Residual) }},
	{"scale", []string{"sqrt-dim", "1"}, func(h *Hypothesis) *int { return (*int)(&h.InputScale) }},
	{"head", []string{"tied", "scaled"}, func(h *Hypothesis) *int { return (*int)(&h.Head) }},
	{"final", []string{"norm", noneName}, func(h *Hypothesis) *int { return (*int)(&h.FinalNorm) }},
	{"prompt", []string{"bos", "bos-speech", "speech", "speech-end", "bos-speech-end", "eos", "pad"}, func(h *Hypothesis) *int { return (*int)(&h.Prompt) }},
	{"memory", []string{"final-norm", "lane-mean"}, func(h *Hypothesis) *int { return (*int)(&h.Memory) }},
	{"xrope", []string{noneName, "both"}, func(h *Hypothesis) *int { return (*int)(&h.CrossRope) }},
	{"lead", []string{noneName, "space"}, func(h *Hypothesis) *int { return (*int)(&h.Lead) }},
	{"engram", []string{"on", "off"}, func(h *Hypothesis) *int { return (*int)(&h.Engram) }},
}

// Axes lists the axes of Hypothesis in String's order.
func Axes() []Axis { return axes }

// Validate reports an unset or unknown value on any axis.
func (h Hypothesis) Validate() error {
	for _, a := range axes {
		v := a.Get(h)
		if v < 1 || v > len(a.Names) {
			return fmt.Errorf("%w: %s = %d", ErrHypothesis, a.Key, v)
		}
	}

	return nil
}

// String is a stable key, key=value for each axis. ParseHypothesis reads
// it back.
func (h Hypothesis) String() string {
	parts := make([]string, len(axes))

	for i, a := range axes {
		v := a.Get(h)

		name := strconv.Itoa(v)
		if v >= 1 && v <= len(a.Names) {
			name = a.Names[v-1]
		}

		parts[i] = a.Key + "=" + name
	}

	return strings.Join(parts, " ")
}

// ParseHypothesis reads a key written by Hypothesis.String.
func ParseHypothesis(key string) (Hypothesis, error) {
	fields := strings.Fields(key)
	if len(fields) != len(axes) {
		return Hypothesis{}, fmt.Errorf("%w: %q has %d fields, want %d", ErrHypothesis, key, len(fields), len(axes))
	}

	var h Hypothesis

	for i, f := range fields {
		a := axes[i]

		name, ok := strings.CutPrefix(f, a.Key+"=")
		if !ok {
			return Hypothesis{}, fmt.Errorf("%w: field %q is not %s=…", ErrHypothesis, f, a.Key)
		}

		v := 0

		for j, n := range a.Names {
			if n == name {
				v = j + 1
			}
		}

		if v == 0 {
			return Hypothesis{}, fmt.Errorf("%w: unknown %s %q", ErrHypothesis, a.Key, name)
		}

		a.Set(&h, v)
	}

	return h, nil
}
