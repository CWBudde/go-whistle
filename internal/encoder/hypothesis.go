package encoder

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/cwbudde/go-whistle/internal/frontend"
)

// ErrHypothesis reports an unset or unknown value in a Hypothesis.
var ErrHypothesis = errors.New("encoder: invalid hypothesis")

// ErrRuledOut reports a hypothesis the weights' shapes exclude before any
// score: it cannot run on the shipped tensors (docs/FINDINGS.md F-014).
var ErrRuledOut = errors.New("encoder: hypothesis ruled out by shapes")

// StemStride is where the conv stem's stride-2 halvings sit.
type StemStride int

const (
	// StrideAll puts stride (2, 2) on the full conv and both depthwise convs:
	// three halvings, 80 mels → 10, 3000 frames → 375.
	StrideAll StemStride = iota + 1
	// StrideDepthwise strides only the two depthwise convs: two halvings,
	// which leave 20 mels (2560 features, not stem.out's 1280).
	StrideDepthwise
)

// StemPad is the stem convs' zero padding.
type StemPad int

const (
	// PadSame is lax "SAME": ceil(n/2) outputs, 0 before and 1 after for an
	// even length, 1 and 1 for an odd one (ops.SamePadding).
	PadSame StemPad = iota + 1
	// PadSymmetric is 1 before and 1 after (torch padding=1): ceil(n/2)
	// outputs, but on an even length each window starts one step earlier
	// than SAME's.
	PadSymmetric
	// PadValid is no padding: floor((n − 3)/2) + 1 outputs.
	PadValid
)

// KernelLayout is which stored kernel axis runs over time.
type KernelLayout int

const (
	// KernelTimeMel reads stem/w [3, 3, 1, 128] as [time, mel, in, out]:
	// the input is (batch, time, mel, 1), Flax NHWC with H = time.
	KernelTimeMel KernelLayout = iota + 1
	// KernelMelTime reads it as [mel, time, in, out] (H = mel).
	KernelMelTime
)

// Flatten is the order of the stem output's (mel, channel) features in the
// 1280-wide row that stem/out multiplies.
type Flatten int

const (
	// FlattenMelChannel is reshape(T, 10·128) of NHWC: channel fastest.
	FlattenMelChannel Flatten = iota + 1
	// FlattenChannelMel is the NCHW order: mel fastest.
	FlattenChannelMel
)

// Act is an activation function.
type Act int

const (
	// ActNone is the identity.
	ActNone Act = iota + 1
	// ActSiLU is x·σ(x) (jax.nn.silu).
	ActSiLU
	// ActGELUTanh is the tanh approximation, jax.nn.gelu's default.
	ActGELUTanh
	// ActGELUErf is the exact x·Φ(x).
	ActGELUErf
	// ActReLU is max(x, 0).
	ActReLU
)

// InputScale multiplies the stem output before the encoder stack.
type InputScale int

const (
	// ScaleOne leaves it unchanged.
	ScaleOne InputScale = iota + 1
	// ScaleSqrtDim multiplies by √512, the decoder's embedding scale.
	ScaleSqrtDim
)

// PE is where the global pe_gate (σ ≈ 0.0022) applies a position signal.
// Each sinusoid placement adds g times the transformer sinusoid of its row
// width w (frequency 10000^(−2i/w) for pair i), laid out [sin | cos] or
// interleaved, with g = σ(pe_gate). The probes rule every placement out
// (docs/FINDINGS.md F-019).
type PE int

const (
	// PENone ignores pe_gate.
	PENone PE = iota + 1
	// PEStemConcat adds σ(pe_gate)·[sin | cos] (first half sines, second half
	// cosines, frequencies 10000^(−2i/512)) to the stem output.
	PEStemConcat
	// PEStemInterleaved adds σ(pe_gate) times the interleaved sinusoid (sin
	// at even, cos at odd channels, as in "Attention is all you need").
	PEStemInterleaved
	// PERopeMix rotates q and k by (1 − g)·x + g·RoPE(x), g = σ(pe_gate),
	// with the Rope axis' θ (1e5 when Rope is RopeNone).
	PERopeMix
	// PEMelConcat adds the [sin | cos] sinusoid to the stem input, the
	// normalised log-mel (width 80).
	PEMelConcat
	// PEMelInterleaved adds the interleaved sinusoid to the stem input.
	PEMelInterleaved
	// PEFlatConcat adds the [sin | cos] sinusoid before stem/out, to the
	// flattened 1280-wide stem features.
	PEFlatConcat
	// PEFlatInterleaved adds the interleaved sinusoid before stem/out.
	PEFlatInterleaved
	// PEBlockConcat adds the [sin | cos] sinusoid to every block's input u
	// (inside mHC, the residual taken around u + PE).
	PEBlockConcat
	// PEBlockInterleaved adds the interleaved sinusoid to every block's
	// input.
	PEBlockInterleaved
	// PEStemUngated adds the [sin | cos] sinusoid to the stem output with
	// gain 1 instead of σ(pe_gate): a control for the probes' sensitivity.
	PEStemUngated
)

// interleaved reports whether a sinusoid placement uses the interleaved
// layout.
func (p PE) interleaved() bool {
	return p == PEStemInterleaved || p == PEMelInterleaved || p == PEFlatInterleaved || p == PEBlockInterleaved
}

// Rope is the encoder attention's rotary embedding.
type Rope int

const (
	// RopeNone uses no rotation.
	RopeNone Rope = iota + 1
	// Rope1e5 is the decoder's half-split RoPE with θ = 1e5 (config "rope").
	Rope1e5
	// Rope1e4 is θ = 1e4, precompute_rope_freqs' default.
	Rope1e4
)

// FFNScale is the macaron weight on both Hadamard MLP residuals.
type FFNScale int

const (
	// FFNFull adds each MLP output unscaled, as the decoder Block does.
	FFNFull FFNScale = iota + 1
	// FFNHalf adds ½ of each (Conformer macaron).
	FFNHalf
)

// GLU is how pw1's 1024 channels become the depthwise conv's 512.
type GLU int

const (
	// GLUValueGate is a·σ(b) with a the first and b the second half
	// (torch.nn.GLU, flax glu).
	GLUValueGate GLU = iota + 1
	// GLUGateValue is σ(a)·b.
	GLUGateValue
	// GLUSwishValue is silu(a)·b (SwiGLU, gate first).
	GLUSwishValue
	// GLUValueSwish is a·silu(b).
	GLUValueSwish
	// GLUNone feeds pw1's 1024 channels straight on, which the 512-wide dw
	// kernel cannot take: ruled out by shapes.
	GLUNone
)

// ConvNormPos is where conv_out_norm sits in the conv module.
type ConvNormPos int

const (
	// NormBeforeAct is dw → conv_out_norm → act → pw2, Conformer's
	// BatchNorm position.
	NormBeforeAct ConvNormPos = iota + 1
	// NormAfterAct is dw → act → conv_out_norm → pw2.
	NormAfterAct
	// NormAfterPW2 is dw → act → pw2 → conv_out_norm, a post-norm like the
	// attention's post_attn_norm.
	NormAfterPW2
)

// DWPad is the block's depthwise conv padding over time (kernel 9).
type DWPad int

const (
	// DWCentred pads 4 frames on each side (non-causal).
	DWCentred DWPad = iota + 1
	// DWCausal pads 8 frames before (causal).
	DWCausal
)

// Order is the order of the four residual modules in a block.
type Order int

const (
	// OrderMacaron is FFN1, attention, conv, FFN2 (Conformer).
	OrderMacaron Order = iota + 1
	// OrderConvFirst is FFN1, conv, attention, FFN2.
	OrderConvFirst
)

// MHC is how the residual stream wraps the blocks.
type MHC int

const (
	// MHCLanes is Stack's mHC: expand to enc_lanes = 4 lanes, one MHC layer
	// per block with lane layer % 4, lane mean at the end (as the decoder).
	MHCLanes MHC = iota + 1
	// MHCNone is a plain stack x = block(x) that ignores encoder/mhc_*.
	MHCNone
)

// HadaPerms is which permutations the encoder's Hadamard MLPs use (U5).
type HadaPerms int

const (
	// PermsShared uses hada_p1/p2 = RandomState(11/13).permutation(512), as
	// the decoder does.
	PermsShared HadaPerms = iota + 1
	// PermsSplit uses _hada_perms(split=True): each half permuted on its own
	// (seeds s and s + 977).
	PermsSplit
	// PermsIdentity uses no permutation.
	PermsIdentity
)

// Output is which tensor needle_embed returns.
type Output int

const (
	// OutFinalNorm is encoder/final_norm of the lane mean (Stack's output).
	OutFinalNorm Output = iota + 1
	// OutLaneMean is the lane mean before final_norm.
	OutLaneMean
)

// Hypothesis picks one value per open axis of the encoder, front end
// included. Every value of every axis is a typed constant starting at 1, so
// the zero Hypothesis is invalid (docs/FINDINGS.md F-014).
type Hypothesis struct {
	Frontend frontend.Options

	StemStride  StemStride
	StemPad     StemPad
	StemKernel  KernelLayout
	StemFlatten Flatten
	StemConvAct Act // after the full conv stem/w
	StemDWAct   Act // after each depthwise conv
	StemPWAct   Act // after each pointwise conv
	InputScale  InputScale

	PE   PE
	Rope Rope

	FFNScale  FFNScale
	GLU       GLU
	ConvAct   Act // after the block's depthwise conv
	ConvNorm  ConvNormPos
	DWPad     DWPad
	Order     Order
	MHC       MHC
	HadaPerms HadaPerms
	Output    Output
}

// Best is the leading hypothesis, mean per-frame cosine 0.9998 against
// needle_embed over the 32 corpus clips with the .cact weights, every clip
// ≥ 0.9990 (docs/FINDINGS.md F-014 to F-020). pe_gate is unused: the
// silence probes match without any position term beyond RoPE (F-019). The
// log is the soft knee 76 dB below the clip's peak, zero centring is
// decided by the DC probes, and the window still ties (F-020).
func Best() Hypothesis {
	return Hypothesis{
		Frontend: frontend.Options{
			Window: frontend.HannSymmetric, Padding: frontend.PadZero, Spectrum: frontend.Power,
			Log: frontend.LogSoft76, Norm: frontend.NormChannel, Band: frontend.BandFull,
			Preemphasis: frontend.PreemphNone, Dither: frontend.DitherNone,
		},
		StemStride: StrideAll, StemPad: PadSymmetric, StemKernel: KernelTimeMel, StemFlatten: FlattenChannelMel,
		StemConvAct: ActSiLU, StemDWAct: ActNone, StemPWAct: ActSiLU, InputScale: ScaleOne,
		PE: PENone, Rope: Rope1e5,
		FFNScale: FFNHalf, GLU: GLUValueGate, ConvAct: ActSiLU, ConvNorm: NormBeforeAct, DWPad: DWCentred,
		Order: OrderMacaron, MHC: MHCLanes, HadaPerms: PermsShared, Output: OutFinalNorm,
	}
}

// Axis describes one encoder field of Hypothesis: its key and its value
// names, Names[v−1] for value v.
type Axis struct {
	Key   string
	Names []string
	field func(*Hypothesis) *int
}

// Get returns the axis' value in h.
func (a Axis) Get(h Hypothesis) int { return *a.field(&h) }

// Set sets the axis' value in h.
func (a Axis) Set(h *Hypothesis, v int) { *a.field(h) = v }

// stemAxes is the number of leading axes the stem output depends on.
const stemAxes = 8

// noneName is the "none" value of several axes.
const noneName = "none"

var actNames = []string{noneName, "silu", "gelu-tanh", "gelu-erf", "relu"}

var axes = []Axis{
	{"stride", []string{"all", "dw-only"}, func(h *Hypothesis) *int { return (*int)(&h.StemStride) }},
	{"stempad", []string{"same", "sym", "valid"}, func(h *Hypothesis) *int { return (*int)(&h.StemPad) }},
	{"kernel", []string{"time-mel", "mel-time"}, func(h *Hypothesis) *int { return (*int)(&h.StemKernel) }},
	{"flatten", []string{"mel-ch", "ch-mel"}, func(h *Hypothesis) *int { return (*int)(&h.StemFlatten) }},
	{"convact", actNames, func(h *Hypothesis) *int { return (*int)(&h.StemConvAct) }},
	{"dwact", actNames, func(h *Hypothesis) *int { return (*int)(&h.StemDWAct) }},
	{"pwact", actNames, func(h *Hypothesis) *int { return (*int)(&h.StemPWAct) }},
	{"scale", []string{"1", "sqrt-dim"}, func(h *Hypothesis) *int { return (*int)(&h.InputScale) }},
	{"pe", []string{
		noneName, "stem-concat", "stem-interleaved", "rope-mix", "mel-concat", "mel-interleaved",
		"flat-concat", "flat-interleaved", "block-concat", "block-interleaved", "stem-ungated",
	}, func(h *Hypothesis) *int { return (*int)(&h.PE) }},
	{"rope", []string{noneName, "1e5", "1e4"}, func(h *Hypothesis) *int { return (*int)(&h.Rope) }},
	{"ffn", []string{"1", "half"}, func(h *Hypothesis) *int { return (*int)(&h.FFNScale) }},
	{"glu", []string{"a*sig(b)", "sig(a)*b", "silu(a)*b", "a*silu(b)", noneName}, func(h *Hypothesis) *int { return (*int)(&h.GLU) }},
	{"blockact", actNames, func(h *Hypothesis) *int { return (*int)(&h.ConvAct) }},
	{"convnorm", []string{"before-act", "after-act", "after-pw2"}, func(h *Hypothesis) *int { return (*int)(&h.ConvNorm) }},
	{"dwpad", []string{"centred", "causal"}, func(h *Hypothesis) *int { return (*int)(&h.DWPad) }},
	{"order", []string{"macaron", "conv-first"}, func(h *Hypothesis) *int { return (*int)(&h.Order) }},
	{"mhc", []string{"lanes", noneName}, func(h *Hypothesis) *int { return (*int)(&h.MHC) }},
	{"perms", []string{"shared", "split", "identity"}, func(h *Hypothesis) *int { return (*int)(&h.HadaPerms) }},
	{"out", []string{"final-norm", "lane-mean"}, func(h *Hypothesis) *int { return (*int)(&h.Output) }},
}

// Axes lists the encoder axes of Hypothesis in String's order. The front
// end's axes are frontend.Options'.
func Axes() []Axis { return axes }

// Validate reports an unset or unknown value on any axis.
func (h Hypothesis) Validate() error {
	err := h.Frontend.Validate()
	if err != nil {
		return fmt.Errorf("%w: %w", ErrHypothesis, err)
	}

	for _, a := range axes {
		v := a.Get(h)
		if v < 1 || v > len(a.Names) {
			return fmt.Errorf("%w: %s = %d", ErrHypothesis, a.Key, v)
		}
	}

	return nil
}

// String is a stable key: the front end's key, then key=value for each
// encoder axis. ParseHypothesis reads it back.
func (h Hypothesis) String() string {
	return h.Frontend.String() + " " + h.EncoderString()
}

// EncoderString is String without the front end's part.
func (h Hypothesis) EncoderString() string {
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

// StemKey identifies what the stem output depends on, the front end and
// the stem axes, for caching it.
func (h Hypothesis) StemKey() string {
	return strings.Join(strings.Fields(h.String())[:frontendFields()+stemAxes], " ")
}

func frontendFields() int { return len(strings.Fields(frontend.NeMo().String())) }

// ParseHypothesis reads a key written by Hypothesis.String.
func ParseHypothesis(key string) (Hypothesis, error) {
	fields := strings.Fields(key)

	nf := frontendFields()
	if len(fields) != nf+len(axes) {
		return Hypothesis{}, fmt.Errorf("%w: %q has %d fields, want %d", ErrHypothesis, key, len(fields), nf+len(axes))
	}

	fe, err := frontend.ParseOptions(strings.Join(fields[:nf], " "))
	if err != nil {
		return Hypothesis{}, fmt.Errorf("%w: %w", ErrHypothesis, err)
	}

	h := Hypothesis{Frontend: fe}

	for i, f := range fields[nf:] {
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
