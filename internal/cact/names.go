package cact

import (
	"errors"
	"fmt"
	"slices"
)

// Transform says how a .cact tensor's values derive from its safetensors
// source (docs/FINDINGS.md F-006).
type Transform uint8

// Transforms. Each one applies after the layer slice of Entry.Layer.
const (
	// Derived tensors have no safetensors source: the exporter computed them
	// (Hadamard permutations, mel filterbank) or they carry file metadata
	// (audio manifest, tokenizer).
	Derived Transform = iota
	// Reshape keeps the values in their order: squeezed unit axes, a layer
	// scalar stored as (1,), stem kernels (3, 3, 1, C) stored as (9, C),
	// engram tables (4, slots, d) stored as (4*slots, d).
	Reshape
	// Transpose stores a Flax (in, out) kernel as an (out, in) matrix.
	Transpose
	// LaneMajor stores an mHC phi (layers, n*C, lanes) as
	// transpose(0, 2, 1).reshape(layers*lanes, n*C), as export.py does.
	LaneMajor
)

func (t Transform) String() string {
	switch t {
	case Derived:
		return "derived"
	case Reshape:
		return "reshape"
	case Transpose:
		return "transpose"
	case LaneMajor:
		return "lane-major"
	default:
		return fmt.Sprintf("transform %d", uint8(t))
	}
}

// Entry names one .cact tensor and says where its values come from.
type Entry struct {
	Name      string // upstream-style .cact name, e.g. "layer03.q_proj"
	Source    string // safetensors tensor name; "" when Derived
	Layer     int    // slice of Source's leading layer axis; -1 = the whole tensor
	Transform Transform
	Shape     []int // .cact shape; nil for the RAW tokenizer
}

// Whistle geometry (safetensors config @ b358ddad, docs/FINDINGS.md F-003).
const (
	whistleLayers  = 8 // dec_layers = enc_layers
	whistleDim     = 512
	whistleQ       = 384 // 8 heads x qk_dim 48
	whistleKV      = 96  // 2 kv heads x qk_dim 48
	whistleV       = 128 // 2 kv heads x v_dim 64
	whistleHead    = 48
	whistleLanes   = 4
	whistleVocab   = 8199
	whistleTaps    = 3
	whistleKernel  = 9
	whistleStem    = 128
	whistleSlots   = 18432
	whistleTables  = 4
	whistleSubDim  = 128
	whistleEngrams = 2
	whistleBins    = 257 // fft 512 / 2 + 1
	whistleMels    = 80
	whistleRank    = 8 // architecture.py HADA_COND_RANK
)

// part is one tensor of a repeated block: its name suffix, its path below
// the block's safetensors prefix, the transform and the .cact shape.
type part struct {
	suffix, path string
	transform    Transform
	shape        []int
}

func vec(suffix, path string, n int) part {
	return part{suffix, path, Reshape, []int{n}}
}

func mat(suffix, path string, rows, cols int) part {
	return part{suffix, path, Reshape, []int{rows, cols}}
}

// kernel is a Flax Dense kernel (dim, out), stored (out, dim).
func kernel(suffix, path string, out int) part {
	return part{suffix, path, Transpose, []int{out, whistleDim}}
}

// hadaParts is a HadamardMLP in export.py _tensors order: 13 tensors. path
// is the module prefix (".../hadamard_mlp/"), suffix the name prefix.
func hadaParts(suffix, path string) []part {
	ps := make([]part, 0, 13)
	for _, d := range []string{"d1", "d2", "b2", "d3", "d4"} {
		ps = append(ps, vec(suffix+d, path+d, whistleDim))
	}

	for _, w := range []string{"w1", "w2", "w3"} {
		ps = append(ps,
			mat(suffix+w+"a", path+w+"a", 16, 16),
			mat(suffix+w+"b", path+w+"b", 32, 32))
	}

	return append(ps,
		mat(suffix+"cond_v", path+"cond_v", whistleDim, whistleRank),
		mat(suffix+"cond_u", path+"cond_u", whistleRank, whistleDim))
}

// attnParts is a self-attention's Q/K/V projections and its gate and output
// projections.
func attnParts() (qkv, gateOut []part) {
	qkv = []part{
		kernel("q_proj", "self_attn/q_proj/kernel", whistleQ),
		kernel("k_proj", "self_attn/k_proj/kernel", whistleKV),
		kernel("v_proj", "self_attn/v_proj/kernel", whistleV),
	}
	gateOut = []part{
		kernel("gate_proj", "self_attn/gate_proj/kernel", whistleDim),
		kernel("out_proj", "self_attn/out_proj/kernel", whistleDim),
	}

	return qkv, gateOut
}

// decoderParts is one decoder block in export.py _tensors order, with
// qkv_conv_taps = 3: 27 tensors.
func decoderParts() []part {
	qkv, gateOut := attnParts()

	ps := []part{vec("norm_in", "ZCRMSNorm_0/scale", whistleDim)}
	ps = append(ps, qkv...)
	ps = append(ps,
		mat("q_taps", "self_attn/q_taps", whistleTaps, whistleQ),
		mat("k_taps", "self_attn/k_taps", whistleTaps, whistleKV),
		mat("v_taps", "self_attn/v_taps", whistleTaps, whistleV),
		vec("q_norm", "self_attn/q_norm/scale", whistleHead),
		vec("k_norm", "self_attn/k_norm/scale", whistleHead))
	ps = append(ps, gateOut...)
	ps = append(ps,
		vec("post_norm", "post_attn_norm/scale", whistleDim),
		vec("attn_gate", "attn_gate", 1),
		vec("pre_hada", "pre_hada_norm/scale", whistleDim))

	return append(ps, hadaParts("", "hadamard_mlp/")...)
}

// crossParts is one decoder block's cross-attention: 10 tensors, named with
// the decoder's suffixes. Q and K both have 8 heads; there is no GQA.
func crossParts() []part {
	return []part{
		vec("norm_in", "cross_norm/scale", whistleDim),
		kernel("q_proj", "cross_attn/q_proj/kernel", whistleQ),
		kernel("k_proj", "cross_attn/k_proj/kernel", whistleQ),
		kernel("v_proj", "cross_attn/v_proj/kernel", whistleDim),
		kernel("gate_proj", "cross_attn/gate_proj/kernel", whistleDim),
		kernel("out_proj", "cross_attn/out_proj/kernel", whistleDim),
		vec("q_norm", "cross_attn/q_norm/scale", whistleHead),
		vec("k_norm", "cross_attn/k_norm/scale", whistleHead),
		vec("post_norm", "post_cross_norm/scale", whistleDim),
		vec("attn_gate", "cross_gate", 1),
	}
}

// encoderParts is one encoder block: the six norms in macaron execution
// order, the attention, the convolution module and both HadamardMLPs:
// 43 tensors.
func encoderParts() []part {
	qkv, gateOut := attnParts()

	ps := []part{
		vec("pre_hada_0", "pre_hada_norm_0/scale", whistleDim),
		vec("norm_in", "ZCRMSNorm_0/scale", whistleDim),
		vec("post_norm", "post_attn_norm/scale", whistleDim),
		vec("conv_norm", "conv_norm/scale", whistleDim),
		vec("conv_out_norm", "conv_out_norm/scale", whistleDim),
		vec("pre_hada", "pre_hada_norm/scale", whistleDim),
	}
	ps = append(ps, qkv...)
	ps = append(ps, gateOut...)
	ps = append(ps,
		vec("q_norm", "self_attn/q_norm/scale", whistleHead),
		vec("k_norm", "self_attn/k_norm/scale", whistleHead),
		vec("attn_gate", "attn_gate", 1),
		kernel("pw1", "pw1/kernel", 2*whistleDim),
		kernel("pw2", "pw2/kernel", whistleDim),
		mat("dw", "dw", whistleKernel, whistleDim))
	ps = append(ps, hadaParts("hada_0.", "hadamard_mlp_0/")...)

	return append(ps, hadaParts("hada.", "hadamard_mlp/")...)
}

// layered expands a block's parts over every layer, layer-major.
func layered(format, prefix string, parts []part) []Entry {
	es := make([]Entry, 0, whistleLayers*len(parts))
	for i := range whistleLayers {
		for _, p := range parts {
			es = append(es, Entry{fmt.Sprintf(format, i) + p.suffix, prefix + p.path, i, p.transform, p.shape})
		}
	}

	return es
}

// mhcEntries is one stack's 9 mHC tensors, each whole (not per layer).
func mhcEntries(name, path string) []Entry {
	l, n, nC := whistleLayers, whistleLanes, whistleLanes*whistleDim

	return []Entry{
		{name + "mhc_a_pre", path + "mhc_a_pre", -1, Reshape, []int{l}},
		{name + "mhc_a_post", path + "mhc_a_post", -1, Reshape, []int{l}},
		{name + "mhc_a_res", path + "mhc_a_res", -1, Reshape, []int{l}},
		{name + "mhc_b_pre", path + "mhc_b_pre", -1, Reshape, []int{l, n}},
		{name + "mhc_b_post", path + "mhc_b_post", -1, Reshape, []int{l, n}},
		{name + "mhc_b_res", path + "mhc_b_res", -1, Reshape, []int{l, n, n}},
		{name + "mhc_phi_pre", path + "mhc_phi_pre", -1, LaneMajor, []int{l * n, nC}},
		{name + "mhc_phi_post", path + "mhc_phi_post", -1, LaneMajor, []int{l * n, nC}},
		{name + "mhc_phi_res", path + "mhc_phi_res", -1, LaneMajor, []int{l * n * n, nC}},
	}
}

// engramEntries is engram site s, the safetensors engrams_s.
func engramEntries(s int) []Entry {
	name, path := fmt.Sprintf("engram%d.", s), fmt.Sprintf("engrams_%d/", s)

	return []Entry{
		{name + "tables", path + "embedding", -1, Reshape, []int{whistleTables * whistleSlots, whistleSubDim}},
		{name + "key_proj", path + "key_proj/kernel", -1, Transpose, []int{whistleDim, whistleDim}},
		{name + "value_proj", path + "value_proj/kernel", -1, Transpose, []int{whistleDim, whistleDim}},
		{name + "taps", path + "taps", -1, Reshape, []int{whistleTables, whistleDim}},
	}
}

// stemEntries is the conv stem: the 3x3 kernels flattened to 9 rows, the
// 1x1 kernels and the output projection transposed.
func stemEntries() []Entry {
	return []Entry{
		{"stem.w", "stem/w", -1, Reshape, []int{9, whistleStem}},
		{"stem.dw_1", "stem/dw_1", -1, Reshape, []int{9, whistleStem}},
		{"stem.pw_1", "stem/pw_1/kernel", -1, Transpose, []int{whistleStem, whistleStem}},
		{"stem.dw_2", "stem/dw_2", -1, Reshape, []int{9, whistleStem}},
		{"stem.pw_2", "stem/pw_2/kernel", -1, Transpose, []int{whistleStem, whistleStem}},
		{"stem.out", "stem/out/kernel", -1, Transpose, []int{whistleDim, 10 * whistleStem}},
	}
}

// WhistleNames returns the tensor map of whistle.cact @ b358ddad: entry i
// describes tensor i. The decoder comes first, as export.py _tensors writes
// it, then the audio part (docs/FINDINGS.md F-006):
//
//	0        embedding
//	1-216    8 decoder layers x 27        layerNN.*
//	217-225  decoder mHC                  mhc_*
//	226-227  hada_p1, hada_p2             derived
//	228-235  engram sites 3 and 7         engram0.*, engram1.*
//	236      final_norm
//	237      audio.manifest               derived
//	238-317  8 cross-attentions x 10      crossNN.*
//	318      pe_gate
//	319-662  8 encoder layers x 43        encNN.*
//	663-671  encoder mHC                  enc.mhc_*
//	672      enc.final_norm
//	673-678  conv stem                    stem.*
//	679      audio.mel_filters            derived
//	680      tokenizer                    derived (RAW)
func WhistleNames() []Entry {
	es := []Entry{{"embedding", "embedding/embedding", -1, Reshape, []int{whistleVocab, whistleDim}}}
	es = append(es, layered("layer%02d.", "stack/layers/block/", decoderParts())...)
	es = append(es, mhcEntries("", "stack/")...)
	es = append(es,
		Entry{"hada_p1", "", -1, Derived, []int{whistleDim}},
		Entry{"hada_p2", "", -1, Derived, []int{whistleDim}})

	for s := range whistleEngrams {
		es = append(es, engramEntries(s)...)
	}

	es = append(es,
		Entry{"final_norm", "stack/final_norm/scale", -1, Reshape, []int{whistleDim}},
		Entry{"audio.manifest", "", -1, Derived, []int{13}})
	es = append(es, layered("cross%02d.", "stack/layers/block/", crossParts())...)
	es = append(es, Entry{"pe_gate", "pe_gate", -1, Reshape, []int{1}})
	es = append(es, layered("enc%02d.", "encoder/layers/block/", encoderParts())...)
	es = append(es, mhcEntries("enc.", "encoder/")...)
	es = append(es, Entry{"enc.final_norm", "encoder/final_norm/scale", -1, Reshape, []int{whistleDim}})
	es = append(es, stemEntries()...)

	return append(es,
		Entry{"audio.mel_filters", "", -1, Derived, []int{whistleBins, whistleMels}},
		Entry{"tokenizer", "", -1, Derived, nil})
}

// ErrDerived reports Apply on an entry without a safetensors source.
var ErrDerived = errors.New("derived tensor has no safetensors source")

// Apply maps the safetensors tensor e.Source, given as row-major values and
// their shape, to e's .cact layout: it takes layer e.Layer of a stacked
// tensor, applies e.Transform and checks the result against e.Shape. The
// result never aliases src.
func (e Entry) Apply(src []float32, shape []int) ([]float32, error) {
	if e.Transform == Derived {
		return nil, fmt.Errorf("cact: %s: %w", e.Name, ErrDerived)
	}

	src, shape, err := e.slice(src, shape)
	if err != nil {
		return nil, err
	}

	switch e.Transform {
	case Reshape:
		return slices.Clone(src), nil
	case Transpose:
		if len(shape) != 2 || len(e.Shape) != 2 || shape[0] != e.Shape[1] {
			return nil, fmt.Errorf("cact: %s: cannot transpose %v to %v", e.Name, shape, e.Shape)
		}

		return transpose(src, shape[0], shape[1]), nil
	case LaneMajor:
		if len(shape) != 3 || len(e.Shape) != 2 || e.Shape[1] != shape[1] {
			return nil, fmt.Errorf("cact: %s: cannot lay out %v lane-major as %v", e.Name, shape, e.Shape)
		}

		return laneMajor(src, shape[0], shape[1], shape[2]), nil
	default:
		return nil, fmt.Errorf("cact: %s: unknown %s", e.Name, e.Transform)
	}
}

// slice checks src against shape, takes layer e.Layer if set, and checks
// the element count against e.Shape.
func (e Entry) slice(src []float32, shape []int) ([]float32, []int, error) {
	if numel(shape) != len(src) {
		return nil, nil, fmt.Errorf("cact: %s: %d values for shape %v", e.Name, len(src), shape)
	}

	if e.Layer >= 0 {
		if len(shape) == 0 || e.Layer >= shape[0] {
			return nil, nil, fmt.Errorf("cact: %s: no layer %d in shape %v", e.Name, e.Layer, shape)
		}

		per := len(src) / shape[0]
		src, shape = src[e.Layer*per:(e.Layer+1)*per], shape[1:]
	}

	if numel(shape) != numel(e.Shape) {
		return nil, nil, fmt.Errorf("cact: %s: source shape %v does not fit %v", e.Name, shape, e.Shape)
	}

	return src, shape, nil
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}

	return n
}

// transpose turns a row-major (rows, cols) matrix into (cols, rows).
func transpose(src []float32, rows, cols int) []float32 {
	out := make([]float32, len(src))

	for r := range rows {
		for c := range cols {
			out[c*rows+r] = src[r*cols+c]
		}
	}

	return out
}

// laneMajor is export.py's phi.transpose(0, 2, 1).reshape(layers*lanes, nC)
// for a row-major (layers, nC, lanes) phi.
func laneMajor(src []float32, layers, nC, lanes int) []float32 {
	out := make([]float32, len(src))

	for l := range layers {
		for c := range nC {
			for k := range lanes {
				out[(l*lanes+k)*nC+c] = src[(l*nC+c)*lanes+k]
			}
		}
	}

	return out
}
