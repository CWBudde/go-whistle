package cact

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
)

// Probe-head codes of a heads.manifest, in export.py's canonical order
// (architecture.py HEADS @ af2654dffa).
const (
	HeadEmbedding  = 1
	HeadConfidence = 2
	HeadRouter     = 3
)

// headKeys are the safetensors module names of the probe heads.
var headKeys = map[int]string{
	HeadEmbedding:  "embedding_head",
	HeadConfidence: "confidence_head",
	HeadRouter:     "router_head",
}

// hadaRank is architecture.py HADA_COND_RANK.
const hadaRank = 8

// The Hadamard permutation records, derived by export.py.
const (
	hadaP1 = "hada_p1"
	hadaP2 = "hada_p2"
)

// textGeom is the text model geometry a .cact header states
// (export.py _pack_cact).
type textGeom struct {
	dim, heads, kvHeads, qk, v, layers, taps int
	hadaN, hadaA, hadaB                      int
	lanes, tables, slots, subDim, convTaps   int
	vocab, sites                             int
}

func newTextGeom(h Header) (textGeom, error) {
	g := textGeom{
		dim: int(h.DModel), heads: int(h.NumHeads), kvHeads: int(h.NumKVHeads),
		qk: int(h.QKHeadDim), v: int(h.VHeadDim), layers: int(h.NumLayers), taps: int(h.QKVConvTaps),
		hadaN: int(h.HadaN), lanes: int(h.MHCLanes), tables: int(h.NumEngramTables),
		slots: int(h.EngramSlots), subDim: int(h.EngramSubDim), convTaps: int(h.EngramConvTaps),
		vocab: int(h.Vocab), sites: int(h.NumEngramSites),
	}

	if g.dim <= 0 || g.heads <= 0 || g.kvHeads <= 0 || g.layers <= 0 || g.lanes <= 0 {
		return textGeom{}, fmt.Errorf("cact: header geometry %+v is incomplete", h)
	}

	// export.py: hada_n = 1 << (d_model - 1).bit_length(); _hada_blocks:
	// b = 1 << ((n - 1).bit_length() // 2).
	if want := 1 << bits.Len(uint(g.dim-1)); g.hadaN != want {
		return textGeom{}, fmt.Errorf("cact: hada_n %d, want %d for d_model %d", g.hadaN, want, g.dim)
	}

	g.hadaA = 1 << (bits.Len(uint(g.hadaN-1)) / 2)
	g.hadaB = g.hadaN / g.hadaA

	return g, nil
}

// blockParts is one text block in export.py _tensors order: 24 tensors, 27
// with Q/K/V taps.
func (g textGeom) blockParts() []part {
	q, kv, v, o := g.heads*g.qk, g.kvHeads*g.qk, g.kvHeads*g.v, g.heads*g.v
	kern := func(suffix, path string, out, in int) part {
		return part{suffix, path, Transpose, []int{out, in}}
	}

	ps := []part{
		vec("norm_in", "ZCRMSNorm_0/scale", g.dim),
		kern("q_proj", "self_attn/q_proj/kernel", q, g.dim),
		kern("k_proj", "self_attn/k_proj/kernel", kv, g.dim),
		kern("v_proj", "self_attn/v_proj/kernel", v, g.dim),
	}

	if g.taps > 0 {
		ps = append(ps,
			mat("q_taps", "self_attn/q_taps", g.taps, q),
			mat("k_taps", "self_attn/k_taps", g.taps, kv),
			mat("v_taps", "self_attn/v_taps", g.taps, v))
	}

	ps = append(ps,
		vec("q_norm", "self_attn/q_norm/scale", g.qk),
		vec("k_norm", "self_attn/k_norm/scale", g.qk),
		kern("gate_proj", "self_attn/gate_proj/kernel", o, g.dim),
		kern("out_proj", "self_attn/out_proj/kernel", g.dim, o),
		vec("post_norm", "post_attn_norm/scale", g.dim),
		vec("attn_gate", "attn_gate", 1),
		vec("pre_hada", "pre_hada_norm/scale", g.dim))

	path := "hadamard_mlp/"
	for _, d := range []string{"d1", "d2", "b2", "d3", "d4"} {
		ps = append(ps, vec(d, path+d, g.hadaN))
	}

	for _, w := range []string{"w1", "w2", "w3"} {
		ps = append(ps,
			mat(w+"a", path+w+"a", g.hadaA, g.hadaA),
			mat(w+"b", path+w+"b", g.hadaB, g.hadaB))
	}

	return append(ps,
		mat("cond_v", path+"cond_v", g.dim, hadaRank),
		mat("cond_u", path+"cond_u", hadaRank, g.hadaN))
}

// HeadShape is one probe head of a heads.manifest: its code and its
// widths, the probes per row k, the queries q and the outputs.
type HeadShape struct {
	Code, Probes, Queries, Out int
}

// TextNames returns the tensor map of a Needle 3 text .cact such as
// needle3.cact @ 93e88ede (PLAN.md Phase 11, docs/FINDINGS.md F-045): entry
// i describes tensor i, in export.py _tensors order, with the geometry from
// the header and the probe heads from the heads.manifest record and the
// head records' shapes:
//
//	0                 embedding
//	1 … 27·L          L layers x 27           layerNN.*
//	+9                mHC                     mhc_*
//	+2                hada_p1, hada_p2        derived
//	+4·S              S engram sites          engramS.*
//	+1                final_norm
//	+1, +6 per head   heads.manifest, heads   derived, <key>_head.*
//	last              tokenizer               derived (RAW)
//
// The names are Whistle's (WhistleNames) wherever both models have the
// tensor, so the shared blocks of internal/needle load either.
func TextNames(f *File) ([]Entry, error) {
	g, err := newTextGeom(f.header)
	if err != nil {
		return nil, err
	}

	at := len(g.trunk())

	extra := f.Len() - at - 1 // what follows final_norm, less the tokenizer
	if extra < 0 {
		return nil, fmt.Errorf("cact: %d tensors, the text trunk alone needs %d", f.Len(), at+1)
	}

	var heads []HeadShape

	if extra > 0 {
		heads, err = g.headShapes(f, at)
		if err != nil {
			return nil, err
		}
	}

	es, err := TextNamesFor(f.header, heads)
	if err != nil {
		return nil, err
	}

	if len(es) != f.Len() {
		return nil, fmt.Errorf("%w: the manifest's heads take %d records, the file has %d", errHeads, len(es)-at-1, extra)
	}

	last, err := f.Record(len(es) - 1)
	if err != nil {
		return nil, err
	}

	if last.DType != RAW {
		return nil, fmt.Errorf("cact: record %d is %s, want the RAW tokenizer last", len(es)-1, last.DType)
	}

	return es, nil
}

// TextNamesFor is TextNames for a header and the probe heads, without a
// file: the map a text .cact with this geometry and these heads has, which
// a safetensors checkpoint serves under the same names. Without heads there
// is no heads.manifest.
func TextNamesFor(h Header, heads []HeadShape) ([]Entry, error) {
	g, err := newTextGeom(h)
	if err != nil {
		return nil, err
	}

	es := g.trunk()

	if len(heads) > 0 {
		es = append(es, Entry{"heads.manifest", "", -1, Derived, []int{len(heads)}})

		for _, hs := range heads {
			he, err := g.headEntries(hs)
			if err != nil {
				return nil, err
			}

			es = append(es, he...)
		}
	}

	return append(es, Entry{"tokenizer", "", -1, Derived, nil}), nil
}

// trunk is everything through final_norm.
func (g textGeom) trunk() []Entry {
	es := []Entry{{"embedding", "embedding/embedding", -1, Reshape, []int{g.vocab, g.dim}}}

	parts := g.blockParts()
	for i := range g.layers {
		for _, p := range parts {
			es = append(es, Entry{fmt.Sprintf("layer%02d.", i) + p.suffix, "stack/layers/block/" + p.path, i, p.transform, p.shape})
		}
	}

	l, n, nC := g.layers, g.lanes, g.lanes*g.dim
	es = append(es,
		Entry{"mhc_a_pre", "stack/mhc_a_pre", -1, Reshape, []int{l}},
		Entry{"mhc_a_post", "stack/mhc_a_post", -1, Reshape, []int{l}},
		Entry{"mhc_a_res", "stack/mhc_a_res", -1, Reshape, []int{l}},
		Entry{"mhc_b_pre", "stack/mhc_b_pre", -1, Reshape, []int{l, n}},
		Entry{"mhc_b_post", "stack/mhc_b_post", -1, Reshape, []int{l, n}},
		Entry{"mhc_b_res", "stack/mhc_b_res", -1, Reshape, []int{l, n, n}},
		Entry{"mhc_phi_pre", "stack/mhc_phi_pre", -1, LaneMajor, []int{l * n, nC}},
		Entry{"mhc_phi_post", "stack/mhc_phi_post", -1, LaneMajor, []int{l * n, nC}},
		Entry{"mhc_phi_res", "stack/mhc_phi_res", -1, LaneMajor, []int{l * n * n, nC}},
		Entry{hadaP1, "", -1, Derived, []int{g.hadaN}},
		Entry{hadaP2, "", -1, Derived, []int{g.hadaN}})

	kv := g.tables * g.subDim

	for s := range g.sites {
		name, path := fmt.Sprintf("engram%d.", s), fmt.Sprintf("engrams_%d/", s)
		es = append(es,
			Entry{name + "tables", path + "embedding", -1, Reshape, []int{g.tables * g.slots, g.subDim}},
			Entry{name + "key_proj", path + "key_proj/kernel", -1, Transpose, []int{g.dim, kv}},
			Entry{name + "value_proj", path + "value_proj/kernel", -1, Transpose, []int{g.dim, kv}},
			Entry{name + "taps", path + "taps", -1, Reshape, []int{g.convTaps, g.dim}})
	}

	return append(es, Entry{"final_norm", "stack/final_norm/scale", -1, Reshape, []int{g.dim}})
}

// errHeads reports a probe-head section export.py would not write.
var errHeads = errors.New("cact: malformed probe heads")

// headTensors is the number of records ProbeHead.export writes per head:
// probes, gain, query, row_bias, proj, bias.
const headTensors = 6

// headShapes reads the heads.manifest at index at and the widths of each
// head from its records: probes ((L+1)·k, d), query (q, d), proj
// (out, q·d) (export.py _head_tensors, ProbeHead.export).
func (g textGeom) headShapes(f *File, at int) ([]HeadShape, error) {
	rec, err := f.Record(at)
	if err != nil {
		return nil, err
	}

	if rec.DType != FP16 || rec.NDim != 1 {
		return nil, fmt.Errorf("%w: record %d is %s %v, want an FP16 heads.manifest", errHeads, at, rec.DType, rec.Dims())
	}

	codes, err := f.FP16(at)
	if err != nil {
		return nil, err
	}

	rows := g.layers + 1 // the embedding, then every block
	next := at + 1
	heads := make([]HeadShape, 0, len(codes))

	for _, c := range codes {
		code := int(math.Round(float64(c)))
		if _, ok := headKeys[code]; !ok || float32(code) != c {
			return nil, fmt.Errorf("%w: unknown head code %v", errHeads, c)
		}

		dims := make([][]int, headTensors)
		for k := range dims {
			r, err := f.Record(next + k)
			if err != nil {
				return nil, fmt.Errorf("%w: %s: %w", errHeads, headKeys[code], err)
			}

			dims[k] = r.Dims()
		}

		if len(dims[0]) != 2 || len(dims[2]) != 2 || len(dims[4]) != 2 || dims[0][0]%rows != 0 {
			return nil, fmt.Errorf("%w: %s shapes %v", errHeads, headKeys[code], dims)
		}

		heads = append(heads, HeadShape{Code: code, Probes: dims[0][0] / rows, Queries: dims[2][0], Out: dims[4][0]})

		next += headTensors
		if code == HeadRouter {
			next++ // the calibration
		}
	}

	return heads, nil
}

// headEntries maps one probe head: probes, gain, query, row_bias, proj and
// bias, and the router's calibration.
func (g textGeom) headEntries(hs HeadShape) ([]Entry, error) {
	key, ok := headKeys[hs.Code]
	if !ok || hs.Probes <= 0 || hs.Queries <= 0 || hs.Out <= 0 {
		return nil, fmt.Errorf("%w: head %+v", errHeads, hs)
	}

	rows, k, q := g.layers+1, hs.Probes, hs.Queries
	name, path := key+".", key+"/"

	es := []Entry{
		{name + "probes", path + "probes", -1, Reshape, []int{rows * k, g.dim}},
		{name + "gain", path + "gain", -1, Reshape, []int{rows, k}},
		{name + "query", path + "query", -1, Reshape, []int{q, g.dim}},
		{name + "row_bias", path + "row_bias", -1, Reshape, []int{q, rows, k}},
		{name + "proj", path + "proj/kernel", -1, Transpose, []int{hs.Out, q * g.dim}},
	}

	// The embedding head's projection has no bias; export.py writes zeros.
	if hs.Code == HeadEmbedding {
		es = append(es, Entry{name + "bias", "", -1, Derived, []int{hs.Out}})
	} else {
		es = append(es, Entry{name + "bias", path + "proj/bias", -1, Reshape, []int{hs.Out}})
	}

	if hs.Code == HeadRouter {
		es = append(es, Entry{name + "calibration", path + "calibration", -1, Reshape, []int{3}})
	}

	return es, nil
}
