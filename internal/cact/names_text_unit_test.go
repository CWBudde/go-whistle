package cact

import (
	"slices"
	"testing"
)

// smallTextHeader is a text geometry small enough to count by hand: width
// 96 (HadamardMLP 128 = 8 x 16), 2 layers without taps, one engram site.
func smallTextHeader() Header {
	return Header{
		Vocab: 300, DModel: 96, NumHeads: 2, NumKVHeads: 1, NumLayers: 2, QKHeadDim: 16, VHeadDim: 24,
		HadaN: 128, MHCLanes: 2, EngramSlots: 10, EngramSubDim: 48, NumEngramTables: 2, EngramConvTaps: 4,
		NumEngramOrders: 1, EngramOrders: [4]uint32{2}, NumEngramSites: 1, EngramSites: [16]uint32{1},
	}
}

func TestTextNamesForCounts(t *testing.T) {
	heads := []HeadShape{
		{Code: HeadEmbedding, Probes: 2, Queries: 3, Out: 32},
		{Code: HeadConfidence, Probes: 4, Queries: 4, Out: 1},
		{Code: HeadRouter, Probes: 4, Queries: 4, Out: 3},
	}

	es, err := TextNamesFor(smallTextHeader(), heads)
	if err != nil {
		t.Fatal(err)
	}

	// embedding, 2 x 24 block tensors, 9 mHC, 2 perms, 4 engram, final_norm,
	// manifest, 6 + 6 + 7 head tensors, tokenizer.
	if want := 1 + 2*24 + 9 + 2 + 4 + 1 + 1 + 6 + 6 + 7 + 1; len(es) != want {
		t.Fatalf("%d entries, want %d", len(es), want)
	}

	byName := map[string]Entry{}
	for _, e := range es {
		byName[e.Name] = e
	}

	for name, want := range map[string][]int{
		"layer00.q_proj":           {32, 96},
		"layer00.out_proj":         {96, 48},
		"layer01.w1a":              {8, 8},
		"layer01.w3b":              {16, 16},
		"layer01.cond_v":           {96, 8},
		"layer01.cond_u":           {8, 128},
		"mhc_phi_res":              {8, 192},
		"hada_p2":                  {128},
		"engram0.tables":           {20, 48},
		"engram0.key_proj":         {96, 96},
		"embedding_head.probes":    {6, 96},
		"embedding_head.proj":      {32, 288},
		"confidence_head.row_bias": {4, 3, 4},
		"router_head.calibration":  {3},
	} {
		if e, ok := byName[name]; !ok || !slices.Equal(e.Shape, want) {
			t.Errorf("%s = %+v, want shape %v", name, e, want)
		}
	}

	if byName["embedding_head.bias"].Transform != Derived || byName["router_head.bias"].Source != "router_head/proj/bias" {
		t.Error("head biases: the embedding head's is derived, the others are sourced")
	}

	if _, ok := byName["layer00.q_taps"]; ok {
		t.Error("taps without qkv_conv_taps")
	}

	if es[len(es)-1].Name != "tokenizer" || es[1+2*24+9+2+4+1].Name != "heads.manifest" {
		t.Error("manifest or tokenizer out of place")
	}

	noHeads, err := TextNamesFor(smallTextHeader(), nil)
	if err != nil || len(noHeads) != 1+2*24+9+2+4+1+1 {
		t.Errorf("without heads: %d entries, %v", len(noHeads), err)
	}

	bad := smallTextHeader()
	bad.HadaN = 96

	_, err = TextNamesFor(bad, nil)
	if err == nil {
		t.Error("accepted hada_n 96 for d_model 96")
	}

	_, err = TextNamesFor(smallTextHeader(), []HeadShape{{Code: 9, Probes: 1, Queries: 1, Out: 1}})
	if err == nil {
		t.Error("accepted head code 9")
	}
}
