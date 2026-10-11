package text

import (
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// Config is the text model's inference geometry, as both formats state it:
// the .cact header or the safetensors config (docs/FINDINGS.md F-045).
// Training-only keys are left out.
type Config struct {
	Vocab         int     // tokenizer pieces and embedding rows
	OutVocab      int     // logits per position: the tied head's rows
	Dim           int     // model width
	HadaN         int     // HadamardMLP width: Dim rounded up to a power of two
	Heads         int     // query heads
	KVHeads       int     // KV heads (GQA)
	QKDim         int     // query/key head width
	VDim          int     // value head width
	Layers        int     // blocks
	MaxSeqLen     int     // longest sequence the model was trained for
	Lanes         int     // mHC lanes
	Taps          int     // causal conv taps on Q/K/V
	SlidingWindow int     // attention width of the non-global layers
	GlobalLayers  []int   // full-attention layers
	KVWindow      int     // the engine's KV window (export.py kv_window)
	KVBits        int     // KV-cache width the model was post-trained for
	Sites         []int   // layers carrying engram memory
	Orders        []int   // engram n-gram orders
	Slots         int     // engram table rows
	Tables        int     // engram tables per site: len(Orders) x hash heads
	SubDim        int     // engram table width
	ConvTaps      int     // engram value taps
	Rope          float64 // RoPE theta
}

// ConfigFromHeader reads the geometry from a .cact header.
func ConfigFromHeader(h cact.Header) Config {
	out := int(h.OutVocab)
	if out == 0 {
		out = int(h.Vocab) // export.py: 0 = the full vocab
	}

	return Config{
		Vocab: int(h.Vocab), OutVocab: out, Dim: int(h.DModel), HadaN: int(h.HadaN),
		Heads: int(h.NumHeads), KVHeads: int(h.NumKVHeads), QKDim: int(h.QKHeadDim), VDim: int(h.VHeadDim),
		Layers: int(h.NumLayers), MaxSeqLen: int(h.MaxSeqLen), Lanes: int(h.MHCLanes), Taps: int(h.QKVConvTaps),
		SlidingWindow: int(h.SlidingWindow), GlobalLayers: h.GlobalLayers(),
		KVWindow: int(h.KVWindow), KVBits: int(h.KVBits),
		Sites: h.Sites(), Orders: h.Orders(),
		Slots: int(h.EngramSlots), Tables: int(h.NumEngramTables), SubDim: int(h.EngramSubDim),
		ConvTaps: int(h.EngramConvTaps), Rope: float64(h.RopeTheta),
	}
}

// ConfigFromSafetensors reads the geometry from the checkpoint's config,
// with TransformerConfig's defaults for the keys it leaves out.
func ConfigFromSafetensors(c safetensors.TextConfig) Config {
	out := c.OutVocab
	if out == 0 {
		out = c.Vocab
	}

	return Config{
		Vocab: c.Vocab, OutVocab: out, Dim: c.Dim, HadaN: c.HadaN(),
		Heads: c.Heads, KVHeads: c.KVHeads, QKDim: c.QKDim, VDim: c.VDim,
		Layers: c.Layers, MaxSeqLen: c.MaxSeqLen, Lanes: c.Lanes, Taps: c.Taps,
		SlidingWindow: c.SlidingWindow, GlobalLayers: slices.Clone(c.GlobalLayers),
		KVWindow: c.KVWindow, KVBits: c.KVBits,
		Sites: slices.Clone(c.Sites), Orders: c.Orders(),
		Slots: c.Slots, Tables: c.Tables(), SubDim: c.SubDim(),
		ConvTaps: engramConvTaps, Rope: c.Rope(),
	}
}

// engramConvTaps is architecture.py ENGRAM_CONV_TAPS.
const engramConvTaps = 4

// Header is the .cact header a text export of c carries (export.py
// _pack_cact), less the tag, the tensor count and the codebook length.
func (c Config) Header() (cact.Header, error) {
	if len(c.Orders) > 4 || len(c.Sites) > 16 || c.Layers > 64 {
		return cact.Header{}, fmt.Errorf("text: %d orders, %d sites, %d layers exceed the .cact header",
			len(c.Orders), len(c.Sites), c.Layers)
	}

	var gmask uint64
	for _, l := range c.GlobalLayers {
		gmask |= 1 << uint(l)
	}

	h := cact.Header{
		KVWindow: u32(c.KVWindow), KVBits: u32(c.KVBits), Vocab: u32(c.Vocab), OutVocab: u32(c.OutVocab),
		DModel: u32(c.Dim), NumHeads: u32(c.Heads), NumKVHeads: u32(c.KVHeads), NumLayers: u32(c.Layers),
		QKHeadDim: u32(c.QKDim), VHeadDim: u32(c.VDim), MaxSeqLen: u32(c.MaxSeqLen), HadaN: u32(c.HadaN),
		MHCLanes: u32(c.Lanes), SlidingWindow: u32(c.SlidingWindow),
		GlobalMaskLo: uint32(gmask), GlobalMaskHi: uint32(gmask >> 32), QKVConvTaps: u32(c.Taps),
		EngramSlots: u32(c.Slots), EngramSubDim: u32(c.SubDim), NumEngramTables: u32(c.Tables),
		EngramConvTaps: u32(c.ConvTaps), EngramConvDilation: u32(slices.Max(append([]int{0}, c.Orders...))),
		NumEngramOrders: u32(len(c.Orders)), NumEngramSites: u32(len(c.Sites)), RopeTheta: float32(c.Rope),
	}

	for i, o := range c.Orders {
		h.EngramOrders[i] = u32(o)
	}

	for i, s := range c.Sites {
		h.EngramSites[i] = u32(s)
	}

	return h, nil
}

func u32(v int) uint32 {
	if v < 0 || v > math.MaxUint32 {
		return 0
	}

	return uint32(v)
}
