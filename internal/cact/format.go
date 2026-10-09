// Package cact reads the .cact deployment container that Needle 3 exports
// (upstream needle/model/export.py @ af2654dffa): a fixed header with the
// model geometry, the CQ codebooks, a nameless tensor directory and 64-byte
// aligned tensor blobs. Tensors are positional; WhistleNames maps Whistle's
// indices to names and safetensors sources (docs/FINDINGS.md F-006).
package cact

import "fmt"

// Tag opens every .cact file.
const Tag uint32 = 0x05E12A84

// Layout constants of the container. All integers are little-endian.
const (
	Align       = 64  // every blob starts on a 64-byte boundary
	HeaderSize  = 196 // export.py _HDR_FMT "<48If"; its docstring's "120 bytes" is stale
	RecordSize  = 44  // export.py _REC_FMT "<BBHIIIIQQII"
	CodebookLen = 28  // cb2[4] | cb3[8] | cb4[16]
	MaxDims     = 4
)

// DType is a directory record's storage type.
type DType uint8

// Record dtypes, as in export.py.
const (
	FP16 DType = 1
	FP32 DType = 2
	CQ   DType = 3 // codebook-quantised [out, in] matrix, Record.Bits wide
	RAW  DType = 4 // opaque bytes; Whistle's only one is the tokenizer
)

func (d DType) String() string {
	switch d {
	case FP16:
		return "FP16"
	case FP32:
		return "FP32"
	case CQ:
		return "CQ"
	case RAW:
		return "RAW"
	default:
		return fmt.Sprintf("dtype %d", uint8(d))
	}
}

// Header is the fixed file header, field for field in export.py order. Its
// geometry describes the text decoder; see docs/FINDINGS.md F-003 for how it
// maps onto the safetensors config.
type Header struct {
	Tag                uint32
	NumTensors         uint32
	CodebookLen        uint32
	KVWindow           uint32 // trained sliding-window width, 0 = size from the KV budget
	KVBits             uint32 // KV-cache width the model was post-trained for
	Vocab              uint32
	OutVocab           uint32 // tied head width, 0 = the full vocab
	DModel             uint32
	NumHeads           uint32
	NumKVHeads         uint32
	NumLayers          uint32
	QKHeadDim          uint32
	VHeadDim           uint32
	MaxSeqLen          uint32
	HadaN              uint32
	MHCLanes           uint32
	SlidingWindow      uint32
	GlobalMaskLo       uint32 // bit i marks layer i as full attention
	GlobalMaskHi       uint32
	QKVConvTaps        uint32
	EngramSlots        uint32
	EngramSubDim       uint32
	NumEngramTables    uint32
	EngramConvTaps     uint32
	EngramConvDilation uint32
	EngramSeedHeads    uint32
	NumEngramOrders    uint32
	EngramOrders       [4]uint32
	NumEngramSites     uint32
	EngramSites        [16]uint32
	RopeTheta          float32
}

// GlobalLayers lists the full-attention layers from the 64-bit global mask.
func (h Header) GlobalLayers() []int {
	mask := uint64(h.GlobalMaskHi)<<32 | uint64(h.GlobalMaskLo)

	var layers []int

	for i := range min(int(h.NumLayers), 64) {
		if mask>>i&1 == 1 {
			layers = append(layers, i)
		}
	}

	return layers
}

// Orders returns the engram n-gram orders. Parse guarantees
// NumEngramOrders <= 4.
func (h Header) Orders() []int {
	return ints(h.EngramOrders[:h.NumEngramOrders])
}

// Sites returns the decoder layers carrying engram memory. Parse guarantees
// NumEngramSites <= 16.
func (h Header) Sites() []int {
	return ints(h.EngramSites[:h.NumEngramSites])
}

func ints(vals []uint32) []int {
	out := make([]int, len(vals))
	for i, v := range vals {
		out[i] = int(v)
	}

	return out
}

// Record is one 44-byte directory entry.
type Record struct {
	DType  DType
	NDim   uint8
	Pad    uint16
	Shape  [MaxDims]uint32 // dims past NDim are zero
	Offset uint64          // absolute file offset of the blob
	NBytes uint64
	Group  uint32 // CQ group size, 0 otherwise
	Bits   uint32 // CQ width: 1, 2, 3, 4, or 5 for ternary crumbs; 0 otherwise
}

// Dims returns the logical shape.
func (r Record) Dims() []int {
	return ints(r.Shape[:min(int(r.NDim), MaxDims)])
}

// Numel is the element count; 1 for a scalar. Records from a parsed File
// are checked not to overflow.
func (r Record) Numel() int {
	n := 1
	for _, d := range r.Dims() {
		n *= d
	}

	return n
}
