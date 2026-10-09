package weights

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// tinyConfig is a small but consistent geometry for the synthetic files.
func tinyConfig() Config {
	return Config{
		Vocab: 16, Text: 9, Dim: 8, HadaN: 8, Heads: 2, KVHeads: 1, QKDim: 4, VDim: 4,
		DecLayers: 2, EncLayers: 2, Lanes: 2, EncLanes: 2, Taps: 3, ConvKernel: 3,
		Sites: []int{1}, Orders: []int{2, 3}, Slots: 4, Tables: 4,
		Stem: 4, StemStages: 1, Rope: 10000,
		Mels: 4, Rate: 16000, FFT: 8, Win: 8, Hop: 4, MaxTokens: 16,
	}
}

// tinyNames is a name map over every transform and dtype: a CQ2 embedding,
// two layer slices of a transposed FP16 kernel, a lane-major FP32 mHC phi,
// an FP16 scalar, the four numeric derived records and a RAW tokenizer.
func tinyNames() []cact.Entry {
	return []cact.Entry{
		{Name: "embedding", Source: "embedding/embedding", Layer: -1, Transform: cact.Reshape, Shape: []int{16, 8}},
		{Name: "layer00.q_proj", Source: "stack/q_proj/kernel", Layer: 0, Transform: cact.Transpose, Shape: []int{8, 8}},
		{Name: "layer01.q_proj", Source: "stack/q_proj/kernel", Layer: 1, Transform: cact.Transpose, Shape: []int{8, 8}},
		{Name: "mhc_phi_pre", Source: "stack/mhc_phi_pre", Layer: -1, Transform: cact.LaneMajor, Shape: []int{4, 16}},
		{Name: "hada_p1", Layer: -1, Transform: cact.Derived, Shape: []int{8}},
		{Name: "hada_p2", Layer: -1, Transform: cact.Derived, Shape: []int{8}},
		{Name: "audio.manifest", Layer: -1, Transform: cact.Derived, Shape: []int{13}},
		{Name: "pe_gate", Source: "pe_gate", Layer: -1, Transform: cact.Reshape, Shape: []int{1}},
		{Name: "audio.mel_filters", Layer: -1, Transform: cact.Derived, Shape: []int{5, 4}},
		{Name: "tokenizer", Layer: -1, Transform: cact.Derived},
	}
}

// fixture is one model in both formats: the .cact holds what the exporter
// would write, the safetensors file the sources it would read. The CQ
// embedding's source is its own dequantisation and every FP16 value is
// exact in FP16, so both backends must agree bit for bit.
type fixture struct {
	cfg         Config
	names       []cact.Entry
	cact, st    []byte
	tokenizer   []byte
	cactTensors map[string][]float32 // .cact values by entry name
}

func newFixture(t *testing.T) fixture {
	t.Helper()

	cfg := tinyConfig()
	fx := fixture{
		cfg: cfg, names: tinyNames(), tokenizer: []byte("tokenizer blob"),
		cactTensors: map[string][]float32{},
	}

	rng := rand.New(rand.NewPCG(1, 2))
	quarters := func(n int) []float32 {
		v := make([]float32, n)
		for k := range v {
			v[k] = float32(rng.IntN(33)-16) / 4
		}

		return v
	}

	// CQ2, group 8: 16 rows of 8 2-bit indices (2 bytes), then 16 FP16 norms.
	cqBlob := make([]byte, 16*2+16*2)
	for k := range 32 {
		cqBlob[k] = byte(rng.Uint32())
	}

	for r := range 16 {
		binary.LittleEndian.PutUint16(cqBlob[32+2*r:], 0x3c00+uint16(r)) // 1.0 and up
	}

	qProj := quarters(2 * 8 * 8) // (layers, in, out)
	phi := quarters(2 * 16 * 2)  // (layers, n*C, lanes)
	peGate := []float32{0.75}

	perm := func(seed uint32) []float32 {
		p := cact.NumpyPermutation(seed, cfg.HadaN)

		v := make([]float32, len(p))
		for k, x := range p {
			v[k] = float32(x)
		}

		return v
	}

	tensors := []fixtureTensor{
		{dtype: cact.CQ, shape: []uint32{16, 8}, blob: cqBlob, group: 8, bits: 2},
		{dtype: cact.FP16, shape: []uint32{8, 8}, blob: f16Blob(transpose(qProj[:64], 8, 8))},
		{dtype: cact.FP16, shape: []uint32{8, 8}, blob: f16Blob(transpose(qProj[64:], 8, 8))},
		{dtype: cact.FP32, shape: []uint32{4, 16}, blob: f32Blob(laneMajor(phi, 2, 16, 2))},
		{dtype: cact.FP32, shape: []uint32{8}, blob: f32Blob(perm(11))},
		{dtype: cact.FP32, shape: []uint32{8}, blob: f32Blob(perm(13))},
		{dtype: cact.FP32, shape: []uint32{13}, blob: f32Blob(cfg.manifestRecord())},
		{dtype: cact.FP16, shape: []uint32{1}, blob: f16Blob(peGate)},
		{dtype: cact.FP32, shape: []uint32{5, 4}, blob: f32Blob(cact.SlaneyMel(cfg.Rate, cfg.FFT, cfg.Mels))},
		{dtype: cact.RAW, blob: fx.tokenizer},
	}

	fx.cact = encodeCact(t, cactHeader(cfg), tensors)

	f, err := cact.Parse(fx.cact)
	if err != nil {
		t.Fatalf("fixture: cact.Parse: %v", err)
	}

	for i, e := range fx.names {
		rec, _ := f.Record(i)

		var v []float32

		switch rec.DType {
		case cact.CQ:
			v, err = f.CQ(i)
		case cact.FP16, cact.FP32:
			v, err = f.Float32(i)
		default:
			continue
		}

		if err != nil {
			t.Fatalf("fixture: decode %s: %v", e.Name, err)
		}

		fx.cactTensors[e.Name] = v
	}

	fx.st = encodeSafetensors(t, []safetensors.Tensor{
		{Name: "embedding/embedding", Shape: []int64{16, 8}, Data: fx.cactTensors["embedding"]},
		{Name: "stack/q_proj/kernel", Shape: []int64{2, 8, 8}, Data: qProj},
		{Name: "stack/mhc_phi_pre", Shape: []int64{2, 16, 2}, Data: phi},
		{Name: "pe_gate", Shape: []int64{1, 1}, Data: peGate},
	}, stConfig(cfg))

	return fx
}

// cactHeader is the decoder geometry export.py writes for cfg (F-003).
func cactHeader(cfg Config) cact.Header {
	h := cact.Header{
		Tag: cact.Tag, KVBits: 8, Vocab: uint32(cfg.Vocab), DModel: uint32(cfg.Dim),
		NumHeads: uint32(cfg.Heads), NumKVHeads: uint32(cfg.KVHeads), NumLayers: uint32(cfg.DecLayers),
		QKHeadDim: uint32(cfg.QKDim), VHeadDim: uint32(cfg.VDim), MaxSeqLen: uint32(cfg.MaxTokens),
		HadaN: uint32(cfg.HadaN), MHCLanes: uint32(cfg.Lanes), QKVConvTaps: uint32(cfg.Taps),
		EngramSlots: uint32(cfg.Slots), EngramSubDim: uint32(cfg.Dim / cfg.Tables),
		NumEngramTables: uint32(cfg.Tables), EngramConvTaps: 4, EngramConvDilation: 3,
		NumEngramOrders: uint32(len(cfg.Orders)), NumEngramSites: uint32(len(cfg.Sites)),
		RopeTheta: float32(cfg.Rope),
	}

	for k, o := range cfg.Orders {
		h.EngramOrders[k] = uint32(o)
	}

	for k, s := range cfg.Sites {
		h.EngramSites[k] = uint32(s)
	}

	return h
}

// stConfig is the safetensors config that maps onto cfg.
func stConfig(cfg Config) safetensors.Config {
	return safetensors.Config{
		Text: cfg.Text, Dim: cfg.Dim, KVHeads: cfg.KVHeads, QKDim: cfg.QKDim, VDim: cfg.VDim,
		EncLayers: cfg.EncLayers, DecLayers: cfg.DecLayers, ConvKernel: cfg.ConvKernel, Taps: cfg.Taps,
		Lanes: cfg.Lanes, EncLanes: cfg.EncLanes, Sites: cfg.Sites, Orders: cfg.Orders, Slots: cfg.Slots,
		Hashes: cfg.Tables / len(cfg.Orders), Stem: cfg.Stem, StemStages: cfg.StemStages, Rope: cfg.Rope,
		Mels: cfg.Mels, Rate: cfg.Rate, FFT: cfg.FFT, Win: cfg.Win, Hop: cfg.Hop, MaxTokens: cfg.MaxTokens,
		Pad: 0, EOS: 1, BOS: 2, DType: "bfloat16", Speed: []float64{1},
	}
}

func encodeSafetensors(t *testing.T, tensors []safetensors.Tensor, cfg safetensors.Config) []byte {
	t.Helper()

	js, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}

	data, err := safetensors.EncodeTensorsWithMetadata(tensors, map[string]string{"config": string(js)})
	if err != nil {
		t.Fatalf("EncodeTensorsWithMetadata: %v", err)
	}

	return data
}

// fixtureTensor is one .cact directory entry plus its blob.
type fixtureTensor struct {
	dtype       cact.DType
	shape       []uint32
	blob        []byte
	group, bits uint32
}

// encodeCact lays a file out as export.py _pack_cact does: header,
// codebook, directory, then 64-byte aligned blobs.
func encodeCact(t *testing.T, h cact.Header, tensors []fixtureTensor) []byte {
	t.Helper()

	h.NumTensors = uint32(len(tensors))
	h.CodebookLen = cact.CodebookLen

	cb := make([]float32, cact.CodebookLen)
	for k := range cb {
		cb[k] = float32(k)/8 - 1
	}

	var buf bytes.Buffer

	err := binary.Write(&buf, binary.LittleEndian, h)
	if err == nil {
		err = binary.Write(&buf, binary.LittleEndian, cb)
	}

	if err != nil {
		t.Fatalf("encode header: %v", err)
	}

	pos := uint64(buf.Len() + len(tensors)*cact.RecordSize)
	records := make([]cact.Record, len(tensors))

	for i, tt := range tensors {
		pos = (pos + cact.Align - 1) &^ (cact.Align - 1)
		records[i] = cact.Record{
			DType: tt.dtype, NDim: uint8(len(tt.shape)), Offset: pos, NBytes: uint64(len(tt.blob)),
			Group: tt.group, Bits: tt.bits,
		}
		copy(records[i].Shape[:], tt.shape)
		pos += uint64(len(tt.blob))
	}

	err = binary.Write(&buf, binary.LittleEndian, records)
	if err != nil {
		t.Fatalf("encode directory: %v", err)
	}

	for i, tt := range tensors {
		buf.Write(make([]byte, int(records[i].Offset)-buf.Len()))
		buf.Write(tt.blob)
	}

	return buf.Bytes()
}

func f32Blob(vals []float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}

	return out
}

// f16Blob encodes values that FP16 holds exactly: quarters with |v| <= 4.
func f16Blob(vals []float32) []byte {
	out := make([]byte, 2*len(vals))

	for i, v := range vals {
		var h uint16

		if v != 0 {
			frac, exp := math.Frexp(math.Abs(float64(v))) // |v| = frac * 2^exp, frac in [0.5, 1)
			h = uint16(exp-1+15)<<10 | uint16(math.Ldexp(frac, 11))&0x3ff
		}

		if v < 0 {
			h |= 0x8000
		}

		binary.LittleEndian.PutUint16(out[2*i:], h)
	}

	return out
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

// laneMajor is export.py's phi.transpose(0, 2, 1).reshape(layers*lanes, nC).
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
