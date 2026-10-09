package cact

import (
	"encoding/binary"
	"errors"
	"math"
	"math/bits"
	"os"
	"reflect"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// Pinned model files as laid out by `whistle download`.
const (
	whistleCactPath        = "../../models/whistle/whistle.cact"
	whistleSafetensorsPath = "../../models/whistle/whistle.safetensors"
)

func openWhistle(t *testing.T) *File {
	t.Helper()

	_, err := os.Stat(whistleCactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", whistleCactPath)
	}

	f, err := Open(whistleCactPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	return f
}

// TestWhistleCact pins whistle.cact @ b358ddad: the header, the codebooks,
// the directory layout and the records PLAN.md's tensor map relies on.
func TestWhistleCact(t *testing.T) {
	f := openWhistle(t)

	wantHeader := Header{
		Tag: Tag, NumTensors: 681, CodebookLen: 28, KVWindow: 0, KVBits: 8,
		Vocab: 8199, OutVocab: 0, DModel: 512, NumHeads: 8, NumKVHeads: 2, NumLayers: 8,
		QKHeadDim: 48, VHeadDim: 64, MaxSeqLen: 320, HadaN: 512, MHCLanes: 4,
		SlidingWindow: 0, GlobalMaskLo: 0, GlobalMaskHi: 0, QKVConvTaps: 3,
		EngramSlots: 18432, EngramSubDim: 128, NumEngramTables: 4, EngramConvTaps: 4, EngramConvDilation: 3,
		EngramSeedHeads: 0, NumEngramOrders: 2, EngramOrders: [4]uint32{2, 3},
		NumEngramSites: 2, EngramSites: [16]uint32{3, 7},
		RopeTheta: 100000,
	}
	if got := f.Header(); got != wantHeader {
		t.Errorf("Header() =\n%+v\nwant\n%+v", got, wantHeader)
	}

	// The Lloyd-Max codebooks, already scaled by 1/sqrt(128).
	wantCodebooks := map[int][]float32{
		2: {-0.1331721395254135, -0.03990209102630615, 0.04003528133034706, 0.13346044719219208},
		3: {
			-0.18926136195659637, -0.11788499355316162, -0.06614968925714493, -0.02101346105337143,
			0.022101031616330147, 0.06678506731987, 0.11826207488775253, 0.18982602655887604,
		},
		4: {
			-0.23953154683113098, -0.18070605397224426, -0.14095793664455414, -0.10901255160570145,
			-0.08151296526193619, -0.056527651846408844, -0.03293915092945099, -0.010116109624505043,
			0.012710800394415855, 0.03570791706442833, 0.059458255767822266, 0.08497702330350876,
			0.112741619348526, 0.1446969211101532, 0.184239000082016, 0.2422737032175064,
		},
	}
	for width, want := range wantCodebooks {
		got, err := f.Codebook(width)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("Codebook(%d) = %v, %v; want %v", width, got, err, want)
		}
	}

	type kind struct {
		dtype      DType
		bits, ndim int
	}

	hist := map[kind]int{}
	end := uint64(0)

	for i := range f.Len() {
		rec, _ := f.Record(i)
		hist[kind{rec.DType, int(rec.Bits), int(rec.NDim)}]++

		if rec.DType == CQ && rec.Group != 128 {
			t.Errorf("record %d: CQ group %d, want 128", i, rec.Group)
		}

		end = rec.Offset + rec.NBytes
	}

	wantHist := map[kind]int{
		{FP16, 0, 1}: 289, {FP16, 0, 2}: 233, {FP16, 0, 3}: 2,
		{FP32, 0, 1}: 3, {FP32, 0, 2}: 1,
		{CQ, 2, 2}: 145, {CQ, 4, 2}: 7,
		{RAW, 0, 0}: 1,
	}
	if !reflect.DeepEqual(hist, wantHist) {
		t.Errorf("record histogram = %v, want %v", hist, wantHist)
	}

	if size := uint64(len(f.data)); end != size {
		t.Errorf("last blob ends at %d, file is %d bytes", end, size)
	}

	first, _ := f.Record(0)
	if first.DType != CQ || first.Bits != 4 || !slices.Equal(first.Dims(), []int{8199, 512}) ||
		first.Offset != HeaderSize+4*CodebookLen+681*RecordSize {
		t.Errorf("record 0 = %+v, want CQ4 (8199, 512) right after the directory", first)
	}

	for i, dims := range map[int][]int{226: {512}, 227: {512}, 237: {13}, 679: {257, 80}} {
		rec, _ := f.Record(i)
		if rec.DType != FP32 || !slices.Equal(rec.Dims(), dims) {
			t.Errorf("record %d = %s %v, want FP32 %v", i, rec.DType, rec.Dims(), dims)
		}
	}

	manifest, err := f.FP32(237)
	if err != nil {
		t.Fatalf("FP32(237): %v", err)
	}

	wantManifest := []float32{3246, 1, 8192, 8, 4, 9, 128, 3, 80, 16000, 512, 400, 160}
	if !slices.Equal(manifest, wantManifest) {
		t.Errorf("audio manifest = %v, want %v", manifest, wantManifest)
	}

	tok, err := f.Raw(680)
	if err != nil {
		t.Fatalf("Raw(680): %v", err)
	}

	if len(tok) != 109231 || binary.LittleEndian.Uint32(tok) != 8199 {
		t.Errorf("tokenizer blob: %d bytes, %d pieces; want 109231 bytes, 8199 pieces",
			len(tok), binary.LittleEndian.Uint32(tok))
	}

	for i := range f.Len() {
		rec, _ := f.Record(i)
		if rec.DType != FP16 && rec.DType != FP32 {
			continue
		}

		vals, err := f.Float32(i)
		if err != nil {
			t.Fatalf("Float32(%d): %v", i, err)
		}

		if len(vals) != rec.Numel() {
			t.Errorf("Float32(%d): %d values, want %d", i, len(vals), rec.Numel())
		}

		for _, v := range vals {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("record %d (%s) holds %v", i, rec.DType, v)

				break
			}
		}
	}
}

// TestWhistleCactMatchesConfig maps the .cact header onto the safetensors
// config of the same checkpoint (docs/FINDINGS.md F-003).
func TestWhistleCactMatchesConfig(t *testing.T) {
	f := openWhistle(t)

	_, err := os.Stat(whistleSafetensorsPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", whistleSafetensorsPath)
	}

	store, err := safetensors.OpenStore(whistleSafetensorsPath, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	cfg, err := store.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}

	run, err := store.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	h := f.Header()
	tables := len(cfg.Orders) * cfg.Hashes

	for _, c := range []struct {
		field     string
		got, want int
	}{
		{"vocab = text + 7 languages", int(h.Vocab), cfg.Text + 7},
		{"d_model = dim", int(h.DModel), cfg.Dim},
		{"num_kv_heads = kv_heads", int(h.NumKVHeads), cfg.KVHeads},
		{"num_layers = dec_layers", int(h.NumLayers), cfg.DecLayers},
		{"qk_head_dim = qk_dim", int(h.QKHeadDim), cfg.QKDim},
		{"v_head_dim = v_dim", int(h.VHeadDim), cfg.VDim},
		{"max_seq_len = max_tokens", int(h.MaxSeqLen), cfg.MaxTokens},
		{"hada_n = next pow2 of dim", int(h.HadaN), 1 << bits.Len(uint(cfg.Dim-1))},
		{"mhc_lanes = lanes", int(h.MHCLanes), cfg.Lanes},
		{"qkv_conv_taps = taps", int(h.QKVConvTaps), cfg.Taps},
		{"engram_slots = slots", int(h.EngramSlots), cfg.Slots},
		{"num_engram_tables = orders * hashes", int(h.NumEngramTables), tables},
		{"engram_sub_dim = dim / tables", int(h.EngramSubDim), cfg.Dim / tables},
		{"engram_conv_dilation = max(orders)", int(h.EngramConvDilation), slices.Max(cfg.Orders)},
		{"kv_bits = run.kv_bits", int(h.KVBits), run.KVBits},
	} {
		if c.got != c.want {
			t.Errorf("%s: header %d, config %d", c.field, c.got, c.want)
		}
	}

	if !slices.Equal(h.Orders(), cfg.Orders) || !slices.Equal(h.Sites(), cfg.Sites) {
		t.Errorf("orders/sites: header %v/%v, config %v/%v", h.Orders(), h.Sites(), cfg.Orders, cfg.Sites)
	}

	if float64(h.RopeTheta) != cfg.Rope {
		t.Errorf("rope_theta: header %v, config %v", h.RopeTheta, cfg.Rope)
	}

	// The encoder and front-end geometry is not in the header but in the
	// audio manifest, tensor 237. Its first two values are still unexplained.
	manifest, err := f.FP32(237)
	if err != nil {
		t.Fatalf("FP32(237): %v", err)
	}

	want := []float32{
		float32(cfg.Text), float32(cfg.EncLayers), float32(cfg.EncLanes), float32(cfg.ConvKernel),
		float32(cfg.Stem), float32(cfg.StemStages), float32(cfg.Mels), float32(cfg.Rate),
		float32(cfg.FFT), float32(cfg.Win), float32(cfg.Hop),
	}
	if !slices.Equal(manifest[2:], want) {
		t.Errorf("manifest[2:] = %v, want config text..hop %v", manifest[2:], want)
	}
}
