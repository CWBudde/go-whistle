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

// TestWhistleCQ dequantises every CQ tensor of whistle.cact @ b358ddad. The
// pinned values come from export.py _cq_unpack run in numpy on the same file
// (docs/FINDINGS.md F-004). Both sides are float32 but round differently:
// numpy's matmul sums a group in up to 128 steps, so they agree to
// (128 + log2(128) + 3) * 2^-24 < 1e-5 of the group magnitude, which is
// the tolerance relative to the row norm.
func TestWhistleCQ(t *testing.T) {
	f := openWhistle(t)

	for width := 2; width <= 4; width++ {
		got, err := f.Codebook(width)
		if err != nil || !slices.Equal(got, portCodebook(width, 128)) {
			t.Errorf("Codebook(%d) = %v, %v; want _cq_codebook_np(%d, 128) = %v",
				width, got, err, width, portCodebook(width, 128))
		}
	}

	type pin struct {
		dims     []int
		rowNorms map[int]float64
		first4   []float64 // w[0][:4]
		last     float64   // w[-1][-1]
	}

	pins := map[int]pin{
		0: { // the CQ4 token embedding
			dims:     []int{8199, 512},
			rowNorms: map[int]float64{0: 186.81284537308602, 1: 57.53578797611011, 4099: 71.17012258767335, 8198: 78.4586356644421},
			first4:   []float64{-0.3075689673423767, -14.557136535644531, 3.948115110397339, 5.2327423095703125},
			last:     0.017739592120051384,
		},
		2: { // the first CQ2 tensor
			dims:     []int{384, 512},
			rowNorms: map[int]float64{0: 6.372649275450949, 1: 5.228630641170833, 192: 8.429754060368738, 383: 9.550146851717102},
			first4:   []float64{0.2964808940887451, -0.6099874973297119, 0.649564266204834, -0.015735018998384476},
			last:     0.24696075916290283,
		},
		671: { // the last CQ4 tensor, 16 groups per row
			dims:     []int{128, 2048},
			rowNorms: map[int]float64{0: 47.7150872785567, 1: 22.116285867410525, 64: 35.72595426974767, 127: 25.951508096145528},
			first4:   []float64{2.292008876800537, -1.0804028511047363, -1.638489842414856, -0.6228991150856018},
			last:     0.9282915592193604,
		},
	}

	const tol = 1e-5

	count := map[uint32]int{}
	maxAbs := 0.0

	for i := range f.Len() {
		rec, _ := f.Record(i)
		if rec.DType != CQ {
			continue
		}

		count[rec.Bits]++

		w, err := f.CQ(i)
		if err != nil {
			t.Fatalf("CQ(%d): %v", i, err)
		}

		if len(w) != rec.Numel() {
			t.Fatalf("CQ(%d): %d values, want %d", i, len(w), rec.Numel())
		}

		var sq, tensorMax float64

		for _, v := range w {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("CQ(%d) holds %v", i, v)
			}

			sq += float64(v) * float64(v)
			tensorMax = math.Max(tensorMax, math.Abs(float64(v)))
		}

		// Sane magnitudes: no dead tensor, nothing near the FP16 norm range.
		if rms := math.Sqrt(sq / float64(len(w))); rms < 1e-3 || rms > 10 || tensorMax > 100 {
			t.Errorf("CQ(%d) %v: rms %.4g, max |w| %.4g", i, rec.Dims(), rms, tensorMax)
		}

		maxAbs = math.Max(maxAbs, tensorMax)

		p, ok := pins[i]
		if !ok {
			continue
		}

		if !slices.Equal(rec.Dims(), p.dims) {
			t.Fatalf("CQ(%d) dims %v, want %v", i, rec.Dims(), p.dims)
		}

		out, in := p.dims[0], p.dims[1]

		for r, want := range p.rowNorms {
			var rowSq float64
			for _, v := range w[r*in : (r+1)*in] {
				rowSq += float64(v) * float64(v)
			}

			if got := math.Sqrt(rowSq); math.Abs(got-want) > tol*want {
				t.Errorf("CQ(%d) |row %d| = %.9g, want %.9g", i, r, got, want)
			}
		}

		values := map[int]float64{(out-1)*in + in - 1: p.last}
		for c, v := range p.first4 {
			values[c] = v
		}

		for at, want := range values {
			norm := p.rowNorms[at/in]
			if diff := math.Abs(float64(w[at]) - want); diff > tol*norm {
				t.Errorf("CQ(%d) w[%d][%d] = %.9g, want %.9g", i, at/in, at%in, w[at], want)
			}
		}
	}

	if count[2] != 145 || count[4] != 7 || len(count) != 2 {
		t.Errorf("CQ tensors by width = %v, want 145 CQ2 and 7 CQ4", count)
	}

	if want := 69.29121398925781; math.Abs(maxAbs-want) > tol*want {
		t.Errorf("max |w| over all CQ tensors = %.9g, want %.9g", maxAbs, want)
	}
}

// TestWhistleCQMatchesSafetensors compares the dequantised CQ4 embedding,
// tensor 0, with the F32 embedding/embedding of the same checkpoint
// (docs/FINDINGS.md F-004).
func TestWhistleCQMatchesSafetensors(t *testing.T) {
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

	emb, err := store.TensorWithShape("embedding/embedding", []int64{8199, 512})
	if err != nil {
		t.Fatalf("embedding/embedding: %v", err)
	}

	w, err := f.CQ(0)
	if err != nil {
		t.Fatalf("CQ(0): %v", err)
	}

	cos := func(a, b []float32) float64 {
		var ab, aa, bb float64
		for k := range a {
			ab += float64(a[k]) * float64(b[k])
			aa += float64(a[k]) * float64(a[k])
			bb += float64(b[k]) * float64(b[k])
		}

		return ab / math.Sqrt(aa*bb)
	}

	// numpy on the same files: 0.9953722094069657 overall, rows >= 0.9898033582667012.
	if got := cos(w, emb.Data); math.Abs(got-0.99537221) > 1e-6 {
		t.Errorf("cosine(CQ(0), embedding/embedding) = %.8f, want 0.99537221", got)
	}

	worst := 1.0
	for r := range 8199 {
		worst = min(worst, cos(w[r*512:(r+1)*512], emb.Data[r*512:(r+1)*512]))
	}

	if math.Abs(worst-0.98980336) > 1e-6 {
		t.Errorf("worst row cosine = %.8f, want 0.98980336", worst)
	}
}
