package cact

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

// testTensor is one directory entry plus its blob for encodeTestFile.
type testTensor struct {
	dtype       DType
	shape       []uint32
	blob        []byte
	group, bits uint32
}

// testHeader is a small but complete geometry; encodeTestFile fills
// NumTensors and CodebookLen.
func testHeader() Header {
	return Header{
		Tag: Tag, KVBits: 8, Vocab: 16, DModel: 8, NumHeads: 2, NumKVHeads: 1, NumLayers: 3,
		QKHeadDim: 4, VHeadDim: 4, MaxSeqLen: 32, HadaN: 8, MHCLanes: 4,
		GlobalMaskLo: 0b101, GlobalMaskHi: 0, QKVConvTaps: 3,
		EngramSlots: 64, EngramSubDim: 2, NumEngramTables: 4, EngramConvTaps: 4, EngramConvDilation: 3,
		NumEngramOrders: 2, EngramOrders: [4]uint32{2, 3},
		NumEngramSites: 1, EngramSites: [16]uint32{1},
		RopeTheta: 10000,
	}
}

func testCodebook() []float32 {
	cb := make([]float32, CodebookLen)
	for i := range cb {
		cb[i] = float32(i) - 14
	}

	return cb
}

// encodeTestFile lays a .cact file out the way export.py _pack_cact does:
// header, codebook, directory, then each blob at the next 64-byte boundary
// with zero padding in between.
func encodeTestFile(t *testing.T, h Header, cb []float32, tensors []testTensor) []byte {
	t.Helper()

	h.NumTensors = uint32(len(tensors))
	h.CodebookLen = uint32(len(cb))

	var buf bytes.Buffer

	err := binary.Write(&buf, binary.LittleEndian, h)
	if err != nil {
		t.Fatalf("write header: %v", err)
	}

	err = binary.Write(&buf, binary.LittleEndian, cb)
	if err != nil {
		t.Fatalf("write codebook: %v", err)
	}

	pos := uint64(buf.Len() + len(tensors)*RecordSize)
	records := make([]Record, len(tensors))

	for i, tt := range tensors {
		pos = (pos + Align - 1) &^ (Align - 1)
		rec := Record{
			DType: tt.dtype, NDim: uint8(len(tt.shape)), Offset: pos, NBytes: uint64(len(tt.blob)),
			Group: tt.group, Bits: tt.bits,
		}
		copy(rec.Shape[:], tt.shape)
		records[i] = rec
		pos += uint64(len(tt.blob))
	}

	err = binary.Write(&buf, binary.LittleEndian, records)
	if err != nil {
		t.Fatalf("write directory: %v", err)
	}

	for i, tt := range tensors {
		buf.Write(make([]byte, int(records[i].Offset)-buf.Len()))
		buf.Write(tt.blob)
	}

	return buf.Bytes()
}

func f32Blob(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}

	return out
}

func f16Blob(bits ...uint16) []byte {
	out := make([]byte, 2*len(bits))
	for i, b := range bits {
		binary.LittleEndian.PutUint16(out[2*i:], b)
	}

	return out
}

// testTensors covers every dtype: FP32 (2, 3), FP16 (3), an FP16 scalar, CQ2
// (2, 200) with group 128 (in_pad 256), CQ ternary (1, 128) and RAW.
func testTensors() []testTensor {
	return []testTensor{
		{dtype: FP32, shape: []uint32{2, 3}, blob: f32Blob(1, 2, 3, 4, 5, -6)},
		{dtype: FP16, shape: []uint32{3}, blob: f16Blob(0x3c00, 0xc000, 0x3800)},
		{dtype: FP16, shape: nil, blob: f16Blob(0x4400)},
		{dtype: CQ, shape: []uint32{2, 200}, blob: make([]byte, 2*256*2/8+2*2*2), group: 128, bits: 2},
		{dtype: CQ, shape: []uint32{1, 128}, blob: make([]byte, 128*2/8+2), group: 128, bits: 5},
		{dtype: RAW, blob: []byte("tokenizer")},
	}
}

func parseTestFile(t *testing.T) *File {
	t.Helper()

	f, err := Parse(encodeTestFile(t, testHeader(), testCodebook(), testTensors()))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	return f
}

func TestLayoutSizes(t *testing.T) {
	if got := binary.Size(Header{}); got != HeaderSize {
		t.Errorf("binary.Size(Header) = %d, want %d", got, HeaderSize)
	}

	if got := binary.Size(Record{}); got != RecordSize {
		t.Errorf("binary.Size(Record) = %d, want %d", got, RecordSize)
	}
}

func TestParse_Header(t *testing.T) {
	f := parseTestFile(t)

	want := testHeader()
	want.NumTensors = 6
	want.CodebookLen = CodebookLen

	if got := f.Header(); got != want {
		t.Errorf("Header() =\n%+v\nwant\n%+v", got, want)
	}

	if f.Len() != 6 {
		t.Errorf("Len() = %d, want 6", f.Len())
	}

	h := f.Header()
	if got := h.GlobalLayers(); !reflect.DeepEqual(got, []int{0, 2}) {
		t.Errorf("GlobalLayers() = %v, want [0 2]", got)
	}

	if got := h.Orders(); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Errorf("Orders() = %v, want [2 3]", got)
	}

	if got := h.Sites(); !reflect.DeepEqual(got, []int{1}) {
		t.Errorf("Sites() = %v, want [1]", got)
	}
}

func TestHeader_GlobalLayersHighWord(t *testing.T) {
	h := Header{NumLayers: 40, GlobalMaskLo: 1 << 31, GlobalMaskHi: 1 << 3}
	if got := h.GlobalLayers(); !reflect.DeepEqual(got, []int{31, 35}) {
		t.Errorf("GlobalLayers() = %v, want [31 35]", got)
	}
}

func TestParse_Codebook(t *testing.T) {
	f := parseTestFile(t)
	cb := testCodebook()

	for _, tc := range []struct {
		bits     int
		from, to int
	}{{2, 0, 4}, {3, 4, 12}, {4, 12, 28}} {
		got, err := f.Codebook(tc.bits)
		if err != nil {
			t.Fatalf("Codebook(%d): %v", tc.bits, err)
		}

		if !reflect.DeepEqual(got, cb[tc.from:tc.to]) {
			t.Errorf("Codebook(%d) = %v, want %v", tc.bits, got, cb[tc.from:tc.to])
		}

		got[0] = 99

		again, _ := f.Codebook(tc.bits)
		if again[0] == 99 {
			t.Errorf("Codebook(%d) returned the file's own slice", tc.bits)
		}
	}

	for _, bits := range []int{0, 1, 5, 8} {
		_, err := f.Codebook(bits)
		if err == nil {
			t.Errorf("Codebook(%d) succeeded", bits)
		}
	}
}

func TestParse_Records(t *testing.T) {
	f := parseTestFile(t)

	rec, err := f.Record(3)
	if err != nil {
		t.Fatalf("Record(3): %v", err)
	}

	if rec.DType != CQ || rec.Group != 128 || rec.Bits != 2 || !reflect.DeepEqual(rec.Dims(), []int{2, 200}) {
		t.Errorf("Record(3) = %+v", rec)
	}

	if rec.Offset%Align != 0 {
		t.Errorf("Record(3).Offset = %d, not %d-aligned", rec.Offset, Align)
	}

	if n := rec.Numel(); n != 400 {
		t.Errorf("Record(3).Numel() = %d, want 400", n)
	}

	scalar, _ := f.Record(2)
	if len(scalar.Dims()) != 0 || scalar.Numel() != 1 {
		t.Errorf("scalar record: dims %v, numel %d", scalar.Dims(), scalar.Numel())
	}

	for _, i := range []int{-1, 6} {
		_, err := f.Record(i)
		if err == nil {
			t.Errorf("Record(%d) succeeded", i)
		}
	}
}

func TestReaders(t *testing.T) {
	f := parseTestFile(t)

	got, err := f.FP32(0)
	if err != nil || !reflect.DeepEqual(got, []float32{1, 2, 3, 4, 5, -6}) {
		t.Errorf("FP32(0) = %v, %v", got, err)
	}

	got, err = f.FP16(1)
	if err != nil || !reflect.DeepEqual(got, []float32{1, -2, 0.5}) {
		t.Errorf("FP16(1) = %v, %v", got, err)
	}

	got, err = f.FP16(2)
	if err != nil || !reflect.DeepEqual(got, []float32{4}) {
		t.Errorf("FP16(2) = %v, %v", got, err)
	}

	for i, want := range map[int][]float32{0: {1, 2, 3, 4, 5, -6}, 1: {1, -2, 0.5}} {
		got, err := f.Float32(i)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("Float32(%d) = %v, %v; want %v", i, got, err, want)
		}
	}

	raw, err := f.Raw(5)
	if err != nil || string(raw) != "tokenizer" {
		t.Errorf("Raw(5) = %q, %v", raw, err)
	}

	raw[0] = 'X'

	again, _ := f.Raw(5)
	if string(again) != "tokenizer" {
		t.Error("Raw returned the file's own bytes")
	}
}

func TestReaders_WrongDType(t *testing.T) {
	f := parseTestFile(t)

	for name, read := range map[string]func() error{
		"FP32(FP16)":   func() error { _, err := f.FP32(1); return err },
		"FP16(FP32)":   func() error { _, err := f.FP16(0); return err },
		"Float32(CQ)":  func() error { _, err := f.Float32(3); return err },
		"Float32(RAW)": func() error { _, err := f.Float32(5); return err },
		"Raw(FP32)":    func() error { _, err := f.Raw(0); return err },
	} {
		err := read()
		if !errors.Is(err, ErrDType) {
			t.Errorf("%s: err = %v, want ErrDType", name, err)
		}
	}

	_, err := f.FP32(99)
	if err == nil || errors.Is(err, ErrDType) {
		t.Errorf("FP32(99): err = %v, want an index error", err)
	}
}

// recordAt is the byte offset of directory entry i in a file encoded with
// the full test codebook.
func recordAt(i int) int { return HeaderSize + 4*CodebookLen + i*RecordSize }

// Field offsets inside a record (<BBHIIIIQQII).
const (
	recDType  = 0
	recNDim   = 1
	recPad    = 2
	recShape  = 4
	recOffset = 20
	recNBytes = 28
	recGroup  = 36
	recBits   = 40
)

func TestParse_Errors(t *testing.T) {
	valid := encodeTestFile(t, testHeader(), testCodebook(), testTensors())

	put32 := func(b []byte, at int, v uint32) { binary.LittleEndian.PutUint32(b[at:], v) }
	put64 := func(b []byte, at int, v uint64) { binary.LittleEndian.PutUint64(b[at:], v) }
	recOff := func(b []byte, i int) uint64 { return binary.LittleEndian.Uint64(b[recordAt(i)+recOffset:]) }

	for _, tc := range []struct {
		name   string
		mutate func(b []byte) []byte
		want   string
	}{
		{"short header", func(b []byte) []byte { return b[:HeaderSize-1] }, "header"},
		{"bad tag", func(b []byte) []byte { put32(b, 0, 0xdeadbeef); return b }, "tag"},
		{"codebook length", func(b []byte) []byte { put32(b, 8, 27); return b }, "codebook"},
		{"too many orders", func(b []byte) []byte { put32(b, 4*26, 5); return b }, "orders"},
		{"too many sites", func(b []byte) []byte { put32(b, 4*31, 17); return b }, "sites"},
		{"directory past EOF", func(b []byte) []byte { put32(b, 4, 1000); return b }, "directory"},
		// 97612894 * 44 = 2^32 + 40: in uint32 the directory would wrap to 40 bytes and pass the EOF check.
		{"directory size wraps uint32", func(b []byte) []byte { put32(b, 4, 97612894); return b }, "directory"},
		{"unknown dtype", func(b []byte) []byte { b[recordAt(0)+recDType] = 9; return b }, "dtype 9"},
		{"ndim > 4", func(b []byte) []byte { b[recordAt(0)+recNDim] = 5; return b }, "ndim"},
		{"dim past ndim", func(b []byte) []byte { put32(b, recordAt(1)+recShape+4, 7); return b }, "past ndim"},
		{"pad", func(b []byte) []byte { b[recordAt(0)+recPad] = 1; return b }, "pad"},
		{"misaligned", func(b []byte) []byte { put64(b, recordAt(0)+recOffset, recOff(b, 0)+4); return b }, "aligned"},
		{"inside directory", func(b []byte) []byte { put64(b, recordAt(0)+recOffset, 0); return b }, "overlaps"},
		{"overlap", func(b []byte) []byte { put64(b, recordAt(1)+recOffset, recOff(b, 0)); return b }, "overlaps"},
		{"blob past EOF", func(b []byte) []byte { return b[:len(b)-1] }, "past EOF"},
		{"FP32 nbytes", func(b []byte) []byte { put32(b, recordAt(0)+recShape, 3); return b }, "nbytes"},
		{"FP16 nbytes", func(b []byte) []byte { put64(b, recordAt(1)+recNBytes, 4); return b }, "nbytes"},
		{"RAW ndim", func(b []byte) []byte {
			b[recordAt(5)+recNDim] = 1
			put32(b, recordAt(5)+recShape, 9)

			return b
		}, "RAW"},
		{"CQ ndim", func(b []byte) []byte {
			b[recordAt(3)+recNDim] = 3
			put32(b, recordAt(3)+recShape+8, 1)

			return b
		}, "CQ"},
		{"CQ group 0", func(b []byte) []byte { put32(b, recordAt(3)+recGroup, 0); return b }, "group"},
		{"CQ group not /8", func(b []byte) []byte { put32(b, recordAt(3)+recGroup, 100); return b }, "group"},
		{"CQ bits 6", func(b []byte) []byte { put32(b, recordAt(3)+recBits, 6); return b }, "bits"},
		{"CQ nbytes", func(b []byte) []byte { put32(b, recordAt(3)+recBits, 4); return b }, "nbytes"},
		{"numel overflow", func(b []byte) []byte {
			b[recordAt(0)+recNDim] = 4
			for k := range 4 {
				put32(b, recordAt(0)+recShape+4*k, math.MaxUint32)
			}

			return b
		}, "overflow"},
	} {
		data := tc.mutate(bytes.Clone(valid))

		_, err := Parse(data)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one mentioning %q", tc.name, err, tc.want)
		}
	}
}

func TestParse_Empty(t *testing.T) {
	f, err := Parse(encodeTestFile(t, testHeader(), testCodebook(), nil))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if f.Len() != 0 {
		t.Errorf("Len() = %d, want 0", f.Len())
	}
}

func TestOpen_Missing(t *testing.T) {
	_, err := Open(t.TempDir() + "/nope.cact")
	if err == nil {
		t.Fatal("Open of a missing file succeeded")
	}
}

func TestDTypeString(t *testing.T) {
	for d, want := range map[DType]string{FP16: "FP16", FP32: "FP32", CQ: "CQ", RAW: "RAW", 7: "dtype 7"} {
		if got := d.String(); got != want {
			t.Errorf("DType(%d).String() = %q, want %q", d, got, want)
		}
	}
}
