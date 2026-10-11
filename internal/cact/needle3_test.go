package cact

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// needle3CactPath is the pinned text model as laid out by `just fetch-needle3`.
const needle3CactPath = "../../models/needle3/needle3.cact"

const needle3NamesGolden = "testdata/needle3_names.txt"

func openNeedle3(t *testing.T) *File {
	t.Helper()

	_, err := os.Stat(needle3CactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model", needle3CactPath)
	}

	f, err := Open(needle3CactPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	return f
}

// TestNeedle3Cact pins needle3.cact @ 93e88ede, which the Whistle reader
// opens unchanged: the header, the codebooks (Whistle's, bit for bit), the
// directory layout and the records after final_norm (docs/FINDINGS.md
// F-045).
func TestNeedle3Cact(t *testing.T) {
	f := openNeedle3(t)

	wantHeader := Header{
		Tag: Tag, NumTensors: 581, CodebookLen: 28, KVWindow: 256, KVBits: 8,
		Vocab: 8192, OutVocab: 8192, DModel: 768, NumHeads: 12, NumKVHeads: 2, NumLayers: 20,
		QKHeadDim: 48, VHeadDim: 64, MaxSeqLen: 8192, HadaN: 1024, MHCLanes: 4,
		SlidingWindow: 1024, GlobalMaskLo: 1<<4 | 1<<9 | 1<<14 | 1<<19, GlobalMaskHi: 0, QKVConvTaps: 3,
		EngramSlots: 18432, EngramSubDim: 128, NumEngramTables: 6, EngramConvTaps: 4, EngramConvDilation: 3,
		EngramSeedHeads: 0, NumEngramOrders: 2, EngramOrders: [4]uint32{2, 3},
		NumEngramSites: 5, EngramSites: [16]uint32{3, 7, 11, 15, 19},
		RopeTheta: 100000,
	}
	if got := f.Header(); got != wantHeader {
		t.Errorf("Header() =\n%+v\nwant\n%+v", got, wantHeader)
	}

	if got := f.Header().GlobalLayers(); !slices.Equal(got, []int{4, 9, 14, 19}) {
		t.Errorf("GlobalLayers() = %v", got)
	}

	// The codebooks depend on the group size only, so they equal Whistle's.
	for width, want := range map[int][]float32{
		2: {-0.1331721395254135, -0.03990209102630615, 0.04003528133034706, 0.13346044719219208},
		4: {
			-0.23953154683113098, -0.18070605397224426, -0.14095793664455414, -0.10901255160570145,
			-0.08151296526193619, -0.056527651846408844, -0.03293915092945099, -0.010116109624505043,
			0.012710800394415855, 0.03570791706442833, 0.059458255767822266, 0.08497702330350876,
			0.112741619348526, 0.1446969211101532, 0.184239000082016, 0.2422737032175064,
		},
	} {
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
		{FP16, 0, 1}: 226, {FP16, 0, 2}: 228, {FP16, 0, 3}: 2,
		{FP32, 0, 1}: 2,
		{CQ, 2, 2}:   115, {CQ, 4, 2}: 7,
		{RAW, 0, 0}: 1,
	}
	if !reflect.DeepEqual(hist, wantHist) {
		t.Errorf("record histogram = %v, want %v", hist, wantHist)
	}

	if size := uint64(len(f.data)); end != size {
		t.Errorf("last blob ends at %d, file is %d bytes", end, size)
	}

	first, _ := f.Record(0)
	if first.DType != CQ || first.Bits != 4 || !slices.Equal(first.Dims(), []int{8192, 768}) ||
		first.Offset != (HeaderSize+4*CodebookLen+581*RecordSize+Align-1)/Align*Align {
		t.Errorf("record 0 = %+v, want CQ4 (8192, 768) at the first 64-byte boundary after the directory", first)
	}

	// heads.manifest right after final_norm: one head, code 2 (confidence).
	manifest, err := f.FP16(573)
	if err != nil || !slices.Equal(manifest, []float32{HeadConfidence}) {
		t.Errorf("heads.manifest = %v, %v; want [2]", manifest, err)
	}

	tok, err := f.Raw(580)
	if err != nil {
		t.Fatalf("Raw(580): %v", err)
	}

	sum := sha256.Sum256(tok)
	if len(tok) != 109140 || binary.LittleEndian.Uint32(tok) != 8192 ||
		hex.EncodeToString(sum[:]) != "9f839cd44027ea9203738897da2bac572a5cc77aec7fceacd4391fc412a3249f" {
		t.Errorf("tokenizer blob: %d bytes, %d pieces, sha256 %x; want 109140 bytes, 8192 pieces",
			len(tok), binary.LittleEndian.Uint32(tok), sum)
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

		for _, v := range vals {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("record %d (%s) holds %v", i, rec.DType, v)

				break
			}
		}
	}
}

// TestNeedle3NamesGolden pins TextNames for needle3.cact: every index's
// name, shape, transform and source.
func TestNeedle3NamesGolden(t *testing.T) {
	f := openNeedle3(t)

	es, err := TextNames(f)
	if err != nil {
		t.Fatalf("TextNames: %v", err)
	}

	if len(es) != f.Len() {
		t.Fatalf("%d entries for %d records", len(es), f.Len())
	}

	for i, e := range es {
		rec, _ := f.Record(i)
		if !slices.Equal(rec.Dims(), e.Shape) && (e.Shape != nil || rec.NDim != 0) {
			t.Errorf("%d %s: record %s %v, map shape %v", i, e.Name, rec.DType, rec.Dims(), e.Shape)
		}
	}

	// renderNames heads the table with Whistle's provenance; swap it.
	body := renderNames(es)
	body = body[bytes.IndexByte(body, '\n')+1:]
	body = body[bytes.IndexByte(body, '\n')+1:]
	got := append([]byte("# TextNames() for needle3.cact @ 93e88ede (docs/FINDINGS.md F-045).\n"+
		"# Regenerate with: go test ./internal/cact -run TestNeedle3NamesGolden -update\n"), body...)

	if *update {
		err := os.WriteFile(needle3NamesGolden, got, 0o644)
		if err != nil {
			t.Fatal(err)
		}
	}

	want, err := os.ReadFile(needle3NamesGolden)
	if err != nil {
		t.Fatalf("%v (run with -update to create it)", err)
	}

	if !bytes.Equal(got, want) {
		gl, wl := strings.Split(string(got), "\n"), strings.Split(string(want), "\n")
		for k := range min(len(gl), len(wl)) {
			if gl[k] != wl[k] {
				t.Fatalf("%s line %d:\n got %q\nwant %q", needle3NamesGolden, k+1, gl[k], wl[k])
			}
		}

		t.Fatalf("%s: %d lines, want %d", needle3NamesGolden, len(gl), len(wl))
	}
}

// TestTextNamesWhistleTrunk maps whistle.cact's decoder through TextNames'
// geometry: the trunk names equal WhistleNames' for every index the two
// share, which keeps internal/needle's loaders valid for both models.
func TestTextNamesWhistleTrunk(t *testing.T) {
	f := openWhistle(t)

	g, err := newTextGeom(f.Header())
	if err != nil {
		t.Fatal(err)
	}

	trunk := g.trunk()
	whistle := WhistleNames()

	for i, e := range trunk {
		w := whistle[i]
		if e.Name != w.Name || e.Source != w.Source || e.Layer != w.Layer || e.Transform != w.Transform ||
			!slices.Equal(e.Shape, w.Shape) {
			t.Errorf("index %d: text %+v, whistle %+v", i, e, w)
		}
	}
}
