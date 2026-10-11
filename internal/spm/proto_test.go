package spm

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

// pbuf builds protobuf wire format by hand.
type pbuf []byte

func (b pbuf) key(num, wire int) pbuf {
	return binary.AppendUvarint(b, uint64(num)<<3|uint64(wire))
}

func (b pbuf) varint(num int, v uint64) pbuf { return binary.AppendUvarint(b.key(num, wireVarint), v) }

func (b pbuf) int32(num int, v int32) pbuf { return b.varint(num, uint64(int64(v))) }

func (b pbuf) bool(num int, v bool) pbuf {
	if v {
		return b.varint(num, 1)
	}

	return b.varint(num, 0)
}

func (b pbuf) bytes(num int, v []byte) pbuf {
	b = binary.AppendUvarint(b.key(num, wireBytes), uint64(len(v)))

	return append(b, v...)
}

func (b pbuf) str(num int, s string) pbuf { return b.bytes(num, []byte(s)) }

func (b pbuf) float(num int, f float32) pbuf {
	return binary.LittleEndian.AppendUint32(b.key(num, wireFixed32), math.Float32bits(f))
}

// testPiece is one vocabulary entry of a test model.
type testPiece struct {
	piece string
	score float32
	typ   Type
}

// testModel is a ModelProto to serialise.
type testModel struct {
	pieces     []testPiece
	modelType  ModelType
	byteFB     bool
	unkSurface *string
	normalizer NormalizerSpec
	extra      func(trainer, normalizer pbuf) (pbuf, pbuf) // appends raw fields
}

func (m testModel) encode() []byte {
	var out pbuf

	for _, p := range m.pieces {
		var pc pbuf

		pc = pc.str(1, p.piece).float(2, p.score)
		if p.typ != TypeNormal {
			pc = pc.varint(3, uint64(p.typ))
		}

		out = out.bytes(1, pc)
	}

	var ts pbuf

	ts = ts.int32(3, int32(m.modelType)).int32(4, int32(len(m.pieces))).bool(35, m.byteFB)
	ts = ts.int32(40, 3).int32(41, 2).int32(42, 1).int32(43, 0)

	if m.unkSurface != nil {
		ts = ts.str(44, *m.unkSurface)
	}

	var ns pbuf

	ns = ns.str(1, m.normalizer.Name).bytes(2, m.normalizer.PrecompiledCharsmap)
	ns = ns.bool(3, m.normalizer.AddDummyPrefix).bool(4, m.normalizer.RemoveExtraWhitespaces)
	ns = ns.bool(5, m.normalizer.EscapeWhitespaces)

	if m.extra != nil {
		ts, ns = m.extra(ts, ns)
	}

	return out.bytes(2, ts).bytes(3, ns)
}

// testVocab is a tiny BPE vocabulary in sentencepiece's layout: specials,
// user-defined symbols, the 256 byte pieces, then NORMAL pieces whose score
// is minus their merge rank, except where a test needs a tie or an UNUSED
// piece.
func testVocab() []testPiece {
	pieces := []testPiece{
		{"<pad>", 0, TypeControl},
		{"</s>", 0, TypeControl},
		{"<s>", 0, TypeControl},
		{"<unk>", 0, TypeUnknown},
		{"<b>", 0, TypeUserDefined},
		{"<b>x", 0, TypeUserDefined},
		{"<speech>", 0, TypeUserDefined},
	}

	for b := range 256 {
		pieces = append(pieces, testPiece{fmt.Sprintf("<0x%02X>", b), 0, TypeByte})
	}

	normals := []string{
		"hi", "▁h", "▁hi", "ab", "bc", "abc", "s>", "<b", "▁", "a", "b", "c", "h", "i", "x", "y", "z", "<", ">", "s",
	}
	for rank, p := range normals {
		pieces = append(pieces, testPiece{p, -float32(rank), TypeNormal})
	}

	return append(pieces,
		// "aa" and "ad" tie: on "aad" the leftmost pair merges first.
		testPiece{"aa", -30, TypeNormal},
		testPiece{"ad", -30, TypeNormal},
		testPiece{"d", -31, TypeNormal},
		// "xy" is UNUSED: it may merge on the way to "xyz", but is split
		// again when it is a final symbol.
		testPiece{"xy", 1, TypeUnused},
		testPiece{"xyz", -32, TypeNormal},
		// "ef" scores -0 and "fg" +0: sentencepiece v0.2.2 ranks +0 higher.
		testPiece{"e", -33, TypeNormal},
		testPiece{"f", -34, TypeNormal},
		testPiece{"g", -35, TypeNormal},
		testPiece{"ef", float32(math.Copysign(0, -1)), TypeNormal},
		testPiece{"fg", 0, TypeNormal},
	)
}

func identity() NormalizerSpec {
	return NormalizerSpec{Name: "identity", EscapeWhitespaces: true}
}

func defaultTestModel() testModel {
	return testModel{pieces: testVocab(), modelType: ModelBPE, byteFB: true, normalizer: identity()}
}

// newTest parses m and builds a Processor.
func newTest(t *testing.T, m testModel) *Processor {
	t.Helper()

	p, err := Parse(m.encode())
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	return p
}

func TestParseModelProto(t *testing.T) {
	m := defaultTestModel()
	m.extra = func(ts, ns pbuf) (pbuf, pbuf) {
		ts = ts.int32(4, 7) // a repeated scalar: the last one wins
		ts = ts.str(31, "<b>").str(31, "<b>x").str(30, "<ctl>")
		ts = ts.varint(999, 5).bytes(998, []byte("skipped"))
		ts = binary.LittleEndian.AppendUint64(ts.key(997, wireFixed64), 1)

		return ts, ns.str(6, "rule\tvalue")
	}

	data := pbuf(m.encode())
	// A second trainer_spec merges into the first; a denormalizer and an
	// unknown top-level field are read and skipped.
	data = data.bytes(2, pbuf(nil).bool(25, true).int32(43, -1)).bytes(5, pbuf(nil).str(1, "denorm")).varint(77, 1)

	got, err := ParseModelProto(data)
	if err != nil {
		t.Fatalf("ParseModelProto: %v", err)
	}

	vocab := testVocab()
	if len(got.Pieces) != len(vocab) {
		t.Fatalf("%d pieces, want %d", len(got.Pieces), len(vocab))
	}

	for i, p := range vocab {
		g := got.Pieces[i]
		if g.Piece != p.piece || math.Float32bits(g.Score) != math.Float32bits(p.score) || g.Type != p.typ {
			t.Errorf("piece %d = (%q, %v, %s), want (%q, %v, %s)", i, g.Piece, g.Score, g.Type, p.piece, p.score, p.typ)
		}
	}

	ts := got.Trainer
	if ts.ModelType != ModelBPE || ts.VocabSize != 7 || !ts.ByteFallback || !ts.SplitDigits {
		t.Errorf("trainer type, vocab size, byte fallback, split digits = %s, %d, %v, %v; want BPE, 7, true, true",
			ts.ModelType, ts.VocabSize, ts.ByteFallback, ts.SplitDigits)
	}

	if ts.UnkID != 3 || ts.BOSID != 2 || ts.EOSID != 1 || ts.PadID != -1 {
		t.Errorf("trainer unk/bos/eos/pad = %d/%d/%d/%d, want 3/2/1/-1", ts.UnkID, ts.BOSID, ts.EOSID, ts.PadID)
	}

	// Absent fields keep the proto2 defaults.
	if !ts.SplitByWhitespace || ts.MaxSentencePieceLength != 16 || ts.UnkSurface != " ⁇ " || ts.UnkPiece != "<unk>" {
		t.Errorf("trainer defaults: split_by_whitespace %v, max length %d, unk_surface %q, unk_piece %q",
			ts.SplitByWhitespace, ts.MaxSentencePieceLength, ts.UnkSurface, ts.UnkPiece)
	}

	if !slices.Equal(ts.UserDefinedSymbols, []string{"<b>", "<b>x"}) || !slices.Equal(ts.ControlSymbols, []string{"<ctl>"}) {
		t.Errorf("user-defined %q, control %q", ts.UserDefinedSymbols, ts.ControlSymbols)
	}

	ns := got.Normalizer
	if ns.Name != "identity" || ns.AddDummyPrefix || ns.RemoveExtraWhitespaces || !ns.EscapeWhitespaces ||
		ns.NormalizationRuleTSV != "rule\tvalue" || len(ns.PrecompiledCharsmap) != 0 {
		t.Errorf("normalizer = %+v", ns)
	}

	if !got.HasDenormalizer || got.Denormalizer.Name != "denorm" || !got.Denormalizer.AddDummyPrefix {
		t.Errorf("denormalizer = %v, %+v; want present, named, with defaults", got.HasDenormalizer, got.Denormalizer)
	}
}

func TestParseModelProtoDefaults(t *testing.T) {
	got, err := ParseModelProto(nil)
	if err != nil {
		t.Fatalf("ParseModelProto(nil): %v", err)
	}

	want := ModelProto{Trainer: defaultTrainerSpec(), Normalizer: defaultNormalizerSpec(), Denormalizer: defaultNormalizerSpec()}
	if fmt.Sprintf("%+v", *got) != fmt.Sprintf("%+v", want) {
		t.Errorf("empty proto = %+v, want %+v", *got, want)
	}

	if got.Trainer.ModelType != ModelUnigram || got.Trainer.PadID != -1 || !got.Normalizer.AddDummyPrefix ||
		!got.Normalizer.RemoveExtraWhitespaces || !got.Normalizer.EscapeWhitespaces {
		t.Errorf("defaults = %+v / %+v", got.Trainer, got.Normalizer)
	}
}

func TestParseModelProtoRejects(t *testing.T) {
	piece := func(fields pbuf) []byte { return pbuf(nil).bytes(1, fields) }

	tests := []struct {
		name string
		data []byte
		want string
	}{
		{"truncated key", []byte{0x80}, "truncated"},
		{"truncated varint", pbuf(nil).key(77, wireVarint), "truncated"},
		{"truncated length", append(pbuf(nil).key(1, wireBytes), 5, 'a'), "truncated"},
		{"truncated fixed32", append(pbuf(nil).key(77, wireFixed32), 1, 2), "truncated"},
		{"truncated fixed64", append(pbuf(nil).key(77, wireFixed64), 1, 2, 3), "truncated"},
		{"group", pbuf(nil).key(77, 3), "unsupported wire type 3"},
		{"field zero", pbuf(nil).varint(0, 1), "invalid field number 0"},
		{"piece as varint", pbuf(nil).varint(1, 1), "piece 0: wire type 0"},
		{"piece string as varint", piece(pbuf(nil).varint(1, 1)), "piece 0: wire type 0"},
		{"score as varint", piece(pbuf(nil).varint(2, 1)), "piece 0: wire type 0, want 5"},
		{"invalid UTF-8 piece", piece(pbuf(nil).str(1, "a\xff")), "not valid UTF-8"},
		{"unknown type", piece(pbuf(nil).str(1, "a").varint(3, 7)), "unknown type 7"},
		{"type zero", piece(pbuf(nil).str(1, "a").varint(3, 0)), "unknown type 0"},
		{"trainer as varint", pbuf(nil).varint(2, 1), "trainer_spec: wire type 0"},
		{"bad trainer field", pbuf(nil).bytes(2, pbuf(nil).str(35, "x")), "trainer_spec: wire type 2"},
		{"charsmap as varint", pbuf(nil).bytes(3, pbuf(nil).varint(2, 1)), "normalizer_spec: field 2: wire type 0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseModelProto(tt.data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestNewRejects(t *testing.T) {
	withPieces := func(edit func([]testPiece) []testPiece) testModel {
		m := defaultTestModel()
		m.pieces = edit(m.pieces)

		return m
	}
	replace := func(i int, p testPiece) testModel {
		return withPieces(func(v []testPiece) []testPiece { v[i] = p; return v })
	}

	unigram := defaultTestModel()
	unigram.modelType = ModelUnigram

	charsmap := defaultTestModel()
	charsmap.normalizer.Name = "nmt_nfkc"
	charsmap.normalizer.PrecompiledCharsmap = []byte{1, 2, 3}

	suffix := defaultTestModel()
	suffix.extra = func(ts, ns pbuf) (pbuf, pbuf) { return ts.bool(24, true), ns }

	noFallback := defaultTestModel()
	noFallback.byteFB = false

	tests := []struct {
		name string
		data []byte
		want string
	}{
		{"unigram", unigram.encode(), "model type UNIGRAM is not supported"},
		{"charsmap", charsmap.encode(), `normalizer "nmt_nfkc" has a precompiled charsmap`},
		{"whitespace suffix", suffix.encode(), "treat_whitespace_as_suffix"},
		{"denormalizer charsmap", pbuf(defaultTestModel().encode()).bytes(5, pbuf(nil).bytes(2, []byte{1})), "denormalizer"},
		{"empty piece", replace(20, testPiece{"", 0, TypeNormal}).encode(), "piece 20 is empty"},
		{"NUL piece", replace(20, testPiece{"a\x00", 0, TypeNormal}).encode(), "contains NUL"},
		{"long piece", replace(20, testPiece{strings.Repeat("a", maxPieceSize), 0, TypeNormal}).encode(), "8000 bytes"},
		{"repeated piece", replace(1, testPiece{"<pad>", 0, TypeControl}).encode(), `piece 1 "<pad>" repeats piece 0`},
		{"second unk", replace(1, testPiece{"<unk2>", 0, TypeUnknown}).encode(), "second UNKNOWN"},
		{"no unk", replace(3, testPiece{"<unk>", 0, TypeControl}).encode(), "no UNKNOWN piece"},
		{"byte without fallback", noFallback.encode(), "byte_fallback is off"},
		{"lower-case byte piece", replace(7+0xAB, testPiece{"<0xab>", 0, TypeByte}).encode(), `malformed byte piece "<0xab>"`},
		{"missing byte piece", replace(7+0x41, testPiece{"A", 0, TypeNormal}).encode(), "<0x41> is missing"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(tt.data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestPieceToByte(t *testing.T) {
	for b := range 256 {
		got, ok := pieceToByte(fmt.Sprintf("<0x%02X>", b))
		if !ok || int(got) != b {
			t.Errorf("pieceToByte(<0x%02X>) = %d, %v", b, got, ok)
		}
	}

	for _, s := range []string{"<0xab>", "<0x4>", "<0x041>", "0x41>", "<0x41", "<0X41>", "<0xG1>", "<0x4g>"} {
		if _, ok := pieceToByte(s); ok {
			t.Errorf("pieceToByte(%q) accepted", s)
		}
	}
}
