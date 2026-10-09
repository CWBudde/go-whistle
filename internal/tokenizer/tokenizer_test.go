package tokenizer

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"slices"
	"strings"
	"testing"
)

// testPiece is one vocabulary entry for encodeTestBlob.
type testPiece struct {
	piece string
	score float32
	typ   Type
}

// testBlobHeader holds the header fields encodeTestBlob writes besides the
// piece count.
type testBlobHeader struct {
	pad, eos, bos, unk     uint32
	addDummy, byteFallback bool
}

func defaultHeader() testBlobHeader {
	return testBlobHeader{pad: 0, eos: 1, bos: 2, unk: 3, byteFallback: true}
}

// encodeTestBlob lays a tokenizer blob out the way export.py
// _tokenizer_blob does: header "<IIIIIBBH", then per piece "<fBH" and the
// UTF-8 surface.
func encodeTestBlob(t *testing.T, h testBlobHeader, pieces []testPiece) []byte {
	t.Helper()

	var buf bytes.Buffer

	hdr := struct {
		N, Pad, EOS, BOS, UNK uint32
		AddDummy, ByteFB      uint8
		Reserved              uint16
	}{
		N: uint32(len(pieces)), Pad: h.pad, EOS: h.eos, BOS: h.bos, UNK: h.unk,
		AddDummy: boolByte(h.addDummy), ByteFB: boolByte(h.byteFallback),
	}

	err := binary.Write(&buf, binary.LittleEndian, hdr)
	if err != nil {
		t.Fatalf("write header: %v", err)
	}

	for _, p := range pieces {
		rec := struct {
			Score float32
			Type  uint8
			Len   uint16
		}{p.score, uint8(p.typ), uint16(len(p.piece))}

		err = binary.Write(&buf, binary.LittleEndian, rec)
		if err != nil {
			t.Fatalf("write record %q: %v", p.piece, err)
		}

		buf.WriteString(p.piece)
	}

	return buf.Bytes()
}

func boolByte(b bool) uint8 {
	if b {
		return 1
	}

	return 0
}

// testVocab is a toy Whistle-shaped vocabulary: the four specials, three
// markers (two sharing a prefix), the 256 byte pieces, then BPE pieces whose
// score is minus their merge rank.
func testVocab() []testPiece {
	normals := []string{"hi", "▁h", "▁hi", "ab", "bc", "abc", "aa", "▁", "a", "b", "c", "h", "i", "x", "y", "<", ">", "s"}

	pieces := make([]testPiece, 0, 7+256+len(normals))
	pieces = append(pieces,
		testPiece{"<pad>", 0, TypeControl},
		testPiece{"</s>", 0, TypeControl},
		testPiece{"<s>", 0, TypeControl},
		testPiece{"<unk>", 0, TypeUnknown},
		testPiece{"<speech>", 0, TypeUserDefined},
		testPiece{"<b>", 0, TypeUserDefined},
		testPiece{"<b>x", 0, TypeUserDefined},
	)

	for b := range 256 {
		pieces = append(pieces, testPiece{fmt.Sprintf("<0x%02X>", b), 0, TypeByte})
	}

	for rank, p := range normals {
		pieces = append(pieces, testPiece{p, -float32(rank), TypeNormal})
	}

	return pieces
}

// testTokenizer parses testVocab under h.
func testTokenizer(t *testing.T, h testBlobHeader) *Tokenizer {
	t.Helper()

	tok, err := Parse(encodeTestBlob(t, h, testVocab()))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	return tok
}

// ids maps pieces of tok to their ids, failing on an unknown piece.
func ids(t *testing.T, tok *Tokenizer, pieces ...string) []int {
	t.Helper()

	out := make([]int, len(pieces))

	for i, p := range pieces {
		id, ok := tok.PieceID(p)
		if !ok {
			t.Fatalf("piece %q not in vocab", p)
		}

		out[i] = id
	}

	return out
}

// byteIDs maps raw bytes to the ids of their byte pieces in testVocab.
func byteIDs(bs ...byte) []int {
	out := make([]int, len(bs))
	for i, b := range bs {
		out[i] = 7 + int(b)
	}

	return out
}

func TestParse(t *testing.T) {
	h := defaultHeader()
	tok := testTokenizer(t, h)

	vocab := testVocab()
	if tok.Len() != len(vocab) {
		t.Fatalf("Len = %d, want %d", tok.Len(), len(vocab))
	}

	for id, p := range vocab {
		if tok.Piece(id) != p.piece || tok.Score(id) != p.score || tok.Type(id) != p.typ {
			t.Errorf("id %d = (%q, %v, %s), want (%q, %v, %s)",
				id, tok.Piece(id), tok.Score(id), tok.Type(id), p.piece, p.score, p.typ)
		}

		got, ok := tok.PieceID(p.piece)
		if !ok || got != id {
			t.Errorf("PieceID(%q) = %d, %v; want %d", p.piece, got, ok, id)
		}
	}

	if tok.PadID() != 0 || tok.EOSID() != 1 || tok.BOSID() != 2 || tok.UnkID() != 3 {
		t.Errorf("specials pad/eos/bos/unk = %d/%d/%d/%d, want 0/1/2/3",
			tok.PadID(), tok.EOSID(), tok.BOSID(), tok.UnkID())
	}

	if tok.AddDummyPrefix() || !tok.ByteFallback() {
		t.Errorf("AddDummyPrefix, ByteFallback = %v, %v; want false, true", tok.AddDummyPrefix(), tok.ByteFallback())
	}

	_, ok := tok.PieceID("nope")
	if ok {
		t.Error(`PieceID("nope") found a piece`)
	}
}

func TestParseRejects(t *testing.T) {
	good := encodeTestBlob(t, defaultHeader(), testVocab())

	withPiece := func(i int, p testPiece) []byte {
		v := testVocab()
		v[i] = p

		return encodeTestBlob(t, defaultHeader(), v)
	}
	withHeader := func(edit func(*testBlobHeader)) []byte {
		h := defaultHeader()
		edit(&h)

		return encodeTestBlob(t, h, testVocab())
	}
	noBytes := slices.DeleteFunc(testVocab(), func(p testPiece) bool { return p.piece == "<0x41>" })

	tests := []struct {
		name string
		blob []byte
		want string
	}{
		{"empty", nil, "header"},
		{"short header", good[:23], "header"},
		{"piece count past the end", good[:27], "pieces need"},
		// The last piece is "s": a 7-byte record, then 1 surface byte.
		{"truncated record", good[:len(good)-3], "piece 280"},
		{"truncated surface", good[:len(good)-1], "piece 280"},
		{"trailing bytes", append(slices.Clone(good), 0), "trailing"},
		{"invalid utf-8", withPiece(270, testPiece{"\xff", -1, TypeNormal}), "UTF-8"},
		{"unknown type", withPiece(270, testPiece{"zz", -1, Type(5)}), "type"},
		{"malformed byte piece", withPiece(7, testPiece{"<0xZZ>", 0, TypeByte}), "byte piece"},
		{"duplicate byte piece", withPiece(8, testPiece{"<0x00>", 0, TypeByte}), "byte piece"},
		{"missing byte piece", encodeTestBlob(t, defaultHeader(), noBytes), "byte"},
		{"unk out of range", withHeader(func(h *testBlobHeader) { h.unk = 9999 }), "unk"},
		{"pad out of range", withHeader(func(h *testBlobHeader) { h.pad = 9999 }), "pad"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(tt.blob)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Parse error = %v, want one mentioning %q", err, tt.want)
			}
		})
	}
}

// TestParseWithoutByteFallback accepts a vocabulary with no byte pieces when
// byte fallback is off.
func TestParseWithoutByteFallback(t *testing.T) {
	h := defaultHeader()
	h.byteFallback = false
	v := slices.DeleteFunc(testVocab(), func(p testPiece) bool { return p.typ == TypeByte })

	_, err := Parse(encodeTestBlob(t, h, v))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
}

func TestDecode(t *testing.T) {
	tok := testTokenizer(t, defaultHeader())

	tests := []struct {
		name string
		ids  []int
		want string
	}{
		{"empty", nil, ""},
		{"meta space becomes space", ids(t, tok, "▁hi", "▁", "abc"), " hi abc"},
		{"leading space kept without dummy prefix", ids(t, tok, "▁", "hi"), " hi"},
		{"control and unk skipped", ids(t, tok, "<s>", "hi", "<unk>", "<pad>", "</s>"), "hi"},
		{"user-defined surface", ids(t, tok, "<speech>", "hi", "<b>x"), "<speech>hi<b>x"},
		{"byte run", byteIDs(0xC3, 0xA9, 0xE2, 0x82, 0xAC), "é€"},
		{"byte run then piece", append(byteIDs(0xF0, 0x9F, 0x99, 0x82), ids(t, tok, "▁hi")...), "🙂 hi"},
		// sentencepiece replaces every invalid byte with one U+FFFD (F-005).
		{"truncated sequence", byteIDs(0xE2, 0x82), "��"},
		{"truncated 4-byte sequence", byteIDs(0xF0, 0x9F, 0x98), "���"},
		{"lone continuation", byteIDs(0x80), "�"},
		{"surrogate", byteIDs(0xED, 0xA0, 0x80), "���"},
		{"overlong", byteIDs(0xC0, 0x80), "��"},
		{"valid U+FFFD bytes", byteIDs(0xEF, 0xBF, 0xBD), "�"},
		{"lead byte before piece", append(byteIDs(0xC3), ids(t, tok, "hi")...), "�hi"},
		// A skipped control or unk piece still ends the byte run (F-005).
		{"run split by control", append(append(byteIDs(0xC3), 0), byteIDs(0xA9)...), "��"},
		{"run split by unk", append(append(byteIDs(0xC3), 3), byteIDs(0xA9)...), "��"},
		{"run split by marker", append(append(byteIDs(0xC3), 4), byteIDs(0xA9)...), "�<speech>�"},
		// Bytes spelling ▁ stay ▁; only piece surfaces are unescaped (F-005).
		{"meta space from bytes", byteIDs(0xE2, 0x96, 0x81), "▁"},
		{"space byte", byteIDs(0x20), " "},
		{"nul byte", byteIDs(0x00), "\x00"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Decode(tt.ids)
			if err != nil {
				t.Fatalf("Decode(%v): %v", tt.ids, err)
			}

			if got != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.ids, got, tt.want)
			}
		})
	}
}

func TestDecodeDummyPrefix(t *testing.T) {
	h := defaultHeader()
	h.addDummy = true
	tok := testTokenizer(t, h)

	tests := []struct {
		ids  []int
		want string
	}{
		{ids(t, tok, "▁hi", "▁", "abc"), "hi abc"},
		{ids(t, tok, "<s>", "▁hi"), "hi"},
		{ids(t, tok, "hi"), "hi"},
		{ids(t, tok, "▁", "▁hi"), " hi"},
	}

	for _, tt := range tests {
		got, err := tok.Decode(tt.ids)
		if err != nil {
			t.Fatalf("Decode(%v): %v", tt.ids, err)
		}

		if got != tt.want {
			t.Errorf("Decode(%v) = %q, want %q", tt.ids, got, tt.want)
		}
	}
}

func TestDecodeRejectsOutOfRange(t *testing.T) {
	tok := testTokenizer(t, defaultHeader())

	for _, id := range []int{-1, tok.Len()} {
		_, err := tok.Decode([]int{4, id})
		if err == nil {
			t.Errorf("Decode with id %d: no error", id)
		}
	}
}

func TestEncode(t *testing.T) {
	tok := testTokenizer(t, defaultHeader())

	tests := []struct {
		name string
		text string
		want []int
	}{
		{"empty", "", nil},
		{"single piece", "hi", ids(t, tok, "hi")},
		// "hi" outranks "▁h", so "▁hi" is reached through "▁"+"hi".
		{"space escaped", " hi", ids(t, tok, "▁hi")},
		{"no dummy prefix", "hi aa", ids(t, tok, "hi", "▁", "aa")},
		// "ab" outranks "bc"; "abc" then merges from "ab"+"c".
		{"best score first", "abc", ids(t, tok, "abc")},
		// Equal pairs: the leftmost one merges first.
		{"leftmost tie", "aaa", ids(t, tok, "aa", "a")},
		{"marker", "hi<speech>hi", ids(t, tok, "hi", "<speech>", "hi")},
		{"longest marker first", "<b>xy", ids(t, tok, "<b>x", "y")},
		{"shorter marker", "<b>y", ids(t, tok, "<b>", "y")},
		{"marker spelled with spaces", " <speech> ", ids(t, tok, "▁", "<speech>", "▁")},
		{"byte fallback", "é", byteIDs(0xC3, 0xA9)},
		{"byte fallback inside text", "hiéhi", append(append(ids(t, tok, "hi"), byteIDs(0xC3, 0xA9)...), ids(t, tok, "hi")...)},
		{"emoji", "🙂", byteIDs(0xF0, 0x9F, 0x99, 0x82)},
		// BPE spans the whole segment: "<s>" is never formed, so no control id.
		{"control spelled out", "<s>", ids(t, tok, "<", "s", ">")},
		{"invalid utf-8 byte", "h\xffi", append(append(ids(t, tok, "h"), byteIDs(0xFF)...), ids(t, tok, "i")...)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.text)
			if !slices.Equal(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}

func TestEncodeDummyPrefix(t *testing.T) {
	h := defaultHeader()
	h.addDummy = true
	tok := testTokenizer(t, h)

	got := tok.Encode("hi ab")

	want := ids(t, tok, "▁hi", "▁", "ab")
	if !slices.Equal(got, want) {
		t.Errorf("Encode = %v, want %v", got, want)
	}

	text, err := tok.Decode(got)
	if err != nil || text != "hi ab" {
		t.Errorf("Decode(Encode) = %q, %v; want %q", text, err, "hi ab")
	}
}

func TestEncodeWithoutByteFallback(t *testing.T) {
	h := defaultHeader()
	h.byteFallback = false
	tok := testTokenizer(t, h)

	got := tok.Encode("hié")

	want := append(ids(t, tok, "hi"), tok.UnkID())
	if !slices.Equal(got, want) {
		t.Errorf("Encode = %v, want %v", got, want)
	}
}

func TestRoundTrip(t *testing.T) {
	tok := testTokenizer(t, defaultHeader())

	for _, text := range []string{
		"hi", " hi  abc ", "abc aa", "Grüße", "Привет", "你好", "مرحبا", "🙂👍🏽", "<speech>hi</speech>", "<b>x<b>y", "a\tb\nc",
	} {
		got, err := tok.Decode(tok.Encode(text))
		if err != nil {
			t.Fatalf("Decode(Encode(%q)): %v", text, err)
		}

		if got != text {
			t.Errorf("Decode(Encode(%q)) = %q", text, got)
		}
	}
}
