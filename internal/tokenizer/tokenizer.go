// Package tokenizer reads Whistle's SentencePiece BPE vocabulary from the RAW
// blob in whistle.cact (export.py _tokenizer_blob) and decodes and encodes
// with it. Decode matches sentencepiece; Encode ports upstream export.py
// RefTokenizer. docs/FINDINGS.md F-005 covers where the two upstream
// references disagree and which one each side follows.
package tokenizer

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// Type is a piece's kind, numbered as export.py TK_*.
type Type uint8

// Piece types.
const (
	TypeNormal      Type = 0 // BPE piece
	TypeUnknown     Type = 1 // <unk>
	TypeControl     Type = 2 // <pad>, </s>, <s>
	TypeUserDefined Type = 3 // chat markers and language tokens, matched whole by Encode
	TypeByte        Type = 4 // <0xHH> byte-fallback piece
)

func (t Type) String() string {
	switch t {
	case TypeNormal:
		return "NORMAL"
	case TypeUnknown:
		return "UNKNOWN"
	case TypeControl:
		return "CONTROL"
	case TypeUserDefined:
		return "USER_DEFINED"
	case TypeByte:
		return "BYTE"
	default:
		return fmt.Sprintf("Type(%d)", uint8(t))
	}
}

const (
	headerSize = 24 // export.py _TK_HDR "<IIIIIBBH"
	recordSize = 7  // export.py _TK_REC "<fBH", followed by the surface

	metaSpace = "▁" // U+2581, SentencePiece's escaped space
)

// Tokenizer is a parsed vocabulary. It is immutable and safe for concurrent
// use.
type Tokenizer struct {
	pieces   []string
	scores   []float32
	types    []Type
	surfaces []string // decoded text per piece; only NORMAL and USER_DEFINED are used
	byteVals []byte   // the byte a BYTE piece stands for

	pieceID map[string]int
	byteID  [256]int // -1 when the vocabulary has no piece for the byte
	markers []string // USER_DEFINED pieces, longest first

	pad, eos, bos, unk           int
	addDummyPrefix, byteFallback bool
}

// FromCact parses the first RAW tensor of f, which export.py writes as the
// tokenizer blob, as RefTokenizer.from_cact does.
func FromCact(f *cact.File) (*Tokenizer, error) {
	for i := range f.Len() {
		rec, err := f.Record(i)
		if err != nil {
			return nil, err
		}

		if rec.DType != cact.RAW {
			continue
		}

		blob, err := f.Raw(i)
		if err != nil {
			return nil, err
		}

		return Parse(blob)
	}

	return nil, fmt.Errorf("tokenizer: no RAW tensor among %d", f.Len())
}

// Parse decodes a tokenizer blob: the header, then one record per piece in
// id order. It validates more strictly than export.py parse_tokenizer_blob:
// trailing bytes, unknown types, malformed or repeated byte pieces, special
// ids out of range, and byte fallback without all 256 byte pieces are errors.
// The returned Tokenizer does not reference blob.
func Parse(blob []byte) (*Tokenizer, error) {
	if len(blob) < headerSize {
		return nil, fmt.Errorf("tokenizer: %d bytes is shorter than the %d-byte header", len(blob), headerSize)
	}

	le := binary.LittleEndian
	n := uint64(le.Uint32(blob))

	if n*recordSize > uint64(len(blob)-headerSize) {
		return nil, fmt.Errorf("tokenizer: %d pieces need at least %d bytes after the header, have %d",
			n, n*recordSize, len(blob)-headerSize)
	}

	t := &Tokenizer{
		pieces:         make([]string, n),
		scores:         make([]float32, n),
		types:          make([]Type, n),
		surfaces:       make([]string, n),
		byteVals:       make([]byte, n),
		pieceID:        make(map[string]int, n),
		pad:            int(le.Uint32(blob[4:])),
		eos:            int(le.Uint32(blob[8:])),
		bos:            int(le.Uint32(blob[12:])),
		unk:            int(le.Uint32(blob[16:])),
		addDummyPrefix: blob[20] != 0,
		byteFallback:   blob[21] != 0,
	}
	for b := range t.byteID {
		t.byteID[b] = -1
	}

	off := headerSize
	for i := range int(n) {
		next, err := t.parsePiece(blob, off, i)
		if err != nil {
			return nil, err
		}

		off = next
	}

	if off != len(blob) {
		return nil, fmt.Errorf("tokenizer: %d trailing bytes after %d pieces", len(blob)-off, n)
	}

	err := t.check()
	if err != nil {
		return nil, err
	}

	for id, typ := range t.types {
		if typ == TypeUserDefined {
			t.markers = append(t.markers, t.pieces[id])
		}
	}

	// RefTokenizer sorts by code-point length with Python's stable sort, so
	// equal lengths stay in id order.
	slices.SortStableFunc(t.markers, func(a, b string) int {
		return utf8.RuneCountInString(b) - utf8.RuneCountInString(a)
	})

	return t, nil
}

// Len is the vocabulary size.
func (t *Tokenizer) Len() int { return len(t.pieces) }

// Piece returns the surface of id, which must be in [0, Len()).
func (t *Tokenizer) Piece(id int) string { return t.pieces[id] }

// Score returns the BPE score of id; a higher score merges first.
func (t *Tokenizer) Score(id int) float32 { return t.scores[id] }

// Type returns the kind of id.
func (t *Tokenizer) Type(id int) Type { return t.types[id] }

// PieceID looks a piece up by its surface.
func (t *Tokenizer) PieceID(piece string) (int, bool) {
	id, ok := t.pieceID[piece]

	return id, ok
}

// PadID is the padding id from the blob header.
func (t *Tokenizer) PadID() int { return t.pad }

// EOSID is the end-of-sequence id from the blob header.
func (t *Tokenizer) EOSID() int { return t.eos }

// BOSID is the begin-of-sequence id from the blob header.
func (t *Tokenizer) BOSID() int { return t.bos }

// UnkID is the unknown-piece id from the blob header.
func (t *Tokenizer) UnkID() int { return t.unk }

// AddDummyPrefix reports whether Encode prepends "▁" and Decode drops one
// leading space.
func (t *Tokenizer) AddDummyPrefix() bool { return t.addDummyPrefix }

// ByteFallback reports whether Encode spells unknown symbols as byte pieces
// rather than <unk>.
func (t *Tokenizer) ByteFallback() bool { return t.byteFallback }

// Decode turns ids into text the way sentencepiece does (F-005): CONTROL and
// UNKNOWN pieces vanish, NORMAL and USER_DEFINED pieces contribute their
// surface with "▁" as a space, and each run of consecutive BYTE pieces is
// decoded as UTF-8 with one U+FFFD per invalid byte. Any other piece ends a
// byte run, including a vanishing one, and "▁" spelled by bytes stays "▁".
// With AddDummyPrefix one leading space is dropped. An id outside
// [0, Len()) is an error.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var (
		sb  strings.Builder
		run []byte
	)

	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			return "", fmt.Errorf("tokenizer: id %d out of range [0, %d)", id, len(t.pieces))
		}

		if t.types[id] == TypeByte {
			run = append(run, t.byteVals[id])

			continue
		}

		writeBytes(&sb, run)
		run = run[:0]

		if t.types[id] == TypeNormal || t.types[id] == TypeUserDefined {
			sb.WriteString(t.surfaces[id])
		}
	}

	writeBytes(&sb, run)

	text := sb.String()
	if t.addDummyPrefix {
		text = strings.TrimPrefix(text, " ")
	}

	return text, nil
}

// writeBytes appends a byte run as UTF-8, replacing every byte that does not
// start a valid sequence with U+FFFD, as sentencepiece does.
func writeBytes(sb *strings.Builder, run []byte) {
	for len(run) > 0 {
		r, size := utf8.DecodeRune(run)
		if r == utf8.RuneError && size == 1 {
			sb.WriteRune(utf8.RuneError)
		} else {
			sb.Write(run[:size])
		}

		run = run[size:]
	}
}

// Encode is upstream RefTokenizer.encode: spaces become "▁" (with a leading
// "▁" under AddDummyPrefix), USER_DEFINED markers are cut out whole (longest
// first), and each stretch between them is BPE-merged by best score. Symbols
// left without a piece become their UTF-8 bytes under ByteFallback, else
// <unk>. Unlike sentencepiece, BPE does not split at spaces first; on the
// fixture's multilingual samples both give the same ids (F-005). Invalid
// UTF-8 in text, which a Python str cannot hold, is carried byte by byte and
// so falls back to byte pieces.
func (t *Tokenizer) Encode(text string) []int {
	if text == "" {
		return nil
	}

	esc := strings.ReplaceAll(text, " ", metaSpace)
	if t.addDummyPrefix {
		esc = metaSpace + esc
	}

	var out []int

	start := 0

	for i := 0; i < len(esc); {
		marker, ok := t.markerAt(esc[i:])
		if !ok {
			_, size := utf8.DecodeRuneInString(esc[i:])
			i += size

			continue
		}

		out = t.bpe(out, esc[start:i])
		out = append(out, t.pieceID[marker])
		i += len(marker)
		start = i
	}

	return t.bpe(out, esc[start:])
}

// parsePiece reads record id at off and returns the offset past it.
func (t *Tokenizer) parsePiece(blob []byte, off, id int) (int, error) {
	if len(blob)-off < recordSize {
		return 0, fmt.Errorf("tokenizer: piece %d: record at %d runs past the end (%d bytes)", id, off, len(blob))
	}

	score := math.Float32frombits(binary.LittleEndian.Uint32(blob[off:]))
	typ := Type(blob[off+4])
	size := int(binary.LittleEndian.Uint16(blob[off+5:]))
	off += recordSize

	if len(blob)-off < size {
		return 0, fmt.Errorf("tokenizer: piece %d: %d-byte surface at %d runs past the end (%d bytes)",
			id, size, off, len(blob))
	}

	piece := string(blob[off : off+size])
	off += size

	switch {
	case !utf8.ValidString(piece):
		return 0, fmt.Errorf("tokenizer: piece %d %q is not valid UTF-8", id, piece)
	case typ > TypeByte:
		return 0, fmt.Errorf("tokenizer: piece %d %q has unknown type %d", id, piece, uint8(typ))
	}

	if typ == TypeByte {
		b, ok := parseBytePiece(piece)
		if !ok {
			return 0, fmt.Errorf("tokenizer: piece %d: malformed byte piece %q", id, piece)
		}

		if t.byteID[b] >= 0 {
			return 0, fmt.Errorf("tokenizer: piece %d: byte piece %q repeats piece %d", id, piece, t.byteID[b])
		}

		t.byteID[b] = id
		t.byteVals[id] = b
	}

	t.pieces[id] = piece
	t.scores[id] = score
	t.types[id] = typ
	t.surfaces[id] = strings.ReplaceAll(piece, metaSpace, " ")
	t.pieceID[piece] = id // a repeated piece maps to its last id, as RefTokenizer's p2id

	return off, nil
}

// parseBytePiece reads "<0xHH>", as RefTokenizer's int(p[3:5], 16).
func parseBytePiece(piece string) (byte, bool) {
	if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") || piece[5] != '>' {
		return 0, false
	}

	v, err := strconv.ParseUint(piece[3:5], 16, 8)
	if err != nil {
		return 0, false
	}

	return byte(v), true
}

// check validates the header's special ids and the byte-fallback table.
func (t *Tokenizer) check() error {
	for _, s := range []struct {
		name string
		id   int
	}{{"pad", t.pad}, {"eos", t.eos}, {"bos", t.bos}, {"unk", t.unk}} {
		if s.id >= len(t.pieces) {
			return fmt.Errorf("tokenizer: %s id %d out of range [0, %d)", s.name, s.id, len(t.pieces))
		}
	}

	if t.byteFallback {
		for b, id := range t.byteID {
			if id < 0 {
				return fmt.Errorf("tokenizer: byte fallback is on, but byte piece <0x%02X> is missing", b)
			}
		}
	}

	return nil
}

// markerAt returns the first marker, longest first, that s starts with.
func (t *Tokenizer) markerAt(s string) (string, bool) {
	for _, m := range t.markers {
		if strings.HasPrefix(s, m) {
			return m, true
		}
	}

	return "", false
}

// span is a symbol of the segment being merged, as seg[lo:hi].
type span struct{ lo, hi int }

// bpe is RefTokenizer._bpe: start from code points and repeatedly merge the
// adjacent pair whose concatenation is the highest-scoring piece, the
// leftmost on ties, until no pair is a piece. It appends the ids to out.
func (t *Tokenizer) bpe(out []int, seg string) []int {
	syms := make([]span, 0, len(seg))

	for i := 0; i < len(seg); {
		_, size := utf8.DecodeRuneInString(seg[i:])
		syms = append(syms, span{i, i + size})
		i += size
	}

	for len(syms) > 1 {
		best, bestJ := float32(0), -1

		for j := range len(syms) - 1 {
			id, ok := t.pieceID[seg[syms[j].lo:syms[j+1].hi]]
			if ok && (bestJ < 0 || t.scores[id] > best) {
				best, bestJ = t.scores[id], j
			}
		}

		if bestJ < 0 {
			break
		}

		syms[bestJ].hi = syms[bestJ+1].hi
		syms = slices.Delete(syms, bestJ+1, bestJ+2)
	}

	for _, s := range syms {
		sym := seg[s.lo:s.hi]

		switch id, ok := t.pieceID[sym]; {
		case ok:
			out = append(out, id)
		case t.byteFallback:
			for k := range len(sym) {
				out = append(out, t.byteID[sym[k]])
			}
		default:
			out = append(out, t.unk)
		}
	}

	return out
}
