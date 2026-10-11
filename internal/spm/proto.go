package spm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unicode/utf8"
)

// ModelProto is the part of sentencepiece_model.proto's ModelProto that
// encoding and decoding need. Fields the reader does not know are skipped;
// self_test_data is ignored.
type ModelProto struct {
	Pieces       []Piece        // field 1
	Trainer      TrainerSpec    // field 2
	Normalizer   NormalizerSpec // field 3
	Denormalizer NormalizerSpec // field 5

	HasDenormalizer bool
}

// Piece is one vocabulary entry, ModelProto.SentencePiece.
type Piece struct {
	Piece string  // field 1
	Score float32 // field 2
	Type  Type    // field 3, NORMAL when absent
}

// TrainerSpec holds the trainer settings that are recorded in the model.
// Only ModelType, ByteFallback, TreatWhitespaceAsSuffix, UnkSurface and the
// special pieces affect encoding or decoding; the rest are kept for
// inspection. Absent fields carry the proto2 defaults.
type TrainerSpec struct {
	ModelType                 ModelType // field 3
	VocabSize                 int32     // field 4
	MaxSentencePieceLength    int32     // field 20
	SplitByUnicodeScript      bool      // field 21
	SplitByWhitespace         bool      // field 22
	SplitByNumber             bool      // field 23
	TreatWhitespaceAsSuffix   bool      // field 24
	SplitDigits               bool      // field 25
	AllowWhitespaceOnlyPieces bool      // field 26
	ControlSymbols            []string  // field 30
	UserDefinedSymbols        []string  // field 31
	ByteFallback              bool      // field 35
	UnkID                     int32     // field 40
	BOSID                     int32     // field 41
	EOSID                     int32     // field 42
	PadID                     int32     // field 43
	UnkSurface                string    // field 44
	UnkPiece                  string    // field 45
	BOSPiece                  string    // field 46
	EOSPiece                  string    // field 47
	PadPiece                  string    // field 48
}

// NormalizerSpec is sentencepiece_model.proto's NormalizerSpec. Absent fields
// carry the proto2 defaults (all three flags true).
type NormalizerSpec struct {
	Name                   string // field 1
	PrecompiledCharsmap    []byte // field 2
	AddDummyPrefix         bool   // field 3
	RemoveExtraWhitespaces bool   // field 4
	EscapeWhitespaces      bool   // field 5
	NormalizationRuleTSV   string // field 6
}

// Type is a piece's kind, numbered as ModelProto.SentencePiece.Type.
type Type uint8

// Piece types.
const (
	TypeNormal      Type = 1 // BPE piece
	TypeUnknown     Type = 2 // <unk>
	TypeControl     Type = 3 // <s>, </s>, <pad>: never produced from text, invisible in Decode
	TypeUserDefined Type = 4 // matched whole in the input, never split or merged
	TypeUnused      Type = 5 // mergeable, but re-split into its parts
	TypeByte        Type = 6 // <0xHH> byte-fallback piece
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
	case TypeUnused:
		return "UNUSED"
	case TypeByte:
		return "BYTE"
	default:
		return fmt.Sprintf("Type(%d)", uint8(t))
	}
}

// ModelType is TrainerSpec.ModelType.
type ModelType int32

// Model types. Only BPE is implemented.
const (
	ModelUnigram ModelType = 1
	ModelBPE     ModelType = 2
	ModelWord    ModelType = 3
	ModelChar    ModelType = 4
)

func (m ModelType) String() string {
	switch m {
	case ModelUnigram:
		return "UNIGRAM"
	case ModelBPE:
		return "BPE"
	case ModelWord:
		return "WORD"
	case ModelChar:
		return "CHAR"
	default:
		return fmt.Sprintf("ModelType(%d)", int32(m))
	}
}

// defaultUnkSurface is sentencepiece's kDefaultUnknownSymbol, " ⁇ ".
const defaultUnkSurface = " ⁇ "

// Protobuf wire types.
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
	wireFixed32 = 5
)

var errTruncated = errors.New("truncated message")

// ParseModelProto decodes a serialised ModelProto (a tokenizer.model file).
// It reads the protobuf wire format by hand, applies the proto2 defaults of
// sentencepiece_model.proto, and, as protobuf does, lets the last occurrence
// of a scalar field win and merges repeated occurrences of a message field.
// Strings must be valid UTF-8.
func ParseModelProto(data []byte) (*ModelProto, error) {
	m := &ModelProto{
		Trainer:      defaultTrainerSpec(),
		Normalizer:   defaultNormalizerSpec(),
		Denormalizer: defaultNormalizerSpec(),
	}

	err := walk(data, func(num int, f field) error {
		switch num {
		case 1:
			p, err := parsePiece(f)
			if err != nil {
				return fmt.Errorf("piece %d: %w", len(m.Pieces), err)
			}

			m.Pieces = append(m.Pieces, p)
		case 2:
			return parseMessage(f, "trainer_spec", m.Trainer.set)
		case 3:
			return parseMessage(f, "normalizer_spec", m.Normalizer.set)
		case 5:
			m.HasDenormalizer = true

			return parseMessage(f, "denormalizer_spec", m.Denormalizer.set)
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("spm: model proto: %w", err)
	}

	return m, nil
}

func defaultTrainerSpec() TrainerSpec {
	return TrainerSpec{
		ModelType:              ModelUnigram,
		VocabSize:              8000,
		MaxSentencePieceLength: 16,
		SplitByUnicodeScript:   true,
		SplitByWhitespace:      true,
		SplitByNumber:          true,
		UnkID:                  0,
		BOSID:                  1,
		EOSID:                  2,
		PadID:                  -1,
		UnkSurface:             defaultUnkSurface,
		UnkPiece:               "<unk>",
		BOSPiece:               "<s>",
		EOSPiece:               "</s>",
		PadPiece:               "<pad>",
	}
}

func defaultNormalizerSpec() NormalizerSpec {
	return NormalizerSpec{AddDummyPrefix: true, RemoveExtraWhitespaces: true, EscapeWhitespaces: true}
}

func parsePiece(f field) (Piece, error) {
	p := Piece{Type: TypeNormal}

	if f.wire != wireBytes {
		return p, fmt.Errorf("wire type %d, want %d", f.wire, wireBytes)
	}

	err := walk(f.bytes, func(num int, f field) error {
		var err error

		switch num {
		case 1:
			p.Piece, err = f.str()
		case 2:
			p.Score, err = f.float()
		case 3:
			var v uint64

			v, err = f.varint()
			if err == nil && (v < uint64(TypeNormal) || v > uint64(TypeByte)) {
				err = fmt.Errorf("unknown type %d", v)
			}

			p.Type = Type(v)
		}

		return err
	})

	return p, err
}

func (s *TrainerSpec) set(num int, f field) error {
	if dst := s.int32Field(num); dst != nil {
		v, err := f.int32()
		*dst = v

		return err
	}

	if dst := s.boolField(num); dst != nil {
		v, err := f.bool()
		*dst = v

		return err
	}

	if dst := s.stringField(num); dst != nil {
		v, err := f.str()
		*dst = v

		return err
	}

	switch num {
	case 30:
		return appendStr(&s.ControlSymbols, f)
	case 31:
		return appendStr(&s.UserDefinedSymbols, f)
	}

	return nil
}

func (s *TrainerSpec) int32Field(num int) *int32 {
	switch num {
	case 3:
		return (*int32)(&s.ModelType)
	case 4:
		return &s.VocabSize
	case 20:
		return &s.MaxSentencePieceLength
	case 40:
		return &s.UnkID
	case 41:
		return &s.BOSID
	case 42:
		return &s.EOSID
	case 43:
		return &s.PadID
	}

	return nil
}

func (s *TrainerSpec) boolField(num int) *bool {
	switch num {
	case 21:
		return &s.SplitByUnicodeScript
	case 22:
		return &s.SplitByWhitespace
	case 23:
		return &s.SplitByNumber
	case 24:
		return &s.TreatWhitespaceAsSuffix
	case 25:
		return &s.SplitDigits
	case 26:
		return &s.AllowWhitespaceOnlyPieces
	case 35:
		return &s.ByteFallback
	}

	return nil
}

func (s *TrainerSpec) stringField(num int) *string {
	switch num {
	case 44:
		return &s.UnkSurface
	case 45:
		return &s.UnkPiece
	case 46:
		return &s.BOSPiece
	case 47:
		return &s.EOSPiece
	case 48:
		return &s.PadPiece
	}

	return nil
}

func (s *NormalizerSpec) set(num int, f field) error {
	var err error

	switch num {
	case 1:
		s.Name, err = f.str()
	case 2:
		if f.wire != wireBytes {
			return fmt.Errorf("field %d: wire type %d, want %d", num, f.wire, wireBytes)
		}

		s.PrecompiledCharsmap = append([]byte(nil), f.bytes...)
	case 3:
		s.AddDummyPrefix, err = f.bool()
	case 4:
		s.RemoveExtraWhitespaces, err = f.bool()
	case 5:
		s.EscapeWhitespaces, err = f.bool()
	case 6:
		s.NormalizationRuleTSV, err = f.str()
	}

	return err
}

func appendStr(dst *[]string, f field) error {
	s, err := f.str()
	if err != nil {
		return err
	}

	*dst = append(*dst, s)

	return nil
}

// parseMessage walks an embedded message field with set.
func parseMessage(f field, name string, set func(int, field) error) error {
	if f.wire != wireBytes {
		return fmt.Errorf("%s: wire type %d, want %d", name, f.wire, wireBytes)
	}

	err := walk(f.bytes, set)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}

	return nil
}

// field is one decoded field: the varint, the fixed-width value or the
// length-delimited payload, by wire type.
type field struct {
	wire  int
	num   uint64 // varint and fixed values
	bytes []byte // length-delimited payload
}

// walk calls fn for every field of the message in data, in wire order.
func walk(data []byte, fn func(num int, f field) error) error {
	for len(data) > 0 {
		key, n := binary.Uvarint(data)
		if n <= 0 {
			return errTruncated
		}

		data = data[n:]
		num := key >> 3

		if num == 0 || num > math.MaxInt32 {
			return fmt.Errorf("invalid field number %d", num)
		}

		f, n, err := readValue(data, int(key&7))
		if err != nil {
			return fmt.Errorf("field %d: %w", num, err)
		}

		data = data[n:]

		err = fn(int(num), f)
		if err != nil {
			return err
		}
	}

	return nil
}

// readValue reads the value of a field of the given wire type at the start
// of data and returns it with its length. Groups (wire types 3 and 4) are
// rejected; sentencepiece_model.proto has none.
func readValue(data []byte, wire int) (field, int, error) {
	f := field{wire: wire}

	switch wire {
	case wireVarint:
		v, n := binary.Uvarint(data)
		if n <= 0 {
			return f, 0, errTruncated
		}

		f.num = v

		return f, n, nil
	case wireFixed64:
		if len(data) < 8 {
			return f, 0, errTruncated
		}

		f.num = binary.LittleEndian.Uint64(data)

		return f, 8, nil
	case wireFixed32:
		if len(data) < 4 {
			return f, 0, errTruncated
		}

		f.num = uint64(binary.LittleEndian.Uint32(data))

		return f, 4, nil
	case wireBytes:
		size, n := binary.Uvarint(data)
		if n <= 0 || size > uint64(len(data)-n) {
			return f, 0, errTruncated
		}

		f.bytes = data[n : n+int(size)]

		return f, n + int(size), nil
	default:
		return f, 0, fmt.Errorf("unsupported wire type %d", wire)
	}
}

func (f field) varint() (uint64, error) {
	if f.wire != wireVarint {
		return 0, fmt.Errorf("wire type %d, want %d", f.wire, wireVarint)
	}

	return f.num, nil
}

// int32 reads an int32 or enum field; negative values are sign-extended
// 10-byte varints, of which protobuf keeps the low 32 bits.
func (f field) int32() (int32, error) {
	v, err := f.varint()

	return int32(uint32(v)), err
}

func (f field) bool() (bool, error) {
	v, err := f.varint()

	return v != 0, err
}

func (f field) float() (float32, error) {
	if f.wire != wireFixed32 {
		return 0, fmt.Errorf("wire type %d, want %d", f.wire, wireFixed32)
	}

	return math.Float32frombits(uint32(f.num)), nil
}

func (f field) str() (string, error) {
	if f.wire != wireBytes {
		return "", fmt.Errorf("wire type %d, want %d", f.wire, wireBytes)
	}

	if !utf8.Valid(f.bytes) {
		return "", fmt.Errorf("string %q is not valid UTF-8", f.bytes)
	}

	return string(f.bytes), nil
}
