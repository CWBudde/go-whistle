// Package spm is a pure-Go SentencePiece BPE tokenizer for Needle 3's text
// model. It reads a tokenizer.model (a serialised sentencepiece ModelProto)
// and encodes and decodes exactly as the Python sentencepiece library
// (v0.2.2) does, which upstream needle/model/tokenizer.py SANTokenizer wraps
// (cactus-compute/needle @ af2654dffa): Encode is
// sp.Encode(text, out_type=int) and Decode is sp.Decode(ids).
//
// The port follows sentencepiece's normalizer.cc, bpe_model.cc,
// model_interface.cc and sentencepiece_processor.cc (EncodeOptimized,
// DecodeOptimized). Only what Needle 3's model configures is implemented: BPE
// with the identity normaliser (no precompiled_charsmap) and whitespace as a
// prefix. New rejects models that need more.
package spm

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"unicode/utf8"
)

const (
	metaSpace = "▁" // U+2581, sentencepiece's kSpaceSymbol

	// maxPieceSize is model_interface.cc kMaxPieceSize.
	maxPieceSize = 8000
)

// Processor is a loaded model. It is immutable and safe for concurrent use.
type Processor struct {
	model *ModelProto

	pieceID map[string]int // every piece; BPE loads with use_reserved_id_map = false
	byteVal []int16        // the byte a BYTE piece stands for, -1 for other pieces
	byteID  [256]int       // the id of <0xHH>, -1 when absent
	users   userSymbols    // USER_DEFINED pieces, the PrefixMatcher

	unk, bos, eos, pad int
}

// Load reads and parses the tokenizer.model at path.
func Load(path string) (*Processor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("spm: %w", err)
	}

	return Parse(data)
}

// Parse decodes a serialised ModelProto and builds a Processor from it.
func Parse(data []byte) (*Processor, error) {
	m, err := ParseModelProto(data)
	if err != nil {
		return nil, err
	}

	return New(m)
}

// New builds a Processor from m, which it keeps referencing; m must not be
// modified afterwards. It checks what sentencepiece's
// ModelInterface::InitializePieces checks (non-empty pieces without NUL and
// shorter than 8000 bytes, no piece twice, exactly one UNKNOWN, BYTE pieces
// only with byte_fallback and then all 256 of them) and rejects what this
// port does not implement: model types other than BPE, a precompiled
// charsmap in the normaliser or denormaliser, and
// treat_whitespace_as_suffix.
func New(m *ModelProto) (*Processor, error) {
	err := checkSupported(m)
	if err != nil {
		return nil, err
	}

	p := &Processor{
		model:   m,
		pieceID: make(map[string]int, len(m.Pieces)),
		byteVal: make([]int16, len(m.Pieces)),
		unk:     -1,
	}
	for b := range p.byteID {
		p.byteID[b] = -1
	}

	var users []string

	for id, pc := range m.Pieces {
		err = p.addPiece(id, pc)
		if err != nil {
			return nil, err
		}

		if pc.Type == TypeUserDefined {
			users = append(users, pc.Piece)
		}
	}

	if p.unk < 0 {
		return nil, errors.New("spm: no UNKNOWN piece")
	}

	if m.Trainer.ByteFallback {
		for b, id := range p.byteID {
			if id < 0 {
				return nil, fmt.Errorf("spm: byte_fallback is on, but byte piece <0x%02X> is missing", b)
			}
		}
	}

	p.users = newUserSymbols(users)

	// sentencepiece_processor.cc Load: the special ids come from the
	// special pieces, not from the trainer spec's ids.
	p.bos = p.specialID(m.Trainer.BOSPiece, TypeControl)
	p.eos = p.specialID(m.Trainer.EOSPiece, TypeControl)
	p.pad = p.specialID(m.Trainer.PadPiece, TypeControl)

	return p, nil
}

func checkSupported(m *ModelProto) error {
	switch {
	case m.Trainer.ModelType != ModelBPE:
		return fmt.Errorf("spm: model type %s is not supported, only BPE", m.Trainer.ModelType)
	case len(m.Normalizer.PrecompiledCharsmap) > 0:
		return fmt.Errorf("spm: normalizer %q has a precompiled charsmap; only the identity normalizer is supported",
			m.Normalizer.Name)
	case m.HasDenormalizer && len(m.Denormalizer.PrecompiledCharsmap) > 0:
		return fmt.Errorf("spm: denormalizer %q has a precompiled charsmap, which is not supported", m.Denormalizer.Name)
	case m.Trainer.TreatWhitespaceAsSuffix:
		return errors.New("spm: treat_whitespace_as_suffix is not supported")
	}

	return nil
}

// pieceToByte reads "<0xHH>" with upper-case hex digits, the only spelling
// sentencepiece's ByteToPiece writes and PieceToByte accepts.
func pieceToByte(piece string) (byte, bool) {
	if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") || piece[5] != '>' {
		return 0, false
	}

	var b byte

	for _, c := range piece[3:5] {
		switch {
		case c >= '0' && c <= '9':
			b = b<<4 | byte(c-'0')
		case c >= 'A' && c <= 'F':
			b = b<<4 | byte(c-'A'+10)
		default:
			return 0, false
		}
	}

	return b, true
}

// Len is the vocabulary size.
func (p *Processor) Len() int { return len(p.model.Pieces) }

// Piece returns the surface of id, which must be in [0, Len()).
func (p *Processor) Piece(id int) string { return p.model.Pieces[id].Piece }

// Score returns the score of id; in a BPE model a higher score merges first.
func (p *Processor) Score(id int) float32 { return p.model.Pieces[id].Score }

// Type returns the kind of id.
func (p *Processor) Type(id int) Type { return p.model.Pieces[id].Type }

// PieceID looks a piece up by its surface.
func (p *Processor) PieceID(piece string) (int, bool) {
	id, ok := p.pieceID[piece]

	return id, ok
}

// UnkID is the id of the UNKNOWN piece, which Encode emits for symbols
// without a piece when byte fallback is off. (sentencepiece's unk_id() looks
// the trainer spec's unk_piece up instead; the two agree unless the model
// renames <unk> without saying so.)
func (p *Processor) UnkID() int { return p.unk }

// BOSID is the id of the trainer spec's bos_piece, or -1 when that is not a
// CONTROL piece, as sentencepiece's bos_id().
func (p *Processor) BOSID() int { return p.bos }

// EOSID is the id of eos_piece, or -1, as BOSID.
func (p *Processor) EOSID() int { return p.eos }

// PadID is the id of pad_piece, or -1, as BOSID.
func (p *Processor) PadID() int { return p.pad }

// Trainer returns the model's trainer spec.
func (p *Processor) Trainer() TrainerSpec { return p.model.Trainer }

// Normalizer returns the model's normalizer spec.
func (p *Processor) Normalizer() NormalizerSpec { return p.model.Normalizer }

// Decode turns ids into text as sentencepiece's DecodeOptimized: CONTROL
// pieces vanish, an UNKNOWN piece becomes the trainer spec's unk_surface
// (" ⁇ " by default), other pieces contribute their surface with "▁" as a
// space, and each run of consecutive BYTE pieces is decoded as UTF-8 with
// one U+FFFD per invalid byte. Any other piece, a vanishing CONTROL piece
// included, ends a byte run. With add_dummy_prefix or
// remove_extra_whitespaces, the leading "▁" of the first piece that is not a
// byte piece is dropped while the output is still empty. An id outside
// [0, Len()) is an error.
func (p *Processor) Decode(ids []int) (string, error) {
	var (
		sb   strings.Builder
		run  []byte
		norm = p.model.Normalizer
		bos  = true // still expecting the dummy-prefix space
	)

	for _, id := range ids {
		if id < 0 || id >= len(p.model.Pieces) {
			return "", fmt.Errorf("spm: id %d out of range [0, %d)", id, len(p.model.Pieces))
		}

		if b := p.byteVal[id]; b >= 0 {
			run = append(run, byte(b))

			continue
		}

		writeBytes(&sb, run)
		run = run[:0]

		if sb.Len() > 0 {
			bos = false
		}

		pc := p.model.Pieces[id]
		if pc.Type == TypeControl {
			continue
		}

		piece, dropped := pc.Piece, false
		if bos && (norm.AddDummyPrefix || norm.RemoveExtraWhitespaces) {
			piece, dropped = strings.CutPrefix(piece, metaSpace)
			dropped = dropped && !norm.RemoveExtraWhitespaces
		}

		if pc.Type == TypeUnknown {
			sb.WriteString(p.model.Trainer.UnkSurface)
		} else {
			sb.WriteString(strings.ReplaceAll(piece, metaSpace, " "))
		}

		if dropped || sb.Len() > 0 {
			bos = false
		}
	}

	writeBytes(&sb, run)

	return sb.String(), nil
}

// addPiece validates and indexes piece id.
func (p *Processor) addPiece(id int, pc Piece) error {
	switch {
	case pc.Piece == "":
		return fmt.Errorf("spm: piece %d is empty", id)
	case len(pc.Piece) >= maxPieceSize:
		return fmt.Errorf("spm: piece %d is %d bytes long", id, len(pc.Piece))
	case strings.IndexByte(pc.Piece, 0) >= 0:
		return fmt.Errorf("spm: piece %d %q contains NUL", id, pc.Piece)
	}

	if prev, ok := p.pieceID[pc.Piece]; ok {
		return fmt.Errorf("spm: piece %d %q repeats piece %d", id, pc.Piece, prev)
	}

	p.pieceID[pc.Piece] = id
	p.byteVal[id] = -1

	switch pc.Type {
	case TypeUnknown:
		if p.unk >= 0 {
			return fmt.Errorf("spm: piece %d %q is a second UNKNOWN piece after %d", id, pc.Piece, p.unk)
		}

		p.unk = id
	case TypeByte:
		if !p.model.Trainer.ByteFallback {
			return fmt.Errorf("spm: byte piece %d %q, but byte_fallback is off", id, pc.Piece)
		}

		b, ok := pieceToByte(pc.Piece)
		if !ok {
			return fmt.Errorf("spm: piece %d: malformed byte piece %q", id, pc.Piece)
		}

		p.byteID[b] = id
		p.byteVal[id] = int16(b)
	}

	return nil
}

// specialID is sentencepiece's PieceToId(piece) when the piece has type
// want, else -1.
func (p *Processor) specialID(piece string, want Type) int {
	id, ok := p.pieceID[piece]
	if !ok || p.model.Pieces[id].Type != want {
		return -1
	}

	return id
}

// writeBytes appends a byte run as UTF-8, replacing every byte that does not
// start a valid sequence with U+FFFD. Go's and sentencepiece's
// (string_util::DecodeUTF8) notions of a valid sequence agree: no overlong
// forms, surrogates or code points above U+10FFFF.
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

// userSymbols is sentencepiece's PrefixMatcher over the USER_DEFINED
// pieces: at each position the longest one that matches wins.
type userSymbols map[byte][]string // by first byte, longest first

func newUserSymbols(pieces []string) userSymbols {
	u := userSymbols{}
	for _, s := range pieces {
		u[s[0]] = append(u[s[0]], s)
	}

	for _, list := range u {
		slices.SortFunc(list, func(a, b string) int { return len(b) - len(a) })
	}

	return u
}

// match returns the length of the longest user-defined symbol s starts
// with, or 0.
func (u userSymbols) match(s string) int {
	if s == "" {
		return 0
	}

	for _, sym := range u[s[0]] {
		if strings.HasPrefix(s, sym) {
			return len(sym)
		}
	}

	return 0
}
