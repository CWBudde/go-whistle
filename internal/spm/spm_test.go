package spm

import (
	"math"
	"slices"
	"strings"
	"testing"
)

// ids maps pieces of p to their ids; "0xHH" stands for the byte piece
// <0xHH>.
func ids(t *testing.T, p *Processor, pieces ...string) []int {
	t.Helper()

	out := make([]int, len(pieces))

	for i, s := range pieces {
		if strings.HasPrefix(s, "0x") {
			s = "<" + s + ">"
		}

		id, ok := p.PieceID(s)
		if !ok {
			t.Fatalf("piece %q not in vocab", s)
		}

		out[i] = id
	}

	return out
}

// noBytesModel is testVocab without byte pieces and byte fallback.
func noBytesModel() testModel {
	m := defaultTestModel()
	m.byteFB = false
	m.pieces = slices.DeleteFunc(m.pieces, func(p testPiece) bool { return p.typ == TypeByte })

	return m
}

func TestAccessors(t *testing.T) {
	p := newTest(t, defaultTestModel())
	vocab := testVocab()

	if p.Len() != len(vocab) {
		t.Fatalf("Len = %d, want %d", p.Len(), len(vocab))
	}

	for id, v := range vocab {
		if p.Piece(id) != v.piece || p.Score(id) != v.score || p.Type(id) != v.typ {
			t.Errorf("id %d = (%q, %v, %s), want (%q, %v, %s)", id, p.Piece(id), p.Score(id), p.Type(id), v.piece, v.score, v.typ)
		}

		if got, ok := p.PieceID(v.piece); !ok || got != id {
			t.Errorf("PieceID(%q) = %d, %v; want %d", v.piece, got, ok, id)
		}
	}

	if p.UnkID() != 3 || p.BOSID() != 2 || p.EOSID() != 1 || p.PadID() != 0 {
		t.Errorf("unk/bos/eos/pad = %d/%d/%d/%d, want 3/2/1/0", p.UnkID(), p.BOSID(), p.EOSID(), p.PadID())
	}

	if p.Trainer().ModelType != ModelBPE || !p.Trainer().ByteFallback || p.Normalizer().Name != "identity" {
		t.Errorf("Trainer, Normalizer = %+v, %+v", p.Trainer(), p.Normalizer())
	}

	// Specials resolve by piece, as sentencepiece's bos_id(): a renamed or
	// non-CONTROL piece gives -1.
	m := defaultTestModel()
	m.extra = func(ts, ns pbuf) (pbuf, pbuf) { return ts.str(46, "<bos>").str(48, "<b>"), ns }

	q := newTest(t, m)
	if q.BOSID() != -1 || q.PadID() != -1 || q.EOSID() != 1 {
		t.Errorf("renamed specials bos/eos/pad = %d/%d/%d, want -1/1/-1", q.BOSID(), q.EOSID(), q.PadID())
	}
}

func TestEncode(t *testing.T) {
	p := newTest(t, defaultTestModel())

	tests := []struct {
		name string
		text string
		want []string
	}{
		{"empty", "", nil},
		{"piece", "hi", []string{"hi"}},
		{"space becomes meta space", "ab hi", []string{"ab", "▁hi"}},
		{"meta space in text acts as space", "▁hi", []string{"▁hi"}},
		{"lone space", " ", []string{"▁"}},
		{"merge order by score", "abc", []string{"abc"}},
		{"tie merges leftmost", "aad", []string{"aa", "d"}},
		{"unused piece is split again", "xy", []string{"x", "y"}},
		{"unused piece merges on", "xyz", []string{"xyz"}},
		{"plus zero beats minus zero", "efg", []string{"e", "fg"}},
		{"longest user-defined symbol", "<b>xy<b>", []string{"<b>x", "y", "<b>"}},
		{"user-defined symbols stay whole", "x<speech>y", []string{"x", "<speech>", "y"}},
		{"user-defined symbols never merge", "<<b>>", []string{"<", "<b>", ">"}},
		{"partial user-defined symbol merges normally", "<b", []string{"<b"}},
		{"control piece is never merged to", "<s>", []string{"<", "s>"}},
		{"byte fallback", "é", []string{"0xC3", "0xA9"}},
		{"tab is not a space", "h\ti", []string{"h", "0x09", "i"}},
		{"invalid UTF-8 becomes U+FFFD", "a\xffb", []string{"a", "0xEF", "0xBF", "0xBD", "b"}},
		{"truncated sequence", "\xe2\x82", []string{"0xEF", "0xBF", "0xBD", "0xEF", "0xBF", "0xBD"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.Encode(tt.text)
			if want := ids(t, p, tt.want...); !slices.Equal(got, want) {
				t.Errorf("Encode(%q) = %v, want %v %q", tt.text, got, want, tt.want)
			}
		})
	}
}

func TestEncodeWithoutByteFallback(t *testing.T) {
	p := newTest(t, noBytesModel())
	unk := p.UnkID()
	a := ids(t, p, "a")[0]

	tests := []struct {
		text string
		want []int
	}{
		{"é", []int{unk}},
		{"éé", []int{unk}}, // a run of unknown symbols is one <unk>
		{"éaé", []int{unk, a, unk}},
		{"a\xff", []int{a, unk}},
	}

	for _, tt := range tests {
		if got := p.Encode(tt.text); !slices.Equal(got, tt.want) {
			t.Errorf("Encode(%q) = %v, want %v", tt.text, got, tt.want)
		}
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		name               string
		dummy, extra, esca bool
		in, want           string
	}{
		{"identity", false, false, true, "  a  b\t\n ", "▁▁a▁▁b\t\n▁"},
		{"identity keeps no-break space", false, false, true, "a b", "a b"},
		{"invalid UTF-8", false, false, true, "a\xff\xc3", "a��"},
		{"U+FFFD stays", false, false, true, "�", "�"},
		{"dummy prefix", true, false, true, "a b", "▁a▁b"},
		{"dummy prefix on empty", true, false, true, "", ""},
		{"dummy prefix on space", true, false, true, " ", "▁▁"},
		{"remove extra whitespaces", false, true, true, "  a   b  ", "a▁b"},
		{"remove extra whitespaces, spaces only", true, true, true, "   ", ""},
		{"remove extra whitespaces keeps tabs", false, true, true, " \ta \t b", "\ta▁\t▁b"},
		{"both", true, true, true, "  a   b  ", "▁a▁b"},
		{"no escaping", true, true, false, "  a   b  ", " a b"},
		{"user-defined symbol passes whole", false, true, true, "x <b>x  y", "x▁<b>x▁y"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := defaultTestModel()
			m.normalizer = NormalizerSpec{
				Name: "identity", AddDummyPrefix: tt.dummy, RemoveExtraWhitespaces: tt.extra, EscapeWhitespaces: tt.esca,
			}

			if got := newTest(t, m).normalize(tt.in); got != tt.want {
				t.Errorf("normalize(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestDecode(t *testing.T) {
	p := newTest(t, defaultTestModel())

	tests := []struct {
		name   string
		pieces []string
		want   string
	}{
		{"empty", nil, ""},
		{"controls vanish", []string{"<s>", "hi", "</s>", "<pad>"}, "hi"},
		{"unknown", []string{"<unk>"}, " ⁇ "},
		{"meta spaces", []string{"▁hi", "▁", "▁h"}, " hi  h"},
		{"user-defined", []string{"<b>x", "<speech>"}, "<b>x<speech>"},
		{"byte run", []string{"0xC3", "0xA9", "0xE2", "0x82", "0xAC"}, "é€"},
		{"truncated run", []string{"0xE2", "0x82", "hi"}, "��hi"},
		{"run split by control", []string{"0xC3", "<pad>", "0xA9"}, "��"},
		{"run split by unknown", []string{"0xC3", "<unk>", "0xA9"}, "� ⁇ �"},
		{"meta space from bytes stays", []string{"0xE2", "0x96", "0x81"}, "▁"},
		{"surrogate bytes", []string{"0xED", "0xA0", "0x80"}, "���"},
		{"U+FFFD bytes", []string{"0xEF", "0xBF", "0xBD"}, "�"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := p.Decode(ids(t, p, tt.pieces...))
			if err != nil || got != tt.want {
				t.Errorf("Decode(%q) = %q, %v; want %q", tt.pieces, got, err, tt.want)
			}
		})
	}

	for _, id := range []int{-1, p.Len()} {
		_, err := p.Decode([]int{id})
		if err == nil || !strings.Contains(err.Error(), "out of range") {
			t.Errorf("Decode([%d]) err = %v, want out of range", id, err)
		}
	}
}

func TestDecodeDummyPrefix(t *testing.T) {
	tests := []struct {
		name         string
		dummy, extra bool
		pieces       []string
		want         string
	}{
		{"identity keeps the first space", false, false, []string{"▁hi", "▁h"}, " hi h"},
		{"dummy prefix drops it", true, false, []string{"▁hi", "▁h"}, "hi h"},
		{"after a control", true, false, []string{"<s>", "▁hi"}, "hi"},
		{"not after bytes", true, false, []string{"0x41", "▁hi"}, "A hi"},
		{"not after unknown", true, false, []string{"<unk>", "▁hi"}, " ⁇  hi"},
		{"only once", true, false, []string{"▁", "▁hi"}, " hi"},
		{"extra whitespaces drop every leading one", false, true, []string{"▁", "▁", "▁hi", "▁h"}, "hi h"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := defaultTestModel()
			m.normalizer.AddDummyPrefix, m.normalizer.RemoveExtraWhitespaces = tt.dummy, tt.extra
			p := newTest(t, m)

			got, err := p.Decode(ids(t, p, tt.pieces...))
			if err != nil || got != tt.want {
				t.Errorf("Decode(%q) = %q, %v; want %q", tt.pieces, got, err, tt.want)
			}
		})
	}
}

func TestDecodeUnkSurface(t *testing.T) {
	for _, surface := range []string{"", "<?>"} {
		m := defaultTestModel()
		m.unkSurface = &surface
		p := newTest(t, m)

		got, err := p.Decode([]int{p.UnkID(), p.UnkID()})
		if want := surface + surface; err != nil || got != want {
			t.Errorf("unk_surface %q: Decode = %q, %v; want %q", surface, got, err, want)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	p := newTest(t, defaultTestModel())

	for _, text := range []string{"", "ab hi", "  abc\taad xyz efg ", "<b>x<speech> <s> é€ 🙂", "\x00\x7f"} {
		got, err := p.Decode(p.Encode(text))
		if err != nil || got != text {
			t.Errorf("Decode(Encode(%q)) = %q, %v", text, got, err)
		}
	}
}

func TestScoreKey(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	nan := float32(math.NaN())
	inc := []float32{float32(math.Inf(-1)), -1e30, -2, -1, -1e-30, negZero, 0, 1e-30, 1, 2, float32(math.Inf(1)), nan}

	for i := 1; i < len(inc); i++ {
		if scoreKey(inc[i-1]) >= scoreKey(inc[i]) {
			t.Errorf("scoreKey(%v) = %d not below scoreKey(%v) = %d", inc[i-1], scoreKey(inc[i-1]), inc[i], scoreKey(inc[i]))
		}
	}
}

func FuzzEncodeRoundTrip(f *testing.F) {
	p, err := Parse(defaultTestModel().encode())
	if err != nil {
		f.Fatal(err)
	}

	for _, s := range []string{"", "ab hi", "<b>x", "aad", "xyz", "\xff"} {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, text string) {
		got, err := p.Decode(p.Encode(text))
		if err != nil {
			t.Fatal(err)
		}

		// Decode gives back the normalised text: invalid UTF-8 as U+FFFD and
		// "▁" as a space.
		want := strings.ReplaceAll(strings.ToValidUTF8(text, "�"), "▁", " ")
		if !strings.ContainsRune(text, '�') && got != want {
			t.Errorf("Decode(Encode(%q)) = %q, want %q", text, got, want)
		}
	})
}
