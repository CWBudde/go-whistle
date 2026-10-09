package tokenizer

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
)

const (
	// whistleCactPath is the pinned model file as laid out by `whistle download`.
	whistleCactPath = "../../models/whistle/whistle.cact"
	// fixturePath is written by scripts/tokenizer_fixture.py (F-005).
	fixturePath = "../../testdata/tokenizer/whistle.json"
)

// openWhistle loads the tokenizer of whistle.cact and returns it with the
// raw blob.
func openWhistle(t *testing.T) (*Tokenizer, []byte) {
	t.Helper()

	_, err := os.Stat(whistleCactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", whistleCactPath)
	}

	f, err := cact.Open(whistleCactPath)
	if err != nil {
		t.Fatalf("cact.Open: %v", err)
	}

	tok, err := FromCact(f)
	if err != nil {
		t.Fatalf("FromCact: %v", err)
	}

	blob, err := f.Raw(680)
	if err != nil {
		t.Fatalf("Raw(680): %v", err)
	}

	return tok, blob
}

// fixture mirrors testdata/tokenizer/whistle.json.
type fixture struct {
	Source struct {
		BlobSHA256 string `json:"blob_sha256"`
		Pieces     int    `json:"pieces"`
	} `json:"source"`
	VocabDecodeSHA256 string `json:"vocab_decode_sha256"`
	Decode            []struct {
		Name          string `json:"name"`
		IDs           []int  `json:"ids"`
		Sentencepiece string `json:"sentencepiece"`
		Ref           string `json:"ref"`
	} `json:"decode"`
	Encode []struct {
		Name string `json:"name"`
		Text string `json:"text"`
		Ref  []int  `json:"ref"`
	} `json:"encode"`
}

func loadFixture(t *testing.T, blob []byte) fixture {
	t.Helper()

	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	var fx fixture

	err = json.Unmarshal(data, &fx)
	if err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	sum := sha256.Sum256(blob)
	if got := hex.EncodeToString(sum[:]); got != fx.Source.BlobSHA256 {
		t.Fatalf("tokenizer blob sha256 %s, fixture was made from %s; regenerate it with `just tokenizer-fixture`",
			got, fx.Source.BlobSHA256)
	}

	return fx
}

// TestWhistleTokenizer pins the vocabulary of whistle.cact @ b358ddad.
func TestWhistleTokenizer(t *testing.T) {
	tok, _ := openWhistle(t)

	if tok.Len() != 8199 {
		t.Fatalf("Len = %d, want 8199", tok.Len())
	}

	if tok.PadID() != 0 || tok.EOSID() != 1 || tok.BOSID() != 2 || tok.UnkID() != 3 {
		t.Errorf("specials pad/eos/bos/unk = %d/%d/%d/%d, want 0/1/2/3",
			tok.PadID(), tok.EOSID(), tok.BOSID(), tok.UnkID())
	}

	if tok.AddDummyPrefix() || !tok.ByteFallback() {
		t.Errorf("AddDummyPrefix, ByteFallback = %v, %v; want false, true", tok.AddDummyPrefix(), tok.ByteFallback())
	}

	hist := map[Type]int{}

	for id := range tok.Len() {
		hist[tok.Type(id)]++
	}

	wantHist := map[Type]int{TypeNormal: 7908, TypeByte: 256, TypeUserDefined: 31, TypeControl: 3, TypeUnknown: 1}
	for typ, n := range wantHist {
		if hist[typ] != n {
			t.Errorf("%d %s pieces, want %d", hist[typ], typ, n)
		}
	}

	pinned := []struct {
		id    int
		piece string
		typ   Type
		score float32
	}{
		{0, "<pad>", TypeControl, 0},
		{3, "<unk>", TypeUnknown, 0},
		{4, "<|im_start|>", TypeUserDefined, 0},
		{18, "<speech>", TypeUserDefined, 0},
		{19, "</speech>", TypeUserDefined, 0},
		{28, "<0x00>", TypeByte, 0},
		{283, "<0xFF>", TypeByte, 0},
		{284, "en", TypeNormal, 0},
		{286, "▁t", TypeNormal, -2},
		{8191, "ę", TypeNormal, -7907},
		{8192, "<|en|>", TypeUserDefined, 0},
		{8198, "<|pl|>", TypeUserDefined, 0},
	}
	for _, p := range pinned {
		if tok.Piece(p.id) != p.piece || tok.Type(p.id) != p.typ || tok.Score(p.id) != p.score {
			t.Errorf("id %d = (%q, %s, %v), want (%q, %s, %v)",
				p.id, tok.Piece(p.id), tok.Type(p.id), tok.Score(p.id), p.piece, p.typ, p.score)
		}
	}

	langs := []string{"en", "de", "fr", "es", "it", "nl", "pl"}
	for k, lang := range langs {
		id, ok := tok.PieceID("<|" + lang + "|>")
		if !ok || id != 8192+k {
			t.Errorf("language token %s = %d, %v; want %d", lang, id, ok, 8192+k)
		}
	}
}

// TestWhistleDecodeMatchesSentencepiece decodes the fixture's id sequences
// and every single id, and compares with sentencepiece on a model rebuilt
// from the same blob (F-005).
func TestWhistleDecodeMatchesSentencepiece(t *testing.T) {
	tok, blob := openWhistle(t)
	fx := loadFixture(t, blob)

	if fx.Source.Pieces != tok.Len() {
		t.Fatalf("fixture has %d pieces, tokenizer %d", fx.Source.Pieces, tok.Len())
	}

	ids, refDiffers := 0, 0

	for _, c := range fx.Decode {
		got, err := tok.Decode(c.IDs)
		if err != nil {
			t.Errorf("%s: Decode: %v", c.Name, err)

			continue
		}

		if got != c.Sentencepiece {
			t.Errorf("%s: Decode(%v) = %q, sentencepiece %q", c.Name, c.IDs, got, c.Sentencepiece)
		}

		ids += len(c.IDs)

		if c.Ref != c.Sentencepiece {
			refDiffers++
		}
	}

	h := sha256.New()

	for id := range tok.Len() {
		s, err := tok.Decode([]int{id})
		if err != nil {
			t.Fatalf("Decode([%d]): %v", id, err)
		}

		h.Write(binary.LittleEndian.AppendUint32(nil, uint32(len(s))))
		h.Write([]byte(s))
	}

	if got := hex.EncodeToString(h.Sum(nil)); got != fx.VocabDecodeSHA256 {
		t.Errorf("single-id decode of all %d ids hashes to %s, sentencepiece to %s", tok.Len(), got, fx.VocabDecodeSHA256)
	}

	t.Logf("%d sequences (%d ids) and all %d single ids match sentencepiece; RefTokenizer differs on %d sequences",
		len(fx.Decode), ids, tok.Len(), refDiffers)
}

// TestWhistleEncodeMatchesRef encodes the fixture's multilingual samples and
// compares with upstream RefTokenizer, then decodes them back.
func TestWhistleEncodeMatchesRef(t *testing.T) {
	tok, blob := openWhistle(t)
	fx := loadFixture(t, blob)

	if len(fx.Encode) == 0 {
		t.Fatal("fixture has no encode samples")
	}

	for _, c := range fx.Encode {
		t.Run(c.Name, func(t *testing.T) {
			ids := tok.Encode(c.Text)
			if !slices.Equal(ids, c.Ref) {
				t.Errorf("Encode(%q) = %v, RefTokenizer %v", c.Text, ids, c.Ref)
			}

			got, err := tok.Decode(ids)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}

			if got != c.Text {
				t.Errorf("Decode(Encode(%q)) = %q", c.Text, got)
			}
		})
	}
}

// TestWhistleRoundTrip round-trips text beyond the fixture: every script the
// fixture covers plus long and marker-heavy input.
func TestWhistleRoundTrip(t *testing.T) {
	tok, _ := openWhistle(t)

	texts := []string{
		"Hello, world!",
		"Ärger über Öl: Größe zählt. Œuvre, façade, naïve, Søren, Łódź, Ğ, ı, ß.",
		"Съешь же ещё этих мягких французских булок, да выпей чаю.",
		"日本語のテキストと中文文本，还有한국어도 있습니다.",
		"العربية: النص يُكتب من اليمين إلى اليسار ١٢٣",
		"Emoji \U0001F9D1\U0001F3FE\u200d\U0001F4BB flags 🇫🇷🇵🇱 and ZWJ \U0001F3F3\ufe0f\u200d\U0001F308, plus \u00a0nbsp and \u200bzwsp.",
		"<|im_start|>system\n<speech></speech><|de|><|im_end|>",
		"<|im_start <speech <|xx|> half markers",
		"   ",
		"\x00\x01\x7f",
		string(slices.Repeat([]byte("Wiederholung "), 64)),
	}

	for _, text := range texts {
		ids := tok.Encode(text)

		got, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode(Encode(%q)): %v", text, err)
		}

		if got != text {
			t.Errorf("Decode(Encode(%q)) = %q (ids %v)", text, got, ids)
		}
	}
}
