package spm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

const (
	// needle3ModelPath is Cactus-Compute/needle3 tokenizer/tokenizer.model @
	// 93e88ede as laid out by hub.PinnedNeedle3.
	needle3ModelPath   = "../../models/needle3/tokenizer.model"
	needle3ModelSHA256 = "97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25"
	// needle3CactPath is the deployed model of the same revision; its last
	// record is export.py's RAW tokenizer blob.
	needle3CactPath  = "../../models/needle3/needle3.cact"
	needle3BlobIndex = 580
	// needle3FixturePath is written by scripts/text_tokenizer_fixture.py.
	needle3FixturePath = "../../testdata/text/tokenizer/needle3.jsonl"
)

// needle3Header is the first line of the fixture.
type needle3Header struct {
	Source struct {
		ModelSHA256    string `json:"model_sha256"`
		CactBlobSHA256 string `json:"cact_blob_sha256"`
		Sentencepiece  string `json:"sentencepiece"`
	} `json:"source"`
	Pieces  int `json:"pieces"`
	Special struct {
		Unk int `json:"unk"`
		BOS int `json:"bos"`
		EOS int `json:"eos"`
		Pad int `json:"pad"`
	} `json:"special"`
	VocabDecodeSHA256 string `json:"vocab_decode_sha256"`
}

// needle3Case is one further line: an encode case (text or text_hex, the
// sentencepiece ids, their decode, and RefTokenizer's ids where they differ)
// or a decode case (ids and their decode in text).
type needle3Case struct {
	Kind    string `json:"kind"`
	Name    string `json:"name"`
	Text    string `json:"text"`
	TextHex string `json:"text_hex"`
	IDs     []int  `json:"ids"`
	Decoded string `json:"decoded"`
	Ref     []int  `json:"ref"`
}

// input is the text to encode, from text_hex for the raw-byte cases.
func (c needle3Case) input(t *testing.T) string {
	t.Helper()

	if c.TextHex == "" {
		return c.Text
	}

	b, err := hex.DecodeString(c.TextHex)
	if err != nil {
		t.Fatalf("%s: text_hex: %v", c.Name, err)
	}

	return string(b)
}

// openNeedle3 loads the pinned tokenizer.model, skipping when it is absent.
func openNeedle3(t *testing.T) (*Processor, []byte) {
	t.Helper()

	data, err := os.ReadFile(needle3ModelPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model",
			needle3ModelPath)
	}

	if err != nil {
		t.Fatal(err)
	}

	p, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	return p, data
}

// loadNeedle3Fixture reads the fixture and checks it was made from data.
func loadNeedle3Fixture(t *testing.T, data []byte) (needle3Header, []needle3Case) {
	t.Helper()

	f, err := os.Open(needle3FixturePath)
	if err != nil {
		t.Fatalf("open fixture: %v", err)
	}
	defer f.Close()

	dec := json.NewDecoder(f)

	var hdr needle3Header

	err = dec.Decode(&hdr)
	if err != nil {
		t.Fatalf("fixture header: %v", err)
	}

	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != hdr.Source.ModelSHA256 {
		t.Fatalf("tokenizer.model sha256 %s, fixture was made from %s; regenerate it with scripts/text_tokenizer_fixture.py",
			got, hdr.Source.ModelSHA256)
	}

	var cases []needle3Case

	for {
		var c needle3Case

		err = dec.Decode(&c)
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("fixture case %d: %v", len(cases), err)
		}

		cases = append(cases, c)
	}

	return hdr, cases
}

// byKind splits the fixture cases into encode and decode cases.
func byKind(t *testing.T, cases []needle3Case) (enc, dec []needle3Case) {
	t.Helper()

	for _, c := range cases {
		switch c.Kind {
		case "encode":
			enc = append(enc, c)
		case "decode":
			dec = append(dec, c)
		default:
			t.Fatalf("%s: unknown kind %q", c.Name, c.Kind)
		}
	}

	return enc, dec
}

// TestNeedle3Model pins the vocabulary and specs of tokenizer.model @
// 93e88ede.
func TestNeedle3Model(t *testing.T) {
	p, data := openNeedle3(t)

	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != needle3ModelSHA256 || len(data) != 126520 {
		t.Fatalf("tokenizer.model sha256 %s, %d bytes; want %s, 126520", got, len(data), needle3ModelSHA256)
	}

	if p.Len() != 8192 {
		t.Fatalf("Len = %d, want 8192", p.Len())
	}

	if p.UnkID() != 3 || p.BOSID() != 2 || p.EOSID() != 1 || p.PadID() != 0 {
		t.Errorf("unk/bos/eos/pad = %d/%d/%d/%d, want 3/2/1/0", p.UnkID(), p.BOSID(), p.EOSID(), p.PadID())
	}

	hist := map[Type]int{}
	for id := range p.Len() {
		hist[p.Type(id)]++
	}

	want := map[Type]int{TypeNormal: 7908, TypeByte: 256, TypeUserDefined: 24, TypeControl: 3, TypeUnknown: 1}
	if !mapsEqual(hist, want) {
		t.Errorf("type histogram %v, want %v", hist, want)
	}

	// Ids 4..27 are the user-defined symbols in trainer order, 28..283 the
	// bytes, and the NORMAL pieces score -0, -1, ... by merge rank.
	ts := p.Trainer()
	for i, s := range ts.UserDefinedSymbols {
		if p.Piece(4+i) != s || p.Type(4+i) != TypeUserDefined {
			t.Errorf("id %d = %q %s, want user-defined %q", 4+i, p.Piece(4+i), p.Type(4+i), s)
		}
	}

	for id := 28; id < 284; id++ {
		if p.Type(id) != TypeByte || p.byteVal[id] != int16(id-28) {
			t.Errorf("id %d = %q %s, want byte piece <0x%02X>", id, p.Piece(id), p.Type(id), id-28)
		}
	}

	for id := 284; id < p.Len(); id++ {
		if want := -float32(id - 284); p.Type(id) != TypeNormal || p.Score(id) != want {
			t.Errorf("id %d = %q %s %v, want NORMAL %v", id, p.Piece(id), p.Type(id), p.Score(id), want)
		}
	}
}

// TestNeedle3Specs pins what tokenizer.model configures: BPE with byte
// fallback and the identity normaliser without a dummy prefix, so Encode
// only escapes ' ' as "▁".
func TestNeedle3Specs(t *testing.T) {
	p, _ := openNeedle3(t)

	ts := p.Trainer()
	if ts.ModelType != ModelBPE || ts.VocabSize != 8192 || !ts.ByteFallback || len(ts.UserDefinedSymbols) != 24 ||
		ts.SplitDigits || !ts.SplitByWhitespace || ts.TreatWhitespaceAsSuffix || ts.UnkSurface != " ⁇ " {
		t.Errorf("trainer spec = %+v", ts)
	}

	ns := p.Normalizer()
	if ns.Name != "identity" || ns.AddDummyPrefix || ns.RemoveExtraWhitespaces || !ns.EscapeWhitespaces ||
		len(ns.PrecompiledCharsmap) != 0 {
		t.Errorf("normalizer spec = %+v", ns)
	}
}

func mapsEqual(a, b map[Type]int) bool {
	if len(a) != len(b) {
		return false
	}

	for k, v := range a {
		if b[k] != v {
			return false
		}
	}

	return true
}

// TestNeedle3Fixture checks Encode and Decode against Python sentencepiece
// on the fixture corpus.
func TestNeedle3Fixture(t *testing.T) {
	p, data := openNeedle3(t)
	hdr, cases := loadNeedle3Fixture(t, data)
	enc, dec := byKind(t, cases)

	if hdr.Pieces != p.Len() || hdr.Special.Unk != p.UnkID() || hdr.Special.BOS != p.BOSID() ||
		hdr.Special.EOS != p.EOSID() || hdr.Special.Pad != p.PadID() {
		t.Errorf("fixture pieces %d, specials %+v; Processor has %d, %d/%d/%d/%d", hdr.Pieces, hdr.Special,
			p.Len(), p.UnkID(), p.BOSID(), p.EOSID(), p.PadID())
	}

	if len(enc) < 100 || len(dec) < 60 {
		t.Fatalf("fixture has %d encode and %d decode cases, want at least 100 and 60", len(enc), len(dec))
	}

	nIDs := 0

	for _, c := range enc {
		text := c.input(t)
		nIDs += len(c.IDs)

		got := p.Encode(text)
		if !slices.Equal(got, c.IDs) {
			t.Errorf("%s: Encode(%q)\n got %v\nwant %v", c.Name, text, got, c.IDs)

			continue
		}

		back, err := p.Decode(got)
		if err != nil || back != c.Decoded {
			t.Errorf("%s: Decode(Encode) = %q, %v; want %q", c.Name, back, err, c.Decoded)
		}

		// Decoding gives the text back unless it holds "▁", which reads as
		// a space, or invalid UTF-8, which reads as U+FFFD.
		if c.TextHex == "" && !strings.Contains(text, metaSpace) && back != text {
			t.Errorf("%s: round trip %q -> %q", c.Name, text, back)
		}
	}

	for _, c := range dec {
		nIDs += len(c.IDs)

		got, err := p.Decode(c.IDs)
		if err != nil || got != c.Text {
			t.Errorf("%s: Decode(%v) = %q, %v; want %q", c.Name, c.IDs, got, err, c.Text)
		}
	}

	t.Logf("%d encode texts and %d decode sequences (%d ids) equal sentencepiece %s",
		len(enc), len(dec), nIDs, hdr.Source.Sentencepiece)

	if got := vocabDecodeDigest(t, p); got != hdr.VocabDecodeSHA256 {
		t.Errorf("single-id decode digest %s, want %s", got, hdr.VocabDecodeSHA256)
	}
}

// vocabDecodeDigest is the fixture's vocab_decode_sha256: sha256 over the
// decode of every single id, each as a u32 LE length and the UTF-8 bytes.
func vocabDecodeDigest(t *testing.T, p *Processor) string {
	t.Helper()

	h := sha256.New()

	for id := range p.Len() {
		s, err := p.Decode([]int{id})
		if err != nil {
			t.Fatal(err)
		}

		h.Write(binary.LittleEndian.AppendUint32(nil, uint32(len(s))))
		h.Write([]byte(s))
	}

	return hex.EncodeToString(h.Sum(nil))
}

// TestNeedle3VersusRefTokenizer compares the two candidate tokenizers of
// the Needle 3 engine: sentencepiece over tokenizer.model (this package) and
// upstream export.py RefTokenizer over needle3.cact's RAW tokenizer blob
// (internal/tokenizer). It checks that the blob holds the same pieces,
// scores and types, and that both encode every fixture text alike, except
// for invalid UTF-8, which sentencepiece reads as U+FFFD and
// internal/tokenizer spells byte by byte. Decode differs only for <unk>,
// which sentencepiece prints as " ⁇ " and RefTokenizer drops.
func TestNeedle3VersusRefTokenizer(t *testing.T) {
	p, data := openNeedle3(t)
	ref, blob := openNeedle3Blob(t)
	hdr, cases := loadNeedle3Fixture(t, data)
	enc, dec := byKind(t, cases)

	sum := sha256.Sum256(blob)
	if got := hex.EncodeToString(sum[:]); got != hdr.Source.CactBlobSHA256 {
		t.Errorf("tokenizer blob sha256 %s, fixture recorded %s", got, hdr.Source.CactBlobSHA256)
	}

	// (c) The blob is the same vocabulary, bit for bit.
	if ref.Len() != p.Len() || ref.PadID() != p.PadID() || ref.EOSID() != p.EOSID() || ref.BOSID() != p.BOSID() ||
		ref.UnkID() != p.UnkID() || ref.AddDummyPrefix() != p.Normalizer().AddDummyPrefix ||
		ref.ByteFallback() != p.Trainer().ByteFallback {
		t.Fatalf("blob header differs: %d pieces, pad/eos/bos/unk %d/%d/%d/%d, dummy %v, byte fallback %v",
			ref.Len(), ref.PadID(), ref.EOSID(), ref.BOSID(), ref.UnkID(), ref.AddDummyPrefix(), ref.ByteFallback())
	}

	refType := map[tokenizer.Type]Type{
		tokenizer.TypeNormal: TypeNormal, tokenizer.TypeUnknown: TypeUnknown, tokenizer.TypeControl: TypeControl,
		tokenizer.TypeUserDefined: TypeUserDefined, tokenizer.TypeByte: TypeByte,
	}
	for id := range p.Len() {
		if ref.Piece(id) != p.Piece(id) || math.Float32bits(ref.Score(id)) != math.Float32bits(p.Score(id)) ||
			refType[ref.Type(id)] != p.Type(id) {
			t.Errorf("id %d: blob (%q, %v, %s), tokenizer.model (%q, %v, %s)", id, ref.Piece(id), ref.Score(id),
				ref.Type(id), p.Piece(id), p.Score(id), p.Type(id))
		}
	}

	// (b) Encode: internal/tokenizer must reproduce upstream RefTokenizer
	// (fixture `ref`, absent where it equals sentencepiece), and both must
	// equal sentencepiece on valid text.
	same, diffs := 0, []string(nil)

	for _, c := range enc {
		got := ref.Encode(c.input(t))
		if slices.Equal(got, c.IDs) {
			same++

			continue
		}

		diffs = append(diffs, c.Name)

		switch {
		case c.Ref != nil && !slices.Equal(got, c.Ref):
			t.Errorf("%s: internal/tokenizer %v, upstream RefTokenizer %v", c.Name, got, c.Ref)
		case c.Ref == nil && c.TextHex == "":
			t.Errorf("%s: internal/tokenizer %v, sentencepiece %v", c.Name, got, c.IDs)
		}
	}

	t.Logf("encode: %d/%d texts equal; the rest are invalid UTF-8: %q", same, len(enc), diffs)

	// Decode differs only where sentencepiece prints <unk> as " ⁇ ".
	same = 0

	for _, c := range dec {
		got, err := ref.Decode(c.IDs)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}

		if got == c.Text {
			same++

			continue
		}

		if !slices.Contains(c.IDs, p.UnkID()) || got != strings.ReplaceAll(c.Text, " ⁇ ", "") {
			t.Errorf("%s: internal/tokenizer decodes %v to %q, sentencepiece to %q", c.Name, c.IDs, got, c.Text)
		}
	}

	t.Logf("decode: %d/%d sequences equal; the rest hold <unk>", same, len(dec))
}

// TestNeedle3VersusRefTokenizerRandom extends the comparison to seeded
// random texts glued from piece surfaces (with "▁" as a space), markers,
// spelled-out special and byte pieces, whitespace and random code points.
func TestNeedle3VersusRefTokenizerRandom(t *testing.T) {
	p, _ := openNeedle3(t)
	ref, _ := openNeedle3Blob(t)

	rng := rand.New(rand.NewPCG(20261011, 1))
	extra := []string{" ", "  ", "\t", "\n", "<", ">", "<s>", "</s>", "<unk>", "<pad>", "<0x41>", "▁", "0x", "|"}

	const n = 20000

	same := 0

	for range n {
		var sb strings.Builder

		for range 1 + rng.IntN(12) {
			switch k := rng.IntN(10); {
			case k < 6:
				sb.WriteString(strings.ReplaceAll(p.Piece(rng.IntN(p.Len())), metaSpace, " "))
			case k < 9:
				sb.WriteString(extra[rng.IntN(len(extra))])
			default:
				sb.WriteRune(rune(rng.IntN(0x30000)))
			}
		}

		// WriteRune spells surrogates as U+FFFD, so text is valid UTF-8;
		// the fixture's raw-byte cases cover invalid input.
		text := sb.String()
		if got, want := ref.Encode(text), p.Encode(text); slices.Equal(got, want) {
			same++
		} else {
			t.Errorf("Encode(%q): internal/tokenizer %v, sentencepiece port %v", text, got, want)
		}
	}

	t.Logf("%d/%d random texts encode equally", same, n)
}

// openNeedle3Blob parses needle3.cact's RAW tokenizer blob with
// internal/tokenizer, skipping when the archive is absent.
func openNeedle3Blob(t *testing.T) (*tokenizer.Tokenizer, []byte) {
	t.Helper()

	_, err := os.Stat(needle3CactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model",
			needle3CactPath)
	}

	f, err := cact.Open(needle3CactPath)
	if err != nil {
		t.Fatalf("cact.Open: %v", err)
	}

	if f.Len() != needle3BlobIndex+1 {
		t.Fatalf("needle3.cact has %d records, want %d", f.Len(), needle3BlobIndex+1)
	}

	blob, err := f.Raw(needle3BlobIndex)
	if err != nil {
		t.Fatalf("Raw(%d): %v", needle3BlobIndex, err)
	}

	tok, err := tokenizer.Parse(blob)
	if err != nil {
		t.Fatalf("tokenizer.Parse: %v", err)
	}

	return tok, blob
}

func BenchmarkNeedle3Encode(b *testing.B) {
	data, err := os.ReadFile(needle3ModelPath)
	if err != nil {
		b.Skipf("%s missing", needle3ModelPath)
	}

	p, err := Parse(data)
	if err != nil {
		b.Fatal(err)
	}

	text := strings.Repeat("<|im_start|>user\n<tools>[{\"name\":\"get_weather\",\"parameters\":{}}]</tools>\n"+
		"What's the weather like in Berlin tomorrow?<|im_end|>\n", 8)

	b.SetBytes(int64(len(text)))

	for b.Loop() {
		p.Encode(text)
	}
}
