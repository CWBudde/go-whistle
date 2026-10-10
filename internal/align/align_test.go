package align

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// toyPieces is a vocabulary for the unit tests: the four specials, the 256
// byte pieces, then pieces shaped like Whistle's.
var toyPieces = []string{
	"▁\"", "You", "▁was", "▁am", ".\"", "▁m", "ouse", "!\"", "▁mind", ",", "▁", ".", "S", "ora", "▁l", "'", "autre",
}

// toyTokenizer parses toyPieces laid out as export.py's tokenizer blob:
// header "<IIIIIBBH", then per piece "<fBH" and its UTF-8 surface.
func toyTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()

	type piece struct {
		s   string
		typ tokenizer.Type
	}

	pieces := make([]piece, 0, 4+256+len(toyPieces))
	pieces = append(pieces,
		piece{"<pad>", tokenizer.TypeControl}, piece{"</s>", tokenizer.TypeControl}, piece{"<s>", tokenizer.TypeControl},
		piece{"<unk>", tokenizer.TypeUnknown},
	)

	for b := range 256 {
		pieces = append(pieces, piece{fmt.Sprintf("<0x%02X>", b), tokenizer.TypeByte})
	}

	for _, p := range toyPieces {
		pieces = append(pieces, piece{p, tokenizer.TypeNormal})
	}

	var buf bytes.Buffer

	hdr := struct {
		N, Pad, EOS, BOS, UNK uint32
		AddDummy, ByteFB      uint8
		Reserved              uint16
	}{N: uint32(len(pieces)), EOS: 1, BOS: 2, UNK: 3, ByteFB: 1}

	err := binary.Write(&buf, binary.LittleEndian, hdr)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range pieces {
		err = binary.Write(&buf, binary.LittleEndian, struct {
			Score float32
			Type  uint8
			Len   uint16
		}{0, uint8(p.typ), uint16(len(p.s))})
		if err != nil {
			t.Fatal(err)
		}

		buf.WriteString(p.s)
	}

	tok, err := tokenizer.Parse(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}

	return tok
}

// toyIDs maps pieces (or "<0xNN>" byte pieces) to ids.
func toyIDs(t *testing.T, tok *tokenizer.Tokenizer, pieces ...string) []int {
	t.Helper()

	out := make([]int, len(pieces))

	for i, p := range pieces {
		id, ok := tok.PieceID(p)
		if !ok {
			t.Fatalf("piece %q not in the toy vocabulary", p)
		}

		out[i] = id
	}

	return out
}

// TestWords pins the grouping at "▁" and Whisper's punctuation words
// (docs/FINDINGS.md F-033): a leading "▁\"" times with its word, ".\""
// is part of the word, "!\"" and "," are not, a word of punctuation keeps
// its tokens, byte pieces continue a word, "'" splits a Whisper word but
// both halves time it.
func TestWords(t *testing.T) {
	tok := toyTokenizer(t)
	tokens := toyIDs(t, tok,
		"▁\"", "You", "▁was", "▁am", ".\"", "▁m", "ouse", "!\"", "▁mind", ",", "▁", ".", "▁", "<0xC3>", "<0xBC>",
		"S", "ora", "▁l", "'", "autre")

	got, err := Words(tok, tokens)
	if err != nil {
		t.Fatal(err)
	}

	want := []Word{
		{Text: `"You`, Start: 0, End: 2, Core: [2]int{0, 2}},
		{Text: "was", Start: 2, End: 3, Core: [2]int{2, 3}},
		{Text: `am."`, Start: 3, End: 5, Core: [2]int{3, 5}},
		{Text: `mouse!"`, Start: 5, End: 8, Core: [2]int{5, 7}},
		{Text: "mind,", Start: 8, End: 10, Core: [2]int{8, 9}},
		{Text: ".", Start: 10, End: 12, Core: [2]int{10, 12}},
		{Text: "üSora", Start: 12, End: 17, Core: [2]int{12, 17}},
		{Text: "l'autre", Start: 17, End: 20, Core: [2]int{17, 20}},
	}

	if !slices.Equal(got, want) {
		t.Errorf("Words:\n got %+v\nwant %+v", got, want)
	}

	_, err = Words(tok, []int{-1})
	if err == nil {
		t.Error("Words accepted token −1")
	}

	empty, err := Words(tok, nil)
	if err != nil || len(empty) != 0 {
		t.Errorf("Words(nil) = %v, %v", empty, err)
	}
}

func TestProbability(t *testing.T) {
	logp := []float64{math.Log(0.5), math.Log(0.9), math.Log(0.2)}
	w := Word{Start: 0, End: 3, Core: [2]int{0, 2}}

	if p := Probability(logp, w); math.Abs(p-0.7) > 1e-12 {
		t.Errorf("Probability = %v, want the mean 0.7 over the core", p)
	}
}

// TestDTW pins the step patterns on small cost matrices: the first and the
// last column of every row, Whisper's preference of the column move on
// ties, and no path for more rows than columns without sharing.
func TestDTW(t *testing.T) {
	// A cheap diagonal band: rows 0, 1, 2 over columns 0–1, 2–3, 4.
	cost := []float64{
		0, 0, 9, 9, 9,
		9, 9, 0, 0, 9,
		9, 9, 9, 9, 0,
	}

	for _, step := range []Step{StepWhisper, StepNoShare} {
		starts, ends := DTW(cost, 3, 5, step)
		if !slices.Equal(starts, []int{0, 2, 4}) || !slices.Equal(ends, []int{1, 3, 4}) {
			t.Errorf("step %d: starts %v ends %v, want [0 2 4] [1 3 4]", step, starts, ends)
		}
	}

	// Three rows, two columns: Whisper shares a column, NoShare cannot.
	tall := []float64{0, 9, 0, 9, 9, 0}

	starts, ends := DTW(tall, 3, 2, StepWhisper)
	if !slices.Equal(starts, []int{0, 0, 1}) || !slices.Equal(ends, []int{0, 0, 1}) {
		t.Errorf("shared column: starts %v ends %v, want [0 0 1] [0 0 1]", starts, ends)
	}

	if s, e := DTW(tall, 3, 2, StepNoShare); s != nil || e != nil {
		t.Errorf("NoShare with 3 rows on 2 columns: %v %v, want no path", s, e)
	}

	// All costs equal: Whisper's ties take the column move, so the last
	// row stretches back to the first column it can reach.
	starts, ends = DTW(make([]float64, 2*3), 2, 3, StepWhisper)
	if !slices.Equal(starts, []int{0, 0}) || !slices.Equal(ends, []int{0, 2}) {
		t.Errorf("ties: starts %v ends %v, want [0 0] [0 2]", starts, ends)
	}
}

func TestMedianFilter(t *testing.T) {
	m := []float64{1, 5, 2, 8, 3}
	MedianFilter(m, 1, 5, 3)

	// Reflected: 5 1 5 2 8 3 8.
	if want := []float64{5, 2, 5, 3, 8}; !slices.Equal(m, want) {
		t.Errorf("width 3: %v, want %v", m, want)
	}

	m = []float64{1, 5, 2}
	MedianFilter(m, 1, 3, 1)

	if !slices.Equal(m, []float64{1, 5, 2}) {
		t.Errorf("width 1 changed the row: %v", m)
	}
}

func TestNormalise(t *testing.T) {
	m := []float64{1, 2, 3, 4, 5, 9}

	Normalise(m, 3, 2, NormFrame)

	for c := range 2 {
		var s, q float64

		for r := range 3 {
			s += m[r*2+c]
			q += m[r*2+c] * m[r*2+c]
		}

		if math.Abs(s) > 1e-12 || math.Abs(q/3-1) > 1e-12 {
			t.Errorf("column %d: mean %v, variance %v after NormFrame", c, s/3, q/3)
		}
	}
}

func TestOnset(t *testing.T) {
	levels := make([]float64, 40)
	for i := range levels {
		levels[i] = -60
	}

	for i := 13; i < 40; i++ {
		levels[i] = 0
	}

	levels[11] = -10 // within 14 dB of the 99th percentile (0 dB): the onset

	if f, ok := Onset(levels, 14); !ok || f != 1 {
		t.Errorf("Onset = %d, %v; want frame round(11/8) = 1", f, ok)
	}

	if f, ok := Onset(levels, 5); !ok || f != 2 {
		t.Errorf("Onset at 5 dB = %d, %v; want round(13/8) = 2", f, ok)
	}

	if _, ok := Onset(nil, 14); ok {
		t.Error("Onset without levels reported one")
	}
}

// toyInput is a capture of two layers of two heads over 8 frames for the
// positions <s>, language, two text tokens and EOS; position p attends to
// the frames bands[p].
func toyInput(bands [][2]int) Input {
	const frames, heads = 8, 2

	in := Input{First: 2, Tokens: 2, Frames: frames}

	for l := range 2 {
		a := decoder.Attention{Layer: l, Tokens: len(bands), Heads: heads, Frames: frames}
		a.Probs = make([]float32, len(bands)*heads*frames)

		for p, b := range bands {
			for h := range heads {
				row := a.Probs[(p*heads+h)*frames : (p*heads+h+1)*frames]
				for f := range row {
					row[f] = 0.01
					if f >= b[0] && f < b[1] {
						row[f] = 1
					}
				}
			}
		}

		in.Attn = append(in.Attn, a)
	}

	return in
}

// TestTimes runs the default alignment on a synthetic capture: <s> (whose
// query predicts the language token) holds frames 0–2, the language token
// (it predicts the first word) 3–4, the first word (it predicts the
// second) 5–6, the second word (it predicts EOS) 7.
func TestTimes(t *testing.T) {
	tok := toyTokenizer(t)
	tokens := toyIDs(t, tok, "▁was", "▁mind")

	words, err := Words(tok, tokens)
	if err != nil {
		t.Fatal(err)
	}

	in := toyInput([][2]int{{0, 3}, {3, 5}, {5, 7}, {7, 8}, {7, 8}})
	opt := Default()
	opt.Onset = 0

	got, err := Times(in, words, opt)
	if err != nil {
		t.Fatal(err)
	}

	if want := []Span{{3, 5}, {5, 7}}; !slices.Equal(got, want) {
		t.Errorf("Times = %v, want %v", got, want)
	}

	timed, err := Align(tok, tokens, []float64{math.Log(0.5), 0}, in, opt)
	if err != nil {
		t.Fatal(err)
	}

	if timed[0].Text != "was" || math.Abs(timed[0].Probability-0.5) > 1e-12 || timed[1].Start != 5 || timed[1].End != 7 {
		t.Errorf("Align = %+v", timed)
	}

	// The onset moves the first start later, never past its end.
	in.Levels = make([]float64, 64)
	for i := range in.Levels {
		in.Levels[i] = -60
		if i >= 36 {
			in.Levels[i] = 0
		}
	}

	opt.Onset = 14

	got, err = Times(in, words, opt)
	if err != nil || got[0] != (Span{Start: 5, End: 5}) {
		t.Errorf("onset at frame 5: %v, %v; want the first word at 5–5", got, err)
	}

	_, err = Times(Input{First: 2, Tokens: 2, Frames: 8}, words, Default())
	if err == nil {
		t.Error("Times without a capture succeeded")
	}

	bad := Default()
	bad.Cost = CostNegLog

	_, err = Times(in, words, bad)
	if err == nil {
		t.Error("CostNegLog with NormFrameAll validated")
	}
}

func TestLastHalf(t *testing.T) {
	in := toyInput([][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}})

	if got := LastHalf(in.Attn); !slices.Equal(got, []Head{{1, 0}, {1, 1}}) {
		t.Errorf("LastHalf of layers 0, 1 = %v, want layer 1's heads", got)
	}
}
