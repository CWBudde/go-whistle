package search

import (
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// The case-folded matchers of the F-044 probe (docs/FINDINGS.md): the
// engine gives "Paare", "paare" and "PAARE" the same text, while
// Keywords matches token ids, the first letter's two cases only. Both are
// off in DefaultBias.

// foldMode is how a keyword's case is matched (Bias.fold).
type foldMode int

const (
	// foldNone matches the spellings of Bias.Forms on token ids.
	foldNone foldMode = iota
	// foldCases adds the keyword lower-cased, and lower-cased with its
	// first letter upper-cased, to the spellings, still on token ids.
	foldCases
	// foldBytes matches the keyword over the case-folded bytes of the
	// text, whatever tokens spell them (byteKeywords).
	foldBytes
)

// foldShare is how byteKeywords gives a matched keyword its token bonus
// (Bias.share).
type foldShare int

const (
	// shareCanon gives each token of the keyword's own spelling, as
	// written, Weight on its last byte (a token that is only the space on
	// the next token's): the keyword's own tokens earn what Keywords
	// gives them, any other spelling the same in total once it reaches
	// the end of one of those tokens.
	shareCanon foldShare = iota
	// shareChar gives Weight to every character of the keyword, spaces
	// not counted.
	shareChar
	// shareToken gives Weight to every token of the text that advances a
	// match by more than a space, however many bytes it spells.
	shareToken
)

// The symbols of byteKeywords besides the bytes 0–255.
const (
	symStart = 256 + iota // the start of the text, in front of a bare spelling
	symOther              // a token that spells no text (a control token)
	symEnd                // EOS
)

// byteKeywords is an Aho-Corasick automaton over the case-folded bytes of
// the keywords' spellings: after a space, and bare behind symStart. A
// token of the text advances it byte by byte, so every spelling of a
// keyword in tokens matches it. Its bookkeeping is Keywords' per byte:
// each matched byte holds the bonus it earned, a failure takes back
// Bias.Revoke of the dropped bytes' bonus, and a completed keyword's
// bytes keep theirs. Byte pieces are matched as they are, so an
// upper-case letter outside ASCII spelled in byte pieces is not folded.
type byteKeywords struct {
	next  []map[int]int
	fail  []int
	depth []int
	w     []float64 // the bonus of the byte into the node (shareCanon, shareChar)
	out   [][]int   // the byte lengths of the spellings that end at the node, suffixes included
	end   []bool
	key   []int   // the keyword a spelling that ends at the node spells
	via   [][]int // the keywords whose spellings pass through the node (Bias.spent)
	syms  [][]int // the folded symbols of every vocabulary token
	eos   int
	bias  Bias
}

// byteState is byteKeywords' state after a text.
type byteState struct {
	n    int
	earn []float64 // the bonus each byte of the match holds, 0 once a completed keyword covers it
	pend []int     // the spellings that ended at the last byte and wait for a word end, by length (Bias.wordEnd)
	done []int     // the keywords completed so far (Bias.spent)
}

// foldSymbols is the symbols a token spells: its piece with "▁" as a
// space, lower-cased, or its byte; symOther for a token that spells no
// text.
func foldSymbols(tok *tokenizer.Tokenizer, id int) []int {
	switch tok.Type(id) {
	case tokenizer.TypeByte:
		b, ok := bytePiece(tok.Piece(id))
		if !ok {
			return []int{symOther}
		}

		return []int{int(b)}
	case tokenizer.TypeNormal, tokenizer.TypeUserDefined:
		s := strings.ToLower(strings.ReplaceAll(tok.Piece(id), "▁", " "))
		out := make([]int, len(s))

		for i := range len(s) {
			out[i] = int(s[i])
		}

		return out
	default:
		return []int{symOther}
	}
}

// bytePiece is the byte of a "<0xNN>" piece.
func bytePiece(p string) (byte, bool) {
	if len(p) != 6 || !strings.HasPrefix(p, "<0x") || p[5] != '>' {
		return 0, false
	}

	var b byte

	for _, c := range p[3:5] {
		var v byte

		switch {
		case c >= '0' && c <= '9':
			v = byte(c - '0')
		case c >= 'A' && c <= 'F':
			v = byte(c - 'A' + 10)
		case c >= 'a' && c <= 'f':
			v = byte(c - 'a' + 10)
		default:
			return 0, false
		}

		b = b<<4 | v
	}

	return b, true
}

// newByteKeywords builds the byte automaton of the keywords under b.
func newByteKeywords(tok *tokenizer.Tokenizer, keywords []string, b Bias) *byteKeywords {
	k := &byteKeywords{
		next: []map[int]int{{}}, fail: []int{0}, depth: []int{0}, w: []float64{0}, end: []bool{false}, key: []int{-1},
		via: [][]int{nil}, eos: tok.EOSID(), bias: b,
	}

	key := 0

	for _, kw := range keywords {
		words := []string{kw}
		if b.Split {
			words = strings.Fields(kw)
		}

		for _, word := range words {
			if b.Forms&FormSpace != 0 {
				syms, ws := k.spelling(tok, " "+word)
				k.add(syms, ws, key)
			}

			if b.Forms&FormBare != 0 {
				syms, ws := k.spelling(tok, word)
				k.add(append([]int{symStart}, syms...), append([]float64{0}, ws...), key)
			}

			key++
		}
	}

	k.link()

	k.syms = make([][]int, tok.Len())
	for id := range k.syms {
		k.syms[id] = foldSymbols(tok, id)
	}

	return k
}

// spelling is the folded symbols of s and each symbol's bonus under
// shareCanon or shareChar (shareToken gives it per token of the text).
func (k *byteKeywords) spelling(tok *tokenizer.Tokenizer, s string) ([]int, []float64) {
	folded := strings.ToLower(s)
	syms := make([]int, len(folded))

	for i := range len(folded) {
		syms[i] = int(folded[i])
	}

	ws := make([]float64, len(syms))

	switch k.bias.share {
	case shareChar:
		for i := range len(folded) {
			if c := folded[i]; c != ' ' && utf8.RuneStart(c) {
				ws[i] = k.bias.Weight
			}
		}
	case shareCanon:
		k.canonWeights(tok, s, syms, ws)
	}

	return syms, ws
}

// canonWeights writes into ws the shareCanon bonus of the symbols syms of
// s: each token of s's own spelling puts Weight on its last byte, and a
// token that spells only the space hands its Weight to the next token, so
// that every spelling after a space holds nothing on the space they
// share. Lower-casing keeps the byte length of the letters in use; a
// spelling where it does not spreads the whole evenly.
func (k *byteKeywords) canonWeights(tok *tokenizer.Tokenizer, s string, syms []int, ws []float64) {
	ids := tok.Encode(s)
	spans := make([]int, 0, len(ids))
	total := 0

	for _, id := range ids {
		n := 1
		if tok.Type(id) != tokenizer.TypeByte {
			n = len(strings.ReplaceAll(tok.Piece(id), "▁", " "))
		}

		spans = append(spans, n)
		total += n
	}

	if total != len(syms) {
		for i := range ws {
			ws[i] = k.bias.Weight * float64(len(spans)) / float64(len(ws))
		}

		return
	}

	at, carry := 0, 0.0

	for _, n := range spans {
		at += n
		carry += k.bias.Weight

		if n == 1 && syms[at-1] == ' ' && at < len(syms) {
			continue
		}

		ws[at-1], carry = carry, 0
	}
}

// add inserts one spelling of keyword key.
func (k *byteKeywords) add(syms []int, ws []float64, key int) {
	if len(syms) == 0 || (len(syms) == 1 && syms[0] == symStart) {
		return
	}

	n := 0

	for i, s := range syms {
		c, ok := k.next[n][s]
		if !ok {
			c = len(k.next)
			k.next = append(k.next, map[int]int{})
			k.fail = append(k.fail, 0)
			k.depth = append(k.depth, k.depth[n]+1)
			k.w = append(k.w, ws[i])
			k.end = append(k.end, false)
			k.key = append(k.key, -1)
			k.via = append(k.via, nil)
			k.next[n][s] = c
		}

		if !slices.Contains(k.via[c], key) {
			k.via[c] = append(k.via[c], key)
		}

		n = c
	}

	if !k.end[n] {
		k.end[n], k.key[n] = true, key
	}
}

// link sets the failure links and the spellings that end at each node.
func (k *byteKeywords) link() { k.out = link(k.next, k.fail, k.end, k.depth) }

// suffix is the node of the last l symbols of the match at c.
func (k *byteKeywords) suffix(c, l int) int {
	for k.depth[c] > l {
		c = k.fail[c]
	}

	return c
}

// spent reports whether every keyword through node c has completed
// already, under Bias.spent.
func (k *byteKeywords) spent(c int, done []int) bool {
	if !k.bias.spent {
		return false
	}

	for _, key := range k.via[c] {
		if !slices.Contains(done, key) {
			return false
		}
	}

	return true
}

// wordEnd reports whether symbol s ends a word: anything but an ASCII
// letter or digit, and the bytes of a multi-byte character count as
// letters.
func wordEnd(s int) bool {
	switch {
	case s >= 256:
		return true
	case s >= 0x80:
		return false
	default:
		c := byte(s)

		return (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9')
	}
}

// token is the state after one token, given by its symbols, at st and the
// bonus change.
func (k *byteKeywords) token(st byteState, syms []int) (byteState, float64) {
	d := 0.0
	earned := -1 // under shareToken: the match position that holds this token's bonus

	for _, s := range syms {
		var dd float64

		st, dd, earned = k.symbol(st, s, earned)
		d += dd
	}

	return st, d
}

// complete gives the spelling of length l that ends at the match's last
// byte its completion, unless Bias.spent and its keyword completed
// before: Bias.Complete, its bytes covered in earn, written in place.
func (k *byteKeywords) complete(n, l int, earn []float64, done []int) (float64, []int) {
	key := k.key[k.suffix(n, l)]
	if k.bias.spent && slices.Contains(done, key) {
		return 0, done
	}

	clear(earn[len(earn)-l:])

	if k.bias.spent {
		done = append(slices.Clip(done), key)
	}

	return k.bias.Complete, done
}

// resolve completes the spellings that wait for a word end at st when s
// ends the word (Bias.wordEnd), and returns the state and the bonus.
func (k *byteKeywords) resolve(st byteState, s int) (byteState, float64) {
	if len(st.pend) == 0 || !wordEnd(s) {
		return st, 0
	}

	d := 0.0
	earn := slices.Clone(st.earn)

	for _, l := range st.pend {
		var c float64

		c, st.done = k.complete(st.n, l, earn, st.done)
		d += c
	}

	st.earn = earn

	return st, d
}

// symbol is the state after symbol s at st and the bonus change; earned
// is the position of the current token's bonus under shareToken (-1 when
// it has none yet).
func (k *byteKeywords) symbol(st byteState, s, earned int) (byteState, float64, int) {
	st, d := k.resolve(st, s)
	done := st.done

	n, f := st.n, st.n

	c, ok := k.next[n][s]
	for !ok && f != 0 {
		f = k.fail[f]
		c, ok = k.next[f][s]
	}

	drop := k.depth[n] - k.depth[f]

	lost := 0.0
	for _, e := range st.earn[:drop] {
		lost += e
	}

	d -= k.bias.Revoke * lost

	if earned >= 0 {
		earned -= drop
	}

	if !ok {
		return byteState{done: done}, d, -1
	}

	earn := make([]float64, 0, k.depth[c])
	earn = append(earn, st.earn[drop:]...)

	w := k.w[c]
	if k.bias.share == shareToken {
		w = 0
		if earned < 0 && s != ' ' && s < 256 {
			w, earned = k.bias.Weight, len(earn)
		}
	}

	if k.spent(c, done) {
		w = 0
	}

	earn = append(earn, w)
	d += w
	next := byteState{n: c, earn: earn, done: done}

	for _, l := range k.out[c] {
		if k.bias.wordEnd {
			next.pend = append(next.pend, l)

			continue
		}

		var cb float64

		cb, next.done = k.complete(c, l, earn, next.done)
		d += cb
	}

	return next, d, earned
}

// start is the state at the start of the text.
func (k *byteKeywords) start() byteState {
	st, _ := k.token(byteState{}, []int{symStart})

	return st
}

// adjust adds to logp every token's change of the bonus after text.
func (k *byteKeywords) adjust(text []int, logp []float64) {
	st := k.start()
	for _, id := range text {
		st, _ = k.token(st, k.syms[id])
	}

	for id := range logp {
		syms := []int{symEnd}
		if id != k.eos {
			syms = k.syms[id]
		}

		_, d := k.token(st, syms)
		logp[id] += d
	}
}

// bonus is the bonus of a whole text, with EOS at the end when eos ≥ 0.
func (k *byteKeywords) bonus(text []int, eos int) float64 {
	st, sum := k.start(), 0.0

	for _, id := range text {
		var d float64

		st, d = k.token(st, k.syms[id])
		sum += d
	}

	if eos >= 0 {
		_, d := k.token(st, []int{symEnd})
		sum += d
	}

	return sum
}
