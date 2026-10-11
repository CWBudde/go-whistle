package spm

import (
	"container/heap"
	"math"
	"strings"
	"unicode/utf8"
)

// maxResegmentDepth is bpe_model.cc kMaxBpeResegmentDepth.
const maxResegmentDepth = 100

// Encode turns text into ids as sentencepiece's EncodeOptimized, without
// BOS or EOS (upstream run.py and finetune.py prepend BOS themselves). The
// text is normalised (normalize), split into user-defined symbols and single
// characters, and BPE-merged (bpe). A resulting symbol that is no piece is
// spelled as its UTF-8 byte pieces under byte_fallback, and otherwise becomes
// the UNKNOWN id, one for each run of such symbols.
func (p *Processor) Encode(text string) []int {
	norm := p.normalize(text)
	if norm == "" {
		return nil
	}

	var (
		out     []int
		prevUnk bool
	)

	for _, w := range p.bpe(norm) {
		id, ok := p.pieceID[w]
		if !ok {
			id = p.unk
		}

		switch typ := p.model.Pieces[id].Type; {
		case typ == TypeControl:
			out = append(out, id)
			prevUnk = false
		case typ == TypeUnknown && p.model.Trainer.ByteFallback:
			for i := range len(w) {
				out = append(out, p.byteID[w[i]])
			}

			prevUnk = true
		case typ == TypeUnknown && prevUnk:
			// sentencepiece merges a run of unknown symbols into one <unk>.
		default:
			out = append(out, id)
			prevUnk = typ == TypeUnknown
		}
	}

	return out
}

// normalize is sentencepiece's Normalizer::Normalize without a precompiled
// charsmap, i.e. the identity normaliser: user-defined symbols pass whole,
// each invalid UTF-8 byte becomes U+FFFD, and with escape_whitespaces every
// ' ' (only U+0020; tabs and newlines stay) becomes "▁". add_dummy_prefix
// prepends one space; remove_extra_whitespaces drops leading and trailing
// spaces and squeezes runs of them.
func (p *Processor) normalize(text string) string {
	ns := p.model.Normalizer

	if ns.RemoveExtraWhitespaces {
		for text != "" {
			sp, n := p.normalizePrefix(text)
			if sp != " " {
				break
			}

			text = text[n:]
		}
	}

	if text == "" {
		return ""
	}

	space := " "
	if ns.EscapeWhitespaces {
		space = metaSpace
	}

	var sb strings.Builder

	sb.Grow(len(text) + len(text)/2)

	if ns.AddDummyPrefix {
		sb.WriteString(space)
	}

	prevSpace := ns.RemoveExtraWhitespaces

	for text != "" {
		sp, n := p.normalizePrefix(text)
		if prevSpace {
			sp = strings.TrimLeft(sp, " ")
		}

		if sp != "" {
			for i := range len(sp) {
				if sp[i] == ' ' {
					sb.WriteString(space)
				} else {
					sb.WriteByte(sp[i])
				}
			}

			prevSpace = strings.HasSuffix(sp, " ")
		}

		text = text[n:]

		if !ns.RemoveExtraWhitespaces {
			prevSpace = false
		}
	}

	out := sb.String()
	if ns.RemoveExtraWhitespaces {
		for strings.HasSuffix(out, space) {
			out = out[:len(out)-len(space)]
		}
	}

	return out
}

// normalizePrefix is Normalizer::NormalizePrefix without a charsmap: the
// longest user-defined symbol at the start of s, else its first character,
// else, for a byte that does not start valid UTF-8, U+FFFD. It returns the
// normalised text and the number of bytes of s it consumed.
func (p *Processor) normalizePrefix(s string) (string, int) {
	if n := p.users.match(s); n > 0 {
		return s[:n], n
	}

	r, size := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError && size == 1 {
		return string(utf8.RuneError), 1
	}

	return s[:size], size
}

// symbol is bpe_model.cc's Symbol: norm[lo:hi] in a doubly linked list; a
// merged-away symbol is empty.
type symbol struct {
	prev, next int
	freeze     bool // a user-defined symbol, never merged
	lo, hi     int
}

// pair is bpe_model.cc's SymbolPair: merging symbols left and right gives a
// piece of size bytes and score key.
type pair struct {
	key         int64 // scoreKey of the merged piece's score
	left, right int
	size        int
}

// agenda is the max-heap of candidate merges, ordered as bpe_model.cc's
// SymbolPairComparator: the highest score first, on equal scores the
// leftmost pair.
type agenda []pair

func (a *agenda) Len() int { return len(*a) }

func (a *agenda) Less(i, j int) bool {
	x, y := (*a)[i], (*a)[j]

	return x.key > y.key || (x.key == y.key && x.left < y.left)
}

func (a *agenda) Swap(i, j int) { (*a)[i], (*a)[j] = (*a)[j], (*a)[i] }

func (a *agenda) Push(x any) { *a = append(*a, x.(pair)) } //nolint:forcetypeassert // only pairs are pushed

func (a *agenda) Pop() any {
	old := *a
	x := old[len(old)-1]
	*a = old[:len(old)-1]

	return x
}

// scoreKey maps a score to an integer in the order sentencepiece v0.2.2's
// SymbolPairComparator compares scores, which is by IEEE 754 bit pattern
// rather than by float value: it is the float order, except that -0 ranks
// below +0 and NaNs rank by their bits.
func scoreKey(f float32) int64 {
	bits := math.Float32bits(f)
	if bits&(1<<31) == 0 {
		return int64(bits)
	}

	return -int64(bits&^(1<<31)) - 1
}

// oneCharLen is string_util::OneCharLen: the length of the UTF-8 sequence
// that lead byte b starts. On normalised (valid UTF-8) text it is the rune
// length.
func oneCharLen(b byte) int {
	return int("\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x02\x02\x03\x04"[b>>4])
}

// bpe is bpe_model.cc Model::SampleEncode without dropout. norm is split
// into user-defined symbols (longest match, frozen) and single characters;
// then, while any adjacent pair of unfrozen symbols concatenates to a
// NORMAL, USER_DEFINED or UNUSED piece, the pair with the best score (the
// leftmost on ties) is merged. Merged UNUSED pieces are finally split back
// into the parts they were merged from. The symbols are returned as
// substrings of norm, in order.
func (p *Processor) bpe(norm string) []string {
	b := bpeState{p: p, norm: norm, syms: make([]symbol, 0, len(norm))}

	for i := 0; i < len(norm); {
		n := p.users.match(norm[i:])
		freeze := n > 0

		if !freeze {
			n = min(oneCharLen(norm[i]), len(norm)-i)
		}

		b.syms = append(b.syms, symbol{prev: len(b.syms) - 1, next: len(b.syms) + 1, freeze: freeze, lo: i, hi: i + n})
		i += n
	}

	b.syms[len(b.syms)-1].next = -1

	for r := 1; r < len(b.syms); r++ {
		if c, ok := b.candidate(r-1, r); ok {
			b.agenda = append(b.agenda, c)
		}
	}

	heap.Init(&b.agenda)

	for b.agenda.Len() > 0 {
		b.merge(heap.Pop(&b.agenda).(pair)) //nolint:forcetypeassert // only pairs are pushed
	}

	out := make([]string, 0, len(b.syms))
	for i := 0; i >= 0; i = b.syms[i].next {
		out = b.resegment(out, b.piece(i), 0)
	}

	return out
}

// bpeState is the working state of one bpe call.
type bpeState struct {
	p        *Processor
	norm     string
	syms     []symbol
	agenda   agenda
	revMerge map[string][2]string // merged UNUSED piece -> its two parts
}

// piece is the current text of symbol i.
func (b *bpeState) piece(i int) string { return b.norm[b.syms[i].lo:b.syms[i].hi] }

// candidate is MaybeAddNewSymbolPair: the agenda entry for merging symbols
// l and r, if they may merge. It records how an UNUSED piece was built.
func (b *bpeState) candidate(l, r int) (pair, bool) {
	if l < 0 || r < 0 || b.syms[l].freeze || b.syms[r].freeze {
		return pair{}, false
	}

	merged := b.norm[b.syms[l].lo:b.syms[r].hi]

	id, ok := b.p.pieceID[merged]
	if !ok {
		return pair{}, false
	}

	pc := b.p.model.Pieces[id]

	switch pc.Type {
	case TypeNormal, TypeUserDefined:
	case TypeUnused:
		if b.revMerge == nil {
			b.revMerge = map[string][2]string{}
		}

		b.revMerge[merged] = [2]string{b.piece(l), b.piece(r)}
	default:
		return pair{}, false // CONTROL, UNKNOWN and BYTE pieces never come from a merge
	}

	return pair{key: scoreKey(pc.Score), left: l, right: r, size: len(merged)}, true
}

// merge applies the agenda entry top unless it is stale, i.e. one side was
// merged away or has grown since it was queued, and queues the pairs the
// merged symbol forms with its neighbours.
func (b *bpeState) merge(top pair) {
	l, r := &b.syms[top.left], &b.syms[top.right]
	if l.lo == l.hi || r.lo == r.hi || (l.hi-l.lo)+(r.hi-r.lo) != top.size {
		return
	}

	l.hi = r.hi
	l.next = r.next

	if r.next >= 0 {
		b.syms[r.next].prev = top.left
	}

	r.lo = r.hi

	if c, ok := b.candidate(l.prev, top.left); ok {
		heap.Push(&b.agenda, c)
	}

	if c, ok := b.candidate(top.left, l.next); ok {
		heap.Push(&b.agenda, c)
	}
}

// resegment appends w to out, splitting a merged UNUSED piece back into its
// parts, recursively up to bpe_model.cc's depth limit.
func (b *bpeState) resegment(out []string, w string, depth int) []string {
	id, ok := b.p.pieceID[w]
	if !ok || depth > maxResegmentDepth || b.p.model.Pieces[id].Type != TypeUnused {
		return append(out, w)
	}

	parts, ok := b.revMerge[w]
	if !ok {
		return append(out, w)
	}

	out = b.resegment(out, parts[0], depth+1)

	return b.resegment(out, parts[1], depth+1)
}
