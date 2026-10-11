package search

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// Form is a set of spellings under which a keyword is matched.
type Form int

const (
	// FormSpace matches the keyword after a space ("▁" + its first
	// piece), a word inside the text.
	FormSpace Form = 1 << iota
	// FormBare matches the keyword without a leading space, a word at the
	// start of the text (the vocabulary has no dummy prefix).
	FormBare
	// FormCapital adds the spellings with the first letter upper-cased.
	FormCapital
	// FormLower adds the spellings with the first letter lower-cased.
	FormLower
)

// String names the forms, e.g. "space+bare+capital".
func (f Form) String() string {
	var names []string

	for i, name := range []string{"space", "bare", "capital", "lower"} {
		if f&(1<<i) != 0 {
			names = append(names, name)
		}
	}

	return strings.Join(names, "+")
}

// Bias is how keyword biasing rewards a keyword in a hypothesis' text
// (docs/FINDINGS.md F-029). The bonus enters the adjusted log-probability
// that ranks, prunes and finally picks the hypotheses.
type Bias struct {
	Weight   float64 // bonus per matched keyword token in nats (per character under PerChar)
	PerChar  bool    // scale a token's bonus by the characters it spells
	Revoke   float64 // the share of a failed partial match's bonus taken back, 0 … 1
	Complete float64 // extra bonus when a keyword completes
	Forms    Form    // the spellings matched
	Split    bool    // match each word of a phrase as a keyword of its own
	Renorm   bool    // add the automaton tokens' bonus to the logits and renormalise

	// Probe axes of docs/FINDINGS.md F-042, off in DefaultBias.
	transient bool // the completion bonus lasts one token: the next token (EOS included) takes it back
	echo      bool // with transient, a completion also earns the keyword's own token bonus for that token (icefall)
	once      bool // a keyword earns its bonus once per text: a repeat's tokens are taken back when it completes
}

// DefaultBias is the bias the keyword fixtures picked (docs/FINDINGS.md
// F-029): 2 nats per matched token, all of it taken back when a partial
// match fails, 1 more on completion, the keyword after a space and bare,
// as written and with its first letter upper- and lower-cased. With
// Default it reproduces the engine's text with the keywords on 45 of 53
// fixture cases.
func DefaultBias() Bias {
	return Bias{Weight: 2, Revoke: 1, Complete: 1, Forms: FormSpace | FormBare | FormCapital | FormLower}
}

// String names the bias, e.g. "w=1.5 revoke=1 complete=0
// forms=space+bare+capital".
func (b Bias) String() string {
	var sb strings.Builder

	fmt.Fprintf(&sb, "w=%g", b.Weight)

	if b.PerChar {
		sb.WriteString("/char")
	}

	fmt.Fprintf(&sb, " revoke=%g complete=%g forms=%s", b.Revoke, b.Complete, b.Forms)

	if b.Split {
		sb.WriteString(" split")
	}

	if b.Renorm {
		sb.WriteString(" renorm")
	}

	return sb.String()
}

// ErrKeywords reports an invalid Bias.
var ErrKeywords = errors.New("search: invalid keyword bias")

// ParseKeywords splits the engine's keyword format, newline-separated words
// and phrases, into keywords: each line trimmed, inner runs of white space
// made one space, empty lines dropped.
func ParseKeywords(s string) []string {
	var out []string

	for line := range strings.SplitSeq(s, "\n") {
		if kw := strings.Join(strings.Fields(line), " "); kw != "" {
			out = append(out, kw)
		}
	}

	return out
}

// Keywords is an Aho-Corasick automaton over the token spellings of a
// keyword list. Its Adjust method is a search Adjust: it reruns the
// automaton over a hypothesis' text and adds to every next token the
// change of the hypothesis' bonus. A token that advances a match earns the
// token's weight, and Bias.Complete on top for every keyword it completes,
// a keyword that ends as a suffix of the match included; a token that
// breaks a partial match falls back to the longest suffix that is still a
// keyword prefix and takes back Bias.Revoke of the bonus the lost tokens
// earned, except for the tokens of a completed keyword, whose bonus is
// never taken back. A bare spelling (FormBare) matches only at the start
// of the text. EOS ends the text, so it breaks any partial match.
type Keywords struct {
	next  []map[int]int // goto, per node; node 0 is the root
	fail  []int         // the longest proper suffix that is a node
	up    []int         // the parent node
	w     []float64     // the bonus of the token into the node
	depth []int         // the node's path length in tokens
	out   [][]int       // the token lengths of the keywords that end at the node, suffixes included
	end   []bool        // a keyword ends here
	key   []int         // the keyword a spelling that ends at the node spells, shared by its forms (Bias.once)
	start kwState       // the state at the start of the text
	bias  Bias
}

// kwState is the automaton's state after a text: the node of the longest
// text suffix that is a keyword prefix, and which of that suffix' tokens
// belong to a completed keyword (bit i for the path's token i).
type kwState struct {
	n    int
	cov  bitset
	held float64 // the transient bonus the next token takes back
	done []int   // the keywords completed so far, by key (Bias.once)
}

// textStart is the token in front of every text that anchors the bare
// spellings; it earns nothing. noToken is a token outside the automaton.
const (
	textStart = -2
	noToken   = -1
)

// bitset is a set of path positions, bit i of word i/64 for position i.
// It is never written in place: a state shares it with the states it was
// derived from.
type bitset []uint64

// has reports whether position i is in b.
func (b bitset) has(i int) bool {
	w := i / 64

	return w < len(b) && b[w]>>(i%64)&1 != 0
}

// shift is b without its first n positions, the rest moved down by n.
func (b bitset) shift(n int) bitset {
	if n == 0 {
		return b
	}

	skip, off := n/64, n%64
	if skip >= len(b) {
		return nil
	}

	out := make(bitset, len(b)-skip)
	for i := range out {
		out[i] = b[skip+i] >> off
		if off > 0 && skip+i+1 < len(b) {
			out[i] |= b[skip+i+1] << (64 - off)
		}
	}

	return out
}

// with is b plus the positions [lo, hi).
func (b bitset) with(lo, hi int) bitset {
	out := make(bitset, max(len(b), (hi+63)/64))
	copy(out, b)

	for i := lo; i < hi; i++ {
		out[i/64] |= 1 << (i % 64)
	}

	return out
}

// NewKeywords tokenises the keywords in the forms of b with tok and builds
// the automaton.
func NewKeywords(tok *tokenizer.Tokenizer, keywords []string, b Bias) (*Keywords, error) {
	switch {
	case math.IsNaN(b.Weight) || math.IsInf(b.Weight, 0) || math.IsNaN(b.Complete) || math.IsInf(b.Complete, 0):
		return nil, fmt.Errorf("%w: weight %v, completion %v", ErrKeywords, b.Weight, b.Complete)
	case !(b.Revoke >= 0 && b.Revoke <= 1):
		return nil, fmt.Errorf("%w: revoke %v", ErrKeywords, b.Revoke)
	case b.Forms&(FormSpace|FormBare) == 0:
		return nil, fmt.Errorf("%w: forms %b match neither after a space nor bare", ErrKeywords, b.Forms)
	}

	ids, keys := encodeKeywords(tok, keywords, b)

	var k *Keywords

	if b.PerChar {
		ws := make([][]float64, len(ids))
		for i, s := range ids {
			ws[i] = charWeights(tok, s, b.Weight)
		}

		k = buildWeighted(ids, ws, b)
	} else {
		k = build(ids, func(int) float64 { return b.Weight }, b)
	}

	k.identify(ids, keys)

	return k, nil
}

// encodeKeywords tokenises the spellings of the keywords under b, a bare
// one behind textStart, and gives each spelling the index of the keyword
// (under Split: of the word) it spells, shared by all its forms.
func encodeKeywords(tok *tokenizer.Tokenizer, keywords []string, b Bias) ([][]int, []int) {
	var (
		ids  [][]int
		keys []int
		key  int
	)

	for _, kw := range keywords {
		words := []string{kw}
		if b.Split {
			words = strings.Fields(kw)
		}

		for _, w := range words {
			for _, s := range spellings(w, b.Forms) {
				e := tok.Encode(s)
				if len(e) > 0 && !strings.HasPrefix(s, " ") {
					e = append([]int{textStart}, e...)
				}

				ids = append(ids, e)
				keys = append(keys, key)
			}

			key++
		}
	}

	return ids, keys
}

// charWeights is the PerChar bonus of each token of a spelling: weight
// times the change of the characters its tokens so far decode to, spaces
// not counted. A byte run decodes as Tokenizer.Decode does it, one U+FFFD
// per byte that starts no valid sequence, so whether a byte is a character
// of its own can depend on the bytes after it; a token's share may then be
// negative, but a whole spelling earns its decoded characters. A byte
// decision is final once 4 bytes follow, so each token costs O(1).
func charWeights(tok *tokenizer.Tokenizer, ids []int, weight float64) []float64 {
	ws := make([]float64, len(ids))

	var (
		run          []byte
		done, commit int // characters before the run's undecided tail; the tail's start
		last         int // the characters after the previous token
	)

	for i, id := range ids {
		switch {
		case id == textStart:
		case tok.Type(id) == tokenizer.TypeByte:
			v, _ := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(tok.Piece(id), "<0x"), ">"), 16, 8)
			run = append(run, byte(v))

			for len(run)-commit >= utf8.UTFMax {
				_, size := utf8.DecodeRune(run[commit:])
				commit += size
				done++
			}
		default:
			done += utf8.RuneCount(run[commit:])
			run, commit = run[:0], 0

			if t := tok.Type(id); t == tokenizer.TypeNormal || t == tokenizer.TypeUserDefined {
				done += utf8.RuneCountInString(strings.ReplaceAll(tok.Piece(id), "▁", ""))
			}
		}

		now := done + utf8.RuneCount(run[commit:])
		ws[i] = weight * float64(now-last)
		last = now
	}

	return ws
}

// build is the automaton over the token sequences ids, a token's bonus
// given by weight; a sequence that starts with textStart is anchored at
// the start of the text.
func build(ids [][]int, weight func(id int) float64, b Bias) *Keywords {
	ws := make([][]float64, len(ids))

	for i, s := range ids {
		ws[i] = make([]float64, len(s))
		for j, id := range s {
			if id != textStart {
				ws[i][j] = weight(id)
			}
		}
	}

	return buildWeighted(ids, ws, b)
}

// buildWeighted is build with the bonus of token j of spelling i in
// ws[i][j]. A node's bonus comes from the first spelling through it; the
// spellings through a node share its prefix, so their bonuses agree.
func buildWeighted(ids [][]int, ws [][]float64, b Bias) *Keywords {
	k := &Keywords{next: []map[int]int{{}}, fail: []int{0}, up: []int{0}, w: []float64{0}, depth: []int{0}, end: []bool{false}, key: []int{0}, bias: b}

	for i, s := range ids {
		k.add(s, ws[i])
	}

	k.link()
	k.start, _ = k.step(kwState{}, textStart)

	return k
}

// spellings is the strings a keyword is tokenised from under forms.
func spellings(kw string, forms Form) []string {
	cases := []string{kw}

	r, size := utf8.DecodeRuneInString(kw)
	if forms&FormCapital != 0 {
		cases = append(cases, string(unicode.ToUpper(r))+kw[size:])
	}

	if forms&FormLower != 0 {
		cases = append(cases, string(unicode.ToLower(r))+kw[size:])
	}

	var out []string

	for _, c := range cases {
		if forms&FormSpace != 0 {
			out = append(out, " "+c)
		}

		if forms&FormBare != 0 {
			out = append(out, c)
		}
	}

	return out
}

func hasNext(m map[int]int, id int) bool {
	_, ok := m[id]

	return ok
}

// Nodes is the number of automaton nodes, the root included.
func (k *Keywords) Nodes() int { return len(k.next) }

// Adjust adds to logp, the next-token log-probabilities after text, each
// token's change of the keyword bonus. It is an Adjust for Options.
func (k *Keywords) Adjust(text []int, logp []float64) {
	all, special := k.changes(text, func(id int) float64 {
		if id < len(logp) {
			return logp[id]
		}

		return math.Inf(-1)
	})

	if all != 0 {
		for i := range logp {
			logp[i] += all
		}
	}

	for id, d := range special {
		if id < len(logp) {
			logp[id] += d - all
		}
	}
}

// Bonus is the keyword bonus of a whole text: the sum of the changes
// Adjust adds along it, with EOS at the end when eos ≥ 0, without the
// normaliser of Bias.Renorm.
func (k *Keywords) Bonus(text []int, eos int) float64 {
	st, sum := k.start, 0.0

	for _, id := range text {
		var d float64

		st, d = k.step(st, id)
		sum += d
	}

	if eos >= 0 {
		_, d := k.step(st, eos)
		sum += d
	}

	return sum
}

// add inserts one spelling.
func (k *Keywords) add(ids []int, ws []float64) {
	if len(ids) == 0 || (len(ids) == 1 && ids[0] == textStart) {
		return
	}

	n := 0

	for i, id := range ids {
		c, ok := k.next[n][id]
		if !ok {
			c = len(k.next)
			k.next = append(k.next, map[int]int{})
			k.fail = append(k.fail, 0)
			k.up = append(k.up, n)
			k.w = append(k.w, ws[i])
			k.depth = append(k.depth, k.depth[n]+1)
			k.end = append(k.end, false)
			k.key = append(k.key, -1-c) // a spelling of its own until identify
			k.next[n][id] = c
		}

		n = c
	}

	k.end[n] = true
}

// identify gives the node each spelling of ids ends at the key of its
// keyword, keys[i] for spelling i, so that Bias.once counts a keyword once
// whatever its form. A node two keywords' spellings share keeps the
// first's key. Without identify each spelling is a keyword of its own.
func (k *Keywords) identify(ids [][]int, keys []int) {
	set := make([]bool, len(k.key))

	for i, s := range ids {
		n := 0
		for _, id := range s {
			n = k.next[n][id]
		}

		if n != 0 && !set[n] {
			k.key[n], set[n] = keys[i], true
		}
	}
}

// link sets the failure links breadth first, and with them the keywords
// that end at each node: its own and those of its failure link.
func (k *Keywords) link() {
	k.out = make([][]int, len(k.next))

	queue := make([]int, 0, len(k.next))
	for _, c := range k.next[0] {
		queue = append(queue, c)
	}

	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]

		if k.end[n] {
			k.out[n] = append(k.out[n], k.depth[n])
		}

		k.out[n] = append(k.out[n], k.out[k.fail[n]]...)

		for id, c := range k.next[n] {
			f := k.fail[n]
			for f != 0 && !hasNext(k.next[f], id) {
				f = k.fail[f]
			}

			if g, ok := k.next[f][id]; ok && g != c {
				k.fail[c] = g
			}

			queue = append(queue, c)
		}
	}
}

// state is the state after text.
func (k *Keywords) state(text []int) kwState {
	st := k.start
	for _, id := range text {
		st, _ = k.step(st, id)
	}

	return st
}

// step is the state after token id at st and the bonus change: the
// revoked bonus of the tokens the match drops, then the weight of id and
// the completions when id extends a match.
func (k *Keywords) step(st kwState, id int) (kwState, float64) {
	n, f := st.n, st.n

	c, ok := k.next[n][id]
	for !ok && f != 0 {
		f = k.fail[f]
		c, ok = k.next[f][id]
	}

	drop := k.depth[n] - k.depth[f]
	d := k.revoke(n, drop, st.cov) - st.held

	if !ok {
		return kwState{done: st.done}, d
	}

	depth := k.depth[c]
	cov := st.cov.shift(drop)
	d += k.w[c]
	next := kwState{n: c, done: st.done}

	for _, l := range k.out[c] {
		kn := k.suffix(c, l)

		if k.bias.once {
			if slices.Contains(next.done, k.key[kn]) {
				d -= k.pathBonus(kn) // a repeat, in any form: its tokens earn nothing
				cov = cov.with(depth-l, depth)

				continue
			}

			next.done = append(slices.Clip(next.done), k.key[kn])
		}

		bonus := k.bias.Complete
		if k.bias.transient && k.bias.echo {
			bonus += k.pathBonus(kn)
		}

		if k.bias.transient {
			next.held += bonus
		}

		d += bonus
		cov = cov.with(depth-l, depth)
	}

	next.cov = cov

	return next, d
}

// suffix is the node of the last l tokens of the match at c: c or one of
// its failure links.
func (k *Keywords) suffix(c, l int) int {
	for k.depth[c] > l {
		c = k.fail[c]
	}

	return c
}

// pathBonus is the token bonus along the path to node n.
func (k *Keywords) pathBonus(n int) float64 {
	sum := 0.0
	for ; k.depth[n] > 0; n = k.up[n] {
		sum += k.w[n]
	}

	return sum
}

// revoke is the bonus taken back when the match at node n drops its
// first drop tokens: theirs, except for those cov marks as in a completed
// keyword. It is signed, so a negative PerChar share is taken back too
// and the dropped tokens' net bonus goes.
func (k *Keywords) revoke(n, drop int, cov bitset) float64 {
	if drop == 0 {
		return 0 // a direct step: nothing to walk
	}

	lost := 0.0

	for a := n; k.depth[a] > 0; a = k.up[a] {
		if i := k.depth[a] - 1; i < drop && !cov.has(i) {
			lost += k.w[a]
		}
	}

	return -k.bias.Revoke * lost
}

// changes is what Adjust adds after text, given the model's log p: all to
// a token outside the automaton, special[id] to each automaton token.
// Without Bias.Renorm these are the bonus changes. With it, the automaton
// tokens' bonus over that of the others is a logit bias: added to their
// logits before the softmax, so the normaliser takes it back where the
// model is already sure of them, while the change every token shares (a
// revocation) stays a score term.
func (k *Keywords) changes(text []int, lp func(id int) float64) (float64, map[int]float64) {
	base, special := k.deltas(text)
	if !k.bias.Renorm {
		return base, special
	}

	z := 1.0
	for id, d := range special {
		z += math.Exp(lp(id)) * math.Expm1(d-base)
	}

	logZ := math.Log(max(z, math.SmallestNonzeroFloat64))

	for id, d := range special {
		special[id] = d - logZ
	}

	return base - logZ, special
}

// deltas is the bonus change of every next token after text: base for a
// token outside the automaton, which falls back to the root, and the
// change for each token with a transition from the node after text or
// from one of its suffixes.
func (k *Keywords) deltas(text []int) (float64, map[int]float64) {
	st := k.state(text)
	_, base := k.step(st, noToken)
	special := map[int]float64{}

	for f := st.n; ; f = k.fail[f] {
		for id := range k.next[f] {
			if _, ok := special[id]; !ok && id >= 0 {
				_, special[id] = k.step(st, id)
			}
		}

		if f == 0 {
			return base, special
		}
	}
}
