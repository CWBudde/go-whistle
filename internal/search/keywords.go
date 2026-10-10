package search

import (
	"errors"
	"fmt"
	"math"
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
// token's weight, and Bias.Complete on top when it completes a keyword; a
// token that breaks a partial match falls back to the longest suffix that
// is still a keyword prefix and takes back Bias.Revoke of the bonus the
// lost part earned. A completed keyword's bonus is never taken back. EOS
// ends the text, so it breaks any partial match.
type Keywords struct {
	next []map[int]int // goto, per node; node 0 is the root
	fail []int         // the longest proper suffix that is a node
	phi  []float64     // the bonus of the partial match since the last completed keyword on the path
	end  []bool        // a keyword ends here
	bias Bias
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

	var ids [][]int

	for _, kw := range keywords {
		words := []string{kw}
		if b.Split {
			words = strings.Fields(kw)
		}

		for _, w := range words {
			for _, s := range spellings(w, b.Forms) {
				ids = append(ids, tok.Encode(s))
			}
		}
	}

	weight := func(int) float64 { return b.Weight }
	if b.PerChar {
		weight = func(id int) float64 {
			return b.Weight * float64(utf8.RuneCountInString(strings.TrimPrefix(tok.Piece(id), "▁")))
		}
	}

	return build(ids, weight, b), nil
}

// build is the automaton over the token sequences ids, a token's bonus
// given by weight.
func build(ids [][]int, weight func(id int) float64, b Bias) *Keywords {
	k := &Keywords{next: []map[int]int{{}}, fail: []int{0}, phi: []float64{0}, end: []bool{false}, bias: b}

	for _, s := range ids {
		k.add(s, weight)
	}

	k.link()

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
	n, sum := 0, 0.0

	for _, id := range text {
		var d float64

		n, d = k.step(n, id)
		sum += d
	}

	if eos >= 0 {
		_, d := k.step(n, eos)
		sum += d
	}

	return sum
}

// add inserts one spelling.
func (k *Keywords) add(ids []int, weight func(int) float64) {
	if len(ids) == 0 {
		return
	}

	n := 0

	for _, id := range ids {
		c, ok := k.next[n][id]
		if !ok {
			base := k.phi[n]
			if k.end[n] {
				base = 0
			}

			c = len(k.next)
			k.next = append(k.next, map[int]int{})
			k.fail = append(k.fail, 0)
			k.phi = append(k.phi, base+weight(id))
			k.end = append(k.end, false)
			k.next[n][id] = c
		}

		n = c
	}

	k.end[n] = true
}

// link sets the failure links breadth first.
func (k *Keywords) link() {
	queue := make([]int, 0, len(k.next))
	for _, c := range k.next[0] {
		queue = append(queue, c)
	}

	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]

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

// state is the node after text.
func (k *Keywords) state(text []int) int {
	n := 0
	for _, id := range text {
		n, _ = k.step(n, id)
	}

	return n
}

// step is the node after token id at node n and the bonus change.
func (k *Keywords) step(n, id int) (int, float64) {
	if c, ok := k.next[n][id]; ok {
		return c, k.gain(n, c)
	}

	f := n
	for f != 0 {
		f = k.fail[f]
		if c, ok := k.next[f][id]; ok {
			return c, k.loss(n, f) + k.gain(f, c)
		}
	}

	return 0, k.loss(n, 0)
}

// gain is the bonus of advancing from n to its child c.
func (k *Keywords) gain(n, c int) float64 {
	g := k.phi[c]
	if !k.end[n] {
		g -= k.phi[n]
	}

	if k.end[c] {
		g += k.bias.Complete
	}

	return g
}

// loss is the bonus taken back when the match at n falls back to its
// suffix f.
func (k *Keywords) loss(n, f int) float64 {
	if k.end[n] {
		return 0
	}

	lost := k.phi[n]
	if !k.end[f] {
		lost -= k.phi[f]
	}

	return -k.bias.Revoke * max(lost, 0)
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
	n := k.state(text)
	_, base := k.step(n, -1)
	special := map[int]float64{}

	for f := n; ; f = k.fail[f] {
		for id := range k.next[f] {
			if _, ok := special[id]; !ok {
				_, special[id] = k.step(n, id)
			}
		}

		if f == 0 {
			return base, special
		}
	}
}
