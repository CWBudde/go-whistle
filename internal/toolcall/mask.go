package toolcall

import (
	"fmt"
	"slices"
	"strings"

	"github.com/cwbudde/go-whistle/internal/spm"
)

// tokenMask is the vocabulary as bytes, for masking tokens under a
// grammar (docs/FINDINGS.md F-055): a token is allowed iff its bytes keep
// the automaton alive. A normal piece stands for its text ("▁" a space), a
// byte-fallback piece <0xNN> for its one byte; control, user-defined and
// unknown pieces never stand inside the call block.
type tokenMask struct {
	bytes [][]byte // each token's bytes; nil for a token that has none
	root  *trieNode
}

// trieNode is the tokens that share a byte prefix: ids end here, edges go
// on.
type trieNode struct {
	ids   []int
	edges []trieEdge
}

type trieEdge struct {
	b    byte
	next *trieNode
}

// newTokenMask indexes the first vocab tokens of p.
func newTokenMask(p *spm.Processor, vocab int) *tokenMask {
	t := &tokenMask{bytes: make([][]byte, vocab), root: &trieNode{}}

	for id := range min(vocab, p.Len()) {
		var bs []byte

		switch p.Type(id) {
		case spm.TypeNormal:
			bs = []byte(strings.ReplaceAll(p.Piece(id), "▁", " "))
		case spm.TypeByte:
			// The piece's surface is <0xNN>.
			var b byte

			_, err := fmt.Sscanf(p.Piece(id), "<0x%02X>", &b)
			if err != nil {
				continue
			}

			bs = []byte{b}
		default:
			continue
		}

		t.bytes[id] = bs
		t.insert(bs, id)
	}

	return t
}

func (t *tokenMask) insert(bs []byte, id int) {
	n := t.root

	for _, b := range bs {
		i, found := slices.BinarySearchFunc(n.edges, b, func(e trieEdge, b byte) int { return int(e.b) - int(b) })
		if !found {
			n.edges = slices.Insert(n.edges, i, trieEdge{b: b, next: &trieNode{}})
		}

		n = n.edges[i].next
	}

	n.ids = append(n.ids, id)
}

// allowed are the tokens whose bytes keep some state of states alive, in
// no particular order; g caches them by the states.
func (t *tokenMask) allowed(g *Grammar, states []*stack) []int {
	key := stateKey(states)

	g.mu.Lock()
	defer g.mu.Unlock()

	if ids, ok := g.cache[key]; ok {
		return ids
	}

	var ids []int

	t.walk(t.root, states, &ids)

	if g.cache == nil {
		g.cache = map[string][]int{}
	}

	g.cache[key] = ids

	return ids
}

// walk collects the tokens below n whose bytes past n keep states alive.
func (t *tokenMask) walk(n *trieNode, states []*stack, ids *[]int) {
	for _, e := range n.edges {
		next := advanceAll(states, e.b)
		if len(next) == 0 {
			continue
		}

		*ids = append(*ids, e.next.ids...)
		t.walk(e.next, next, ids)
	}
}

// stateKey identifies a set of states: matchers print their nodes and
// grammar by address, which the grammar keeps alive.
func stateKey(states []*stack) string {
	var b strings.Builder

	for _, st := range states {
		for ; st != nil; st = st.below {
			fmt.Fprintf(&b, "%T%v;", st.top, st.top)
		}

		b.WriteByte('|')
	}

	return b.String()
}
