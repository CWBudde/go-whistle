package needle

import (
	"fmt"
	"slices"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// Engram hash constants (architecture.py _ENGRAM_SEED, _ENGRAM_PRIME).
const (
	ngramSeedBase = 0x9E3779B9
	ngramPrime    = 0x01000193
)

// EngramKV is one engram site's keys and values for a sequence, each t
// rows of the model width; Block gates them into its input.
type EngramKV struct {
	K, V []float32
}

// Engram is architecture.py Engram with engram_indices: n-gram hashes of
// the token history select one row per table, the rows are concatenated,
// and key_proj/value_proj map them to the model width; the values then
// pass a causal 4-tap conv dilated by the largest order:
//
//	idx[t, c] = hash_c(tok[t − order_c + 1 … t]) mod slots
//	e[t]      = concat_c tables[c, idx[t, c]] · (t ≥ order_c − 1)
//	k         = e @ key_proj
//	v         = Σ_j taps[j] · shift_right(e @ value_proj, dilation · j)
//
// Table c = oi·hashes + h serves order orders[oi] with hash head h
// (Whistle: orders (2, 3), 2 hashes, 4 tables of 18432 × 128). The hash is
// uint32 arithmetic and reads only the last order tokens, so a decoder can
// keep a rolling history of MaxOrder tokens (IndicesAt; docs/FINDINGS.md
// F-013).
type Engram struct {
	dim, sub, slots, taps, dilation, maxOrder int

	orders           []int     // order of table c
	tables           Linear    // (len(orders)·slots, sub): table c's slot s is row c·slots + s
	keyProj, valProj Linear    // [dim, len(orders)·sub]
	tapW             []float32 // (taps, dim)
}

// LoadEngram loads the site with the names prefix+"tables", … ("engram0."
// and "engram1." for Whistle's sites at layers 3 and 7). orders are the
// n-gram orders and tables the table count, len(orders) × hashes; the
// weights' Config holds both (Orders, Tables). The slots, the sub-width,
// the model width and the tap count come from the shapes.
func LoadEngram(src weights.Source, prefix string, orders []int, tables int) (*Engram, error) {
	if len(orders) == 0 || tables <= 0 || tables%len(orders) != 0 || slices.Min(orders) < 1 {
		return nil, fmt.Errorf("needle: %sengram: %d tables for orders %v", prefix, tables, orders)
	}

	l := &loader{src: src, prefix: prefix}
	e := &Engram{dilation: slices.Max(orders), maxOrder: slices.Max(orders)}

	e.tables = l.table("tables", 0, 0)
	rows := e.tables.Out()
	e.sub = e.tables.In()

	if l.err == nil && (rows == 0 || rows%tables != 0) {
		l.failf("%stables has %d rows, not a multiple of %d tables", prefix, rows, tables)
	}

	e.slots = rows / tables
	e.keyProj = l.linear("key_proj", 0, tables*e.sub)
	e.dim = e.keyProj.Out()
	e.valProj = l.linear("value_proj", e.dim, tables*e.sub)
	e.tapW, e.taps, _ = l.matrix("taps", 0, e.dim)

	if l.err != nil {
		return nil, l.err
	}

	hashes := tables / len(orders)
	for _, o := range orders {
		for range hashes {
			e.orders = append(e.orders, o)
		}
	}

	return e, nil
}

// Dim is the model width of the keys and values.
func (e *Engram) Dim() int { return e.dim }

// Tables is the number of hash tables, one index per position each.
func (e *Engram) Tables() int { return len(e.orders) }

// Slots is the number of rows per table.
func (e *Engram) Slots() int { return e.slots }

// MaxOrder is the largest n-gram order: the indices of a position depend
// on its last MaxOrder tokens only.
func (e *Engram) MaxOrder() int { return e.maxOrder }

// IndicesAt writes the Tables() slot indices of the position whose token
// history is hist, hist[len(hist) − 1] being the token at the position;
// tokens before the start of hist count as 0, which is upstream's zero
// padding when hist starts at position 0. A decoder passes its rolling
// window of the last MaxOrder tokens.
func (e *Engram) IndicesAt(dst, hist []int) {
	for c, order := range e.orders {
		dst[c] = int(ngramHash(c, order, hist) % uint32(e.slots))
	}
}

// Indices is architecture.py engram_indices for one sequence: t rows of
// Tables() slot indices.
func (e *Engram) Indices(tokens []int) []int {
	n := len(e.orders)
	out := make([]int, len(tokens)*n)

	for i := range tokens {
		e.IndicesAt(out[i*n:(i+1)*n], tokens[max(0, i+1-e.maxOrder):i+1])
	}

	return out
}

// Forward returns the keys and values of the token ids of one sequence,
// each len(tokens) rows of Dim values, with the causal mask.
func (e *Engram) Forward(tokens []int) (*EngramKV, error) {
	return e.forward(tokens, nil)
}

// ngramSeed is table c's hash seed, _ENGRAM_SEED · (c + 1) mod 2³². The
// seed stride between orders is the hash count, upstream's default
// (Whistle's engram_seed_heads is 0, F-003).
func ngramSeed(c int) uint32 { return ngramSeedBase * uint32(c+1) }

// ngramHash is engram_indices' hash of table c before the mod: acc = seed,
// then for j < order acc = (acc ^ tok[t − j]) · _ENGRAM_PRIME, then
// acc ^= acc >> 15, all mod 2³². hist ends with tok[t]; tokens before its
// start count as 0.
func ngramHash(c, order int, hist []int) uint32 {
	acc := ngramSeed(c)

	for j := range order {
		var tok uint32
		if j < len(hist) {
			tok = uint32(hist[len(hist)-1-j])
		}

		acc = (acc ^ tok) * ngramPrime
	}

	return acc ^ acc>>15
}

// engramTrace receives the intermediates for the parity tests.
type engramTrace struct {
	indices []int
	ngramOK []float32 // (t, tables)
	e, vRaw []float32
}

func (e *Engram) forward(tokens []int, tr *engramTrace) (*EngramKV, error) {
	for i, tok := range tokens {
		if tok < 0 {
			return nil, fmt.Errorf("needle: engram token %d at %d is negative", tok, i)
		}
	}

	t, n := len(tokens), len(e.orders)
	w := n * e.sub
	idx := e.Indices(tokens)
	emb := make([]float32, t*w)
	ok := make([]float32, t*n)

	// fetched · ngram_ok: positions before order − 1 keep zero rows.
	for i := range t {
		for c, order := range e.orders {
			if i+1 < order {
				continue
			}

			row := c*e.slots + idx[i*n+c]
			e.tables.Row(emb[i*w+c*e.sub:i*w+(c+1)*e.sub], row)
			ok[i*n+c] = 1
		}
	}

	k, vRaw := make([]float32, t*e.dim), make([]float32, t*e.dim)
	e.keyProj.Apply(k, emb, t)
	e.valProj.Apply(vRaw, emb, t)

	if tr != nil {
		tr.indices, tr.ngramOK, tr.e, tr.vRaw = idx, ok, emb, slices.Clone(vRaw)
	}

	return &EngramKV{K: k, V: applyTaps(vRaw, t, e.dim, e.tapW, e.taps, e.dilation)}, nil
}
