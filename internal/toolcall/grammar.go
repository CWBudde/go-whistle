package toolcall

import (
	"math/big"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// The call block's grammar (docs/FINDINGS.md F-055). The engine decodes the
// <tool_call> list under "a byte-level grammar compiled from your schemas"
// (llms.txt): a pushdown automaton over the bytes of the JSON the model
// writes, compiled here from the schemas as rendered (Normalise), whose
// live states decide which tokens the decoder may take (tokenMask).

// Grammar is the call block of one turn, the JSON list between <tool_call>
// and </tool_call>: compact JSON (no white space outside strings), each
// call {"name":…,"arguments":…} with the two keys in that order, a name of
// the tools and arguments that match its parameters.
type Grammar struct {
	tools []string         // the model's names, in declaration order
	args  map[string]*node // the model's name -> its arguments
	// required asks for at least one call: a trigger routed the request
	// (docs/FINDINGS.md F-054, F-055).
	required bool

	mu    sync.Mutex
	cache map[string][]int // the allowed tokens by the automaton's states (tokenMask)
}

// Caps of the grammar from upstream's tool-design guide: "a numeric
// literal stops at twenty characters", "an array stops after sixty-four
// items".
const (
	maxNumberLen = 20
	maxItems     = 64
)

// Grammar is the call block's grammar over all of n's tools; an empty list
// (a refusal) is allowed.
func (n *Normalised) Grammar() *Grammar {
	return n.grammar(n.tools, false)
}

// GrammarFor is the grammar of the answer to input: Grammar, unless a
// trigger matches input and the request is not negated. Then the engine
// routes the request (docs/FINDINGS.md F-054, F-055): the grammar holds
// only the triggered tools and requires a call, and the answer must open
// one right after the reasoning.
func (n *Normalised) GrammarFor(input string) *Grammar {
	tools := n.Triggered(input)
	if len(tools) == 0 || negation.MatchString(input) {
		return n.Grammar()
	}

	return n.grammar(tools, true)
}

func (n *Normalised) grammar(tools []string, required bool) *Grammar {
	g := &Grammar{args: map[string]*node{}, required: required}

	for _, t := range tools {
		if _, dup := g.args[t]; dup {
			continue
		}

		g.tools = append(g.tools, t)

		params := n.params[t]

		switch {
		case n.never[t]:
			g.args[t] = &node{kind: kindNever}

			continue
		case params == nil:
			// A tool without parameters takes no arguments (port-side).
			g.args[t] = &node{kind: kindObject}

			continue
		}

		c := &compiler{root: params, refs: map[string]*node{}}
		g.args[t] = c.compile(params, 0)
	}

	// A tool whose arguments no value matches cannot be called (PR #65
	// review); a routed grammar without a callable tool falls back to the
	// refusal (port-side).
	g.tools = slices.DeleteFunc(g.tools, func(t string) bool { return !g.args[t].admits() })
	g.required = g.required && len(g.tools) > 0

	return g
}

// Accepts reports whether block is a whole call block of g.
func (g *Grammar) Accepts(block []byte) bool {
	states := g.feed(g.start(), block)

	return slices.ContainsFunc(states, (*stack).accepting)
}

// Alive reports whether prefix starts a call block of g.
func (g *Grammar) Alive(prefix []byte) bool {
	return len(g.feed(g.start(), prefix)) > 0
}

func (g *Grammar) start() []*stack {
	return []*stack{{top: callsM{g: g}}}
}

// feed advances the states over bs.
func (g *Grammar) feed(states []*stack, bs []byte) []*stack {
	for _, b := range bs {
		states = advanceAll(states, b)
		if len(states) == 0 {
			return nil
		}
	}

	return states
}

// nodeKind is what a schema node admits.
type nodeKind uint8

const (
	kindAny nodeKind = iota
	kindNever
	kindString
	kindInteger
	kindNumber
	kindBool
	kindNull
	kindObject
	kindArray
	kindLiterals // enum, const
	kindAlt      // anyOf, oneOf, a type list
)

// node is a compiled schema.
type node struct {
	kind nodeKind
	alts []*node  // kindAlt
	lits []string // kindLiterals: each value's compact JSON

	// strings
	date           bool // format: date, YYYY-MM-DD
	minLen, maxLen int  // in characters; maxLen < 0 is unbounded

	// numbers
	lo, hi         *big.Rat // nil: unbounded
	loExcl, hiExcl bool

	// objects
	props    map[string]*node
	order    []string // property names, declared order
	required []string
	extra    *node // the value of a key outside props; nil: no such key

	// arrays
	items              *node // nil: any value
	minItems, maxItems int
}

// anyNode admits every JSON value.
var anyNode = &node{kind: kindAny}

// compiler compiles the schemas of one tool's parameters; $ref resolves
// against root.
type compiler struct {
	root *object
	refs map[string]*node // a bare {"$ref": …} by its target, for recursion
}

// maxDepth bounds $ref chains and merges that never reach a schema.
const maxDepth = 64

// annotations are keys that do not constrain a value.
var annotations = map[string]bool{
	"description": true, "title": true, "default": true, "examples": true, "$comment": true,
	"readOnly": true, "writeOnly": true, "deprecated": true,
}

// compile is the node of schema s (a JSON schema object, or true/false).
func (c *compiler) compile(s any, depth int) *node {
	o, ok := s.(*object)
	if !ok {
		if s == false {
			return &node{kind: kindNever}
		}

		return anyNode
	}

	if depth > maxDepth {
		return anyNode
	}

	if ref, ok := o.vals["$ref"].(string); ok {
		return c.ref(o, ref, depth)
	}

	if all, ok := o.vals[allOf].([]any); ok {
		merged := o.without(allOf)

		for _, b := range all {
			r := c.resolve(b, depth)
			if r == false {
				return &node{kind: kindNever}
			}

			merged = merge(merged, r)
		}

		return c.compile(merged, depth+1)
	}

	for _, key := range []string{"anyOf", "oneOf"} {
		if list, ok := o.vals[key].([]any); ok {
			base := o.without(key)
			alt := &node{kind: kindAlt}

			for _, b := range list {
				r := c.resolve(b, depth)
				if r == false {
					continue // a branch no value matches (PR #65 review)
				}

				alt.alts = append(alt.alts, c.compile(merge(base, r), depth+1))
			}

			return alt
		}
	}

	return c.plain(o, depth)
}

// ref compiles a {"$ref": …} schema: the target merged with the other
// keys. A bare reference compiles once per target, so recursive schemas
// close on themselves.
func (c *compiler) ref(o *object, ref string, depth int) *node {
	target, ok := c.pointer(ref)
	if !ok {
		return anyNode
	}

	bare := true

	for _, k := range o.keys {
		if k != "$ref" && !annotations[k] {
			bare = false
		}
	}

	if !bare {
		return c.compile(merge(o.without("$ref"), target), depth+1)
	}

	if n, ok := c.refs[ref]; ok {
		return n
	}

	n := &node{}
	c.refs[ref] = n
	*n = *c.compile(target, depth+1)

	return n
}

// resolve is a branch schema with its $ref resolved, for merging.
func (c *compiler) resolve(s any, depth int) any {
	o, ok := s.(*object)
	if !ok {
		return s
	}

	for range maxDepth - depth {
		ref, ok := o.vals["$ref"].(string)
		if !ok {
			return o
		}

		target, ok := c.pointer(ref)
		if !ok {
			return o.without("$ref")
		}

		t, ok := target.(*object)
		if !ok {
			return target
		}

		o = merge(o.without("$ref"), t)
	}

	return o
}

// pointer resolves a local JSON pointer ("#", "#/$defs/x") against the
// root.
func (c *compiler) pointer(ref string) (any, bool) {
	path, ok := strings.CutPrefix(ref, "#")
	if !ok {
		return nil, false
	}

	var v any = c.root

	if path == "" {
		return v, true
	}

	if !strings.HasPrefix(path, "/") {
		return nil, false
	}

	for seg := range strings.SplitSeq(path[1:], "/") {
		seg = strings.ReplaceAll(strings.ReplaceAll(seg, "~1", "/"), "~0", "~")

		switch t := v.(type) {
		case *object:
			v, ok = t.vals[seg]
			if !ok {
				return nil, false
			}
		case []any:
			i, err := strconv.Atoi(seg)
			if err != nil || i < 0 || i >= len(t) {
				return nil, false
			}

			v = t[i]
		default:
			return nil, false
		}
	}

	return v, true
}

// merge is the schema a and b both describe, approximated (a port-side
// rule, docs/FINDINGS.md F-055): b's keys over a's, but "properties"
// united (a's first) and "required" united.
func merge(a *object, b any) *object {
	bo, ok := b.(*object)
	if !ok {
		return a
	}

	out := a

	for _, k := range bo.keys {
		v := bo.vals[k]

		switch k {
		case "properties":
			if ap, ok := a.vals[k].(*object); ok {
				if bp, ok := v.(*object); ok {
					u := ap
					for _, pk := range bp.keys {
						if _, has := u.vals[pk]; !has {
							u = u.with(pk, bp.vals[pk])
						}
					}

					v = u
				}
			}
		case "required":
			if ar, ok := a.vals[k].([]any); ok {
				if br, ok := v.([]any); ok {
					v = append(slices.Clone(ar), br...)
				}
			}
		}

		out = out.with(k, v)
	}

	return out
}

// plain compiles a schema without $ref and composition keywords.
func (c *compiler) plain(o *object, depth int) *node {
	if v, ok := o.vals["const"]; ok {
		return &node{kind: kindLiterals, lits: []string{compact(v)}}
	}

	if list, ok := o.vals["enum"].([]any); ok {
		n := &node{kind: kindLiterals}
		for _, v := range list {
			if lit := compact(v); !slices.Contains(n.lits, lit) {
				n.lits = append(n.lits, lit)
			}
		}

		return n
	}

	switch t := o.vals["type"].(type) {
	case string:
		return c.typed(o, t, depth)
	case []any:
		alt := &node{kind: kindAlt}

		for _, x := range t {
			if s, ok := x.(string); ok {
				alt.alts = append(alt.alts, c.typed(o, s, depth))
			}
		}

		return alt
	}

	// Without a type, the keywords that only an object or an array has
	// decide; else any value.
	switch {
	case o.vals["properties"] != nil || o.vals["required"] != nil || o.vals["additionalProperties"] != nil:
		return c.typed(o, "object", depth)
	case o.vals["items"] != nil:
		return c.typed(o, "array", depth)
	}

	return anyNode
}

// typed compiles o as a value of the JSON type t.
func (c *compiler) typed(o *object, t string, depth int) *node {
	switch t {
	case "string":
		n := &node{kind: kindString, minLen: intKey(o, "minLength", 0), maxLen: intKey(o, "maxLength", -1)}
		n.date = o.vals["format"] == "date"

		return n
	case "integer", "number":
		return numberNode(o, t == "integer")
	case "boolean":
		return &node{kind: kindBool}
	case jsonNull:
		return &node{kind: kindNull}
	case "object":
		return c.objectNode(o, depth)
	case "array":
		n := &node{kind: kindArray, minItems: intKey(o, "minItems", 0), maxItems: min(intKey(o, "maxItems", maxItems), maxItems)}
		if it, ok := o.vals["items"]; ok {
			if _, list := it.([]any); !list {
				n.items = c.compile(it, depth+1)
			}
		}

		return n
	}

	return anyNode
}

// objectNode compiles an object schema. An object with "properties" takes
// those keys only, whatever its "additionalProperties": the engine writes
// no other key under true or a schema either (docs/FINDINGS.md F-055). An
// object without "properties" takes any keys, their values under an
// "additionalProperties" schema, none under false. A required key that no
// property declares is left out, so an object can always close
// (port-side).
func (c *compiler) objectNode(o *object, depth int) *node {
	n := &node{kind: kindObject, props: map[string]*node{}}

	props, hasProps := o.vals["properties"].(*object)
	if hasProps {
		for _, k := range props.keys {
			n.order = append(n.order, k)
			n.props[k] = c.compile(props.vals[k], depth+1)
		}
	}

	switch extra := o.vals["additionalProperties"].(type) {
	case nil:
		n.extra = anyNode
	case bool:
		if extra {
			n.extra = anyNode
		}
	default:
		n.extra = c.compile(extra, depth+1)
	}

	if hasProps {
		n.extra = nil
	}

	n.required = n.requiredOf(o)

	return n
}

// requiredOf are the "required" keys of o that the object n can hold.
func (n *node) requiredOf(o *object) []string {
	req, _ := o.vals["required"].([]any)

	var out []string

	for _, r := range req {
		s, ok := r.(string)
		if !ok || slices.Contains(out, s) {
			continue
		}

		if v := n.props[s]; v != nil && v.admits() || n.extra != nil && n.extra.admits() {
			out = append(out, s)
		}
	}

	return out
}

// admits reports whether some value matches n, so that the automaton only
// enters values it can finish (PR #65 review); a chain deeper than maxDepth
// (a schema that refers to itself) counts as admitting.
func (n *node) admits() bool { return n.admitsAt(0) }

func (n *node) admitsAt(depth int) bool {
	if depth > maxDepth {
		return true
	}

	switch n.kind {
	case kindNever:
		return false
	case kindLiterals:
		return len(n.lits) > 0
	case kindAlt:
		return slices.ContainsFunc(n.alts, func(a *node) bool { return a.admitsAt(depth + 1) })
	case kindString:
		if n.date {
			return n.minLen <= dateLen && (n.maxLen < 0 || n.maxLen >= dateLen)
		}

		return n.maxLen < 0 || n.minLen <= n.maxLen
	case kindInteger, kindNumber:
		return n.inhabited()
	case kindArray:
		return n.minItems <= n.maxItems && (n.minItems == 0 || n.items == nil || n.items.admitsAt(depth+1))
	}

	return true
}

// dateLen is the length of a format: date string, YYYY-MM-DD.
const dateLen = 10

// inhabited reports whether some number, or some integer, lies within n's
// bounds.
func (n *node) inhabited() bool {
	lo, hi := n.lo, n.hi
	if lo == nil || hi == nil {
		return true
	}

	if n.kind == kindInteger {
		return ceilRat(lo, n.loExcl).Cmp(floorRat(hi, n.hiExcl)) <= 0
	}

	c := lo.Cmp(hi)

	return c < 0 || c == 0 && !n.loExcl && !n.hiExcl
}

// ceilRat is the least integer above r, or at r unless excl.
func ceilRat(r *big.Rat, excl bool) *big.Int {
	q := new(big.Int).Div(r.Num(), r.Denom()) // Euclidean: the floor, the denominator being positive
	if !r.IsInt() || excl {
		q.Add(q, big.NewInt(1))
	}

	return q
}

// floorRat is the greatest integer below r, or at r unless excl.
func floorRat(r *big.Rat, excl bool) *big.Int {
	q := new(big.Int).Div(r.Num(), r.Denom())
	if r.IsInt() && excl {
		q.Sub(q, big.NewInt(1))
	}

	return q
}

// numberNode compiles a number or integer schema with its bounds.
func numberNode(o *object, integer bool) *node {
	n := &node{kind: kindNumber}
	if integer {
		n.kind = kindInteger
	}

	bound := func(key string) *big.Rat {
		v, ok := o.vals[key]
		if !ok {
			return nil
		}

		lit, ok := v.(interface{ String() string })
		if !ok {
			return nil
		}

		r, ok := new(big.Rat).SetString(lit.String())
		if !ok {
			return nil
		}

		return r
	}

	n.lo, n.hi = bound("minimum"), bound("maximum")

	if x := bound("exclusiveMinimum"); x != nil && (n.lo == nil || x.Cmp(n.lo) >= 0) {
		n.lo, n.loExcl = x, true
	}

	if x := bound("exclusiveMaximum"); x != nil && (n.hi == nil || x.Cmp(n.hi) <= 0) {
		n.hi, n.hiExcl = x, true
	}

	return n
}

// intKey is o's non-negative integer key, or def.
func intKey(o *object, key string, def int) int {
	v, ok := o.vals[key].(interface{ String() string })
	if !ok {
		return def
	}

	i, err := strconv.Atoi(v.String())
	if err != nil || i < 0 {
		return def
	}

	return i
}

// compact is v's JSON as the rendered tools write it.
func compact(v any) string {
	var b strings.Builder

	writeValue(&b, v)

	return b.String()
}
