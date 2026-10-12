package toolcall

import (
	"encoding/json"
	"math/big"
	"strings"
)

// The grammar's automaton: a set of states, each a stack of matchers, the
// value being matched on top and below it the value that contains it, in
// the phase that follows the inner value. advance feeds one byte to every
// state; a state whose top can end may also pass the byte to the matcher
// below. Matchers are comparable values, so equal states merge.

// stack is one state of the automaton, a persistent list.
type stack struct {
	top   matcher
	below *stack
}

// matcher matches one JSON value, or the call block.
type matcher interface {
	// feed appends the states that follow from consuming b, the matcher
	// sitting on below.
	feed(b byte, below *stack, out []*stack) []*stack
	// final reports whether the value may end here.
	final() bool
}

// accepting reports whether st may end here: every matcher on it may.
func (st *stack) accepting() bool {
	for ; st != nil; st = st.below {
		if !st.top.final() {
			return false
		}
	}

	return true
}

// advance appends the states that follow st on b.
func advance(st *stack, b byte, out []*stack) []*stack {
	out = st.top.feed(b, st.below, out)

	if st.below != nil && st.top.final() {
		out = advance(st.below, b, out)
	}

	return out
}

// advanceAll is the set of states that follow states on b.
func advanceAll(states []*stack, b byte) []*stack {
	var out []*stack

	for _, st := range states {
		out = advance(st, b, out)
	}

	if len(out) < 2 {
		return out
	}

	uniq := out[:0]

	for _, st := range out {
		dup := false

		for _, u := range uniq {
			if sameStack(st, u) {
				dup = true

				break
			}
		}

		if !dup {
			uniq = append(uniq, st)
		}
	}

	return uniq
}

func sameStack(a, b *stack) bool {
	for ; a != nil && b != nil; a, b = a.below, b.below {
		if a == b {
			return true
		}

		if a.top != b.top {
			return false
		}
	}

	return a == nil && b == nil
}

// step appends m on below, when ok.
func step(m matcher, ok bool, below *stack, out []*stack) []*stack {
	if !ok {
		return out
	}

	return append(out, &stack{top: m, below: below})
}

// The JSON literals.
const (
	jsonTrue  = "true"
	jsonFalse = "false"
	jsonNull  = "null"
)

// The values every JSON value may be.
var (
	anyString = &node{kind: kindString, maxLen: -1}
	anyNumber = &node{kind: kindNumber}
	anyObject = &node{kind: kindObject, extra: anyNode}
	anyArray  = &node{kind: kindArray, maxItems: maxItems}
)

// starts are the matchers of a value of n before its first byte.
func starts(n *node) []matcher {
	switch n.kind {
	case kindAny:
		return []matcher{
			strM{n: anyString},
			numM{n: anyNumber},
			litM{s: jsonTrue},
			litM{s: jsonFalse},
			litM{s: jsonNull},
			objM{n: anyObject},
			arrM{n: anyArray},
		}
	case kindString:
		return []matcher{strM{n: n}}
	case kindInteger, kindNumber:
		return []matcher{numM{n: n}}
	case kindBool:
		return []matcher{litM{s: jsonTrue}, litM{s: jsonFalse}}
	case kindNull:
		return []matcher{litM{s: jsonNull}}
	case kindObject:
		return []matcher{objM{n: n}}
	case kindArray:
		return []matcher{arrM{n: n}}
	case kindLiterals:
		out := make([]matcher, len(n.lits))
		for i, l := range n.lits {
			out[i] = litM{s: l}
		}

		return out
	case kindAlt:
		var out []matcher
		for _, a := range n.alts {
			out = append(out, starts(a)...)
		}

		return out
	}

	return nil
}

// push appends the states of a value of n on parent, which takes over
// once the value ends.
func push(n *node, parent matcher, below *stack, out []*stack) []*stack {
	under := &stack{top: parent, below: below}
	for _, m := range starts(n) {
		out = append(out, &stack{top: m, below: under})
	}

	return out
}

// pushFeed is push with the value's first byte b fed.
func pushFeed(n *node, parent matcher, below *stack, b byte, out []*stack) []*stack {
	under := &stack{top: parent, below: below}
	for _, m := range starts(n) {
		out = m.feed(b, under, out)
	}

	return out
}

// litM matches the literal s: an enum or const value, true, false, null.
type litM struct {
	s string
	i int
}

func (m litM) feed(b byte, below *stack, out []*stack) []*stack {
	ok := m.i < len(m.s) && m.s[m.i] == b
	m.i++

	return step(m, ok, below, out)
}

func (m litM) final() bool { return m.i == len(m.s) }

// strM matches a JSON string of n: valid UTF-8, no raw control
// characters, the escapes JSON has, the length bounds and format: date.
type strM struct {
	n      *node
	open   bool
	done   bool
	esc    uint8 // 1 after '\', 2…5 within \uXXXX
	need   uint8 // continuation bytes of a UTF-8 sequence still to come
	lo, hi byte  // the range of the next continuation byte
	length int   // characters, counted only when n bounds them
	prev   byte  // the date's previous character
}

func (m strM) feed(b byte, below *stack, out []*stack) []*stack {
	m, ok := m.next(b)

	return step(m, ok, below, out)
}

func (m strM) final() bool { return m.done }

// counts reports whether the string's length matters.
func (m strM) counts() bool { return m.n.date || m.n.minLen > 0 || m.n.maxLen >= 0 }

func (m strM) next(b byte) (strM, bool) {
	switch {
	case !m.open:
		m.open = b == '"'

		return m, m.open
	case m.done:
		return m, false
	case m.need > 0:
		return m.continuation(b)
	case m.esc > 0:
		return m.escape(b)
	case b == '"':
		m.done = m.length >= m.n.minLen && (!m.n.date || m.length == 10)

		return m, m.done
	case b == '\\':
		m.esc = 1

		return m, !m.n.date && m.room()
	case b < 0x20:
		return m, false
	case b < 0x80:
		if m.n.date && !dateChar(m.length, m.prev, b) {
			return m, false
		}

		m.prev = b

		return m.char()
	}

	return m.lead(b)
}

// continuation feeds a UTF-8 continuation byte.
func (m strM) continuation(b byte) (strM, bool) {
	if b < m.lo || b > m.hi {
		return m, false
	}

	m.need--
	m.lo, m.hi = 0x80, 0xBF

	if m.need > 0 {
		return m, true
	}

	return m.char()
}

// escape feeds a byte of an escape sequence.
func (m strM) escape(b byte) (strM, bool) {
	if m.esc >= 2 {
		if !isHex(b) {
			return m, false
		}

		m.esc++
		if m.esc < 6 {
			return m, true
		}

		m.esc = 0

		return m.char()
	}

	switch b {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		m.esc = 0

		return m.char()
	case 'u':
		m.esc = 2

		return m, true
	}

	return m, false
}

// lead starts a multi-byte UTF-8 sequence, without overlong forms,
// surrogates or code points past U+10FFFF.
func (m strM) lead(b byte) (strM, bool) {
	if m.n.date || !m.room() {
		return m, false
	}

	m.lo, m.hi = 0x80, 0xBF

	switch {
	case b >= 0xC2 && b <= 0xDF:
		m.need = 1
	case b == 0xE0:
		m.need, m.lo = 2, 0xA0
	case b == 0xED:
		m.need, m.hi = 2, 0x9F
	case b >= 0xE1 && b <= 0xEF:
		m.need = 2
	case b == 0xF0:
		m.need, m.lo = 3, 0x90
	case b >= 0xF1 && b <= 0xF3:
		m.need = 3
	case b == 0xF4:
		m.need, m.hi = 3, 0x8F
	default:
		return m, false
	}

	return m, true
}

// room reports whether another character fits: a sequence that starts
// must be able to end.
func (m strM) room() bool {
	return !m.counts() || m.n.maxLen < 0 || m.length < m.n.maxLen
}

// char counts a character that ended.
func (m strM) char() (strM, bool) {
	if !m.counts() {
		return m, true
	}

	m.length++

	return m, m.n.maxLen < 0 || m.length <= m.n.maxLen
}

func isHex(b byte) bool {
	return b >= '0' && b <= '9' || b >= 'a' && b <= 'f' || b >= 'A' && b <= 'F'
}

// dateChar reports whether b may be character i of a YYYY-MM-DD date after
// prev: months 01–12, days 01–31.
func dateChar(i int, prev, b byte) bool {
	switch i {
	case 4, 7:
		return b == '-'
	case 5:
		return b == '0' || b == '1'
	case 6:
		return monthDigit(prev, b)
	case 8:
		return b >= '0' && b <= '3'
	case 9:
		return dayDigit(prev, b)
	}

	return i < 4 && isDigit(b)
}

// monthDigit reports whether b may follow the month's first digit prev.
func monthDigit(prev, b byte) bool {
	if prev == '0' {
		return b >= '1' && b <= '9'
	}

	return b >= '0' && b <= '2'
}

// dayDigit reports whether b may follow the day's first digit prev.
func dayDigit(prev, b byte) bool {
	switch prev {
	case '0':
		return b >= '1' && b <= '9'
	case '3':
		return b == '0' || b == '1'
	}

	return isDigit(b)
}

// numM matches a JSON number of n: an integer without fraction or
// exponent, at most maxNumberLen characters, within the bounds. A bounded
// number has no exponent (port-side).
type numM struct {
	n   *node
	lit string
}

func (m numM) feed(b byte, below *stack, out []*stack) []*stack {
	lit := m.lit + string(b)
	if len(lit) > maxNumberLen {
		return out
	}

	// An unfinished literal needs one more character, a finished one at the
	// cap must be a number of the bounds: no state without a way out.
	valid, complete := m.syntax(lit)
	if !valid || !complete && len(lit) == maxNumberLen || m.bounded() && !m.reachable(lit) {
		return out
	}

	if len(lit) == maxNumberLen && !(numM{n: m.n, lit: lit}).final() {
		return out
	}

	return append(out, &stack{top: numM{n: m.n, lit: lit}, below: below})
}

func (m numM) final() bool {
	_, complete := m.syntax(m.lit)
	if !complete {
		return false
	}

	if !m.bounded() {
		return true
	}

	v, ok := new(big.Rat).SetString(m.lit)

	return ok && m.within(v, true, v, true)
}

func (m numM) bounded() bool { return m.n.lo != nil || m.n.hi != nil }

// syntax reports whether s starts a number of m's kind and whether it is
// one.
func (m numM) syntax(s string) (valid, complete bool) {
	integer := m.n.kind == kindInteger

	i, valid := intRun(s, len(s)-len(strings.TrimPrefix(s, "-")))

	if valid && i >= 0 && i < len(s) && s[i] == '.' && !integer {
		i, valid = digitRun(s, i+1)
	}

	if valid && i >= 0 && i < len(s) && (s[i] == 'e' || s[i] == 'E') && !integer && !m.bounded() {
		i, valid = expRun(s, i+1)
	}

	if !valid || i < 0 {
		return valid, false
	}

	return i == len(s), i == len(s)
}

// expRun reads the exponent's digits after "e" at i, with their sign.
func expRun(s string, i int) (end int, valid bool) {
	if i < len(s) && (s[i] == '+' || s[i] == '-') {
		i++
	}

	return digitRun(s, i)
}

// intRun reads the integer part at i: its end, or -1 with valid when s
// ends before it.
func intRun(s string, i int) (end int, valid bool) {
	switch {
	case i == len(s):
		return -1, true
	case s[i] == '0':
		return i + 1, true
	case s[i] >= '1' && s[i] <= '9':
		return digitsFrom(s, i), true
	}

	return 0, false
}

// digitRun reads the digits that must follow at i: their end, or -1 with
// valid when s ends before the first.
func digitRun(s string, i int) (end int, valid bool) {
	if i == len(s) {
		return -1, true
	}

	j := digitsFrom(s, i)

	return j, j > i
}

func digitsFrom(s string, i int) int {
	for i < len(s) && isDigit(s[i]) {
		i++
	}

	return i
}

// reachable reports whether some number that starts with the prefix lit
// (no exponent) lies within the bounds: "a bounded number cannot grow past
// its maximum".
func (m numM) reachable(lit string) bool {
	neg := strings.HasPrefix(lit, "-")
	body := strings.TrimPrefix(lit, "-")

	if body == "" {
		// Any number of the sign: (-∞, 0] or [0, ∞).
		return m.span(new(big.Rat), nil, false, neg)
	}

	whole, frac, dot := strings.Cut(body, ".")
	v, _ := new(big.Rat).SetString(whole)

	if dot {
		// [w, w + 10^-d) for the d fraction digits so far.
		w, _ := new(big.Rat).SetString(whole + "." + frac + "0")
		ulp := new(big.Rat).SetFrac(big.NewInt(1), pow10(len(frac)))

		return m.span(w, new(big.Rat).Add(w, ulp), false, neg)
	}

	integer := m.n.kind == kindInteger

	if whole == "0" {
		// 0, or 0.x below 1.
		if integer {
			return m.span(v, v, true, neg)
		}

		return m.span(v, big.NewRat(1, 1), false, neg)
	}

	// [v·10^k, (v+1)·10^k) for each count k of digits still to come.
	for k := range maxNumberLen - len(lit) + 1 {
		p := new(big.Rat).SetInt(pow10(k))
		lo := new(big.Rat).Mul(v, p)
		hi := new(big.Rat).Add(lo, p)

		if integer {
			hi.Sub(hi, big.NewRat(1, 1))
		}

		if m.span(lo, hi, integer, neg) {
			return true
		}
	}

	return false
}

// span reports whether the numbers from a (included) to b (nil:
// unbounded; included with bIn), negated with neg, meet the bounds.
func (m numM) span(a, b *big.Rat, bIn, neg bool) bool {
	if neg {
		return m.within(negRat(b), bIn, negRat(a), true)
	}

	return m.within(a, true, b, bIn)
}

func negRat(r *big.Rat) *big.Rat {
	if r == nil {
		return nil
	}

	return new(big.Rat).Neg(r)
}

// within reports whether the interval from a to b (nil: unbounded; aIn,
// bIn for closed ends) meets the bounds.
func (m numM) within(a *big.Rat, aIn bool, b *big.Rat, bIn bool) bool {
	n := m.n

	lo, loIn := a, aIn
	if n.lo != nil && (lo == nil || tighter(n.lo, lo, n.loExcl, 1)) {
		lo, loIn = n.lo, !n.loExcl
	}

	hi, hiIn := b, bIn
	if n.hi != nil && (hi == nil || tighter(n.hi, hi, n.hiExcl, -1)) {
		hi, hiIn = n.hi, !n.hiExcl
	}

	if lo == nil || hi == nil {
		return true
	}

	c := lo.Cmp(hi)

	return c < 0 || c == 0 && loIn && hiIn
}

// tighter reports whether the bound x (exclusive with excl) cuts more than
// the end y: past it in the direction dir, or at it and open.
func tighter(x, y *big.Rat, excl bool, dir int) bool {
	c := x.Cmp(y)

	return c == dir || c == 0 && excl
}

func pow10(k int) *big.Int {
	return new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(k)), nil)
}

// Object phases.
const (
	objOpen  = iota // before '{'
	objFirst        // after '{': a key or '}'
	objKey          // within a key
	objColon        // after a key
	objNext         // after a value: ',' or '}'
	objComma        // after ',': a key
	objDone         // after '}'
)

// objM matches a JSON object of n: each key once, a property or, under
// additionalProperties, another key, every required key before '}'.
type objM struct {
	n     *node
	phase uint8
	raw   string // the key's bytes so far; the key once it ends
	ks    strM   // the key as a string
	seen  string // the keys so far, each followed by NUL
}

func (m objM) final() bool { return m.phase == objDone }

func (m objM) feed(b byte, below *stack, out []*stack) []*stack {
	switch m.phase {
	case objOpen:
		m.phase = objFirst

		return step(m, b == '{', below, out)
	case objFirst, objNext:
		return m.after(b, below, out)
	case objComma:
		return step(m.keyStart(), b == '"', below, out)
	case objKey:
		return m.keyByte(b, below, out)
	case objColon:
		if b != ':' {
			return out
		}

		key := m.raw
		m.phase, m.raw, m.ks = objNext, "", strM{}
		m.seen += key + "\x00"

		return push(m.value(key), m, below, out)
	}

	return out
}

// after feeds b after '{' or after a value.
func (m objM) after(b byte, below *stack, out []*stack) []*stack {
	switch {
	case b == '}':
		m.phase = objDone

		return step(m, m.complete(), below, out)
	case m.phase == objFirst && b == '"':
		return step(m.keyStart(), true, below, out)
	case m.phase == objNext && b == ',':
		m.phase = objComma

		return step(m, m.more(), below, out)
	}

	return out
}

func (m objM) keyStart() objM {
	m.phase, m.raw = objKey, ""
	m.ks, _ = strM{n: anyString}.next('"')

	return m
}

// keyByte feeds b to the key. A key ends as one of the properties not seen
// yet or, under additionalProperties, as any other key not seen yet.
func (m objM) keyByte(b byte, below *stack, out []*stack) []*stack {
	ks, ok := m.ks.next(b)
	if !ok {
		return out
	}

	m.ks = ks
	if !ks.done {
		m.raw += string(b)

		return step(m, m.n.extra != nil || m.prefixOfProperty(), below, out)
	}

	var key string

	err := json.Unmarshal([]byte(`"`+m.raw+`"`), &key)
	if err != nil || m.hasSeen(key) {
		return out
	}

	if v := m.value(key); v == nil || !v.admits() {
		return out
	}

	m.phase, m.raw = objColon, key

	return step(m, true, below, out)
}

// prefixOfProperty reports whether raw starts the spelling of a property
// not seen yet that admits a value.
func (m objM) prefixOfProperty() bool {
	for _, p := range m.n.order {
		if !m.hasSeen(p) && m.n.props[p].admits() && strings.HasPrefix(jsonBody(p), m.raw) {
			return true
		}
	}

	return false
}

func (m objM) hasSeen(key string) bool {
	return strings.Contains("\x00"+m.seen, "\x00"+key+"\x00")
}

// complete reports whether every required key was seen.
func (m objM) complete() bool {
	for _, r := range m.n.required {
		if !m.hasSeen(r) {
			return false
		}
	}

	return true
}

// more reports whether another key may follow.
func (m objM) more() bool {
	if m.n.extra != nil {
		return true
	}

	for _, p := range m.n.order {
		if !m.hasSeen(p) && m.n.props[p].admits() {
			return true
		}
	}

	return false
}

// value is the node of key's value.
func (m objM) value(key string) *node {
	if v, ok := m.n.props[key]; ok {
		return v
	}

	return m.n.extra
}

// jsonBody is s as a JSON string without its quotes, as the rendered tools
// spell it.
func jsonBody(s string) string {
	var b strings.Builder

	writeString(&b, s)

	q := b.String()

	return q[1 : len(q)-1]
}

// Array phases.
const (
	arrOpen  = iota // before '['
	arrFirst        // after '[': a value or ']'
	arrNext         // after a value: ',' or ']'
	arrComma        // after ',': a value
	arrDone         // after ']'
)

// arrM matches a JSON array of n, within its item counts.
type arrM struct {
	n     *node
	phase uint8
	count int
}

func (m arrM) final() bool { return m.phase == arrDone }

func (m arrM) feed(b byte, below *stack, out []*stack) []*stack {
	switch m.phase {
	case arrOpen:
		m.phase = arrFirst

		return step(m, b == '[', below, out)
	case arrFirst, arrComma:
		if b == ']' && m.phase == arrFirst {
			m.phase = arrDone

			return step(m, m.n.minItems == 0, below, out)
		}

		if m.count >= m.n.maxItems {
			return out
		}

		items := m.n.items
		if items == nil {
			items = anyNode
		}

		m.phase = arrNext
		m.count++

		return pushFeed(items, m, below, b, out)
	case arrNext:
		switch b {
		case ',':
			m.phase = arrComma

			return step(m, m.count < m.n.maxItems, below, out)
		case ']':
			m.phase = arrDone

			return step(m, m.count >= m.n.minItems, below, out)
		}
	}

	return out
}

// callsM matches the call block, a list of calls of g; an empty list only
// when g does not require a call.
type callsM struct {
	g     *Grammar
	phase uint8
}

func (m callsM) final() bool { return m.phase == arrDone }

func (m callsM) feed(b byte, below *stack, out []*stack) []*stack {
	switch m.phase {
	case arrOpen:
		m.phase = arrFirst

		return step(m, b == '[', below, out)
	case arrFirst, arrComma:
		if b == ']' && m.phase == arrFirst {
			m.phase = arrDone

			return step(m, !m.g.required, below, out)
		}

		m.phase = arrNext

		return step(callM{g: m.g}, b == '{', &stack{top: m, below: below}, out)
	case arrNext:
		switch b {
		case ',':
			m.phase = arrComma

			return step(m, true, below, out)
		case ']':
			m.phase = arrDone

			return step(m, true, below, out)
		}
	}

	return out
}

// Call phases, after its '{'.
const (
	callName = iota // `"name":"`
	callTool        // the tool's name and its closing quote
	callArgs        // `,"arguments":`
	callEnd         // after the arguments: '}'
	callDone
)

const (
	callNameLit = `"name":"`
	callArgsLit = `,"arguments":`
)

// callM matches one call of g after its '{': the name, then the arguments
// of that tool, then '}'.
type callM struct {
	g     *Grammar
	phase uint8
	i     int    // position in a literal
	raw   string // the name's bytes so far, its closing quote included
}

func (m callM) final() bool { return m.phase == callDone }

func (m callM) feed(b byte, below *stack, out []*stack) []*stack {
	switch m.phase {
	case callName, callArgs:
		lit := callNameLit
		if m.phase == callArgs {
			lit = callArgsLit
		}

		if b != lit[m.i] {
			return out
		}

		m.i++
		if m.i < len(lit) {
			return step(m, true, below, out)
		}

		m.i = 0
		if m.phase == callName {
			m.phase = callTool

			return step(m, true, below, out)
		}

		m.phase = callEnd

		return push(m.g.args[m.g.toolOf(m.raw)], m, below, out)
	case callTool:
		m.raw += string(b)

		for _, t := range m.g.tools {
			body := jsonBody(t) + `"`
			if strings.HasPrefix(body, m.raw) {
				if body == m.raw {
					m.phase = callArgs
				}

				return step(m, true, below, out)
			}
		}

		return out
	case callEnd:
		m.phase = callDone

		return step(m, b == '}', below, out)
	}

	return out
}

// toolOf is the tool whose name, spelled and closed, is raw.
func (g *Grammar) toolOf(raw string) string {
	for _, t := range g.tools {
		if jsonBody(t)+`"` == raw {
			return t
		}
	}

	return ""
}
