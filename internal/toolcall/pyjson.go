package toolcall

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ToolsJSON returns the tool schemas raw, a JSON array, as upstream's
// render_example writes them into the prompt: Python's
// json.dumps(json.loads(raw), separators=(",", ":"), ensure_ascii=False)
// (docs/FINDINGS.md F-048). That keeps the caller's key order (a repeated
// key keeps its first place and its last value, as a dict does), writes
// non-ASCII text raw and escapes only what Python escapes, writes integers
// in full and every number with a '.', 'e' or 'E' as Python's float repr
// ("100.0" for 1e2, "1e-07", "Infinity" past the float range). Like
// json.loads it also reads NaN, Infinity and -Infinity.
func ToolsJSON(raw []byte) (string, error) {
	// json.loads raises on bytes that are not UTF-8, where encoding/json
	// would put U+FFFD in their place (PR #55 review).
	if !utf8.Valid(raw) {
		return "", errors.New("toolcall: tools: not valid UTF-8")
	}

	p := &pyParser{s: raw}

	v, err := p.value()
	if err != nil {
		return "", fmt.Errorf("toolcall: tools: %w", err)
	}

	p.space()

	if p.i != len(p.s) {
		return "", fmt.Errorf("toolcall: tools: data after the JSON value at byte %d", p.i)
	}

	if _, ok := v.([]any); !ok {
		return "", errors.New("toolcall: tools: not a JSON array")
	}

	var b strings.Builder

	writeValue(&b, v)

	return b.String(), nil
}

// object is a JSON object in Python's dict order.
type object struct {
	keys []string
	vals map[string]any
}

// pyParser reads JSON as Python's json.loads does: standard JSON plus the
// literals NaN, Infinity and -Infinity, which encoding/json rejects.
// Values are an *object, a []any, a string, a json.Number (the literal,
// non-finite ones included), a bool or nil.
type pyParser struct {
	s []byte
	i int
}

// number is a JSON number literal.
var number = regexp.MustCompile(`^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?`)

func (p *pyParser) space() {
	for p.i < len(p.s) && strings.IndexByte(" \t\n\r", p.s[p.i]) >= 0 {
		p.i++
	}
}

func (p *pyParser) errorf(format string, a ...any) error {
	return fmt.Errorf("byte %d: %s", p.i, fmt.Sprintf(format, a...))
}

func (p *pyParser) value() (any, error) {
	p.space()

	if p.i >= len(p.s) {
		return nil, p.errorf("unexpected end of input")
	}

	rest := p.s[p.i:]

	for _, lit := range []struct {
		text string
		v    any
	}{
		{"true", true},
		{"false", false},
		{"null", nil},
		{"NaN", json.Number("NaN")},
		{"Infinity", json.Number("Infinity")},
		{"-Infinity", json.Number("-Infinity")},
	} {
		if bytes.HasPrefix(rest, []byte(lit.text)) {
			p.i += len(lit.text)

			return lit.v, nil
		}
	}

	switch rest[0] {
	case '[':
		return p.array()
	case '{':
		return p.object()
	case '"':
		return p.str()
	}

	if m := number.Find(rest); m != nil {
		p.i += len(m)

		return json.Number(m), nil
	}

	return nil, p.errorf("invalid character %q", rest[0])
}

func (p *pyParser) array() (any, error) {
	p.i++ // [

	out := []any{}

	p.space()

	if p.i < len(p.s) && p.s[p.i] == ']' {
		p.i++

		return out, nil
	}

	for {
		v, err := p.value()
		if err != nil {
			return nil, err
		}

		out = append(out, v)

		done, err := p.next(']')
		if err != nil || done {
			return out, err
		}
	}
}

func (p *pyParser) object() (any, error) {
	p.i++ // {

	o := &object{vals: map[string]any{}}

	p.space()

	if p.i < len(p.s) && p.s[p.i] == '}' {
		p.i++

		return o, nil
	}

	for {
		p.space()

		if p.i >= len(p.s) || p.s[p.i] != '"' {
			return nil, p.errorf("want an object key")
		}

		key, err := p.str()
		if err != nil {
			return nil, err
		}

		p.space()

		if p.i >= len(p.s) || p.s[p.i] != ':' {
			return nil, p.errorf("want ':'")
		}

		p.i++

		v, err := p.value()
		if err != nil {
			return nil, err
		}

		if _, seen := o.vals[key]; !seen {
			o.keys = append(o.keys, key)
		}

		o.vals[key] = v

		done, err := p.next('}')
		if err != nil || done {
			return o, err
		}
	}
}

// next consumes ',' (false) or the closing delimiter (true).
func (p *pyParser) next(closing byte) (bool, error) {
	p.space()

	if p.i < len(p.s) {
		switch p.s[p.i] {
		case ',':
			p.i++

			return false, nil
		case closing:
			p.i++

			return true, nil
		}
	}

	return false, p.errorf("want ',' or %q", closing)
}

// str reads a string literal; encoding/json decodes its escapes and, as
// json.loads does, rejects raw control characters.
func (p *pyParser) str() (string, error) {
	start := p.i

	for p.i++; p.i < len(p.s); p.i++ {
		switch p.s[p.i] {
		case '\\':
			p.i++
		case '"':
			p.i++

			err := loneSurrogate(p.s[start+1 : p.i-1])
			if err != nil {
				return "", p.errorf("string: %v", err)
			}

			var out string

			err = json.Unmarshal(p.s[start:p.i], &out)
			if err != nil {
				return "", p.errorf("string: %v", err)
			}

			return out, nil
		}
	}

	return "", p.errorf("unterminated string")
}

func writeValue(b *strings.Builder, v any) {
	switch t := v.(type) {
	case *object:
		b.WriteByte('{')

		for i, k := range t.keys {
			if i > 0 {
				b.WriteByte(',')
			}

			writeString(b, k)
			b.WriteByte(':')
			writeValue(b, t.vals[k])
		}

		b.WriteByte('}')
	case []any:
		b.WriteByte('[')

		for i, e := range t {
			if i > 0 {
				b.WriteByte(',')
			}

			writeValue(b, e)
		}

		b.WriteByte(']')
	case string:
		writeString(b, t)
	case json.Number:
		b.WriteString(pyNumber(string(t)))
	case bool:
		b.WriteString(strconv.FormatBool(t))
	default:
		b.WriteString("null")
	}
}

// writeString writes s as Python's json.dumps with ensure_ascii=False: '"'
// and '\\' escaped, \n \r \t \b \f by name, other control characters as
// \u00xx in lower case, everything else as is.
func writeString(b *strings.Builder, s string) {
	b.WriteByte('"')

	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}

	b.WriteByte('"')
}

// loneSurrogate reports an escaped surrogate (\uD800–\uDFFF) that is not
// a high one followed by an escaped low one. Python's json.loads keeps
// such a code point, which its tokenizer then cannot encode, while
// encoding/json would put U+FFFD in its place and change the prompt
// silently (PR #55 review).
func loneSurrogate(lit []byte) error {
	for i := 0; i < len(lit); i++ {
		if lit[i] != '\\' {
			continue
		}

		i++

		if i >= len(lit) || lit[i] != 'u' {
			continue
		}

		r := hexRune(lit[i+1:])
		i += 4

		switch {
		case r >= 0xd800 && r < 0xdc00:
			if len(lit) > i+2 && lit[i+1] == '\\' && lit[i+2] == 'u' {
				if lo := hexRune(lit[i+3:]); lo >= 0xdc00 && lo < 0xe000 {
					i += 6

					continue
				}
			}

			return fmt.Errorf("lone surrogate \\u%04x", r)
		case r >= 0xdc00 && r < 0xe000:
			return fmt.Errorf("lone surrogate \\u%04x", r)
		}
	}

	return nil
}

// hexRune is the code point of the 4 hex digits at the start of b, -1 if
// there are none.
func hexRune(b []byte) rune {
	if len(b) < 4 {
		return -1
	}

	n, err := strconv.ParseUint(string(b[:4]), 16, 32)
	if err != nil {
		return -1
	}

	return rune(n)
}

// pyNumber is a JSON number literal as Python's json module reads and
// writes it: an integer in full ("-0" is 0), anything with a fraction or an
// exponent as a float in repr.
func pyNumber(lit string) string {
	if !strings.ContainsAny(lit, ".eE") {
		n, ok := new(big.Int).SetString(lit, 10)
		if ok {
			return n.String()
		}
	}

	f, err := strconv.ParseFloat(lit, 64)
	if err != nil && !math.IsInf(f, 0) {
		return lit // json.Decoder has checked the syntax
	}

	return pyFloat(f)
}

// pyFloat is Python's float repr as json.dumps writes it: the shortest
// digits that round-trip, positional for decimal exponents from -4 up to
// 16 with ".0" on integral values, else d.ddde±XX, and Infinity, -Infinity
// and NaN.
func pyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case math.IsNaN(f):
		return "NaN"
	}

	sign := ""
	if math.Signbit(f) {
		sign, f = "-", -f
	}

	if f == 0 {
		return sign + "0.0"
	}

	// FormatFloat 'e' with -1 gives the shortest digits as d.ddde±x.
	e := strconv.FormatFloat(f, 'e', -1, 64)
	mant, exp, _ := strings.Cut(e, "e")
	digits := strings.Replace(mant, ".", "", 1)
	x, _ := strconv.Atoi(exp)
	decpt := x + 1 // f = 0.digits × 10^decpt

	if decpt > -4 && decpt <= 16 {
		switch {
		case decpt <= 0:
			return sign + "0." + strings.Repeat("0", -decpt) + digits
		case decpt >= len(digits):
			return sign + digits + strings.Repeat("0", decpt-len(digits)) + ".0"
		default:
			return sign + digits[:decpt] + "." + digits[decpt:]
		}
	}

	m := digits[:1]
	if len(digits) > 1 {
		m += "." + digits[1:]
	}

	es := "+"
	if x < 0 {
		es, x = "-", -x
	}

	return fmt.Sprintf("%s%se%s%02d", sign, m, es, x)
}
