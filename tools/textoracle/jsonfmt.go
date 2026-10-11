//go:build oracle

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// kind is the JSON type of a node.
type kind int

const (
	kindNull kind = iota
	kindBool
	kindNumber
	kindString
	kindArray
	kindObject
)

// node is a decoded JSON value that keeps what encoding/json's maps lose: the order of an object's keys (the engine
// emits them in a fixed order, and the order of a call's arguments is the order the model generated them) and the
// literal of a number.
type node struct {
	kind  kind
	str   string // the string value, or the number literal
	b     bool
	elems []node   // array elements, or object values
	keys  []string // object keys, parallel to elems
}

// parseJSON decodes exactly one JSON value.
func parseJSON(b []byte) (node, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()

	n, err := decodeNode(dec)
	if err != nil {
		return node{}, err
	}

	_, err = dec.Token()
	if !errors.Is(err, io.EOF) {
		return node{}, fmt.Errorf("trailing data after the JSON value: %q", b)
	}

	return n, nil
}

func decodeNode(dec *json.Decoder) (node, error) {
	tok, err := dec.Token()
	if err != nil {
		return node{}, fmt.Errorf("decode json: %w", err)
	}

	switch t := tok.(type) {
	case nil:
		return node{kind: kindNull}, nil
	case bool:
		return node{kind: kindBool, b: t}, nil
	case json.Number:
		return node{kind: kindNumber, str: t.String()}, nil
	case string:
		return node{kind: kindString, str: t}, nil
	case json.Delim:
		if t == '[' {
			return decodeArray(dec)
		}

		return decodeObject(dec)
	default:
		return node{}, fmt.Errorf("unexpected json token %v", tok)
	}
}

func decodeArray(dec *json.Decoder) (node, error) {
	n := node{kind: kindArray}

	for dec.More() {
		e, err := decodeNode(dec)
		if err != nil {
			return node{}, err
		}

		n.elems = append(n.elems, e)
	}

	_, err := dec.Token() // ']'
	if err != nil {
		return node{}, fmt.Errorf("decode json: %w", err)
	}

	return n, nil
}

func decodeObject(dec *json.Decoder) (node, error) {
	n := node{kind: kindObject}

	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return node{}, fmt.Errorf("decode json: %w", err)
		}

		key, ok := tok.(string)
		if !ok {
			return node{}, fmt.Errorf("object key %v is not a string", tok)
		}

		if n.get(key) != nil {
			return node{}, fmt.Errorf("duplicate object key %q", key)
		}

		v, err := decodeNode(dec)
		if err != nil {
			return node{}, err
		}

		n.keys = append(n.keys, key)
		n.elems = append(n.elems, v)
	}

	_, err := dec.Token() // '}'
	if err != nil {
		return node{}, fmt.Errorf("decode json: %w", err)
	}

	return n, nil
}

// get returns the value of an object's key, or nil.
func (n node) get(key string) *node {
	for i, k := range n.keys {
		if k == key {
			return &n.elems[i]
		}
	}

	return nil
}

// without returns a copy of the object n without the keys drop reports.
func (n node) without(drop func(string) bool) node {
	out := node{kind: kindObject}

	for i, k := range n.keys {
		if drop(k) {
			continue
		}

		out.keys = append(out.keys, k)
		out.elems = append(out.elems, n.elems[i])
	}

	return out
}

// timingSuffixes mark the engine's per-run measurements: prefill_tps, decode_tps and peak_ram_mb on every answer, and
// audio_ttft_ms and audio_decode_tps when the turn was speech. They vary from run to run and machine to machine.
var timingSuffixes = []string{"_tps", "_ms", "_mb"}

// isTimingKey reports whether a top-level engine key is a measurement rather than part of the answer.
func isTimingKey(k string) bool {
	for _, s := range timingSuffixes {
		if strings.HasSuffix(k, s) {
			return true
		}
	}

	return false
}

// normaliseResponse parses one engine answer (a JSON object) and drops its timing fields; everything else, the key
// order included, is kept. A word time such as audio_words[].start is data, not timing: only top-level keys go.
func normaliseResponse(out []byte) (node, error) {
	n, err := parseJSON(bytes.TrimSpace(out))
	if err != nil {
		return node{}, fmt.Errorf("engine output: %w", err)
	}

	if n.kind != kindObject {
		return node{}, fmt.Errorf("engine output is not a JSON object: %q", out)
	}

	return n.without(isTimingKey), nil
}

// printWidth is prettier's default line width; the fixture writer lays arrays out as prettier would, so that
// `treefmt --fail-on-change` leaves the fixtures alone.
const printWidth = 80

// marshalPretty encodes n the way prettier 3 prints a .json file whose objects are all written expanded: every
// non-empty object is broken one key per line (prettier keeps an object expanded when its first key starts a new
// line), an array is printed on one line when it fits and holds no non-empty object, else one element per line,
// or, for an array of numbers, filled. Numbers are rewritten in prettier's canonical form (canonicalNumber), which
// keeps their value. The output ends with a newline.
func marshalPretty(n node) []byte {
	var b strings.Builder

	writePretty(&b, n, 0, 0, 0)
	b.WriteByte('\n')

	return []byte(b.String())
}

// writePretty writes n starting at column col with the given indentation; tail is the number of characters that
// follow n on its line (1 for a separating comma).
func writePretty(b *strings.Builder, n node, indent, col, tail int) {
	switch n.kind {
	case kindObject:
		if len(n.keys) == 0 {
			b.WriteString("{}")

			return
		}

		pad := strings.Repeat(" ", indent+2)

		b.WriteString("{\n")

		for i, k := range n.keys {
			key := quote(k) + ": "
			t := 0

			if i < len(n.keys)-1 {
				t = 1
			}

			b.WriteString(pad + key)
			writePretty(b, n.elems[i], indent+2, indent+2+width(key), t)

			if t == 1 {
				b.WriteByte(',')
			}

			b.WriteByte('\n')
		}

		b.WriteString(strings.Repeat(" ", indent) + "}")
	case kindArray:
		writeArray(b, n, indent, col, tail)
	default:
		b.WriteString(flat(n))
	}
}

func writeArray(b *strings.Builder, n node, indent, col, tail int) {
	if f, ok := flatArray(n); ok && col+width(f)+tail <= printWidth {
		b.WriteString(f)

		return
	}

	pad := strings.Repeat(" ", indent+2)

	b.WriteString("[\n" + pad)

	if allNumbers(n) {
		// prettier's fill: as many numbers per line as fit, each but the last followed by its comma.
		pos := indent + 2

		for i, e := range n.elems {
			s := flat(e)
			if i < len(n.elems)-1 {
				s += ","
			}

			switch {
			case i == 0:
			case pos+1+width(s) <= printWidth:
				b.WriteByte(' ')

				pos++
			default:
				b.WriteString("\n" + pad)

				pos = indent + 2
			}

			b.WriteString(s)

			pos += width(s)
		}
	} else {
		for i, e := range n.elems {
			t := 0
			if i < len(n.elems)-1 {
				t = 1
			}

			writePretty(b, e, indent+2, indent+2, t)

			if t == 1 {
				b.WriteString(",\n" + pad)
			}
		}
	}

	b.WriteString("\n" + strings.Repeat(" ", indent) + "]")
}

// flatArray returns the one-line form of an array, and false when prettier would break it whatever the width: it
// holds a non-empty object, or an array that must break, or more than one element, all arrays of two or more
// elements.
func flatArray(n node) (string, bool) {
	if len(n.elems) == 0 {
		return "[]", true
	}

	if len(n.elems) > 1 && allNested(n) {
		return "", false
	}

	parts := make([]string, len(n.elems))

	for i, e := range n.elems {
		switch {
		case e.kind == kindObject && len(e.keys) > 0:
			return "", false
		case e.kind == kindArray:
			f, ok := flatArray(e)
			if !ok {
				return "", false
			}

			parts[i] = f
		default:
			parts[i] = flat(e)
		}
	}

	return "[" + strings.Join(parts, ", ") + "]", true
}

// allNested reports prettier's forced break of an array: every element an array of two or more elements.
func allNested(n node) bool {
	for _, e := range n.elems {
		if e.kind != kindArray || len(e.elems) < 2 {
			return false
		}
	}

	return true
}

func allNumbers(n node) bool {
	if len(n.elems) < 2 {
		return false
	}

	for _, e := range n.elems {
		if e.kind != kindNumber {
			return false
		}
	}

	return true
}

// flat is a scalar or an empty object's one-line form.
func flat(n node) string {
	switch n.kind {
	case kindNull:
		return "null"
	case kindBool:
		if n.b {
			return "true"
		}

		return "false"
	case kindNumber:
		return canonicalNumber(n.str)
	case kindString:
		return quote(n.str)
	case kindObject:
		return "{}"
	default:
		f, _ := flatArray(n)

		return f
	}
}

// width is the printed width of s in columns. The fixtures hold Latin text only, where a rune is one column.
func width(s string) int { return utf8.RuneCountInString(s) }

// quote encodes s as a JSON string without encoding/json's HTML escaping, so that "<" stays "<".
func quote(s string) string {
	var b bytes.Buffer

	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)

	err := enc.Encode(s)
	if err != nil {
		panic(err) // unreachable: every Go string encodes
	}

	return strings.TrimSuffix(b.String(), "\n")
}

// canonicalNumber rewrites a JSON number literal as prettier prints it: lower-case exponent without "+" or leading
// zeros, no exponent of zero, and no trailing zeros after the first fraction digit ("0.9490" becomes "0.949",
// "1.0000" becomes "1.0"). The value is unchanged.
func canonicalNumber(lit string) string {
	mant, exp, hasExp := strings.Cut(strings.ToLower(lit), "e")
	if hasExp {
		exp, hasExp = canonicalExponent(exp)
	}

	if intPart, frac, ok := strings.Cut(mant, "."); ok {
		frac = strings.TrimRight(frac, "0")
		if frac == "" && len(mant) > len(intPart)+1 {
			frac = "0" // "1.0000" keeps one zero
		}

		mant = intPart
		if frac != "" {
			mant += "." + frac
		}
	}

	if hasExp {
		return mant + "e" + exp
	}

	return mant
}

// canonicalExponent drops an exponent's "+" and leading zeros; it reports false for an exponent of zero.
func canonicalExponent(exp string) (string, bool) {
	sign := ""

	switch {
	case strings.HasPrefix(exp, "-"):
		sign, exp = "-", exp[1:]
	case strings.HasPrefix(exp, "+"):
		exp = exp[1:]
	}

	exp = strings.TrimLeft(exp, "0")
	if exp == "" {
		return "", false
	}

	return sign + exp, true
}

// pyDumps encodes n as Python's json.dumps does with its defaults: ", " and ": " separators and non-ASCII escaped.
// upstream Needle.run feeds tool results back to the engine as json.dumps(results), so a fixture turn that carries
// tool results is spelled the same way.
func pyDumps(n node) string {
	var b strings.Builder

	writePy(&b, n)

	return b.String()
}

func writePy(b *strings.Builder, n node) {
	switch n.kind {
	case kindArray:
		b.WriteByte('[')

		for i, e := range n.elems {
			if i > 0 {
				b.WriteString(", ")
			}

			writePy(b, e)
		}

		b.WriteByte(']')
	case kindObject:
		b.WriteByte('{')

		for i, k := range n.keys {
			if i > 0 {
				b.WriteString(", ")
			}

			b.WriteString(asciiQuote(k) + ": ")
			writePy(b, n.elems[i])
		}

		b.WriteByte('}')
	case kindString:
		b.WriteString(asciiQuote(n.str))
	case kindNumber:
		b.WriteString(n.str)
	default:
		b.WriteString(flat(n))
	}
}

// asciiQuote is Python's json.dumps string encoding with ensure_ascii: every rune above U+007E as \uXXXX (a UTF-16
// surrogate pair above U+FFFF).
func asciiQuote(s string) string {
	var b strings.Builder

	b.WriteByte('"')

	for _, r := range s {
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20 || (r > 0x7e && r <= 0xffff):
			fmt.Fprintf(&b, `\u%04x`, r)
		case r > 0xffff:
			r -= 0x10000
			fmt.Fprintf(&b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
		default:
			b.WriteRune(r)
		}
	}

	b.WriteByte('"')

	return b.String()
}

// compact encodes n on one line, key order and number literals kept, for embedding in a struct as json.RawMessage.
func compact(n node) json.RawMessage {
	var b strings.Builder

	writeCompact(&b, n)

	return json.RawMessage(b.String())
}

func writeCompact(b *strings.Builder, n node) {
	switch n.kind {
	case kindArray:
		b.WriteByte('[')

		for i, e := range n.elems {
			if i > 0 {
				b.WriteByte(',')
			}

			writeCompact(b, e)
		}

		b.WriteByte(']')
	case kindObject:
		b.WriteByte('{')

		for i, k := range n.keys {
			if i > 0 {
				b.WriteByte(',')
			}

			b.WriteString(quote(k) + ":")
			writeCompact(b, n.elems[i])
		}

		b.WriteByte('}')
	case kindNumber:
		b.WriteString(n.str) // the literal, not yet canonical
	default:
		b.WriteString(flat(n))
	}
}

// prettyJSON encodes v with encoding/json (struct fields in declaration order, map keys sorted, json.RawMessage
// verbatim) and lays the result out with marshalPretty.
func prettyJSON(v any) ([]byte, error) {
	js, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("encode json: %w", err)
	}

	n, err := parseJSON(js)
	if err != nil {
		return nil, err
	}

	return marshalPretty(n), nil
}
