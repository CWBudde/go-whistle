package toolcall

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// ToolsJSON returns the tool schemas raw, a JSON array, as upstream's
// render_example writes them into the prompt: Python's
// json.dumps(json.loads(raw), separators=(",", ":"), ensure_ascii=False)
// (docs/FINDINGS.md F-048). That keeps the caller's key order (a repeated
// key keeps its first place and its last value, as a dict does), writes
// non-ASCII text raw and escapes only what Python escapes, writes integers
// in full and every number with a '.', 'e' or 'E' as Python's float repr
// ("100.0" for 1e2, "1e-07", "Infinity" past the float range).
func ToolsJSON(raw []byte) (string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()

	v, err := decodeValue(dec)
	if err != nil {
		return "", fmt.Errorf("toolcall: tools: %w", err)
	}

	_, err = dec.Token()
	if !errors.Is(err, io.EOF) {
		return "", errors.New("toolcall: tools: data after the JSON value")
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

// decodeValue reads one JSON value: an object, a []any, a string, a
// json.Number, a bool or nil.
func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}

	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '[':
			return decodeArray(dec)
		case '{':
			return decodeObject(dec)
		}

		return nil, fmt.Errorf("unexpected %v", t)
	default:
		return t, nil
	}
}

func decodeArray(dec *json.Decoder) (any, error) {
	out := []any{}

	for dec.More() {
		v, err := decodeValue(dec)
		if err != nil {
			return nil, err
		}

		out = append(out, v)
	}

	_, err := dec.Token() // ]

	return out, err
}

func decodeObject(dec *json.Decoder) (any, error) {
	o := &object{vals: map[string]any{}}

	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}

		key, ok := tok.(string)
		if !ok {
			return nil, fmt.Errorf("object key %v", tok)
		}

		v, err := decodeValue(dec)
		if err != nil {
			return nil, err
		}

		if _, seen := o.vals[key]; !seen {
			o.keys = append(o.keys, key)
		}

		o.vals[key] = v
	}

	_, err := dec.Token() // }

	return o, err
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
