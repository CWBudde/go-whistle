package needle

import (
	"encoding/json"
	"math"
	"math/big"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// The Python client's grounding checks on top of the engine's envelope
// (upstream needle/__init__.py: _source_years, _relative_cue,
// _licensed_years, _walk_grounding, _numeric_paths, _source_numbers,
// _grounded_number_paths; docs/FINDINGS.md F-064). Python's re reads \d, \s
// and \w as Unicode classes and has look-arounds, which RE2 does not: the
// patterns below match the parts without them, and the code around them
// checks the look-arounds and the word boundaries as Python does.

// pyWord is Python's \w for one rune: str.isalnum or '_'.
func pyWord(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r) }

// pySpace is Python's \s for one rune (str.isspace): Unicode white space
// and the separators U+001C–U+001F.
func pySpace(r rune) bool { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }

// asciiAlnum is the class [0-9A-Za-z] of the look-aheads.
func asciiAlnum(r rune) bool {
	return r >= '0' && r <= '9' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z'
}

// jsonNull is JSON's null as text.
const jsonNull = "null"

// isDigit is Python's \d for one rune: a decimal digit of any script.
func isDigit(r rune) bool { return unicode.Is(unicode.Nd, r) }

// boundary reports Python's \b at byte offset i of s.
func boundary(s string, i int) bool {
	before, after := false, false

	if i > 0 {
		r, _ := utf8.DecodeLastRuneInString(s[:i])
		before = pyWord(r)
	}

	if i < len(s) {
		r, _ := utf8.DecodeRuneInString(s[i:])
		after = pyWord(r)
	}

	return before != after
}

// runeAt is the rune at byte offset i of s and its length; -1 past the end.
func runeAt(s string, i int) (rune, int) {
	if i >= len(s) {
		return -1, 0
	}

	return utf8.DecodeRuneInString(s[i:])
}

const (
	// pyS is Python's \s as the body of an RE2 class (pySpace).
	pyS = `\t\n\v\f\r \x{1c}-\x{1f}\x{85}\p{Z}`
	// months is _source_years' month names.
	months = `(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|` +
		`sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)`
)

// yearPattern is one of _source_years' patterns from after its leading \b
// to its year group: the group's shortest length and whether the
// look-ahead after it also rejects the hour of a time.
type yearPattern struct {
	re       *regexp.Regexp
	min      int
	timeAway bool
}

var yearPatterns = []yearPattern{
	// \b\d{1,2}(?:st|nd|rd|th)?\s+{months}[\s,]+(\d{1,4})(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)
	{regexp.MustCompile(`^\p{Nd}{1,2}(?:st|nd|rd|th)?[` + pyS + `]+` + months + `[` + pyS + `,]+(\p{Nd}{1,4})`), 1, true},
	// \b{months}\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s+(\d{1,4})(?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b)
	{
		regexp.MustCompile(`^` + months + `[` + pyS + `]+\p{Nd}{1,2}(?:st|nd|rd|th)?[` + pyS + `]*,?[` + pyS +
			`]+(\p{Nd}{1,4})`), 1, true,
	},
	// \b{months}[\s,]+(\d{3,4})(?![0-9A-Za-z])
	{regexp.MustCompile(`^` + months + `[` + pyS + `,]+(\p{Nd}{3,4})`), 3, false},
	// \byear\s+(\d{1,4})(?![0-9A-Za-z])
	{regexp.MustCompile(`^year[` + pyS + `]+(\p{Nd}{1,4})`), 1, false},
}

// sourceYears is _source_years: the years text writes after a day and a
// month, after a month alone (three or four digits), after "year", and at
// the start of a year-first numeric date (2024-03-15, 2024/3/15).
func sourceYears(text string) map[int]bool {
	s := pyLower(text)
	years := map[int]bool{}

	for _, p := range yearPatterns {
		for i := 0; i < len(s); {
			if y, end, ok := p.matchAt(s, i); ok {
				years[y] = true
				i = end

				continue
			}

			_, n := utf8.DecodeRuneInString(s[i:])
			i += n
		}
	}

	numericDateYears(s, years)

	return years
}

// pyLower is Python's str.lower: strings.ToLower, but 'İ' (U+0130)
// becomes "i̇", 'i' and the combining dot U+0307, as in Unicode's full
// case mapping.
func pyLower(s string) string { return strings.ToLower(strings.ReplaceAll(s, "İ", "i\u0307")) }

// matchAt matches p at byte offset i of s as Python's re does, giving the
// year's digits back one by one until the look-ahead holds: the year and
// the match's end.
func (p yearPattern) matchAt(s string, i int) (year, end int, ok bool) {
	if !boundary(s, i) {
		return 0, 0, false
	}

	m := p.re.FindStringSubmatchIndex(s[i:])
	if m == nil {
		return 0, 0, false
	}

	digits := []rune(s[i+m[2] : i+m[3]])

	for n := len(digits); n >= p.min; n-- {
		end := i + m[2] + len(string(digits[:n]))
		if p.ahead(s, end) {
			return digitsValue(digits[:n]), end, true
		}
	}

	return 0, 0, false
}

// ahead is the negative look-ahead after a year that ends at byte offset
// i: (?![0-9A-Za-z]), and with timeAway (?![0-9A-Za-z]|:\d|\s*[ap]\.?m\b).
func (p yearPattern) ahead(s string, i int) bool {
	r, n := runeAt(s, i)
	if asciiAlnum(r) {
		return false
	}

	if !p.timeAway {
		return true
	}

	if d, _ := runeAt(s, i+n); r == ':' && isDigit(d) {
		return false
	}

	j := i
	for r, n := runeAt(s, j); pySpace(r); r, n = runeAt(s, j) {
		j += n
	}

	if j >= len(s) || s[j] != 'a' && s[j] != 'p' {
		return true
	}

	j++
	if j+1 < len(s) && s[j] == '.' && s[j+1] == 'm' {
		j++
	}

	return j >= len(s) || s[j] != 'm' || !boundary(s, j+1)
}

// numericDateYears adds the years that _source_years' last pattern,
// (?<![0-9])(\d{4})(?=[-/]\d{1,2}[-/]\d{1,2}(?![0-9])), finds in s.
func numericDateYears(s string, years map[int]bool) {
	rs := []rune(s)

	for i := 0; i+4 <= len(rs); i++ {
		if i > 0 && asciiDigitAt(rs, i-1) || !digitsAt(rs, i, 4) || !dateAhead(rs, i+4) {
			continue
		}

		years[digitsValue(rs[i:i+4])] = true
		i += 3
	}
}

// dateAhead is the look-ahead after a numeric date's year at i,
// [-/]\d{1,2}[-/]\d{1,2}(?![0-9]), its digits tried two before one.
func dateAhead(rs []rune, i int) bool {
	sep := func(i int) bool { return i < len(rs) && (rs[i] == '-' || rs[i] == '/') }

	// part is \d{1,2} at i followed by what next accepts.
	part := func(i int, next func(int) bool) bool {
		return digitsAt(rs, i, 1) && (digitsAt(rs, i+1, 1) && next(i+2) || next(i+1))
	}

	return sep(i) && part(i+1, func(j int) bool {
		return sep(j) && part(j+1, func(k int) bool { return !asciiDigitAt(rs, k) })
	})
}

// digitsAt reports n decimal digits of any script at i.
func digitsAt(rs []rune, i, n int) bool {
	if i+n > len(rs) {
		return false
	}

	for _, r := range rs[i : i+n] {
		if !isDigit(r) {
			return false
		}
	}

	return true
}

// asciiDigitAt reports [0-9] at i.
func asciiDigitAt(rs []rune, i int) bool { return i < len(rs) && rs[i] >= '0' && rs[i] <= '9' }

// digitsValue is Python's int() of decimal digits of any script.
func digitsValue(ds []rune) int {
	v := 0
	for _, d := range ds {
		v = 10*v + digitValue(d)
	}

	return v
}

// digitValue is the value of a decimal digit (category Nd) of any script:
// Unicode lays out each script's digits as runs of ten from a zero.
func digitValue(r rune) int {
	if r >= '0' && r <= '9' {
		return int(r - '0')
	}

	for _, rg := range unicode.Nd.R16 {
		if r >= rune(rg.Lo) && r <= rune(rg.Hi) {
			return int(r-rune(rg.Lo)) % 10
		}
	}

	for _, rg := range unicode.Nd.R32 {
		if r >= rune(rg.Lo) && r <= rune(rg.Hi) {
			return int(r-rune(rg.Lo)) % 10
		}
	}

	return 0
}

// relativeCues are _relative_cue's alternatives, every spelling on its
// own, so that a failing word boundary after one moves on to the next as
// Python's backtracking does.
var relativeCues = func() []*regexp.Regexp {
	words := []string{
		"today", "tonight", "tomorrow", "yesterday", "next", "this", "coming", "now",
		"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
		`in \p{Nd}+ days`, `in \p{Nd}+ day`, `in \p{Nd}+ weeks`, `in \p{Nd}+ week`,
		`in \p{Nd}+ months`, `in \p{Nd}+ month`, `in \p{Nd}+ years`, `in \p{Nd}+ year`,
	}

	// Python's IGNORECASE also takes 'ı' (U+0131) and 'İ' (U+0130) for
	// 'i', which RE2's case folding does not.
	res := make([]*regexp.Regexp, len(words))
	for i, w := range words {
		res[i] = regexp.MustCompile(`^(?i:` + strings.ReplaceAll(w, "i", "[iıİ]") + `)`)
	}

	return res
}()

// relativeCue is _relative_cue: whether text reasons relatively (today,
// tomorrow, next, this, in 3 days, a weekday, …), as whole words in any
// case.
func relativeCue(text string) bool {
	for i := 0; i < len(text); {
		if boundary(text, i) {
			for _, re := range relativeCues {
				if m := re.FindStringIndex(text[i:]); m != nil && boundary(text, i+m[1]) {
					return true
				}
			}
		}

		_, n := utf8.DecodeRuneInString(text[i:])
		i += n
	}

	return false
}

// licensedYears is _licensed_years: the years a date argument may carry,
// those seen in the input and, when there are any and the input reasons
// relatively, those of the system facts.
func licensedYears(seen map[int]bool, system string, relative bool) map[int]bool {
	years := make(map[int]bool, len(seen))
	for y := range seen {
		years[y] = true
	}

	if len(years) > 0 && system != "" && relative {
		for y := range sourceYears(system) {
			years[y] = true
		}
	}

	return years
}

// isoYear is _walk_grounding's ^(\d{4})-.
var isoYear = regexp.MustCompile(`^(\p{Nd}{4})-`)

// walkGrounding is _walk_grounding: the paths of the date and date-time
// values of arguments whose year the walk of schema checked against years,
// and those of them whose year is not among years. schema is a tool (its
// "parameters", or the schema itself without them) and arguments its call's
// arguments, both as decodeJSON gives them.
func walkGrounding(schema, arguments any, years map[int]bool) (checked, failures map[string]bool) {
	root := schemaParameters(schema)
	checked, failures = map[string]bool{}, map[string]bool{}

	var walk func(value, node any, path string)

	walk = func(value, node any, path string) {
		obj := concrete(asObject(resolveRef(node, root)), root)

		if f := obj["format"]; f == "date" || f == "date-time" {
			if s, ok := value.(string); ok {
				checkYear(s, path, years, checked, failures)

				return
			}
		}

		switch v := value.(type) {
		case map[string]any:
			props := asObject(obj["properties"])

			for key, item := range v {
				if p, ok := props[key]; ok {
					walk(item, p, joinKey(path, key))
				}
			}
		case []any:
			if items, ok := obj["items"]; ok {
				for i, item := range v {
					walk(item, items, path+"["+strconv.Itoa(i)+"]")
				}
			}
		}
	}

	walk(arguments, root, "")

	return checked, failures
}

// concrete is the one branch of obj's anyOf (or else oneOf) that is not
// the null type, when there is exactly one, else obj.
func concrete(obj map[string]any, root any) map[string]any {
	variants, _ := obj["anyOf"].([]any)
	if len(variants) == 0 {
		variants, _ = obj["oneOf"].([]any)
	}

	var branches []any

	for _, v := range variants {
		if asObject(resolveRef(v, root))["type"] != jsonNull {
			branches = append(branches, v)
		}
	}

	if len(branches) == 1 {
		return asObject(resolveRef(branches[0], root))
	}

	return obj
}

// checkYear checks the year of the date or date-time s at path against
// years, when s starts with one and there are years: path is then checked,
// and a failure when its year is not among them.
func checkYear(s, path string, years map[int]bool, checked, failures map[string]bool) {
	m := isoYear.FindStringSubmatch(s)
	if m == nil || len(years) == 0 {
		return
	}

	checked[path] = true

	if !years[digitsValue([]rune(m[1]))] {
		failures[path] = true
	}
}

// schemaParameters is _schema_parameters for a schema given as JSON: its
// "parameters" when it has them, else itself.
func schemaParameters(schema any) any {
	obj, ok := schema.(map[string]any)
	if !ok {
		return map[string]any{}
	}

	if p, ok := obj["parameters"]; ok {
		return p
	}

	return obj
}

// resolveRef is _resolve_ref: node with its "$ref" followed through root
// ("#/a/b", ~1 and ~0 unescaped) until a reference repeats or points at
// the node itself; a missing target is the empty schema.
func resolveRef(node, root any) any {
	seen := map[string]bool{}

	for {
		obj, ok := node.(map[string]any)
		if !ok {
			return node
		}

		ref, ok := obj["$ref"].(string)
		if !ok || seen[ref] {
			return node
		}

		seen[ref] = true

		target := root

		for part := range strings.SplitSeq(strings.TrimPrefix(ref, "#/"), "/") {
			part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")

			t, ok := asObject(target)[part]
			if !ok {
				t = map[string]any{}
			}

			target = t
		}

		if t, ok := target.(map[string]any); ok && reflect.ValueOf(t).UnsafePointer() == reflect.ValueOf(obj).UnsafePointer() {
			return node
		}

		node = target
	}
}

// asObject is v as a JSON object; anything else is the empty one.
func asObject(v any) map[string]any {
	obj, _ := v.(map[string]any)

	return obj
}

// joinKey is the path of key below path: "path.key", or "key" at the top.
func joinKey(path, key string) string {
	if path == "" {
		return key
	}

	return path + "." + key
}

// dateFact is _DATE_FACT: a date fact as WithDateFact writes it, and an ISO
// date-time.
var dateFact = regexp.MustCompile(`date:[` + pyS + `]*\p{Nd}{4}-\p{Nd}{2}-\p{Nd}{2}(?:[` + pyS +
	`]+[A-Za-z]{3})?(?:[` + pyS + `]+\p{Nd}{2}:\p{Nd}{2})?|` +
	`\p{Nd}{4}-\p{Nd}{2}-\p{Nd}{2}T\p{Nd}{2}:\p{Nd}{2}(?::\p{Nd}{2})?`)

// sourceNumbers is _source_numbers: the numbers the sources write outside
// their date facts, "1,234" as 1234, as the exact values big.Rat.RatString
// writes.
func sourceNumbers(sources ...string) map[string]bool {
	var kept []string

	for _, s := range sources {
		if s = dateFact.ReplaceAllString(s, " "); s != "" {
			kept = append(kept, s)
		}
	}

	numbers := map[string]bool{}

	for _, tok := range numberTokens(strings.Join(kept, "\n")) {
		lit := strings.TrimPrefix(asciiDigits(strings.ReplaceAll(tok, ",", "")), "+")
		if r, ok := new(big.Rat).SetString(lit); ok {
			numbers[r.RatString()] = true
		}
	}

	return numbers
}

// asciiDigits writes the decimal digits of any script in s as ASCII ones.
func asciiDigits(s string) string {
	return strings.Map(func(r rune) rune {
		if r > '9' && isDigit(r) {
			return '0' + rune(digitValue(r))
		}

		return r
	}, s)
}

// numberTokens are the matches of _NUMBER_TOKEN in s as Python's finditer
// finds them:
// (?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d).
func numberTokens(s string) []string {
	rs := []rune(s)

	var toks []string

	for i := 0; i < len(rs); {
		if i > 0 && (pyWord(rs[i-1]) || rs[i-1] == '.' || rs[i-1] == ',') {
			i++

			continue
		}

		end := numberEnd(rs, i)
		if end < 0 {
			i++

			continue
		}

		toks = append(toks, string(rs[i:end]))
		i = end
	}

	return toks
}

// numberEnd is the end of _NUMBER_TOKEN's alternatives at i, -1 when
// neither matches: [-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d) with as many
// groups as there are, given back one by one until no digit follows, then
// [-+]?\d+(?:\.\d+)?, which no digit can follow.
func numberEnd(rs []rune, i int) int {
	j := i
	if j < len(rs) && (rs[j] == '-' || rs[j] == '+') {
		j++
	}

	lead := digitRun(rs, j)
	if lead == j {
		return -1
	}

	var groups []int

	for g := lead; lead-j <= 3 && g < len(rs) && rs[g] == ',' && digitsAt(rs, g+1, 3); g += 4 {
		groups = append(groups, g+4)
	}

	for _, g := range slices.Backward(groups) {
		if e := fractionEnd(rs, g); !digitsAt(rs, e, 1) {
			return e
		}
	}

	return fractionEnd(rs, lead)
}

// digitRun is the end of the decimal digits from i.
func digitRun(rs []rune, i int) int {
	for digitsAt(rs, i, 1) {
		i++
	}

	return i
}

// fractionEnd is the end of an optional \.\d+ at i.
func fractionEnd(rs []rune, i int) int {
	if i < len(rs) && rs[i] == '.' && digitsAt(rs, i+1, 1) {
		return digitRun(rs, i+1)
	}

	return i
}

// numericPaths is _numeric_paths: the path of every number of arguments
// (a boolean is none) with its value as Python's decimal.Decimal(str(v))
// has it, written by big.Rat.RatString; a number past the float range,
// which Decimal holds as infinite, is left out.
func numericPaths(arguments any, path string, yield func(path, value string)) {
	switch v := arguments.(type) {
	case json.Number:
		if r := pyDecimal(string(v)); r != nil {
			yield(path, r.RatString())
		}
	case map[string]any:
		for key, item := range v {
			numericPaths(item, joinKey(path, key), yield)
		}
	case []any:
		for i, item := range v {
			numericPaths(item, path+"["+strconv.Itoa(i)+"]", yield)
		}
	}
}

// pyDecimal is the value of decimal.Decimal(str(json.loads(lit))): an
// integer literal exactly, any other one as the float json.loads makes of
// it; nil past the float range.
func pyDecimal(lit string) *big.Rat {
	if !strings.ContainsAny(lit, ".eE") {
		r, ok := new(big.Rat).SetString(lit)
		if !ok {
			return nil
		}

		return r
	}

	f, err := strconv.ParseFloat(lit, 64)
	if err != nil || math.IsInf(f, 0) {
		return nil
	}

	r, _ := new(big.Rat).SetString(strconv.FormatFloat(f, 'g', -1, 64))

	return r
}

// groundedNumberPaths is _grounded_number_paths: the paths of the numbers
// of arguments that one of the sources writes.
func groundedNumberPaths(arguments any, sources ...string) map[string]bool {
	grounded := map[string]bool{}

	numbers := sourceNumbers(sources...)
	if len(numbers) == 0 {
		return grounded
	}

	numericPaths(arguments, "", func(path, value string) {
		if numbers[value] {
			grounded[path] = true
		}
	})

	return grounded
}

// decodeJSON is raw as JSON values with json.Number numbers; nil when raw
// is not JSON.
func decodeJSON(raw []byte) any {
	var v any

	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()

	if dec.Decode(&v) != nil {
		return nil
	}

	return v
}
