package needle

import (
	"bytes"
	"encoding/json"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

// This file ports upstream's needle/environments/_harness.py (af2654dffa):
// how run_tests scores an answer against a suite's case, so the Go agent's
// answers are judged as upstream judges the engine's (docs/FINDINGS.md
// F-065).

// casefold is Python's str.casefold for the code points the suites and
// their answers can hold: per rune the simple folding, which for all but
// Cherokee is unicode.ToLower(unicode.ToUpper(r)) (ς and Σ are σ, the
// Kelvin sign k, ſ s, µ μ), Cherokee folding to its capitals, and the full
// foldings of ß and ẞ (ss), İ (i and a combining dot above) and the Latin
// ligatures ﬀ to ﬆ. Not ported, since golang.org/x/text is not a
// dependency: the other full foldings of CaseFolding.txt (ŉ, ǰ, ẖ to ẚ,
// the Greek letters with a diacritic or a iota subscript and the Armenian
// ligatures), which Python expands into several runes. The suites' calls
// and the engine's answers to them hold ASCII and á only
// (TestEnvironmentCasefold pins the ported cases against Python 3.14).
func casefold(s string) string {
	var b strings.Builder

	for _, r := range s {
		switch {
		case r == 'ß' || r == 'ẞ':
			b.WriteString("ss")
		case r == 'İ':
			b.WriteString("i̇")
		case r >= 'ﬀ' && r <= 'ﬆ':
			b.WriteString(ligatures[r-'ﬀ'])
		case unicode.Is(unicode.Cherokee, r):
			b.WriteRune(unicode.ToUpper(r))
		default:
			b.WriteRune(unicode.ToLower(unicode.ToUpper(r)))
		}
	}

	return b.String()
}

// ligatures are the full foldings of U+FB00 to U+FB06.
var ligatures = [...]string{"ff", "fi", "fl", "ffi", "ffl", "st", "st"}

// harnessFold is _fold over a value json.Decoder decodes with UseNumber:
// strings casefolded, an integral float made an int, objects' values and
// lists' elements folded (keys stay). A number literal without a fraction
// or exponent is a Python int and stays an int; any other is a float,
// written as the int it equals when integral, else as the float64's
// shortest form. Python's json.dumps writes the same float64 as one text
// and different ones as different texts, so the folded forms compare as
// its dumps do.
func harnessFold(v any) any {
	switch x := v.(type) {
	case string:
		return casefold(x)
	case json.Number:
		return foldNumber(x)
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = harnessFold(e)
		}

		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = harnessFold(e)
		}

		return out
	}

	return v
}

func foldNumber(n json.Number) json.Number {
	s := string(n)
	if !strings.ContainsAny(s, ".eE") {
		i, err := strconv.ParseInt(s, 10, 64)
		if err == nil {
			return json.Number(strconv.FormatInt(i, 10)) // "-0" is 0
		}

		return n // a big int: Python keeps every digit
	}

	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsInf(f, 0) {
		return n
	}

	if f == math.Trunc(f) {
		return json.Number(strconv.FormatFloat(f+0, 'f', 0, 64)) // int(-0.0) is 0
	}

	return json.Number(strconv.FormatFloat(f, 'g', -1, 64))
}

// harnessKey is _key, json.dumps(_fold(call), sort_keys=True), as a form
// that compares as it does: encoding/json sorts the keys of a map.
func harnessKey(call json.RawMessage) string {
	d := json.NewDecoder(bytes.NewReader(call))
	d.UseNumber()

	var v any

	err := d.Decode(&v)
	if err != nil {
		return "invalid: " + string(call)
	}

	js, err := json.Marshal(harnessFold(v))
	if err != nil {
		return "invalid: " + string(call)
	}

	return string(js)
}

// harnessCase is a case of a suite as the harness reads it: the calls it
// wants, whether it is critical, and the answer.
type harnessCase struct {
	want     []json.RawMessage
	critical bool
	answer   harnessAnswer
}

// harnessAnswer is what run_tests reads of a response.
type harnessAnswer struct {
	Calls      []json.RawMessage `json:"function_calls"`
	Confidence *float64          `json:"confidence"`
	Validation *struct {
		Ungrounded []string `json:"ungrounded"`
		Negation   bool     `json:"negation"`
	} `json:"validation"`
}

// harnessOutcome is what run_tests prints and returns.
type harnessOutcome struct {
	Passed, Total, Critical int
	Pass                    bool
	Failed                  []int // the failing cases' indices
}

// runHarness is run_tests at the gate minConfidence over the cases'
// answers: an answer's calls count as none when the validation flags a
// field as ungrounded or the request as negated, or when the confidence
// is below the gate (a missing confidence is 0.0); a case passes when the
// sorted keys of the calls equal those it wants; the suite passes when
// round(0.9·n) cases pass (Python's round, half to even) and no critical
// case fails.
func runHarness(cases []harnessCase, minConfidence float64) harnessOutcome {
	o := harnessOutcome{Total: len(cases)}

	for i, c := range cases {
		got := c.answer.Calls

		if v := c.answer.Validation; len(got) > 0 && v != nil && (len(v.Ungrounded) > 0 || v.Negation) {
			got = nil
		}

		conf := 0.0
		if c.answer.Confidence != nil {
			conf = *c.answer.Confidence
		}

		if len(got) > 0 && conf < minConfidence {
			got = nil
		}

		if slices.Equal(sortedKeys(got), sortedKeys(c.want)) {
			o.Passed++

			continue
		}

		o.Failed = append(o.Failed, i)

		if c.critical {
			o.Critical++
		}
	}

	o.Pass = float64(o.Passed) >= math.RoundToEven(0.9*float64(o.Total)) && o.Critical == 0

	return o
}

func sortedKeys(calls []json.RawMessage) []string {
	keys := make([]string, len(calls))
	for i, c := range calls {
		keys[i] = harnessKey(c)
	}

	slices.Sort(keys)

	return keys
}
