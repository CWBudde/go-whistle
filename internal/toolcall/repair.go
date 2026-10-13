package toolcall

import (
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
)

// The engine's deterministic repair of the decoded calls (llms.txt:
// "Arguments arrive after the engine's deterministic repair step, which
// grounds them in the request text"), as far as the repair probes pin it
// (docs/FINDINGS.md F-056). Every word list holds only what llms.txt names,
// what a recorded probe shows the engine to act on, or plain English
// (weekdays, months, cardinal numbers); nothing is taken from the engine's
// binaries (AGENTS.md).

// repairCall is one call under repair: the model's tool name and its
// arguments, parsed; args is nil when they are no object, and such a call
// passes through as it is.
type repairCall struct {
	name    string
	args    *object
	raw     []byte
	changed bool
}

// repairer holds what the rules read: the tools, the request and the date
// fact.
type repairer struct {
	n       *Normalised
	input   string
	lower   string
	fact    time.Time
	hasFact bool
}

// repair is the engine's repair of the calls the model wrote for input,
// with the system facts system, in the order the probes show
// (docs/FINDINGS.md F-056): the values grounded in the request first
// (phone numbers, quoted titles, names, places, pickup and destination,
// "by N", a defaulted number), then dates, the paired tool, polarity and
// named enum options, capitalisation, and last the dropped arguments and
// calls.
func (n *Normalised) repair(calls []modelCall, input, system string) []modelCall {
	r := &repairer{n: n, input: input, lower: strings.ToLower(input)}
	r.fact, r.hasFact = dateFactOf(system)

	cs := make([]*repairCall, len(calls))

	for i, c := range calls {
		cs[i] = &repairCall{name: c.name, raw: c.args}

		v, err := (&pyParser{s: c.args}).value()
		if o, ok := v.(*object); ok && err == nil {
			cs[i].args = o
		}
	}

	rules := []func(*repairCall){
		r.phone, r.quoted, r.nameSplit, r.place, r.fromTo, r.byN, r.defaultNumber, r.date, r.paired, r.polarity,
		r.enumNamed, r.capitalise, r.dropArguments,
	}

	for _, rule := range rules {
		for _, c := range cs {
			if c.args != nil {
				rule(c)
			}
		}
	}

	cs = r.dropCalls(cs)

	out := make([]modelCall, len(cs))
	for i, c := range cs {
		out[i] = modelCall{name: c.name, args: c.bytes()}
	}

	return out
}

// bytes are c's arguments as JSON: as the model wrote them unless the
// repair changed them.
func (c *repairCall) bytes() []byte {
	if !c.changed {
		return c.raw
	}

	var b strings.Builder

	writeValue(&b, c.args)

	return []byte(b.String())
}

// set sets the argument key of c to v.
func (c *repairCall) set(key string, v any) {
	c.args = c.args.with(key, v)
	c.changed = true
}

// drop removes the argument key of c.
func (c *repairCall) drop(key string) {
	c.args = c.args.without(key)
	c.changed = true
}

// stringArgs are c's top-level string arguments, in order, as key and
// value.
func (c *repairCall) stringArgs(yield func(key, v string) bool) {
	for _, k := range c.args.keys {
		if s, ok := c.args.vals[k].(string); ok && !yield(k, s) {
			return
		}
	}
}

// property is the schema of the tool's top-level property key, nil when the
// tool does not declare it: from the parameters' own properties, else from
// the first allOf, anyOf or oneOf branch that declares it (PR #66 review).
func (r *repairer) property(tool, key string) *object {
	p := r.n.params[tool]
	if p == nil {
		return nil
	}

	for _, s := range composed(p) {
		if m := member(member(s, "properties"), key); m != nil {
			return m
		}
	}

	return nil
}

// required reports whether key is a required argument of tool: required by
// the parameters or by one of their allOf, anyOf or oneOf branches, which
// the grammar may have held the call to (PR #66 review).
func (r *repairer) required(tool, key string) bool {
	p := r.n.params[tool]
	if p == nil {
		return false
	}

	for _, s := range composed(p) {
		if req, _ := s.vals["required"].([]any); slices.Contains(req, any(key)) {
			return true
		}
	}

	return false
}

// properties are the names of the tool's top-level properties, those of
// the parameters first, then those only its branches declare.
func (r *repairer) properties(tool string) []string {
	p := r.n.params[tool]
	if p == nil {
		return nil
	}

	var out []string

	for _, s := range composed(p) {
		if ps := member(s, "properties"); ps != nil {
			for _, k := range ps.keys {
				if !slices.Contains(out, k) {
					out = append(out, k)
				}
			}
		}
	}

	return out
}

// typeOf is a schema's "type" when it is one name.
func typeOf(s *object) string {
	if s == nil {
		return ""
	}

	t, _ := s.vals["type"].(string)

	return t
}

// typeInteger is JSON Schema's integer type.
const typeInteger = "integer"

// isNumberType reports whether a schema is of type number or integer.
func isNumberType(s *object) bool {
	t := typeOf(s)

	return t == "number" || t == typeInteger
}

// phoneRun is a phone number as written: digits, spaces, parentheses,
// dots and dashes, with an optional leading "+" or "(".
var phoneRun = regexp.MustCompile(`[+(]?\d[\d ().-]*\d`)

// minPhoneDigits is the fewest digits a run needs to be a phone number (a
// port-side bound).
const minPhoneDigits = 7

// phone completes a truncated phone number: an argument whose key holds
// "phone" and whose value is part of a longer phone number in the request
// takes the whole number (docs/FINDINGS.md F-056: phone_truncated, and
// phone_split, whose two parts then are one call twice).
func (r *repairer) phone(c *repairCall) {
	var runs []string

	for _, m := range phoneRun.FindAllString(r.input, -1) {
		if digits(m) >= minPhoneDigits {
			runs = append(runs, m)
		}
	}

	for k, v := range c.stringArgs {
		if !strings.Contains(k, "phone") || digits(v) == 0 {
			continue
		}

		for _, run := range runs {
			if run != v && strings.Contains(run, strings.TrimSpace(v)) {
				c.set(k, run)

				break
			}
		}
	}
}

// digits counts the decimal digits of s.
func digits(s string) int {
	n := 0

	for i := range len(s) {
		if isDigit(s[i]) {
			n++
		}
	}

	return n
}

// quotedSpan is a span of the request in double quotes, typographic double
// quotes, or single quotes that open after white space (not an
// apostrophe).
var quotedSpan = regexp.MustCompile(`"([^"]+)"|“([^”]+)”|(?:^|\s)'([^']+)'(?:[\s.,!?;:]|$)`)

// quoted restores a quoted title verbatim: a string argument that is the
// start of a quoted span of the request, ending at a word, takes the whole
// span (docs/FINDINGS.md F-056: quoted_truncated; which arguments it
// reaches is port-side).
func (r *repairer) quoted(c *repairCall) {
	matches := quotedSpan.FindAllStringSubmatch(r.input, -1)
	spans := make([]string, 0, len(matches))

	for _, m := range matches {
		spans = append(spans, m[1]+m[2]+m[3])
	}

	for k, v := range c.stringArgs {
		for _, q := range spans {
			// The offsets are the folded strings': folding may change the
			// length in bytes (PR #66 review).
			lv, lq := strings.ToLower(v), strings.ToLower(q)
			if v == "" || v == q || !strings.HasPrefix(lq, lv) {
				continue
			}

			if len(lv) == len(lq) || lq[len(lv)] == ' ' {
				c.set(k, q)

				break
			}
		}
	}
}

// The first and last keys of a name pair the probes show split
// (docs/FINDINGS.md F-056: name_given, name_first, name_first_name).
var (
	firstKeys = []string{"first_name", "first", "given_name"}
	lastKeys  = []string{"last_name", "last", "family_name"}
)

// nameSplit splits a full name in the first of a first/last pair: the
// first key keeps the first word; the last key is not filled
// (docs/FINDINGS.md F-056: name_given, name_first, name_first_name).
func (r *repairer) nameSplit(c *repairCall) {
	if !slices.ContainsFunc(r.properties(c.name), func(k string) bool { return slices.Contains(lastKeys, k) }) {
		return
	}

	for k, v := range c.stringArgs {
		if !slices.Contains(firstKeys, k) {
			continue
		}

		if f := strings.Fields(v); len(f) > 1 {
			c.set(k, f[0])
		}
	}
}

// placeLinks join a place to its street address in the request, as the
// probes show (docs/FINDINGS.md F-056: place_located_at, place_which_is_at;
// " which is located at " and a bare " at " leave the place alone).
var placeLinks = []string{" located at ", " which is at "}

// place recovers a following street address: a string argument that the
// request writes right before a place link and an address starting with a
// house number takes the address after a space (docs/FINDINGS.md F-056:
// place_located_at, place_which_is_at).
func (r *repairer) place(c *repairCall) {
	for k, v := range c.stringArgs {
		if v == "" {
			continue
		}

		for _, link := range placeLinks {
			i := strings.Index(r.lower, strings.ToLower(v)+link)
			if i < 0 {
				continue
			}

			rest := r.input[i+len(v)+len(link):]
			addr := strings.TrimSpace(rest[:clauseEnd(rest)])

			if addr != "" && isDigit(addr[0]) {
				c.set(k, v+" "+addr)
			}

			break
		}
	}
}

// The keys of a pickup and of a destination slot: llms.txt's "pickup and
// destination", and the probes' origin (docs/FINDINGS.md F-056:
// from_to_missing, from_to_article).
var (
	pickupKeys      = []string{"pickup", "origin"}
	destinationKeys = []string{"destination"}
)

// fromToWords is "from X to Y" in the request.
var fromToWords = regexp.MustCompile(`(?i)\bfrom\s+(.+?)\s+to\s+(.+)`)

// fromTo reads pickup and destination from "from X to Y": a pickup or a
// destination slot the call leaves out takes X or Y, without a leading
// "the" (docs/FINDINGS.md F-056: from_to_missing, from_to_article; a slot
// the model filled stays, from_to_given).
func (r *repairer) fromTo(c *repairCall) {
	m := fromToWords.FindStringSubmatch(r.input)
	if m == nil || clauseEnd(m[1]) < len(m[1]) {
		return
	}

	from, to := withoutThe(m[1]), withoutThe(m[2][:clauseEnd(m[2])])

	for _, k := range r.properties(c.name) {
		if _, ok := c.args.vals[k]; ok || typeOf(r.property(c.name, k)) != "string" {
			continue
		}

		switch {
		case slices.Contains(pickupKeys, k) && from != "":
			c.set(k, from)
		case slices.Contains(destinationKeys, k) && to != "":
			c.set(k, to)
		}
	}
}

// withoutThe is s trimmed, without a leading "the ".
func withoutThe(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 4 && strings.EqualFold(s[:4], "the ") {
		return s[4:]
	}

	return s
}

// byNumber is "by N" in the request.
var byNumber = regexp.MustCompile(`(?i)\bby\s+(\d+(?:\.\d+)?)\b`)

// byN fills a tool's only number argument from "by N" in the request
// (docs/FINDINGS.md F-056: by_n).
func (r *repairer) byN(c *repairCall) {
	m := byNumber.FindStringSubmatch(r.input)
	if m == nil {
		return
	}

	var key string

	for _, k := range r.properties(c.name) {
		if isNumberType(r.property(c.name, k)) {
			if key != "" {
				return
			}

			key = k
		}
	}

	if key == "" || typeOf(r.property(c.name, key)) == typeInteger && strings.Contains(m[1], ".") {
		return
	}

	if v, ok := c.args.vals[key].(json.Number); !ok || !sameNumber(string(v), m[1]) {
		c.set(key, json.Number(m[1]))
	}
}

// sameNumber reports whether two number literals have the same value.
func sameNumber(a, b string) bool {
	x, errX := strconv.ParseFloat(a, 64)
	y, errY := strconv.ParseFloat(b, 64)

	return errX == nil && errY == nil && x == y
}

// defaultNumber gives a required number with a default its default when
// the request names no quantity (llms.txt; the model already writes the
// default on every probe, docs/FINDINGS.md F-056: default_number).
func (r *repairer) defaultNumber(c *repairCall) {
	if namesQuantity(r.lower) {
		return
	}

	for _, k := range r.properties(c.name) {
		s := r.property(c.name, k)
		if s == nil { // a boolean schema has no default (PR #66 review)
			continue
		}

		d, ok := s.vals["default"].(json.Number)
		if !ok || !isNumberType(s) || !r.required(c.name, k) {
			continue
		}

		if v, ok := c.args.vals[k].(json.Number); !ok || !sameNumber(string(v), string(d)) {
			c.set(k, d)
		}
	}
}

// namesQuantity reports whether the request lowered has a digit or a number
// word.
func namesQuantity(lowered string) bool {
	if strings.ContainsFunc(lowered, func(r rune) bool { return r >= '0' && r <= '9' }) {
		return true
	}

	return slices.ContainsFunc(words(lowered), func(w string) bool {
		_, ok := numberWords[w]

		return ok
	})
}

// numberWords are the English cardinal numbers with their values; "half"
// also stands for 50 (of a percentage), a reading the probes pin
// (docs/FINDINGS.md F-056: optional_number_half).
var numberWords = map[string][]float64{
	"zero": {0}, "one": {1}, "two": {2}, "three": {3}, "four": {4}, "five": {5}, "six": {6}, "seven": {7},
	"eight": {8}, "nine": {9}, "ten": {10}, "eleven": {11}, "twelve": {12}, "thirteen": {13}, "fourteen": {14},
	"fifteen": {15}, "sixteen": {16}, "seventeen": {17}, "eighteen": {18}, "nineteen": {19}, "twenty": {20},
	"thirty": {30}, "forty": {40}, "fifty": {50}, "sixty": {60}, "seventy": {70}, "eighty": {80}, "ninety": {90},
	"hundred": {100}, "thousand": {1000}, "million": {1e6}, "billion": {1e9}, "half": {0.5, 50},
}

// wordNumber reports whether the request lowered states v in words: a
// number word, a tens word and a unit ("twenty five", "twenty-five"), or a
// number before a scale word ("one hundred", "a hundred").
func wordNumber(lowered string, v float64) bool {
	ws := words(strings.ReplaceAll(lowered, "-", " "))

	for i, w := range ws {
		vals, ok := numberWords[w]
		if !ok {
			continue
		}

		var next []float64
		if i+1 < len(ws) {
			next = numberWords[ws[i+1]]
		}

		switch {
		case len(next) == 1 && isScale(ws[i+1]):
			if len(vals) == 1 && vals[0]*next[0] == v {
				return true
			}
		case slices.Contains(vals, v):
			return true
		case len(vals) == 1 && len(next) == 1 && isTens(vals[0]) && next[0] < 10 && vals[0]+next[0] == v:
			return true
		}
	}

	return false
}

// isScale reports whether w multiplies the number before it.
func isScale(w string) bool {
	return w == "hundred" || w == "thousand" || w == "million" || w == "billion"
}

// isTens reports whether v is twenty, thirty, … ninety.
func isTens(v float64) bool { return v >= 20 && v < 100 && int(v)%10 == 0 }

// wordRun is a word of letters and apostrophes.
var wordRun = regexp.MustCompile(`[\p{L}']+`)

// words are the words of s.
func words(s string) []string { return wordRun.FindAllString(s, -1) }

// Weekdays and months as the engine lists them.
var (
	weekdays = map[string]time.Weekday{
		"sunday": time.Sunday, "monday": time.Monday, "tuesday": time.Tuesday, "wednesday": time.Wednesday,
		"thursday": time.Thursday, "friday": time.Friday, "saturday": time.Saturday,
	}
	months = map[string]time.Month{
		"january": time.January, "february": time.February, "march": time.March, "april": time.April, "may": time.May,
		"june": time.June, "july": time.July, "august": time.August, "september": time.September,
		"october": time.October, "november": time.November, "december": time.December,
	}
)

const (
	monthNames = `(january|february|march|april|may|june|july|august|september|october|november|december)`
	dayNumber  = `(\d{1,2})(?:st|nd|rd|th)?`
	yearNumber = `(?:,?\s+(\d{4})\b)?`
)

var (
	// dateFactDate is the date of the system facts' date: fact.
	dateFactDate = regexp.MustCompile(`(?:^|[^A-Za-z])date:\s*(\d{4}-\d{2}-\d{2})`)
	// weekdayName is a weekday written in full.
	weekdayName = regexp.MustCompile(`(?i)\b(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b`)
	// monthDay is "March 4th", "March 4" with an optional year.
	monthDay = regexp.MustCompile(`(?i)\b` + monthNames + `\s+` + dayNumber + `\b` + yearNumber)
	// dayMonth is "4 March", "4th of March" with an optional year.
	dayMonth = regexp.MustCompile(`(?i)\b` + dayNumber + `\s+(?:of\s+)?` + monthNames + `\b` + yearNumber)
	// isoDateValue is a value that is a calendar date and nothing else.
	isoDateValue = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
)

// dateFactOf is the date of the system facts' date: fact.
func dateFactOf(system string) (time.Time, bool) {
	m := dateFactDate.FindStringSubmatch(system)
	if m == nil {
		return time.Time{}, false
	}

	t, err := time.Parse(time.DateOnly, m[1])

	return t, err == nil
}

// date re-resolves a date against the date: fact: when the request names
// one day, a weekday (the next one after the fact's day, "next Friday"
// too) or a month and day ("March 4th", "4 March": on or after the fact's
// day, or in the year written), every string argument that is a calendar
// date takes it (docs/FINDINGS.md F-056: the date_ probes; a date with a
// time stays, date_time_weekday).
func (r *repairer) date(c *repairCall) {
	d, ok := r.namedDay()
	if !ok {
		return
	}

	day := d.Format(time.DateOnly)

	for k, v := range c.stringArgs {
		if isoDateValue.MatchString(v) && v != day {
			c.set(k, day)
		}
	}
}

// namedDay is the one day the request names against the date fact; a day
// that does not exist ("February 30th") or two different days resolve
// nothing.
func (r *repairer) namedDay() (time.Time, bool) {
	if !r.hasFact {
		return time.Time{}, false
	}

	var days []time.Time

	for _, m := range weekdayName.FindAllStringSubmatch(r.input, -1) {
		ahead := (int(weekdays[strings.ToLower(m[1])])-int(r.fact.Weekday())+6)%7 + 1
		days = append(days, r.fact.AddDate(0, 0, ahead))
	}

	for _, m := range monthDay.FindAllStringSubmatch(r.input, -1) {
		days = append(days, r.monthDay(m[1], m[2], m[3]))
	}

	for _, m := range dayMonth.FindAllStringSubmatch(r.input, -1) {
		days = append(days, r.monthDay(m[2], m[1], m[3]))
	}

	if len(days) == 0 || days[0].IsZero() || slices.ContainsFunc(days, func(t time.Time) bool { return !t.Equal(days[0]) }) {
		return time.Time{}, false
	}

	return days[0], true
}

// monthDay is the day of a month and day, in the year given or else on or
// after the date fact; zero when there is no such day.
func (r *repairer) monthDay(month, day, year string) time.Time {
	mo := months[strings.ToLower(month)]
	dd, _ := strconv.Atoi(day)

	y := r.fact.Year()
	if year != "" {
		y, _ = strconv.Atoi(year)
	}

	t := time.Date(y, mo, dd, 0, 0, 0, 0, time.UTC)
	if year == "" && t.Before(r.fact) {
		t = time.Date(y+1, mo, dd, 0, 0, 0, 0, time.UTC)
	}

	if t.Month() != mo || t.Day() != dd {
		return time.Time{}
	}

	return t
}

// paired makes a call follow the request's polarity between a tool and its
// "un" pair (lock_door and unlock_door): the call's clause naming one
// verb and not the other picks that tool, unless the clause is negated
// (docs/FINDINGS.md F-056: paired_unlock, paired_second; "don't unlock"
// keeps lock_door, bool_own_clause).
func (r *repairer) paired(c *repairCall) {
	base := strings.TrimPrefix(c.name, "un")
	if r.n.params[base] == nil || r.n.params["un"+base] == nil {
		return
	}

	clause := r.clauseOf(c)
	if negation.MatchString(clause) {
		return
	}

	verb, _, _ := strings.Cut(base, "_")
	ws := words(strings.ToLower(clause))

	switch un := slices.Contains(ws, "un"+verb); {
	case un && !slices.Contains(ws, verb):
		c.name = "un" + base
	case !un && slices.Contains(ws, verb):
		c.name = base
	}
}

// Polar phrases: the request verbs llms.txt names ("turn on, unlock,
// louder, turn down") and their opposites.
var (
	onPhrases   = phrases("turn on")
	offPhrases  = phrases("turn off")
	upPhrases   = phrases("louder", "turn up")
	downPhrases = phrases("turn down")
)

// phrases is a case-insensitive pattern for any of the phrases as words.
func phrases(ps ...string) *regexp.Regexp {
	return regexp.MustCompile(`(?i)\b(?:` + strings.Join(ps, "|") + `)\b`)
}

// polarKeys are the boolean arguments that follow an on or off verb: "on",
// which the probes pin.
var polarKeys = []string{"on"}

// polarity makes polar booleans and enums follow the verb of the call's
// clause, unless it is negated: on and off phrases set an on/off boolean
// and an enum of "on" and "off", raising and lowering phrases an enum of
// "up" and "down"; of two, the earlier phrase wins (docs/FINDINGS.md
// F-056: bool_own_clause, enum_louder).
func (r *repairer) polarity(c *repairCall) {
	clause := r.clauseOf(c)
	if negation.MatchString(clause) {
		return
	}

	on := earlier(clause, onPhrases, offPhrases)
	up := earlier(clause, upPhrases, downPhrases)

	for _, k := range r.properties(c.name) {
		if _, ok := c.args.vals[k]; !ok {
			continue
		}

		s := r.property(c.name, k)

		switch {
		case typeOf(s) == "boolean" && slices.Contains(polarKeys, k) && on != 0:
			if c.args.vals[k] != (on > 0) {
				c.set(k, on > 0)
			}
		case hasOptions(s, "on", "off") && on != 0:
			setOption(c, k, on, "on", "off")
		case hasOptions(s, "up", "down") && up != 0:
			setOption(c, k, up, "up", "down")
		}
	}
}

// setOption sets c's argument k to yes for a positive polarity, else to no.
func setOption(c *repairCall, k string, polarity int, yes, no string) {
	v := no
	if polarity > 0 {
		v = yes
	}

	if c.args.vals[k] != v {
		c.set(k, v)
	}
}

// earlier is +1 when pos matches s before neg does, -1 when neg matches
// first, 0 when neither matches.
func earlier(s string, pos, neg *regexp.Regexp) int {
	p, n := pos.FindStringIndex(s), neg.FindStringIndex(s)

	switch {
	case p != nil && (n == nil || p[0] <= n[0]):
		return 1
	case n != nil:
		return -1
	}

	return 0
}

// hasOptions reports whether a schema's enum holds every option.
func hasOptions(s *object, options ...string) bool {
	if s == nil {
		return false
	}

	enum, _ := s.vals["enum"].([]any)

	for _, o := range options {
		if !slices.Contains(enum, any(o)) {
			return false
		}
	}

	return true
}

// enumNamed moves a string enum to the one option the request names as a
// word (docs/FINDINGS.md F-056: enum_named_high, enum_named_low; two named
// or none keep the model's).
func (r *repairer) enumNamed(c *repairCall) {
	for _, k := range r.properties(c.name) {
		var enum []any
		if s := r.property(c.name, k); s != nil {
			enum, _ = s.vals["enum"].([]any)
		}

		var named []string

		for _, o := range enum {
			if s, ok := o.(string); ok && s != "" && phrases(regexp.QuoteMeta(s)).MatchString(r.input) {
				named = append(named, s)
			}
		}

		if v, ok := c.args.vals[k]; ok && len(named) == 1 && v != named[0] {
			c.set(k, named[0])
		}
	}
}

// capitalKeys are the arguments whose value the engine capitalises, each
// shown by a probe (docs/FINDINGS.md F-056: cap_*).
var capitalKeys = []string{"label", "title", "message", "subject", "note", "event_title", "event_name"}

// capitalise upper-cases the first byte of a top-level string argument
// named in capitalKeys when it is an ASCII lower-case letter
// (docs/FINDINGS.md F-056: the cap_ probes; "über" and "3 eggs" stay,
// "iPhone" becomes "IPhone", nested titles stay).
func (r *repairer) capitalise(c *repairCall) {
	for k, v := range c.stringArgs {
		if slices.Contains(capitalKeys, k) && v != "" && v[0] >= 'a' && v[0] <= 'z' {
			c.set(k, string(v[0]-'a'+'A')+v[1:])
		}
	}
}

// dropArguments drops the optional arguments the request does not ground:
// empty strings, and numbers the request states neither in digits nor in
// words (docs/FINDINGS.md F-056: optional_number, optional_zero,
// empty_optional). Strings the request does not state stay: the engine
// keeps a time "15:00" for "3pm" (optional_time).
func (r *repairer) dropArguments(c *repairCall) {
	for _, k := range slices.Clone(c.args.keys) {
		if r.required(c.name, k) {
			continue
		}

		switch v := c.args.vals[k].(type) {
		case string:
			if v == "" {
				c.drop(k)
			}
		case json.Number:
			var schemas []*object
			if s := r.property(c.name, k); s != nil {
				schemas = composed(s)
			}

			f, err := strconv.ParseFloat(string(v), 64)
			if !grounded(v, schemas, r.lower) && (err != nil || !wordNumber(r.lower, f)) {
				c.drop(k)
			}
		}
	}
}

// exclusion is a phrase that excludes what follows it, as the probes show
// (docs/FINDINGS.md F-056: exclude_except, exclude_but_not,
// exclude_apart_from).
var exclusion = regexp.MustCompile(`(?i)\b(?:except|but not|apart from)\s+`)

// negatedVerb is a negation, spelled as the envelope's (F-054), and the
// word after it.
var negatedVerb = regexp.MustCompile(`(?i)\b(?:don['’]?t|do not|never)\s+(\w+)`)

// dropCalls drops the calls for something the request excludes ("except
// the hallway": a call with that value, unless it is the only call) or
// negates ("don't lock": a call to a tool whose name starts with that
// verb), and duplicate calls (docs/FINDINGS.md F-056). A negated verb drops
// its calls from every clause, not only its own: the engine drops the jazz
// of "play jazz in the kitchen and don't play anything in the bedroom"
// (negated_verb).
func (r *repairer) dropCalls(cs []*repairCall) []*repairCall {
	links := exclusion.FindAllStringIndex(r.input, -1)
	excluded := make([]string, 0, len(links))

	for _, m := range links {
		rest := r.input[m[1]:]
		excluded = append(excluded, strings.ToLower(withoutThe(rest[:clauseEnd(rest)])))
	}

	if len(cs) > 1 && len(excluded) > 0 {
		cs = slices.DeleteFunc(slices.Clone(cs), func(c *repairCall) bool {
			if c.args == nil {
				return false
			}

			for _, v := range c.stringArgs {
				if slices.Contains(excluded, strings.ToLower(v)) {
					return true
				}
			}

			return false
		})
	}

	negations := negatedVerb.FindAllStringSubmatch(r.input, -1)
	verbs := make([]string, 0, len(negations))

	for _, m := range negations {
		verbs = append(verbs, strings.ToLower(m[1]))
	}

	seen := map[string]bool{}

	return slices.DeleteFunc(cs, func(c *repairCall) bool {
		verb, _, _ := strings.Cut(c.name, "_")
		if slices.Contains(verbs, verb) {
			return true
		}

		key := c.name + "\x00" + c.identity()
		if seen[key] {
			return true
		}

		seen[key] = true

		return false
	})
}

// identity is c's arguments in an order-free form: two calls with the same
// arguments in another key order are duplicates (PR #66 review).
func (c *repairCall) identity() string {
	if c.args == nil {
		return string(c.bytes())
	}

	var b strings.Builder

	canonical(&b, c.args)

	return b.String()
}

// canonical writes v with every object's keys sorted.
func canonical(b *strings.Builder, v any) {
	switch t := v.(type) {
	case *object:
		keys := slices.Sorted(slices.Values(t.keys))

		b.WriteByte('{')

		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}

			b.WriteString(strconv.Quote(k))
			b.WriteByte(':')
			canonical(b, t.vals[k])
		}

		b.WriteByte('}')
	case []any:
		b.WriteByte('[')

		for i, e := range t {
			if i > 0 {
				b.WriteByte(',')
			}

			canonical(b, e)
		}

		b.WriteByte(']')
	case string:
		b.WriteString(strconv.Quote(t))
	default:
		fmt.Fprint(b, t)
	}
}

// clauseSplit separates the clauses of a request: punctuation, a full stop
// before white space, and "and", "but", "then".
var clauseSplit = regexp.MustCompile(`(?i)\s*(?:[,;!?]|\.(?:\s|$)|\s(?:and|but|then)(?:\s+then)?\s)\s*`)

// clauses are the request's clauses, without empty ones.
func clauses(s string) []string {
	var out []string

	for _, c := range clauseSplit.Split(s, -1) {
		if c = strings.TrimSpace(c); c != "" {
			out = append(out, c)
		}
	}

	return out
}

// clauseEnd is the length of the first clause of s.
func clauseEnd(s string) int {
	if m := clauseSplit.FindStringIndex(s); m != nil {
		return m[0]
	}

	return len(s)
}

// clauseOf is the clause of the request a call belongs to: the first that
// contains one of its string arguments, else the whole request.
func (r *repairer) clauseOf(c *repairCall) string {
	cls := clauses(r.input)

	for _, v := range c.stringArgs {
		if v == "" {
			continue
		}

		for _, cl := range cls {
			if strings.Contains(strings.ToLower(cl), strings.ToLower(v)) {
				return cl
			}
		}
	}

	return r.input
}
