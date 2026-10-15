package needle

import (
	"maps"
	"slices"
	"testing"
)

// The expectations below are what upstream's Python helpers return
// (needle/__init__.py at the pinned commit, Python 3.14) for the same
// inputs.

// TestSourceYears pins _source_years: a year after a day and a month or a
// month and a day (not the hour of a time), after a month alone, after
// "year", and at the start of a year-first numeric date; digits of any
// script.
func TestSourceYears(t *testing.T) {
	for _, c := range []struct {
		text string
		want []int
	}{
		{"meet on 5 june 2024 and june 5, 2025", []int{2024, 2025}},
		{"march 1999", []int{1999}},
		{"year 12", []int{12}},
		{"the year  2030x", nil},
		{"2023-01-02", []int{2023}},
		{"2023/1/2", []int{2023}},
		{"12023-01-02", nil},
		{"2023-01-022", nil},
		{"2023-01-02x", []int{2023}},
		{"5 june 19:30", nil},
		{"june 5 7 pm", nil},
		{"june 5 7pm", nil},
		{"june 5, 2024 at 7 a.m.", []int{2024}},
		{"5 june 2024am", nil},
		{"5 june 2024 amazing", []int{2024}},
		{"5th june 2024", []int{2024}},
		{"june 5th, 2024", []int{2024}},
		{"june 1, 20245", nil},
		{"dec 2024", []int{2024}},
		{"decimal 2024", nil},
		{"sept 2024", []int{2024}},
		{"sep 5 2024", []int{2024}},
		{"on 5 june 2024: meet", []int{2024}},
		{"5 june 2024:30", []int{2024}},
		{"may 2024 and may 5 2025", []int{2024, 2025}},
		{"x5 june 2024", []int{2024}},
		{"5 june ٢٠٢٤", []int{2024}},
		{"5 june 2024٥", []int{2024}},
		{"١٢ june 2024", []int{2024}},
		{"2024-٠١-02", []int{2024}},
		{"june 5 2024", []int{2024}},
		{"5 june, 2024", []int{2024}},
		{"june 5 ,2024", nil},
		{"may 12 2024p.m.", nil},
		{"mar 12 2024 p.mx", []int{2024}},
		{"2024-10-15 Thu 12:00", []int{2024}},
		{"Meet on 5 JUNE 2024", []int{2024}},
		{"İjune 1993", []int{1993}},
		{"İyear 12", []int{12}},
		{"june İ 1993", nil},
	} {
		if got := slices.Sorted(maps.Keys(sourceYears(c.text))); !slices.Equal(got, c.want) {
			t.Errorf("sourceYears(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// TestRelativeCue pins _relative_cue: the cue words as whole words (a
// Unicode word boundary on both sides) in any case.
func TestRelativeCue(t *testing.T) {
	for _, c := range []struct {
		text string
		want bool
	}{
		{"Tomorrow at 5", true},
		{"next week", true},
		{"in 3 days", true},
		{"in 3 dayss", false},
		{"in 12 weeks", true},
		{"thistle", false},
		{"nowhere", false},
		{"now", true},
		{"émonday", false},
		{"on Monday", true},
		{"", false},
		{"in ٣ days", true},
		{"the coming days", true},
		{"today_", false},
		{"_today", false},
		{"tonıght", true},
		{"TONİGHT", true},
		{"frıday", true},
		{"İN 3 DAYS", true},
	} {
		if got := relativeCue(c.text); got != c.want {
			t.Errorf("relativeCue(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// TestLicensedYears pins _licensed_years: the system facts' years join
// only when the input has years of its own and reasons relatively.
func TestLicensedYears(t *testing.T) {
	system := "date: 2026-10-15 Thu 12:00"

	for _, c := range []struct {
		seen     []int
		system   string
		relative bool
		want     []int
	}{
		{nil, system, true, nil},
		{[]int{2024}, system, true, []int{2024, 2026}},
		{[]int{2024}, system, false, []int{2024}},
		{[]int{2024}, "", true, []int{2024}},
	} {
		seen := map[int]bool{}
		for _, y := range c.seen {
			seen[y] = true
		}

		got := slices.Sorted(maps.Keys(licensedYears(seen, c.system, c.relative)))
		if !slices.Equal(got, c.want) {
			t.Errorf("licensedYears(%v, %q, %v) = %v, want %v", c.seen, c.system, c.relative, got, c.want)
		}
	}
}

// TestWalkGrounding pins _walk_grounding: date and date-time values found
// through properties, items, $ref (one that points at itself stops) and an
// anyOf or oneOf with one branch besides null, checked only when there
// are years.
func TestWalkGrounding(t *testing.T) {
	schema := decodeJSON([]byte(`{"name":"book","parameters":{"type":"object",` +
		`"$defs":{"D":{"type":"string","format":"date"}},"properties":{` +
		`"day":{"type":"string","format":"date"},"when":{"type":"string","format":"date-time"},` +
		`"ref":{"$ref":"#/$defs/D"},"maybe":{"anyOf":[{"type":"string","format":"date"},{"type":"null"}]},` +
		`"two":{"oneOf":[{"type":"string","format":"date"},{"type":"integer"}]},` +
		`"trip":{"type":"object","properties":{"start":{"type":"string","format":"date"}}},` +
		`"stops":{"type":"array","items":{"type":"object","properties":{"on":{"$ref":"#/$defs/D"}}}},` +
		`"loop":{"$ref":"#/properties/loop"},"note":{"type":"string"}}}}`))
	args := decodeJSON([]byte(`{"day":"2024-03-01","when":"2025-01-01T10:00","ref":"2024-05-05",` +
		`"maybe":"2023-01-01","two":"2023-02-02","trip":{"start":"2024-07-07"},` +
		`"stops":[{"on":"2024-01-01"},{"on":"2026-01-01"}],"loop":"2020-01-01","note":"2020-01-01",` +
		`"extra":"2020-01-01","bad":"24-01-01"}`))
	all := []string{"day", "maybe", "ref", "stops[0].on", "stops[1].on", "trip.start", "when"}

	for _, c := range []struct {
		years            []int
		checked, failing []string
	}{
		{nil, nil, nil},
		{[]int{2024}, all, []string{"maybe", "stops[1].on", "when"}},
		{[]int{2024, 2025}, all, []string{"maybe", "stops[1].on"}},
	} {
		years := map[int]bool{}
		for _, y := range c.years {
			years[y] = true
		}

		checked, failures := walkGrounding(schema, args, years)
		if got := slices.Sorted(maps.Keys(checked)); !slices.Equal(got, c.checked) {
			t.Errorf("years %v: checked %v, want %v", c.years, got, c.checked)
		}

		if got := slices.Sorted(maps.Keys(failures)); !slices.Equal(got, c.failing) {
			t.Errorf("years %v: failures %v, want %v", c.years, got, c.failing)
		}
	}
}

// TestGroundedNumberPaths pins _grounded_number_paths: numbers as written
// in the sources ("1,234" is 1234, "1.50" is 1.5, a sign kept), not inside
// a word or after '.' or ',', not in a date fact; booleans are no numbers.
func TestGroundedNumberPaths(t *testing.T) {
	for _, c := range []struct {
		args    string
		sources []string
		want    []string
	}{
		{
			`{"n":1234,"m":1.5,"b":true,"l":[7,{"x":1e3}],"s":"12","neg":-3,"z":0.1}`,
			[]string{"order 1,234 items at 1.50 each, -3 and 0.10 and 1000", ""},
			[]string{"l[1].x", "m", "n", "neg", "z"},
		},
		{
			`{"n":2026,"t":10,"d":15}`,
			[]string{"", "date: 2026-10-15 Thu 12:00; 2026-10-15T09:30:00 ok 15"},
			[]string{"d"},
		},
		{`{"n":5}`, []string{"x5 and 5.5 and a,5 and .5"}, nil},
		{`{"n":5,"k":1234567}`, []string{"1,234,567 and 5,"}, []string{"k", "n"}},
		{`{"n":12,"m":1234}`, []string{"1,2345 and 12,34"}, []string{"n"}},
		{`{"n":3}`, []string{"+3"}, []string{"n"}},
		{`{"n":0}`, []string{"-0"}, []string{"n"}},
		{`{"n":7}`, []string{"٧ apples"}, []string{"n"}},
		{`{"n":1e400}`, []string{"1"}, nil},
		{`{}`, []string{"nothing"}, nil},
	} {
		got := slices.Sorted(maps.Keys(groundedNumberPaths(decodeJSON([]byte(c.args)), c.sources...)))
		if !slices.Equal(got, c.want) {
			t.Errorf("groundedNumberPaths(%s, %q) = %v, want %v", c.args, c.sources, got, c.want)
		}
	}
}
