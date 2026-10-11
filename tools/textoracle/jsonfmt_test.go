//go:build oracle

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCanonicalNumber(t *testing.T) {
	for in, want := range map[string]string{
		"0.9492":    "0.9492",
		"0.0920":    "0.092",
		"1.0000":    "1.0",
		"0.0000":    "0.0",
		"10.0":      "10.0",
		"100.50":    "100.5",
		"-0.50":     "-0.5",
		"2":         "2",
		"1E5":       "1e5",
		"1e+05":     "1e5",
		"1.50e-007": "1.5e-7",
		"1e+00":     "1",
		"2.0E0":     "2.0",
	} {
		if got := canonicalNumber(in); got != want {
			t.Errorf("canonicalNumber(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestMarshalPretty pins the layouts prettier 3.7 gives the same documents (checked with `prettier --check`).
func TestMarshalPretty(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"scalars", `{"b":true,"n":null,"x":0.9490,"s":"a<b> é \"q\""}`, `{
  "b": true,
  "n": null,
  "x": 0.949,
  "s": "a<b> é \"q\""
}
`},
		{"empty", `{"a":[],"o":{},"e":[{},{}]}`, `{
  "a": [],
  "o": {},
  "e": [{}, {}]
}
`},
		{"short array", `{"ungrounded":["create_event.time"],"m":[[],[1]],"n":[[1,2],[3]]}`, `{
  "ungrounded": ["create_event.time"],
  "m": [[], [1]],
  "n": [[1, 2], [3]]
}
`},
		{"array of objects", `{"calls":[{"name":"x"}]}`, `{
  "calls": [
    {
      "name": "x"
    }
  ]
}
`},
		{"nested arrays break", `{"k":[[1,2],[3,4]]}`, `{
  "k": [
    [1, 2],
    [3, 4]
  ]
}
`},
		{"long array", `{"a_quite_long_key_name_here_for_testing_purposes":["a_string_value","another_string"]}`, `{
  "a_quite_long_key_name_here_for_testing_purposes": [
    "a_string_value",
    "another_string"
  ]
}
`},
		{"comma counts", `{"k":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],"z":1}`, `{
  "k": [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  ],
  "z": 1
}
`},
		{"last member fits", `{"z":1,"k":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]}`, `{
  "z": 1,
  "k": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
}
`},
		{"number fill", `{"d":[1,2,3,0.5000000000000000000000000000001,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]}`, `{
  "d": [
    1, 2, 3, 0.5000000000000000000000000000001, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22
  ]
}
`},
		{"long string stays", `{"reasoning":"'10 minutes' -> duration_minutes 10; '10 minutes' -> duration_minutes 10; '10 minutes'"}`, `{
  "reasoning": "'10 minutes' -> duration_minutes 10; '10 minutes' -> duration_minutes 10; '10 minutes'"
}
`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n, err := parseJSON([]byte(tc.in))
			if err != nil {
				t.Fatal(err)
			}

			if got := string(marshalPretty(n)); got != tc.want {
				t.Errorf("got\n%s\nwant\n%s", got, tc.want)
			}
		})
	}
}

func TestParseJSONKeepsOrderAndLiterals(t *testing.T) {
	n, err := parseJSON([]byte(`{"z":1.50,"a":{"y":[true,"x"],"b":null}}`))
	if err != nil {
		t.Fatal(err)
	}

	if got := string(compact(n)); got != `{"z":1.50,"a":{"y":[true,"x"],"b":null}}` {
		t.Errorf("compact = %s", got)
	}

	for _, bad := range []string{`{"a":1,"a":2}`, `{"a":1} {}`, `{"a":`, ``} {
		_, err := parseJSON([]byte(bad))
		if err == nil {
			t.Errorf("parseJSON(%q) succeeded", bad)
		}
	}
}

func TestNormaliseResponse(t *testing.T) {
	out := `{"type":"call","confidence":0.9492,"prefill_tps":4300.0,"decode_tps":850.0,"peak_ram_mb":91.8,` +
		`"validation":{"ungrounded":[],"negation":false},"audio_text":"Hi.","audio_words":[{"word":"Hi.",` +
		`"start":0.40,"end":0.56}],"audio_ttft_ms":4.6,"audio_decode_tps":1427.0}` + "\n"

	n, err := normaliseResponse([]byte(out))
	if err != nil {
		t.Fatal(err)
	}

	want := `{"type":"call","confidence":0.9492,"validation":{"ungrounded":[],"negation":false},"audio_text":"Hi.",` +
		`"audio_words":[{"word":"Hi.","start":0.40,"end":0.56}]}`
	if got := string(compact(n)); got != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}

	for _, bad := range []string{`[1]`, `not json`, `{"a":1}{"b":2}`} {
		_, err := normaliseResponse([]byte(bad))
		if err == nil {
			t.Errorf("normaliseResponse(%q) succeeded", bad)
		}
	}
}

func TestPyDumps(t *testing.T) {
	n, err := parseJSON([]byte(`[{"status":"ok","n":10,"ends_at":"09:40","x":"\u00e9\n\ud83d\ude00","f":[1.5,true,null]}]`))
	if err != nil {
		t.Fatal(err)
	}

	// Python's json.dumps of the decoded value; "x" holds an e-acute, a newline and an emoji (U+1F600).
	want := `[{"status": "ok", "n": 10, "ends_at": "09:40", "x": "\u00e9\n\ud83d\ude00", "f": [1.5, true, null]}]`
	if got := pyDumps(n); got != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
}

func TestPrettyJSONEscapesNoHTML(t *testing.T) {
	js, err := prettyJSON(struct {
		A string   `json:"a"`
		B []string `json:"b"`
	}{"<&>", []string{}})
	if err != nil {
		t.Fatal(err)
	}

	if want := "{\n  \"a\": \"<&>\",\n  \"b\": []\n}\n"; string(js) != want {
		t.Errorf("got %q, want %q", js, want)
	}
}

// TestFixturesRoundTrip checks that every committed JSON fixture (cases.json, hand-written, aside) is what the
// writer gives for its own content, so a regeneration from the same engine answers diffs clean.
func TestFixturesRoundTrip(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", "..", "testdata", "text", "engine", "*.json"))
	if err != nil {
		t.Fatal(err)
	}

	if len(paths) == 0 {
		t.Skip("no fixtures in testdata/text/engine (run `go run -tags oracle ./tools/textoracle all`)")
	}

	for _, p := range paths {
		if strings.HasSuffix(p, "cases.json") {
			continue
		}

		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}

		n, err := parseJSON(raw)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}

		if got := marshalPretty(n); string(got) != string(raw) {
			t.Errorf("%s does not round-trip through marshalPretty", p)
		}
	}
}
