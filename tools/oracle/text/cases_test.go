//go:build oracle

package main

import (
	"encoding/binary"
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const tinyCorpus = `{
  "system": "date: 2026-10-11 Sun 09:30",
  "toolsets": {"t": [{"name": "set_timer", "parameters": {"type": "object"}}]},
  "cases": [
    {"id": "one", "covers": "x", "toolset": "t", "query": "set a timer"},
    {"id": "two", "covers": "x", "toolset": "t", "system": "date: 2026-01-01 Thu 08:00; locale: de-DE",
     "query": "q", "results": [[{"status": "ok", "n": 1}]], "max": 8}
  ],
  "audio": [{"id": "clip_case", "clip": "ls-x", "toolset": "t"}]
}`

func TestParseCorpus(t *testing.T) {
	c, err := parseCorpus([]byte(tinyCorpus))
	if err != nil {
		t.Fatal(err)
	}

	one, two := c.Cases[0], c.Cases[1]

	if got := c.system(one); got != "date: 2026-10-11 Sun 09:30" {
		t.Errorf("default system = %q", got)
	}

	if got := c.system(two); !strings.HasSuffix(got, "locale: de-DE") {
		t.Errorf("case system = %q", got)
	}

	if one.max() != defaultMax || two.max() != 8 {
		t.Errorf("max = %d, %d", one.max(), two.max())
	}

	if got := c.tools("t"); got != `[{"name":"set_timer","parameters":{"type":"object"}}]` {
		t.Errorf("tools = %s", got)
	}

	want := []string{"q", `[{"status": "ok", "n": 1}]`}
	if got := two.turns(); !slices.Equal(got, want) {
		t.Errorf("turns = %q, want %q", got, want)
	}
}

func TestParseCorpusRejects(t *testing.T) {
	for name, edit := range map[string][2]string{
		"unknown field":   {`"covers": "x", "toolset": "t", "query": "set a timer"`, `"covers": "x", "toolset": "t", "query": "q", "extra": 1`},
		"duplicate id":    {`"id": "two"`, `"id": "one"`},
		"bad id":          {`"id": "one"`, `"id": "One-1"`},
		"unknown toolset": {`"toolset": "t", "query": "set a timer"`, `"toolset": "u", "query": "set a timer"`},
		"empty query":     {`"query": "set a timer"`, `"query": ""`},
		"no date fact":    {`"system": "date: 2026-10-11 Sun 09:30"`, `"system": "update: 2026-10-11"`},
		"case date fact":  {`"system": "date: 2026-01-01 Thu 08:00; locale: de-DE"`, `"system": "locale: de-DE"`},
		"bad result":      {`"results": [[{"status": "ok", "n": 1}]]`, `"results": ["x"]`},
		"negative max":    {`"max": 8`, `"max": -1`},
		"nameless tool":   {`{"name": "set_timer", `, `{"title": "set_timer", `},
		"audio no clip":   {`"clip": "ls-x"`, `"clip": ""`},
		"audio dup id":    {`"id": "clip_case"`, `"id": "one"`},
	} {
		t.Run(name, func(t *testing.T) {
			in := strings.Replace(tinyCorpus, edit[0], edit[1], 1)
			if in == tinyCorpus {
				t.Fatalf("edit %q does not apply", edit[0])
			}

			_, err := parseCorpus([]byte(in))
			if err == nil {
				t.Error("parseCorpus succeeded")
			}
		})
	}
}

// TestCommittedCorpus checks testdata/text/engine/cases.json: it parses, every case has a fixed date fact, and the
// cases the date and depth subcommands name exist.
func TestCommittedCorpus(t *testing.T) {
	c, err := readCorpus(filepath.Join("..", "..", "testdata", "text", "engine", "cases.json"))
	if err != nil {
		t.Fatal(err)
	}

	if n := len(c.Cases); n < 12 || n > 20 {
		t.Errorf("%d cases, want 12 to 20", n)
	}

	for _, k := range c.Cases {
		if !strings.HasPrefix(c.system(k), "date: 2026-10-11 Sun 09:30") {
			t.Errorf("%s: system %q does not start with the fixed date fact", k.ID, c.system(k))
		}
	}

	for _, id := range slices.Concat(dateCases, depthCases) {
		_, err := c.find(id)
		if err != nil {
			t.Error(err)
		}
	}
}

func TestSameCheck(t *testing.T) {
	parse := func(s string) node {
		n, err := parseJSON([]byte(s))
		if err != nil {
			t.Fatal(err)
		}

		return n
	}

	ref := []node{parse(`{"type":"call","confidence":0.5,"reasoning":"a"}`)}
	s := newCheck("x")
	s.note("same", ref, []node{parse(`{"type":"call","confidence":0.5,"reasoning":"a"}`)})
	s.note("conf", ref, []node{parse(`{"type":"call","confidence":0.6,"reasoning":"a"}`)})
	s.note("both", ref, []node{parse(`{"type":"call","confidence":0.6,"reasoning":"b","extra":1}`)})

	if s.Identical || !slices.Equal(s.Differing, []string{"conf", "both"}) {
		t.Errorf("differing = %v", s.Differing)
	}

	want := map[string]int{"confidence": 2, "reasoning": 1, "extra": 1}
	if got := mustJSON(t, s.Fields); got != mustJSON(t, want) {
		t.Errorf("fields = %s", got)
	}

	if got := diffKeys(ref, nil); !slices.Equal(got, []string{"turns"}) {
		t.Errorf("diffKeys on a missing turn = %v", got)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()

	js, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}

	return string(js)
}

func TestTally(t *testing.T) {
	man := textManifest{Types: map[string]int{}, Calls: map[string]int{}}

	for _, s := range []string{
		`{"type":"call","function_calls":[{"name":"a"}],"confidence":0.9490}`,
		`{"type":"call","function_calls":[],"confidence":0.0315}`,
		`{"type":"respond","function_calls":[],"confidence":0.6369}`,
	} {
		n, err := parseJSON([]byte(s))
		if err != nil {
			t.Fatal(err)
		}

		tally(&man, n)
	}

	if man.Types["call"] != 2 || man.Types["respond"] != 1 || man.Calls["0"] != 2 || man.Calls["1"] != 1 {
		t.Errorf("types %v, calls %v", man.Types, man.Calls)
	}

	if man.MinConf != "0.0315" || man.MaxConf != "0.949" {
		t.Errorf("confidence range %s .. %s", man.MinConf, man.MaxConf)
	}
}

func TestReadPCM16(t *testing.T) {
	wav := func(rate uint32, bits uint16, samples ...int16) []byte {
		b := []byte("RIFF\x00\x00\x00\x00WAVEfmt ")
		b = binary.LittleEndian.AppendUint32(b, 16)
		b = binary.LittleEndian.AppendUint16(b, 1)
		b = binary.LittleEndian.AppendUint16(b, 1)
		b = binary.LittleEndian.AppendUint32(b, rate)
		b = binary.LittleEndian.AppendUint32(b, rate*2)
		b = binary.LittleEndian.AppendUint16(b, 2)
		b = binary.LittleEndian.AppendUint16(b, bits)
		b = append(b, "data"...)
		b = binary.LittleEndian.AppendUint32(b, uint32(2*len(samples)))

		for _, s := range samples {
			b = binary.LittleEndian.AppendUint16(b, uint16(s))
		}

		return b
	}

	pcm, err := readPCM16(wav(sampleRate, 16, 0, 16384, -32768))
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(pcm, []float32{0, 0.5, -1}) {
		t.Errorf("pcm = %v", pcm)
	}

	for _, bad := range [][]byte{wav(8000, 16, 1), wav(sampleRate, 8, 1), []byte("RIFF")} {
		_, err := readPCM16(bad)
		if err == nil {
			t.Error("readPCM16 accepted a non-16 kHz-16-bit WAV")
		}
	}
}

func TestSlicesWith(t *testing.T) {
	got := slicesWith([]string{"--model", "a", "--tools", "t"}, "--model", "b")
	if want := []string{"--model", "a", "--model", "b", "--tools", "t"}; !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}
