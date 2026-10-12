package toolcall

import (
	"encoding/json"
	"errors"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/spm"
)

// Pinned files as laid out by `just fetch-needle3`, the fixture of `just
// text-prompt-fixtures` (scripts/text_prompt_fixtures.py) and the engine
// corpus of `just oracle-text`.
const (
	tokenizerPath  = "../../models/needle3/tokenizer.model"
	renderFixture  = "../../testdata/text/prompt/render.json"
	engineCases    = "../../testdata/text/engine/cases.json"
	engineLibrary  = "../../testdata/text/engine/library.json"
	engineResponse = "../../testdata/text/engine/responses.json"
)

type renderCase struct {
	Name   string `json:"name"`
	System string `json:"system"`
	Tools  string `json:"tools"`
	Query  string `json:"query"`
	Prompt string `json:"prompt"`
	IDs    []int  `json:"ids"`
}

type renderFile struct {
	Now   string       `json:"now"`
	Cases []renderCase `json:"cases"`
	Dates []struct {
		System string `json:"system"`
		Want   string `json:"want"`
	} `json:"dates"`
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	err = json.Unmarshal(b, v)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func readRender(t *testing.T) renderFile {
	t.Helper()

	var f renderFile

	readJSON(t, renderFixture, &f)

	if len(f.Cases) == 0 || len(f.Dates) == 0 {
		t.Fatalf("%s: %d cases, %d date cases", renderFixture, len(f.Cases), len(f.Dates))
	}

	return f
}

var (
	tokOnce sync.Once
	tok     *spm.Processor
	errTok  error
)

// loadTokenizer loads the pinned tokenizer.model once, skipping without it.
func loadTokenizer(t *testing.T) *spm.Processor {
	t.Helper()

	_, err := os.Stat(tokenizerPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model", tokenizerPath)
	}

	tokOnce.Do(func() { tok, errTok = spm.Load(tokenizerPath) })

	if errTok != nil {
		t.Fatal(errTok)
	}

	return tok
}

// TestRenderMatchesPython renders every fixture case and requires
// upstream's render_example prompt byte for byte, the tools JSON included.
func TestRenderMatchesPython(t *testing.T) {
	for _, c := range readRender(t).Cases {
		tools, err := ToolsJSON([]byte(c.Tools))
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)

			continue
		}

		if got := Render(c.System, tools, c.Query); got != c.Prompt {
			t.Errorf("%s:\n got %q\nwant %q", c.Name, got, c.Prompt)
		}
	}
}

// TestEncodeMatchesPython requires [BOS] + sentencepiece.encode(prompt) id
// for id.
func TestEncodeMatchesPython(t *testing.T) {
	p := loadTokenizer(t)

	for _, c := range readRender(t).Cases {
		if got := Encode(p, c.Prompt); !slices.Equal(got, c.IDs) {
			t.Errorf("%s: %d ids, Python %d; first difference at %d", c.Name, len(got), len(c.IDs), firstDiff(got, c.IDs))
		}
	}
}

func firstDiff(a, b []int) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}

	return min(len(a), len(b))
}

// TestWithDateFactMatchesPython runs upstream's _with_date_fact cases at
// the fixture's fixed clock.
func TestWithDateFactMatchesPython(t *testing.T) {
	f := readRender(t)

	now, err := time.Parse("2006-01-02T15:04:05", f.Now)
	if err != nil {
		t.Fatal(err)
	}

	for _, d := range f.Dates {
		if got := WithDateFact(d.System, now); got != d.Want {
			t.Errorf("WithDateFact(%q) = %q, want %q", d.System, got, d.Want)
		}
	}
}

// TestToolsJSONErrors: invalid JSON, trailing data and tools that are not
// an array are rejected.
func TestToolsJSONErrors(t *testing.T) {
	for _, raw := range []string{``, `[`, `{"name":"x"}`, `"tools"`, `[1] [2]`, `[{"a":}]`, `[1,]`} {
		_, err := ToolsJSON([]byte(raw))
		if err == nil {
			t.Errorf("ToolsJSON(%q) accepted", raw)
		}
	}
}

// TestPyFloat pins Python's float repr on values the fixture does not
// reach: the exponent switches below 1e-4 and from 1e16, at least two
// exponent digits, a mantissa without a trailing zero.
func TestPyFloat(t *testing.T) {
	tenth := 0.1 // a variable: Go folds the constant 0.1 + 0.2 to exactly 0.3

	for _, c := range []struct {
		in   float64
		want string
	}{
		{0, "0.0"},
		{1, "1.0"},
		{0.5, "0.5"},
		{1e-4, "0.0001"},
		{1.5e-5, "1.5e-05"},
		{1e15, "1000000000000000.0"},
		{1e16, "1e+16"},
		{1.25e22, "1.25e+22"},
		{123.456, "123.456"},
		{5e-324, "5e-324"},
		{1e100, "1e+100"},
		{-2.5e-8, "-2.5e-08"},
		{tenth + 0.2, "0.30000000000000004"},
	} {
		if got := pyFloat(c.in); got != c.want {
			t.Errorf("pyFloat(%v) = %s, want %s", c.in, got, c.want)
		}
	}
}

// TestStaticPrefixMatchesEngine requires the token count of [BOS] and the
// rendering up to </tools> to equal needle_init's on every case of the
// engine corpus (docs/FINDINGS.md F-048).
func TestStaticPrefixMatchesEngine(t *testing.T) {
	p := loadTokenizer(t)

	var corpus struct {
		System   string                     `json:"system"`
		Toolsets map[string]json.RawMessage `json:"toolsets"`
		Cases    []struct {
			ID      string `json:"id"`
			Toolset string `json:"toolset"`
			System  string `json:"system"`
		} `json:"cases"`
	}

	var lib struct {
		Cases map[string]struct {
			Init int `json:"init"`
		} `json:"cases"`
	}

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineLibrary, &lib)

	counts := map[int]int{}

	for _, c := range corpus.Cases {
		system := corpus.System
		if c.System != "" {
			system = c.System
		}

		tools, err := ToolsJSON(corpus.Toolsets[c.Toolset])
		if err != nil {
			t.Fatal(err)
		}

		want, ok := lib.Cases[c.ID]
		if !ok {
			t.Fatalf("%s: no needle_init count", c.ID)
		}

		got := len(StaticPrefix(p, system, tools))
		if got != want.Init {
			t.Errorf("%s: static prefix %d tokens, needle_init %d", c.ID, got, want.Init)
		}

		counts[got]++
	}

	if len(corpus.Cases) != len(lib.Cases) {
		t.Errorf("%d cases, %d needle_init counts", len(corpus.Cases), len(lib.Cases))
	}

	t.Logf("%d cases: static prefix counts %v (count: cases)", len(corpus.Cases), counts)
}

// TestStaticPrefixIsPromptPrefix: the static prefix is the start of the
// whole prompt's ids, so a cached prefix can be extended by the query.
func TestStaticPrefixIsPromptPrefix(t *testing.T) {
	p := loadTokenizer(t)

	for _, c := range readRender(t).Cases {
		tools, err := ToolsJSON([]byte(c.Tools))
		if err != nil {
			t.Fatal(err)
		}

		prefix := StaticPrefix(p, c.System, tools)
		if !slices.Equal(prefix, c.IDs[:min(len(prefix), len(c.IDs))]) {
			t.Errorf("%s: the static prefix is not the start of the prompt's ids", c.Name)
		}

		if !strings.Contains(c.Prompt, toolsEnd) {
			t.Errorf("%s: no %s in the prompt", c.Name, toolsEnd)
		}
	}
}
