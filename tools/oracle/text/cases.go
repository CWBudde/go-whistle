//go:build oracle

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"slices"
)

// corpus is testdata/text/engine/cases.json, the hand-written tool-calling requests the engine answers: named tool
// sets (JSON arrays of function schemas, what `needle --tools` reads), the system facts every case gets unless it
// names its own, the text cases and the speech cases.
type corpus struct {
	System   string                     `json:"system"`
	Toolsets map[string]json.RawMessage `json:"toolsets"`
	Cases    []textCase                 `json:"cases"`
	Audio    []audioCase                `json:"audio"`
}

// textCase is one request: a query, optionally followed by tool results that are fed back as further turns of the
// same conversation (upstream Needle.run sends json.dumps(results), a list with one entry per executed call).
type textCase struct {
	ID      string            `json:"id"`
	Covers  string            `json:"covers"`
	Toolset string            `json:"toolset"`
	System  string            `json:"system,omitempty"`
	Query   string            `json:"query"`
	Results []json.RawMessage `json:"results,omitempty"`
	Max     int               `json:"max,omitempty"`
}

// audioCase is one speech request: a clip of testdata/audio answered against a tool set.
type audioCase struct {
	ID      string `json:"id"`
	Clip    string `json:"clip"`
	Toolset string `json:"toolset"`
}

// defaultMax is needle --max's default response token limit, which the library calls pass explicitly.
const defaultMax = 512

var (
	caseID = regexp.MustCompile(`^[a-z0-9]+(?:_[a-z0-9]+)*$`)
	// dateFact is upstream's _DATE_KEY: the `date:` key itself, not the tail of `update:`. Every system text must carry
	// one, else the answers to relative dates would depend on what the engine does without one.
	dateFact = regexp.MustCompile(`(?:^|[^A-Za-z])date:`)
)

// readCorpus reads and checks a case corpus.
func readCorpus(p string) (corpus, error) {
	raw, err := os.ReadFile(p) // #nosec G304 -- the corpus path is a flag.
	if err != nil {
		return corpus{}, fmt.Errorf("read cases: %w", err)
	}

	return parseCorpus(raw)
}

// parseCorpus decodes a case corpus, rejecting unknown fields, and checks it.
func parseCorpus(raw []byte) (corpus, error) {
	var c corpus

	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()

	err := dec.Decode(&c)
	if err != nil {
		return corpus{}, fmt.Errorf("cases: %w", err)
	}

	err = c.check()
	if err != nil {
		return corpus{}, fmt.Errorf("cases: %w", err)
	}

	return c, nil
}

func (c corpus) check() error {
	for name, ts := range c.Toolsets {
		var tools []map[string]any

		err := json.Unmarshal(ts, &tools)
		if err != nil {
			return fmt.Errorf("toolset %s: not a JSON array of objects: %w", name, err)
		}

		for i, t := range tools {
			if _, ok := t["name"].(string); !ok {
				return fmt.Errorf("toolset %s: tool %d has no name", name, i)
			}
		}
	}

	if len(c.Cases) == 0 {
		return errors.New("no cases")
	}

	seen := map[string]bool{}

	for _, k := range c.Cases {
		err := c.checkCase(k, seen)
		if err != nil {
			return fmt.Errorf("case %q: %w", k.ID, err)
		}
	}

	for _, a := range c.Audio {
		if !caseID.MatchString(a.ID) || seen[a.ID] {
			return fmt.Errorf("audio case %q: id must be unique snake_case", a.ID)
		}

		seen[a.ID] = true

		if _, ok := c.Toolsets[a.Toolset]; !ok {
			return fmt.Errorf("audio case %q: unknown toolset %q", a.ID, a.Toolset)
		}

		if a.Clip == "" {
			return fmt.Errorf("audio case %q: no clip", a.ID)
		}
	}

	if !dateFact.MatchString(c.System) {
		return errors.New("the default system text has no date: fact")
	}

	return nil
}

func (c corpus) checkCase(k textCase, seen map[string]bool) error {
	if !caseID.MatchString(k.ID) {
		return errors.New("id must be snake_case")
	}

	if seen[k.ID] {
		return errors.New("duplicate id")
	}

	seen[k.ID] = true

	if _, ok := c.Toolsets[k.Toolset]; !ok {
		return fmt.Errorf("unknown toolset %q", k.Toolset)
	}

	if k.Query == "" {
		return errors.New("empty query")
	}

	if k.System != "" && !dateFact.MatchString(k.System) {
		return errors.New("system text has no date: fact")
	}

	if k.Max < 0 {
		return errors.New("negative max")
	}

	for i, r := range k.Results {
		n, err := parseJSON(r)
		if err != nil {
			return fmt.Errorf("result %d: %w", i, err)
		}

		if n.kind != kindArray {
			return fmt.Errorf("result %d is not an array, one entry per executed call", i)
		}
	}

	return nil
}

// system is the system text of case k.
func (c corpus) system(k textCase) string {
	if k.System != "" {
		return k.System
	}

	return c.System
}

// tools is the tools JSON of a tool set, compact.
func (c corpus) tools(name string) string {
	var b bytes.Buffer

	_ = json.Compact(&b, c.Toolsets[name]) // checked by parseCorpus

	return b.String()
}

// toolsetNames are the corpus's tool sets, sorted.
func (c corpus) toolsetNames() []string {
	names := make([]string, 0, len(c.Toolsets))
	for n := range c.Toolsets {
		names = append(names, n)
	}

	slices.Sort(names)

	return names
}

// turns are the inputs of case k in order: the query, then each tool result as Python's json.dumps spells it.
func (k textCase) turns() []string {
	out := make([]string, 0, 1+len(k.Results))
	out = append(out, k.Query)

	for _, r := range k.Results {
		n, _ := parseJSON(r) // checked by parseCorpus
		out = append(out, pyDumps(n))
	}

	return out
}

// max is the case's response token limit.
func (k textCase) max() int {
	if k.Max > 0 {
		return k.Max
	}

	return defaultMax
}
