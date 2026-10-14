package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// defaultMax is needle --max's default response token limit, the budget of
// every case that sets none.
const defaultMax = 512

// textCase is one first turn of the engine corpus
// (testdata/text/engine/cases.json): its tools, system text, query and
// token budget, and the function_calls the engine answered it with
// (responses.json).
type textCase struct {
	ID, System, Query string
	Tools             []byte // a JSON array of function schemas, as `needle --tools` reads
	Max               int
	EngineCalls       json.RawMessage
}

// loadCases reads the corpus at casesPath and the engine's recorded
// answers at responsesPath: every case's first turn, its tool results
// left out.
func loadCases(casesPath, responsesPath string) ([]textCase, error) {
	var corpus struct {
		System   string                     `json:"system"`
		Toolsets map[string]json.RawMessage `json:"toolsets"`
		Cases    []struct {
			ID      string `json:"id"`
			Toolset string `json:"toolset"`
			System  string `json:"system"`
			Query   string `json:"query"`
			Max     int    `json:"max"`
		} `json:"cases"`
	}

	err := readJSON(casesPath, &corpus)
	if err != nil {
		return nil, err
	}

	var responses map[string][]struct {
		Calls json.RawMessage `json:"function_calls"`
	}

	err = readJSON(responsesPath, &responses)
	if err != nil {
		return nil, err
	}

	if len(corpus.Cases) == 0 {
		return nil, fmt.Errorf("%s: no cases", casesPath)
	}

	out := make([]textCase, 0, len(corpus.Cases))

	for _, c := range corpus.Cases {
		tools, ok := corpus.Toolsets[c.Toolset]
		if !ok {
			return nil, fmt.Errorf("%s: unknown toolset %q", c.ID, c.Toolset)
		}

		answers := responses[c.ID]
		if len(answers) == 0 {
			return nil, fmt.Errorf("%s: no recorded answer in %s", c.ID, responsesPath)
		}

		k := textCase{
			ID: c.ID, System: corpus.System, Query: c.Query, Tools: tools, Max: defaultMax,
			EngineCalls: answers[0].Calls,
		}

		if c.System != "" {
			k.System = c.System
		}

		if c.Max > 0 {
			k.Max = c.Max
		}

		if k.Query == "" || k.System == "" {
			return nil, fmt.Errorf("%s: empty query or system text", c.ID)
		}

		out = append(out, k)
	}

	return out, nil
}

// readJSON reads path into v.
func readJSON(path string, v any) error {
	raw, err := os.ReadFile(path) // #nosec G304 -- a flag or a default path.
	if err != nil {
		return fmt.Errorf("read: %w", err)
	}

	err = json.Unmarshal(raw, v)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}

	return nil
}
