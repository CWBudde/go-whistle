package toolcall

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/text"
)

const needle3Safetensors = "../../models/needle3/needle3.safetensors"

// call is one function call of a <tool_call> block or the engine's envelope.
type call struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// greedy decodes from ids with the float model until </tool_call>,
// <|im_end|> or EOS, at most maxNew tokens, and returns the generated text
// and the smallest top-1 margin (top-1 minus top-2 logit) along the way.
func greedy(t *testing.T, m *text.Model, p *spm.Processor, ids []int, maxNew int) (string, float64) {
	t.Helper()

	stop := map[int]bool{p.EOSID(): true}

	for _, piece := range []string{"</tool_call>", imEnd} {
		id, ok := p.PieceID(piece)
		if !ok {
			t.Fatalf("no piece %s", piece)
		}

		stop[id] = true
	}

	s := m.NewState()
	v := m.Config().OutVocab

	logits, err := m.Extend(s, ids)
	if err != nil {
		t.Fatal(err)
	}

	row := logits[len(logits)-v:]
	margin := math.Inf(1)

	var out []int

	for len(out) < maxNew {
		best, second := top2(row)
		margin = min(margin, float64(row[best]-row[second]))
		out = append(out, best)

		if stop[best] {
			break
		}

		row, err = m.Extend(s, []int{best})
		if err != nil {
			t.Fatal(err)
		}
	}

	txt, err := p.Decode(out)
	if err != nil {
		t.Fatal(err)
	}

	return txt, margin
}

func top2(row []float32) (best, second int) {
	best, second = 0, 1
	if row[second] > row[best] {
		best, second = second, best
	}

	for i := 2; i < len(row); i++ {
		switch {
		case row[i] > row[best]:
			best, second = i, best
		case row[i] > row[second]:
			second = i
		}
	}

	return best, second
}

// parseCalls is the <tool_call>[…]</tool_call> block of a completion; no
// block is no calls.
func parseCalls(completion string) ([]call, error) {
	_, body, ok := strings.Cut(completion, "<tool_call>")
	if !ok {
		return nil, nil
	}

	body, _, ok = strings.Cut(body, "</tool_call>")
	if !ok {
		return nil, errors.New("no </tool_call>")
	}

	var calls []call

	err := json.Unmarshal([]byte(body), &calls)

	return calls, err
}

// diffCalls names how got differs from want: "" when equal, else the
// differences in the names or the argument values.
func diffCalls(got, want []call) string {
	if len(got) != len(want) {
		return fmt.Sprintf("%d calls against %d", len(got), len(want))
	}

	var diffs []string

	for i := range got {
		if got[i].Name != want[i].Name {
			return fmt.Sprintf("call %d: %s against %s", i, got[i].Name, want[i].Name)
		}

		keys := slices.Sorted(func(yield func(string) bool) {
			for k := range got[i].Arguments {
				yield(k)
			}

			for k := range want[i].Arguments {
				if _, ok := got[i].Arguments[k]; !ok {
					yield(k)
				}
			}
		})

		for _, k := range keys {
			g, gok := got[i].Arguments[k]
			w, wok := want[i].Arguments[k]

			switch {
			case !gok:
				diffs = append(diffs, fmt.Sprintf("%s.%s missing (engine %v)", got[i].Name, k, w))
			case !wok:
				diffs = append(diffs, fmt.Sprintf("%s.%s=%v extra", got[i].Name, k, g))
			case !reflect.DeepEqual(g, w):
				diffs = append(diffs, fmt.Sprintf("%s.%s %v, engine %v", got[i].Name, k, g, w))
			}
		}
	}

	return strings.Join(diffs, "; ")
}

// TestFloatGreedyAgainstEngine is the decode half of PLAN.md 11.3's
// Accept, recorded rather than asserted (the user's decision of
// 2026-10-12): each engine case's first turn rendered and decoded greedily
// by the float model, its calls set against the engine's function_calls
// and suppressed_calls. The engine rewrites its calls after decoding and
// suppresses those below confidence 0.1 (11.4's envelope), so the log
// names the differences; the test requires only that every completion
// parses (docs/FINDINGS.md F-048).
func TestFloatGreedyAgainstEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes 19 prompts with the float model")
	}

	p := loadTokenizer(t)

	_, err := os.Stat(needle3Safetensors)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3`", needle3Safetensors)
	}

	src, err := text.Open(needle3Safetensors)
	if err != nil {
		t.Fatal(err)
	}

	m, err := text.Load(src, src.TextConfig())
	if err != nil {
		t.Fatal(err)
	}

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

	var responses map[string][]struct {
		Calls      []call `json:"function_calls"`
		Suppressed []call `json:"suppressed_calls"`
		ErrorCode  string `json:"error_code"`
	}

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineResponse, &responses)

	tally := map[string]int{}

	for _, c := range corpus.Cases {
		system := corpus.System
		if c.System != "" {
			system = c.System
		}

		tools, err := ToolsJSON(corpus.Toolsets[c.Toolset])
		if err != nil {
			t.Fatal(err)
		}

		maxNew := 512 // needle --max's default
		if c.Max > 0 {
			maxNew = c.Max
		}

		completion, margin := greedy(t, m, p, Encode(p, Render(system, tools, c.Query)), maxNew)
		eng := responses[c.ID][0]

		got, err := parseCalls(completion)
		if err != nil {
			if eng.ErrorCode == "truncated" && c.Max > 0 {
				tally["equal"]++

				t.Logf("%-20s margin %6.3f  truncated at %d tokens, as the engine", c.ID, margin, maxNew)

				continue
			}

			t.Errorf("%s: %v in %q", c.ID, err, completion)

			continue
		}

		verdict, key := "equal", "equal"

		switch d := diffCalls(got, eng.Calls); {
		case d == "":
		case len(eng.Calls) == 0 && len(eng.Suppressed) > 0 && diffCalls(got, eng.Suppressed) == "":
			verdict, key = "equal to the suppressed calls", verdict
		case len(eng.Calls) == 0 && len(eng.Suppressed) > 0:
			verdict, key = "against the suppressed calls: "+diffCalls(got, eng.Suppressed), "differs"
		default:
			verdict, key = d, "differs"
		}

		tally[key]++

		t.Logf("%-20s margin %6.3f  %s", c.ID, margin, verdict)
	}

	t.Logf("%d cases: %v", len(corpus.Cases), tally)
}
