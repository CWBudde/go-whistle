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

// loadModel loads the text model at path, skipping without it. The
// switch NEEDLE_TEST_NUMERICS loads the packed needle3.cact instead, in
// float32 or the engine's int8 numerics (numericsEnv, docs/FINDINGS.md
// F-060).
func loadModel(t *testing.T, path string) *text.Model {
	t.Helper()

	if n := testNumerics(t); n != numericsFloat {
		return loadNumerics(t, n)
	}

	requireModel(t, path)

	src, err := text.Open(path)
	if err != nil {
		t.Fatal(err)
	}

	m, err := text.Load(src, src.TextConfig())
	if err != nil {
		t.Fatal(err)
	}

	return m
}

// requireModel skips without the model file at path.
func requireModel(t *testing.T, path string) {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3`", path)
	}
}

// engineCase is one case of the engine corpus (testdata/text/engine).
type engineCase struct {
	ID      string            `json:"id"`
	Toolset string            `json:"toolset"`
	System  string            `json:"system"`
	Query   string            `json:"query"`
	Max     int               `json:"max"`
	Results []json.RawMessage `json:"results"`
}

type engineCorpus struct {
	System   string                     `json:"system"`
	Toolsets map[string]json.RawMessage `json:"toolsets"`
	Cases    []engineCase               `json:"cases"`
}

// engineAnswer is one turn's envelope of responses.json.
type engineAnswer struct {
	Reasoning  *string `json:"reasoning"`
	Calls      []call  `json:"function_calls"`
	Suppressed []call  `json:"suppressed_calls"`
	ErrorCode  string  `json:"error_code"`
}

// turnOut is one decoded turn: the answer, its confidence and its envelope.
type turnOut struct {
	a    *Decoded
	conf float64
	env  *Envelope
}

// decodeTurns decodes the conversation inputs (a query, then tool results)
// over the tools raw as the engine does, each answer under the grammar of
// its input (docs/FINDINGS.md F-055) and its shipped calls deciding whether
// the next input is a tool result (docs/FINDINGS.md F-051).
func decodeTurns(t *testing.T, d *Decoder, m *text.Model, p *spm.Processor, system string, raw []byte,
	inputs []string, maxNew int,
) []turnOut {
	t.Helper()

	n, err := Normalise(raw)
	if err != nil {
		t.Fatal(err)
	}

	conv := NewConversation(p, system, n.JSON, m.Config().MaxSeqLen)
	s := m.NewState()

	out := make([]turnOut, 0, len(inputs))

	for i, in := range inputs {
		ids, err := conv.Turn(in, maxNew, true)
		if err != nil {
			t.Fatal(err)
		}

		if i == 0 {
			ids = append(StaticPrefix(p, system, n.JSON), ids...)
		}

		a, err := d.Decode(s, ids, maxNew, n.GrammarFor(in))
		if err != nil {
			t.Fatal(err)
		}

		conf, err := d.Confidence(s)
		if err != nil {
			t.Fatal(err)
		}

		said := conv.Inputs()

		env, err := n.Envelope(a, in, system, said[:len(said)-1], conf)
		if err != nil {
			t.Fatal(err)
		}

		conv.Answered(len(env.FunctionCalls))

		out = append(out, turnOut{a: a, conf: conf, env: env})
	}

	return out
}

// decodeCase decodes every turn of c as the engine does: the query, then
// each recorded tool result after the answer before it.
func decodeCase(t *testing.T, d *Decoder, m *text.Model, p *spm.Processor, corpus *engineCorpus, c engineCase) []turnOut {
	t.Helper()

	system := corpus.System
	if c.System != "" {
		system = c.System
	}

	maxNew := 512 // needle --max's default
	if c.Max > 0 {
		maxNew = c.Max
	}

	inputs := make([]string, 0, 1+len(c.Results))
	inputs = append(inputs, c.Query)

	for _, r := range c.Results {
		inputs = append(inputs, string(r))
	}

	return decodeTurns(t, d, m, p, system, corpus.Toolsets[c.Toolset], inputs, maxNew)
}

// thinkOf is the reasoning between <think>\n and \n</think>.
func thinkOf(answer string) string {
	_, rest, _ := strings.Cut(answer, thinkOpen)
	r, _, _ := strings.Cut(rest, "\n</think>")

	return r
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

// minMargin is the smallest top-1 margin of the tokens in [from, to).
func minMargin(a *Decoded, from, to int) float64 {
	m := math.Inf(1)
	for _, x := range a.Margins[from:min(to, len(a.Margins))] {
		m = min(m, x)
	}

	return m
}

// Turns of the engine corpus whose decode differs from the engine's, and
// why (docs/FINDINGS.md F-052).
var (
	// The reasoning differs at a near-tie: float activations against the
	// engine's int8 ones (11.5).
	reasoningNearTie = map[string]bool{"alarm_morning": true, "event_no_year": true, "catalogue_reminder": true}
	// The reasoning and the call are the engine's, the token count is not:
	// the engine ends the same call one token later, which neither the
	// schema grammar nor a near-tie explains (docs/FINDINGS.md F-055).
	countOff = map[string]bool{"thermostat_decimal": true}
)

// TestDecodeMatchesEngine decodes every turn of the engine corpus with the
// dequantised needle3.cact as the engine does (docs/FINDINGS.md F-052,
// F-055): the forced "<think>\n", greedy, the call block under the schema
// grammar, </tool_call> counted, <|im_end|> not, --max counting the forced
// tokens, and the history closed with <|im_end|>. Each
// turn's reasoning is the engine's and its generated-token count is
// needle_complete's, but on the turns listed above; the truncated turn ends
// at the budget as the engine's.
func TestDecodeMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the engine corpus")
	}

	p := loadTokenizer(t)
	m := loadModel(t, needle3Cact)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var (
		corpus    engineCorpus
		responses map[string][]engineAnswer
		library   struct {
			Cases map[string]struct {
				Complete []int `json:"complete"`
			} `json:"cases"`
		}
	)

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineResponse, &responses)
	readJSON(t, engineLibrary, &library)

	var equal, total int

	for _, c := range corpus.Cases {
		for i, turn := range decodeCase(t, d, m, p, &corpus, c) {
			a := turn.a
			eng := responses[c.ID][i]
			want := library.Cases[c.ID].Complete[i]

			reasoning := ""
			if eng.Reasoning != nil {
				reasoning = *eng.Reasoning
			}

			think := thinkOf(a.Text) == reasoning
			margin := minMargin(a, 0, len(a.Tokens))
			total++

			if think {
				equal++
			}

			t.Logf("%-20s turn %d  tokens %3d engine %3d  truncated %-5v  reasoning equal %-5v  min margin %.3f",
				c.ID, i, len(a.Tokens), want, a.Truncated, think, margin)

			switch {
			case eng.ErrorCode == "truncated":
				if !a.Truncated || len(a.Tokens) != want {
					listedf(t, "%s: %d tokens, truncated %v; the engine truncates at %d", c.ID, len(a.Tokens), a.Truncated,
						want)
				}
			case reasoningNearTie[c.ID]:
				if think || margin > 0.06 {
					listedf(t, "%s: listed as a near-tie, but reasoning equal %v at margin %.3f", c.ID, think, margin)
				}
			case !think:
				listedf(t, "%s turn %d: reasoning %q, engine %q", c.ID, i, thinkOf(a.Text), reasoning)
			case countOff[c.ID] == (len(a.Tokens) == want):
				listedf(t, "%s turn %d: %d tokens, engine %d (listed: %v)", c.ID, i, len(a.Tokens), want, countOff[c.ID])
			}
		}
	}

	int8Floor(t, "reasoning equal", equal, total, int8DecodeReasoning)
}

// TestFloatGreedyAgainstEngine is the decode half of PLAN.md 11.3's
// Accept, recorded rather than asserted (the user's decision of
// 2026-10-12): each engine case's first turn rendered and decoded greedily
// by the float model, and the calls of its envelope (repaired as the
// engine does, dispatched and suppressed) set against the engine's. The
// log names the differences; the test requires only that every answer
// builds an envelope (docs/FINDINGS.md F-048, F-056).
func TestFloatGreedyAgainstEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the engine corpus with the float model")
	}

	p := loadTokenizer(t)
	m := loadModel(t, needle3Safetensors)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var (
		corpus    engineCorpus
		responses map[string][]engineAnswer
	)

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineResponse, &responses)

	tally := map[string]int{}

	for _, c := range corpus.Cases {
		c.Results = nil // the first turn only
		turn := decodeCase(t, d, m, p, &corpus, c)[0]
		a, eng := turn.a, responses[c.ID][0]
		margin := minMargin(a, 0, len(a.Tokens))

		if a.Truncated {
			verdict, key := "truncated at the budget, as the engine", "equal"
			if eng.ErrorCode != "truncated" {
				verdict, key = "truncated, the engine is not", "differs"
			}

			tally[key]++

			t.Logf("%-20s margin %6.3f  %s (%d tokens)", c.ID, margin, verdict, len(a.Tokens))

			continue
		}

		js, err := json.Marshal(turn.env)
		if err != nil {
			t.Fatal(err)
		}

		var got engineEnvelope

		err = json.Unmarshal(js, &got)
		if err != nil {
			t.Fatal(err)
		}

		verdict, key := "equal", "equal"
		if d := diffCalls(append(got.Calls, got.Suppressed...), append(eng.Calls, eng.Suppressed...)); d != "" {
			verdict, key = d, "differs"
		}

		tally[key]++

		t.Logf("%-20s margin %6.3f  %s", c.ID, margin, verdict)
	}

	t.Logf("%d cases: %v", len(corpus.Cases), tally)
}
