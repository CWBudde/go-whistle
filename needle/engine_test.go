package needle

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// The weights and the engine fixtures, relative to the package.
const (
	modelDir      = "../models/needle3"
	engineCases   = "../testdata/text/engine/cases.json"
	engineAnswers = "../testdata/text/engine/responses.json"
	engineHistory = "../testdata/text/engine/history.json"
)

var (
	modelOnce sync.Once
	model     *Model
	errModel  error
)

// requireWeights skips without the weights.
func requireWeights(t *testing.T) {
	t.Helper()

	_, err := os.Stat(modelDir + "/" + cactFile)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3`", modelDir)
	}
}

// testModel is the model in the engine's numerics, Load's default, shared
// by the tests and never closed.
func testModel(t *testing.T) *Model {
	t.Helper()

	requireWeights(t)

	modelOnce.Do(func() { model, errModel = Load(modelDir) })

	if errModel != nil {
		t.Fatal(errModel)
	}

	return model
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	err = json.Unmarshal(data, v)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

// engineCorpus is testdata/text/engine/cases.json.
type engineCorpus struct {
	System   string                     `json:"system"`
	Toolsets map[string]json.RawMessage `json:"toolsets"`
	Cases    []struct {
		ID      string            `json:"id"`
		Toolset string            `json:"toolset"`
		System  string            `json:"system"`
		Query   string            `json:"query"`
		Max     int               `json:"max"`
		Results []json.RawMessage `json:"results"`
	} `json:"cases"`
}

type engineCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// engineEnvelope is an envelope as recorded or as Response marshals it,
// parsed.
type engineEnvelope struct {
	Type       string       `json:"type"`
	Calls      []engineCall `json:"function_calls"`
	Suppressed []engineCall `json:"suppressed_calls"`
	Reasoning  *string      `json:"reasoning"`
	Confidence float64      `json:"confidence"`
}

// sameCalls reports equal calls: the names and every argument's value.
func sameCalls(got, want []engineCall) bool {
	if len(got) != len(want) {
		return false
	}

	for i := range got {
		if got[i].Name != want[i].Name || len(got[i].Arguments) != len(want[i].Arguments) {
			return false
		}

		for k, v := range want[i].Arguments {
			if g, ok := got[i].Arguments[k]; !ok || !reflect.DeepEqual(g, v) {
				return false
			}
		}
	}

	return true
}

// against is how a turn's envelope compares with the engine's, as
// internal/toolcall's TestEnvelopeMatchesEngine and TestHistoryMatchesEngine
// count it.
type against struct {
	typeAndFC bool // type and function_calls
	reasoning bool
	calls     bool // type and the calls, dispatched and suppressed
	conf      float64
}

func compare(t *testing.T, js []byte, want engineEnvelope) against {
	t.Helper()

	var got engineEnvelope

	err := json.Unmarshal(js, &got)
	if err != nil {
		t.Fatal(err)
	}

	return against{
		typeAndFC: got.Type == want.Type && sameCalls(got.Calls, want.Calls),
		reasoning: reflect.DeepEqual(got.Reasoning, want.Reasoning),
		calls: got.Type == want.Type &&
			sameCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...)),
		conf: math.Abs(got.Confidence - want.Confidence),
	}
}

// pipeline answers inputs as one conversation the way internal/toolcall's
// tests compose the pipeline (decodeTurns): Conversation, Decoder and
// Envelope on the model's own weights, each envelope as JSON.
func pipeline(t *testing.T, m *Model, system string, tools []byte, inputs []string, maxNew int) [][]byte {
	t.Helper()

	n, err := toolcall.Normalise(tools)
	if err != nil {
		t.Fatal(err)
	}

	conv := toolcall.NewConversation(m.p, system, n.JSON, m.tm.Config().MaxSeqLen)
	s := m.tm.NewState()

	out := make([][]byte, 0, len(inputs))

	for _, in := range inputs {
		ids, err := conv.Turn(in, maxNew, true)
		if err != nil {
			t.Fatal(err)
		}

		if conv.Restarted() {
			s = m.tm.NewState()
		}

		if s.Len() == 0 {
			ids = append(toolcall.StaticPrefix(m.p, system, n.JSON), ids...)
		}

		a, err := m.d.Decode(s, ids, maxNew, n.GrammarFor(in))
		if err != nil {
			t.Fatal(err)
		}

		conf, err := m.d.Confidence(s)
		if err != nil {
			t.Fatal(err)
		}

		said := conv.Inputs()

		e, err := n.Envelope(a, in, system, said[:len(said)-1], conf)
		if err != nil {
			t.Fatal(err)
		}

		conv.Answered(a, len(e.FunctionCalls))

		js, err := e.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}

		out = append(out, js)
	}

	return out
}

// TestCompleteMatchesPipeline answers every turn of the engine corpus
// through the API, one agent per case, and requires each envelope to be
// the internal pipeline's byte for byte (the API loses nothing), then
// counts them against the engine as TestEnvelopeMatchesEngine does under
// NEEDLE_TEST_NUMERICS=int8: on arm64 (M5 Pro), type and function_calls
// 24 of 25, reasoning 22 of 25, and the confidence within 0.02 on 20 of
// the 22 turns decoded as the engine's (docs/FINDINGS.md F-062, F-064).
func TestCompleteMatchesPipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the engine corpus")
	}

	m := testModel(t)

	var (
		corpus    engineCorpus
		responses map[string][]engineEnvelope
	)

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineAnswers, &responses)

	var typeAndFC, reasoning, decoded, within, total int

	for _, c := range corpus.Cases {
		system := corpus.System
		if c.System != "" {
			system = c.System
		}

		maxNew := DefaultMaxNewTokens
		if c.Max > 0 {
			maxNew = c.Max
		}

		inputs := []string{c.Query}
		for _, r := range c.Results {
			inputs = append(inputs, string(r))
		}

		tools := corpus.Toolsets[c.Toolset]
		want := pipeline(t, m, system, tools, inputs, maxNew)

		a, err := m.NewAgent(tools, AgentOptions{System: system, FailInputOverflow: true})
		if err != nil {
			t.Fatal(err)
		}

		for i, in := range inputs {
			r, err := a.Complete(in, CompleteOptions{MaxNewTokens: maxNew})
			if err != nil {
				t.Fatal(err)
			}

			js, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}

			if !bytes.Equal(js, want[i]) {
				t.Errorf("%s/%d: the API's envelope\n%s\nthe pipeline's\n%s", c.ID, i, js, want[i])
			}

			x := compare(t, js, responses[c.ID][i])
			same := x.calls && x.reasoning
			total++

			for _, f := range []struct {
				ok bool
				n  *int
			}{{x.typeAndFC, &typeAndFC}, {x.reasoning, &reasoning}, {same, &decoded}, {same && x.conf <= 0.02, &within}} {
				if f.ok {
					*f.n++
				}
			}

			t.Logf("%-20s turn %d  type+function_calls %-5v reasoning %-5v confidence off %.4f", c.ID, i, x.typeAndFC,
				x.reasoning, x.conf)
		}
	}

	t.Logf("type and function_calls equal on %d of %d", typeAndFC, total)
	t.Logf("reasoning equal on %d of %d", reasoning, total)
	t.Logf("confidence within 0.02 on %d of %d turns decoded as the engine's", within, decoded)

	// internal/toolcall's int8 counts less one turn for another
	// architecture's rounding at the near-ties.
	if typeAndFC < 23 || reasoning < 21 || within < 19 {
		t.Errorf("below the int8 floors (23, 21 and 19)")
	}
}

// turnText is a recorded input of history.json, or a reset.
type turnText struct {
	Head   string `json:"head"`
	Unit   string `json:"unit"`
	Repeat int    `json:"repeat"`
	Tail   string `json:"tail"`
	Reset  bool   `json:"reset"`
}

func (in turnText) text() string { return in.Head + strings.Repeat(in.Unit, in.Repeat) + in.Tail }

// historyFile is the part of testdata/text/engine/history.json the API
// replays: its conversations.
type historyFile struct {
	System        string `json:"system"`
	Conversations []struct {
		ID      string `json:"id"`
		Group   string `json:"group"`
		Toolset string `json:"toolset"`
		System  string `json:"system"`
		Max     int    `json:"max"`
		Refuse  bool   `json:"fail_input_overflow"`
		Steps   []struct {
			Input  turnText        `json:"input"`
			Max    int             `json:"max"`
			SHA256 string          `json:"sha256"`
			Answer json.RawMessage `json:"answer"`
		} `json:"steps"`
	} `json:"conversations"`
}

// overflowError is the counts in the engine's --fail-input-overflow error.
var overflowError = regexp.MustCompile(`^input exceeds the context window \((\d+) tokens, (\d+) available\)$`)

// answerKind is how an answer ended, for the turn after it.
func answerKind(r *Response) string {
	switch {
	case r.ErrorCode != nil && *r.ErrorCode == "truncated":
		return "truncated"
	case r.Type == "respond":
		return "respond"
	case len(r.FunctionCalls) > 0:
		return "call"
	case len(r.SuppressedCalls) > 0:
		return "suppressed"
	}

	return "empty"
}

// historyTally counts the history's turns as internal/toolcall's
// TestHistoryMatchesEngine does: the later turns (after an answer of the
// conversation) and the cut turns, apart.
type historyTally struct {
	later, think, calls, decoded, within, cut, cutSame int
}

func (h *historyTally) add(x against, cut, later bool) {
	same := x.reasoning && x.calls

	switch {
	case cut:
		h.cut++

		if same {
			h.cutSame++
		}
	case later:
		h.later++

		for _, f := range []struct {
			ok bool
			n  *int
		}{{x.reasoning, &h.think}, {x.calls, &h.calls}, {same, &h.decoded}, {same && x.conf <= 0.02, &h.within}} {
			if f.ok {
				*f.n++
			}
		}
	}
}

// int8Env is the switch of the repository's int8 parity runs
// (internal/toolcall's numericsEnv). The API always runs the engine's
// numerics; set to "int8", TestHistoryMatchesEngine also replays the seven
// long conversations of 13 turns near the context, which take it from
// minutes to half an hour.
const int8Env = "NEEDLE_TEST_NUMERICS"

// TestHistoryMatchesEngine replays every conversation of history.json
// through the API, one agent per conversation (Reset at a reset step, the
// engine's --fail-input-overflow as FailInputOverflow), and counts the
// later and the cut turns against the engine as internal/toolcall's
// TestHistoryMatchesEngine does under NEEDLE_TEST_NUMERICS=int8. Like its
// replayHistory it stops a conversation at the turn that would take it past
// the context: the API answers that turn alone, a port choice the engine's
// answers do not pin (docs/FINDINGS.md F-064), so it is logged, not
// counted. The refused turns carry the engine's numbers. The long
// conversations and the floors need int8Env.
func TestHistoryMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the history probes")
	}

	m := testModel(t)

	var (
		fx     historyFile
		corpus engineCorpus
	)

	readJSON(t, engineHistory, &fx)
	readJSON(t, engineCases, &corpus)

	var (
		h       historyTally
		refused int
		past    []string
	)

	full := os.Getenv(int8Env) == "int8"

	for _, c := range fx.Conversations {
		if c.Group == "long" && !full {
			continue
		}

		system := fx.System
		if c.System != "" {
			system = c.System
		}

		a, err := m.NewAgent(corpus.Toolsets[c.Toolset], AgentOptions{System: system, FailInputOverflow: c.Refuse})
		if err != nil {
			t.Fatal(err)
		}

		sess, _ := a.c.(*session)
		previous := ""

	steps:
		for i, st := range c.Steps {
			key := c.ID + "/" + strconv.Itoa(i)

			if st.Input.Reset {
				a.Reset()

				previous = ""

				continue
			}

			in := st.Input.text()
			if sum := sha256.Sum256([]byte(in)); hex.EncodeToString(sum[:]) != st.SHA256 {
				t.Fatalf("%s: the rebuilt input is not the engine's", key)
			}

			maxNew := c.Max
			if st.Max > 0 {
				maxNew = st.Max
			}

			var failed struct {
				Error string `json:"error"`
			}

			_ = json.Unmarshal(st.Answer, &failed)

			dropped := sess.dropped
			r, err := a.Complete(in, CompleteOptions{MaxNewTokens: maxNew})

			if mm := overflowError.FindStringSubmatch(failed.Error); mm != nil {
				refused++

				var oe *OverflowError
				if !errors.As(err, &oe) || !errors.Is(err, ErrContextOverflow) ||
					strconv.Itoa(oe.Tokens) != mm[1] || strconv.Itoa(oe.Available) != mm[2] {
					t.Errorf("%s: engine refused %s, the API: %v", key, mm[0], err)
				}

				continue
			}

			if err != nil {
				t.Fatalf("%s: engine answered, the API: %v", key, err)
			}

			var want engineEnvelope

			err = json.Unmarshal(st.Answer, &want)
			if err != nil {
				t.Fatal(err)
			}

			js, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}

			x := compare(t, js, want)

			if sess.dropped != dropped {
				t.Logf("%s: past the context, answered alone: reasoning %v, calls %v, confidence off %.4f: %s", key,
					x.reasoning, x.calls, x.conf, js)

				past = append(past, key)

				break steps
			}

			t.Logf("%-32s after %-10s reasoning %-5v calls %-5v confidence off %.4f", key, previous, x.reasoning,
				x.calls, x.conf)

			h.add(x, c.Group == "cut" && i == len(c.Steps)-1, previous != "")

			previous = answerKind(r)
		}
	}

	t.Logf("refused turns with the engine's numbers: %d; past the context, answered alone: %v", refused, past)
	t.Logf("later turns with the engine's reasoning: %d of %d", h.think, h.later)
	t.Logf("later turns with the engine's calls: %d of %d", h.calls, h.later)
	t.Logf("later turns decoded as the engine's: %d of %d", h.decoded, h.later)
	t.Logf("confidence within 0.02 of the later turns decoded as the engine's: %d of %d", h.within, h.decoded)
	t.Logf("cut turns decoded as the engine's: %d of %d", h.cutSame, h.cut)

	if refused != 8 {
		t.Errorf("%d refusals, want 8", refused)
	}

	// The floors: the counts measured on arm64 (M5 Pro) less two, those of
	// internal/toolcall's int8 run with the long conversations, without
	// them reasoning 27 and calls 71 of 92, the confidence 14 of 22.
	floors := [3]int{25, 69, 12}
	if full {
		floors = [3]int{50, 110, 35}
	}

	if h.think < floors[0] || h.calls < floors[1] || h.within < floors[2] {
		t.Errorf("below the floors (reasoning %d, calls %d, confidence %d)", floors[0], floors[1], floors[2])
	}

	if !full {
		t.Logf("the long conversations run with %s=int8", int8Env)

		return
	}

	if len(past) != 4 {
		t.Errorf("%d conversations past the context, want 4", len(past))
	}
}

// TestConcurrentAgents: agents of one Model answer at the same time as
// they answer one after another, byte for byte. Run it with -race: the
// agents share the weights and their kernels' scratch pools, the tokenizer
// and the decoder. The tool set is small and the answers short (truncated),
// which keeps the run under the race detector to minutes; the decoder's
// vocabulary mask, which only a call block reads, is not written after
// NewDecoder, and each agent's grammar guards its own cache.
func TestConcurrentAgents(t *testing.T) {
	m := testModel(t)

	tools := []byte(`[{"name":"set_timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer"}},` +
		`"required":["minutes"]}}]`)
	queries := []string{"set a timer for 5 minutes", "timer 20 min", "what is the weather"}

	answer := func(q string) []byte {
		a, err := m.NewAgent(tools, AgentOptions{})
		if err != nil {
			t.Error(err)

			return nil
		}

		r, err := a.Complete(q, CompleteOptions{MaxNewTokens: 6})
		if err != nil {
			t.Error(err)

			return nil
		}

		js, err := json.Marshal(r)
		if err != nil {
			t.Error(err)
		}

		return js
	}

	want := make([][]byte, len(queries))
	for i, q := range queries {
		want[i] = answer(q)
	}

	got := make([][]byte, len(queries))

	var wg sync.WaitGroup

	for i, q := range queries {
		wg.Go(func() { got[i] = answer(q) })
	}

	wg.Wait()

	for i, q := range queries {
		if !bytes.Equal(got[i], want[i]) {
			t.Errorf("%q: concurrently %s, alone %s", q, got[i], want[i])
		}
	}
}

// TestModelErrors: a missing directory, a turn the engine refuses, tools
// whose rendering fills the context, and a closed model.
func TestModelErrors(t *testing.T) {
	_, err := Load(t.TempDir())
	if err == nil {
		t.Error("Load of an empty directory succeeded")
	}

	requireWeights(t)

	m, err := Load(modelDir, WithoutInt8())
	if err != nil {
		t.Fatal(err)
	}

	tools := []byte(`[{"name":"set_timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer"}}}}]`)

	a, err := m.NewAgent(tools, AgentOptions{FailInputOverflow: true})
	if err != nil {
		t.Fatal(err)
	}

	_, err = a.Complete(strings.Repeat("timer ", 8000), CompleteOptions{})

	room := m.context() - len(a.c.(*session).prefix) - DefaultMaxNewTokens

	var oe *OverflowError
	if !errors.As(err, &oe) || !errors.Is(err, ErrContextOverflow) || oe.Available != room {
		t.Errorf("an oversized turn: %v, want the room %d", err, room)
	}

	huge := `[{"name":"t","description":"` + strings.Repeat("word ", 9000) + `"}]`

	_, err = m.NewAgent([]byte(huge), AgentOptions{})
	if !errors.Is(err, ErrContextBudget) {
		t.Errorf("tools past the context: %v", err)
	}

	err = m.Close()
	if err != nil {
		t.Fatal(err)
	}

	_, err = a.Complete("set a timer", CompleteOptions{})
	if !errors.Is(err, ErrClosed) {
		t.Errorf("Complete after Close: %v", err)
	}

	_, err = m.NewAgent(tools, AgentOptions{})
	if !errors.Is(err, ErrClosed) {
		t.Errorf("NewAgent after Close: %v", err)
	}

	err = m.Close()
	if err != nil {
		t.Errorf("a second Close: %v", err)
	}
}
