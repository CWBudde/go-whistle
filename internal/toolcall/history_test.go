package toolcall

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/text"
)

// engineHistory is the fixture of `just oracle-text history`.
const engineHistory = "../../testdata/text/engine/history.json"

// turnText is a recorded input as Head + Unit×Repeat + Tail (tools/oracle/text
// turns.go), or a reset.
type turnText struct {
	Head   string `json:"head"`
	Unit   string `json:"unit"`
	Repeat int    `json:"repeat"`
	Tail   string `json:"tail"`
	Reset  bool   `json:"reset"`
}

func (in turnText) text() string { return in.Head + strings.Repeat(in.Unit, in.Repeat) + in.Tail }

// historyStep is one recorded turn of history.json.
type historyStep struct {
	Input     turnText        `json:"input"`
	Max       int             `json:"max"`
	SHA256    string          `json:"sha256"`
	Answer    json.RawMessage `json:"answer"`
	Generated *int            `json:"generated"`
}

// historyConv is one recorded conversation of history.json.
type historyConv struct {
	ID      string        `json:"id"`
	Group   string        `json:"group"`
	Covers  string        `json:"covers"`
	Toolset string        `json:"toolset"`
	System  string        `json:"system"`
	Library bool          `json:"library"`
	Max     int           `json:"max"`
	Refuse  bool          `json:"fail_input_overflow"`
	Steps   []historyStep `json:"steps"`
}

// historyFile is testdata/text/engine/history.json (tools/oracle/text
// history.go).
type historyFile struct {
	System  string `json:"system"`
	Context int    `json:"context"`
	Wire    []struct {
		ID      string        `json:"id"`
		Escaped string        `json:"escaped"`
		Max     int           `json:"max"`
		Steps   []historyStep `json:"steps"`
	} `json:"wire"`
	Conversations []historyConv `json:"conversations"`
	Scans         []struct {
		ID      string     `json:"id"`
		Toolset string     `json:"toolset"`
		Refuse  bool       `json:"fail_input_overflow"`
		Inputs  []turnText `json:"inputs"`
		Runs    []struct {
			Max     int               `json:"max"`
			Answers []json.RawMessage `json:"answers"`
		} `json:"runs"`
	} `json:"scans"`
}

// checkSHA fails when a rebuilt input is not the text the oracle sent.
func checkSHA(t *testing.T, at, text, want string) {
	t.Helper()

	if sum := sha256.Sum256([]byte(text)); hex.EncodeToString(sum[:]) != want {
		t.Fatalf("%s: the rebuilt input is not the engine's", at)
	}
}

// asRead is the input the engine reads from a request body that spells
// the runes of escaped as JSON \u escapes: its request parser drops the
// escape's backslash and keeps the four hex digits as text
// (docs/FINDINGS.md F-063). Go's json.Marshal, which the turns probes were
// sent with, escapes '<', '&' and '>' that way.
func asRead(input, escaped string) string {
	var b strings.Builder

	for _, r := range input {
		if strings.ContainsRune(escaped, r) {
			fmt.Fprintf(&b, "u%04x", r)

			continue
		}

		b.WriteRune(r)
	}

	return b.String()
}

// goEscaped are the runes json.Marshal sends as \u escapes in a string:
// '<', '&', '>', U+2028 and U+2029.
var goEscaped = "<&>" + string(rune(0x2028)) + string(rune(0x2029))

// TestHistoryWire: the engine counts a tool result or a user turn as the
// port renders it when the request body sends it unescaped, and as the
// text with each \u escape's backslash dropped when the body escapes a
// rune (docs/FINDINGS.md F-063). Every wire probe the engine refused has
// the port's count of the input as read.
func TestHistoryWire(t *testing.T) {
	p := loadTokenizer(t)

	var (
		fx     historyFile
		corpus struct {
			Toolsets map[string]json.RawMessage `json:"toolsets"`
		}
	)

	readJSON(t, engineHistory, &fx)
	readJSON(t, engineCases, &corpus)

	tools, err := ToolsJSON(corpus.Toolsets["assistant"])
	if err != nil {
		t.Fatal(err)
	}

	refused := 0

	for _, w := range fx.Wire {
		c := NewConversation(p, fx.System, tools, fx.Context)

		for i, st := range w.Steps {
			at := w.ID + " step " + strconv.Itoa(i)
			text := st.Input.text()
			checkSHA(t, at, text, st.SHA256)

			var a struct {
				Error string `json:"error"`
				Calls []call `json:"function_calls"`
			}

			err := json.Unmarshal(st.Answer, &a)
			if err != nil {
				t.Fatal(err)
			}

			ids, err := c.Turn(asRead(text, w.Escaped), w.Max, true)

			if m := overflowError.FindStringSubmatch(a.Error); m != nil {
				refused++

				checkRefusal(t, at, m, len(ids), err)

				continue
			}

			if err != nil {
				t.Errorf("%s: engine answered, Turn: %v", at, err)
			}

			c.Answered(nil, len(a.Calls))
		}
	}

	// One probe, the control user_A_plain, fits the room.
	if refused != len(fx.Wire)-1 {
		t.Errorf("%d refused wire probes, want %d", refused, len(fx.Wire)-1)
	}

	t.Logf("%d wire probes, %d refusals with the engine's numbers", len(fx.Wire), refused)
}

// historyTurn is one replayed turn of a history conversation, against the
// engine's answer.
type historyTurn struct {
	key      string
	previous string // how the answer before it ended: call, respond, truncated, … ("" for a first turn)
	toolTurn bool   // a tool result after a call
	cut      bool   // an oversized turn the engine cut
	same     bool   // type, reasoning and calls (dispatched and suppressed) equal
	think    bool   // reasoning equal
	calls    bool   // type and calls equal
	conf     float64
	engine   float64
	margin   float64 // the margin where the reasoning leaves the engine's, +Inf when it does not
	counted  *bool   // the token count equals needle_complete's (library conversations)
}

// answerKind is how an answer ended, for the turn after it.
func answerKind(a *Decoded, e *Envelope) string {
	switch {
	case a.Truncated:
		return "truncated"
	case e.Type == "respond":
		return "respond"
	case len(e.FunctionCalls) > 0:
		return "call"
	case len(e.SuppressedCalls) > 0:
		return "suppressed"
	}

	return "empty"
}

// replayHistory replays one conversation of history.json through
// Conversation, Decoder and Envelope. It stops before a turn that would take
// the conversation past the model's context, which the port cannot feed
// (the engine's drop rule is not pinned, docs/FINDINGS.md F-063), and
// reports it.
func replayHistory(t *testing.T, d *Decoder, m *text.Model, p *spm.Processor, fx *historyFile, tools []byte,
	c historyConv,
) (turns []historyTurn, refused int, past string) {
	t.Helper()

	system := fx.System
	if c.System != "" {
		system = c.System
	}

	n, err := Normalise(tools)
	if err != nil {
		t.Fatal(err)
	}

	conv := NewConversation(p, system, n.JSON, fx.Context)
	s := m.NewState()
	previous := ""

	for i, st := range c.Steps {
		key := c.ID + "/" + strconv.Itoa(i)

		if st.Input.Reset {
			conv.Reset()

			s, previous = m.NewState(), ""

			continue
		}

		in := st.Input.text()
		checkSHA(t, key, in, st.SHA256)

		maxNew := c.Max
		if st.Max > 0 {
			maxNew = st.Max
		}

		var want engineEnvelope

		err := json.Unmarshal(st.Answer, &want)
		if err != nil {
			t.Fatal(err)
		}

		var failed struct {
			Error string `json:"error"`
		}

		_ = json.Unmarshal(st.Answer, &failed)

		head, _, _ := conv.render(in)
		toolTurn := strings.HasPrefix(head, "\n"+imStart+"tool\n")

		ids, err := conv.Turn(in, maxNew, c.Refuse)

		if mm := overflowError.FindStringSubmatch(failed.Error); mm != nil {
			refused++

			checkRefusal(t, key, mm, len(ids), err)

			continue
		}

		if err != nil {
			t.Fatalf("%s: engine answered, Turn: %v", key, err)
		}

		if conv.Restarted() {
			s = m.NewState()
		}

		if s.Len() == 0 {
			ids = append(StaticPrefix(p, system, n.JSON), ids...)
		}

		if s.Len()+len(ids) > fx.Context {
			return turns, refused, fmt.Sprintf("%s: %d positions", key, s.Len()+len(ids))
		}

		a, err := d.Decode(s, ids, maxNew, n.GrammarFor(in))
		if errors.Is(err, text.ErrTooLong) {
			return turns, refused, key + ": the answer passes the context"
		}

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

		conv.Answered(a, len(env.FunctionCalls))

		js, err := json.Marshal(env)
		if err != nil {
			t.Fatal(err)
		}

		var got engineEnvelope

		err = json.Unmarshal(js, &got)
		if err != nil {
			t.Fatal(err)
		}

		think := reflect.DeepEqual(got.Reasoning, want.Reasoning)
		calls := got.Type == want.Type &&
			diffCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...)) == ""

		h := historyTurn{
			key: key, previous: previous, toolTurn: toolTurn, cut: c.Group == "cut" && i == len(c.Steps)-1,
			same: think && calls, think: think, calls: calls, conf: got.Confidence, engine: want.Confidence,
			margin: math.Inf(1),
		}

		if !think {
			ref := p.Encode(thinkOpen + derefString(want.Reasoning) + "\n</think>")
			for k := range min(len(ref), len(a.Tokens)) {
				if ref[k] != a.Tokens[k] {
					h.margin = a.Margins[k]

					break
				}
			}
		}

		if st.Generated != nil && h.same {
			ok := len(a.Tokens) == *st.Generated
			h.counted = &ok
		}

		turns = append(turns, h)
		previous = answerKind(a, env)
	}

	return turns, refused, ""
}

// historyMiss names why a later or cut turn differs from the engine's, ""
// when it does not: earlier tells whether every earlier turn of the
// conversation decoded as the engine's.
func historyMiss(h historyTurn, earlier bool) string {
	switch {
	case h.same && math.Abs(h.conf-h.engine) <= 0.02:
		return ""
	case !earlier:
		return "an earlier turn differs"
	case h.margin < historyNearTie:
		return "a near-tie"
	case h.previous == "truncated":
		return "after a truncated answer"
	case h.toolTurn:
		return "a tool result after a call"
	case h.same:
		return "the confidence"
	}

	return "unexplained"
}

// historyNearTie is the margin below which a reasoning that leaves the
// engine's counts as a near-tie: float32 or the port's int8 sums against
// the engine's (docs/FINDINGS.md F-060).
const historyNearTie = 0.1

// historyDiffer are the turns of history.json that differ from the
// engine's in float32 for none of the reasons TestHistoryMatchesEngine
// names itself, and why (docs/FINDINGS.md F-063). Keys are
// "conversation/turn". A cut turn's answer degenerates where the model
// lost the request: a list of guessed calls, suppressed at a confidence
// near 0; the engine's and the port's guesses differ.
var historyDiffer = map[string]string{
	"alarm_change/2":       "after a respond answer to a tool result: the decode probability 0.576 against 0.676",
	"result_thanks_same/2": "the engine's 0.0947 withholds the call the port dispatches at 0.510, at the engine's reasoning",
	"refusal_then_timer/1": "after an empty call block: the confidence 0.444 against 0.473",
	"weather_message/2":    "after a respond answer: the message's arguments differ at the engine's reasoning",
	"long_turn_fact/1":     "the reasoning leaves the engine's at a margin of 0.337 after a turn of filler",
	"cut_first_120_-8/0":   "a cut first turn: both answers degenerate (0.0009, engine 0.0016), other guesses",
	"cut_first_120_-4/0":   "a cut first turn decoded as the engine's: the confidence 0.327 against 0.306",
	"cut_first_120_-3/0":   "a cut first turn decoded as the engine's: the confidence 0.419 against 0.373",
	"cut_first_120_+1/0":   "a cut first turn: both answers degenerate (0.0011, engine 0.0012), other guesses",
	"cut_first_120_+2/0":   "a cut first turn decoded as the engine's: the confidence 0.416 against 0.385",
	"cut_first_120_+3/0":   "a cut first turn: both answers degenerate (0.0008, engine 0.0020), other guesses",
	"cut_first_120_+12/0":  "a cut first turn: both answers degenerate (0.0015, engine 0.0010), other guesses",
	"cut_later_120_-6/1":   "a cut later turn: both answers degenerate (0.0396, engine 0.0046), other guesses",
	"cut_later_120_-2/1":   "a cut later turn: the engine's answer degenerates (0.0041), the port's answers the request (0.4318)",
	"cut_later_120_-1/1":   "a cut later turn: the engine's answer degenerates (0.0008), the port's answers the request (0.2906)",
	"cut_later_120_+0/1":   "a cut later turn: the engine's answer degenerates (0.0012), the port's answers the request (0.3948)",
	"cut_later_120_+3/1":   "a cut later turn: both answers degenerate (0.0025, engine 0.0013), other guesses",
	"cut_first_200_-8/0":   "a cut first turn: both answers degenerate (0.0331, engine 0.0317), other guesses",
	"cut_first_200_-4/0":   "a cut first turn decoded as the engine's: the confidence 0.503 against 0.482",
	"cut_later_200_-4/1":   "a cut later turn: both answers degenerate (0.0040, engine 0.0029), other guesses",
	"cut_later_200_-2/1":   "a cut later turn: both answers degenerate (0.0011, engine 0.0010), other guesses",
	"cut_later_200_+0/1":   "a cut later turn: both answers degenerate (0.0010, engine 0.0012), other guesses",
	"cut_later_200_+1/1":   "a cut later turn: both answers degenerate (0.0022, engine 0.0012), other guesses",
	"cut_later_200_+3/1":   "a cut later turn: both answers degenerate (0.0009, engine 0.0008), other guesses",
	"cut_later_200_+8/1":   "a cut later turn: both answers degenerate (0.0116, engine 0.0010), other guesses",
	"cut_later_200_+12/1":  "a cut later turn: both answers degenerate (0.0014, engine 0.0024), other guesses",
}

// TestHistoryMatchesEngine replays every conversation of the history probe
// set (testdata/text/engine/history.json, `just oracle-text history`)
// through Conversation, Decoder and Envelope with the dequantised
// needle3.cact under the grammar (docs/FINDINGS.md F-063): the turns the
// engine refused have its numbers, and the later turns (and the turns the
// engine cut) are counted with the engine's reasoning, calls and, among
// those decoded as the engine's, confidence within 0.02. A turn that
// differs is logged with its reason: an earlier turn of the conversation
// differs, its reasoning leaves the engine's at a near-tie, it follows a
// truncated answer (the engine's decode under a small limit is not the
// port's, F-063's Open), or it is a tool result after a call (the closing
// there is F-052's, which F-063 keeps against the better aggregate fit of
// the closing as trained). Any other differing turn must be listed in
// historyDiffer. Turns that would take the conversation past the context
// stop the replay: the engine's drop rule is not pinned.
func TestHistoryMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the history probes")
	}

	p := loadTokenizer(t)
	m := loadModel(t, needle3Cact)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var (
		fx     historyFile
		corpus engineCorpus
	)

	readJSON(t, engineHistory, &fx)
	readJSON(t, engineCases, &corpus)

	var (
		later, think, calls, decoded, within int
		cut, cutSame                         int
		refused, counted, countTotal         int
		past                                 []string
		reasons                              = map[string]int{}
	)

	for _, c := range fx.Conversations {
		turns, r, stop := replayHistory(t, d, m, p, &fx, corpus.Toolsets[c.Toolset], c)
		refused += r

		if stop != "" {
			past = append(past, stop)
		}

		earlier := true

		for _, h := range turns {
			off := math.Abs(h.conf - h.engine)

			t.Logf("%-32s after %-10s tool %-5v reasoning %-5v calls %-5v margin %6.3f confidence %.4f engine %.4f",
				h.key, h.previous, h.toolTurn, h.think, h.calls, h.margin, h.conf, h.engine)

			if h.counted != nil {
				countTotal++

				if *h.counted {
					counted++
				}
			}

			switch {
			case h.cut:
				cut++

				if h.same {
					cutSame++
				}
			case h.previous != "":
				later++

				for _, x := range []struct {
					ok bool
					n  *int
				}{{h.think, &think}, {h.calls, &calls}, {h.same, &decoded}, {h.same && off <= 0.02, &within}} {
					if x.ok {
						*x.n++
					}
				}
			}

			if h.previous == "" && !h.cut {
				earlier = earlier && h.same

				continue
			}

			why := historyMiss(h, earlier)

			earlier = earlier && h.same
			listed, isListed := historyDiffer[h.key]

			if why == "" {
				if isListed {
					listedf(t, "%s: listed (%q) but equal to the engine's", h.key, listed)
				}

				continue
			}

			reasons[why]++

			switch {
			case why == "unexplained" || why == "the confidence":
				if !isListed {
					listedf(t, "%s: %s (reasoning %v, calls %v, confidence %.4f engine %.4f), not listed", h.key, why,
						h.think, h.calls, h.conf, h.engine)
				}
			case isListed:
				listedf(t, "%s: listed (%q) but %s", h.key, listed, why)
			}
		}
	}

	for key := range historyDiffer {
		found := false

		for _, c := range fx.Conversations {
			found = found || strings.HasPrefix(key, c.ID+"/")
		}

		if !found {
			t.Errorf("historyDiffer lists %s, which is no turn", key)
		}
	}

	t.Logf("refused turns with the engine's numbers: %d; past the context, not replayed: %v", refused, past)
	t.Logf("misses by reason: %v", reasons)
	int8Floor(t, "later turns with the engine's reasoning", think, later, int8HistoryReasoning)
	int8Floor(t, "later turns with the engine's calls", calls, later, int8HistoryCalls)
	logCount(t, "later turns decoded as the engine's", decoded, later)
	int8Floor(t, "confidence within 0.02 of the later turns decoded as the engine's", within, decoded,
		int8HistoryConfidence)
	logCount(t, "cut turns decoded as the engine's", cutSame, cut)
	logCount(t, "token counts equal needle_complete's", counted, countTotal)

	if refused != historyRefused {
		t.Errorf("%d refusals, want %d", refused, historyRefused)
	}
}

// historyRefused are the turns of history.json the engine refused with
// --fail-input-overflow.
const historyRefused = 8

// The floors of TestHistoryMatchesEngine under the int8 numerics: the
// counts measured on arm64 (M5 Pro) less two, as numericsEnv's floors.
// Float32 gives reasoning 28 and calls 76 of 153 later turns, and the
// confidence within 0.02 on 13 of the 24 decoded as the engine's.
const (
	int8HistoryReasoning  = 50  // 52 of 154
	int8HistoryCalls      = 110 // 112 of 154
	int8HistoryConfidence = 35  // 37 of the 47 decoded as the engine's
)

// TestHistoryScans: the history does not count against --max. The engine
// answers a conversation the same at both ends of the scan, where its turns
// fit alone although its history and --max together pass the context, and
// Turn accepts every turn at each of them (docs/FINDINGS.md F-063).
func TestHistoryScans(t *testing.T) {
	p := loadTokenizer(t)

	var (
		fx     historyFile
		corpus engineCorpus
	)

	readJSON(t, engineHistory, &fx)
	readJSON(t, engineCases, &corpus)

	if len(fx.Scans) == 0 {
		t.Fatal("no scans")
	}

	for _, sc := range fx.Scans {
		last := 0

		n, err := Normalise(corpus.Toolsets[sc.Toolset])
		if err != nil {
			t.Fatal(err)
		}

		for _, run := range sc.Runs {
			if !slices.EqualFunc(run.Answers, sc.Runs[0].Answers, func(a, b json.RawMessage) bool {
				return string(a) == string(b)
			}) {
				t.Errorf("%s: the engine answers otherwise at --max %d than at %d", sc.ID, run.Max, sc.Runs[0].Max)
			}

			c := NewConversation(p, fx.System, n.JSON, fx.Context)
			fed := len(StaticPrefix(p, fx.System, n.JSON))

			for i, in := range sc.Inputs {
				ids, err := c.Turn(in.text(), run.Max, sc.Refuse)
				if err != nil {
					t.Errorf("%s at --max %d, turn %d: %v", sc.ID, run.Max, i, err)
				}

				fed += len(ids)

				var a struct {
					Calls []call `json:"function_calls"`
				}

				_ = json.Unmarshal(run.Answers[i], &a)
				c.Answered(nil, len(a.Calls))
			}

			t.Logf("%s at --max %d: the static prefix and the turns take %d positions, with --max %d of the %d of "+
				"the context", sc.ID, run.Max, fed, fed+run.Max, fx.Context)

			last = fed + run.Max
		}

		// Even without the answers, the history and the largest --max pass the
		// context, and the engine still answers as at the smallest.
		if last <= fx.Context {
			t.Errorf("%s: the scan's largest --max leaves the history %d positions of the context", sc.ID, last)
		}
	}
}

// TestHistoryIDs: every probe of history.json has its own id, which keys
// its turns in the logs and in historyDiffer.
func TestHistoryIDs(t *testing.T) {
	var fx historyFile

	readJSON(t, engineHistory, &fx)

	seen := map[string]bool{}
	check := func(id string) {
		if seen[id] {
			t.Errorf("id %q recorded twice", id)
		}

		seen[id] = true
	}

	for _, w := range fx.Wire {
		check(w.ID)
	}

	for _, c := range fx.Conversations {
		check(c.ID)
	}

	for _, sc := range fx.Scans {
		check(sc.ID)
	}
}
