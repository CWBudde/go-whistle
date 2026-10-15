package toolcall

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/spm"
)

// engineTurns is the fixture of `just oracle-text turns`.
const engineTurns = "../../testdata/text/engine/turns.json"

// turnsFile is testdata/text/engine/turns.json (tools/oracle/text
// turns.go).
type turnsFile struct {
	System    string `json:"system"`
	Toolset   string `json:"toolset"`
	Context   int    `json:"context"`
	Scenarios []struct {
		ID     string `json:"id"`
		Max    int    `json:"max"`
		Refuse bool   `json:"fail_input_overflow"`
		Steps  []struct {
			Input struct {
				Head   string `json:"head"`
				Unit   string `json:"unit"`
				Repeat int    `json:"repeat"`
				Tail   string `json:"tail"`
				Reset  bool   `json:"reset"`
			} `json:"input"`
			SHA256 string `json:"sha256"`
			Answer struct {
				Error string `json:"error"`
				Calls []call `json:"function_calls"`
			} `json:"answer"`
		} `json:"steps"`
	} `json:"scenarios"`
}

// overflowError is the counts in the engine's --fail-input-overflow error.
var overflowError = regexp.MustCompile(`^input exceeds the context window \((\d+) tokens, (\d+) available\)$`)

// TestConversationMatchesEngine replays every recorded conversation: each
// turn the engine refused with --fail-input-overflow is refused by Turn
// with the engine's length and room, each turn it answered fits, and the
// answers' calls move the conversation as they moved the engine's
// (docs/FINDINGS.md F-051). The refusals pin the first turn, later user
// turns, tool results (compact JSON after a call), reset, and that a
// refused turn still counts. The probes went out through json.Marshal,
// whose \u escapes of '<', '&' and '>' the engine reads as their hex
// digits, so each input is replayed as the engine read it
// (docs/FINDINGS.md F-063).
func TestConversationMatchesEngine(t *testing.T) {
	p := loadTokenizer(t)

	var (
		fx     turnsFile
		corpus struct {
			Toolsets map[string]json.RawMessage `json:"toolsets"`
		}
	)

	readJSON(t, engineTurns, &fx)
	readJSON(t, engineCases, &corpus)

	tools, err := ToolsJSON(corpus.Toolsets[fx.Toolset])
	if err != nil {
		t.Fatal(err)
	}

	refused := 0

	for _, sc := range fx.Scenarios {
		c := NewConversation(p, fx.System, tools, fx.Context)

		for i, st := range sc.Steps {
			if st.Input.Reset {
				c.Reset()

				continue
			}

			text := st.Input.Head + strings.Repeat(st.Input.Unit, st.Input.Repeat) + st.Input.Tail
			if sum := sha256.Sum256([]byte(text)); hex.EncodeToString(sum[:]) != st.SHA256 {
				t.Fatalf("%s step %d: the rebuilt input is not the engine's", sc.ID, i)
			}

			ids, err := c.Turn(asRead(text, goEscaped), sc.Max, sc.Refuse)

			if m := overflowError.FindStringSubmatch(st.Answer.Error); m != nil {
				refused++

				checkRefusal(t, sc.ID+" step "+strconv.Itoa(i), m, len(ids), err)

				continue
			}

			if err != nil {
				t.Errorf("%s step %d: engine answered, Turn: %v", sc.ID, i, err)

				continue
			}

			if len(ids) > c.Available(sc.Max) {
				t.Errorf("%s step %d: turn of %d tokens past the room of %d", sc.ID, i, len(ids), c.Available(sc.Max))
			}

			c.Answered(nil, len(st.Answer.Calls))
		}
	}

	if refused != 53 {
		t.Errorf("%d refused probes, want 53", refused)
	}

	t.Logf("%d scenarios, %d refusals with the engine's numbers", len(fx.Scenarios), refused)
}

// checkRefusal compares Turn's result for a turn the engine refused, m
// being the engine's error and its two counts.
func checkRefusal(t *testing.T, at string, m []string, got int, err error) {
	t.Helper()

	var oe *OverflowError
	if !errors.As(err, &oe) || !errors.Is(err, ErrContextOverflow) {
		t.Errorf("%s: engine refused %s, Turn gave %d tokens (%v)", at, m[0], got, err)

		return
	}

	if strconv.Itoa(oe.Tokens) != m[1] || strconv.Itoa(oe.Available) != m[2] {
		t.Errorf("%s: engine (%s tokens, %s available), Turn (%d, %d)", at, m[1], m[2], oe.Tokens, oe.Available)
	}

	if !strings.Contains(oe.Error(), m[0][len("input exceeds the context window"):]) {
		t.Errorf("%s: %q", at, oe.Error())
	}
}

// TestConversationTrim: without --fail-input-overflow the engine answers
// the request at the end of an oversized turn and not the one at its
// start, so it keeps the end; Turn's cut does the same.
func TestConversationTrim(t *testing.T) {
	p := loadTokenizer(t)

	var fx turnsFile

	readJSON(t, engineTurns, &fx)

	tools, err := ToolsJSON([]byte(`[]`))
	if err != nil {
		t.Fatal(err)
	}

	for _, sc := range fx.Scenarios {
		if sc.Refuse {
			continue
		}

		st := sc.Steps[len(sc.Steps)-1]
		text := st.Input.Head + strings.Repeat(st.Input.Unit, st.Input.Repeat) + st.Input.Tail
		last := strings.HasSuffix(text, "eggs.")

		called := false

		for _, k := range st.Answer.Calls {
			called = called || k.Name == "set_timer"
		}

		if called != last {
			t.Errorf("%s: the engine called set_timer %v", sc.ID, called)
		}

		c := NewConversation(p, fx.System, tools, fx.Context)

		ids, err := c.Turn(text, sc.Max, false)
		if err != nil {
			t.Fatal(err)
		}

		kept := strings.Contains(decode(t, p, ids), "Set a timer")
		if kept != last || len(ids) != c.Available(sc.Max) {
			t.Errorf("%s: %d tokens, the request kept %v", sc.ID, len(ids), kept)
		}
	}
}

func TestConversation(t *testing.T) {
	p := loadTokenizer(t)
	c := NewConversation(p, "date: 2026-10-11 Sun 09:30", `[]`, 8192)

	render := func(input string) string {
		ids, err := c.Turn(input, 512, true)
		if err != nil {
			t.Fatal(err)
		}

		return decode(t, p, ids)
	}

	tail := "<|im_end|>\n<|im_start|>assistant\n"

	if got := render("hi"); got != "\nhi"+tail {
		t.Errorf("first turn %q", got)
	}

	c.Answered(nil, 1)

	if got := render(` [ {"a": 1.50} ] `); got != "\n<|im_start|>tool\n<tool_result>[{\"a\":1.50}]</tool_result>"+tail {
		t.Errorf("tool result %q", got)
	}

	if got := render(`{"a": 1}`); got != "\n<|im_start|>user\n{\"a\": 1}"+tail {
		t.Errorf("JSON after a tool result %q", got)
	}

	c.Answered(nil, 1)

	if got := render(`not json`); got != "\n<|im_start|>user\nnot json"+tail {
		t.Errorf("text after a call %q", got)
	}

	c.Reset()

	if got := render(`[1]`); got != "\n[1]"+tail {
		t.Errorf("first turn after reset %q", got)
	}

	if got := c.Available(512); got != 8192-len(StaticPrefix(p, "date: 2026-10-11 Sun 09:30", `[]`))-512 {
		t.Errorf("Available = %d", got)
	}

	// No room even for the markers: refused without refuse too (PR #61 review).
	maxNew := c.Available(0) - 5

	ids, err := c.Turn("hello there", maxNew, false)
	if !errors.Is(err, ErrContextOverflow) || c.Available(maxNew) != 5 {
		t.Errorf("room %d: Turn gave %d tokens, %v", c.Available(maxNew), len(ids), err)
	}

	for in, want := range map[string]string{
		`1.50e2`: `1.50e2`, `"é\/ b"`: `"é\/ b"`, ` true `: `true`, `{"a": 1, "a": 2}`: `{"a":1,"a":2}`,
		"[1, \n \"a \\\" b\" ]": `[1,"a \" b"]`,
	} {
		if got, ok := compactJSON(in); !ok || got != want {
			t.Errorf("compactJSON(%q) = %q %v, want %q", in, got, ok, want)
		}
	}

	for _, in := range []string{``, `[1`, `[1] x`, "\"\xff\""} {
		if _, ok := compactJSON(in); ok {
			t.Errorf("compactJSON(%q) is JSON", in)
		}
	}
}

func decode(t *testing.T, p *spm.Processor, ids []int) string {
	t.Helper()

	s, err := p.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}

	return s
}

// TestConversationClosing: Turn closes the answer before it as the engine
// does (docs/FINDINGS.md F-063): a call block before a tool result with
// <|im_end|>, before a user turn with </tool_call><|im_end|>, any other
// answer with <|im_end|>; a refused turn leaves the closing for the next
// one, and the closing does not count towards the turn.
func TestConversationClosing(t *testing.T) {
	p := loadTokenizer(t)
	c := NewConversation(p, "", `[]`, 8192)

	id := func(piece string) int {
		n, ok := p.PieceID(piece)
		if !ok {
			t.Fatalf("no piece %s", piece)
		}

		return n
	}

	end, call := id(imEnd), id(toolCallEnd)
	callBlock := &Decoded{Tokens: []int{7, 8, call}}

	turn := func(input string, maxNew int, refuse bool) []int {
		t.Helper()

		ids, err := c.Turn(input, maxNew, refuse)
		if err != nil {
			t.Fatal(err)
		}

		return ids
	}

	starts := func(name string, ids []int, closing ...int) {
		t.Helper()

		if len(ids) < len(closing) || !slices.Equal(ids[:len(closing)], closing) ||
			strings.HasPrefix(decode(t, p, ids[len(closing):]), imEnd) {
			t.Errorf("%s: %q, want the closing %v first", name, decode(t, p, ids), closing)
		}
	}

	starts("first turn", turn("set a timer", 512, true))
	c.Answered(callBlock, 1)
	starts("tool result after a call", turn(`{"ok":true}`, 512, true), end)
	c.Answered(&Decoded{Tokens: []int{7, 8}}, 0)
	starts("user turn after a respond answer", turn("thanks", 512, true), end)
	c.Answered(&Decoded{Tokens: []int{7, call}}, 0)
	starts("user turn after an empty or suppressed call block", turn(`{"ok":true}`, 512, true), call, end)
	c.Answered(&Decoded{Tokens: []int{7, call}, Truncated: true}, 0)
	starts("after a truncated answer", turn("go on", 512, true), end)

	// A refused turn feeds nothing and keeps the closing; its count is the
	// turn's alone.
	c.Answered(callBlock, 1)

	body := `[` + strings.Repeat(`{"a":1},`, 400) + `1]`
	compact, _ := compactJSON(body)
	bare := len(p.Encode("\n"+imStart+"tool\n"+toolResultBeg)) + len(p.Encode(compact)) +
		len(p.Encode(toolResultEnd+imEnd+"\n"+imStart+"assistant\n"))

	var oe *OverflowError

	_, err := c.Turn(body, 7000, true)
	if !errors.As(err, &oe) || oe.Tokens != bare || oe.Available != c.Available(7000) {
		t.Errorf("the long tool result: %v, want %d tokens refused", err, bare)
	}

	starts("user turn after a call and a refused turn", turn("and now?", 512, true), call, end)

	c.Answered(callBlock, 1)
	c.Reset()
	starts("first turn after a reset", turn("hello", 512, true))

	c.Answered(nil, 1)
	starts("a turn whose answer the caller closed", turn(`{"ok":true}`, 512, true))
}

// TestConversationCut: a turn longer than the room, without the flag, is
// cut to the room and drops the conversation before it (docs/FINDINGS.md
// F-063): a first turn keeps the last tokens of its rendering, its leading
// newline cut first; a later turn keeps its markers, and the closing of the
// answer before it goes with the history.
func TestConversationCut(t *testing.T) {
	p := loadTokenizer(t)
	long := strings.Repeat("the weather was nice and we walked along the river. ", 40) + "Set a timer."
	tail := p.Encode(imEnd + "\n" + imStart + "assistant\n")

	c := NewConversation(p, "", `[]`, 8192)
	maxNew := c.Available(0) - 120

	ids, err := c.Turn(long, maxNew, false)
	if err != nil {
		t.Fatal(err)
	}

	whole := append(append(p.Encode("\n"), p.Encode(long)...), tail...)
	if !c.Restarted() || !slices.Equal(ids, whole[len(whole)-120:]) {
		t.Errorf("first turn: %d tokens, restarted %v, want the last 120 of %d", len(ids), c.Restarted(), len(whole))
	}

	ids, err = c.Turn("hello", maxNew, false)
	if err != nil || c.Restarted() || len(ids) != len(p.Encode("hello"))+9 {
		t.Errorf("a turn that fits: %d tokens, restarted %v, %v", len(ids), c.Restarted(), err)
	}

	end, _ := p.PieceID(toolCallEnd)
	c.Answered(&Decoded{Tokens: []int{7, end}}, 1)

	ids, err = c.Turn(long, maxNew, false)
	if err != nil {
		t.Fatal(err)
	}

	head := p.Encode("\n" + imStart + "user\n")
	if !c.Restarted() || len(ids) != 120 || !slices.Equal(ids[:len(head)], head) ||
		!slices.Equal(ids[len(ids)-len(tail):], tail) {
		t.Errorf("later turn: %q, restarted %v", decode(t, p, ids), c.Restarted())
	}
}
