package toolcall

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"regexp"
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
// refused turn still counts.
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

	var (
		refused      int
		unidentified []string
	)

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

			ids, err := c.Turn(text, sc.Max, sc.Refuse)

			if m := overflowError.FindStringSubmatch(st.Answer.Error); m != nil {
				refused++

				if !checkRefusal(t, sc.ID+" step "+strconv.Itoa(i), text, m, len(ids), err) {
					unidentified = append(unidentified, sc.ID)
				}

				continue
			}

			if err != nil {
				t.Errorf("%s step %d: engine answered, Turn: %v", sc.ID, i, err)

				continue
			}

			if len(ids) > c.Available(sc.Max) {
				t.Errorf("%s step %d: turn of %d tokens past the room of %d", sc.ID, i, len(ids), c.Available(sc.Max))
			}

			c.Answered(len(st.Answer.Calls))
		}
	}

	if refused-len(unidentified) < 25 {
		t.Errorf("%d refused probes; the fixture does not pin the rendering", refused-len(unidentified))
	}

	t.Logf("%d scenarios, %d refusals matched, %d with '<', '&' or '>' in a tool result unidentified: %v",
		len(fx.Scenarios), refused-len(unidentified), len(unidentified), unidentified)
}

// checkRefusal compares Turn's result for a turn the engine refused, m
// being the engine's error and its two counts. It returns false for a turn
// whose length the engine's writing of '<', '&' and '>' in a tool result's
// strings changes (F-051: not identified), where the engine counts more.
func checkRefusal(t *testing.T, at, text string, m []string, got int, err error) bool {
	t.Helper()

	var oe *OverflowError
	if !errors.As(err, &oe) || !errors.Is(err, ErrContextOverflow) {
		t.Errorf("%s: engine refused %s, Turn gave %d tokens (%v)", at, m[0], got, err)

		return true
	}

	if strings.ContainsAny(text, "<&>") && strconv.Itoa(oe.Tokens) != m[1] {
		if n, _ := strconv.Atoi(m[1]); n <= oe.Tokens {
			t.Errorf("%s: engine %s tokens, Turn %d", at, m[1], oe.Tokens)
		}

		return false
	}

	if strconv.Itoa(oe.Tokens) != m[1] || strconv.Itoa(oe.Available) != m[2] {
		t.Errorf("%s: engine (%s tokens, %s available), Turn (%d, %d)", at, m[1], m[2], oe.Tokens, oe.Available)
	}

	if !strings.Contains(oe.Error(), m[0][len("input exceeds the context window"):]) {
		t.Errorf("%s: %q", at, oe.Error())
	}

	return true
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

	c.Answered(1)

	if got := render(` [ {"a": 1.50} ] `); got != "\n<|im_start|>tool\n<tool_result>[{\"a\":1.50}]</tool_result>"+tail {
		t.Errorf("tool result %q", got)
	}

	if got := render(`{"a": 1}`); got != "\n<|im_start|>user\n{\"a\": 1}"+tail {
		t.Errorf("JSON after a tool result %q", got)
	}

	c.Answered(1)

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
