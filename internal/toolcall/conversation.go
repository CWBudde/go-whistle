package toolcall

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/cwbudde/go-whistle/internal/spm"
)

// ErrContextOverflow is the engine's refusal of a turn that does not fit
// the context window even with the conversation's earlier turns dropped.
var ErrContextOverflow = errors.New("toolcall: input exceeds the context window")

// OverflowError is ErrContextOverflow with the engine's two numbers: the
// turn's length and the room a turn has.
type OverflowError struct {
	Tokens, Available int
}

func (e *OverflowError) Error() string {
	return fmt.Sprintf("%v (%d tokens, %d available)", ErrContextOverflow, e.Tokens, e.Available)
}

func (e *OverflowError) Unwrap() error { return ErrContextOverflow }

// Conversation renders the turns of one engine session after the static
// prefix, as `needle --serve` and needle_complete continue it
// (docs/FINDINGS.md F-051):
//
//	the first turn       \n{input}<|im_end|>\n<|im_start|>assistant\n
//	a later user turn    \n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
//	a tool result        \n<|im_start|>tool\n<tool_result>{json}</tool_result><|im_end|>\n<|im_start|>assistant\n
//
// The first turn continues the user message the static prefix opened. A
// later input is a tool result when the answer before it made at least one
// call (suppressed calls do not count) and the input is JSON; it is then
// written without the white space outside its strings (compactJSON). A
// turn has at most context − static prefix − the answer's token limit
// tokens; a longer one is refused (with the engine's --fail-input-overflow)
// or loses its start. A refused turn still ends the first turn and the call
// before it.
type Conversation struct {
	p         *spm.Processor
	static    int
	context   int
	started   bool
	afterCall bool
}

// NewConversation starts a conversation over the static prefix of system
// and tools (the tools JSON Normalise or ToolsJSON returns) for a model
// with the given context length (max_seq_len, 8192 for needle3).
func NewConversation(p *spm.Processor, system, tools string, context int) *Conversation {
	return &Conversation{p: p, static: len(StaticPrefix(p, system, tools)), context: context}
}

// Available is the most tokens a turn may have when the answer may take
// maxNew tokens: the context less the static prefix and maxNew. Earlier
// turns do not count; the engine drops them to make room.
func (c *Conversation) Available(maxNew int) int { return c.context - c.static - maxNew }

// Turn renders the next input, returns its token ids and moves the
// conversation on. A turn longer than Available(maxNew) is refused with an
// *OverflowError when refuse is set, as the engine's --fail-input-overflow
// does; otherwise the input loses its first tokens until the turn fits, as
// the engine keeps an oversized turn's end (docs/FINDINGS.md F-051: the
// engine's exact cut is not measured). Either way the turn is no longer the
// first, and the call before it is answered.
func (c *Conversation) Turn(input string, maxNew int, refuse bool) ([]int, error) {
	head, body, tail := c.render(input)

	c.started, c.afterCall = true, false

	h, b, t := c.p.Encode(head), c.p.Encode(body), c.p.Encode(tail)
	room := c.Available(maxNew)

	if n := len(h) + len(b) + len(t); n > room {
		if refuse {
			return nil, &OverflowError{Tokens: n, Available: room}
		}

		keep := max(room-len(h)-len(t), 0)
		if keep < len(b) {
			b = b[len(b)-keep:]
		}
	}

	return append(append(h, b...), t...), nil
}

// Answered records the model's answer to the last turn: calls is the
// number of function calls it made, suppressed ones left out.
func (c *Conversation) Answered(calls int) { c.afterCall = calls > 0 }

// Reset starts a new conversation over the same static prefix.
func (c *Conversation) Reset() { c.started, c.afterCall = false, false }

// render splits a turn into its markers before the input, the input as the
// model sees it, and the markers after.
func (c *Conversation) render(input string) (head, body, tail string) {
	tail = imEnd + "\n" + imStart + "assistant\n"

	switch {
	case !c.started:
		return "\n", input, tail
	case c.afterCall:
		if js, ok := compactJSON(input); ok {
			return "\n" + imStart + "tool\n" + toolResultBeg, js, toolResultEnd + tail
		}
	}

	return "\n" + imStart + "user\n", input, tail
}

const (
	toolResultBeg = "<tool_result>"
	toolResultEnd = "</tool_result>"
)

// compactJSON is s without the white space outside its strings, and
// whether s is JSON at all (white space around the value allowed). Every
// literal stays as written: numbers ("1.50", "1e2", "-0"), escapes ("\/",
// \u-escapes) and repeated keys, which is what the engine's counts show
// (docs/FINDINGS.md F-051). The engine writes '<', '&' and '>' inside
// strings in a way the counts do not identify; they stay as they are here.
func compactJSON(s string) (string, bool) {
	if !utf8.ValidString(s) {
		return "", false
	}

	p := &pyParser{s: []byte(s)}

	_, err := p.value()
	if err != nil {
		return "", false
	}

	p.space()

	if p.i != len(p.s) {
		return "", false
	}

	var b strings.Builder

	in := false

	for i := 0; i < len(s); i++ {
		c := s[i]

		switch {
		case in && c == '\\':
			b.WriteByte(c)

			i++
			c = s[i]
		case c == '"':
			in = !in
		case !in && strings.IndexByte(" \t\n\r", c) >= 0:
			continue
		}

		b.WriteByte(c)
	}

	return b.String(), true
}
