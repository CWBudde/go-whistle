package toolcall

import (
	"errors"
	"fmt"
	"slices"
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
// or cut, which drops the conversation before it (Turn, Restarted). Earlier
// turns do not count against that room (docs/FINDINGS.md F-063): how the
// engine makes room for them past the context is not pinned. A refused
// turn still ends the first turn and the call
// before it, and leaves nothing in the history. Inputs keeps the inputs of
// the turns, which ground the values of later calls (docs/FINDINGS.md
// F-057).
//
// The assistant's turn before a later turn is closed when that turn comes
// (docs/FINDINGS.md F-063): Decoder.Decode leaves the answer open, less the
// token that ended it, and Turn puts the closing in front of the next turn
// it renders, outside the turn's count:
//
//	after a call block, before a tool result   <|im_end|>
//	after a call block, before a user turn     </tool_call><|im_end|>
//	after any other answer                     <|im_end|>
type Conversation struct {
	p         *spm.Processor
	static    int
	context   int
	started   bool
	afterCall bool
	restarted bool
	closing   closing
	inputs    []string
	end, call int // <|im_end|>, </tool_call>
}

// closing is how the answer before the next turn closes.
type closing uint8

const (
	closeNone closing = iota // nothing open: no answer, or one Answered did not see
	closeTurn                // <|im_end|>
	closeCall                // a call block: </tool_call> too before a user turn
)

// NewConversation starts a conversation over the static prefix of system
// and tools (the tools JSON Normalise or ToolsJSON returns) for a model
// with the given context length (max_seq_len, 8192 for needle3).
func NewConversation(p *spm.Processor, system, tools string, context int) *Conversation {
	end, _ := p.PieceID(imEnd)
	call, _ := p.PieceID(toolCallEnd)

	return &Conversation{p: p, static: len(StaticPrefix(p, system, tools)), context: context, end: end, call: call}
}

// Available is the most tokens a turn may have when the answer may take
// maxNew tokens: the context less the static prefix and maxNew. Earlier
// turns do not count; the engine drops them to make room.
func (c *Conversation) Available(maxNew int) int { return c.context - c.static - maxNew }

// Turn renders the next input, returns its token ids and moves the
// conversation on. The ids start with the closing of the answer before it
// (docs/FINDINGS.md F-063), which does not count towards the turn. A turn
// longer than Available(maxNew) is refused with an *OverflowError when
// refuse is set, as the engine's --fail-input-overflow does, or when even
// the markers around the input do not fit. Otherwise the engine cuts the
// turn to the room and drops the conversation before it, so the turn
// follows the static prefix (Restarted; docs/FINDINGS.md F-063): a first
// turn keeps the last Available(maxNew) tokens of its rendering, its
// leading newline cut with the input's start; a later turn keeps its
// markers and the end of its input. Either way the turn is no longer the
// first, and the call before it is answered; a refused turn leaves the
// closing for the next one.
func (c *Conversation) Turn(input string, maxNew int, refuse bool) ([]int, error) {
	head, body, tail := c.render(input)
	closed := c.closingIDs(strings.HasPrefix(head, "\n"+imStart+"tool\n"))
	first := !c.started

	c.started, c.afterCall, c.restarted = true, false, false

	h, b, t := c.p.Encode(head), c.p.Encode(body), c.p.Encode(tail)
	room := c.Available(maxNew)

	if n := len(h) + len(b) + len(t); n > room {
		// Without room for the markers no cut fits either (PR #61 review).
		if refuse || room < len(h)+len(t) {
			return nil, &OverflowError{Tokens: n, Available: room}
		}

		if first {
			hb := slices.Concat(h, b)
			h, b = nil, hb[len(hb)-(room-len(t)):]
		} else {
			b = b[len(b)-(room-len(h)-len(t)):]
		}

		closed, c.restarted = nil, true
	}

	c.inputs = append(c.inputs, input)
	c.closing = closeNone

	return append(append(append(closed, h...), b...), t...), nil
}

// Restarted reports whether the last Turn was cut to fit, after which the
// engine keeps nothing of the conversation before it but the static prefix
// (docs/FINDINGS.md F-063): the caller feeds the turn to a state that holds
// the static prefix alone.
func (c *Conversation) Restarted() bool { return c.restarted }

// Inputs are the inputs of the turns since the start or the last Reset, as
// given, the last one included: the user turns and tool results the engine
// grounds a call's values in (Envelope's earlier inputs are all but the
// last; docs/FINDINGS.md F-057). A refused turn is not one of them.
func (c *Conversation) Inputs() []string { return slices.Clone(c.inputs) }

// Answered records the model's answer to the last turn: a is the answer as
// Decoder.Decode left it open, nil when the caller closes it itself, and
// calls the number of function calls it made, suppressed ones left out.
func (c *Conversation) Answered(a *Decoded, calls int) {
	c.afterCall = calls > 0

	switch {
	case a == nil:
		c.closing = closeNone
	case !a.Truncated && len(a.Tokens) > 0 && a.Tokens[len(a.Tokens)-1] == c.call:
		c.closing = closeCall
	default:
		c.closing = closeTurn
	}
}

// Reset starts a new conversation over the same static prefix.
func (c *Conversation) Reset() {
	c.started, c.afterCall, c.restarted, c.closing, c.inputs = false, false, false, closeNone, nil
}

// closingIDs are the tokens that close the open answer before a tool
// result or, unless toolResult, a user turn.
func (c *Conversation) closingIDs(toolResult bool) []int {
	switch {
	case c.closing == closeCall && !toolResult:
		return []int{c.call, c.end}
	case c.closing != closeNone:
		return []int{c.end}
	}

	return nil
}

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
// (docs/FINDINGS.md F-051). '<', '&' and '>' count as written too; what
// F-051 measured for them was the engine's request parser reading Go's
// \u escapes of them as their hex digits (docs/FINDINGS.md F-063).
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
