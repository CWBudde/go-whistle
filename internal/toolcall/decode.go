package toolcall

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/text"
)

// thinkOpen is what the engine puts at the start of every answer before
// the model's first token (docs/FINDINGS.md F-052).
const thinkOpen = "<think>\n"

// Decoded is one answer as the engine decodes it (docs/FINDINGS.md F-052):
// the forced "<think>\n", then the greedy tokens up to and including
// </tool_call>, or up to <|im_end|> or EOS, which do not count.
type Decoded struct {
	// Tokens are the answer's tokens, the forced ones first; their number
	// is needle_complete's generated-token count.
	Tokens []int
	// Text is Tokens decoded.
	Text string
	// Probs is each token's softmax probability where the model chose it
	// (1 for the forced tokens).
	Probs []float64
	// Allowed is each token's softmax probability among the tokens the
	// grammar allows (Probs where it allows any; 1 for the forced tokens).
	Allowed []float64
	// Margins is each token's top-1 minus top-2 logit among the tokens the
	// grammar allows (+Inf for the forced tokens and where one is allowed).
	Margins []float64
	// Truncated reports a budget that ran out before the answer ended.
	Truncated bool
	// stop is the probability of the uncounted token that ended the answer
	// (<|im_end|> or EOS), 0 when none did.
	stop float64
	// block is the index in Tokens of the <tool_call> that opens the call
	// block, -1 without one.
	block int
}

// Probability is the decode probability of the answer, which the engine's
// confidence is the min of with the ConfidenceHead's score (llms.txt: "the
// min of a post-hoc calibration head and the decode probability of the
// call"; docs/FINDINGS.md F-062):
//
//   - an answer with a call block: the smallest probability of a block
//     token, <tool_call> to </tool_call>, among the tokens the grammar
//     allows (Allowed);
//   - a respond answer: the geometric mean of every token's probability
//     (Probs), the reasoning and the forced "<think>\n" (as 1) included,
//     and of the uncounted <|im_end|> or EOS that ended it;
//   - a truncated answer, or one without recorded probabilities: 1.
func (a *Decoded) Probability() float64 {
	if a.Truncated || len(a.Probs) == 0 {
		return 1
	}

	if a.block < 0 {
		var sum float64
		for _, p := range a.Probs {
			sum += math.Log(p)
		}

		n := len(a.Probs)
		if a.stop > 0 {
			sum += math.Log(a.stop)
			n++
		}

		return math.Exp(sum / float64(n))
	}

	p := 1.0
	for _, x := range a.Allowed[min(a.block, len(a.Allowed)):] {
		p = min(p, x)
	}

	return p
}

// take appends the chosen token best of the logits row, with its
// probability over row and among the tokens allowed (nil for any), its
// margin over second, and whether it opens the call block.
func (a *Decoded) take(row []float32, best, second int, allowed []int, opens bool) {
	a.Tokens = append(a.Tokens, best)
	a.Probs = append(a.Probs, softmaxAt(row, best))
	a.Allowed = append(a.Allowed, softmaxAmong(row, best, allowed))
	a.Margins = append(a.Margins, margin(row, best, second))

	if opens && a.block < 0 {
		a.block = len(a.Tokens) - 1
	}
}

// Decoder decodes answers greedily, as the engine does.
type Decoder struct {
	m        *text.Model
	p        *spm.Processor
	think    []int
	thinkEnd int // </think>
	open     int // <tool_call>
	call     int // </tool_call>, the last counted token of a call
	end      int // <|im_end|>
	eos      int
	mask     *tokenMask
}

// NewDecoder is a decoder for m with the tokenizer p.
func NewDecoder(m *text.Model, p *spm.Processor) (*Decoder, error) {
	d := &Decoder{
		m: m, p: p, think: p.Encode(thinkOpen), eos: p.EOSID(),
		mask: newTokenMask(p, m.Config().OutVocab),
	}

	for piece, id := range map[string]*int{
		thinkClose: &d.thinkEnd, toolCallStart: &d.open, toolCallEnd: &d.call, imEnd: &d.end,
	} {
		var ok bool

		*id, ok = p.PieceID(piece)
		if !ok {
			return nil, fmt.Errorf("toolcall: no piece %s", piece)
		}
	}

	return d, nil
}

const (
	thinkClose    = "</think>"
	toolCallStart = "<tool_call>"
	toolCallEnd   = "</tool_call>"
)

// Decode feeds input (a turn's tokens: the static prefix and the first
// turn, or a later turn) to s and decodes the answer within maxNew tokens.
// It leaves s holding the answer as fed, its last generated token left out
// (a closing </tool_call>, or the one that met the budget), and
// <|im_end|>, as the engine keeps the assistant's turn in the history
// (docs/FINDINGS.md F-052).
//
// Under the grammar g (nil: none) the reasoning stays free; from
// <tool_call> on, each token is the most likely one whose bytes keep g's
// automaton alive, and </tool_call> may only close a whole call block. A
// grammar that requires a call (a routed request) also forces "\n" and
// <tool_call> right after </think> (docs/FINDINGS.md F-055).
func (d *Decoder) Decode(s *text.State, input []int, maxNew int, g *Grammar) (*Decoded, error) {
	if maxNew < 1 {
		return nil, errors.New("toolcall: no token budget")
	}

	out := &Decoded{block: -1}

	forced := d.think[:min(len(d.think), maxNew)]
	for range forced {
		out.Probs = append(out.Probs, 1)
		out.Allowed = append(out.Allowed, 1)
		out.Margins = append(out.Margins, math.Inf(1))
	}

	out.Tokens = append(out.Tokens, forced...)

	logits, err := d.m.Extend(s, append(append([]int(nil), input...), forced...))
	if err != nil {
		return nil, fmt.Errorf("toolcall: %w", err)
	}

	v := d.m.Config().OutVocab
	row := logits[len(logits)-v:]

	out.Truncated = len(out.Tokens) == maxNew
	c := &constraint{d: d, g: g}

	for !out.Truncated {
		best, second, allowed, err := c.choose(row)
		if err != nil {
			return nil, err
		}

		if best == d.end || best == d.eos {
			out.stop = softmaxAmong(row, best, allowed)

			break
		}

		out.take(row, best, second, allowed, best == d.open)

		if best == d.call {
			break
		}

		c.took(best)

		if len(out.Tokens) == maxNew {
			out.Truncated = true

			break
		}

		row, err = d.m.Extend(s, []int{best})
		if err != nil {
			return nil, fmt.Errorf("toolcall: %w", err)
		}
	}

	// <|im_end|> closes the turn in s. The answer's last token counts but is
	// not fed, a closing </tool_call> as the token that meets the budget:
	// only that history gives the engine's next answer (timer_then_result's
	// tool-result turn) and confidence (timer_budget; docs/FINDINGS.md
	// F-052, F-053).
	_, err = d.m.Extend(s, []int{d.end})
	if err != nil {
		return nil, fmt.Errorf("toolcall: %w", err)
	}

	out.Text, err = d.p.Decode(out.Tokens)
	if err != nil {
		return nil, fmt.Errorf("toolcall: %w", err)
	}

	return out, nil
}

// Confidence is the ConfidenceHead's score of the conversation s holds after
// Decode: the sigmoid of its logit pooled over every token fed, the turn's
// closing <|im_end|> left out (docs/FINDINGS.md F-053, F-062). The engine
// reports the min of it and the answer's decode probability
// (Decoded.Probability), which Normalised.Envelope takes.
func (d *Decoder) Confidence(s *text.State) (float64, error) {
	toks := s.Tokens()
	if len(toks) < 2 {
		return 0, errors.New("toolcall: confidence of an empty conversation")
	}

	toks = toks[:len(toks)-1]

	out, err := d.m.Forward(toks, true)
	if err != nil {
		return 0, fmt.Errorf("toolcall: %w", err)
	}

	h, err := d.m.Confidence(out, len(toks))
	if err != nil {
		return 0, fmt.Errorf("toolcall: %w", err)
	}

	return 1 / (1 + math.Exp(-float64(h))), nil
}

// constraint tracks where an answer is against a grammar: the free
// reasoning, the part after </think> and the call block.
type constraint struct {
	d       *Decoder
	g       *Grammar
	phase   uint8
	prelude string   // what a routed answer still has to write before <tool_call>
	states  []*stack // the call block's automaton
}

// Answer phases.
const (
	inThink = iota
	afterThink
	inCall
)

// allowed are the tokens the grammar allows next, nil for any.
func (c *constraint) allowed() ([]int, error) {
	switch {
	case c.g == nil || c.phase == inThink:
		return nil, nil
	case c.phase == afterThink && !c.g.required:
		return nil, nil
	case c.phase == afterThink && c.prelude == "":
		return []int{c.d.open}, nil
	case c.phase == afterThink:
		var ids []int

		for id, bs := range c.d.mask.bytes {
			if len(bs) > 0 && strings.HasPrefix(c.prelude, string(bs)) {
				ids = append(ids, id)
			}
		}

		return ids, nil
	}

	ids := c.d.mask.allowed(c.g, c.states)
	if slices.ContainsFunc(c.states, (*stack).accepting) {
		ids = append(slices.Clip(ids), c.d.call)
	}

	if len(ids) == 0 {
		return nil, errors.New("toolcall: the grammar allows no token")
	}

	return ids, nil
}

// choose are the most and the second most likely tokens of row the
// grammar allows, and the tokens it allows (nil for any).
func (c *constraint) choose(row []float32) (best, second int, allowed []int, err error) {
	allowed, err = c.allowed()
	if err != nil {
		return 0, 0, nil, err
	}

	if allowed == nil {
		best, second = top2(row)

		return best, second, nil, nil
	}

	best, second = top2Of(row, allowed)

	return best, second, allowed, nil
}

// took moves on past the token id.
func (c *constraint) took(id int) {
	switch {
	case c.phase == inThink && id == c.d.thinkEnd:
		c.phase, c.prelude = afterThink, "\n"
	case c.phase == afterThink && id == c.d.open && c.g != nil:
		c.phase, c.states = inCall, c.g.start()
	case c.phase == afterThink:
		c.prelude = strings.TrimPrefix(c.prelude, string(c.d.mask.bytes[id]))
	case c.phase == inCall:
		c.states = c.g.feed(c.states, c.d.mask.bytes[id])
	}
}

// top2Of are the indices of the largest and second-largest values of row
// among ids, the first id on a tie; one id is both.
func top2Of(row []float32, ids []int) (best, second int) {
	best, second = ids[0], -1

	for _, id := range ids[1:] {
		switch {
		case row[id] > row[best]:
			best, second = id, best
		case second < 0 || row[id] > row[second]:
			second = id
		}
	}

	if second < 0 {
		second = best
	}

	return best, second
}

// margin is row[best] - row[second], +Inf when they are one token.
func margin(row []float32, best, second int) float64 {
	if best == second {
		return math.Inf(1)
	}

	return float64(row[best] - row[second])
}

// top2 are the indices of row's largest and second-largest values, the
// first on a tie; a row of one value is both (PR #64 review).
func top2(row []float32) (best, second int) {
	if len(row) == 1 {
		return 0, 0
	}

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

// softmaxAmong is row[i]'s softmax probability among the tokens ids, over
// all of row when ids is nil.
func softmaxAmong(row []float32, i int, ids []int) float64 {
	if ids == nil {
		return softmaxAt(row, i)
	}

	hi := float64(row[ids[0]])
	for _, id := range ids {
		hi = max(hi, float64(row[id]))
	}

	var sum float64
	for _, id := range ids {
		sum += math.Exp(float64(row[id]) - hi)
	}

	return math.Exp(float64(row[i])-hi) / sum
}

// softmaxAt is softmax(row)[i].
func softmaxAt(row []float32, i int) float64 {
	hi := float64(row[0])
	for _, x := range row {
		hi = max(hi, float64(x))
	}

	var sum float64
	for _, x := range row {
		sum += math.Exp(float64(x) - hi)
	}

	return math.Exp(float64(row[i])-hi) / sum
}
