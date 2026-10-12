package toolcall

import (
	"errors"
	"fmt"
	"math"

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
	// Margins is each token's top-1 minus top-2 logit (+Inf for the forced
	// tokens).
	Margins []float64
	// Truncated reports a budget that ran out before the answer ended.
	Truncated bool
}

// Decoder decodes answers greedily, as the engine does.
type Decoder struct {
	m     *text.Model
	p     *spm.Processor
	think []int
	call  int // </tool_call>, the last counted token of a call
	end   int // <|im_end|>
	eos   int
}

// NewDecoder is a decoder for m with the tokenizer p.
func NewDecoder(m *text.Model, p *spm.Processor) (*Decoder, error) {
	d := &Decoder{m: m, p: p, think: p.Encode(thinkOpen), eos: p.EOSID()}

	for piece, id := range map[string]*int{toolCallEnd: &d.call, imEnd: &d.end} {
		var ok bool

		*id, ok = p.PieceID(piece)
		if !ok {
			return nil, fmt.Errorf("toolcall: no piece %s", piece)
		}
	}

	return d, nil
}

const toolCallEnd = "</tool_call>"

// Decode feeds input (a turn's tokens: the static prefix and the first
// turn, or a later turn) to s and decodes the answer within maxNew tokens.
// It leaves s holding the answer as fed, its last generated token left out
// (a closing </tool_call>, or the one that met the budget), and
// <|im_end|>, as the engine keeps the assistant's turn in the history
// (docs/FINDINGS.md F-052).
func (d *Decoder) Decode(s *text.State, input []int, maxNew int) (*Decoded, error) {
	if maxNew < 1 {
		return nil, errors.New("toolcall: no token budget")
	}

	out := &Decoded{}

	forced := d.think[:min(len(d.think), maxNew)]
	for range forced {
		out.Probs = append(out.Probs, 1)
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

	for !out.Truncated {
		best, second := top2(row)
		if best == d.end || best == d.eos {
			break
		}

		out.Tokens = append(out.Tokens, best)
		out.Probs = append(out.Probs, softmaxAt(row, best))
		out.Margins = append(out.Margins, float64(row[best]-row[second]))

		if best == d.call {
			break
		}

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
// closing <|im_end|> left out (docs/FINDINGS.md F-053). The engine's min
// with the call's decode probability is not applied: the corpus never shows
// it binding.
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

// top2 are the indices of row's largest and second-largest values, the
// first on a tie.
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
