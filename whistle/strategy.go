package whistle

import (
	"fmt"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
)

// strategy turns a clip's projected encoder memory into a token sequence.
// Transcribe sees only this seam, so beam search (PLAN.md 6.6) can replace
// greedy decoding without changing the public API.
type strategy interface {
	decode(d *decoder.Decoder, mem *decoder.Memory, lang string, clk *clock) (decoded, error)
}

// decoded is a strategy's transcript before detokenisation.
type decoded struct {
	tokens []int  // the text tokens, without the prompt, the language token and EOS
	lang   string // the language code, detected or forced
}

// clock times one transcription the way the engine reports it
// (docs/FINDINGS.md F-027): TTFT from start to the first token after the
// language token, the decode rate over the tokens after that one.
type clock struct {
	start, first, last time.Time
	after              int // tokens decoded after the first
	now                func() time.Time
}

func newClock(now func() time.Time) *clock {
	return &clock{start: now(), now: now}
}

// token records that the strategy picked a token after the language token,
// text or EOS.
func (c *clock) token() {
	t := c.now()
	if c.first.IsZero() {
		c.first = t
	} else {
		c.after++
	}

	c.last = t
}

// ttft is the time to the first token, 0 when none was decoded.
func (c *clock) ttft() time.Duration {
	if c.first.IsZero() {
		return 0
	}

	return c.first.Sub(c.start)
}

// tps is the decode rate after the first token, 0 without one.
func (c *clock) tps() float64 {
	d := c.last.Sub(c.first).Seconds()
	if c.after == 0 || d <= 0 {
		return 0
	}

	return float64(c.after) / d
}

// greedy is decoder.GreedyMemory step for step through the decoder's
// exported API, with the clock between the steps: the prompt, the language
// token (forced, or the argmax over the language tokens), then the argmax
// until EOS or MaxTokens predicted tokens (docs/FINDINGS.md F-023, F-024).
type greedy struct{}

func (greedy) decode(d *decoder.Decoder, mem *decoder.Memory, lang string, clk *clock) (decoded, error) {
	s := d.NewState(mem)

	logits, err := d.Extend(s, d.Prompt())
	if err != nil {
		return decoded{}, fmt.Errorf("greedy: %w", err)
	}

	l, err := language(d, logits[len(logits)-d.Vocab():], lang)
	if err != nil {
		return decoded{}, err
	}

	out := decoded{lang: l.Code}
	next := l.ID
	eos := d.Tokenizer().EOSID()

	for range d.MaxTokens() - 1 {
		logits, err = d.Extend(s, []int{next})
		if err != nil {
			return decoded{}, fmt.Errorf("greedy: %w", err)
		}

		next = argmax(logits)

		clk.token()

		if next == eos {
			break
		}

		out.tokens = append(out.tokens, next)
	}

	return out, nil
}

// language picks the language token from the logits at the language step:
// the forced code, or the argmax over the language tokens, the first on
// ties.
func language(d *decoder.Decoder, logits []float32, code string) (decoder.Language, error) {
	langs := d.Languages()

	if code != "" {
		for _, l := range langs {
			if l.Code == code {
				return l, nil
			}
		}

		return decoder.Language{}, fmt.Errorf("unknown language %q", code)
	}

	best := langs[0]
	for _, l := range langs[1:] {
		if logits[l.ID] > logits[best.ID] {
			best = l
		}
	}

	return best, nil
}

// argmax is the index of the largest value, the first on ties.
func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}

	return best
}
