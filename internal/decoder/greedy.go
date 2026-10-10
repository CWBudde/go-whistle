package decoder

import (
	"fmt"
	"math"
	"strings"
)

// Result is one greedy transcript.
type Result struct {
	Tokens   []int  // the text tokens, without the prompt, the language token and EOS
	Text     string // Tokens decoded
	Language string // the language code, detected or forced
	EOS      bool   // decoding stopped at EOS, not at the token limit
}

// Greedy decodes enc, frames rows of the encoder output, greedily: the
// prompt, then the language token (forced to lang, or the argmax over the
// language tokens when lang is empty), then the argmax token until EOS or
// MaxTokens tokens (docs/FINDINGS.md F-023, F-024).
func (d *Decoder) Greedy(enc []float32, frames int, lang string) (Result, error) {
	mem, err := d.Memory(enc, frames)
	if err != nil {
		return Result{}, err
	}

	return d.GreedyMemory(mem, lang)
}

// GreedyMemory is Greedy on a projected memory.
func (d *Decoder) GreedyMemory(mem *Memory, lang string) (Result, error) {
	s := d.NewState(mem)

	logits, err := d.Extend(s, d.prompt)
	if err != nil {
		return Result{}, err
	}

	last := logits[len(logits)-d.vocab:]

	l, err := d.language(last, lang)
	if err != nil {
		return Result{}, err
	}

	res := Result{Language: l.Code}
	next := l.ID
	eos := d.tok.EOSID()

	for range d.maxTok - 1 {
		logits, err = d.Extend(s, []int{next})
		if err != nil {
			return Result{}, err
		}

		next = argmax(logits)
		if next == eos {
			res.EOS = true

			break
		}

		res.Tokens = append(res.Tokens, next)
	}

	res.Text, err = d.Decode(res.Tokens)

	return res, err
}

// language picks the language token from the logits at the language step:
// the forced code, or the argmax over the language tokens.
func (d *Decoder) language(logits []float32, code string) (Language, error) {
	if code != "" {
		for _, l := range d.langs {
			if l.Code == code {
				return l, nil
			}
		}

		return Language{}, fmt.Errorf("decoder: unknown language %q", code)
	}

	best := d.langs[0]
	for _, l := range d.langs[1:] {
		if logits[l.ID] > logits[best.ID] {
			best = l
		}
	}

	return best, nil
}

// Decode turns text tokens into the transcript: the tokenizer's decode,
// without the leading space of the LeadSpace format.
func (d *Decoder) Decode(tokens []int) (string, error) {
	text, err := d.tok.Decode(tokens)
	if err != nil {
		return "", err
	}

	if d.h.Lead == LeadSpace {
		text = strings.TrimPrefix(text, " ")
	}

	return text, nil
}

// Encode turns a transcript into its text tokens in the Lead format.
func (d *Decoder) Encode(text string) []int {
	if d.h.Lead == LeadSpace {
		text = " " + text
	}

	return d.tok.Encode(text)
}

// Targets is the sequence a transcript in language code teacher-forces:
// the prompt, the language token, the text tokens and EOS. The first
// len(Prompt()) − 1 positions predict nothing; every later one predicts the
// next token.
func (d *Decoder) Targets(code, text string) ([]int, error) {
	seq := d.Prompt()

	for _, l := range d.langs {
		if l.Code == code {
			seq = append(seq, l.ID)
			seq = append(seq, d.Encode(text)...)

			return append(seq, d.tok.EOSID()), nil
		}
	}

	return nil, fmt.Errorf("decoder: unknown language %q", code)
}

// Forced is the teacher-forced score of one target sequence.
type Forced struct {
	LogProbs []float64 // log p of each predicted token: the language token, the text tokens, EOS
	Top1     []bool    // whether each predicted token is the argmax
	LangRank int       // the language token's rank among all tokens at the language step, 0 = top
	LangTop  bool      // whether the language token wins among the language tokens
}

// TeacherForce runs seq (Targets) in one pass against mem and scores the
// tokens after the prompt.
func (d *Decoder) TeacherForce(mem *Memory, seq []int) (Forced, error) {
	p := len(d.prompt)
	if len(seq) <= p {
		return Forced{}, fmt.Errorf("decoder: %d target tokens for a prompt of %d", len(seq), p)
	}

	logits, err := d.Extend(d.NewState(mem), seq[:len(seq)-1])
	if err != nil {
		return Forced{}, err
	}

	var f Forced

	for i := p - 1; i < len(seq)-1; i++ {
		row := logits[i*d.vocab : (i+1)*d.vocab]
		want := seq[i+1]
		f.LogProbs = append(f.LogProbs, logSoftmaxAt(row, want))
		f.Top1 = append(f.Top1, argmax(row) == want)

		if i == p-1 {
			for _, v := range row {
				if v > row[want] {
					f.LangRank++
				}
			}

			l, _ := d.language(row, "")
			f.LangTop = l.ID == want
		}
	}

	return f, nil
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

// logSoftmaxAt is log softmax(v)[i], computed in float64.
func logSoftmaxAt(v []float32, i int) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = math.Max(m, float64(x))
	}

	var s float64
	for _, x := range v {
		s += math.Exp(float64(x) - m)
	}

	return float64(v[i]) - m - math.Log(s)
}
