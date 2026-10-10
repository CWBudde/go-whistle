package align

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/silence"
)

// Head is one cross-attention head of one decoder layer.
type Head struct{ Layer, Head int }

// Rows names the query row that times a token.
type Rows int

const (
	_ Rows = iota
	// RowsPredict times token k by the query that predicts it, whose input
	// is token k−1: Whisper's find_alignment.
	RowsPredict
	// RowsInput times token k by its own query, the row whose input it is.
	RowsInput
)

// Norm is the normalisation of each head's attention before the heads are
// averaged.
type Norm int

const (
	_ Norm = iota
	// NormNone keeps the probabilities.
	NormNone
	// NormFrame z-scores every frame over the rows: Whisper's std_mean over
	// the token axis.
	NormFrame
	// NormRow z-scores every row over the frames.
	NormRow
	// NormFrameAll z-scores every frame over all positions of the pass,
	// the prompt's and EOS' included, before the DTW rows are cut out:
	// exactly Whisper's order.
	NormFrameAll
)

// Cost is the DTW cost of a matrix entry a.
type Cost int

const (
	_ Cost = iota
	// CostNeg is −a, Whisper's dtw(−matrix).
	CostNeg
	// CostNegLog is −log a (the probabilities, NormNone only).
	CostNegLog
)

// End is how a word's end is read off the path.
type End int

const (
	_ End = iota
	// EndNext is the first frame of the row after the word's last token:
	// Whisper's jump times, so consecutive words share their boundary.
	EndNext
	// EndLast is one past the last frame of the word's last token's row.
	EndLast
)

// Options are the timing algorithm's axes (docs/FINDINGS.md F-033).
type Options struct {
	// Heads are the heads averaged; empty is Whisper's default alignment
	// heads: every head of the last half of the layers captured, the
	// layers that ran (layers 4–7 of eight; on a ladder rung its last half).
	Heads  []Head
	Gate   bool // weight each head's row by its mean output gate on that row
	Rows   Rows
	Lead   int // DTW rows before the first text token's, absorbing leading silence
	Norm   Norm
	Filter int // median filter width over the frames, odd; 1 or 0: none
	Cost   Cost
	Step   Step
	End    End
	Core   bool // time a word by its core tokens, without its punctuation (Word.Core)
	Offset int  // frames added to every start and end
	// Onset moves the first word's start to the speech onset when that is
	// later: the first 10 ms level (Input.Levels) within Onset dB of the
	// clip's 99th percentile, rounded to the 80 ms grid. 0 turns it off.
	Onset float64
}

// Default is the timing that reproduces the engine (docs/FINDINGS.md
// F-033): Whisper's find_alignment with its default alignment heads (every
// head of the last half of the decoder layers that run), each head
// z-scored per frame over all positions of the pass, no median filter,
// averaged; the DTW rows are the queries that predict the language token,
// each text token and EOS; −matrix, Whisper's step pattern and jump times;
// words timed without their punctuation in Whisper's sense; the first
// word's start moved to the speech onset 14 dB below the clip's 99th
// percentile level when that is later (fitted, not identified).
func Default() Options {
	return Options{
		Rows: RowsPredict, Lead: 1, Norm: NormFrameAll, Filter: 1, Cost: CostNeg, Step: StepWhisper, End: EndNext,
		Core: true, Onset: 14,
	}
}

// Validate reports an inconsistent combination.
func (o Options) Validate() error {
	switch {
	case o.Rows != RowsPredict && o.Rows != RowsInput:
		return fmt.Errorf("align: rows %d", o.Rows)
	case o.Norm < NormNone || o.Norm > NormFrameAll:
		return fmt.Errorf("align: norm %d", o.Norm)
	case o.Cost != CostNeg && o.Cost != CostNegLog, o.Cost == CostNegLog && o.Norm != NormNone:
		return fmt.Errorf("align: cost %d with norm %d", o.Cost, o.Norm)
	case o.Step != StepWhisper && o.Step != StepNoShare:
		return fmt.Errorf("align: step %d", o.Step)
	case o.End != EndNext && o.End != EndLast:
		return fmt.Errorf("align: end %d", o.End)
	case o.Lead < 0, o.Filter < 0, o.Filter > 1 && o.Filter%2 == 0, o.Onset < 0:
		return fmt.Errorf("align: lead %d, filter %d, onset %g", o.Lead, o.Filter, o.Onset)
	}

	return nil
}

// String is a compact key of o; a layer whose eight heads all take part
// prints as "L<layer>".
func (o Options) String() string {
	heads := "last-half"

	if len(o.Heads) > 0 {
		byLayer := map[int][]int{}
		for _, h := range o.Heads {
			byLayer[h.Layer] = append(byLayer[h.Layer], h.Head)
		}

		var hs []string

		for l := range 64 {
			switch hl := byLayer[l]; {
			case len(hl) == 0:
			case len(hl) == 8 && slices.Equal(slices.Sorted(slices.Values(hl)), []int{0, 1, 2, 3, 4, 5, 6, 7}):
				hs = append(hs, "L"+strconv.Itoa(l))
			default:
				for _, h := range hl {
					hs = append(hs, fmt.Sprintf("%d.%d", l, h))
				}
			}
		}

		heads = strings.Join(hs, ",")
	}

	return fmt.Sprintf("heads=%s gate=%t rows=%s lead=%d norm=%s filter=%d cost=%s step=%s end=%s core=%t offset=%d "+
		"onset=%g", heads, o.Gate, pick(o.Rows, "predict", "input"), o.Lead, pick(o.Norm, "none", "frame", "row",
		"frameall"), o.Filter, pick(o.Cost, "neg", "neglog"), pick(o.Step, "whisper", "noshare"), pick(o.End, "next",
		"last"), o.Core, o.Offset, o.Onset)
}

func pick[T ~int](v T, names ...string) string {
	if v < 1 || int(v) > len(names) {
		return strconv.Itoa(int(v))
	}

	return names[v-1]
}

// Span is one word's start and end on the frame grid.
type Span struct{ Start, End int }

// Input is one transcript's teacher-forced pass: the captured attention
// (decoder.State.SetCapture) of the prompt, the language token, the text
// tokens and EOS fed in one or more Extend calls.
type Input struct {
	Attn   []decoder.Attention
	First  int // the position of the first text token
	Tokens int // the text tokens
	Frames int // the encoder frames
	// Levels is the clip's speech-band level per 10 ms front-end frame in
	// dB (silence.Gate.Levels), eight to an encoder frame; Options.Onset
	// needs it.
	Levels []float64
}

// ErrNoPath reports a transcript the DTW cannot place on the frames, such
// as more rows than frames under StepNoShare.
var ErrNoPath = errors.New("align: no monotonic path")

// Times places every word on the frames: start and end frame, monotone and
// within [0, Frames] (docs/FINDINGS.md F-033).
func Times(in Input, words []Word, opt Options) ([]Span, error) {
	err := opt.Validate()
	if err != nil {
		return nil, err
	}

	m, rows, err := Matrix(in, opt)
	if err != nil {
		return nil, err
	}

	starts, ends := DTW(m, rows, in.Frames, opt.Step)
	if starts == nil {
		return nil, ErrNoPath
	}

	out := spans(starts, ends, words, opt, in.Frames)

	if opt.Onset > 0 && len(out) > 0 {
		on, ok := Onset(in.Levels, opt.Onset)
		if !ok {
			return nil, errors.New("align: Options.Onset needs Input.Levels")
		}

		out[0].Start = min(max(out[0].Start, on), out[0].End)
	}

	return out, nil
}

// Onset is the speech onset on the 80 ms grid: the first of levels (dB per
// 10 ms frame) within db of their nearest-rank 99th percentile (the
// silence gate's, docs/FINDINGS.md F-031), its index over 8 rounded. ok is
// false without levels.
func Onset(levels []float64, db float64) (frame int, ok bool) {
	if len(levels) == 0 {
		return 0, false
	}

	sorted := slices.Clone(levels)
	slices.Sort(sorted)

	p99 := silence.Percentile(sorted, 99)

	for i, v := range levels {
		if v >= p99-db {
			return (i + 4) / 8, true
		}
	}

	return (len(levels) + 4) / 8, true
}

// Matrix is the alignment matrix of opt: the selected heads' attention on
// the DTW rows (Lead rows, then one per text token and one for EOS),
// normalised, filtered and averaged, as costs, rows × in.Frames values.
func Matrix(in Input, opt Options) ([]float64, int, error) {
	shift := 0
	if opt.Rows == RowsPredict {
		shift = -1
	}

	top := in.First + shift - opt.Lead
	rows := opt.Lead + in.Tokens + 1

	if top < 0 {
		return nil, 0, fmt.Errorf("align: %d lead rows before position %d", opt.Lead, in.First+shift)
	}

	heads := opt.Heads
	if len(heads) == 0 {
		heads = LastHalf(in.Attn)
	}

	if len(heads) == 0 {
		return nil, 0, errors.New("align: no attention captured")
	}

	f := in.Frames
	sum := make([]float64, rows*f)

	for _, h := range heads {
		buf, err := HeadMatrix(in, h, top, rows, opt)
		if err != nil {
			return nil, 0, err
		}

		for i, v := range buf {
			sum[i] += v
		}
	}

	for i, v := range sum {
		a := v / float64(len(heads))
		if opt.Cost == CostNegLog {
			sum[i] = -math.Log(math.Max(a, 1e-30))
		} else {
			sum[i] = -a
		}
	}

	return sum, rows, nil
}

// LastHalf is Whisper's default alignment heads on a capture: every head
// of the last half of the layers it holds, all_heads[n_text_layer // 2:].
func LastHalf(attn []decoder.Attention) []Head {
	heads := map[int]int{}
	for _, a := range attn {
		heads[a.Layer] = a.Heads
	}

	layers := make([]int, 0, len(heads))
	for l := range heads {
		layers = append(layers, l)
	}

	slices.Sort(layers)

	var out []Head

	for _, l := range layers[len(layers)/2:] {
		for h := range heads[l] {
			out = append(out, Head{Layer: l, Head: h})
		}
	}

	return out
}

// HeadMatrix is head h's attention on the positions top … top+rows−1,
// rows × in.Frames values, times its mean output gate per row under
// opt.Gate, normalised by opt.Norm and median filtered by opt.Filter.
func HeadMatrix(in Input, h Head, top, rows int, opt Options) ([]float64, error) {
	lo, n := top, rows
	if opt.Norm == NormFrameAll {
		lo, n = 0, positions(in, h.Layer)
	}

	if lo > top || lo+n < top+rows {
		return nil, fmt.Errorf("align: layer %d captured %d positions, want %d…%d", h.Layer, n, top, top+rows-1)
	}

	m := make([]float64, n*in.Frames)

	for r := range n {
		err := headRow(m[r*in.Frames:(r+1)*in.Frames], in, h, lo+r, opt.Gate)
		if err != nil {
			return nil, err
		}
	}

	norm := opt.Norm
	if norm == NormFrameAll {
		norm = NormFrame
	}

	Normalise(m, n, in.Frames, norm)
	m = m[(top-lo)*in.Frames : (top-lo+rows)*in.Frames]
	MedianFilter(m, rows, in.Frames, opt.Filter)

	return m, nil
}

// positions is the number of positions captured for layer l from 0 on.
func positions(in Input, l int) int {
	n := 0

	for _, a := range in.Attn {
		if a.Layer == l {
			n = max(n, a.Pos+a.Tokens)
		}
	}

	return n
}

// headRow copies head h's probabilities at position p into dst, times the
// head's mean output gate there when gate is set.
func headRow(dst []float64, in Input, h Head, p int, gate bool) error {
	for _, a := range in.Attn {
		if a.Layer != h.Layer || p < a.Pos || p >= a.Pos+a.Tokens {
			continue
		}

		if h.Head < 0 || h.Head >= a.Heads || a.Frames != in.Frames {
			break
		}

		i := (p-a.Pos)*a.Heads + h.Head
		g := 1.0

		if v := len(a.Gate) / max(1, a.Tokens*a.Heads); gate && v > 0 {
			g = 0
			for _, x := range a.Gate[i*v : (i+1)*v] {
				g += float64(x)
			}

			g /= float64(v)
		}

		for c, x := range a.Probs[i*a.Frames : (i+1)*a.Frames] {
			dst[c] = g * float64(x)
		}

		return nil
	}

	return fmt.Errorf("align: no capture of layer %d head %d at position %d over %d frames", h.Layer, h.Head, p,
		in.Frames)
}

// Normalise applies norm to m, rows × cols, in place.
func Normalise(m []float64, rows, cols int, norm Norm) {
	switch norm {
	case NormFrame:
		for c := range cols {
			zscore(m[c:], rows, cols)
		}
	case NormRow:
		for r := range rows {
			zscore(m[r*cols:], cols, 1)
		}
	}
}

// zscore standardises n values of v at the given stride (population
// standard deviation, as torch.std_mean(unbiased=False)).
func zscore(v []float64, n, stride int) {
	var mean float64
	for i := range n {
		mean += v[i*stride]
	}

	mean /= float64(n)

	var vr float64

	for i := range n {
		d := v[i*stride] - mean
		vr += d * d
	}

	sd := math.Sqrt(vr / float64(n))
	if sd == 0 {
		sd = 1
	}

	for i := range n {
		v[i*stride] = (v[i*stride] - mean) / sd
	}
}

// MedianFilter replaces every value of m, rows × cols, by the median of the
// width values around it in its row, the row reflected at its ends:
// Whisper's median_filter (torch reflect padding). Rows shorter than the
// half width are left alone.
func MedianFilter(m []float64, rows, cols, width int) {
	half := width / 2
	if half == 0 || cols <= half {
		return
	}

	pad := make([]float64, cols+2*half)
	win := make([]float64, width)

	for r := range rows {
		row := m[r*cols : (r+1)*cols]

		for i := range pad {
			j := i - half
			if j < 0 {
				j = -j
			}

			if j >= cols {
				j = 2*(cols-1) - j
			}

			pad[i] = row[j]
		}

		for c := range row {
			copy(win, pad[c:c+width])
			slices.Sort(win)
			row[c] = win[half]
		}
	}
}

// spans reads each word's frames off the path.
func spans(starts, ends []int, words []Word, opt Options, frames int) []Span {
	out := make([]Span, len(words))
	prev := 0

	for i, w := range words {
		a, b := w.Start, w.End
		if opt.Core {
			a, b = w.Core[0], w.Core[1]
		}

		s := starts[opt.Lead+a]

		e := starts[opt.Lead+b]
		if opt.End == EndLast {
			e = ends[opt.Lead+b-1] + 1
		}

		s = min(max(s+opt.Offset, prev, 0), frames)
		e = min(max(e+opt.Offset, s), frames)
		out[i] = Span{Start: s, End: e}
		prev = s
	}

	return out
}
