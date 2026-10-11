package encoder

import (
	"fmt"
	"math"

	"github.com/cwbudde/algo-tensor/ops"
	"github.com/cwbudde/algo-tensor/tensor"
	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// stemStages is the number of depthwise-separable stages after the full
// conv (stem/dw_{1,2}, stem/pw_{1,2}).
const stemStages = 2

// stem is the 2-D conv stem over time × mel (PLAN.md card, step 3):
//
//	conv 3×3, 1 → C (stem/w)                    → act
//	2 × [depthwise 3×3 (stem/dw_k) → act → pointwise C → C (stem/pw_k) → act]
//	flatten (mel', C) → stem/out [1280 → 512]
//
// None of these tensors has a bias. Strides, padding, the kernel's axis
// order, the flatten order and the activations are Hypothesis axes
// (docs/FINDINGS.md F-014).
type stem struct {
	h        Hypothesis
	channels int

	w    *tensor.Tensor    // [3, 3, 1, C]
	dw   [2]*tensor.Tensor // [3, 3, 1, C]
	pw   [2]needle.Linear  // [C, C] (out, in)
	out  needle.Linear     // [dim, mels'·C]
	dim  int
	in   int // stem/out's input width
	mels int // the input's mel bins

	peGate float32 // σ(pe_gate), for PEFlatConcat and PEFlatInterleaved
}

func loadStem(src weights.Source, h Hypothesis, mels int) (*stem, error) {
	s := &stem{h: h, mels: mels}

	var err error

	s.w, s.channels, err = loadKernel(src, "stem.w", h.StemKernel)
	if err != nil {
		return nil, err
	}

	for k := range stemStages {
		s.dw[k], _, err = loadKernel(src, fmt.Sprintf("stem.dw_%d", k+1), h.StemKernel)
		if err != nil {
			return nil, err
		}

		s.pw[k], err = needle.LoadLinear(src, fmt.Sprintf("stem.pw_%d", k+1), s.channels, s.channels)
		if err != nil {
			return nil, err
		}
	}

	shape, err := src.Shape("stem.out")
	if err != nil {
		return nil, fmt.Errorf("encoder: stem.out: %w", err)
	}

	if len(shape) != 2 {
		return nil, fmt.Errorf("encoder: stem.out has shape %v, want rank 2", shape)
	}

	s.dim, s.in = shape[0], shape[1]

	s.out, err = needle.LoadLinear(src, "stem.out", s.dim, s.in)
	if err != nil {
		return nil, err
	}

	if m := s.length(mels); m*s.channels != s.in {
		return nil, fmt.Errorf("%w: stride=%s pad=%s leaves %d mel positions × %d channels, stem.out takes %d",
			ErrRuledOut, axes[0].Names[h.StemStride-1], axes[1].Names[h.StemPad-1], m, s.channels, s.in)
	}

	return s, nil
}

// loadKernel reads a (9, C) stem kernel as HWIO [3, 3, 1, C]; KernelMelTime
// swaps the two spatial axes, so that H runs over time either way.
func loadKernel(src weights.Source, name string, layout KernelLayout) (*tensor.Tensor, int, error) {
	data, shape, err := src.Tensor(name)
	if err != nil {
		return nil, 0, fmt.Errorf("encoder: %s: %w", name, err)
	}

	if len(shape) != 2 || shape[0] != 9 {
		return nil, 0, fmt.Errorf("encoder: %s has shape %v, want (9, C)", name, shape)
	}

	c := shape[1]
	k := make([]float32, len(data))

	for i := range 3 {
		for j := range 3 {
			from, to := (i*3+j)*c, (i*3+j)*c
			if layout == KernelMelTime {
				to = (j*3 + i) * c
			}

			copy(k[to:to+c], data[from:from+c])
		}
	}

	t, err := tensor.New(k, []int64{3, 3, 1, int64(c)})

	return t, c, err
}

// matrix reads a rank-2 tensor of the given shape.
func matrix(src weights.Source, name string, rows, cols int) ([]float32, error) {
	data, shape, err := src.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("encoder: %s: %w", name, err)
	}

	if len(shape) != 2 || shape[0] != rows || shape[1] != cols {
		return nil, fmt.Errorf("encoder: %s has shape %v, want (%d, %d)", name, shape, rows, cols)
	}

	return data, nil
}

// stride is the (time, mel) stride of the full conv (stage 0) or of a
// depthwise stage.
func (s *stem) stride(stage int) int {
	if stage == 0 && s.h.StemStride == StrideDepthwise {
		return 1
	}

	return 2
}

// pad is the padding of a 3-tap conv over n positions.
func (s *stem) pad(n, stride int) (lo, hi int) {
	switch s.h.StemPad {
	case PadSymmetric:
		return 1, 1
	case PadValid:
		return 0, 0
	default:
		return ops.SamePadding(n, 3, stride)
	}
}

// length is the stem's output length along an axis of n positions (mel
// frames, or mel bins); 0 when the padded axis is shorter than a kernel.
func (s *stem) length(n int) int {
	for stage := range stemStages + 1 {
		st := s.stride(stage)

		lo, hi := s.pad(n, st)
		if n+lo+hi < 3 {
			return 0
		}

		n = (n+lo+hi-3)/st + 1
	}

	return n
}

func (s *stem) conv(x, k *tensor.Tensor, stage, groups int) (*tensor.Tensor, error) {
	shape := x.Shape()
	st := s.stride(stage)
	top, bottom := s.pad(int(shape[1]), st)
	left, right := s.pad(int(shape[2]), st)

	return ops.Conv2D(x, k, nil, ops.Conv2DParams{
		StrideH: st, StrideW: st, PadTop: top, PadBottom: bottom, PadLeft: left, PadRight: right, Groups: groups,
	})
}

// stemChunk is the number of output frames the stem computes at once
// (docs/FINDINGS.md F-038). The whole first conv of a 30 s clip is 1500 ×
// 40 × 128 floats, 29 MiB live at once (F-037); a chunk of 8 frames needs
// 35 of its 1500 rows, and the chunks run on tensor.Parallel. A multiple
// of 4 starts every chunk on a 4-row block of the pointwise products.
const stemChunk = 8

// forward maps mel (frames rows of s.mels values) to T rows of dim values,
// stemChunk output frames at a time. Each chunk recomputes the few rows
// its convs share with the next one (one per stride-2 stage). A conv
// pixel reads the same taps and skips the same padding as in whole, and
// the activations work value by value. The pointwise products sum some
// rows in blocks of 4 (the NEON float GEMM; cq's nibble4 kernel, which
// purego and amd64 without AVX2 run), so the outputs equal whole's bit for
// bit where every chunk starts on such a block: for Best's mel widths of
// 20 and 10, not for VALID padding's 19 and 9.
func (s *stem) forward(mel []float32, frames int) ([]float32, int, error) {
	t := s.length(frames)
	if t == 0 || s.length(s.mels)*s.channels != s.in {
		return s.whole(mel, frames) // reports the error
	}

	var n [stemStages + 2]int // input length of each conv stage, then T

	n[0] = frames

	for stage := range stemStages + 1 {
		st := s.stride(stage)
		lo, hi := s.pad(n[stage], st)
		n[stage+1] = (n[stage]+lo+hi-3)/st + 1
	}

	out := make([]float32, t*s.dim)
	chunks := (t + stemChunk - 1) / stemChunk
	errs := make([]error, chunks)

	tensor.Parallel(chunks, 1, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			o0, o1 := c*stemChunk, min(t, (c+1)*stemChunk)
			errs[c] = s.chunk(out[o0*s.dim:o1*s.dim], mel, n, o0, o1)
		}
	})

	for _, err := range errs {
		if err != nil {
			return nil, 0, err
		}
	}

	return out, t, nil
}

// chunk writes output frames [o0, o1) of the stem to dst: the rows each
// conv stage needs, back from the output, then the stages on those rows.
func (s *stem) chunk(dst, mel []float32, n [stemStages + 2]int, o0, o1 int) error {
	var lo, hi [stemStages + 2]int // rows [lo, hi) of each stage's input, then of the output

	lo[stemStages+1], hi[stemStages+1] = o0, o1
	for stage := stemStages; stage >= 0; stage-- {
		st := s.stride(stage)
		pad, _ := s.pad(n[stage], st)
		lo[stage] = max(0, lo[stage+1]*st-pad)
		hi[stage] = min(n[stage], (hi[stage+1]-1)*st-pad+3)
	}

	x, err := tensor.New(mel[lo[0]*s.mels:hi[0]*s.mels], []int64{1, int64(hi[0] - lo[0]), int64(s.mels), 1})
	if err != nil {
		return err
	}

	x, err = s.convRows(x, s.w, 0, 1, n[0], lo[0], lo[1], hi[1])
	if err != nil {
		return fmt.Errorf("encoder: stem conv: %w", err)
	}

	applyAct(x.RawData(), s.h.StemConvAct)

	for k := range stemStages {
		x, err = s.convRows(x, s.dw[k], k+1, s.channels, n[k+1], lo[k+1], lo[k+2], hi[k+2])
		if err != nil {
			return fmt.Errorf("encoder: stem dw_%d: %w", k+1, err)
		}

		s.pointwise(x.RawData(), k)
	}

	s.output(dst, x, o0, o1-o0)

	return nil
}

// convRows is conv for output rows [o0, o1) of stage on x, the input rows
// from inLo on of an axis of n rows. It pads the chunk where the whole
// axis is padded and nowhere else, so each output pixel reads the taps
// that conv reads for it, in the same order.
func (s *stem) convRows(x, k *tensor.Tensor, stage, groups, n, inLo, o0, o1 int) (*tensor.Tensor, error) {
	shape := x.Shape()
	st := s.stride(stage)
	pad, _ := s.pad(n, st)
	top := pad + inLo - o0*st
	bottom := max(0, (o1-o0-1)*st+3-int(shape[1])-top)
	left, right := s.pad(int(shape[2]), st)

	return ops.Conv2D(x, k, nil, ops.Conv2DParams{
		StrideH: st, StrideW: st, PadTop: top, PadBottom: bottom, PadLeft: left, PadRight: right, Groups: groups,
	})
}

// pointwise runs depthwise stage k's activation, pointwise product and its
// activation on d, rows of channels values, in place.
func (s *stem) pointwise(d []float32, k int) {
	applyAct(d, s.h.StemDWAct)

	y := make([]float32, len(d))
	s.pw[k].Apply(y, d, len(d)/s.channels)
	applyAct(y, s.h.StemPWAct)
	copy(d, y)
}

// output flattens x, t time rows of the last stage starting at frame p0,
// adds a flat pe_gate signal and writes stem.out's t rows to dst.
func (s *stem) output(dst []float32, x *tensor.Tensor, p0, t int) {
	shape := x.Shape()
	m, c := int(shape[2]), int(shape[3])

	flat := x.RawData()
	if s.h.StemFlatten == FlattenChannelMel {
		nhwc := flat
		flat = make([]float32, len(nhwc))

		for i := range t {
			for f := range m {
				for ch := range c {
					flat[(i*c+ch)*m+f] = nhwc[(i*m+f)*c+ch]
				}
			}
		}
	}

	if s.h.PE == PEFlatConcat || s.h.PE == PEFlatInterleaved {
		addSinusoidFrom(flat, p0, t, s.in, s.peGate, s.h.PE.interleaved())
	}

	s.out.Apply(dst, flat, t)
}

// whole is forward on the whole clip at once, as the stem ran before the
// chunks: the reference of TestStemChunks, and the path that reports a
// clip or a hypothesis the stem does not fit.
func (s *stem) whole(mel []float32, frames int) ([]float32, int, error) {
	x, err := tensor.New(mel, []int64{1, int64(frames), int64(s.mels), 1})
	if err != nil {
		return nil, 0, err
	}

	x, err = s.conv(x, s.w, 0, 1)
	if err != nil {
		return nil, 0, fmt.Errorf("encoder: stem conv: %w", err)
	}

	applyAct(x.RawData(), s.h.StemConvAct)

	for k := range stemStages {
		x, err = s.conv(x, s.dw[k], k+1, s.channels)
		if err != nil {
			return nil, 0, fmt.Errorf("encoder: stem dw_%d: %w", k+1, err)
		}

		s.pointwise(x.RawData(), k)
	}

	shape := x.Shape()
	t, m, c := int(shape[1]), int(shape[2]), int(shape[3])

	if m*c != s.in {
		return nil, 0, fmt.Errorf("%w: stem output %d × %d does not fit stem.out (%d)", ErrRuledOut, m, c, s.in)
	}

	out := make([]float32, t*s.dim)
	s.output(out, x, 0, t)

	return out, t, nil
}

// applyAct applies an activation in place.
func applyAct(x []float32, a Act) {
	switch a {
	case ActSiLU:
		for i, v := range x {
			x[i] = v * sigmoid(v)
		}
	case ActGELUTanh:
		const c = 0.7978845608028654 // √(2/π)

		for i, v := range x {
			f := float64(v)
			x[i] = float32(0.5 * f * (1 + math.Tanh(c*(f+0.044715*f*f*f))))
		}
	case ActGELUErf:
		for i, v := range x {
			f := float64(v)
			x[i] = float32(0.5 * f * (1 + math.Erf(f/math.Sqrt2)))
		}
	case ActReLU:
		for i, v := range x {
			x[i] = max(v, 0)
		}
	}
}

// sigmoid is jax.nn.sigmoid in float32.
func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}
