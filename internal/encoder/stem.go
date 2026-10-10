package encoder

import (
	"fmt"
	"math"

	"github.com/cwbudde/go-whistle/internal/ops"
	"github.com/cwbudde/go-whistle/internal/tensor"
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

	w   *tensor.Tensor    // [3, 3, 1, C]
	dw  [2]*tensor.Tensor // [3, 3, 1, C]
	pw  [2][]float32      // [C, C] (out, in)
	out []float32         // [dim, mels'·C]
	dim int
	in  int // stem/out's input width
}

func loadStem(src weights.Source, h Hypothesis, mels int) (*stem, error) {
	s := &stem{h: h}

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

		s.pw[k], err = matrix(src, fmt.Sprintf("stem.pw_%d", k+1), s.channels, s.channels)
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

	s.out, err = matrix(src, "stem.out", s.dim, s.in)
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

// forward maps mel (frames rows of mels values) to T rows of dim values.
func (s *stem) forward(mel []float32, frames, mels int) ([]float32, int, error) {
	x, err := tensor.New(mel, []int64{1, int64(frames), int64(mels), 1})
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

		d := x.RawData()
		applyAct(d, s.h.StemDWAct)

		y := make([]float32, len(d))
		tensor.MatMulTransB(y, s.channels, d, s.pw[k], len(d)/s.channels, s.channels, s.channels)
		applyAct(y, s.h.StemPWAct)
		copy(d, y)
	}

	shape := x.Shape()
	t, m, c := int(shape[1]), int(shape[2]), int(shape[3])

	if m*c != s.in {
		return nil, 0, fmt.Errorf("%w: stem output %d × %d does not fit stem.out (%d)", ErrRuledOut, m, c, s.in)
	}

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

	out := make([]float32, t*s.dim)
	tensor.MatMulTransB(out, s.dim, flat, s.out, t, s.dim, s.in)

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
