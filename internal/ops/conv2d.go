package ops

import (
	"errors"
	"fmt"

	"github.com/cwbudde/go-whistle/internal/tensor"
)

// Conv2DParams are a 2-D convolution's strides, zero padding per side and
// group count. Groups = in channels with one input channel per group is a
// depthwise convolution.
type Conv2DParams struct {
	StrideH, StrideW  int
	PadTop, PadBottom int
	PadLeft, PadRight int
	Groups            int
}

// SamePadding is lax.conv's "SAME" padding along one axis of length n for a
// kernel of k taps and a stride: the output has ceil(n / stride) positions,
// and of the total padding max((out − 1)·stride + k − n, 0) the low side
// gets the smaller half. For an even n, kernel 3 and stride 2 that is 0
// before and 1 after (docs/FINDINGS.md F-014).
func SamePadding(n, k, stride int) (lo, hi int) {
	out := (n + stride - 1) / stride
	total := max((out-1)*stride+k-n, 0)

	return total / 2, total - total/2
}

// Conv2D is a deterministic CPU 2-D convolution in Flax's layout, the one
// Whistle's conv stem stores (docs/FINDINGS.md F-014):
//
//	input  [batch, height, width, in_channels]          (NHWC)
//	kernel [kh, kw, in_channels/groups, out_channels]   (HWIO)
//	bias   [out_channels] or nil
//
// Output channel o belongs to group o / (out/groups) and reads that group's
// in/groups input channels. Products accumulate in float32 in kernel-row,
// kernel-column, input-channel order.
func Conv2D(input, kernel, bias *tensor.Tensor, p Conv2DParams) (*tensor.Tensor, error) {
	g, err := prepareConv2D(input, kernel, bias, p)
	if err != nil {
		return nil, err
	}

	out, err := tensor.Zeros([]int64{int64(g.batch), int64(g.outH), int64(g.outW), int64(g.outC)})
	if err != nil {
		return nil, err
	}

	x, k, o := input.RawData(), kernel.RawData(), out.RawData()

	var b []float32
	if bias != nil {
		b = bias.RawData()
	}

	img := g.inH * g.inW * g.inC

	for n := range g.batch {
		for oh := range g.outH {
			for ow := range g.outW {
				off := ((n*g.outH+oh)*g.outW + ow) * g.outC
				g.pixel(o[off:off+g.outC], x[n*img:(n+1)*img], k, b, oh, ow, p)
			}
		}
	}

	return out, nil
}

// conv2DGeom holds the checked sizes of one Conv2D call.
type conv2DGeom struct {
	batch, inH, inW, inC  int
	kh, kw, inPer, outPer int
	outH, outW, outC      int
}

// pixel accumulates one output position's out channels into dst from one
// image x (inH × inW × inC).
func (g conv2DGeom) pixel(dst, x, k, b []float32, oh, ow int, p Conv2DParams) {
	if b != nil {
		copy(dst, b)
	}

	for i := range g.kh {
		ih := oh*p.StrideH + i - p.PadTop
		if ih < 0 || ih >= g.inH {
			continue
		}

		for j := range g.kw {
			iw := ow*p.StrideW + j - p.PadLeft
			if iw < 0 || iw >= g.inW {
				continue
			}

			px := x[(ih*g.inW+iw)*g.inC : (ih*g.inW+iw+1)*g.inC]
			taps := k[(i*g.kw+j)*g.inPer*g.outC : (i*g.kw+j+1)*g.inPer*g.outC]

			for c := range g.inPer {
				row := taps[c*g.outC : (c+1)*g.outC]
				if g.outPer == g.outC { // one group: every output reads input channel c
					tensor.Axpy(dst, px[c], row)

					continue
				}

				for oc, w := range row {
					dst[oc] += px[oc/g.outPer*g.inPer+c] * w
				}
			}
		}
	}
}

//nolint:cyclop // one flat list of shape checks reads better than split helpers
func prepareConv2D(input, kernel, bias *tensor.Tensor, p Conv2DParams) (conv2DGeom, error) {
	if input == nil || kernel == nil {
		return conv2DGeom{}, errors.New("ops: conv2d requires non-nil input/kernel")
	}

	if p.StrideH <= 0 || p.StrideW <= 0 || p.Groups <= 0 {
		return conv2DGeom{}, errors.New("ops: conv2d strides and groups must be > 0")
	}

	if p.PadTop < 0 || p.PadBottom < 0 || p.PadLeft < 0 || p.PadRight < 0 {
		return conv2DGeom{}, errors.New("ops: conv2d padding must be >= 0")
	}

	in, ks := input.Shape(), kernel.Shape()
	if len(in) != 4 || len(ks) != 4 {
		return conv2DGeom{}, fmt.Errorf("ops: conv2d expects input/kernel rank 4, got %v and %v", in, ks)
	}

	g := conv2DGeom{
		batch: int(in[0]), inH: int(in[1]), inW: int(in[2]), inC: int(in[3]),
		kh: int(ks[0]), kw: int(ks[1]), inPer: int(ks[2]), outC: int(ks[3]),
	}

	if g.inC%p.Groups != 0 || g.outC%p.Groups != 0 || g.inPer != g.inC/p.Groups {
		return conv2DGeom{}, fmt.Errorf("ops: conv2d kernel %v does not fit %d input channels in %d groups", ks, g.inC, p.Groups)
	}

	if bias != nil && (bias.Rank() != 1 || bias.Shape()[0] != int64(g.outC)) {
		return conv2DGeom{}, fmt.Errorf("ops: conv2d bias shape %v does not match out_channels %d", bias.Shape(), g.outC)
	}

	if g.inH+p.PadTop+p.PadBottom < g.kh || g.inW+p.PadLeft+p.PadRight < g.kw {
		return conv2DGeom{}, fmt.Errorf("ops: conv2d padded input %d×%d is smaller than kernel %d×%d",
			g.inH+p.PadTop+p.PadBottom, g.inW+p.PadLeft+p.PadRight, g.kh, g.kw)
	}

	g.outPer = g.outC / p.Groups
	g.outH = (g.inH+p.PadTop+p.PadBottom-g.kh)/p.StrideH + 1
	g.outW = (g.inW+p.PadLeft+p.PadRight-g.kw)/p.StrideW + 1

	return g, nil
}
