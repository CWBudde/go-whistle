// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/ops/ops_bench_test.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: benchmarks use Whistle shapes instead of FlowLM/Mimi ones; the helpers are unchanged.

package ops

import (
	"fmt"
	"testing"

	"github.com/cwbudde/go-whistle/internal/tensor"
)

// The benchmarks use Whistle's shapes (PLAN.md reference card): width 512,
// 8 query heads of qk 48 and v 64, 375 encoder frames for a 30 s clip, and
// the depthwise kernel-9 conv of the encoder's conv module.

func BenchmarkLinearWhistleQProj(b *testing.B) {
	x := mustTensor(b, seqData(375*512), []int64{1, 375, 512})
	w := mustTensor(b, seqData(384*512), []int64{384, 512})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.Linear(x, w, nil)
		if err != nil {
			b.Fatalf("linear: %v", err)
		}
	}
}

func BenchmarkMatMulWhistleScores(b *testing.B) {
	q := mustTensor(b, seqData(8*375*48), []int64{1, 8, 375, 48})
	kT := mustTensor(b, seqData(8*48*375), []int64{1, 8, 48, 375})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := tensor.MatMul(q, kT)
		if err != nil {
			b.Fatalf("matmul: %v", err)
		}
	}
}

func BenchmarkAttentionWhistleEncoder(b *testing.B) {
	q := mustTensor(b, seqData(8*375*48), []int64{1, 8, 375, 48})
	k := mustTensor(b, seqData(8*375*48), []int64{1, 8, 375, 48})
	v := mustTensor(b, seqData(8*375*64), []int64{1, 8, 375, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, false, 0)
		if err != nil {
			b.Fatalf("attention: %v", err)
		}
	}
}

func BenchmarkAttentionWhistleDecodeStep(b *testing.B) {
	q := mustTensor(b, seqData(8*48), []int64{1, 8, 1, 48})
	k := mustTensor(b, seqData(8*64*48), []int64{1, 8, 64, 48})
	v := mustTensor(b, seqData(8*64*64), []int64{1, 8, 64, 64})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Attention(q, k, v, true, 63)
		if err != nil {
			b.Fatalf("attention: %v", err)
		}
	}
}

// BenchmarkConv1DWhistleDepthwise runs the grouped (scalar) path; a
// depthwise kernel would replace it.
func BenchmarkConv1DWhistleDepthwise(b *testing.B) {
	input := mustTensor(b, seqData(512*375), []int64{1, 512, 375})
	kernel := mustTensor(b, seqData(512*9), []int64{512, 1, 9})
	bias := mustTensor(b, seqData(512), []int64{512})
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		_, err := Conv1D(input, kernel, bias, 1, 4, 1, 512)
		if err != nil {
			b.Fatalf("conv1d: %v", err)
		}
	}
}

func BenchmarkConv1DWhistlePointwiseParallel(b *testing.B) {
	for _, workers := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
			SetConvWorkers(workers)
			defer SetConvWorkers(1)

			input := mustTensor(b, seqData(512*375), []int64{1, 512, 375})
			kernel := mustTensor(b, seqData(1024*512), []int64{1024, 512, 1})
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				_, err := Conv1D(input, kernel, nil, 1, 0, 1, 1)
				if err != nil {
					b.Fatalf("conv1d: %v", err)
				}
			}
		})
	}
}

func seqData(n int) []float32 {
	out := make([]float32, n)
	for i := range n {
		out[i] = float32((i%17)-8) / 17
	}

	return out
}

func mustTensor(tb testing.TB, data []float32, shape []int64) *tensor.Tensor {
	tb.Helper()

	t, err := tensor.New(data, shape)
	if err != nil {
		tb.Fatalf("new tensor: %v", err)
	}

	return t
}
