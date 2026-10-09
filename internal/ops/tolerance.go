// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/ops/tolerance.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: without the layer_norm, mlp and convtranspose1d entries; values unchanged.

package ops

import "fmt"

// Tolerance defines acceptable numeric drift of a kernel versus a float64 or
// framework reference output.
type Tolerance struct {
	Abs float64
	Rel float64
}

// KernelTolerances defines per-kernel parity targets. The values are
// go-pocket-tts' ONNX parity targets, kept unchanged for the kernels this
// package still has (docs/FINDINGS.md F-008). They are kernel-level bounds;
// the Needle block fixtures of Phase 2 set their own, tighter acceptance.
var KernelTolerances = map[string]Tolerance{
	"matmul":      {Abs: 1e-4, Rel: 1e-4},
	"linear":      {Abs: 1e-4, Rel: 1e-4},
	"softmax":     {Abs: 1e-4, Rel: 1e-4},
	"causal_mask": {Abs: 0, Rel: 0},
	"rope":        {Abs: 2e-4, Rel: 2e-4},
	"attention":   {Abs: 2e-4, Rel: 2e-4},
	"conv1d":      {Abs: 2e-4, Rel: 2e-4},
}

func KernelTolerance(name string) (Tolerance, error) {
	t, ok := KernelTolerances[name]
	if !ok {
		return Tolerance{}, fmt.Errorf("ops: no tolerance configured for kernel %q", name)
	}

	return t, nil
}
