package ops

import (
	"maps"
	"testing"
)

// TestKernelTolerancesPinned pins the tolerances copied from go-pocket-tts
// internal/runtime/ops/tolerance.go @ 9faba38: any change must be a decision
// recorded in docs/FINDINGS.md, not drift. The layer_norm, mlp and
// convtranspose1d entries were dropped together with their kernels (F-008).
func TestKernelTolerancesPinned(t *testing.T) {
	want := map[string]Tolerance{
		"matmul":      {Abs: 1e-4, Rel: 1e-4},
		"linear":      {Abs: 1e-4, Rel: 1e-4},
		"softmax":     {Abs: 1e-4, Rel: 1e-4},
		"causal_mask": {Abs: 0, Rel: 0},
		"rope":        {Abs: 2e-4, Rel: 2e-4},
		"attention":   {Abs: 2e-4, Rel: 2e-4},
		"conv1d":      {Abs: 2e-4, Rel: 2e-4},
	}

	if !maps.Equal(KernelTolerances, want) {
		t.Fatalf("KernelTolerances = %v, want %v", KernelTolerances, want)
	}
}
