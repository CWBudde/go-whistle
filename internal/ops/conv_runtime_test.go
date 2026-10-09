// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/ops/conv_runtime_test.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: clamps the largest int instead of maxInt32 + 123, which overflows a
// 32-bit int and broke the build on 386 and arm.

package ops

import "testing"

func TestSetConvWorkersClamp(t *testing.T) {
	SetConvWorkers(-5)

	if got := getConvWorkers(); got != 0 {
		t.Fatalf("getConvWorkers() = %d, want 0", got)
	}

	const maxInt32 = int(^uint32(0) >> 1)

	SetConvWorkers(int(^uint(0) >> 1)) // above maxInt32 where int has 64 bits

	if got := getConvWorkers(); got != maxInt32 {
		t.Fatalf("getConvWorkers() = %d, want %d", got, maxInt32)
	}

	SetConvWorkers(1)
}
