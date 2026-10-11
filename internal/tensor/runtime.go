// Copied from github.com/cwbudde/go-pocket-tts internal/runtime/tensor/runtime.go @ 9faba38;
// port fixes in either direction (PLAN.md, Architecture of the Go port).
// Change: the default is GOMAXPROCS instead of 1, and Parallel shares one
// helper budget between all callers (docs/FINDINGS.md F-038).

package tensor

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// workers controls goroutine parallelism for tensor math kernels such as
// Linear, MatMul, MatMulTransB and Parallel. 0, the default, stands for
// defaultWorkers; 1 disables parallel execution.
var workers atomic.Int32

// defaultWorkers is GOMAXPROCS at start-up. It is read once: GOMAXPROCS(0)
// takes the scheduler's lock, which every Parallel call contended for
// (docs/FINDINGS.md F-038).
var defaultWorkers = max(1, runtime.GOMAXPROCS(0))

// SetWorkers sets the maximum number of goroutines used by tensor kernels.
// n <= 1 disables kernel parallelism. The default is GOMAXPROCS at
// start-up.
func SetWorkers(n int) {
	const maxInt32 = int(^uint32(0) >> 1)

	if n < 1 {
		n = 1
	}

	if n > maxInt32 {
		n = maxInt32
	}

	workers.Store(int32(n))
}

func getWorkers() int {
	n := int(workers.Load())
	if n < 1 {
		return defaultWorkers
	}

	return n
}

// Workers returns the currently configured tensor-kernel worker count.
func Workers() int {
	return getWorkers()
}

func parallelFor(n, maxWorkers int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}

	if maxWorkers <= 1 || n == 1 {
		fn(0, n)
		return
	}

	if maxWorkers > n {
		maxWorkers = n
	}

	chunk := (n + maxWorkers - 1) / maxWorkers
	var wg sync.WaitGroup

	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)

		wg.Add(1)

		go func(lo, hi int) {
			defer wg.Done()

			fn(lo, hi)
		}(lo, hi)
	}

	wg.Wait()
}

// parallelSplit is the number of Parallel pieces per worker: more than
// one, so that the goroutines balance a core that is busy elsewhere
// (docs/FINDINGS.md F-038).
const parallelSplit = 4

// helpers counts the goroutines that Parallel runs besides its callers,
// over all calls at once. Parallel starts a helper only while fewer than
// Workers() − 1 run, so nested calls (a GEMM inside a parallel loop over
// heads) and concurrent ones (the decoder's beams, each running its own
// GEMMs) never add more than Workers() − 1 goroutines in total: a caller
// that finds the budget used up runs its pieces itself.
var helpers atomic.Int32

// Parallel calls fn on pieces [lo, hi) that cover [0, n) without overlap,
// on the calling goroutine and up to Workers() − 1 helpers. Every piece
// starts at a multiple of grain and, except the last, ends at one, so a
// caller can keep blocks of grain items together (MatMulTransB's 4-row
// blocks): if fn computes each block the same wherever a piece starts,
// the result does not depend on the worker count. There are up to
// parallelSplit · Workers() pieces whose sizes in grain units differ by at
// most one, whether or not helpers are free; the goroutines take them in
// turn, so a core that other work holds up takes fewer. Parallel returns
// when all pieces are done.
func Parallel(n, grain int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}

	grain = max(grain, 1)
	units := (n + grain - 1) / grain

	w := min(getWorkers(), units)
	if w <= 1 {
		fn(0, n)
		return
	}

	pieces := min(units, parallelSplit*w)

	var next atomic.Int32

	run := func() {
		for {
			p := int(next.Add(1)) - 1
			if p >= pieces {
				return
			}

			fn(p*units/pieces*grain, min(n, (p+1)*units/pieces*grain))
		}
	}

	limit := int32(getWorkers() - 1)

	var wg sync.WaitGroup

	for range pieces - 1 {
		if helpers.Add(1) > limit {
			helpers.Add(-1)
			break
		}

		wg.Go(func() {
			defer helpers.Add(-1)

			run()
		})
	}

	run()
	wg.Wait()
}
