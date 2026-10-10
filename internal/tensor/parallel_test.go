package tensor

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

// withWorkers runs fn with SetWorkers(n) and restores the setting after.
func withWorkers(t *testing.T, n int, fn func()) {
	t.Helper()

	prev := workers.Load()
	defer workers.Store(prev)

	SetWorkers(n)
	fn()
}

// TestWorkersDefault checks that the default is GOMAXPROCS at start-up,
// which go test's -cpu flag may have changed since.
func TestWorkersDefault(t *testing.T) {
	prev := workers.Load()
	defer workers.Store(prev)

	workers.Store(0)

	if got := Workers(); got != defaultWorkers || got < 1 {
		t.Fatalf("Workers() = %d by default, want %d", got, defaultWorkers)
	}

	t.Logf("default %d workers, GOMAXPROCS now %d", defaultWorkers, runtime.GOMAXPROCS(0))
}

// TestParallelPieces checks that the pieces cover [0, n) once, start at
// multiples of grain, end at one except the last, are at most
// parallelSplit · Workers() many and differ by at most one grain.
func TestParallelPieces(t *testing.T) {
	for _, w := range []int{1, 2, 3, 8, 64} {
		for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 63, 64, 65, 1000} {
			for _, grain := range []int{0, 1, 4, 16} {
				withWorkers(t, w, func() {
					var mu sync.Mutex

					var got [][2]int

					hits := make([]int, n)

					Parallel(n, grain, func(lo, hi int) {
						mu.Lock()
						defer mu.Unlock()

						got = append(got, [2]int{lo, hi})

						for i := lo; i < hi; i++ {
							hits[i]++
						}
					})

					g := max(grain, 1)
					if len(got) > parallelSplit*w {
						t.Errorf("w %d n %d grain %d: %d pieces", w, n, grain, len(got))
					}

					short, long := n, 0

					for _, p := range got {
						if p[0]%g != 0 || (p[1] != n && p[1]%g != 0) || p[0] >= p[1] {
							t.Errorf("w %d n %d grain %d: piece %v", w, n, grain, p)
						}

						if p[1] != n {
							short, long = min(short, p[1]-p[0]), max(long, p[1]-p[0])
						}
					}

					if long > short+g {
						t.Errorf("w %d n %d grain %d: pieces of %d to %d items", w, n, grain, short, long)
					}

					for i, h := range hits {
						if h != 1 {
							t.Fatalf("w %d n %d grain %d: item %d run %d times", w, n, grain, i, h)
						}
					}
				})
			}
		}
	}
}

// TestParallelBoundsHelpers runs Parallel inside Parallel from several
// goroutines at once (the decoder's beams around a split GEMM) and checks
// that no more than Workers() − 1 helpers ever run besides the callers.
func TestParallelBoundsHelpers(t *testing.T) {
	const w, callers = 4, 5

	withWorkers(t, w, func() {
		var running, peak atomic.Int32

		busy := func() {
			r := running.Add(1)

			for {
				p := peak.Load()
				if r <= p || peak.CompareAndSwap(p, r) {
					break
				}
			}

			runtime.Gosched()
			running.Add(-1)
		}

		var wg sync.WaitGroup

		for range callers {
			wg.Go(func() {
				for range 50 {
					Parallel(64, 1, func(lo, hi int) {
						Parallel(hi-lo, 1, func(int, int) { busy() })
					})
				}
			})
		}

		wg.Wait()

		if h := helpers.Load(); h != 0 {
			t.Fatalf("%d helpers still counted after all calls returned", h)
		}

		if p := peak.Load(); p > callers+w-1 {
			t.Fatalf("%d goroutines ran fn at once, want at most %d", p, callers+w-1)
		}
	})
}

// TestMatMulTransBWorkers checks that the split product equals the
// one-worker product bit for bit, over row and column splits, leftover
// rows and columns, k tails and an ldc wider than n.
func TestMatMulTransBWorkers(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))

	shapes := [][3]int{
		{375, 384, 512},
		{375, 64, 375},
		{375, 375, 48},
		{301, 1027, 130},
		{1, 8199, 512},
		{5, 8199, 512},
		{7, 2051, 513},
		{9, 1024, 515},
		{4, 4096, 128},
	}

	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		if m*n*k < 2*gemmPieceFMAs {
			t.Fatalf("%v is below the split threshold", s)
		}

		a, b := randGemmData(rng, m*k), randGemmData(rng, n*k)
		ldc := n + 3

		var want []float32

		for _, w := range []int{1, 2, 3, 7, 16} {
			c := make([]float32, (m-1)*ldc+n+1)
			for i := range c {
				c[i] = gemmSentinel
			}

			withWorkers(t, w, func() { MatMulTransB(c, ldc, a, b, m, n, k) })

			if want == nil {
				want = c

				continue
			}

			if !slices.EqualFunc(c, want, func(x, y float32) bool {
				return math.Float32bits(x) == math.Float32bits(y)
			}) {
				t.Fatalf("m %d n %d k %d: %d workers differ from 1 worker", m, n, k, w)
			}
		}
	}
}
