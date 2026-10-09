package needle

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Pinned inputs as laid out by `whistle download` and `just needle-fixtures`.
const (
	whistleSafetensorsPath = "../../models/whistle/whistle.safetensors"
	whistleCactPath        = "../../models/whistle/whistle.cact"
	fixtureDir             = "../../testdata/needle"
)

// openWhistle opens whistle.safetensors, the float32 weights the fixtures
// were made from (testdata/needle/README.md), or skips the test.
func openWhistle(t *testing.T) weights.Source {
	t.Helper()

	return openModel(t, whistleSafetensorsPath)
}

func openModel(t *testing.T, path string) weights.Source {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", path)
	}

	src, err := weights.Open(path, weights.Options{})
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}

	t.Cleanup(func() { _ = src.Close() })

	return src
}

// fixture is one testdata/needle file.
type fixture struct {
	t     *testing.T
	name  string
	store *safetensors.Store
}

func openFixture(t *testing.T, name string) fixture {
	t.Helper()

	store, err := safetensors.OpenStore(filepath.Join(fixtureDir, name+".safetensors"), safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("fixture %s: %v", name, err)
	}

	t.Cleanup(store.Close)

	return fixture{t: t, name: name, store: store}
}

// get returns a fixture tensor and its shape.
func (f fixture) get(name string) ([]float32, []int) {
	f.t.Helper()

	ts, err := f.store.Tensor(name)
	if err != nil {
		f.t.Fatalf("%s.%s: %v", f.name, name, err)
	}

	shape := make([]int, len(ts.Shape))
	for i, d := range ts.Shape {
		shape[i] = int(d)
	}

	return ts.Data, shape
}

// metadata returns a metadata value of the fixture.
func (f fixture) metadata(key string) string {
	f.t.Helper()

	v, ok := f.store.Metadata()[key]
	if !ok {
		f.t.Fatalf("%s: no metadata %q", f.name, key)
	}

	return v
}

// stats are the contract's two distances (testdata/needle/README.md).
type stats struct {
	maxAbs, refMax, cos float64
}

func compare(got, want []float32) stats {
	var s stats

	var dot, gg, ww float64

	for i, w := range want {
		g := float64(got[i])
		wf := float64(w)
		s.maxAbs = math.Max(s.maxAbs, math.Abs(g-wf))
		s.refMax = math.Max(s.refMax, math.Abs(wf))
		dot += g * wf
		gg += g * g
		ww += wf * wf
	}

	switch {
	case gg == 0 && ww == 0:
		s.cos = 1
	case gg == 0 || ww == 0:
		s.cos = 0
	default:
		s.cos = dot / math.Sqrt(gg*ww)
	}

	return s
}

// Contract of testdata/needle/README.md (docs/FINDINGS.md F-009).
const (
	contractRel = 1e-5
	contractCos = 0.99999
)

// checkContract holds got to the fixture contract, max |Δ| ≤ 1e-5 ·
// max(1, max |ref|) and cosine ≥ 0.99999, and logs both numbers.
func checkContract(t *testing.T, name string, got, want []float32) {
	t.Helper()

	if len(got) != len(want) {
		t.Errorf("%s: %d values, fixture has %d", name, len(got), len(want))

		return
	}

	s := compare(got, want)
	bound := contractRel * math.Max(1, s.refMax)
	t.Logf("%-12s max|ref| %9.3f  max abs %.3e (bound %.3e, %.2f of it)  cosine %.12f",
		name, s.refMax, s.maxAbs, bound, s.maxAbs/bound, s.cos)

	if s.maxAbs > bound || !(s.cos >= contractCos) {
		t.Errorf("%s: max abs %.3e > %.3e or cosine %.12f < %v", name, s.maxAbs, bound, s.cos, contractCos)
	}
}

// headsToRows reorders a per-head tensor (heads, T, d) into row layout
// (T, heads·d), the layout the blocks use.
func headsToRows(x []float32, heads, t, d int) []float32 {
	out := make([]float32, len(x))

	for h := range heads {
		for i := range t {
			copy(out[(i*heads+h)*d:(i*heads+h+1)*d], x[(h*t+i)*d:(h*t+i+1)*d])
		}
	}

	return out
}

func randVec(r *rand.Rand, n int, scale float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64() * scale)
	}

	return v
}

func maxAbsDiff64(got []float32, want []float64) float64 {
	var m float64
	for i, w := range want {
		m = math.Max(m, math.Abs(float64(got[i])-w))
	}

	return m
}

// equalInts reports whether a float32 fixture tensor holds exactly ints.
func equalInts(got []float32, want []int) bool {
	return slices.EqualFunc(got, want, func(g float32, w int) bool { return g == float32(w) })
}
