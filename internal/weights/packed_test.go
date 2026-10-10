package weights

import (
	"errors"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
)

// TestWhistlePacked opens whistle.cact packed and in float mode
// (docs/FINDINGS.md F-037). Packed, Matrix serves exactly the 152 CQ
// records, at their shapes, cached per name, without a float32 decode, and
// each matrix's rows equal the float source's values bit for bit. In float
// mode and from safetensors no tensor is packed.
func TestWhistlePacked(t *testing.T) {
	requireWhistle(t)

	packed, err := OpenCact(whistleCactPath, Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	defer packed.Close()

	float := openWhistleSource(t, whistleCactPath, Options{})
	st := openWhistleSource(t, whistleSafetensorsPath, Options{Packed: true})

	if _, ok := st.(Packer); ok {
		t.Error("the safetensors backend is a Packer")
	}

	f, err := cact.Parse(mustRead(t, whistleCactPath))
	if err != nil {
		t.Fatal(err)
	}

	n := 0

	for i, e := range cact.WhistleNames() {
		rec, _ := f.Record(i)
		m, ok, err := packed.Matrix(e.Name)

		if err != nil || ok != (rec.DType == cact.CQ) {
			t.Fatalf("%s (%s): Matrix ok %v, %v", e.Name, rec.DType, ok, err)
		}

		_, fok, ferr := PackedMatrix(float, e.Name)
		if fok || ferr != nil {
			t.Fatalf("%s: float mode serves it packed (%v)", e.Name, ferr)
		}

		if !ok {
			continue
		}

		n++

		if m.Out() != e.Shape[0] || m.In() != e.Shape[1] {
			t.Errorf("%s: matrix [%d, %d], shape %v", e.Name, m.Out(), m.In(), e.Shape)
		}

		if again, _, _ := packed.Matrix(e.Name); again != m {
			t.Errorf("%s: a second Matrix call built a new matrix", e.Name)
		}

		checkRows(t, e.Name, m, float)
	}

	if n != 152 {
		t.Errorf("%d packed tensors, want 152", n)
	}

	if c := packed.Cached(); len(c) != 0 {
		t.Errorf("Matrix decoded float32 values: %v", c)
	}
}

// checkRows compares the first, a middle and the last row of m with the
// float source's values for name.
func checkRows(t *testing.T, name string, m *cq.Matrix, float Source) {
	t.Helper()

	want, _, err := float.Tensor(name)
	if err != nil {
		t.Fatal(err)
	}

	defer float.Release(name)

	row := make([]float32, m.In())

	for _, r := range []int{0, m.Out() / 2, m.Out() - 1} {
		m.Row(row, r)

		if !slices.Equal(row, want[r*m.In():(r+1)*m.In()]) {
			t.Errorf("%s: row %d differs from the dequantised tensor", name, r)
		}
	}
}

// TestWhistlePackedLifecycle checks Matrix's cache rules: concurrent first
// calls share one matrix, Release drops it, unknown names fail with
// ErrNotFound and a closed source with ErrClosed.
func TestWhistlePackedLifecycle(t *testing.T) {
	requireWhistle(t)

	src, err := OpenCact(whistleCactPath, Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	const name = "enc00.q_proj"

	got := make([]*cq.Matrix, 8)

	var wg sync.WaitGroup

	for g := range got {
		wg.Go(func() { got[g], _, _ = src.Matrix(name) })
	}

	wg.Wait()

	for _, m := range got {
		if m == nil || m != got[0] {
			t.Fatal("concurrent Matrix calls built different matrices")
		}
	}

	src.Release(name)

	if m, _, _ := src.Matrix(name); m == got[0] {
		t.Error("Release kept the matrix")
	}

	_, _, err = src.Matrix("nope")
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("unknown name: %v, want ErrNotFound", err)
	}

	err = src.Close()
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = src.Matrix(name)
	if !errors.Is(err, ErrClosed) {
		t.Errorf("after Close: %v, want ErrClosed", err)
	}
}

func openWhistleSource(t *testing.T, path string, opts Options) Source {
	t.Helper()

	src, err := Open(path, opts)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = src.Close() })

	return src
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	return data
}
