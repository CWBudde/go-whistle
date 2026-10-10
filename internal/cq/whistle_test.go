package cq

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/tensor"
)

const whistleCactPath = "../../models/whistle/whistle.cact"

func openWhistle(tb testing.TB) *cact.File {
	tb.Helper()

	_, err := os.Stat(whistleCactPath)
	if errors.Is(err, os.ErrNotExist) {
		tb.Skipf("%s missing; run `whistle download` to fetch the pinned weights", whistleCactPath)
	}

	f, err := cact.Open(whistleCactPath)
	if err != nil {
		tb.Fatalf("Open: %v", err)
	}

	return f
}

// whistleIndex finds a tensor of cact.WhistleNames by name.
func whistleIndex(tb testing.TB, name string) int {
	tb.Helper()

	i := slices.IndexFunc(cact.WhistleNames(), func(e cact.Entry) bool { return e.Name == name })
	if i < 0 {
		tb.Fatalf("no tensor %q in WhistleNames", name)
	}

	return i
}

// whistleTensors are the real tensors of the accuracy table in F-028: each
// CQ shape family and both widths. The engram table is a gather in the
// model; it is multiplied here as one more large CQ2 matrix.
var whistleTensors = []string{
	"layer00.q_proj",  // CQ2 (384, 512)
	"layer03.k_proj",  // CQ2 (96, 512)
	"cross05.v_proj",  // CQ2 (512, 512)
	"enc00.pw1",       // CQ2 (1024, 512)
	"enc07.out_proj",  // CQ2 (512, 512)
	"stem.pw_1",       // CQ2 (128, 128)
	"stem.out",        // CQ2 (512, 1280)
	"engram0.tables",  // CQ2 (73728, 128)
	"embedding",       // CQ4 (8199, 512), the tied head
	"mhc_phi_pre",     // CQ4 (32, 2048)
	"enc.mhc_phi_res", // CQ4 (128, 2048)
}

// TestWhistleGEMM is the 9.1 Accept: on real .cact tensors every kernel
// equals cact.CQ's dequantisation followed by the float GEMM of
// internal/tensor at cosine ≥ 0.99999. The activations are Gaussian rows
// and one heavy-tailed row (Gaussian cubed), five rows per GEMM: one block
// of four and one leftover row.
func TestWhistleGEMM(t *testing.T) {
	f := openWhistle(t)
	rng := rand.New(rand.NewPCG(13, 14))

	for _, name := range whistleTensors {
		i := whistleIndex(t, name)

		p, err := f.CQPacked(i)
		if err != nil {
			t.Fatalf("%s: CQPacked: %v", name, err)
		}

		m, err := New(p)
		if err != nil {
			t.Fatalf("%s: New: %v", name, err)
		}

		w, err := f.CQ(i)
		if err != nil {
			t.Fatalf("%s: CQ: %v", name, err)
		}

		const rows = 5

		x := make([]float32, rows*m.in)
		for k := range x {
			v := rng.NormFloat64()
			if k >= (rows-1)*m.in {
				v = v * v * v
			}

			x[k] = float32(v)
		}

		want := make([]float32, rows*m.out)
		tensor.MatMulTransB(want, m.out, x, w, rows, m.out, m.in)

		want64 := make([]float64, len(want))
		for k, v := range want {
			want64[k] = float64(v)
		}

		var wantMax float64
		for _, v := range want64 {
			wantMax = math.Max(wantMax, math.Abs(v))
		}

		for _, k := range kernels {
			got := make([]float32, rows*m.out)
			m.gemm(k, got, x, rows)

			minCos := 1.0
			for r := range rows {
				minCos = math.Min(minCos, cosine(got[r*m.out:(r+1)*m.out], want64[r*m.out:(r+1)*m.out]))
			}

			var maxAbs float64
			for j, v := range got {
				maxAbs = math.Max(maxAbs, math.Abs(float64(v)-want64[j]))
			}

			t.Logf("%-16s CQ%d (%d, %d) %-6s 1 − min cosine %.2e  max |Δ| %.3g (max |y| %.4g)",
				name, m.width, m.out, m.in, k, 1-minCos, maxAbs, wantMax)

			if minCos < 0.99999 {
				t.Errorf("%s %s: cosine %.9f < 0.99999", name, k, minCos)
			}
		}

		// Row is CQ's arithmetic, so it must agree bit for bit.
		row := make([]float32, m.in)
		for _, r := range []int{0, m.out / 2, m.out - 1} {
			m.Row(row, r)

			if !slices.Equal(row, w[r*m.in:(r+1)*m.in]) {
				t.Errorf("%s: Row(%d) differs from CQ", name, r)
			}
		}
	}
}

// TestWhistlePhiRows checks the per-layer view of a stacked mHC φ: layer
// l's lanes are rows [4l, 4l+4) (internal/needle mhc.go).
func TestWhistlePhiRows(t *testing.T) {
	f := openWhistle(t)

	p, err := f.CQPacked(whistleIndex(t, "mhc_phi_post"))
	if err != nil {
		t.Fatal(err)
	}

	m, err := New(p)
	if err != nil {
		t.Fatal(err)
	}

	x := make([]float32, m.in)
	for k := range x {
		x[k] = float32(math.Sin(float64(k)))
	}

	full := make([]float32, m.out)
	m.GEMV(full, x)

	for l := range m.out / 4 {
		part := make([]float32, 4)
		m.Rows(4*l, 4*l+4).GEMV(part, x)

		if !slices.Equal(part, full[4*l:4*l+4]) {
			t.Errorf("layer %d: Rows view differs from the full product", l)
		}
	}
}

// TestWhistleAllMatrices builds a Matrix from every CQ record of
// whistle.cact (all are 2- or 4-bit, group 128, stored [out, in]) and logs
// the memory they hold against float32 weights.
func TestWhistleAllMatrices(t *testing.T) {
	f := openWhistle(t)

	var n, packed, float int

	for i := range f.Len() {
		rec, _ := f.Record(i)
		if rec.DType != cact.CQ {
			continue
		}

		p, err := f.CQPacked(i)
		if err != nil {
			t.Fatalf("CQPacked(%d): %v", i, err)
		}

		m, err := New(p)
		if err != nil {
			t.Fatalf("New(%d): %v", i, err)
		}

		n++
		packed += m.Bytes()
		float += 4 * m.out * m.in
	}

	t.Logf("%d CQ matrices: %.2f MB packed (index + float32 norms + codebook), %.2f MB as float32 (%.1f×)",
		n, float64(packed)/1e6, float64(float)/1e6, float64(float)/float64(packed))

	if n != 152 {
		t.Errorf("built %d matrices, want 152", n)
	}
}
