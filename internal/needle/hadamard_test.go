package needle

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// TestKronApplyMatchesDense checks _kron_apply against the dense product
// z · (a ⊗ b): with np.kron's layout, (a ⊗ b)[i·nb + j, k·nb + l] =
// a[i, k] · b[j, l], which is exactly the einsum "ij,ik,jl->kl".
func TestKronApplyMatchesDense(t *testing.T) {
	for _, dims := range [][2]int{{2, 3}, {4, 4}, {16, 32}} {
		na, nb := dims[0], dims[1]
		n := na * nb
		r := rand.New(rand.NewPCG(uint64(na), uint64(nb)))
		a := randVec(r, na*na, 1)
		b := randVec(r, nb*nb, 1)
		z := randVec(r, n, 1)

		want := make([]float64, n)

		for row := range n {
			i, j := row/nb, row%nb
			for col := range n {
				k, l := col/nb, col%nb
				want[col] += float64(z[row]) * float64(a[i*na+k]) * float64(b[j*nb+l])
			}
		}

		got := make([]float32, n)
		kronApply(got, z, a, na, b, nb, make([]float32, n))

		var refMax float64
		for _, w := range want {
			refMax = math.Max(refMax, math.Abs(w))
		}

		if d := maxAbsDiff64(got, want); d > 1e-5*math.Max(1, refMax) {
			t.Errorf("%d×%d: max |Δ| %.3g at max|ref| %.3g", na, nb, d, refMax)
		}
	}
}

// tinyHadamard is a random HadamardMLP of width 8 (blocks 2 × 4) with
// rank-3 conditioning and its own permutations.
func tinyHadamard(r *rand.Rand, prefix string) *mapSource {
	src := newMapSource()

	for _, name := range []string{"d1", "d2", "b2", "d3", "d4"} {
		src.put(prefix+name, randVec(r, 8, 0.8), 8)
	}

	for _, s := range []string{"1", "2", "3"} {
		src.put(prefix+"w"+s+"a", randVec(r, 4, 0.7), 2, 2)
		src.put(prefix+"w"+s+"b", randVec(r, 16, 0.5), 4, 4)
	}

	src.put(prefix+"cond_v", randVec(r, 8*3, 0.6), 8, 3)
	src.put(prefix+"cond_u", randVec(r, 3*8, 0.6), 3, 8)
	src.put("hada_p1", []float32{3, 0, 7, 5, 1, 6, 2, 4}, 8)
	src.put("hada_p2", []float32{6, 2, 0, 1, 7, 4, 5, 3}, 8)

	return src
}

// refHadamard transcribes architecture.py HadamardMLP in float64 for one
// row.
func refHadamard(src *mapSource, prefix string, x []float64, na, nb, rank int) []float64 {
	w := func(name string) []float64 { return toFloat64(src.data[name]) }
	n := na * nb
	p1, p2 := w("hada_p1"), w("hada_p2")
	d1, d2, b2, d3, d4 := w(prefix+"d1"), w(prefix+"d2"), w(prefix+"b2"), w(prefix+"d3"), w(prefix+"d4")
	cv, cu := w(prefix+"cond_v"), w(prefix+"cond_u")

	logits := make([]float64, rank)
	for k := range rank {
		for i := range n {
			logits[k] += x[i] * cv[i*rank+k]
		}
	}

	m := slices.Max(logits)

	var sum float64

	for k := range logits {
		logits[k] = math.Exp(logits[k] - m)
		sum += logits[k]
	}

	cond := make([]float64, n)
	for c := range n {
		cond[c] = 1
		for k := range rank {
			cond[c] += logits[k] / sum * cu[k*n+c]
		}
	}

	kron := func(z, a, b []float64) []float64 {
		out := make([]float64, n)

		for i := range na {
			for j := range nb {
				for k := range na {
					for l := range nb {
						out[k*nb+l] += z[i*nb+j] * a[i*na+k] * b[j*nb+l]
					}
				}
			}
		}

		return out
	}
	gather := func(z, p []float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = z[int(p[i])]
		}

		return out
	}
	scale := func(d, z []float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = d[i] * z[i]
		}

		return out
	}

	z := gather(kron(scale(d1, x), w(prefix+"w1a"), w(prefix+"w1b")), p1)
	for i := range z {
		a := d2[i]*cond[i]*z[i] + b2[i]
		z[i] = a / (1 + math.Exp(-a))
	}

	z = gather(kron(z, w(prefix+"w2a"), w(prefix+"w2b")), p2)
	z = kron(scale(d3, z), w(prefix+"w3a"), w(prefix+"w3b"))

	return scale(d4, z)
}

func TestHadamardMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 13))
	src := tinyHadamard(r, "enc00.hada_0.")

	h, err := LoadHadamardMLP(src, "enc00.hada_0.")
	if err != nil {
		t.Fatal(err)
	}

	const steps = 3

	x := randVec(r, steps*8, 1.5)

	got, err := h.Forward(x, steps)
	if err != nil {
		t.Fatal(err)
	}

	for i := range steps {
		want := refHadamard(src, "enc00.hada_0.", toFloat64(x[i*8:(i+1)*8]), 2, 4, 3)
		if d := maxAbsDiff64(got[i*8:(i+1)*8], want); d > 1e-5 {
			t.Errorf("row %d: max |Δ| %.3g: %v vs %v", i, d, got[i*8:(i+1)*8], want)
		}
	}
}

func TestHadamardLoadErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))

	cases := map[string]func(*mapSource){
		"perm not a permutation": func(s *mapSource) { s.put("hada_p1", []float32{0, 0, 1, 2, 3, 4, 5, 6}, 8) },
		"perm not an integer":    func(s *mapSource) { s.put("hada_p2", []float32{0.5, 1, 2, 3, 4, 5, 6, 7}, 8) },
		"blocks do not tile n":   func(s *mapSource) { s.put("w1b", randVec(r, 9, 1), 3, 3) },
		"factor not square":      func(s *mapSource) { s.put("w2a", randVec(r, 4, 1), 1, 4) },
		"cond_u rank mismatch":   func(s *mapSource) { s.put("cond_u", randVec(r, 16, 1), 2, 8) },
		"missing d4":             func(s *mapSource) { delete(s.data, "d4") },
	}

	for name, mutate := range cases {
		src := tinyHadamard(r, "")
		mutate(src)

		_, err := LoadHadamardMLP(src, "")
		if err == nil {
			t.Errorf("%s: LoadHadamardMLP accepted it", name)
		}
	}

	h, err := LoadHadamardMLP(tinyHadamard(r, ""), "")
	if err != nil {
		t.Fatal(err)
	}

	_, err = h.Forward(make([]float32, 9), 1)
	if err == nil {
		t.Error("Forward accepted 9 values for one row of 8")
	}
}

// TestHadaPermsFixture pins _hada_perms(512) of the fixture, which
// upstream's numpy RandomState(11 / 13).permutation(512) drew, to the Go
// MT19937 port the safetensors backend derives hada_p1/p2 from.
func TestHadaPermsFixture(t *testing.T) {
	fx := openFixture(t, "hadamard_mlp")

	for name, seed := range map[string]uint32{"p1": 11, "p2": 13} {
		got, _ := fx.get(name)
		if want := cact.NumpyPermutation(seed, 512); !equalInts(got, want) {
			t.Errorf("fixture %s differs from NumpyPermutation(%d, 512)", name, seed)
		}
	}
}

// TestHadaPermsBothBackends pins the permutations a HadamardMLP reads from
// .cact records 226/227 to the ones the safetensors backend derives with
// the numpy port, and both to the fixture (docs/FINDINGS.md F-011).
// internal/weights TestWhistleCrossFormat compares every derived record
// across the backends as well.
func TestHadaPermsBothBackends(t *testing.T) {
	st := openModel(t, whistleSafetensorsPath)
	ct := openModel(t, whistleCactPath)
	fx := openFixture(t, "hadamard_mlp")

	hs, err := LoadHadamardMLP(st, "layer00.")
	if err != nil {
		t.Fatal(err)
	}

	hc, err := LoadHadamardMLP(ct, "layer00.")
	if err != nil {
		t.Fatal(err)
	}

	for i, name := range []string{"p1", "p2"} {
		want, _ := fx.get(name)
		ps, pc := hs.perm[i], hc.perm[i]

		if !slices.Equal(ps, pc) {
			t.Errorf("hada_%s: .cact record and safetensors derivation differ", name)
		}

		if !equalInts(want, ps) {
			t.Errorf("hada_%s differs from the fixture's", name)
		}
	}
}

// TestHadamardFixture is decoder layer 0's HadamardMLP against
// testdata/needle/hadamard_mlp.safetensors, every intermediate included.
func TestHadamardFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "hadamard_mlp")

	h, err := LoadHadamardMLP(src, "layer00.")
	if err != nil {
		t.Fatal(err)
	}

	if fx.metadata("blocks") != "16x32" || fx.metadata("cond_rank") != "8" || fx.metadata("split_perms") != "False" {
		t.Fatalf("fixture geometry %s rank %s split %s", fx.metadata("blocks"), fx.metadata("cond_rank"),
			fx.metadata("split_perms"))
	}

	if h.na != 16 || h.nb != 32 || h.rank != 8 {
		t.Fatalf("layer00 geometry %d×%d rank %d", h.na, h.nb, h.rank)
	}

	x, shape := fx.get("x")
	rows := shape[0] * shape[1]
	tr := &hadaTrace{}

	out, err := h.forward(x, rows, tr)
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name string
		got  []float32
	}{
		{"cond_logits", tr.condLogits},
		{"cond", tr.cond},
		{"z1", tr.z1},
		{"a2", tr.a2},
		{"z2", tr.z2},
		{"z3", tr.z3},
		{"out", out},
	} {
		want, _ := fx.get(c.name)
		checkContract(t, c.name, c.got, want)
	}
}
