package needle

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"testing"
)

// TestNgramHashHandComputed pins the hash to values worked out by hand
// (the first also on paper): seed = 0x9E3779B9 · (col + 1) mod 2³², then
// for j < order acc = (acc ^ tok[t − j]) · 0x01000193 mod 2³², then
// acc ^= acc >> 15. Every case wraps around 2³² several times.
func TestNgramHashHandComputed(t *testing.T) {
	cases := []struct {
		col, order int
		hist       []int // hist[len − 1] is the current token
		want       uint32
	}{
		{0, 1, []int{5}, 0xCD55385D},
		{0, 2, []int{7, 42}, 0x1ED6CE87},
		{3, 3, []int{1, 2, 3}, 0x420599E6},
		{1, 2, []int{8198}, 0x7922F0D0},        // the token before the start counts as 0
		{2, 3, []int{0, 0, 0}, 0xEBB8DBEB},     // pad ids
		{0, 2, nil, 0xBE346A8B},                // no history at all
		{0, 2, []int{9, 9, 7, 42}, 0x1ED6CE87}, // only the last order tokens count
	}

	for _, c := range cases {
		if got := ngramHash(c.col, c.order, c.hist); got != c.want {
			t.Errorf("ngramHash(%d, %d, %v) = %#x, want %#x", c.col, c.order, c.hist, got, c.want)
		}
	}

	// col 3's seed 0x9E3779B9 · 4 = 0x278DDE6E4 keeps its low 32 bits.
	if got := ngramSeed(3); got != 0x78DDE6E4 {
		t.Errorf("ngramSeed(3) = %#x, want 0x78DDE6E4", got)
	}
}

// tinyEngram is a random Engram source: orders (2, 3), 2 hashes, so 4
// tables of slots × 3, width 5 and 4 taps.
func tinyEngram(r *rand.Rand, prefix string, slots int) *mapSource {
	const tables, sub, dim = 4, 3, 5

	src := newMapSource()
	src.put(prefix+"tables", randVec(r, tables*slots*sub, 1), tables*slots, sub)
	src.put(prefix+"key_proj", randVec(r, dim*tables*sub, 0.5), dim, tables*sub)
	src.put(prefix+"value_proj", randVec(r, dim*tables*sub, 0.5), dim, tables*sub)
	src.put(prefix+"taps", randVec(r, 4*dim, 0.7), 4, dim)

	return src
}

// refEngram transcribes architecture.py engram_indices and Engram in
// float64 for one sequence: indices by the uint64 hash masked to 32 bits.
func refEngram(src *mapSource, prefix string, tokens, orders []int, hashes, slots int) (k, v []float64) {
	const sub, dim = 3, 5

	steps, tables := len(tokens), len(orders)*hashes
	w := func(name string) []float64 { return toFloat64(src.data[prefix+name]) }
	tab := w("tables")
	e := make([]float64, steps*tables*sub)

	for i := range steps {
		for oi, order := range orders {
			for h := range hashes {
				c := oi*hashes + h
				acc := (uint64(0x9E3779B9) * uint64(c+1)) & math.MaxUint32

				for j := range order {
					var tok uint64
					if i-j >= 0 {
						tok = uint64(tokens[i-j])
					}

					acc = ((acc ^ tok) * 0x01000193) & math.MaxUint32
				}

				acc ^= acc >> 15

				if i < order-1 {
					continue // ngram_ok = 0
				}

				row := c*slots + int(acc%uint64(slots))
				copy(e[(i*tables+c)*sub:], tab[row*sub:(row+1)*sub])
			}
		}
	}

	k = refLinear(e, steps, tables*sub, w("key_proj"), dim)
	raw := refLinear(e, steps, tables*sub, w("value_proj"), dim)
	taps := w("taps")
	dil := slices.Max(orders)
	v = make([]float64, len(raw))

	for j := range 4 {
		for i := j * dil; i < steps; i++ {
			for c := range dim {
				v[i*dim+c] += taps[j*dim+c] * raw[(i-j*dil)*dim+c]
			}
		}
	}

	return k, v
}

func TestEngramMatchesReference(t *testing.T) {
	const slots = 7

	r := rand.New(rand.NewPCG(31, 37))
	src := tinyEngram(r, "engram1.", slots)

	e, err := LoadEngram(src, "engram1.", []int{2, 3}, 4)
	if err != nil {
		t.Fatal(err)
	}

	if e.Dim() != 5 || e.Tables() != 4 || e.Slots() != slots || e.MaxOrder() != 3 {
		t.Fatalf("dim %d tables %d slots %d max order %d", e.Dim(), e.Tables(), e.Slots(), e.MaxOrder())
	}

	tokens := []int{0, 8198, 5, 17, 5, 17, 5, 3, 1 << 20, 2, 2, 2, 9}

	kv, err := e.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}

	wantK, wantV := refEngram(src, "engram1.", tokens, []int{2, 3}, 2, slots)
	if d := maxAbsDiff64(kv.K, wantK); d > 1e-5 {
		t.Errorf("k: max |Δ| %.3g", d)
	}

	if d := maxAbsDiff64(kv.V, wantV); d > 1e-5 {
		t.Errorf("v: max |Δ| %.3g", d)
	}
}

// TestEngramIndicesIncremental: the indices of every position from a
// rolling history of the last MaxOrder tokens equal the whole-sequence
// indices, as a token-by-token decoder computes them.
func TestEngramIndicesIncremental(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))

	e, err := LoadEngram(tinyEngram(r, "", 11), "", []int{2, 3}, 4)
	if err != nil {
		t.Fatal(err)
	}

	tokens := make([]int, 40)
	for i := range tokens {
		tokens[i] = r.IntN(8199)
	}

	all := e.Indices(tokens)
	hist := make([]int, 0, e.MaxOrder())
	got := make([]int, e.Tables())

	for i, tok := range tokens {
		if len(hist) == e.MaxOrder() {
			hist = append(hist[:0], hist[1:]...)
		}

		hist = append(hist, tok)
		e.IndicesAt(got, hist)

		if !slices.Equal(got, all[i*e.Tables():(i+1)*e.Tables()]) {
			t.Fatalf("position %d: rolling %v, sequence %v", i, got, all[i*e.Tables():(i+1)*e.Tables()])
		}
	}
}

func TestEngramErrors(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	src := tinyEngram(r, "", 7)
	call := func(_ any, err error) error { return err }

	e, err := LoadEngram(src, "", []int{2, 3}, 4)
	if err != nil {
		t.Fatal(err)
	}

	bad := tinyEngram(r, "", 7)
	bad.put("key_proj", randVec(r, 5*10, 1), 5, 10)

	for name, err := range map[string]error{
		"tables not a multiple of the orders": call(LoadEngram(src, "", []int{2, 3}, 3)),
		"rows not a multiple of the tables":   call(LoadEngram(src, "", []int{2, 3, 4}, 6)),
		"no orders":                           call(LoadEngram(src, "", nil, 4)),
		"order 0":                             call(LoadEngram(src, "", []int{0, 3}, 4)),
		"key_proj of the wrong width":         call(LoadEngram(bad, "", []int{2, 3}, 4)),
		"missing prefix":                      call(LoadEngram(src, "engram0.", []int{2, 3}, 4)),
		"negative token":                      call(e.Forward([]int{1, -2})),
	} {
		if err == nil {
			t.Errorf("accepted %s", name)
		}
	}
}

// refGate is Block's engram gate in float64 for one row:
// x + σ(⟨rms_unit(x), rms_unit(k)⟩ / √d) · v.
func refGate(x, k, v []float64) (out []float64, alpha float64) {
	unit := func(z []float64) []float64 {
		var ms float64
		for _, a := range z {
			ms += a * a
		}

		inv := 1 / math.Sqrt(ms/float64(len(z))+1e-6)
		out := make([]float64, len(z))

		for i, a := range z {
			out[i] = a * inv
		}

		return out
	}

	ux, uk := unit(x), unit(k)

	var dot float64
	for i := range ux {
		dot += ux[i] * uk[i]
	}

	alpha = 1 / (1 + math.Exp(-dot/math.Sqrt(float64(len(x)))))
	out = make([]float64, len(x))

	for i := range x {
		out[i] = x[i] + alpha*v[i]
	}

	return out, alpha
}

func TestEngramGateMatchesReference(t *testing.T) {
	const dim, steps = 16, 4

	r := rand.New(rand.NewPCG(3, 4))
	x, k, v := randVec(r, steps*dim, 2), randVec(r, steps*dim, 1), randVec(r, steps*dim, 3)

	// One row's k equals x, so its gate is σ(√d) up to the epsilon.
	copy(k[dim:2*dim], x[dim:2*dim])

	alpha := make([]float32, steps)
	got := slices.Clone(x)
	applyEngram(got, &EngramKV{K: k, V: v}, steps, dim, alpha)

	for i := range steps {
		want, wantAlpha := refGate(toFloat64(x[i*dim:(i+1)*dim]), toFloat64(k[i*dim:(i+1)*dim]),
			toFloat64(v[i*dim:(i+1)*dim]))
		if d := maxAbsDiff64(got[i*dim:(i+1)*dim], want); d > 1e-5 || math.Abs(float64(alpha[i])-wantAlpha) > 1e-6 {
			t.Errorf("row %d: max |Δ| %.3g, alpha %v vs %v", i, d, alpha[i], wantAlpha)
		}
	}
}

// TestEngramFixture is testdata/needle/engram.safetensors: site 0 (decoder
// layer 3) from the token ids through the hash indices, the table rows,
// both projections and the dilated taps, and the gate Block applies at
// layer 3 (docs/FINDINGS.md F-013).
func TestEngramFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "engram")
	cfg := src.Config()

	var orders []int

	err := json.Unmarshal([]byte(fx.metadata("orders")), &orders)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(orders, cfg.Orders) || fx.metadata("slots") != strconv.Itoa(cfg.Slots) ||
		fx.metadata("hashes") != strconv.Itoa(cfg.Tables/len(cfg.Orders)) || fx.metadata("site") != "0" ||
		len(cfg.Sites) == 0 || fx.metadata("layer") != strconv.Itoa(cfg.Sites[0]) {
		t.Fatalf("fixture orders %v slots %s hashes %s layer %s, config %+v", orders, fx.metadata("slots"),
			fx.metadata("hashes"), fx.metadata("layer"), cfg)
	}

	e, err := LoadEngram(src, "engram0.", cfg.Orders, cfg.Tables)
	if err != nil {
		t.Fatal(err)
	}

	if fx.metadata("dilation") != strconv.Itoa(e.dilation) {
		t.Fatalf("fixture dilation %s, block %d", fx.metadata("dilation"), e.dilation)
	}

	tokF, shape := fx.get("tokens")
	batch, steps := shape[0], shape[1]
	all := map[string][]float32{}
	indices := make([]float32, 0, batch*steps*e.Tables())

	for b := range batch {
		tokens := make([]int, steps)
		for i, v := range tokF[b*steps : (b+1)*steps] {
			tokens[i] = int(v)
		}

		tr := &engramTrace{}

		kv, err := e.forward(tokens, tr)
		if err != nil {
			t.Fatal(err)
		}

		for _, idx := range tr.indices {
			indices = append(indices, float32(idx))
		}

		if b == 0 {
			all["ngram_ok"] = tr.ngramOK
		}

		all["e"] = append(all["e"], tr.e...)
		all["k"] = append(all["k"], kv.K...)
		all["v_raw"] = append(all["v_raw"], tr.vRaw...)
		all["v"] = append(all["v"], kv.V...)
	}

	wantIdx, _ := fx.get("indices")
	if !slices.Equal(indices, wantIdx) {
		t.Error("indices differ from the fixture")
	} else {
		t.Logf("indices      all %d equal", len(wantIdx))
	}

	// tap_ok[j, t] = 1 where t ≥ dilation·j: where the shift is defined.
	tapOK, _ := fx.get("tap_ok")

	for j := range e.taps {
		for i := range steps {
			want := float32(0)
			if i >= j*e.dilation {
				want = 1
			}

			if tapOK[j*steps+i] != want {
				t.Errorf("tap_ok[%d, %d] = %v, want %v", j, i, tapOK[j*steps+i], want)
			}
		}
	}

	for _, name := range []string{"ngram_ok", "e", "k", "v_raw", "v"} {
		want, _ := fx.get(name)
		checkContract(t, name, all[name], want)
	}

	// The gate at layer 3 on the fixture's k and v, so it is checked on its
	// own; the Go k and v are checked above.
	x, _ := fx.get("x")
	k, _ := fx.get("k")
	v, _ := fx.get("v")
	n := steps * e.Dim()
	alpha := make([]float32, batch*steps)
	xOut := slices.Clone(x)

	for b := range batch {
		applyEngram(xOut[b*n:(b+1)*n], &EngramKV{K: k[b*n : (b+1)*n], V: v[b*n : (b+1)*n]}, steps, e.Dim(),
			alpha[b*steps:(b+1)*steps])
	}

	wantAlpha, _ := fx.get("alpha")
	checkContract(t, "alpha", alpha, wantAlpha)

	wantOut, _ := fx.get("x_out")
	checkContract(t, "x_out", xOut, wantOut)
}
