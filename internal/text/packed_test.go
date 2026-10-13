package text

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// The packed model against the dequantised float model on the same
// needle3.cact (docs/FINDINGS.md F-059): both run the same CQ weights, so
// only the lut kernel's table rounding and the summation order differ.
const (
	minCosPacked  = 0.99999
	confPackedTol = 1e-4
	// packedLimit is PLAN.md 11.5's bound on the packed weights, 40 MB,
	// read as MiB as F-037 reads 9.2's 64 MB.
	packedLimit = 40 << 20
)

// needle3CQ is the number of CQ records in needle3.cact (115 CQ2, 7 CQ4;
// docs/FINDINGS.md F-045).
const needle3CQ = 122

// cactModel is a model on needle3.cact with its source, loaded once per
// test binary.
type cactModel struct {
	once sync.Once
	src  *Source
	m    *Model
	err  error
}

var dequantised, packedCact, packedLUT cactModel

func (c *cactModel) load(t testing.TB, opts OpenOptions) (*Model, *Source) {
	t.Helper()
	requireFile(t, needle3Cact)

	c.once.Do(func() {
		c.src, c.err = OpenWith(needle3Cact, opts)
		if c.err == nil {
			c.m, c.err = Load(c.src, c.src.TextConfig())
		}
	})

	if c.err != nil {
		t.Fatal(c.err)
	}

	return c.m, c.src
}

// loadDequantised is the float model on the dequantised needle3.cact.
func loadDequantised(t testing.TB) *Model {
	t.Helper()

	m, _ := dequantised.load(t, OpenOptions{})

	return m
}

// loadPacked is the model on needle3.cact's CQ records packed, and its
// source.
func loadPacked(t testing.TB) (*Model, *Source) {
	t.Helper()

	return packedCact.load(t, OpenOptions{Packed: true})
}

// cqNames lists needle3.cact's CQ records by name, counted from the file.
func cqNames(t *testing.T) map[string]bool {
	t.Helper()

	f, err := cact.Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	names, err := cact.TextNames(f)
	if err != nil {
		t.Fatal(err)
	}

	cq := map[string]bool{}

	for i, e := range names {
		if rec, _ := f.Record(i); rec.DType == cact.CQ {
			cq[e.Name] = true
		}
	}

	return cq
}

// TestSourcePacked opens needle3.cact packed: Matrix serves exactly its
// CQ records, at their shapes, once per name, without a float32 decode,
// and each matrix's rows equal the float source's values bit for bit. In
// float mode and from safetensors no tensor is packed.
func TestSourcePacked(t *testing.T) {
	requireFile(t, needle3Cact)
	requireFile(t, needle3Safetensors)

	cq := cqNames(t)
	if len(cq) != needle3CQ {
		t.Errorf("needle3.cact has %d CQ records, want %d", len(cq), needle3CQ)
	}

	packed, err := OpenWith(needle3Cact, OpenOptions{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	defer packed.Close()

	float, err := Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	defer float.Close()

	st, err := OpenWith(needle3Safetensors, OpenOptions{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	defer st.Close()

	n := 0

	for _, name := range packed.Names() {
		m, ok, err := packed.Matrix(name)
		if err != nil || ok != cq[name] {
			t.Fatalf("%s: Matrix ok %v, %v; CQ %v", name, ok, err, cq[name])
		}

		for what, src := range map[string]*Source{"float mode": float, "safetensors": st} {
			_, ok, err := src.Matrix(name)
			if ok || err != nil {
				t.Fatalf("%s: %s serves it packed (%v)", name, what, err)
			}
		}

		if !ok {
			continue
		}

		n++

		shape, _ := packed.Shape(name)
		if m.Out() != shape[0] || m.In() != shape[1] {
			t.Errorf("%s: matrix [%d, %d], shape %v", name, m.Out(), m.In(), shape)
		}

		if again, _, _ := packed.Matrix(name); again != m {
			t.Errorf("%s: a second Matrix call built a new matrix", name)
		}

		want, _, err := float.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}

		row := make([]float32, m.In())
		for _, r := range []int{0, m.Out() / 2, m.Out() - 1} {
			m.Row(row, r)

			if !slices.Equal(row, want[r*m.In():(r+1)*m.In()]) {
				t.Fatalf("%s: row %d differs from the dequantised values", name, r)
			}
		}

		float.Release(name)
	}

	t.Logf("%d of %d records served packed", n, len(packed.Names()))

	if n != needle3CQ {
		t.Errorf("%d packed records, want %d", n, needle3CQ)
	}

	if c := packed.cached(); len(c) != 0 {
		t.Errorf("Matrix decoded float32 values: %v", c)
	}
}

// TestSourcePackedLifecycle: Release drops a packed matrix, Close fails
// Matrix with weights.ErrClosed, and an unknown name is not found.
func TestSourcePackedLifecycle(t *testing.T) {
	requireFile(t, needle3Cact)

	src, err := OpenWith(needle3Cact, OpenOptions{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	const name = "layer00.q_proj"

	m, ok, err := src.Matrix(name)
	if !ok || err != nil {
		t.Fatalf("Matrix: %v, %v", ok, err)
	}

	src.Release(name)

	if again, _, _ := src.Matrix(name); again == m {
		t.Error("Release kept the packed matrix")
	}

	_, _, err = src.Matrix("no.such")
	if !errors.Is(err, weights.ErrNotFound) {
		t.Errorf("Matrix of an unknown name: %v, want ErrNotFound", err)
	}

	err = src.Close()
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = src.Matrix(name)
	if !errors.Is(err, weights.ErrClosed) {
		t.Errorf("Matrix after Close: %v, want ErrClosed", err)
	}
}

// TestLoadPackedNoFloatCopy loads the model on the packed source: no CQ
// record sits in the source's float cache, every one of them was built as
// a packed matrix, and the CQ consumers the model holds directly (the
// embedding, the tied head and the confidence head) are packed.
func TestLoadPackedNoFloatCopy(t *testing.T) {
	m, src := loadPacked(t)
	cq := cqNames(t)

	for _, name := range src.cached() {
		if cq[name] {
			t.Errorf("%s: a CQ record in the float cache", name)
		}
	}

	built := src.packedNames()
	for _, name := range built {
		if !cq[name] {
			t.Errorf("%s: packed but not CQ", name)
		}
	}

	if len(built) != len(cq) {
		t.Errorf("%d packed matrices built, want all %d CQ records", len(built), len(cq))
	}

	t.Logf("after Load: %d float32 tensors cached, %d packed matrices", len(src.cached()), len(built))

	h := m.confidence
	for what, l := range map[string]needle.Linear{
		"embedding": m.emb, "head": m.head, "probes": h.probes, "query": h.query, "proj": h.proj,
	} {
		if !l.Packed() {
			t.Errorf("%s is not packed", what)
		}
	}

	float := loadDequantised(t)
	if float.emb.Packed() || float.confidence.probes.Packed() {
		t.Error("the float model is packed")
	}
}

// parity is how far a packed model is from the dequantised float model
// over the forward cases and tiny: the worst logits row cosine, the
// positions whose top-1 differs with the dequantised model's margin
// there, and the largest difference of the confidence logit over both
// spans of every case.
type parity struct {
	worst      float64
	worstCase  string
	positions  int
	mismatches []mismatch
	confDiff   float64
}

type mismatch struct {
	what   string
	margin float64 // the dequantised model's top-1 less its top-2 logit
}

func compareDequantised(t *testing.T, packed *Model) parity {
	t.Helper()
	requireFile(t, forwardFixture)

	float := loadDequantised(t)
	fx := openFixture(t)
	v := packed.cfg.OutVocab
	pr := parity{worst: 2}

	for _, name := range confidenceCases {
		tokens, spans := confidenceSpans(t, fx, name)

		got, err := packed.Forward(tokens, true)
		if err != nil {
			t.Fatal(err)
		}

		want, err := float.Forward(tokens, true)
		if err != nil {
			t.Fatal(err)
		}

		c, row := rowCos(got.Logits, want.Logits, v)
		if c < pr.worst {
			pr.worst, pr.worstCase = c, fmt.Sprintf("%s at %d", name, row)
		}

		pr.positions += len(tokens)

		for p := range tokens {
			wantRow := want.Logits[p*v : (p+1)*v]
			if g, w := argmax(got.Logits[p*v:(p+1)*v]), argmax(wantRow); g != w {
				pr.mismatches = append(pr.mismatches, mismatch{
					what:   fmt.Sprintf("%s position %d: top-1 %d, dequantised %d", name, p, g, w),
					margin: margin(wantRow),
				})
			}
		}

		for _, keep := range spans {
			gc, err := packed.Confidence(got, keep)
			if err != nil {
				t.Fatal(err)
			}

			wc, err := float.Confidence(want, keep)
			if err != nil {
				t.Fatal(err)
			}

			pr.confDiff = max(pr.confDiff, math.Abs(float64(gc)-float64(wc)))
		}
	}

	t.Logf("worst logits row cosine %.12f, 1 − cos %.2g (%s); top-1 equal at %d/%d positions; "+
		"largest confidence |diff| %.3g", pr.worst, 1-pr.worst, pr.worstCase, pr.positions-len(pr.mismatches),
		pr.positions, pr.confDiff)

	for _, m := range pr.mismatches {
		t.Logf("%s, dequantised margin %.3g", m.what, m.margin)
	}

	return pr
}

// margin is a row's largest value less its second largest.
func margin(row []float32) float64 {
	top, second := math.Inf(-1), math.Inf(-1)

	for _, x := range row {
		switch v := float64(x); {
		case v > top:
			top, second = v, top
		case v > second:
			second = v
		}
	}

	return top - second
}

// TestForwardCactPacked runs the packed model and the dequantised float
// model on the inputs of TestForwardCactDequantised: the logits at every
// position within cosine minCosPacked and the same top-1, and the
// confidence head's raw logit over both spans of every case within
// confPackedTol (docs/FINDINGS.md F-059).
func TestForwardCactPacked(t *testing.T) {
	packed, _ := loadPacked(t)
	pr := compareDequantised(t, packed)

	if pr.worst < minCosPacked {
		t.Errorf("logits cosine %.9f (%s) < %v", pr.worst, pr.worstCase, minCosPacked)
	}

	for _, m := range pr.mismatches {
		t.Error(m.what)
	}

	if !(pr.confDiff <= confPackedTol) {
		t.Errorf("confidence moved by %.3g, want at most %v", pr.confDiff, confPackedTol)
	}
}

// TestForwardCactPackedLUT is TestForwardCactPacked on the lut kernel
// (OpenOptions.LUT), whose 14-bit tables move the model further: bounds
// just outside the worst measured (docs/FINDINGS.md F-059), and a top-1
// may change only where the dequantised model's margin is below
// lutMaxMargin.
func TestForwardCactPackedLUT(t *testing.T) {
	const (
		lutMinCos    = 0.9999999
		lutConfTol   = 1e-3
		lutMaxMargin = 1e-2
	)

	packed, _ := packedLUT.load(t, OpenOptions{Packed: true, LUT: true})
	pr := compareDequantised(t, packed)

	if pr.worst < lutMinCos {
		t.Errorf("logits cosine %.9f (%s) < %v", pr.worst, pr.worstCase, lutMinCos)
	}

	for _, m := range pr.mismatches {
		if m.margin >= lutMaxMargin {
			t.Errorf("%s at a margin of %.3g", m.what, m.margin)
		}
	}

	if !(pr.confDiff <= lutConfTol) {
		t.Errorf("confidence moved by %.3g, want at most %v", pr.confDiff, lutConfTol)
	}
}

// TestPackedMemory measures the packed model's resident weight bytes
// (docs/FINDINGS.md F-059): the packed matrices' indices, float32 norms,
// codebooks and lut copies (cq.Matrix.Bytes) and every float32 tensor the
// source's cache holds, by group (weightBytes), and what the loaders
// derive from those (the live heap after Load less the counted bytes on
// the heap). Their sum must stay under packedLimit (PLAN.md 11.5). The
// bytes of the file buffer that no index aliases (the FP16 records and
// norms as stored, the tokenizer) are logged on top. The lut kernel's
// model is measured too, without a bound.
func TestPackedMemory(t *testing.T) {
	requireFile(t, needle3Cact)

	data, err := os.ReadFile(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	for _, mode := range []struct {
		name string
		opts OpenOptions
	}{{"exact", OpenOptions{Packed: true}}, {"lut", OpenOptions{Packed: true, LUT: true}}} {
		t.Run(mode.name, func(t *testing.T) {
			weightsBytes, unaliased := packedMemory(t, data, mode.opts)
			total := weightsBytes + unaliased

			t.Logf("weights %.3f MiB (%d bytes); with the file buffer %.3f MiB (%d bytes); limit %.0f MiB",
				mib(weightsBytes), weightsBytes, mib(total), total, mib(packedLimit))

			if !mode.opts.LUT && weightsBytes >= packedLimit {
				t.Errorf("packed weights take %.3f MiB, want under %.0f", mib(weightsBytes), mib(packedLimit))
			}
		})
	}
}

// packedMemory loads the model packed from data and returns its weight
// bytes, as TestPackedMemory counts them, and the file bytes no packed
// index aliases.
func packedMemory(t *testing.T, data []byte, opts OpenOptions) (weightsBytes, unaliased int) {
	t.Helper()

	var before, after runtime.MemStats

	runtime.GC()
	runtime.ReadMemStats(&before)

	src, err := OpenCactWith(data, opts)
	if err != nil {
		t.Fatal(err)
	}

	m, err := Load(src, src.TextConfig())
	if err != nil {
		t.Fatal(err)
	}

	runtime.GC()
	runtime.ReadMemStats(&after)

	groups := src.weightBytes()
	heap := int(after.HeapAlloc) - int(before.HeapAlloc)

	var counted, index int

	for _, g := range groups {
		counted += g.total()
		index += g.index

		t.Logf("%-13s %7.3f MiB: index %.3f, norms %.3f, lut copy %.3f, float32 %.3f", g.group, mib(g.total()),
			mib(g.index), mib(g.norms), mib(g.lut), mib(g.float))
	}

	derived := max(heap-(counted-index), 0)
	t.Logf("%-13s %7.3f MiB (live heap after Load %.3f MiB)", "loaders", mib(derived), mib(heap))
	t.Logf("%-13s %7.3f MiB of %.3f (FP16 records and norms as stored, tokenizer, directory)", "file, unaliased",
		mib(len(data)-index), mib(len(data)))

	runtime.KeepAlive(m)

	return counted + derived, len(data) - index
}

func mib(n int) float64 { return float64(n) / (1 << 20) }
