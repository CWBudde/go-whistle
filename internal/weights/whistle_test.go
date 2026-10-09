package weights

import (
	"errors"
	"math"
	"os"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"testing"
	"unsafe"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// Pinned model files as laid out by `whistle download`.
const (
	whistleCactPath        = "../../models/whistle/whistle.cact"
	whistleSafetensorsPath = "../../models/whistle/whistle.safetensors"
)

// F-006's measured minimum cosine per record type against the source,
// pinned there to 1e-9; every record of the type meets it.
var pinnedMinCos = map[string]float64{
	"FP16": 0.999999963923, // enc00.q_norm
	"CQ2":  0.931853685259, // cross07.k_proj
	"CQ4":  0.994312193718, // enc.mhc_phi_post
}

func wantWhistleConfig() Config {
	return Config{
		Vocab: 8199, Text: 8192, Dim: 512, HadaN: 512, Heads: 8, KVHeads: 2, QKDim: 48, VDim: 64,
		DecLayers: 8, EncLayers: 8, Lanes: 4, EncLanes: 4, Taps: 3, ConvKernel: 9,
		Sites: []int{3, 7}, Orders: []int{2, 3}, Slots: 18432, Tables: 4,
		Stem: 128, StemStages: 3, Rope: 100000,
		Mels: 80, Rate: 16000, FFT: 512, Win: 400, Hop: 160, MaxTokens: 320,
	}
}

func requireWhistle(t *testing.T) {
	t.Helper()

	for _, path := range []string{whistleCactPath, whistleSafetensorsPath} {
		_, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", path)
		}
	}
}

// TestWhistleCrossFormat opens whistle.cact and whistle.safetensors @
// b358ddad and compares every one of the 681 names across the backends
// (docs/FINDINGS.md F-007), by F-006's rules:
//
//   - shapes are equal for every name;
//   - derived records are equal (the safetensors side recomputes them);
//   - FP16 records are within half an FP16 ulp of the source, value by
//     value, and meet F-006's rounding bound on the cosine;
//   - CQ records meet F-006's pinned minimum cosine of their width, and the
//     minimum per type equals F-006's to 1e-9.
//
// Each backend must also equal its format's plain reader bit for bit
// (cact.File, safetensors.Store + Entry.Apply), so TestWhistleNameMap's
// per-record checks carry over unchanged.
func TestWhistleCrossFormat(t *testing.T) {
	requireWhistle(t)

	c, err := OpenCact(whistleCactPath, Options{})
	if err != nil {
		t.Fatalf("OpenCact: %v", err)
	}
	defer c.Close()

	if n := c.decodes.Load(); n != 0 {
		t.Errorf("OpenCact decoded %d tensors", n)
	}

	s, err := OpenSafetensors(whistleSafetensorsPath, Options{})
	if err != nil {
		t.Fatalf("OpenSafetensors: %v", err)
	}
	defer s.Close()

	if c.Mapped() != canMmap || s.Mapped() != canMmap {
		t.Errorf("Mapped() = %v (cact), %v (safetensors), want %v", c.Mapped(), s.Mapped(), canMmap)
	}

	for format, cfg := range map[string]Config{"cact": c.Config(), "safetensors": s.Config()} {
		if !reflect.DeepEqual(cfg, wantWhistleConfig()) {
			t.Errorf("%s Config() = %+v\nwant %+v", format, cfg, wantWhistleConfig())
		}
	}

	f, err := cact.Open(whistleCactPath)
	if err != nil {
		t.Fatalf("cact.Open: %v", err)
	}

	store, err := safetensors.OpenStore(whistleSafetensorsPath, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	names := cact.WhistleNames()
	if !slices.Equal(c.Names(), s.Names()) || len(c.Names()) != len(names) {
		t.Fatalf("Names: %d (cact) vs %d (safetensors), want %d equal", len(c.Names()), len(s.Names()), len(names))
	}

	minCos := map[string]float64{}
	count := map[string]int{}

	for i, e := range names {
		rec, _ := f.Record(i)

		kind := rec.DType.String()
		if rec.DType == cact.CQ {
			kind += strconv.Itoa(int(rec.Bits))
		}

		count[kind]++

		cs, errC := c.Shape(e.Name)
		ss, errS := s.Shape(e.Name)

		if errC != nil || errS != nil || !slices.Equal(cs, ss) || !slices.Equal(cs, e.Shape) {
			t.Errorf("%d %s: shapes %v (%v) and %v (%v), want %v", i, e.Name, cs, errC, ss, errS, e.Shape)
		}

		if e.Shape == nil {
			checkWhistleTokenizer(t, c, s, e.Name)

			continue
		}

		cv, sv := whistlePair(t, c, s, f, store, i, e)
		if cv == nil {
			continue
		}

		cos := checkWhistleRecord(t, i, e, rec, kind, cv, sv)
		if e.Transform != cact.Derived {
			if m, seen := minCos[kind]; !seen || cos < m {
				minCos[kind] = cos
			}
		}

		c.Release(e.Name)
		s.Release(e.Name)
	}

	wantCount := map[string]int{"FP16": 524, "FP32": 4, "CQ2": 145, "CQ4": 7, "RAW": 1}
	if !reflect.DeepEqual(count, wantCount) {
		t.Errorf("records by type %v, want %v", count, wantCount)
	}

	for kind, want := range pinnedMinCos {
		t.Logf("min cosine %-4s %.12f (F-006 %.12f)", kind, minCos[kind], want)

		if math.Abs(minCos[kind]-want) > 1e-9 {
			t.Errorf("min cosine %s = %.12f, F-006 pins %.12f", kind, minCos[kind], want)
		}
	}
}

// checkWhistleRecord compares one record across the backends by F-006's
// rule for its type and returns the cosine.
func checkWhistleRecord(t *testing.T, i int, e cact.Entry, rec cact.Record, kind string, cv, sv []float32) float64 {
	t.Helper()

	cos := cosine(cv, sv)

	switch {
	case e.Transform == cact.Derived:
		if !slices.Equal(cv, sv) {
			t.Errorf("%d %s: recomputed record differs from the .cact one", i, e.Name)
		}
	case rec.DType == cact.FP16:
		checkWhistleFP16(t, e.Name, cv, sv, cos)
	case rec.DType == cact.CQ:
		if cos < pinnedMinCos[kind] {
			t.Errorf("%d %s: %s cosine %.12f < F-006's %.12f", i, e.Name, kind, cos, pinnedMinCos[kind])
		}
	default:
		t.Errorf("%d %s: sourced %s record", i, e.Name, kind)
	}

	return cos
}

// whistlePair returns name's tensor from both backends after checking each
// against its format's plain reader; nil after an error.
func whistlePair(
	t *testing.T, c *Cact, s *Safetensors, f *cact.File, store *safetensors.Store, i int, e cact.Entry,
) (cv, sv []float32) {
	t.Helper()

	cv, _, err := c.Tensor(e.Name)
	if err != nil {
		t.Errorf("%d %s: cact: %v", i, e.Name, err)

		return nil, nil
	}

	sv, _, err = s.Tensor(e.Name)
	if err != nil {
		t.Errorf("%d %s: safetensors: %v", i, e.Name, err)

		return nil, nil
	}

	rec, _ := f.Record(i)

	var refC []float32
	if rec.DType == cact.CQ {
		refC, err = f.CQ(i)
	} else {
		refC, err = f.Float32(i)
	}

	if err != nil || !slices.Equal(cv, refC) {
		t.Errorf("%d %s: cact backend differs from cact.File (%v)", i, e.Name, err)
	}

	if e.Transform == cact.Derived {
		return cv, sv
	}

	src, err := store.Tensor(e.Source)
	if err != nil {
		t.Fatalf("%s: %v", e.Source, err)
	}

	shape := make([]int, len(src.Shape))
	for k, d := range src.Shape {
		shape[k] = int(d)
	}

	refS, err := e.Apply(src.Data, shape)
	if err != nil || !slices.Equal(sv, refS) {
		t.Errorf("%d %s: safetensors backend differs from Entry.Apply (%v)", i, e.Name, err)
	}

	return cv, sv
}

// checkWhistleFP16 is F-006's FP16 rule: each value within half an FP16
// ulp of its source, and the cosine at least √(1 − ‖h‖²/‖x‖²).
func checkWhistleFP16(t *testing.T, name string, got, want []float32, cos float64) {
	t.Helper()

	var hh, xx float64

	for k, x := range want {
		h := halfULP16(float64(x))
		if math.Abs(float64(got[k])-float64(x)) > h {
			t.Errorf("%s[%d] = %v, more than half an FP16 ulp from %v", name, k, got[k], x)

			return
		}

		hh += h * h
		xx += float64(x) * float64(x)
	}

	bound := 1.0
	if xx > 0 {
		bound = math.Sqrt(max(0, 1-hh/xx))
	}

	// Scalars (gates) have a cosine of ±1; the value check decides them.
	if len(want) > 1 && cos < bound {
		t.Errorf("%s: cosine %.12f < FP16 rounding bound %.12f", name, cos, bound)
	}
}

// checkWhistleTokenizer: the .cact blob parses to the config's vocabulary;
// the checkpoint has none (F-007).
func checkWhistleTokenizer(t *testing.T, c *Cact, s *Safetensors, name string) {
	t.Helper()

	blob, err := c.Raw(name)
	if err != nil {
		t.Fatalf("cact Raw(%s): %v", name, err)
	}

	tok, err := tokenizer.Parse(blob)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}

	if tok.Len() != c.Config().Vocab {
		t.Errorf("tokenizer: %d pieces, want %d", tok.Len(), c.Config().Vocab)
	}

	_, err = s.Raw(name)
	if !errors.Is(err, ErrUnavailable) {
		t.Errorf("safetensors Raw(%s): %v, want ErrUnavailable", name, err)
	}
}

// TestWhistleMmapFallback opens both Whistle files mapped and with the
// read fallback forced; every tensor must be identical.
func TestWhistleMmapFallback(t *testing.T) {
	requireWhistle(t)

	for _, path := range []string{whistleCactPath, whistleSafetensorsPath} {
		mapped, read := openBoth(t, path, nil)

		if got := coreOf(mapped).Mapped(); got != canMmap {
			t.Errorf("%s: Mapped() = %v, want %v", path, got, canMmap)
		}

		n := 0

		for _, name := range mapped.Names() {
			if shape, _ := mapped.Shape(name); shape == nil {
				continue
			}

			a, _, errA := mapped.Tensor(name)
			b, _, errB := read.Tensor(name)

			if errA != nil || errB != nil || !slices.Equal(a, b) {
				t.Errorf("%s %s: mapped and read differ (%v, %v)", path, name, errA, errB)
			}

			mapped.Release(name)
			read.Release(name)

			n++
		}

		t.Logf("%s: %d tensors identical mapped and read", path, n)
	}
}

// TestWhistleConcurrentCQ dequantises the CQ4 embedding from many
// goroutines at once: one decode, one shared slice.
func TestWhistleConcurrentCQ(t *testing.T) {
	requireWhistle(t)

	c, err := OpenCact(whistleCactPath, Options{})
	if err != nil {
		t.Fatalf("OpenCact: %v", err)
	}
	defer c.Close()

	const callers = 8

	ptrs := make([]*float32, callers)

	var wg sync.WaitGroup

	for g := range callers {
		wg.Go(func() {
			v, _, err := c.Tensor("embedding")
			if err != nil {
				t.Errorf("Tensor(embedding): %v", err)

				return
			}

			ptrs[g] = unsafe.SliceData(v)
		})
	}

	wg.Wait()

	if n := c.decodes.Load(); n != 1 {
		t.Errorf("%d decodes, want 1", n)
	}

	for _, p := range ptrs {
		if p != ptrs[0] {
			t.Fatal("callers got different slices")
		}
	}
}

// halfULP16 is half the FP16 spacing at x: 2^(e-11) for |x| in [2^e,
// 2^(e+1)) in the normal range, 2^-25 among the subnormals.
func halfULP16(x float64) float64 {
	a := math.Abs(x)
	if a < 0x1p-14 {
		return 0x1p-25
	}

	_, e := math.Frexp(a) // a = m * 2^e, m in [0.5, 1)

	return math.Ldexp(1, e-12)
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64

	for k := range a {
		ab += float64(a[k]) * float64(b[k])
		aa += float64(a[k]) * float64(a[k])
		bb += float64(b[k]) * float64(b[k])
	}

	if aa == 0 && bb == 0 {
		return 1
	}

	return ab / math.Sqrt(aa*bb)
}
