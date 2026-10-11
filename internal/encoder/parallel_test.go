package encoder

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/algo-tensor/tensor"
	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// The tests of docs/FINDINGS.md F-038: the stem in time chunks equals the
// whole stem bit for bit, and the encoder gives the same bits on one
// worker as on several.

// testSource is a weights.Source over in-memory tensors that serves the
// names in packed as CQ matrices (weights.Packer).
type testSource struct {
	data   map[string][]float32
	shape  map[string][]int
	packed map[string]*cq.Matrix
}

func (s *testSource) Names() []string {
	names := make([]string, 0, len(s.shape))
	for n := range s.shape {
		names = append(names, n)
	}

	slices.Sort(names)

	return names
}

func (s *testSource) Has(name string) bool { _, ok := s.shape[name]; return ok }

func (s *testSource) Shape(name string) ([]int, error) {
	if sh, ok := s.shape[name]; ok {
		return sh, nil
	}

	return nil, weights.ErrNotFound
}

func (s *testSource) Tensor(name string) ([]float32, []int, error) {
	d, ok := s.data[name]
	if !ok {
		return nil, nil, weights.ErrNotFound
	}

	return d, s.shape[name], nil
}

func (s *testSource) Matrix(name string) (*cq.Matrix, bool, error) {
	m, ok := s.packed[name]

	return m, ok, nil
}

func (s *testSource) Raw(string) ([]byte, error) { return nil, weights.ErrUnavailable }
func (s *testSource) Config() weights.Config     { return weights.Config{} }
func (s *testSource) Release(string)             {}
func (s *testSource) Close() error               { return nil }

// randomStem is a source with random stem tensors of c channels and dim
// outputs, whose stem.out takes in values; the pointwise and output
// matrices are packed CQ2 if packed is set.
func randomStem(t *testing.T, r *rand.Rand, c, dim, in int, packed bool) *testSource {
	t.Helper()

	s := &testSource{data: map[string][]float32{}, shape: map[string][]int{}, packed: map[string]*cq.Matrix{}}
	vec := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(r.NormFloat64() * scale)
		}

		return v
	}

	for _, name := range []string{"stem.w", "stem.dw_1", "stem.dw_2"} {
		s.data[name], s.shape[name] = vec(9*c, 0.4), []int{9, c}
	}

	cb := vec(4, 0.1)
	slices.Sort(cb)

	for _, m := range []struct {
		name    string
		out, in int
	}{{"stem.pw_1", c, c}, {"stem.pw_2", c, c}, {"stem.out", dim, in}} {
		s.shape[m.name] = []int{m.out, m.in}
		if !packed {
			s.data[m.name] = vec(m.out*m.in, 1/math.Sqrt(float64(m.in)))

			continue
		}

		p := cact.Packed{Out: m.out, In: m.in, Group: 16, Bits: 2, Width: 2, Codebook: cb}
		p.Index = make([]byte, m.out*p.RowBytes())

		for i := range p.Index {
			p.Index[i] = byte(r.Uint32())
		}

		for range m.out * p.Groups() {
			p.Norms = binary.LittleEndian.AppendUint16(p.Norms, uint16(13+r.IntN(3))<<10|uint16(r.IntN(1024)))
		}

		q, err := cq.New(p)
		if err != nil {
			t.Fatal(err)
		}

		s.packed[m.name] = q
	}

	return s
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// withWorkers runs fn with tensor.SetWorkers(n) and restores the count.
func withWorkers(n int, fn func()) {
	prev := tensor.Workers()
	defer tensor.SetWorkers(prev)

	tensor.SetWorkers(n)
	fn()
}

// TestStemChunks compares the chunked stem with whole on random weights
// for clip lengths around the chunk size and the 30 s limit, over the stem
// geometries the hypothesis search tried: bit for bit where the mel widths
// after the first conv are multiples of 4, as Best's are (stem.forward),
// and to float32 rounding elsewhere.
func TestStemChunks(t *testing.T) {
	const c, dim = 16, 24

	geoms := []struct {
		name  string
		edit  func(*Hypothesis)
		exact bool // mel widths that keep the 4-row blocks
	}{
		{"best", func(*Hypothesis) {}, true},
		{"same", func(h *Hypothesis) { h.StemPad = PadSame }, true},
		{"dw stride", func(h *Hypothesis) { h.StemStride = StrideDepthwise }, true},
		{"flat PE", func(h *Hypothesis) { h.PE = PEFlatConcat }, true},
		{"mel-channel", func(h *Hypothesis) { h.StemFlatten = FlattenMelChannel; h.StemDWAct = ActGELUTanh }, true},
		{"valid", func(h *Hypothesis) { h.StemPad = PadValid }, false}, // mel widths 39, 19, 9
	}

	for _, g := range geoms {
		h := Best()
		g.edit(&h)

		in := (&stem{h: h}).length(frontend.Mels) * c

		for _, packed := range []bool{true, false} {
			r := rand.New(rand.NewPCG(38, 20))

			s, err := loadStem(randomStem(t, r, c, dim, in, packed), h, frontend.Mels)
			if err != nil {
				t.Fatal(err)
			}

			s.peGate = 0.7

			for _, frames := range []int{3, 8, 9, 63, 64, 65, 129, 1001, 3000} {
				name := fmt.Sprintf("%s packed=%v frames=%d", g.name, packed, frames)
				mel := make([]float32, frames*frontend.Mels)

				for i := range mel {
					mel[i] = float32(r.NormFloat64())
				}

				want, wt, err := s.whole(mel, frames)
				if err != nil { // too short for the stem: the chunks must fail too
					_, _, err = s.forward(mel, frames)
					if err == nil {
						t.Errorf("%s: no error where the whole stem fails", name)
					}

					continue
				}

				for _, w := range []int{1, 16} {
					var (
						got []float32
						gt  int
					)

					withWorkers(w, func() { got, gt, err = s.forward(mel, frames) })

					if err != nil || gt != wt {
						t.Fatalf("%s, %d workers: %d frames, want %d, err %v", name, w, gt, wt, err)
					}

					if g.exact && !sameBits(got, want) || !g.exact && maxRel(got, want) > 1e-5 {
						t.Errorf("%s, %d workers: chunks differ from the whole stem (max relative %.2e)", name, w,
							maxRel(got, want))
					}
				}
			}
		}
	}
}

// TestStemTooShort checks that a clip too short for the stem fails as the
// whole stem does.
func TestStemTooShort(t *testing.T) {
	h := Best()
	h.StemPad = PadValid

	s, err := loadStem(randomStem(t, rand.New(rand.NewPCG(38, 21)), 16, 24, 9*16, false), h, frontend.Mels)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = s.forward(make([]float32, 2*frontend.Mels), 2)
	if err == nil {
		t.Error("2 frames with VALID padding: no error")
	}
}

// TestWhistleStemChunks runs Best's stem on whistle.cact, packed and
// float, on a short clip and on 30 s: the chunks equal the whole stem bit
// for bit.
func TestWhistleStemChunks(t *testing.T) {
	float := openModel(t, whistleCactPath)

	packed, err := weights.Open(whistleCactPath, weights.Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = packed.Close() })

	clip := loadClips(t)[0]

	for name, src := range map[string]weights.Source{"packed": packed, "float": float} {
		e, err := Load(src, Best())
		if err != nil {
			t.Fatal(err)
		}

		for _, samples := range [][]float32{clip.Samples, tiled(clip.Samples, 30*frontend.SampleRate)} {
			mel, err := e.fe.LogMel(samples)
			if err != nil {
				t.Fatal(err)
			}

			want, _, err := e.stem.whole(mel.Data, mel.Frames)
			if err != nil {
				t.Fatal(err)
			}

			got, _, err := e.stem.forward(mel.Data, mel.Frames)
			if err != nil || !sameBits(got, want) {
				t.Errorf("%s, %d frames: chunks differ from the whole stem (err %v)", name, mel.Frames, err)
			}
		}
	}
}

// TestWhistleEmbedWorkers runs Embed on whistle.cact, packed and float, on
// one worker and on several: the same bits, on a short clip and on 30 s.
func TestWhistleEmbedWorkers(t *testing.T) {
	float := openModel(t, whistleCactPath)

	packed, err := weights.Open(whistleCactPath, weights.Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = packed.Close() })

	clip := loadClips(t)[0]

	for name, src := range map[string]weights.Source{"packed": packed, "float": float} {
		e, err := Load(src, Best())
		if err != nil {
			t.Fatal(err)
		}

		for _, samples := range [][]float32{clip.Samples, tiled(clip.Samples, 30*frontend.SampleRate)} {
			var want []float32

			for _, w := range []int{1, 3, 16} {
				var got []float32

				withWorkers(w, func() { got, _, err = e.Embed(samples) })

				if err != nil {
					t.Fatal(err)
				}

				if want == nil {
					want = got

					continue
				}

				if !sameBits(got, want) {
					t.Errorf("%s, %d samples: %d workers differ from 1", name, len(samples), w)
				}
			}
		}
	}
}

// maxRel is max |a − b| over max |b|.
func maxRel(a, b []float32) float64 {
	var d, m float64
	for i := range b {
		d = max(d, math.Abs(float64(a[i])-float64(b[i])))
		m = max(m, math.Abs(float64(b[i])))
	}

	return d / m
}

// tiled repeats x to n samples.
func tiled(x []float32, n int) []float32 {
	out := make([]float32, n)
	for i := 0; i < n; i += len(x) {
		copy(out[i:], x)
	}

	return out
}
