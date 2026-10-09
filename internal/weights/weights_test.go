package weights

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// backends opens the fixture in both formats from bytes.
func backends(t *testing.T, fx fixture) map[string]*core {
	t.Helper()

	c, err := ParseCact(fx.cact, Options{Names: fx.names})
	if err != nil {
		t.Fatalf("ParseCact: %v", err)
	}

	s, err := ParseSafetensors(fx.st, Options{Names: fx.names})
	if err != nil {
		t.Fatalf("ParseSafetensors: %v", err)
	}

	t.Cleanup(func() {
		_ = c.Close()
		_ = s.Close()
	})

	return map[string]*core{"cact": c.core, "safetensors": s.core}
}

// writeFiles stores the fixture as whistle.cact and whistle.safetensors.
func writeFiles(t *testing.T, fx fixture) (cactPath, stPath string) {
	t.Helper()

	dir := t.TempDir()
	cactPath, stPath = filepath.Join(dir, "whistle.cact"), filepath.Join(dir, "whistle.safetensors")

	for path, data := range map[string][]byte{cactPath: fx.cact, stPath: fx.st} {
		err := os.WriteFile(path, data, 0o600)
		if err != nil {
			t.Fatal(err)
		}
	}

	return cactPath, stPath
}

func TestSourceInterface(t *testing.T) {
	var _ Source = (*Cact)(nil)
	var _ Source = (*Safetensors)(nil)
}

// TestCrossFormat requires both backends to serve every name with the same
// shape and the same values, and the same config: the safetensors backend
// applies the layer slice, transpose and lane-major layout, and recomputes
// the derived records.
func TestCrossFormat(t *testing.T) {
	fx := newFixture(t)
	bs := backends(t, fx)
	c, s := bs["cact"], bs["safetensors"]

	want := make([]string, len(fx.names))
	for i, e := range fx.names {
		want[i] = e.Name
	}

	for format, b := range bs {
		if got := b.Names(); !slices.Equal(got, want) {
			t.Errorf("%s: Names() = %v, want %v", format, got, want)
		}

		if got := b.Config(); !reflect.DeepEqual(got, fx.cfg) {
			t.Errorf("%s: Config() = %+v\nwant %+v", format, got, fx.cfg)
		}
	}

	for _, e := range fx.names {
		if e.Shape == nil {
			continue
		}

		cv, cs, err := c.Tensor(e.Name)
		if err != nil {
			t.Fatalf("cact %s: %v", e.Name, err)
		}

		sv, ss, err := s.Tensor(e.Name)
		if err != nil {
			t.Fatalf("safetensors %s: %v", e.Name, err)
		}

		if !slices.Equal(cs, e.Shape) || !slices.Equal(ss, e.Shape) {
			t.Errorf("%s: shapes %v (cact) and %v (safetensors), want %v", e.Name, cs, ss, e.Shape)
		}

		if !slices.Equal(cv, fx.cactTensors[e.Name]) {
			t.Errorf("%s: cact backend differs from cact.File", e.Name)
		}

		if !slices.Equal(cv, sv) {
			t.Errorf("%s: backends differ:\n cact %v\n st   %v", e.Name, cv, sv)
		}
	}
}

func TestShapeAndHas(t *testing.T) {
	for format, b := range backends(t, newFixture(t)) {
		shape, err := b.Shape("layer01.q_proj")
		if err != nil || !slices.Equal(shape, []int{8, 8}) {
			t.Errorf("%s: Shape(layer01.q_proj) = %v, %v", format, shape, err)
		}

		shape[0] = 99

		again, _ := b.Shape("layer01.q_proj")
		if again[0] != 8 {
			t.Errorf("%s: Shape hands out the map's slice", format)
		}

		blobShape, err := b.Shape("tokenizer")
		if blobShape != nil || err != nil {
			t.Errorf("%s: Shape(tokenizer) = %v, %v, want nil, nil", format, blobShape, err)
		}

		if !b.Has("pe_gate") || b.Has("nope") {
			t.Errorf("%s: Has is wrong", format)
		}

		for _, call := range []func() error{
			func() error { _, err := b.Shape("nope"); return err },
			func() error { _, _, err := b.Tensor("nope"); return err },
			func() error { _, err := b.Raw("nope"); return err },
		} {
			err := call()
			if !errors.Is(err, ErrNotFound) {
				t.Errorf("%s: unknown name: %v, want ErrNotFound", format, err)
			}
		}
	}
}

// TestDerivedTensors pins the F-007 decision: the safetensors backend
// recomputes the permutations, the mel bank and the manifest, and reports
// the tokenizer as unavailable.
func TestDerivedTensors(t *testing.T) {
	fx := newFixture(t)
	bs := backends(t, fx)

	for _, name := range []string{"hada_p1", "hada_p2", "audio.manifest", "audio.mel_filters"} {
		got, _, err := bs["safetensors"].Tensor(name)
		if err != nil || !slices.Equal(got, fx.cactTensors[name]) {
			t.Errorf("safetensors %s = %v, %v, want %v", name, got, err, fx.cactTensors[name])
		}
	}

	blob, err := bs["cact"].Raw("tokenizer")
	if err != nil || string(blob) != string(fx.tokenizer) {
		t.Fatalf("cact Raw(tokenizer) = %q, %v", blob, err)
	}

	blob[0] = 'X'

	again, _ := bs["cact"].Raw("tokenizer")
	if again[0] != fx.tokenizer[0] {
		t.Error("Raw hands out shared bytes")
	}

	_, err = bs["safetensors"].Raw("tokenizer")
	if !errors.Is(err, ErrUnavailable) {
		t.Errorf("safetensors Raw(tokenizer): %v, want ErrUnavailable", err)
	}

	for format, b := range bs {
		_, _, err := b.Tensor("tokenizer")
		if !errors.Is(err, cact.ErrDType) {
			t.Errorf("%s Tensor(tokenizer): %v, want ErrDType", format, err)
		}

		_, err = b.Raw("pe_gate")
		if !errors.Is(err, cact.ErrDType) {
			t.Errorf("%s Raw(pe_gate): %v, want ErrDType", format, err)
		}
	}
}

// TestUnknownDerived: a derived name the safetensors backend cannot
// recompute is unavailable, not a zero tensor.
func TestUnknownDerived(t *testing.T) {
	fx := newFixture(t)
	names := slices.Clone(fx.names)
	names[5].Name = "hada_p3"

	s, err := ParseSafetensors(fx.st, Options{Names: names})
	if err != nil {
		t.Fatalf("ParseSafetensors: %v", err)
	}
	defer s.Close()

	_, _, err = s.Tensor("hada_p3")
	if !errors.Is(err, ErrUnavailable) {
		t.Errorf("Tensor(hada_p3): %v, want ErrUnavailable", err)
	}
}

// TestLazy: opening decodes nothing; the first Tensor call decodes once and
// later calls return the cached slice.
func TestLazy(t *testing.T) {
	for format, b := range backends(t, newFixture(t)) {
		if n := b.decodes.Load(); n != 0 {
			t.Errorf("%s: %d tensors decoded at open", format, n)
		}

		first, _, err := b.Tensor("embedding")
		if err != nil {
			t.Fatalf("%s: %v", format, err)
		}

		again, _, _ := b.Tensor("embedding")

		if n := b.decodes.Load(); n != 1 {
			t.Errorf("%s: %d decodes after two calls, want 1", format, n)
		}

		if unsafe.SliceData(first) != unsafe.SliceData(again) {
			t.Errorf("%s: second call did not return the cached slice", format)
		}
	}
}

// TestConcurrentTensor runs many goroutines on the same names; with -race
// it checks the cache, and every caller must get the one decoded slice.
func TestConcurrentTensor(t *testing.T) {
	const callers = 32

	for format, b := range backends(t, newFixture(t)) {
		names := []string{"embedding", "layer00.q_proj", "hada_p1", "audio.mel_filters"}
		got := make([][]*float32, len(names))

		var wg sync.WaitGroup

		for k, name := range names {
			got[k] = make([]*float32, callers)

			for g := range callers {
				wg.Go(func() {
					v, _, err := b.Tensor(name)
					if err != nil {
						t.Errorf("%s %s: %v", format, name, err)

						return
					}

					got[k][g] = unsafe.SliceData(v)
				})
			}
		}

		wg.Wait()

		if n := b.decodes.Load(); n != int64(len(names)) {
			t.Errorf("%s: %d decodes for %d names", format, n, len(names))
		}

		for k, ptrs := range got {
			for _, p := range ptrs {
				if p != ptrs[0] {
					t.Errorf("%s %s: callers got different slices", format, names[k])

					break
				}
			}
		}
	}
}

// TestRelease: a released name decodes again on the next call; the old
// slice stays valid.
func TestRelease(t *testing.T) {
	for format, b := range backends(t, newFixture(t)) {
		first, _, _ := b.Tensor("embedding")
		want := slices.Clone(first)

		b.Release("embedding")
		b.Release("embedding") // idempotent
		b.Release("nope")      // unknown names are ignored

		again, _, err := b.Tensor("embedding")
		if err != nil {
			t.Fatalf("%s: %v", format, err)
		}

		if n := b.decodes.Load(); n != 2 {
			t.Errorf("%s: %d decodes, want 2", format, n)
		}

		if unsafe.SliceData(first) == unsafe.SliceData(again) {
			t.Errorf("%s: Release kept the cached slice", format)
		}

		if !slices.Equal(first, want) || !slices.Equal(again, want) {
			t.Errorf("%s: values changed across Release", format)
		}
	}
}

// openBoth opens path with mmap allowed and with the read fallback forced.
func openBoth(t *testing.T, path string, names []cact.Entry) (mapped, read Source) {
	t.Helper()

	mapped, err := Open(path, Options{Names: names})
	if err != nil {
		t.Fatalf("Open %s: %v", path, err)
	}

	read, err = Open(path, Options{Names: names, NoMmap: true})
	if err != nil {
		t.Fatalf("Open %s (NoMmap): %v", path, err)
	}

	t.Cleanup(func() {
		_ = mapped.Close()
		_ = read.Close()
	})

	return mapped, read
}

func coreOf(s Source) *core {
	switch b := s.(type) {
	case *Cact:
		return b.core
	case *Safetensors:
		return b.core
	default:
		return nil
	}
}

// TestMmapAndFallback opens both formats from disk, mapped where the
// platform allows it and with the os.ReadFile fallback forced, and requires
// identical tensors.
func TestMmapAndFallback(t *testing.T) {
	fx := newFixture(t)
	cactPath, stPath := writeFiles(t, fx)

	for _, path := range []string{cactPath, stPath} {
		mapped, read := openBoth(t, path, fx.names)

		if got := coreOf(mapped).Mapped(); got != canMmap {
			t.Errorf("%s: Mapped() = %v, want %v on this platform", path, got, canMmap)
		}

		if coreOf(read).Mapped() {
			t.Errorf("%s: NoMmap still mapped the file", path)
		}

		wantType := reflect.TypeFor[*Cact]()
		if strings.HasSuffix(path, ".safetensors") {
			wantType = reflect.TypeFor[*Safetensors]()
		}

		if reflect.TypeOf(mapped) != wantType {
			t.Errorf("Open(%s) = %T, want %v", path, mapped, wantType)
		}

		for _, e := range fx.names {
			if e.Shape == nil {
				continue
			}

			a, _, errA := mapped.Tensor(e.Name)
			b, _, errB := read.Tensor(e.Name)

			if errA != nil || errB != nil || !slices.Equal(a, b) || !slices.Equal(a, fx.cactTensors[e.Name]) {
				t.Errorf("%s %s: mapped %v (%v), read %v (%v)", path, e.Name, a, errA, b, errB)
			}
		}
	}
}

// TestCloseNoAlias: Close unmaps; nothing handed out before aliases the
// mapping, so it stays readable; Tensor and Raw fail afterwards.
func TestCloseNoAlias(t *testing.T) {
	fx := newFixture(t)
	cactPath, stPath := writeFiles(t, fx)

	for _, path := range []string{cactPath, stPath} {
		src, err := Open(path, Options{Names: fx.names})
		if err != nil {
			t.Fatal(err)
		}

		c := coreOf(src)
		mapping := c.file.data
		lo := uintptr(unsafe.Pointer(unsafe.SliceData(mapping)))
		hi := lo + uintptr(len(mapping))

		inside := func(p unsafe.Pointer, n int) bool {
			a := uintptr(p)
			return n > 0 && a < hi && a+uintptr(n) > lo
		}

		got := map[string][]float32{}
		copies := map[string][]float32{}

		for _, e := range fx.names {
			if e.Shape == nil {
				continue
			}

			v, _, err := src.Tensor(e.Name)
			if err != nil {
				t.Fatal(err)
			}

			if inside(unsafe.Pointer(unsafe.SliceData(v)), 4*len(v)) {
				t.Errorf("%s %s: tensor aliases the file data", path, e.Name)
			}

			got[e.Name], copies[e.Name] = v, slices.Clone(v)
		}

		blob, rawErr := src.Raw("tokenizer")
		if rawErr == nil && inside(unsafe.Pointer(unsafe.SliceData(blob)), len(blob)) {
			t.Errorf("%s: Raw aliases the file data", path)
		}

		err = src.Close()
		if err != nil {
			t.Errorf("%s: Close: %v", path, err)
		}

		if c.file.data != nil {
			t.Errorf("%s: Close kept the file data", path)
		}

		err = src.Close()
		if err != nil {
			t.Errorf("%s: second Close: %v", path, err)
		}

		for name, v := range got {
			if !slices.Equal(v, copies[name]) {
				t.Errorf("%s %s: changed after Close", path, name)
			}
		}

		if rawErr == nil && string(blob) != string(fx.tokenizer) {
			t.Errorf("%s: Raw bytes changed after Close", path)
		}

		_, _, err = src.Tensor("embedding")
		if !errors.Is(err, ErrClosed) {
			t.Errorf("%s: Tensor after Close: %v, want ErrClosed", path, err)
		}

		_, err = src.Raw("tokenizer")
		if !errors.Is(err, ErrClosed) {
			t.Errorf("%s: Raw after Close: %v, want ErrClosed", path, err)
		}

		if !src.Has("embedding") || len(src.Names()) != len(fx.names) {
			t.Errorf("%s: the name map is gone after Close", path)
		}
	}
}

// TestCloseWhileReading closes a source while goroutines decode from it;
// each call either succeeds or reports ErrClosed (run with -race).
func TestCloseWhileReading(t *testing.T) {
	fx := newFixture(t)
	cactPath, _ := writeFiles(t, fx)

	src, err := OpenCact(cactPath, Options{Names: fx.names})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	for g := range 16 {
		wg.Go(func() {
			for k := range 50 {
				name := fx.names[(g+k)%4].Name
				src.Release(name)

				_, _, err := src.Tensor(name)
				if err != nil && !errors.Is(err, ErrClosed) {
					t.Errorf("Tensor(%s): %v", name, err)
				}
			}
		})
	}

	wg.Go(func() { _ = src.Close() })
	wg.Wait()
}

func TestOpenErrors(t *testing.T) {
	fx := newFixture(t)
	cactPath, stPath := writeFiles(t, fx)

	short := slices.Clone(fx.names[:len(fx.names)-1])

	reshaped := slices.Clone(fx.names)
	reshaped[1].Shape = []int{4, 16}

	dup := slices.Clone(fx.names)
	dup[1].Name = "embedding"

	missing := slices.Clone(fx.names)
	missing[7].Source = "no/such/tensor"

	badLayer := slices.Clone(fx.names)
	badLayer[2].Layer = 2

	noManifest := slices.Clone(fx.names)
	noManifest[6].Name = "audio.other"

	openCact := func(names []cact.Entry) func() error {
		return func() error { _, err := OpenCact(cactPath, Options{Names: names}); return err }
	}

	openST := func(names []cact.Entry) func() error {
		return func() error { _, err := OpenSafetensors(stPath, Options{Names: names}); return err }
	}

	for _, tc := range []struct {
		name  string
		open  func() error
		match string
	}{
		{"cact: map shorter than the file", openCact(short), "10 tensors"},
		{"cact: shape mismatch", openCact(reshaped), "layer00.q_proj"},
		{"cact: duplicate name", openCact(dup), "twice"},
		{"cact: no manifest", openCact(noManifest), "audio.manifest"},
		{"cact: whistle map on a tiny file", openCact(nil), "681"},
		{"safetensors: missing source", openST(missing), "no/such/tensor"},
		{"safetensors: no such layer", openST(badLayer), "layer 2"},
		{"safetensors: shape mismatch", openST(reshaped), "layer00.q_proj"},
		{"safetensors: duplicate name", openST(dup), "twice"},
		{"cact as safetensors", func() error {
			_, err := OpenSafetensors(cactPath, Options{Names: fx.names})
			return err
		}, "safetensors"},
		{"missing file", func() error {
			_, err := Open(filepath.Join(t.TempDir(), "x.cact"), Options{})
			return err
		}, "no such file"},
		{"empty file", func() error { return openEmpty(t) }, "short"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.open()
			if err == nil || !strings.Contains(err.Error(), tc.match) {
				t.Errorf("error %v, want one mentioning %q", err, tc.match)
			}
		})
	}
}

func openEmpty(t *testing.T) error {
	t.Helper()

	path := filepath.Join(t.TempDir(), "empty")

	err := os.WriteFile(path, nil, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Open(path, Options{})

	return err
}

// TestManifestErrors: the .cact config needs the manifest's magic, version
// and integral values, as the engine does (F-006).
func TestManifestErrors(t *testing.T) {
	for _, tc := range []struct {
		name  string
		patch func(v []float32)
		match string
	}{
		{"magic", func(v []float32) { v[0] = 3247 }, "magic"},
		{"version", func(v []float32) { v[1] = 2 }, "version"},
		{"fraction", func(v []float32) { v[9] = 0.5 }, "not a count"},
		{"negative", func(v []float32) { v[3] = -1 }, "not a count"},
		{"short", func([]float32) {}, "want 13"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := tinyConfig()
			v := want.manifestRecord()
			tc.patch(v)

			if tc.name == "short" {
				v = v[:12]
			}

			var cfg Config

			err := cfg.setManifest(v)
			if err == nil || !strings.Contains(err.Error(), tc.match) {
				t.Errorf("error %v, want one mentioning %q", err, tc.match)
			}
		})
	}
}

// TestSafetensorsConfigNeedsQProj: the safetensors config does not state
// the head count; it comes from layer00.q_proj (F-007).
func TestSafetensorsConfigNeedsQProj(t *testing.T) {
	fx := newFixture(t)
	names := slices.DeleteFunc(slices.Clone(fx.names), func(e cact.Entry) bool { return e.Name == "layer00.q_proj" })

	_, err := ParseSafetensors(fx.st, Options{Names: names})
	if err == nil || !strings.Contains(err.Error(), "layer00.q_proj") {
		t.Errorf("error %v, want one about layer00.q_proj", err)
	}
}

// TestParseDoesNotOwnBytes: a source parsed from caller bytes maps
// nothing, and Close leaves the bytes alone.
func TestParseDoesNotOwnBytes(t *testing.T) {
	fx := newFixture(t)
	data := slices.Clone(fx.st)

	s, err := ParseSafetensors(data, Options{Names: fx.names})
	if err != nil {
		t.Fatal(err)
	}

	if s.Mapped() {
		t.Error("ParseSafetensors reports a mapping")
	}

	err = s.Close()
	if err != nil {
		t.Fatal(err)
	}

	_, err = safetensors.OpenStoreFromBytes(data, safetensors.StoreOptions{})
	if err != nil || !slices.Equal(data, fx.st) {
		t.Errorf("Close damaged the caller's bytes: %v", err)
	}
}
