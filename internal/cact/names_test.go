package cact

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "rewrite testdata/whistle_names.txt from WhistleNames")

const namesGolden = "testdata/whistle_names.txt"

// renderNames is the golden form of the map: one line per index with the
// .cact name, shape, transform and the safetensors source with its layer.
func renderNames(es []Entry) []byte {
	var b bytes.Buffer

	b.WriteString("# WhistleNames() for whistle.cact @ b358ddad (docs/FINDINGS.md F-006).\n")
	b.WriteString("# Regenerate with: go test ./internal/cact -run TestWhistleNamesGolden -update\n")
	b.WriteString("# index  name  shape  transform  source[layer]\n")

	for i, e := range es {
		src := e.Source
		if src == "" {
			src = "-"
		}

		if e.Layer >= 0 {
			src += fmt.Sprintf("[%d]", e.Layer)
		}

		shape := strings.Trim(strings.ReplaceAll(fmt.Sprint(e.Shape), " ", ","), "[]")
		fmt.Fprintf(&b, "%3d  %-22s %-12s %-10s %s\n", i, e.Name, "("+shape+")", e.Transform, src)
	}

	return b.Bytes()
}

// TestWhistleNamesGolden pins every index's name, shape, transform and
// source.
func TestWhistleNamesGolden(t *testing.T) {
	got := renderNames(WhistleNames())

	if *update {
		err := os.WriteFile(namesGolden, got, 0o644)
		if err != nil {
			t.Fatal(err)
		}
	}

	want, err := os.ReadFile(namesGolden)
	if err != nil {
		t.Fatalf("%v (run with -update to create it)", err)
	}

	if !bytes.Equal(got, want) {
		gl, wl := strings.Split(string(got), "\n"), strings.Split(string(want), "\n")
		for k := range max(len(gl), len(wl)) {
			g, w := "", ""
			if k < len(gl) {
				g = gl[k]
			}

			if k < len(wl) {
				w = wl[k]
			}

			if g != w {
				t.Fatalf("%s line %d:\n got %q\nwant %q", namesGolden, k+1, g, w)
			}
		}
	}
}

// TestWhistleNamesInvariants checks the map's structure: the record count,
// unique names, every (source, layer) slice used once, every layer of a
// stacked source used, and exactly the derived tensors F-006 lists.
func TestWhistleNamesInvariants(t *testing.T) {
	es := WhistleNames()
	if len(es) != 681 {
		t.Fatalf("%d entries, want 681", len(es))
	}

	names := map[string]int{}
	slicesUsed := map[string]int{}
	layers := map[string][]int{}

	var derived []string

	for i, e := range es {
		if j, dup := names[e.Name]; dup {
			t.Errorf("name %q at %d and %d", e.Name, j, i)
		}

		names[e.Name] = i

		if e.Transform == Derived {
			derived = append(derived, e.Name)

			if e.Source != "" || e.Layer != -1 {
				t.Errorf("%d %s: derived with source %q layer %d", i, e.Name, e.Source, e.Layer)
			}

			continue
		}

		key := fmt.Sprintf("%s[%d]", e.Source, e.Layer)
		if j, dup := slicesUsed[key]; dup {
			t.Errorf("%s used by %d and %d", key, j, i)
		}

		slicesUsed[key] = i

		if e.Layer >= 0 {
			layers[e.Source] = append(layers[e.Source], e.Layer)
		}

		if numel(e.Shape) == 0 || len(e.Shape) == 0 {
			t.Errorf("%d %s: empty shape %v", i, e.Name, e.Shape)
		}
	}

	wantDerived := []string{"hada_p1", "hada_p2", "audio.manifest", "audio.mel_filters", "tokenizer"}
	if !slices.Equal(derived, wantDerived) {
		t.Errorf("derived = %v, want %v", derived, wantDerived)
	}

	for src, ls := range layers {
		if !slices.Equal(ls, []int{0, 1, 2, 3, 4, 5, 6, 7}) {
			t.Errorf("%s: layers %v, want 0..7 in order", src, ls)
		}
	}

	// 116 safetensors tensors: 80 stacked per layer (encoder 43, decoder 37
	// with the cross-attention) and 36 used whole.
	whole := len(slicesUsed) - 8*len(layers)
	if len(layers) != 80 || whole != 36 {
		t.Errorf("%d stacked and %d whole sources, want 80 and 36", len(layers), whole)
	}

	for name, i := range map[string]int{
		"embedding": 0, "layer00.norm_in": 1, "layer07.cond_u": 216, "mhc_a_pre": 217, "hada_p1": 226,
		"engram0.tables": 228, "engram1.taps": 235, "final_norm": 236, "audio.manifest": 237,
		"cross00.norm_in": 238, "cross07.attn_gate": 317, "pe_gate": 318, "enc00.pre_hada_0": 319,
		"enc07.hada.cond_u": 662, "enc.mhc_a_pre": 663, "enc.final_norm": 672, "stem.w": 673,
		"stem.out": 678, "audio.mel_filters": 679, "tokenizer": 680,
	} {
		if names[name] != i {
			t.Errorf("%s at %d, want %d", name, names[name], i)
		}
	}
}

func TestEntryApply(t *testing.T) {
	seq := func(n int) []float32 {
		v := make([]float32, n)
		for k := range v {
			v[k] = float32(k)
		}

		return v
	}

	for _, c := range []struct {
		name  string
		e     Entry
		src   []float32
		shape []int
		want  []float32
	}{
		{"whole reshape", Entry{"a", "s", -1, Reshape, []int{6}}, seq(6), []int{2, 3}, seq(6)},
		{"layer slice", Entry{"a", "s", 1, Reshape, []int{3}}, seq(6), []int{2, 3}, []float32{3, 4, 5}},
		{"layer scalar", Entry{"a", "s", 2, Reshape, []int{1}}, seq(3), []int{3}, []float32{2}},
		{"scalar", Entry{"a", "s", -1, Reshape, []int{1}}, []float32{7}, []int{}, []float32{7}},
		{
			"transpose layer",
			Entry{"a", "s", 1, Transpose, []int{3, 2}},
			seq(12),
			[]int{2, 2, 3},
			[]float32{6, 9, 7, 10, 8, 11},
		},
		{
			// phi (layers 2, nC 3, lanes 2) -> rows layer*lanes+lane over nC.
			"lane-major",
			Entry{"a", "s", -1, LaneMajor, []int{4, 3}},
			seq(12),
			[]int{2, 3, 2},
			[]float32{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11},
		},
	} {
		got, err := c.e.Apply(c.src, c.shape)
		if err != nil || !slices.Equal(got, c.want) {
			t.Errorf("%s: Apply = %v, %v; want %v", c.name, got, err, c.want)
		}

		if len(got) > 0 && &got[0] == &c.src[0] {
			t.Errorf("%s: result aliases the source", c.name)
		}
	}
}

func TestEntryApplyErrors(t *testing.T) {
	for _, c := range []struct {
		name  string
		e     Entry
		n     int // source values
		shape []int
	}{
		{"short source", Entry{"a", "s", -1, Reshape, []int{6}}, 6, []int{7}},
		{"layer past end", Entry{"a", "s", 2, Reshape, []int{3}}, 6, []int{2, 3}},
		{"layer of a scalar", Entry{"a", "s", 0, Reshape, []int{1}}, 1, []int{}},
		{"numel", Entry{"a", "s", -1, Reshape, []int{5}}, 6, []int{6}},
		{"transpose 1-D", Entry{"a", "s", -1, Transpose, []int{6}}, 6, []int{6}},
		{"transpose same", Entry{"a", "s", -1, Transpose, []int{2, 3}}, 6, []int{2, 3}},
		{"lane-major 2-D", Entry{"a", "s", -1, LaneMajor, []int{2, 3}}, 6, []int{2, 3}},
		{"lane-major nC", Entry{"a", "s", -1, LaneMajor, []int{3, 2}}, 6, []int{1, 3, 2}},
		{"unknown", Entry{"a", "s", -1, Transform(9), []int{6}}, 6, []int{6}},
	} {
		got, err := c.e.Apply(make([]float32, c.n), c.shape)
		if err == nil {
			t.Errorf("%s: Apply = %v, want an error", c.name, got)
		}
	}

	_, err := Entry{"a", "", -1, Derived, []int{6}}.Apply(make([]float32, 6), []int{6})
	if !errors.Is(err, ErrDerived) {
		t.Errorf("derived: err = %v, want ErrDerived", err)
	}
}

// TestRequantCQMatchesPortPack pins the fast RequantCQ, which the
// model-gated name map test uses, to the matrix reference portPack.
func TestRequantCQMatchesPortPack(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 9))

	for _, c := range []struct{ out, in, width, group int }{
		{3, 128, 2, 128}, {5, 300, 2, 128}, {4, 256, 4, 128}, {2, 40, 3, 16}, {7, 64, 1, 32},
	} {
		w := make([]float32, c.out*c.in)
		for k := range w {
			w[k] = float32(rng.NormFloat64())
		}

		cb := portCodebook(c.width, c.group)
		want := portPack(w, c.out, c.in, c.width, c.group, cb)

		got, margin := RequantCQ(w, c.out, c.in, c.width, c.group, cb)
		if !bytes.Equal(got, want.blob) {
			t.Errorf("%+v: RequantCQ differs from portPack", c)
		}

		for n, u := range want.unit {
			if m := BoundaryMargin(u, cb); math.Abs(margin[n]-m) > 1e-12 {
				t.Fatalf("%+v: margin[%d] = %v, want %v", c, n, margin[n], m)
			}
		}
	}
}

func TestBoundaryMargin(t *testing.T) {
	cb := []float32{-1, 0, 2} // boundaries at -0.5 and 1
	for u, want := range map[float64]float64{-3: 2.5, -0.5: 0, -0.25: 0.25, 0.9: 0.1, 1: 0, 5: 4} {
		if got := BoundaryMargin(u, cb); math.Abs(got-want) > 1e-15 {
			t.Errorf("BoundaryMargin(%v) = %v, want %v", u, got, want)
		}
	}
}
