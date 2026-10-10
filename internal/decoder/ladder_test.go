package decoder

import (
	"slices"
	"testing"
)

// TestLadderOrder pins the port of upstream's _ladder_layer_order (the
// orders below are its output) and ladder_layer_indices: 0, 7, 3, 5, 1, 2,
// 4, 6 for Whistle's 8 layers (PLAN.md reference card).
func TestLadderOrder(t *testing.T) {
	for _, tc := range []struct {
		layers int
		want   []int
	}{
		{1, []int{0}},
		{2, []int{0, 1}},
		{3, []int{0, 2, 1}},
		{5, []int{0, 4, 2, 1, 3}},
		{8, []int{0, 7, 3, 5, 1, 2, 4, 6}},
		{12, []int{0, 11, 5, 8, 2, 3, 6, 9, 1, 4, 7, 10}},
	} {
		if got := LadderOrder(tc.layers); !slices.Equal(got, tc.want) {
			t.Errorf("LadderOrder(%d) = %v, want %v", tc.layers, got, tc.want)
		}
	}

	rungs := [][]int{
		2: {0, 7}, 3: {0, 3, 7}, 4: {0, 3, 5, 7}, 5: {0, 1, 3, 5, 7}, 6: {0, 1, 2, 3, 5, 7},
		7: {0, 1, 2, 3, 4, 5, 7}, 8: {0, 1, 2, 3, 4, 5, 6, 7},
	}

	for depth := 2; depth <= 8; depth++ {
		got, err := LadderLayers(8, depth)
		if err != nil || !slices.Equal(got, rungs[depth]) {
			t.Errorf("LadderLayers(8, %d) = %v, %v; want %v", depth, got, err, rungs[depth])
		}
	}

	for _, depth := range []int{-1, 0, 1, 9} {
		_, err := LadderLayers(8, depth)
		if err == nil {
			t.Errorf("LadderLayers(8, %d): no error", depth)
		}
	}
}

// TestWithDepth: the full rung is the decoder itself, logit for logit; a
// shorter rung runs its layers on the shared Memory without touching the
// parent decoder; depths outside [2, 8] are refused.
func TestWithDepth(t *testing.T) {
	src := openModel(t, whistleCactPath)
	clips := loadClips(t, fastClips[0])
	encs, frames := encode(t, src, clips, MemFinalNorm)

	d, err := Load(src, nil, Best())
	if err != nil {
		t.Fatal(err)
	}

	mem, err := d.Memory(encs[0], frames[0])
	if err != nil {
		t.Fatal(err)
	}

	seq, err := d.Targets(clips[0].Language, clips[0].Text)
	if err != nil {
		t.Fatal(err)
	}

	want, err := d.Extend(d.NewState(mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	full, err := d.WithDepth(8)
	if err != nil {
		t.Fatal(err)
	}

	got, err := full.Extend(full.NewState(mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, want) || full.Depth() != 8 || d.Depth() != 8 {
		t.Errorf("depth 8 differs from the decoder (depths %d, %d)", full.Depth(), d.Depth())
	}

	short, err := d.WithDepth(2)
	if err != nil {
		t.Fatal(err)
	}

	two, err := short.Extend(short.NewState(mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	_, _, cos := compare(two, want)
	if !slices.Equal(short.Layers(), []int{0, 7}) || cos > 0.999 {
		t.Errorf("depth 2: layers %v, logits cosine to the full depth %.6f", short.Layers(), cos)
	}

	again, err := d.Extend(d.NewState(mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(again, want) || d.Depth() != 8 {
		t.Error("WithDepth changed the parent decoder")
	}

	for _, depth := range []int{1, 9} {
		_, err := d.WithDepth(depth)
		if err == nil {
			t.Errorf("WithDepth(%d): no error", depth)
		}
	}
}
