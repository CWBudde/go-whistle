package decoder

import (
	"math"
	"slices"
	"testing"
)

// TestCapture pins the cross-attention hook (PLAN.md 7.1, docs/FINDINGS.md
// F-033): it reports every layer that runs once per Extend, in layer order,
// with probabilities that sum to 1 over the frames and the output gate in
// [0, 1] (σ saturates in float32); turning it on leaves the logits bit for
// bit unchanged; feeding the sequence row by row reports the rows of a
// one-pass feed; a fork does not inherit it; a ladder rung reports only its
// layers.
func TestCapture(t *testing.T) {
	src := openModel(t, whistleCactPath)
	clips := loadClips(t, fastClips[2])
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

	plain, err := d.Extend(d.NewState(mem), seq)
	if err != nil {
		t.Fatal(err)
	}

	var got []Attention

	s := d.NewState(mem)
	s.SetCapture(func(a Attention) { got = append(got, a) })

	logits, err := d.Extend(s, seq)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(logits, plain) {
		t.Fatal("the capture changed the logits")
	}

	checkAttention(t, got, len(seq), mem.Frames())

	// Row by row, through a fork of a captured state.
	parent := d.NewState(mem)
	parent.SetCapture(func(Attention) { t.Error("the parent reported its fork's attention") })

	f := parent.Fork()
	worst := 0.0

	for i, tok := range seq {
		var rows []Attention

		f.SetCapture(func(a Attention) { rows = append(rows, a) })

		_, err := d.Extend(f, []int{tok})
		if err != nil {
			t.Fatal(err)
		}

		for l, a := range rows {
			if a.Pos != i || a.Tokens != 1 {
				t.Fatalf("step %d: pos %d, tokens %d", i, a.Pos, a.Tokens)
			}

			n := a.Heads * a.Frames
			for j, p := range a.Probs {
				worst = math.Max(worst, math.Abs(float64(p-got[l].Probs[i*n+j])))
			}
		}
	}

	if worst > 1e-5 {
		t.Errorf("row by row: max |Δp| %.3g against one pass", worst)
	}

	rung, err := d.WithDepth(2)
	if err != nil {
		t.Fatal(err)
	}

	var layers []int

	s = rung.NewState(mem)
	s.SetCapture(func(a Attention) { layers = append(layers, a.Layer) })

	_, err = rung.Extend(s, seq[:2])
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(layers, []int{0, 7}) {
		t.Errorf("depth 2 reported layers %v, want [0 7]", layers)
	}

	t.Logf("%d tokens × 8 layers × 8 heads × %d frames; row by row max |Δp| %.3g", len(seq), mem.Frames(), worst)
}

// checkAttention checks one Extend's captures of the full decoder on tokens
// rows: one per layer in order, the shapes, rows of probabilities summing
// to 1 and gates in [0, 1].
func checkAttention(t *testing.T, got []Attention, tokens, frames int) {
	t.Helper()

	if len(got) != 8 {
		t.Fatalf("%d attentions for one Extend, want 8", len(got))
	}

	for l, a := range got {
		if a.Layer != l || a.Pos != 0 || a.Tokens != tokens || a.Heads != 8 || a.Frames != frames {
			t.Fatalf("attention %d: layer %d pos %d tokens %d heads %d frames %d", l, a.Layer, a.Pos, a.Tokens,
				a.Heads, a.Frames)
		}

		if len(a.Probs) != a.Tokens*a.Heads*a.Frames || len(a.Gate) != a.Tokens*a.Heads*64 {
			t.Fatalf("layer %d: %d probabilities, %d gates", l, len(a.Probs), len(a.Gate))
		}

		for r := range a.Tokens * a.Heads {
			var sum float64
			for _, p := range a.Probs[r*a.Frames : (r+1)*a.Frames] {
				sum += float64(p)
			}

			if math.Abs(sum-1) > 1e-5 {
				t.Fatalf("layer %d row %d sums to %v", l, r, sum)
			}
		}

		for _, g := range a.Gate {
			if g < 0 || g > 1 {
				t.Fatalf("layer %d: gate %v outside [0, 1]", l, g)
			}
		}
	}
}
