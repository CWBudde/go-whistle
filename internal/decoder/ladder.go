package decoder

import (
	"fmt"
	"slices"

	"github.com/cwbudde/go-whistle/internal/needle"
)

// LadderOrder is upstream's _ladder_layer_order (architecture.py): both
// endpoint layers first, then one layer at a time from the middle of the
// largest gap between the layers chosen so far, the leftmost of equal
// gaps. For 8 layers it is 0, 7, 3, 5, 1, 2, 4, 6.
func LadderOrder(layers int) []int {
	if layers < 2 {
		return make([]int, max(layers, 0))
	}

	selected := []int{0, layers - 1}
	order := slices.Clone(selected)

	for len(order) < layers {
		slices.Sort(selected)

		gap, left := 0, 0

		for i := range len(selected) - 1 {
			if g := selected[i+1] - selected[i]; g > 1 && g > gap {
				gap, left = g, selected[i]
			}
		}

		mid := left + gap/2
		selected = append(selected, mid)
		order = append(order, mid)
	}

	return order
}

// LadderLayers is upstream's ladder_layer_indices: the first depth layers
// of LadderOrder, ascending. depth must be in [2, layers].
func LadderLayers(layers, depth int) ([]int, error) {
	if depth < 2 || depth > layers {
		return nil, fmt.Errorf("decoder: ladder depth must be in [2, %d], got %d", layers, depth)
	}

	out := LadderOrder(layers)[:depth]
	slices.Sort(out)

	return out, nil
}

// Depth is the number of decoder layers the decoder runs.
func (d *Decoder) Depth() int { return len(d.run) }

// Layers is the decoder layers the decoder runs, ascending.
func (d *Decoder) Layers() []int { return slices.Clone(d.run) }

// WithDepth returns the decoder that runs the endpoint-preserving ladder
// rung of depth layers, the engine's --audio-depth (docs/FINDINGS.md
// F-030). It is upstream's sub-network (Stack with exit_depth, or
// ladder_slice): the layers LadderLayers picks run in their original order
// with their own weights, every other layer passes the stream on
// unchanged, the i-th layer that runs takes mHC lane i % 4, an engram site
// acts only when its layer runs, and the final norm and the head stay. The
// weights, the tokenizer and the Memory format are shared with d, so a
// Memory serves every depth; a State belongs to the decoder that started
// it. depth = the full depth gives an equal decoder.
func (d *Decoder) WithDepth(depth int) (*Decoder, error) {
	run, err := LadderLayers(len(d.blocks), depth)
	if err != nil {
		return nil, err
	}

	t := *d
	t.run = run
	t.runMHC = make([]*needle.MHC, len(run))

	for i, l := range run {
		t.runMHC[i] = d.mhc[l].WithLane(i)
	}

	return &t, nil
}
