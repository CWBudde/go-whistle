package decoder

import (
	"math"
	"testing"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// TestPackedMatchesFloat decodes the corpus (fastClips under -short)
// greedily with Best() on whistle.cact packed and in float mode
// (docs/FINDINGS.md F-037): every transcript and language is the same, and
// the teacher-forced mean log-probabilities of the engine's transcripts
// differ by at most 3e-4 (measured 7.2e-5 on arm64, 1.9e-4 on amd64 and
// 1.9e-7 under purego). Loading the encoder and the decoder from the
// packed source decodes no CQ tensor to float32.
func TestPackedMatchesFloat(t *testing.T) {
	float := openModel(t, whistleCactPath)

	packed, err := weights.Open(whistleCactPath, weights.Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = packed.Close() })

	var ids []string
	if testing.Short() {
		ids = fastClips
	}

	clips := loadClips(t, ids...)
	run := map[string]func(i int) (Result, float64){}

	for name, src := range map[string]weights.Source{"packed": packed, "float": float} {
		encs, frames := encode(t, src, clips, MemFinalNorm)

		d, err := Load(src, nil, Best())
		if err != nil {
			t.Fatal(err)
		}

		run[name] = func(i int) (Result, float64) {
			mem, err := d.Memory(encs[i], frames[i])
			if err != nil {
				t.Fatal(err)
			}

			r, err := d.GreedyMemory(mem, "")
			if err != nil {
				t.Fatal(err)
			}

			seq, err := d.Targets(clips[i].Language, clips[i].Text)
			if err != nil {
				t.Fatal(err)
			}

			f, err := d.TeacherForce(mem, seq)
			if err != nil {
				t.Fatal(err)
			}

			return r, mean64(f.LogProbs)
		}
	}

	checkNoFloatCQ(t, packed)

	worst := 0.0

	for i, c := range clips {
		rp, lp := run["packed"](i)
		rf, lf := run["float"](i)

		if rp.Text != rf.Text || rp.Language != rf.Language {
			t.Errorf("%s: packed [%s] %q, float [%s] %q", c.ID, rp.Language, rp.Text, rf.Language, rf.Text)
		}

		worst = max(worst, math.Abs(lp-lf))
	}

	t.Logf("%d clips: greedy transcripts equal; teacher-forced mean logp differs by at most %.2e", len(clips), worst)

	if worst > 3e-4 {
		t.Errorf("teacher-forced mean logp differs by %.2e > 3e-4", worst)
	}
}

// checkNoFloatCQ fails if src's float32 cache holds a tensor it serves
// packed, after the encoder and the decoder were loaded from it.
func checkNoFloatCQ(t *testing.T, src weights.Source) {
	t.Helper()

	cached, ok := src.(interface{ Cached() []string })
	if !ok {
		t.Fatalf("%T has no Cached", src)
	}

	floats := 0

	for _, name := range cached.Cached() {
		floats++

		if _, packed, _ := weights.PackedMatrix(src, name); packed {
			t.Errorf("%s is packed but also decoded to float32", name)
		}
	}

	t.Logf("after Load: %d float32 tensors cached, no CQ tensor among them", floats)

	if floats == 0 {
		t.Error("nothing cached: the check would pass vacuously")
	}
}
