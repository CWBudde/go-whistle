package encoder

import (
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// TestBestPackedMatchesFloat runs Best on whistle.cact packed and in float
// mode over the corpus (docs/FINDINGS.md F-037): the packed embeddings
// match needle_embed as well as the float ones (mean frame cosine 0.999827
// against 0.999828 on all 32 clips, arm64 and amd64), and the float ones
// at a mean frame 1 − cos of 4.8e-8 on all 32 clips and 2.3e-7 on the
// -short ones (arm64; amd64 7.0e-8 and 4.5e-7) and a worst frame cosine of
// 0.999945 (arm64; amd64 0.999893, against its AVX2+FMA float GEMM). That is the lut kernel's table
// rounding (F-034); under purego the two agree to 1.0000000. The floors
// sit just below the worst architecture.
func TestBestPackedMatchesFloat(t *testing.T) {
	float := openModel(t, whistleCactPath)

	packed, err := weights.Open(whistleCactPath, weights.Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = packed.Close() })

	ep, err := Load(packed, Best())
	if err != nil {
		t.Fatal(err)
	}

	ef, err := Load(float, Best())
	if err != nil {
		t.Fatal(err)
	}

	var ids []string
	if testing.Short() {
		ids = slices.Sorted(func(yield func(string) bool) {
			for id := range shortClips[whistleCactPath] {
				if !yield(id) {
					return
				}
			}
		})
	}

	var sumP, sumF, sumPF float64

	worst := 1.0
	clips := loadClips(t, ids...)

	for _, c := range clips {
		yp, n, err := ep.Embed(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		yf, _, err := ef.Embed(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		sumP += mean(frameCosines(yp, c.Embed, n, ep.dim))
		sumF += mean(frameCosines(yf, c.Embed, n, ef.dim))
		pf := frameCosines(yp, yf, n, ep.dim)
		sumPF += mean(pf)
		worst = min(worst, slices.Min(pf))
	}

	n := float64(len(clips))
	mp, mf, mpf := sumP/n, sumF/n, sumPF/n
	t.Logf("%d clips: mean frame cosine to needle_embed packed %.6f, float %.6f; packed vs float mean 1 − cos %.2e, worst frame %.7f",
		len(clips), mp, mf, 1-mpf, worst)

	if mp < 0.9998 || mp < mf-2e-5 {
		t.Errorf("packed mean cosine %.6f, float %.6f", mp, mf)
	}

	if 1-mpf > 6e-7 || worst < 0.99985 {
		t.Errorf("packed vs float: mean frame 1 − cos %.2e > 6e-7 or worst cosine %.7f < 0.99985", 1-mpf, worst)
	}
}
