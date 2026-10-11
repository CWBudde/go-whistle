package cact_test

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// Pinned text model files as laid out by `just fetch-needle3`.
const (
	needle3CactPath        = "../../models/needle3/needle3.cact"
	needle3SafetensorsPath = "../../models/needle3/needle3.safetensors"
)

// headBits is quantize.py HEAD_BITS: export.py writes the probe-head
// matrices at this width whatever the run's plan says.
const headBits = 4

// Measured minimum cosine per kind against the source (docs/FINDINGS.md
// F-045). FP16 is exact: the checkpoint is F16 already.
var pinnedNeedle3MinCos = map[string]float64{
	"FP16": 1,
	"CQ2":  0.933338362129, // layer17.out_proj
	"CQ4":  0.994432977749, // confidence_head.proj
}

func requireNeedle3(t *testing.T, path string) {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model", path)
	}
}

// TestNeedle3NameMap walks all 581 tensors of needle3.cact @ 93e88ede and
// checks each against its TextNames source in needle3.safetensors, with
// TestWhistleNameMap's checks: FP16 records are the source's values, CQ
// records are bit for bit what the CQ quantiser makes of the source (up to
// near-ties and 1-ulp norms), and derived records equal a recomputation.
func TestNeedle3NameMap(t *testing.T) {
	requireNeedle3(t, needle3CactPath)
	requireNeedle3(t, needle3SafetensorsPath)

	f, err := cact.Open(needle3CactPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	store, err := safetensors.OpenStore(needle3SafetensorsPath, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	cfg, err := store.TextConfig()
	if err != nil {
		t.Fatalf("TextConfig: %v", err)
	}

	tensors, err := store.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	names, err := cact.TextNames(f)
	if err != nil {
		t.Fatalf("TextNames: %v", err)
	}

	if len(names) != f.Len() {
		t.Fatalf("%d map entries for %d tensors", len(names), f.Len())
	}

	bitsPlan := parseWeightBits(t, cfg.WeightBits)
	used := map[string]bool{}
	minCos := map[string]float64{}

	t.Logf("%3s  %-24s %-4s  %-11s  %-13s  %s", "idx", "name", "type", "cos", "threshold", "check")

	for i, e := range names {
		rec, _ := f.Record(i)
		kind := rec.DType.String()

		if rec.DType == cact.CQ {
			kind += strconv.Itoa(int(rec.Bits))
		}

		var r result

		if e.Transform == cact.Derived {
			r = checkTextDerived(f, i, e, cfg)
		} else {
			used[e.Source] = true

			src, ok := tensors[e.Source]
			if !ok {
				t.Errorf("%d %s: no safetensors %q", i, e.Name, e.Source)

				continue
			}

			plan := bitsPlan
			if strings.HasSuffix(strings.SplitN(e.Source, "/", 2)[0], "_head") {
				plan = map[string]uint32{"default": headBits}
			}

			r = checkSourced(f, i, e, src, plan)

			if c, seen := minCos[kind]; !seen || r.cos < c {
				minCos[kind] = r.cos
			}
		}

		threshold := fmt.Sprintf("≥ %.9f", r.threshold)
		if r.exact {
			threshold = "exact"
		}

		line := fmt.Sprintf("%3d  %-24s %-4s  %11.9f  %-13s  %s", i, e.Name, kind, r.cos, threshold, r.note)
		if math.IsNaN(r.cos) {
			line = fmt.Sprintf("%3d  %-24s %-4s  %11s  %-13s  %s", i, e.Name, kind, "n/a", threshold, r.note)
		}

		t.Log(line)

		if !r.ok {
			t.Errorf("FAIL %s", line)
		}
	}

	checkSourcesUsed(t, store.Names(), used)

	for _, kind := range []string{"FP16", "CQ2", "CQ4"} {
		t.Logf("min cosine %-4s %.12f (pinned %.12f)", kind, minCos[kind], pinnedNeedle3MinCos[kind])

		if math.Abs(minCos[kind]-pinnedNeedle3MinCos[kind]) > 1e-9 {
			t.Errorf("min cosine %s = %.12f, pinned %.12f", kind, minCos[kind], pinnedNeedle3MinCos[kind])
		}
	}
}

// checkTextDerived checks a text tensor without a safetensors source: the
// Hadamard permutations, the heads.manifest and the tokenizer.
func checkTextDerived(f *cact.File, i int, e cact.Entry, cfg safetensors.TextConfig) result {
	var want []float32

	switch e.Name {
	case "hada_p1", "hada_p2":
		seed, _ := cact.HadaPermSeed(e.Name)

		for _, v := range cact.NumpyPermutation(seed, cfg.HadaN()) {
			want = append(want, float32(v))
		}

		got, err := f.FP32(i)
		if err != nil {
			return result{math.NaN(), 0, true, err.Error(), false}
		}

		note := fmt.Sprintf("= RandomState(%d).permutation(%d)", seed, cfg.HadaN())

		return result{cosine(got, want), 1, true, note, slices.Equal(got, want)}
	case "heads.manifest":
		// The checkpoint holds a confidence head only (config.json extras.heads).
		got, err := f.FP16(i)
		if err != nil {
			return result{math.NaN(), 0, true, err.Error(), false}
		}

		return result{
			math.NaN(), 0, true, fmt.Sprintf("codes %v = [confidence]", got),
			slices.Equal(got, []float32{cact.HeadConfidence}),
		}
	case "tokenizer":
		tok, err := tokenizer.FromCact(f)
		if err != nil {
			return result{math.NaN(), 0, true, err.Error(), false}
		}

		ok := tok.Len() == cfg.Vocab && tok.PadID() == 0 && tok.EOSID() == 1 && tok.BOSID() == 2

		return result{math.NaN(), 0, true, fmt.Sprintf("parse: %d pieces, pad/eos/bos %d/%d/%d",
			tok.Len(), tok.PadID(), tok.EOSID(), tok.BOSID()), ok}
	default:
		return result{math.NaN(), 0, true, "no check for derived " + e.Name, false}
	}
}
