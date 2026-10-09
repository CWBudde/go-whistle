package cact_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
)

// Pinned model files as laid out by `whistle download`, and the tokenizer
// fixture of docs/FINDINGS.md F-005.
const (
	whistleCactPath        = "../../models/whistle/whistle.cact"
	whistleSafetensorsPath = "../../models/whistle/whistle.safetensors"
	tokenizerFixturePath   = "../../testdata/tokenizer/whistle.json"
)

// The audio manifest's first two values (docs/FINDINGS.md F-006): the
// engine finds the manifest as the FP32 (13,) tensor that holds these.
const (
	audioManifestMagic   = 0xCAE
	audioManifestVersion = 1
)

// Hadamard permutation seeds, architecture.py _HADA_PERM_SEEDS.
var hadaPermSeeds = map[string]uint32{"hada_p1": 11, "hada_p2": 13}

// Measured minimum cosine per dtype against the source (F-006). They pin
// the files; drift past 1e-9 fails.
var pinnedMinCos = map[string]float64{
	"FP16": 0.999999963923, // enc00.q_norm
	"CQ2":  0.931853685259, // cross07.k_proj
	"CQ4":  0.994312193718, // enc.mhc_phi_post
}

// lloydMaxCos is sqrt(1 - D) for the Gaussian Lloyd-Max distortion D (Max
// 1960): the cosine CQ would reach on i.i.d. Gaussian weights. Reported for
// reference only (F-006).
var lloydMaxCos = map[uint32]float64{2: math.Sqrt(1 - 0.1175), 4: math.Sqrt(1 - 0.009497)}

// result is one tensor's check. exact checks compare every value; the others
// compare the cosine with its threshold. cos is NaN where no numeric
// comparison exists (the RAW tokenizer).
type result struct {
	cos, threshold float64
	exact          bool
	note           string
	ok             bool
}

func requireFile(t *testing.T, path string) {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", path)
	}
}

// TestWhistleNameMap walks all 681 tensors of whistle.cact @ b358ddad and
// checks each against its WhistleNames source (docs/FINDINGS.md F-006):
//
//   - FP16: every value is the round-to-nearest-even FP16 of the F32 source,
//     and the cosine meets the FP16 rounding bound.
//   - CQ: the record is bit for bit what the CQ quantiser makes of the
//     source (indices equal, norms within 1 FP16 ulp), and the cosine meets
//     the cosine of that requantisation, less the angle 1-ulp norms allow.
//   - derived: equal to a Go recomputation (permutations, mel filterbank),
//     to the config (audio manifest), or parsed (tokenizer).
func TestWhistleNameMap(t *testing.T) {
	requireFile(t, whistleCactPath)
	requireFile(t, whistleSafetensorsPath)

	f, err := cact.Open(whistleCactPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	store, err := safetensors.OpenStore(whistleSafetensorsPath, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	cfg, err := store.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}

	run, err := store.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	tensors, err := store.ReadAll()
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}

	names := cact.WhistleNames()
	if len(names) != f.Len() {
		t.Fatalf("%d map entries for %d tensors", len(names), f.Len())
	}

	bitsPlan := parseWeightBits(t, run.WeightBits)
	used := map[string]bool{}
	minCos := map[string]float64{}

	var below [5]int // CQ tensors under the Gaussian Lloyd-Max cosine, by width

	t.Logf("%3s  %-20s %-4s  %-11s  %-13s  %s", "idx", "name", "type", "cos", "threshold", "check")

	for i, e := range names {
		rec, _ := f.Record(i)
		kind := rec.DType.String()

		if rec.DType == cact.CQ {
			kind += strconv.Itoa(int(rec.Bits))
		}

		if !slices.Equal(rec.Dims(), e.Shape) && (e.Shape != nil || rec.NDim != 0) {
			t.Errorf("%d %s: record %s %v, map shape %v", i, e.Name, kind, rec.Dims(), e.Shape)

			continue
		}

		var r result

		if e.Transform == cact.Derived {
			r = checkDerived(t, f, i, e, cfg)
		} else {
			used[e.Source] = true

			src, ok := tensors[e.Source]
			if !ok {
				t.Errorf("%d %s: no safetensors %q", i, e.Name, e.Source)

				continue
			}

			r = checkSourced(f, i, e, src, bitsPlan)

			if c, seen := minCos[kind]; !seen || r.cos < c {
				minCos[kind] = r.cos
			}

			if rec.DType == cact.CQ && r.cos < lloydMaxCos[rec.Bits] {
				below[rec.Bits]++
			}
		}

		threshold := fmt.Sprintf("≥ %.9f", r.threshold)
		if r.exact {
			threshold = "exact"
		}

		line := fmt.Sprintf("%3d  %-20s %-4s  %11.9f  %-13s  %s", i, e.Name, kind, r.cos, threshold, r.note)
		if math.IsNaN(r.cos) {
			line = fmt.Sprintf("%3d  %-20s %-4s  %11s  %-13s  %s", i, e.Name, kind, "n/a", threshold, r.note)
		}

		t.Log(line)

		if !r.ok {
			t.Errorf("FAIL %s", line)
		}
	}

	checkSourcesUsed(t, store.Names(), used)

	for _, kind := range []string{"FP16", "CQ2", "CQ4"} {
		t.Logf("min cosine %-4s %.12f (pinned %.12f)", kind, minCos[kind], pinnedMinCos[kind])

		if math.Abs(minCos[kind]-pinnedMinCos[kind]) > 1e-9 {
			t.Errorf("min cosine %s = %.12f, pinned %.12f", kind, minCos[kind], pinnedMinCos[kind])
		}
	}

	t.Logf("below the Gaussian Lloyd-Max cosine: CQ2 %d of 145 (%.6f), CQ4 %d of 7 (%.6f)",
		below[2], lloydMaxCos[2], below[4], lloydMaxCos[4])
}

// checkSourced checks a tensor with a safetensors source.
func checkSourced(f *cact.File, i int, e cact.Entry, src *safetensors.Tensor, bitsPlan map[string]uint32) result {
	shape := make([]int, len(src.Shape))
	for k, d := range src.Shape {
		shape[k] = int(d)
	}

	want, err := e.Apply(src.Data, shape)
	if err != nil {
		return result{math.NaN(), 0, false, err.Error(), false}
	}

	rec, _ := f.Record(i)

	switch rec.DType {
	case cact.FP16:
		got, err := f.FP16(i)
		if err != nil {
			return result{math.NaN(), 0, false, err.Error(), false}
		}

		return checkFP16(got, want)
	case cact.CQ:
		if planned := planBits(bitsPlan, e.Source); rec.Bits != planned {
			return result{math.NaN(), 0, false, fmt.Sprintf("CQ%d, run plan says %d bits", rec.Bits, planned), false}
		}

		return checkCQ(f, i, rec, want)
	default:
		return result{math.NaN(), 0, false, "sourced tensor stored as " + rec.DType.String(), false}
	}
}

// checkFP16 requires every stored value to be the round-to-nearest-even
// FP16 of the source, and the cosine to meet the bound that rounding
// guarantees: with |x̂ᵢ - xᵢ| <= hᵢ, half an FP16 ulp at xᵢ (2^-25 below the
// normal range), the angle has sin θ <= ‖h‖/‖x‖.
func checkFP16(got, want []float32) result {
	mismatch := 0

	var hh, xx float64

	for k, x := range want {
		if got[k] != cact.Float16ToFloat32(cact.Float32ToFloat16(x)) {
			mismatch++
		}

		h := halfULP16(float64(x))
		hh += h * h
		xx += float64(x) * float64(x)
	}

	threshold := 1.0
	if xx > 0 {
		threshold = math.Sqrt(max(0, 1-hh/xx))
	}

	c := cosine(got, want)
	note := fmt.Sprintf("fp16(src) %d/%d", len(want)-mismatch, len(want))

	return result{c, threshold, false, note, mismatch == 0 && c >= threshold}
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

// tieTol bounds how far upstream's float32 rotation can move a unit
// coordinate u = (g @ H)_j / ‖g @ H‖: H's entries round 1/√128 (2^-24
// relative), the 128-term sum adds γ₁₂₈ ≈ 128·2^-24 of ‖g‖₁/√128 ≤ ‖g‖, and
// the norm in the denominator at most as much again: 2·(128+2)·2^-24.
const tieTol = 260 * 0x1p-24

// checkCQ requantises the source with the CQ quantiser (export.py _cq_pack,
// in float64) and requires the record to match it: FP16 norms within 1 ulp,
// and every packed index equal, except near-ties: an index may sit on the
// adjacent level where the exact u lies within tieTol of the decision
// boundary between the two, so float32 rounding decides it. The cosine
// threshold is the cosine Q(src) reaches, less the angle 1-ulp norms can
// add: a 1-ulp change scales a group by at most 2^-10, so the dequantised
// tensor turns by at most asin(2^-10). A near-tie flip barely moves the
// cosine, since u is about equally far from both levels.
func checkCQ(f *cact.File, i int, rec cact.Record, want []float32) result {
	out, in, group, width := int(rec.Shape[0]), int(rec.Shape[1]), int(rec.Group), int(rec.Bits)

	cb, err := f.Codebook(width)
	if err != nil {
		return result{math.NaN(), 0, false, err.Error(), false}
	}

	req, margin := cact.RequantCQ(want, out, in, width, group, cb)
	blob := f.Blob(i)
	groups := (in + group - 1) / group
	idxBytes := out * groups * group * width / 8

	flips, badFlips, worst := 0, 0, 0.0

	for k := range idxBytes {
		if blob[k] == req[k] {
			continue
		}

		for n := 8 * k / width; n*width < 8*(k+1); n++ {
			a, b := index(blob, n, width), index(req, n, width)
			if a == b {
				continue
			}

			flips++
			worst = max(worst, margin[n])

			if a-b != 1 && b-a != 1 || margin[n] > tieTol {
				badFlips++
			}
		}
	}

	normDiff, maxULP := 0, 0

	for k := idxBytes; k < len(blob); k += 2 {
		a, b := int(binary.LittleEndian.Uint16(blob[k:])), int(binary.LittleEndian.Uint16(req[k:]))
		if a != b {
			normDiff++
			maxULP = max(maxULP, a-b, b-a)
		}
	}

	deq, err := cact.DequantCQ(rec, req, cb)
	if err != nil {
		return result{math.NaN(), 0, false, err.Error(), false}
	}

	got, err := f.CQ(i)
	if err != nil {
		return result{math.NaN(), 0, false, err.Error(), false}
	}

	threshold := math.Cos(math.Acos(cosine(deq, want)) + math.Asin(0x1p-10))
	c := cosine(got, want)

	note := "= Q(src)"
	if normDiff > 0 {
		note += fmt.Sprintf(", norm ±%d ulp ×%d", maxULP, normDiff)
	}

	if flips > 0 {
		note += fmt.Sprintf(", %d near-tie idx (margin ≤ %.1e)", flips, worst)
	}

	return result{c, threshold, false, note, badFlips == 0 && maxULP <= 1 && c >= threshold}
}

// index reads index n of an LSB-first stream of width-bit indices.
func index(stream []byte, n, width int) int {
	bit := n * width

	v := int(stream[bit/8])
	if bit/8+1 < len(stream) {
		v |= int(stream[bit/8+1]) << 8
	}

	return v >> (bit % 8) & (1<<width - 1)
}

// checkDerived checks a tensor without a safetensors source.
func checkDerived(t *testing.T, f *cact.File, i int, e cact.Entry, cfg safetensors.Config) result {
	t.Helper()

	var want []float32

	switch e.Name {
	case "hada_p1", "hada_p2":
		hadaN := 1 << bits.Len(uint(cfg.Dim-1)) // next power of two
		for _, v := range numpyPermutation(hadaPermSeeds[e.Name], hadaN) {
			want = append(want, float32(v))
		}
	case "audio.manifest":
		want = []float32{audioManifestMagic, audioManifestVersion}
		for _, v := range []int{
			cfg.Text, cfg.EncLayers, cfg.EncLanes, cfg.ConvKernel, cfg.Stem, cfg.StemStages,
			cfg.Mels, cfg.Rate, cfg.FFT, cfg.Win, cfg.Hop,
		} {
			want = append(want, float32(v))
		}
	case "audio.mel_filters":
		want = slaneyMel(cfg.Rate, cfg.FFT, cfg.Mels)
	case "tokenizer":
		return checkTokenizer(t, f, i, cfg)
	default:
		return result{math.NaN(), 0, true, "no check for derived " + e.Name, false}
	}

	got, err := f.FP32(i)
	if err != nil {
		return result{math.NaN(), 0, true, err.Error(), false}
	}

	notes := map[string]string{
		"hada_p1":           "= RandomState(11).permutation(512)",
		"hada_p2":           "= RandomState(13).permutation(512)",
		"audio.manifest":    "= [0xCAE, 1, text … hop] from the config",
		"audio.mel_filters": "= Slaney mel in float64, rounded to float32",
	}

	return result{cosine(got, want), 1, true, notes[e.Name], slices.Equal(got, want)}
}

// checkTokenizer parses the RAW blob, matches its size and special ids to
// the header and config, and requires the bytes F-005's sentencepiece
// fixture was generated from.
func checkTokenizer(t *testing.T, f *cact.File, i int, cfg safetensors.Config) result {
	t.Helper()

	blob, err := f.Raw(i)
	if err != nil {
		return result{math.NaN(), 0, true, err.Error(), false}
	}

	tok, err := tokenizer.Parse(blob)
	if err != nil {
		return result{math.NaN(), 0, true, err.Error(), false}
	}

	data, err := os.ReadFile(tokenizerFixturePath)
	if err != nil {
		t.Fatalf("tokenizer fixture: %v", err)
	}

	var fixture struct {
		Source struct {
			BlobSHA256 string `json:"blob_sha256"`
		} `json:"source"`
	}

	err = json.Unmarshal(data, &fixture)
	if err != nil {
		t.Fatalf("tokenizer fixture: %v", err)
	}

	sum := sha256.Sum256(blob)
	ok := hex.EncodeToString(sum[:]) == fixture.Source.BlobSHA256 &&
		tok.Len() == int(f.Header().Vocab) && tok.Len() == cfg.Text+7 &&
		tok.PadID() == cfg.Pad && tok.EOSID() == cfg.EOS && tok.BOSID() == cfg.BOS

	note := fmt.Sprintf("parse: %d pieces, pad/eos/bos %d/%d/%d; sha256 = F-005 fixture",
		tok.Len(), tok.PadID(), tok.EOSID(), tok.BOSID())

	return result{math.NaN(), 0, true, note, ok}
}

// checkSourcesUsed requires every safetensors tensor to feed some .cact
// tensor. None is training-only: the checkpoint holds inference weights.
func checkSourcesUsed(t *testing.T, all []string, used map[string]bool) {
	t.Helper()

	var unused []string

	for _, name := range all {
		if !used[name] {
			unused = append(unused, name)
		}
	}

	if len(unused) > 0 || len(used) != len(all) {
		t.Errorf("safetensors: %d used of %d, unused %v", len(used), len(all), unused)
	}

	t.Logf("safetensors: %d of %d tensors used, %d unused", len(used), len(all), len(unused))
}

// parseWeightBits reads the run's per-module plan, e.g.
// "embedding=4,stack/mhc=4,encoder/mhc=4,default=2".
func parseWeightBits(t *testing.T, plan string) map[string]uint32 {
	t.Helper()

	out := map[string]uint32{}

	for item := range strings.SplitSeq(plan, ",") {
		key, val, ok := strings.Cut(item, "=")

		width, err := strconv.ParseUint(val, 10, 32)
		if !ok || err != nil {
			t.Fatalf("weight_bits %q: bad item %q", plan, item)
		}

		out[key] = uint32(width)
	}

	return out
}

// planBits is the width the plan gives a safetensors name: the longest
// matching module prefix, else the default.
func planBits(plan map[string]uint32, name string) uint32 {
	best, width := -1, plan["default"]

	for prefix, b := range plan {
		if prefix != "default" && strings.HasPrefix(name, prefix) && len(prefix) > best {
			best, width = len(prefix), b
		}
	}

	return width
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
