package safetensors

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

// needle3ModelPath is the pinned text checkpoint as laid out by `just fetch-needle3`.
const needle3ModelPath = "../../models/needle3/needle3.safetensors"

func wantNeedle3Config() TextConfig {
	return TextConfig{
		Vocab: 8192, Dim: 768, Heads: 12, KVHeads: 2, Layers: 20, QKDim: 48, VDim: 64,
		MaxSeqLen: 8192, DType: "bfloat16", Slots: 18432, Sites: []int{3, 7, 11, 15, 19},
		SlidingWindow: 1024, GlobalLayers: []int{4, 9, 14, 19}, KVWindow: 256,
		LadderDepths: []int{}, LadderWidths: []int{}, Lanes: 4, Taps: 3, OutVocab: 8192,
		KVBits: 8, ActBits: 8, WeightBits: "embedding=4,stack/mhc=4,default=2",
		ConfProbes: 4, ConfQueries: 4,
	}
}

func TestTextConfigDefaults(t *testing.T) {
	cfg := wantNeedle3Config()

	if got := cfg.Orders(); !reflect.DeepEqual(got, []int{2, 3}) {
		t.Errorf("Orders() = %v", got)
	}

	if cfg.Rope() != 1e5 {
		t.Errorf("Rope() = %v", cfg.Rope())
	}

	// engram_geometry: heads = 768 / (2 * 128) = 3, 6 tables of 128.
	if cfg.Hashes() != 3 || cfg.Tables() != 6 || cfg.SubDim() != 128 {
		t.Errorf("engram geometry = %d hashes, %d tables, sub_dim %d; want 3, 6, 128",
			cfg.Hashes(), cfg.Tables(), cfg.SubDim())
	}

	if cfg.HadaN() != 1024 {
		t.Errorf("HadaN() = %d, want 1024", cfg.HadaN())
	}

	small := TextConfig{Dim: 512}
	if small.Hashes() != 2 || small.Tables() != 4 || small.HadaN() != 512 {
		t.Errorf("d_model 512: %d hashes, %d tables, HadaN %d; want Whistle's 2, 4, 512",
			small.Hashes(), small.Tables(), small.HadaN())
	}
}

func TestTextConfigStrict(t *testing.T) {
	store := storeWithMetadata(t, map[string]string{"config": `{"vocab_size": 8192}`})

	_, err := store.TextConfig()
	if err == nil {
		t.Fatal("TextConfig accepted a config without most keys")
	}

	store = storeWithMetadata(t, map[string]string{"config": `{"text": 8192}`})

	_, err = store.TextConfig()
	if err == nil {
		t.Fatal("TextConfig accepted Whistle's key text")
	}
}

// TestTextConfigDim rejects a d_model HadaN cannot round up to a power of two (PR #44 review).
func TestTextConfigDim(t *testing.T) {
	for _, dim := range []int{0, -1, maxTextDim + 1, math.MaxInt} {
		cfg := wantNeedle3Config()
		cfg.Dim = dim

		raw, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}

		store := storeWithMetadata(t, map[string]string{"config": string(raw)})

		_, err = store.TextConfig()
		if err == nil || !strings.Contains(err.Error(), "d_model") {
			t.Errorf("d_model %d: err = %v, want one mentioning d_model", dim, err)
		}
	}

	raw, err := json.Marshal(wantNeedle3Config())
	if err != nil {
		t.Fatal(err)
	}

	got, err := storeWithMetadata(t, map[string]string{"config": string(raw)}).TextConfig()
	if err != nil || got.Dim != 768 {
		t.Errorf("needle3 config: Dim %d, err %v", got.Dim, err)
	}
}

// TestNeedle3Checkpoint pins needle3.safetensors @ 93e88ede: 64 F16 tensors, the config and run in __metadata__.
// Every tensor that differs from Whistle's checkpoint is listed in docs/FINDINGS.md F-045.
func TestNeedle3Checkpoint(t *testing.T) {
	_, err := os.Stat(needle3ModelPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3` to fetch the pinned text model", needle3ModelPath)
	}

	store, err := OpenStore(needle3ModelPath, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	const (
		wantTensors = 64
		// config.json total_parameters is 121,021,910: 2049 more, the size of the derived .cact records
		// hada_p1 + hada_p2 (1024 each) and heads.manifest (1).
		wantParams = 121_019_861
	)

	names := store.Names()
	if len(names) != wantTensors {
		t.Errorf("%d tensors, want %d", len(names), wantTensors)
	}

	params := 0

	for _, name := range names {
		tensor, err := store.Tensor(name)
		if err != nil {
			t.Fatalf("Tensor(%s): %v", name, err)
		}

		if tensor.DType != dtypeF16 {
			t.Errorf("%s is %s, want F16", name, tensor.DType)
		}

		params += len(tensor.Data)
	}

	if params != wantParams {
		t.Errorf("%d parameters, want %d", params, wantParams)
	}

	for name, shape := range map[string][]int64{
		"embedding/embedding":                        {8192, 768},
		"engrams_0/embedding":                        {6, 18432, 128},
		"stack/layers/block/hadamard_mlp/d1":         {20, 1024},
		"stack/layers/block/hadamard_mlp/cond_v":     {20, 768, 8},
		"stack/layers/block/self_attn/q_proj/kernel": {20, 768, 576},
		"stack/mhc_phi_res":                          {20, 3072, 16},
		"confidence_head/probes":                     {21, 4, 768},
		"confidence_head/proj/kernel":                {3072, 1},
	} {
		_, err = store.TensorWithShape(name, shape)
		if err != nil {
			t.Error(err)
		}
	}

	cfg, err := store.TextConfig()
	if err != nil {
		t.Fatalf("TextConfig: %v", err)
	}

	if want := wantNeedle3Config(); !reflect.DeepEqual(cfg, want) {
		t.Errorf("TextConfig() =\n%+v\nwant\n%+v", cfg, want)
	}

	run, err := store.TextRun()
	if err != nil {
		t.Fatalf("TextRun: %v", err)
	}

	// tokenizer_hash is the MD5 of the pinned tokenizer/tokenizer.model.
	if run.Stage != "qapt" || run.BaseCheckpoint != "needle3_rl.pkl" || run.MaxSteps != 10000 ||
		run.BitsEnd != 2 || run.TokenizerHash != "9697fdd2fbb126dd8de173771e49d99d" {
		t.Errorf("TextRun() = %+v", run)
	}

	step, err := store.Step()
	if err != nil || step != 10000 {
		t.Errorf("Step() = %d, %v; want 10000", step, err)
	}

	// The Whistle schema must reject the text config rather than decode it to zeros.
	_, err = store.Config()
	if err == nil {
		t.Error("Config() decoded the text config with Whistle's schema")
	}
}
