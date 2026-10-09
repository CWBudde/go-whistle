package safetensors

import (
	"encoding/binary"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

// whistleConfigJSON is the __metadata__ "config" entry of whistle.safetensors
// at b358ddad, verbatim.
const whistleConfigJSON = `{"text": 8192, "dim": 512, "kv_heads": 2, "qk_dim": 48, "v_dim": 64, ` +
	`"enc_layers": 8, "dec_layers": 8, "conv_kernel": 9, "taps": 3, "lanes": 4, "enc_lanes": 4, ` +
	`"sites": [3, 7], "orders": [2, 3], "slots": 18432, "hashes": 2, "stem": 128, "stem_stages": 3, ` +
	`"rope": 100000.0, "mels": 80, "rate": 16000, "fft": 512, "win": 400, "hop": 160, ` +
	`"freq_masks": 2, "freq_width": 17, "time_masks": 10, "time_width": 0.05, "dither": 1e-05, ` +
	`"per_augment_prob": 0.1, "speed": [0.9, 1.1], "max_tokens": 320, "pad": 0, "eos": 1, "bos": 2, ` +
	`"dtype": "bfloat16"}`

const whistleRunJSON = `{"stage": "qapt", "base_checkpoint": "avg230-250.pkl", ` +
	`"weight_bits": "embedding=4,stack/mhc=4,encoder/mhc=4,default=2", "kv_bits": 8, "act_bits": 8}`

// whistleModelPath is the pinned checkpoint as laid out by `whistle download`.
const whistleModelPath = "../../models/whistle/whistle.safetensors"

func wantWhistleConfig() Config {
	return Config{
		Text: 8192, Dim: 512, KVHeads: 2, QKDim: 48, VDim: 64,
		EncLayers: 8, DecLayers: 8, ConvKernel: 9, Taps: 3, Lanes: 4, EncLanes: 4,
		Sites: []int{3, 7}, Orders: []int{2, 3}, Slots: 18432, Hashes: 2,
		Stem: 128, StemStages: 3, Rope: 100000,
		Mels: 80, Rate: 16000, FFT: 512, Win: 400, Hop: 160,
		MaxTokens: 320, Pad: 0, EOS: 1, BOS: 2, DType: "bfloat16",
		FreqMasks: 2, FreqWidth: 17, TimeMasks: 10, TimeWidth: 0.05, Dither: 1e-5,
		PerAugmentProb: 0.1, Speed: []float64{0.9, 1.1},
	}
}

func wantWhistleRun() Run {
	return Run{
		Stage:          "qapt",
		BaseCheckpoint: "avg230-250.pkl",
		WeightBits:     "embedding=4,stack/mhc=4,encoder/mhc=4,default=2",
		KVBits:         8,
		ActBits:        8,
	}
}

func storeWithMetadata(t *testing.T, meta map[string]string) *Store {
	t.Helper()

	blob, err := EncodeTensorsWithMetadata([]Tensor{{Name: "x", Shape: []int64{1}, Data: []float32{1}}}, meta)
	if err != nil {
		t.Fatalf("EncodeTensorsWithMetadata: %v", err)
	}

	store, err := OpenStoreFromBytes(blob, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStoreFromBytes: %v", err)
	}

	t.Cleanup(store.Close)

	return store
}

func TestStore_MetadataRoundTrip(t *testing.T) {
	store := storeWithMetadata(t, map[string]string{"format": "pt", "step": "7"})

	meta := store.Metadata()
	if len(meta) != 2 || meta["format"] != "pt" || meta["step"] != "7" {
		t.Fatalf("Metadata() = %v", meta)
	}

	meta["format"] = "changed"

	if store.Metadata()["format"] != "pt" {
		t.Error("Metadata() returned the store's own map")
	}

	if names := store.Names(); len(names) != 1 || names[0] != "x" {
		t.Errorf("Names() = %v, want [x]", names)
	}
}

func TestStore_MetadataAbsent(t *testing.T) {
	store := storeWithMetadata(t, nil)

	if meta := store.Metadata(); len(meta) != 0 {
		t.Errorf("Metadata() = %v, want empty", meta)
	}

	_, err := store.Config()
	if err == nil || !strings.Contains(err.Error(), `"config"`) {
		t.Errorf("Config() err = %v, want one naming the missing entry", err)
	}
}

func TestStore_MetadataNonStringRejected(t *testing.T) {
	header := `{"__metadata__":{"step":25000},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	data := make([]byte, 8+len(header)+4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)

	_, err := OpenStoreFromBytes(data, StoreOptions{})
	if err == nil || !strings.Contains(err.Error(), "__metadata__") {
		t.Fatalf("err = %v, want a __metadata__ error", err)
	}
}

func TestEncodeTensorsWithMetadata_ReservedName(t *testing.T) {
	_, err := EncodeTensorsWithMetadata([]Tensor{{Name: "__metadata__", Shape: []int64{1}, Data: []float32{1}}}, nil)
	if err == nil {
		t.Fatal("a tensor named __metadata__ was accepted")
	}
}

func TestStore_Config(t *testing.T) {
	store := storeWithMetadata(t, map[string]string{"config": whistleConfigJSON})

	got, err := store.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}

	if want := wantWhistleConfig(); !reflect.DeepEqual(got, want) {
		t.Errorf("Config() =\n%+v\nwant\n%+v", got, want)
	}
}

func TestStore_ConfigErrors(t *testing.T) {
	for _, tc := range []struct {
		name, config, want string
	}{
		{name: "malformed", config: `{"dim":`, want: "config"},
		{name: "unknown key", config: strings.Replace(whistleConfigJSON, `"dim"`, `"width": 1, "dim"`, 1), want: "width"},
		{name: "missing key", config: strings.Replace(whistleConfigJSON, `"hop": 160, `, "", 1), want: "hop"},
		{name: "wrong type", config: strings.Replace(whistleConfigJSON, `"dim": 512`, `"dim": "512"`, 1), want: "dim"},
	} {
		store := storeWithMetadata(t, map[string]string{"config": tc.config})

		_, err := store.Config()
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: err = %v, want one mentioning %q", tc.name, err, tc.want)
		}
	}
}

func TestStore_RunAndStep(t *testing.T) {
	store := storeWithMetadata(t, map[string]string{"run": whistleRunJSON, "step": "25000"})

	run, err := store.Run()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if run != wantWhistleRun() {
		t.Errorf("Run() = %+v, want %+v", run, wantWhistleRun())
	}

	step, err := store.Step()
	if err != nil || step != 25000 {
		t.Errorf("Step() = %d, %v; want 25000", step, err)
	}
}

func TestStore_RunAndStepErrors(t *testing.T) {
	store := storeWithMetadata(t, map[string]string{
		"run":  strings.Replace(whistleRunJSON, `"kv_bits": 8, `, "", 1),
		"step": "25k",
	})

	_, err := store.Run()
	if err == nil || !strings.Contains(err.Error(), "kv_bits") {
		t.Errorf("Run() err = %v, want one naming kv_bits", err)
	}

	_, err = store.Step()
	if err == nil || !strings.Contains(err.Error(), "25k") {
		t.Errorf("Step() err = %v, want one quoting the bad value", err)
	}

	empty := storeWithMetadata(t, nil)

	_, err = empty.Run()
	if err == nil || !strings.Contains(err.Error(), `"run"`) {
		t.Errorf("Run() on no metadata err = %v", err)
	}

	_, err = empty.Step()
	if err == nil || !strings.Contains(err.Error(), `"step"`) {
		t.Errorf("Step() on no metadata err = %v", err)
	}
}

// TestWhistleCheckpoint pins the downloaded whistle.safetensors: tensor count,
// dtypes, parameter count, a known shape and the metadata against PLAN.md's
// model reference card.
func TestWhistleCheckpoint(t *testing.T) {
	_, err := os.Stat(whistleModelPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", whistleModelPath)
	}

	store, err := OpenStore(whistleModelPath, StoreOptions{})
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer store.Close()

	const (
		wantTensors = 116
		wantParams  = 55_151_433 // HF model card
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

		if tensor.DType != dtypeF32 {
			t.Errorf("%s is %s, want F32", name, tensor.DType)
		}

		params += len(tensor.Data)
	}

	if params != wantParams {
		t.Errorf("%d parameters, want %d", params, wantParams)
	}

	_, err = store.TensorWithShape("embedding/embedding", []int64{8199, 512})
	if err != nil {
		t.Error(err)
	}

	cfg, err := store.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}

	if want := wantWhistleConfig(); !reflect.DeepEqual(cfg, want) {
		t.Errorf("Config() =\n%+v\nwant\n%+v", cfg, want)
	}

	run, err := store.Run()
	if err != nil || run != wantWhistleRun() {
		t.Errorf("Run() = %+v, %v; want %+v", run, err, wantWhistleRun())
	}

	step, err := store.Step()
	if err != nil || step != 25000 {
		t.Errorf("Step() = %d, %v; want 25000", step, err)
	}
}
