package safetensors

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// Metadata keys the Whistle checkpoint stores next to its tensors.
const (
	metaConfig = "config"
	metaRun    = "run"
	metaStep   = "step"
)

// Config is the Whistle model config, stored as JSON in the checkpoint's
// __metadata__ "config" entry. Field names follow upstream; every key is
// required and unknown keys are rejected, so an upstream schema change fails
// loudly instead of decoding to zeros.
type Config struct {
	Text       int     `json:"text"`        // text vocabulary; the 7 language tokens follow it
	Dim        int     `json:"dim"`         // model width
	KVHeads    int     `json:"kv_heads"`    // decoder self-attention KV heads (GQA)
	QKDim      int     `json:"qk_dim"`      // query/key head width
	VDim       int     `json:"v_dim"`       // value head width
	EncLayers  int     `json:"enc_layers"`  // encoder blocks
	DecLayers  int     `json:"dec_layers"`  // decoder blocks
	ConvKernel int     `json:"conv_kernel"` // encoder depthwise conv kernel
	Taps       int     `json:"taps"`        // causal conv taps on decoder Q/K/V
	Lanes      int     `json:"lanes"`       // decoder mHC lanes
	EncLanes   int     `json:"enc_lanes"`   // encoder mHC lanes
	Sites      []int   `json:"sites"`       // decoder layers carrying engram memory
	Orders     []int   `json:"orders"`      // engram n-gram orders
	Slots      int     `json:"slots"`       // engram table rows
	Hashes     int     `json:"hashes"`      // engram hash heads per order
	Stem       int     `json:"stem"`        // conv stem channels
	StemStages int     `json:"stem_stages"` // stride-2 stem stages
	Rope       float64 `json:"rope"`        // RoPE theta
	Mels       int     `json:"mels"`        // mel bins
	Rate       int     `json:"rate"`        // sample rate in Hz
	FFT        int     `json:"fft"`         // STFT size
	Win        int     `json:"win"`         // STFT window in samples
	Hop        int     `json:"hop"`         // STFT hop in samples
	MaxTokens  int     `json:"max_tokens"`  // decode limit
	Pad        int     `json:"pad"`         // token ids
	EOS        int     `json:"eos"`
	BOS        int     `json:"bos"`
	DType      string  `json:"dtype"` // training compute dtype; the checkpoint itself is F32

	// Training-time augmentation. Inference ignores these; they are decoded
	// so the config round-trips completely.
	FreqMasks      int       `json:"freq_masks"`
	FreqWidth      int       `json:"freq_width"`
	TimeMasks      int       `json:"time_masks"`
	TimeWidth      float64   `json:"time_width"`
	Dither         float64   `json:"dither"`
	PerAugmentProb float64   `json:"per_augment_prob"`
	Speed          []float64 `json:"speed"`
}

// Run describes how the checkpoint was trained, from the __metadata__ "run"
// entry. WeightBits is the per-module quantisation plan, e.g.
// "embedding=4,stack/mhc=4,encoder/mhc=4,default=2".
type Run struct {
	Stage          string `json:"stage"`
	BaseCheckpoint string `json:"base_checkpoint"`
	WeightBits     string `json:"weight_bits"`
	KVBits         int    `json:"kv_bits"`
	ActBits        int    `json:"act_bits"`
}

// Config decodes the __metadata__ "config" entry.
func (s *Store) Config() (Config, error) {
	var cfg Config

	err := s.decodeMetadataJSON(metaConfig, &cfg)

	return cfg, err
}

// Run decodes the __metadata__ "run" entry.
func (s *Store) Run() (Run, error) {
	var run Run

	err := s.decodeMetadataJSON(metaRun, &run)

	return run, err
}

// Step returns the training step from the __metadata__ "step" entry.
func (s *Store) Step() (int, error) {
	raw, err := s.metadataEntry(metaStep)
	if err != nil {
		return 0, err
	}

	step, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("safetensors: metadata %q = %q is not an integer: %w", metaStep, raw, err)
	}

	return step, nil
}

func (s *Store) metadataEntry(key string) (string, error) {
	raw, ok := s.metadata[key]
	if !ok {
		return "", fmt.Errorf("safetensors: %s has no %q entry", metadataKey, key)
	}

	return raw, nil
}

// decodeMetadataJSON strictly decodes the JSON metadata entry key into the
// struct pointed to by dst: unknown keys and missing keys are errors.
func (s *Store) decodeMetadataJSON(key string, dst any) error {
	raw, err := s.metadataEntry(key)
	if err != nil {
		return err
	}

	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()

	err = dec.Decode(dst)
	if err != nil {
		return fmt.Errorf("safetensors: metadata %q: %w", key, err)
	}

	missing, err := missingJSONKeys([]byte(raw), reflect.TypeOf(dst).Elem())
	if err != nil {
		return fmt.Errorf("safetensors: metadata %q: %w", key, err)
	}

	if len(missing) > 0 {
		return fmt.Errorf("safetensors: metadata %q lacks %s", key, strings.Join(missing, ", "))
	}

	return nil
}

// missingJSONKeys lists the json tags of struct type t that the JSON object
// raw does not contain.
func missingJSONKeys(raw []byte, t reflect.Type) ([]string, error) {
	var present map[string]json.RawMessage

	err := json.NewDecoder(bytes.NewReader(raw)).Decode(&present)
	if err != nil {
		return nil, fmt.Errorf("decode keys: %w", err)
	}

	var missing []string

	for field := range t.Fields() {
		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		if _, ok := present[name]; !ok {
			missing = append(missing, name)
		}
	}

	return missing, nil
}
