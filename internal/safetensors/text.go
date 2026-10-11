package safetensors

import (
	"encoding/json"
	"fmt"
)

// TextConfig is the Needle 3 text model config (needle3.safetensors @
// 93e88ede, PLAN.md Phase 11), stored as JSON in the checkpoint's
// __metadata__ "config" entry. Unlike Whistle's it uses upstream's
// TransformerConfig field names (needle/model/architecture.py @ af2654dffa)
// and leaves fields at their dataclass defaults out; the methods below supply
// those defaults as TransformerConfig.from_saved does. Decoding is strict:
// every key is required and unknown keys are rejected.
type TextConfig struct {
	Vocab           int    `json:"vocab_size"`
	Dim             int    `json:"d_model"`
	Heads           int    `json:"num_heads"`
	KVHeads         int    `json:"num_kv_heads"`
	Layers          int    `json:"num_layers"`
	QKDim           int    `json:"qk_head_dim"`
	VDim            int    `json:"v_head_dim"`
	MaxSeqLen       int    `json:"max_seq_len"`
	DType           string `json:"dtype"` // training compute dtype; the checkpoint itself is F16
	Slots           int    `json:"engram_slots"`
	Sites           []int  `json:"engram_layers"`  // layers carrying engram memory
	SlidingWindow   int    `json:"sliding_window"` // local attention width of the non-global layers
	GlobalLayers    []int  `json:"global_layers"`  // full-attention layers
	KVWindow        int    `json:"kv_window"`      // the engine's KV window (export.py kv_window)
	LadderDepths    []int  `json:"ladder_depths"`
	LadderSample    bool   `json:"ladder_sample"`
	LadderWidths    []int  `json:"ladder_widths"`
	Lanes           int    `json:"mhc_lanes"`
	Taps            int    `json:"qkv_conv_taps"`
	OutVocab        int    `json:"out_vocab"` // tied head rows, 0 = the full vocab
	EngramSeedHeads int    `json:"engram_seed_heads"`
	KVBits          int    `json:"kv_bits"`
	ActBits         int    `json:"act_bits"`
	WeightBits      string `json:"weight_bits"`
	ConfProbes      int    `json:"confidence_probes"`
	ConfQueries     int    `json:"confidence_queries"`

	// Image-code fields of the multimodal branch; all zero for the text
	// model. Decoded so the config round-trips completely.
	ImageCodeOffset int `json:"image_code_offset"`
	ImageSemCodes   int `json:"image_sem_codes"`
	ImageSemDim     int `json:"image_sem_dim"`
	ImageTexBooks   int `json:"image_tex_books"`
	ImageTexCodes   int `json:"image_tex_codes"`
	ImageTexStages  int `json:"image_tex_stages"`
	ImageTexDim     int `json:"image_tex_dim"`
}

// TransformerConfig defaults for keys the text config leaves out
// (architecture.py @ af2654dffa).
const (
	textDefaultRope  = 100000.0
	textEngramSubDim = 128 // architecture.py ENGRAM_SUB_DIM

	// maxTextDim bounds d_model: far above any Needle model (768), far below
	// where HadaN's power of two overflows.
	maxTextDim = 1 << 20
)

// Orders returns the engram n-gram orders, TransformerConfig's default (2, 3):
// the text config does not store them.
func (c TextConfig) Orders() []int { return []int{2, 3} }

// Rope returns the RoPE theta, TransformerConfig's default 1e5: the text
// config does not store it.
func (c TextConfig) Rope() float64 { return textDefaultRope }

// Hashes returns the engram hash heads per order, engram_geometry's
// max(1, d_model / (len(orders) * 128)) since engram_heads is unset.
func (c TextConfig) Hashes() int {
	return max(1, c.Dim/(len(c.Orders())*textEngramSubDim))
}

// Tables returns the engram tables per site: orders x hash heads.
func (c TextConfig) Tables() int { return len(c.Orders()) * c.Hashes() }

// SubDim returns the engram table width, d_model / tables.
func (c TextConfig) SubDim() int { return c.Dim / c.Tables() }

// HadaN returns the HadamardMLP width: d_model rounded up to a power of two.
func (c TextConfig) HadaN() int {
	n := 1
	for n < c.Dim {
		n <<= 1
	}

	return n
}

// TextRun describes how the text checkpoint was trained, from the
// __metadata__ "run" entry. The quantisation plan and the KV and activation
// widths live in TextConfig here, not in the run as for Whistle. Compiled
// and MergedHeads are kept raw.
type TextRun struct {
	Stage          string          `json:"stage"`
	BaseCheckpoint string          `json:"base_checkpoint"`
	Dataset        string          `json:"dataset"`
	MaxSteps       int             `json:"max_steps"`
	Seed           int             `json:"seed"`
	BitsStart      float64         `json:"bits_start"`
	BitsEnd        float64         `json:"bits_end"`
	AnnealFrac     float64         `json:"anneal_frac"`
	PhaseBFrac     float64         `json:"phase_b_frac"`
	PhaseBBits     string          `json:"phase_b_bits"`
	PhaseCFrac     float64         `json:"phase_c_frac"`
	KDWeight       float64         `json:"kd_weight"`
	AB             bool            `json:"ab"`
	ABSigns        bool            `json:"ab_signs"`
	ActPhaseA      bool            `json:"act_phase_a"`
	LadderDepths   []int           `json:"ladder_depths"`
	LadderWidths   []int           `json:"ladder_widths"`
	TokenizerHash  string          `json:"tokenizer_hash"` // MD5 of the training tokenizer.model
	Compiled       json.RawMessage `json:"compiled"`
	MergedHeads    json.RawMessage `json:"merged_heads"`
}

// TextConfig decodes the __metadata__ "config" entry of a Needle 3 text
// checkpoint.
func (s *Store) TextConfig() (TextConfig, error) {
	var cfg TextConfig

	err := s.decodeMetadataJSON(metaConfig, &cfg)
	if err != nil {
		return cfg, err
	}

	// HadaN rounds d_model up to a power of two; a width past maxTextDim
	// would overflow it, so a malformed config fails here instead.
	if cfg.Dim < 1 || cfg.Dim > maxTextDim {
		return cfg, fmt.Errorf("safetensors: text config d_model %d outside [1, %d]", cfg.Dim, maxTextDim)
	}

	return cfg, nil
}

// TextRun decodes the __metadata__ "run" entry of a Needle 3 text
// checkpoint.
func (s *Store) TextRun() (TextRun, error) {
	var run TextRun

	err := s.decodeMetadataJSON(metaRun, &run)

	return run, err
}
