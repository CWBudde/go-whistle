package weights

import (
	"errors"
	"fmt"
	"math/bits"
	"slices"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// languageTokens is the number of language tokens after the text
// vocabulary, <|en|> to <|pl|> (docs/FINDINGS.md F-003, F-005).
const languageTokens = 7

// manifestLen is the audio manifest's length: magic, version and the eleven
// values of Config.manifest (docs/FINDINGS.md F-006).
const manifestLen = 13

// manifestName is the audio manifest record.
const manifestName = "audio.manifest"

// Config is the inference geometry, as both formats state it: the .cact
// header and audio manifest, or the safetensors config (docs/FINDINGS.md
// F-003, F-007). Training-only keys, the token ids (which the tokenizer
// holds) and the run are left out.
type Config struct {
	Vocab      int     // tokenizer pieces: Text plus the language tokens
	Text       int     // text vocabulary
	Dim        int     // model width
	HadaN      int     // HadamardMLP width: Dim rounded up to a power of two
	Heads      int     // decoder self-attention query heads
	KVHeads    int     // decoder self-attention KV heads (GQA)
	QKDim      int     // query/key head width
	VDim       int     // value head width
	DecLayers  int     // decoder blocks
	EncLayers  int     // encoder blocks
	Lanes      int     // decoder mHC lanes
	EncLanes   int     // encoder mHC lanes
	Taps       int     // causal conv taps on decoder Q/K/V
	ConvKernel int     // encoder depthwise conv kernel
	Sites      []int   // decoder layers carrying engram memory
	Orders     []int   // engram n-gram orders
	Slots      int     // engram table rows
	Tables     int     // engram tables: len(Orders) x hash heads
	Stem       int     // conv stem channels
	StemStages int     // stride-2 stem stages
	Rope       float64 // RoPE theta
	Mels       int     // mel bins
	Rate       int     // sample rate in Hz
	FFT        int     // STFT size
	Win        int     // STFT window in samples
	Hop        int     // STFT hop in samples
	MaxTokens  int     // decode limit
}

func (c *Config) clone() Config {
	out := *c
	out.Sites, out.Orders = slices.Clone(c.Sites), slices.Clone(c.Orders)

	return out
}

// manifestFields are the audio manifest's values after magic and version,
// in order (F-003).
func (c *Config) manifestFields() []*int {
	return []*int{
		&c.Text, &c.EncLayers, &c.EncLanes, &c.ConvKernel, &c.Stem, &c.StemStages,
		&c.Mels, &c.Rate, &c.FFT, &c.Win, &c.Hop,
	}
}

// manifest returns the audio manifest's values after magic and version.
func (c *Config) manifest() []int {
	fields := c.manifestFields()

	out := make([]int, len(fields))
	for k, p := range fields {
		out[k] = *p
	}

	return out
}

// manifestRecord is the audio manifest record for c, as .cact stores it.
func (c *Config) manifestRecord() []float32 {
	out := make([]float32, 0, manifestLen)
	out = append(out, cact.AudioManifestMagic, cact.AudioManifestVersion)

	for _, v := range c.manifest() {
		out = append(out, float32(v))
	}

	return out
}

// setManifest fills the manifest's fields from a .cact audio manifest. Like
// the engine, it requires the magic and version 1 (F-006).
func (c *Config) setManifest(v []float32) error {
	if len(v) != manifestLen {
		return fmt.Errorf("weights: audio manifest has %d values, want %d", len(v), manifestLen)
	}

	if v[0] != cact.AudioManifestMagic {
		return fmt.Errorf("weights: audio manifest magic %v, want %d", v[0], cact.AudioManifestMagic)
	}

	if v[1] != cact.AudioManifestVersion {
		return fmt.Errorf("weights: audio manifest version %v, want %d", v[1], cact.AudioManifestVersion)
	}

	for k, p := range c.manifestFields() {
		x := v[2+k]
		if x < 0 || x != float32(int(x)) {
			return fmt.Errorf("weights: audio manifest value %d = %v is not a count", 2+k, x)
		}

		*p = int(x)
	}

	return nil
}

// cactConfig reads the decoder geometry from the header and the encoder and
// front end from the audio manifest (F-003).
func cactConfig(f *cact.File, entries []cact.Entry) (Config, error) {
	i := slices.IndexFunc(entries, func(e cact.Entry) bool { return e.Name == manifestName })
	if i < 0 {
		return Config{}, errors.New("weights: the name map has no audio.manifest")
	}

	man, err := f.FP32(i)
	if err != nil {
		return Config{}, fmt.Errorf("weights: audio.manifest: %w", err)
	}

	h := f.Header()
	cfg := Config{
		Vocab: int(h.Vocab), Dim: int(h.DModel), HadaN: int(h.HadaN),
		Heads: int(h.NumHeads), KVHeads: int(h.NumKVHeads), QKDim: int(h.QKHeadDim), VDim: int(h.VHeadDim),
		DecLayers: int(h.NumLayers), Lanes: int(h.MHCLanes), Taps: int(h.QKVConvTaps),
		Sites: h.Sites(), Orders: h.Orders(), Slots: int(h.EngramSlots), Tables: int(h.NumEngramTables),
		Rope: float64(h.RopeTheta), MaxTokens: int(h.MaxSeqLen),
	}

	return cfg, cfg.setManifest(man)
}

// safetensorsConfig maps the checkpoint config (F-003). The config does not
// state the query heads; they are the rows of layer00.q_proj over QKDim,
// which newSafetensors has checked against the source (F-007).
func safetensorsConfig(c safetensors.Config, entries []cact.Entry) (Config, error) {
	cfg := Config{
		Vocab: c.Text + languageTokens, Text: c.Text, Dim: c.Dim, HadaN: nextPow2(c.Dim),
		KVHeads: c.KVHeads, QKDim: c.QKDim, VDim: c.VDim, DecLayers: c.DecLayers, EncLayers: c.EncLayers,
		Lanes: c.Lanes, EncLanes: c.EncLanes, Taps: c.Taps, ConvKernel: c.ConvKernel,
		Sites: slices.Clone(c.Sites), Orders: slices.Clone(c.Orders), Slots: c.Slots,
		Tables: len(c.Orders) * c.Hashes, Stem: c.Stem, StemStages: c.StemStages, Rope: c.Rope,
		Mels: c.Mels, Rate: c.Rate, FFT: c.FFT, Win: c.Win, Hop: c.Hop, MaxTokens: c.MaxTokens,
	}

	i := slices.IndexFunc(entries, func(e cact.Entry) bool { return e.Name == "layer00.q_proj" })
	if i < 0 || len(entries[i].Shape) != 2 || cfg.QKDim <= 0 || entries[i].Shape[0]%cfg.QKDim != 0 {
		return Config{}, fmt.Errorf("weights: no head count: needs layer00.q_proj with rows a multiple of qk_dim %d",
			cfg.QKDim)
	}

	cfg.Heads = entries[i].Shape[0] / cfg.QKDim

	return cfg, nil
}

// nextPow2 is architecture.py's hada_n: n rounded up to a power of two.
func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}

	return 1 << bits.Len(uint(n-1))
}
