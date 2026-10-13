package text

import (
	"encoding/binary"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Source serves the text model's tensors by their cact.TextNames names
// from either format, both in the .cact layout, so that internal/needle's
// loaders read it as they read Whistle's weights.Source. Tensors are
// decoded, and CQ tensors dequantised, on first use and cached
// (docs/FINDINGS.md F-045).
//
// Opened with OpenOptions.Packed, a .cact source is also a weights.Packer:
// Matrix serves the CQ records as cq.Matrix values that multiply straight
// from the stored indices, and the needle loaders then take those instead
// of a float32 copy (docs/FINDINGS.md F-059).
type Source struct {
	cfg   Config
	names []cact.Entry
	index map[string]int

	file   *cact.File         // .cact backend
	store  *safetensors.Store // safetensors backend
	packed bool               // Matrix serves the CQ records (OpenOptions.Packed)
	lut    bool               // their products run the lut kernel (OpenOptions.LUT)

	mu     sync.Mutex
	cache  map[string][]float32
	mats   map[string]*cq.Matrix // the packed matrices, as cache (F-059)
	closed bool
}

var (
	_ weights.Source = (*Source)(nil)
	_ weights.Packer = (*Source)(nil)
)

// OpenOptions configure OpenWith and OpenCactWith; the zero value is Open's
// float32 source.
type OpenOptions struct {
	// Packed makes a .cact source serve its CQ records packed (Matrix), so
	// that Load runs the model on them without a float32 copy, as
	// whistle.Load does for Whistle (docs/FINDINGS.md F-037, F-059). The
	// matrices alias the file data. The safetensors backend has no packed
	// form and ignores it.
	//
	// The products run internal/cq's nibble kernels (cq.Matrix.Exact),
	// which sum the codebook levels in float32: the packed model then gives
	// the dequantised float model's logits and confidence up to the float32
	// summation order, and holds no lut copy (F-059).
	Packed bool
	// LUT runs the packed products on the lut kernel instead, as Whistle
	// does (F-034): faster than float32 where its assembly runs, but its
	// tables round to 14 bits, which moves the confidence logit by up to
	// 3.5e-4, and the blocked copies add 15 MiB (F-059). It needs Packed.
	LUT bool
}

// Open reads a text .cact (recognised by its tag) or a text safetensors
// checkpoint into memory and serves it in float32.
func Open(path string) (*Source, error) {
	return OpenWith(path, OpenOptions{})
}

// OpenWith is Open with options: OpenWith(path, OpenOptions{Packed: true})
// serves a .cact's CQ records packed.
func OpenWith(path string, opts OpenOptions) (*Source, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("text: %w", err)
	}

	if len(data) >= 4 && binary.LittleEndian.Uint32(data) == cact.Tag {
		return OpenCactWith(data, opts)
	}

	return OpenSafetensors(data)
}

// OpenCact serves a text .cact held in data in float32.
func OpenCact(data []byte) (*Source, error) {
	return OpenCactWith(data, OpenOptions{})
}

// OpenCactWith serves a text .cact held in data with options. Packed
// matrices alias data, which must not change while the source is in use.
func OpenCactWith(data []byte, opts OpenOptions) (*Source, error) {
	f, err := cact.Parse(data)
	if err != nil {
		return nil, err
	}

	names, err := cact.TextNames(f)
	if err != nil {
		return nil, err
	}

	s := newSource(ConfigFromHeader(f.Header()), names, f, nil)
	s.packed, s.lut = opts.Packed, opts.LUT

	return s, nil
}

// OpenSafetensors serves a text safetensors checkpoint held in data, under
// the names its .cact export would have: the trunk, and the probe heads
// the checkpoint holds.
func OpenSafetensors(data []byte) (*Source, error) {
	store, err := safetensors.OpenStoreFromBytes(data, safetensors.StoreOptions{})
	if err != nil {
		return nil, err
	}

	tc, err := store.TextConfig()
	if err != nil {
		return nil, err
	}

	cfg := ConfigFromSafetensors(tc)

	h, err := cfg.Header()
	if err != nil {
		return nil, err
	}

	var heads []cact.HeadShape

	for _, code := range []int{cact.HeadEmbedding, cact.HeadConfidence, cact.HeadRouter} {
		hs, ok, err := storeHead(store, code, cfg)
		if err != nil {
			return nil, err
		}

		if ok {
			heads = append(heads, hs)
		}
	}

	names, err := cact.TextNamesFor(h, heads)
	if err != nil {
		return nil, err
	}

	return newSource(cfg, names, nil, store), nil
}

// headKey are the safetensors module names of the probe heads by code.
var headKey = map[int]string{
	cact.HeadEmbedding:  "embedding_head",
	cact.HeadConfidence: "confidence_head",
	cact.HeadRouter:     "router_head",
}

// storeHead reads a probe head's widths from the checkpoint's shapes:
// probes (L+1, k, d), query (q, d), proj kernel (q·d, out).
func storeHead(store *safetensors.Store, code int, cfg Config) (cact.HeadShape, bool, error) {
	key := headKey[code]
	if !store.Has(key + "/probes") {
		return cact.HeadShape{}, false, nil
	}

	var shapes [3][]int64

	for k, name := range []string{"/probes", "/query", "/proj/kernel"} {
		shape, err := store.Shape(key + name)
		if err != nil {
			return cact.HeadShape{}, false, err
		}

		shapes[k] = shape
	}

	probes, query, proj := shapes[0], shapes[1], shapes[2]
	if len(probes) != 3 || len(query) != 2 || len(proj) != 2 || int(probes[0]) != cfg.Layers+1 {
		return cact.HeadShape{}, false, fmt.Errorf("text: %s shapes %v %v %v", key, probes, query, proj)
	}

	return cact.HeadShape{Code: code, Probes: int(probes[1]), Queries: int(query[0]), Out: int(proj[1])}, true, nil
}

func newSource(cfg Config, names []cact.Entry, f *cact.File, store *safetensors.Store) *Source {
	s := &Source{
		cfg: cfg, names: names, index: make(map[string]int, len(names)), file: f, store: store,
		cache: map[string][]float32{}, mats: map[string]*cq.Matrix{},
	}

	for i, e := range names {
		s.index[e.Name] = i
	}

	return s
}

// TextConfig is the text geometry.
func (s *Source) TextConfig() Config {
	c := s.cfg
	c.GlobalLayers, c.Sites, c.Orders = slices.Clone(c.GlobalLayers), slices.Clone(c.Sites), slices.Clone(c.Orders)

	return c
}

// Names lists every name in .cact index order.
func (s *Source) Names() []string {
	out := make([]string, len(s.names))
	for i, e := range s.names {
		out[i] = e.Name
	}

	return out
}

// Has reports whether name exists.
func (s *Source) Has(name string) bool {
	_, ok := s.index[name]

	return ok
}

// HasHead reports whether the model carries the probe head with the given
// heads.manifest code (cact.HeadEmbedding, HeadConfidence, HeadRouter).
// The engine retrieves tools per turn only with an embedding head
// (docs/FINDINGS.md F-049).
func (s *Source) HasHead(code int) bool {
	key, ok := headKey[code]

	return ok && s.Has(key+".probes")
}

// Shape is a name's .cact shape, nil for the tokenizer.
func (s *Source) Shape(name string) ([]int, error) {
	i, ok := s.index[name]
	if !ok {
		return nil, fmt.Errorf("text: %s: %w", name, weights.ErrNotFound)
	}

	return slices.Clone(s.names[i].Shape), nil
}

// Tensor returns a tensor's values in the .cact layout and its shape. The
// values are decoded on the first call and cached; callers must not modify
// them.
func (s *Source) Tensor(name string) ([]float32, []int, error) {
	i, ok := s.index[name]
	if !ok {
		return nil, nil, fmt.Errorf("text: %s: %w", name, weights.ErrNotFound)
	}

	e := s.names[i]

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, nil, weights.ErrClosed
	}

	if v, ok := s.cache[name]; ok {
		return v, slices.Clone(e.Shape), nil
	}

	var (
		v   []float32
		err error
	)

	if s.file != nil {
		v, err = s.decodeCact(i)
	} else {
		v, err = s.decodeSafetensors(e)
	}

	if err != nil {
		return nil, nil, fmt.Errorf("text: %s: %w", name, err)
	}

	s.cache[name] = v

	return v, slices.Clone(e.Shape), nil
}

// Matrix serves CQ record name packed when the source was opened with
// OpenOptions.Packed (weights.Packer, docs/FINDINGS.md F-059):
// cq.NewGather on the record as stored, an Exact view unless
// OpenOptions.LUT, built on the first call and cached until Release or
// Close, with no float32 copy. Its indices alias the file data. Other records, CQ widths internal/cq has no kernel for (it has 2
// and 4 bits, all needle3.cact uses), every record in float mode and the
// safetensors backend give nil and false.
func (s *Source) Matrix(name string) (*cq.Matrix, bool, error) {
	i, ok := s.index[name]
	if !ok {
		return nil, false, fmt.Errorf("text: %s: %w", name, weights.ErrNotFound)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, false, fmt.Errorf("text: %s: %w", name, weights.ErrClosed)
	}

	if !s.packed || s.file == nil {
		return nil, false, nil
	}

	if m, ok := s.mats[name]; ok {
		return m, true, nil
	}

	rec, err := s.file.Record(i)
	if err != nil || rec.DType != cact.CQ || rec.Bits != 2 && rec.Bits != 4 {
		return nil, false, err
	}

	p, err := s.file.CQPacked(i)
	if err != nil {
		return nil, false, fmt.Errorf("text: %s: %w", name, err)
	}

	m, err := cq.NewGather(p)
	if err != nil {
		return nil, false, fmt.Errorf("text: %s: %w", name, err)
	}

	if !s.lut {
		m = m.Exact()
	}

	s.mats[name] = m

	return m, true, nil
}

// Raw returns a copy of the tokenizer blob; the safetensors checkpoint has
// none.
func (s *Source) Raw(name string) ([]byte, error) {
	i, ok := s.index[name]
	if !ok {
		return nil, fmt.Errorf("text: %s: %w", name, weights.ErrNotFound)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, weights.ErrClosed
	}

	if s.file == nil {
		return nil, fmt.Errorf("text: %s: %w", name, weights.ErrUnavailable)
	}

	b, err := s.file.Raw(i)
	if err != nil {
		return nil, err
	}

	return slices.Clone(b), nil
}

// Config is the geometry in weights.Config's fields, for code written
// against weights.Source; Text and Vocab are the text vocabulary, DecLayers
// the blocks. TextConfig has the rest.
func (s *Source) Config() weights.Config {
	c := s.cfg

	return weights.Config{
		Vocab: c.Vocab, Text: c.Vocab, Dim: c.Dim, HadaN: c.HadaN, Heads: c.Heads, KVHeads: c.KVHeads,
		QKDim: c.QKDim, VDim: c.VDim, DecLayers: c.Layers, Lanes: c.Lanes, Taps: c.Taps,
		Sites: slices.Clone(c.Sites), Orders: slices.Clone(c.Orders), Slots: c.Slots, Tables: c.Tables,
		Rope: c.Rope, MaxTokens: c.MaxSeqLen,
	}
}

// Release drops a name's cached values and packed matrix.
func (s *Source) Release(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.cache, name)
	delete(s.mats, name)
}

// Close drops the caches; Tensor, Raw and Matrix then fail with
// weights.ErrClosed. Closing twice is a no-op.
func (s *Source) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed, s.cache, s.mats = true, nil, nil

	if s.store != nil {
		s.store.Close()
	}

	return nil
}

func (s *Source) decodeCact(i int) ([]float32, error) {
	rec, err := s.file.Record(i)
	if err != nil {
		return nil, err
	}

	switch rec.DType {
	case cact.CQ:
		return s.file.CQ(i)
	case cact.FP16, cact.FP32:
		return s.file.Float32(i)
	default:
		return nil, fmt.Errorf("%w: a %s record", weights.ErrUnavailable, rec.DType)
	}
}

func (s *Source) decodeSafetensors(e cact.Entry) ([]float32, error) {
	if e.Transform == cact.Derived {
		return s.derived(e)
	}

	t, err := s.store.Tensor(e.Source)
	if err != nil {
		return nil, err
	}

	shape := make([]int, len(t.Shape))
	for k, d := range t.Shape {
		shape[k] = int(d)
	}

	return e.Apply(t.Data, shape)
}

// derived recomputes the records export.py derives: the Hadamard
// permutations, the heads.manifest and the embedding head's zero bias. The
// tokenizer is not in the checkpoint.
func (s *Source) derived(e cact.Entry) ([]float32, error) {
	switch {
	case e.Name == "hada_p1" || e.Name == "hada_p2":
		seed, _ := cact.HadaPermSeed(e.Name)
		p := cact.NumpyPermutation(seed, s.cfg.HadaN)

		out := make([]float32, len(p))
		for k, v := range p {
			out[k] = float32(v)
		}

		return out, nil
	case e.Name == "heads.manifest":
		var codes []float32

		for _, code := range []int{cact.HeadEmbedding, cact.HeadConfidence, cact.HeadRouter} {
			if s.Has(headKey[code] + ".probes") {
				codes = append(codes, float32(code))
			}
		}

		return codes, nil
	case strings.HasSuffix(e.Name, ".bias"):
		return make([]float32, e.Shape[0]), nil
	default:
		return nil, weights.ErrUnavailable
	}
}
