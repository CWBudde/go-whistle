package weights

import (
	"fmt"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/safetensors"
)

// Safetensors serves the float32 checkpoint under the .cact names: each
// source is sliced and laid out by cact.Entry.Apply, and the derived
// records are recomputed from the config. The tokenizer is not in the
// checkpoint, so Raw reports it as ErrUnavailable (docs/FINDINGS.md F-007).
type Safetensors struct {
	*core
}

// OpenSafetensors opens a safetensors file, mapped where the platform
// allows it.
func OpenSafetensors(path string, opts Options) (*Safetensors, error) {
	file, err := openFile(path, opts.NoMmap)
	if err != nil {
		return nil, err
	}

	s, err := newSafetensors(file, opts)
	if err != nil {
		return nil, closeOnError(file, err)
	}

	return s, nil
}

// ParseSafetensors serves safetensors bytes the caller owns; they must not
// change while the source is open.
func ParseSafetensors(data []byte, opts Options) (*Safetensors, error) {
	return newSafetensors(&mapping{data: data}, opts)
}

// newSafetensors reads the header and checks every source's shape against
// its entry, so a mismatch fails here rather than on first use. It decodes
// no tensor.
func newSafetensors(file *mapping, opts Options) (*Safetensors, error) {
	store, err := safetensors.OpenStoreFromBytes(file.data, safetensors.StoreOptions{})
	if err != nil {
		return nil, err
	}

	stCfg, err := store.Config()
	if err != nil {
		return nil, err
	}

	entries := opts.names()

	err = checkNames(entries)
	if err != nil {
		return nil, err
	}

	for _, e := range entries {
		if e.Transform == cact.Derived {
			continue
		}

		shape, err := store.Shape(e.Source)
		if err != nil {
			return nil, fmt.Errorf("weights: %s: %w", e.Name, err)
		}

		err = checkSource(e, shape)
		if err != nil {
			return nil, err
		}
	}

	cfg, err := safetensorsConfig(stCfg, entries)
	if err != nil {
		return nil, err
	}

	err = checkDerived(cfg, entries)
	if err != nil {
		return nil, err
	}

	be := &safetensorsBackend{store: store, entries: entries, cfg: cfg}

	return &Safetensors{newCore(entries, cfg, file, be)}, nil
}

// checkDerived requires a config that recomputes every derived record at
// its mapped shape, so a malformed config fails here rather than as a panic
// or a huge allocation in the first Tensor call.
func checkDerived(cfg Config, entries []cact.Entry) error {
	for _, e := range entries {
		if e.Transform != cact.Derived {
			continue
		}

		var ok bool

		if _, perm := cact.HadaPermSeed(e.Name); perm {
			ok = cfg.Dim > 0 && cfg.HadaN == numel(e.Shape)
		} else {
			switch e.Name {
			case "audio.mel_filters":
				ok = cfg.Rate > 0 && cfg.FFT > 0 && cfg.Mels > 0 && len(e.Shape) == 2 &&
					cfg.FFT/2+1 == e.Shape[0] && cfg.Mels == e.Shape[1]
			case manifestName:
				ok = numel(e.Shape) == manifestLen
			default:
				ok = true // not recomputed: derive reports it unavailable
			}
		}

		if !ok {
			return fmt.Errorf("weights: %s: config (dim %d, rate %d, fft %d, mels %d) does not recompute shape %v",
				e.Name, cfg.Dim, cfg.Rate, cfg.FFT, cfg.Mels, e.Shape)
		}
	}

	return nil
}

// checkSource requires a source shape that e.Apply accepts: the layer
// exists, and the layer's shape transposes, lays out lane-major or
// reshapes to e.Shape.
func checkSource(e cact.Entry, shape64 []int64) error {
	shape := make([]int, len(shape64))
	for k, d := range shape64 {
		shape[k] = int(d)
	}

	src := shape

	if e.Layer >= 0 {
		if len(shape) == 0 || e.Layer >= shape[0] {
			return fmt.Errorf("weights: %s: no layer %d in %s %v", e.Name, e.Layer, e.Source, shape)
		}

		src = shape[1:]
	}

	var fits bool

	switch e.Transform {
	case cact.Transpose:
		fits = len(src) == 2 && len(e.Shape) == 2 && src[0] == e.Shape[1] && src[1] == e.Shape[0]
	case cact.LaneMajor:
		fits = len(src) == 3 && len(e.Shape) == 2 && src[1] == e.Shape[1] && src[0]*src[2] == e.Shape[0]
	default:
		fits = numel(src) == numel(e.Shape)
	}

	if !fits {
		return fmt.Errorf("weights: %s: %s %v does not %s to %v", e.Name, e.Source, shape, e.Transform, e.Shape)
	}

	return nil
}

type safetensorsBackend struct {
	store   *safetensors.Store
	entries []cact.Entry
	cfg     Config
}

// decode applies the entry's transform to a freshly decoded source, or
// recomputes a derived record; neither aliases the file data.
func (b *safetensorsBackend) decode(i int) ([]float32, error) {
	e := b.entries[i]
	if e.Transform == cact.Derived {
		return b.derive(e)
	}

	t, err := b.store.Tensor(e.Source)
	if err != nil {
		return nil, err
	}

	shape := make([]int, len(t.Shape))
	for k, d := range t.Shape {
		shape[k] = int(d)
	}

	return e.Apply(t.Data, shape)
}

// derive recomputes a derived record from the config, as F-006 showed the
// exporter computed it: the Hadamard permutations (numpy MT19937), the
// Slaney mel bank in float64 rounded once, and the audio manifest.
func (b *safetensorsBackend) derive(e cact.Entry) ([]float32, error) {
	var v []float32

	if seed, ok := cact.HadaPermSeed(e.Name); ok {
		for _, p := range cact.NumpyPermutation(seed, b.cfg.HadaN) {
			v = append(v, float32(p))
		}
	} else {
		switch e.Name {
		case "audio.mel_filters":
			v = cact.SlaneyMel(b.cfg.Rate, b.cfg.FFT, b.cfg.Mels)
		case manifestName:
			v = b.cfg.manifestRecord()
		default:
			return nil, fmt.Errorf("weights: safetensors cannot recompute %s: %w", e.Name, ErrUnavailable)
		}
	}

	if len(v) != numel(e.Shape) {
		return nil, fmt.Errorf("weights: %s: recomputed %d values for shape %v", e.Name, len(v), e.Shape)
	}

	return v, nil
}

// raw serves no byte blob: the checkpoint has no tokenizer, and none can
// be recomputed (F-007).
func (b *safetensorsBackend) raw(i int) ([]byte, error) {
	e := b.entries[i]
	if e.Shape == nil {
		return nil, fmt.Errorf("weights: safetensors has no %s: %w", e.Name, ErrUnavailable)
	}

	return nil, fmt.Errorf("weights: %s is a tensor, not a byte blob: %w", e.Name, cact.ErrDType)
}

func (b *safetensorsBackend) close() {
	b.store.Close()
	b.store = nil
}
