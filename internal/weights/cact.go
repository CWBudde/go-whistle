package weights

import (
	"fmt"
	"slices"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// Cact serves a .cact file: entry i of the name map is record i. FP16 and
// FP32 records are widened, CQ records dequantised (cact.File.CQ, F-004).
type Cact struct {
	*core
}

// OpenCact opens a .cact file, mapped where the platform allows it.
func OpenCact(path string, opts Options) (*Cact, error) {
	file, err := openFile(path, opts.NoMmap)
	if err != nil {
		return nil, err
	}

	c, err := newCact(file, opts)
	if err != nil {
		return nil, closeOnError(file, err)
	}

	return c, nil
}

// ParseCact serves .cact bytes the caller owns; they must not change while
// the source is open.
func ParseCact(data []byte, opts Options) (*Cact, error) {
	return newCact(&mapping{data: data}, opts)
}

// newCact parses the directory and checks every record against the name
// map. It decodes no tensor; the config reads the 13-float audio manifest.
func newCact(file *mapping, opts Options) (*Cact, error) {
	f, err := cact.Parse(file.data)
	if err != nil {
		return nil, err
	}

	entries := opts.names()
	if len(entries) != f.Len() {
		return nil, fmt.Errorf("weights: %d names for %d tensors", len(entries), f.Len())
	}

	err = checkNames(entries)
	if err != nil {
		return nil, err
	}

	for i, e := range entries {
		rec, _ := f.Record(i)

		blob := rec.DType == cact.RAW
		if blob != (e.Shape == nil) || !blob && !slices.Equal(rec.Dims(), e.Shape) {
			return nil, fmt.Errorf("weights: %s: record %d is %s %v, the map says %v", e.Name, i, rec.DType, rec.Dims(), e.Shape)
		}
	}

	cfg, err := cactConfig(f, entries)
	if err != nil {
		return nil, err
	}

	return &Cact{newCore(entries, cfg, file, &cactBackend{f: f})}, nil
}

type cactBackend struct {
	f *cact.File
}

// decode copies record i out of the file data: cact's readers always
// allocate the result.
func (b *cactBackend) decode(i int) ([]float32, error) {
	rec, err := b.f.Record(i)
	if err != nil {
		return nil, err
	}

	if rec.DType == cact.CQ {
		return b.f.CQ(i)
	}

	return b.f.Float32(i)
}

func (b *cactBackend) raw(i int) ([]byte, error) { return b.f.Raw(i) }

func (b *cactBackend) close() { b.f = nil }
