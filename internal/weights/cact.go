package weights

import (
	"fmt"
	"slices"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
)

// Cact serves a .cact file: entry i of the name map is record i. FP16 and
// FP32 records are widened, CQ records dequantised (cact.File.CQ, F-004).
// Opened with Options.Packed, Matrix also serves the CQ records packed.
type Cact struct {
	*core

	packed bool
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

	return &Cact{core: newCore(entries, cfg, file, &cactBackend{f: f}), packed: opts.Packed}, nil
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

// Matrix serves CQ tensor name packed when the source was opened with
// Options.Packed (Packer, docs/FINDINGS.md F-037): cq.NewGather on the
// record as stored, so the lut kernel's blocked copy waits for the
// caller's Prepare or first product. Other tensors, CQ widths internal/cq
// has no kernel for (only 2 and 4 bits; the caller falls back to Tensor)
// and every tensor in float mode give nil and false.
func (c *Cact) Matrix(name string) (*cq.Matrix, bool, error) {
	i, _, err := c.entry(name)
	if err != nil {
		return nil, false, err
	}

	if !c.packed {
		return nil, false, nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, false, fmt.Errorf("weights: %s: %w", name, ErrClosed)
	}

	be, _ := c.be.(*cactBackend)
	if rec, _ := be.f.Record(i); rec.DType != cact.CQ || rec.Bits != 2 && rec.Bits != 4 {
		return nil, false, nil
	}

	mc := c.matCell(name)
	mc.once.Do(func() {
		var p cact.Packed

		p, mc.err = be.f.CQPacked(i)
		if mc.err == nil {
			mc.m, mc.err = cq.NewGather(p)
		}

		if mc.err != nil {
			mc.err = fmt.Errorf("weights: %s: %w", name, mc.err)
		}

		mc.ok = mc.err == nil
	})

	return mc.m, mc.ok, mc.err
}
