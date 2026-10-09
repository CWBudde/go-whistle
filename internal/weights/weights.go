// Package weights serves a model's tensors by their upstream-style .cact
// names (cact.WhistleNames, docs/FINDINGS.md F-006) from either format: the
// deployed .cact file (Cact) or the float32 safetensors checkpoint
// (Safetensors). Both return the same shape and layout for every name, so
// model code never sees record indices or Flax paths (docs/FINDINGS.md
// F-007).
//
// Files are memory-mapped where the platform allows it (unix) and read
// into memory elsewhere. Opening decodes nothing: a tensor is decoded, and
// a CQ tensor dequantised, on its first Tensor call and cached. Every
// tensor and byte slice handed out is a fresh heap copy that never aliases
// the file data, so it stays valid after Close.
package weights

import (
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// Lookup errors.
var (
	ErrNotFound    = errors.New("no tensor of that name")
	ErrClosed      = errors.New("weights source is closed")
	ErrUnavailable = errors.New("not available from this format") // F-007
)

// Source serves a model's tensors by name.
type Source interface {
	// Names lists every name in map order (for Whistle, .cact index order).
	Names() []string
	Has(name string) bool
	// Shape is a name's shape, nil for a byte blob (the tokenizer). It
	// reads no tensor data.
	Shape(name string) ([]int, error)
	// Tensor returns a tensor's row-major values and its shape. The values
	// are decoded on the first call and cached; later calls, from any
	// goroutine, return the same slice, which callers must not modify.
	Tensor(name string) ([]float32, []int, error)
	// Raw returns a fresh copy of a byte blob (the tokenizer).
	Raw(name string) ([]byte, error)
	// Config is the model geometry both formats state.
	Config() Config
	// Release drops a name's cached values; the next Tensor call decodes
	// it again. Slices handed out stay valid.
	Release(name string)
	// Close unmaps the file and drops the cache. Tensor and Raw then fail
	// with ErrClosed; the name map and the config stay readable.
	Close() error
}

// Options configure Open, OpenCact, OpenSafetensors and the Parse
// functions.
type Options struct {
	// Names is the name map, entry i for .cact record i; nil means
	// cact.WhistleNames().
	Names []cact.Entry
	// NoMmap reads the file into memory even where it could be mapped.
	NoMmap bool
}

func (o Options) names() []cact.Entry {
	if o.Names == nil {
		return cact.WhistleNames()
	}

	return o.Names
}

// Open opens a .cact file (recognised by its tag) or a safetensors file.
func Open(path string, opts Options) (Source, error) {
	file, err := openFile(path, opts.NoMmap)
	if err != nil {
		return nil, err
	}

	if len(file.data) >= 4 && binary.LittleEndian.Uint32(file.data) == cact.Tag {
		c, err := newCact(file, opts)
		if err != nil {
			return nil, closeOnError(file, err)
		}

		return c, nil
	}

	s, err := newSafetensors(file, opts)
	if err != nil {
		return nil, closeOnError(file, err)
	}

	return s, nil
}

func closeOnError(file *mapping, err error) error {
	return errors.Join(err, file.close())
}

// backend decodes the entries of one format. core serialises it against
// close.
type backend interface {
	decode(i int) ([]float32, error)
	raw(i int) ([]byte, error)
	close()
}

// core is the name lookup, the cache and the close handling both formats
// share.
type core struct {
	entries []cact.Entry
	index   map[string]int
	cfg     Config
	be      backend
	file    *mapping

	mu     sync.RWMutex // Tensor and Raw hold it shared, Close exclusively
	closed bool

	cacheMu sync.Mutex
	cache   map[string]*cell // unbounded until Release or Close (F-007)

	decodes atomic.Int64 // tensors decoded so far, for the laziness tests
}

// cell is one name's decode, done once however many callers wait for it.
type cell struct {
	once sync.Once
	data []float32
	err  error
}

func newCore(entries []cact.Entry, cfg Config, file *mapping, be backend) *core {
	return &core{
		entries: entries, index: indexNames(entries), cfg: cfg, be: be, file: file,
		cache: map[string]*cell{},
	}
}

// indexNames maps each name to its entry; checkNames has ruled out
// duplicates.
func indexNames(entries []cact.Entry) map[string]int {
	index := make(map[string]int, len(entries))
	for i, e := range entries {
		index[e.Name] = i
	}

	return index
}

func checkNames(entries []cact.Entry) error {
	seen := make(map[string]bool, len(entries))
	for _, e := range entries {
		if seen[e.Name] {
			return fmt.Errorf("weights: name %q appears twice in the map", e.Name)
		}

		seen[e.Name] = true
	}

	return nil
}

func (c *core) Names() []string {
	names := make([]string, len(c.entries))
	for i, e := range c.entries {
		names[i] = e.Name
	}

	return names
}

func (c *core) Has(name string) bool {
	_, ok := c.index[name]

	return ok
}

func (c *core) Shape(name string) ([]int, error) {
	_, e, err := c.entry(name)
	if err != nil {
		return nil, err
	}

	return slices.Clone(e.Shape), nil
}

func (c *core) Config() Config { return c.cfg.clone() }

// Mapped reports whether the file was memory-mapped when it was opened.
func (c *core) Mapped() bool { return c.file.mapped }

func (c *core) Tensor(name string) ([]float32, []int, error) {
	i, e, err := c.entry(name)
	if err != nil {
		return nil, nil, err
	}

	if e.Shape == nil {
		return nil, nil, fmt.Errorf("weights: %s is a byte blob, read it with Raw: %w", name, cact.ErrDType)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, nil, fmt.Errorf("weights: %s: %w", name, ErrClosed)
	}

	ce := c.cell(name)
	ce.once.Do(func() {
		c.decodes.Add(1)

		ce.data, ce.err = c.be.decode(i)
		if ce.err == nil && len(ce.data) != numel(e.Shape) {
			ce.data, ce.err = nil, fmt.Errorf("weights: %s: %d values for shape %v", name, len(ce.data), e.Shape)
		}
	})

	if ce.err != nil {
		return nil, nil, ce.err
	}

	return ce.data, slices.Clone(e.Shape), nil
}

func (c *core) Raw(name string) ([]byte, error) {
	i, _, err := c.entry(name)
	if err != nil {
		return nil, err
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.closed {
		return nil, fmt.Errorf("weights: %s: %w", name, ErrClosed)
	}

	return c.be.raw(i)
}

func (c *core) Release(name string) {
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()

	delete(c.cache, name)
}

func (c *core) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	c.closed = true

	c.cacheMu.Lock()
	c.cache = nil
	c.cacheMu.Unlock()

	c.be.close()

	return c.file.close()
}

func (c *core) entry(name string) (int, cact.Entry, error) {
	i, ok := c.index[name]
	if !ok {
		return 0, cact.Entry{}, fmt.Errorf("weights: %s: %w", name, ErrNotFound)
	}

	return i, c.entries[i], nil
}

// cell returns name's cache cell, adding an empty one. The caller holds mu
// shared and has checked closed, so the cache exists.
func (c *core) cell(name string) *cell {
	c.cacheMu.Lock()
	defer c.cacheMu.Unlock()

	ce := c.cache[name]
	if ce == nil {
		ce = &cell{}
		c.cache[name] = ce
	}

	return ce
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}

	return n
}
