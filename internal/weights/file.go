package weights

import (
	"fmt"
	"math"
	"os"
)

// mapping is a weights file's bytes: memory-mapped where the platform
// allows it (mmap_unix.go), read into memory otherwise (mmap_nommap.go),
// or bytes the caller passed to a Parse function.
type mapping struct {
	data   []byte
	mapped bool
}

// openFile maps path read-only, or reads it when noMmap is set or the
// platform cannot map. An empty file is read, since mmap rejects length 0,
// and so is a file the system refuses to map (some network and virtual
// file systems): both give the same bytes, only without the sharing.
func openFile(path string, noMmap bool) (*mapping, error) {
	if noMmap || !canMmap {
		return readFile(path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("weights: %w", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("weights: %w", err)
	}

	size := info.Size()

	switch {
	case size == 0:
		return &mapping{data: []byte{}}, nil
	case size > math.MaxInt:
		return nil, fmt.Errorf("weights: %s: %d bytes do not fit the address space", path, size)
	}

	data, err := mmap(f, int(size))
	if err != nil {
		return readFile(path)
	}

	return &mapping{data: data, mapped: true}, nil
}

func readFile(path string) (*mapping, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("weights: %w", err)
	}

	return &mapping{data: data}, nil
}

// close unmaps a mapping and forgets the bytes. Nothing decoded from them
// aliases them (package comment), so this invalidates no handed-out slice.
func (m *mapping) close() error {
	data := m.data
	m.data = nil

	if m.mapped && data != nil {
		err := munmap(data)
		if err != nil {
			return fmt.Errorf("weights: munmap: %w", err)
		}
	}

	return nil
}
