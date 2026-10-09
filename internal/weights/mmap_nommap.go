//go:build !unix

package weights

import (
	"errors"
	"os"
)

// canMmap: windows, js/wasm, wasip1 and plan9 read the files with
// os.ReadFile instead. Windows could map with CreateFileMapping, but that
// is left for when a Windows user needs it (docs/FINDINGS.md F-007).
const canMmap = false

func mmap(*os.File, int) ([]byte, error) {
	return nil, errors.ErrUnsupported
}

func munmap([]byte) error {
	return nil
}
