//go:build unix

package weights

import (
	"os"
	"syscall"
)

// canMmap: every unix GOOS has syscall.Mmap, so the files are mapped
// read-only and shared, without cgo (docs/FINDINGS.md F-007).
const canMmap = true

func mmap(f *os.File, size int) ([]byte, error) {
	return syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
}

func munmap(data []byte) error {
	return syscall.Munmap(data)
}
