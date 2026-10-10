//go:build unix

package main

import (
	"syscall"

	"github.com/cwbudde/go-whistle/bench"
)

// maxRSS is this process's peak resident set in bytes, 0 if unknown.
func maxRSS() int64 {
	var ru syscall.Rusage

	err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if err != nil {
		return 0
	}

	return bench.MaxRSSBytes(ru.Maxrss)
}
