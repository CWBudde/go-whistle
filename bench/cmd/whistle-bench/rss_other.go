//go:build !unix

package main

// maxRSS is unknown outside unix.
func maxRSS() int64 { return 0 }
