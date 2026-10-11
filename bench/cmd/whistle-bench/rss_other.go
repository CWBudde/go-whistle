//go:build !unix

package main

// maxRSS is unknown outside unix: 0, which the report shows as missing.
func maxRSS() int64 { return 0 }
