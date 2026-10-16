//go:build race

package main

// raceDetector reports a build with the race detector, under which the
// weights-gated tests do not finish within go test's 10 minutes.
const raceDetector = true
