//go:build !js || !wasm

package main

import (
	"fmt"
	"os"
)

// main explains the build: the JS API exists only under js/wasm.
func main() {
	fmt.Fprintln(os.Stderr, "whistle-wasm runs in a browser: build it with GOOS=js GOARCH=wasm (just wasm)")
	os.Exit(2)
}
