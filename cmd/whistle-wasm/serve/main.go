// Command serve serves the web demo (web/) for local use: a static file
// server that sends every file uncached, so a rebuilt whistle.wasm is
// picked up on reload. Run it with `just wasm-serve`.
package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8080", "Address to listen on")
	dir := flag.String("dir", "web", "Directory to serve")

	flag.Parse()

	err := run(*addr, *dir)
	if err != nil {
		log.Fatal(err)
	}
}

func run(addr, dir string) error {
	_, err := os.Stat(filepath.Join(dir, "whistle.wasm"))
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("%s has no whistle.wasm; build it with: just wasm", dir)
	}

	srv := &http.Server{
		Addr:              addr,
		Handler:           handler(dir),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("serving %s on http://%s/", dir, addr)

	return srv.ListenAndServe()
}

// handler serves dir with caching off; Go's MIME table sends .wasm as
// application/wasm, which the worker compiles while it downloads.
func handler(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache")
		files.ServeHTTP(w, r)
	})
}
