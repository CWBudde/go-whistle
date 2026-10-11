package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHandler(t *testing.T) {
	dir := t.TempDir()

	for name, body := range map[string]string{"index.html": "<!doctype html>", "whistle.wasm": "\x00asm"} {
		err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600)
		if err != nil {
			t.Fatal(err)
		}
	}

	err := run("127.0.0.1:0", t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "just wasm") {
		t.Errorf("run without whistle.wasm: %v, want a pointer to just wasm", err)
	}

	srv := httptest.NewServer(handler(dir))
	defer srv.Close()

	for path, want := range map[string]string{"/": "text/html; charset=utf-8", "/whistle.wasm": "application/wasm"} {
		res, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}

		_ = res.Body.Close()

		if res.StatusCode != http.StatusOK || res.Header.Get("Content-Type") != want || res.Header.Get("Cache-Control") != "no-cache" {
			t.Errorf("%s: %d, %q, cache %q; want 200, %q, no-cache", path, res.StatusCode, res.Header.Get("Content-Type"),
				res.Header.Get("Cache-Control"), want)
		}
	}
}
