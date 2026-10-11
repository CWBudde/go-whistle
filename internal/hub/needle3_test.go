package hub

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// needle3Dir is the Needle 3 text model as laid out by `just fetch-needle3`.
const needle3Dir = "../../models/needle3"

func TestPinnedNeedle3(t *testing.T) {
	m, err := PinnedNeedle3()
	if err != nil {
		t.Fatalf("PinnedNeedle3: %v", err)
	}

	if m.Repo != "Cactus-Compute/needle3" {
		t.Errorf("repo = %q", m.Repo)
	}

	if m.Revision != "93e88ede867e4311c39151285e74df513e4b835b" {
		t.Errorf("revision = %q", m.Revision)
	}

	want := map[string]struct {
		local, sha256 string
		size          int64
	}{
		"needle3.cact": {
			"needle3.cact", "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38", 35335380,
		},
		"checkpoints/needle3.safetensors": {
			"needle3.safetensors", "c234c70dccc7a9115e7c41ac2e41d3655fea3b85c245dd898b46179fb90c6c0c", 242047978,
		},
		"config.json": {
			"config.json", "d65e8e076d42b26ed473313e4890dfd8397274a238fd6c1715fff539b464c04e", 1273,
		},
		"tokenizer/tokenizer.model": {
			"tokenizer.model", "97dfd5666620b19875deba9e55953d312364e7763bd08bee428b94f1b9491b25", 126520,
		},
	}
	if len(m.Files) != len(want) {
		t.Fatalf("got %d files, want %d", len(m.Files), len(want))
	}

	for _, f := range m.Files {
		w, ok := want[f.Path]
		if !ok {
			t.Errorf("unexpected file %s", f.Path)

			continue
		}

		if f.Local != w.local || f.SHA256 != w.sha256 || f.Size != w.size {
			t.Errorf("%s = (%s, %s, %d), want (%s, %s, %d)", f.Path, f.Local, f.SHA256, f.Size, w.local, w.sha256, w.size)
		}
	}
}

// TestPinnedManifestsDisjoint keeps the two pins apart: both download into their own directory, so they may reuse
// local names (config.json), but each must name its own repository.
func TestPinnedManifestsDisjoint(t *testing.T) {
	w, err := Pinned()
	if err != nil {
		t.Fatal(err)
	}

	n, err := PinnedNeedle3()
	if err != nil {
		t.Fatal(err)
	}

	if w.Repo == n.Repo || w.Revision == n.Revision {
		t.Errorf("whistle %s@%s and needle3 %s@%s overlap", w.Repo, w.Revision, n.Repo, n.Revision)
	}
}

// TestNeedle3FilesVerify checks the downloaded files against the pin offline: Download in offline mode makes no
// request and fails on any missing or corrupt file.
func TestNeedle3FilesVerify(t *testing.T) {
	m, err := PinnedNeedle3()
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range m.Files {
		_, err := os.Stat(filepath.Join(needle3Dir, f.Local))
		if err != nil {
			t.Skipf("%s missing (run `just fetch-needle3`): %v", f.Local, err)
		}
	}

	c := &Client{Offline: true}

	err = c.Download(context.Background(), m, needle3Dir)
	if err != nil {
		t.Fatalf("offline verify: %v", err)
	}
}

// TestLiveHubNeedle3 downloads the small pinned Needle 3 files from the real Hugging Face Hub. It needs network
// access and is opt-in (WHISTLE_HUB_LIVE=1 go test -run Live ./internal/hub/).
func TestLiveHubNeedle3(t *testing.T) {
	if os.Getenv("WHISTLE_HUB_LIVE") != "1" {
		t.Skip("set WHISTLE_HUB_LIVE=1 to download from huggingface.co")
	}

	m, err := PinnedNeedle3()
	if err != nil {
		t.Fatal(err)
	}

	// Skip the 242 MB safetensors checkpoint; needle3.cact (35 MB) still exercises the CDN redirect.
	small := m
	small.Files = nil

	for _, f := range m.Files {
		if f.Size < 50<<20 {
			small.Files = append(small.Files, f)
		}
	}

	c := NewClientFromEnv()
	c.Offline = false
	c.Progress = t.Output()

	err = c.Download(context.Background(), small, t.TempDir())
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
}
