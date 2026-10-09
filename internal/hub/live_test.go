package hub

import (
	"context"
	"os"
	"testing"
)

// TestLiveHub downloads the small pinned files from the real Hugging Face Hub. It needs network access and is
// opt-in: WHISTLE_HUB_LIVE=1 go test -run Live ./internal/hub/
func TestLiveHub(t *testing.T) {
	if os.Getenv("WHISTLE_HUB_LIVE") != "1" {
		t.Skip("set WHISTLE_HUB_LIVE=1 to download from huggingface.co")
	}

	m, err := Pinned()
	if err != nil {
		t.Fatal(err)
	}

	// Skip the 220 MB safetensors checkpoint; whistle.cact (17 MB) still exercises the CDN redirect.
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

	dir := t.TempDir()

	err = c.Download(context.Background(), small, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	// The files are verified locally now, so an offline re-run must succeed without the network.
	c.Offline = true

	err = c.Download(context.Background(), small, dir)
	if err != nil {
		t.Fatalf("offline re-run: %v", err)
	}
}
