package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/hub"
)

const testRev = "0123456789abcdef0123456789abcdef01234567"

func TestRootHasDownload(t *testing.T) {
	root := NewRootCmd()

	cmd, _, err := root.Find([]string{"download"})
	if err != nil || cmd.Name() != "download" {
		t.Fatalf("root has no download subcommand: %v", err)
	}

	if got := cmd.Flags().Lookup("dir").DefValue; got != defaultModelDir {
		t.Errorf("--dir default = %q, want %q", got, defaultModelDir)
	}

	if got := cmd.Flags().Lookup("offline").DefValue; got != "false" {
		t.Errorf("--offline default = %q, want false", got)
	}
}

func TestDownloadCmdLayout(t *testing.T) {
	if defaultModelDir != filepath.Join("models", "whistle") {
		t.Errorf("defaultModelDir = %q, want models/whistle", defaultModelDir)
	}

	m, err := hub.Pinned()
	if err != nil {
		t.Fatal(err)
	}

	got := make([]string, 0, len(m.Files))
	for _, f := range m.Files {
		got = append(got, f.Local)
	}

	slices.Sort(got)

	want := []string{"config.json", "whistle.cact", "whistle.safetensors"}
	if !slices.Equal(got, want) {
		t.Errorf("local layout = %v, want %v", got, want)
	}
}

func TestRunDownload(t *testing.T) {
	files := map[string][]byte{
		"config.json":          []byte(`{"d_model":256}`),
		"checkpoints/model.st": bytes.Repeat([]byte("whistle"), 1000),
	}

	m := hub.Manifest{Repo: "test/whistle", Revision: testRev}

	for path, data := range files {
		sum := sha256.Sum256(data)
		m.Files = append(m.Files, hub.File{
			Path:   path,
			Local:  filepath.Base(path),
			SHA256: hex.EncodeToString(sum[:]),
			Size:   int64(len(data)),
		})
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/test/whistle/revision/"+testRev {
			_, _ = w.Write([]byte(`{"sha":"` + testRev + `"}`))
			return
		}

		path, ok := strings.CutPrefix(r.URL.Path, "/test/whistle/resolve/"+testRev+"/")
		if data, found := files[path]; ok && found {
			_, _ = w.Write(data)
			return
		}

		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)

	dir := filepath.Join(t.TempDir(), "models", "whistle")
	c := &hub.Client{HTTP: srv.Client(), Endpoint: srv.URL}

	var out bytes.Buffer

	err := runDownload(context.Background(), c, m, dir, &out)
	if err != nil {
		t.Fatalf("runDownload: %v", err)
	}

	for path, data := range files {
		got, err := os.ReadFile(filepath.Join(dir, filepath.Base(path)))
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(got, data) {
			t.Errorf("%s: content differs", path)
		}
	}

	want := "whistle 01234567 ready in " + dir + "\n"
	if out.String() != want {
		t.Errorf("stdout = %q, want %q", out.String(), want)
	}
}

// noNetwork points HF_ENDPOINT at a server that fails the test on any request.
func noNetwork(t *testing.T) {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request in offline mode: %s", r.URL)
	}))
	t.Cleanup(srv.Close)

	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_TOKEN", "")
}

func runCmd(t *testing.T, args ...string) (string, error) {
	t.Helper()

	root := NewRootCmd()

	var out, errOut bytes.Buffer

	root.SetOut(&out)
	root.SetErr(&errOut)
	root.SetArgs(args)

	err := root.ExecuteContext(context.Background())

	return out.String(), err
}

func TestDownloadCmdOffline(t *testing.T) {
	noNetwork(t)
	t.Setenv("HF_HUB_OFFLINE", "")

	out, err := runCmd(t, "download", "--offline", "--dir", t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "HF_HUB_OFFLINE") {
		t.Fatalf("err = %v, want an error naming HF_HUB_OFFLINE", err)
	}

	if out != "" {
		t.Errorf("stdout = %q, want empty on failure", out)
	}
}

func TestDownloadCmdEnvOffline(t *testing.T) {
	noNetwork(t)
	t.Setenv("HF_HUB_OFFLINE", "1")

	_, err := runCmd(t, "download", "--dir", t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "HF_HUB_OFFLINE") {
		t.Fatalf("err = %v, want an error naming HF_HUB_OFFLINE", err)
	}
}
