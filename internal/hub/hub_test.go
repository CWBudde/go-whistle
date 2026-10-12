package hub

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const (
	fakeRepo = "Org/model"
	fakeSHA  = "0123456789abcdef0123456789abcdef01234567"
)

// fakeHub serves the two Hugging Face endpoints hub uses: revision resolution and file resolve.
type fakeHub struct {
	files    map[string][]byte // repo path -> content
	sha      string            // the only commit the hub knows
	token    string            // when set, requests without "Bearer <token>" get 401
	ranges   bool              // honour Range requests
	mu       sync.Mutex
	gets     int
	apiCalls int
	rangeHdr []string
}

func newFakeHub(t *testing.T, files map[string][]byte) (*fakeHub, *httptest.Server) {
	t.Helper()

	fh := &fakeHub{files: files, sha: fakeSHA, ranges: true}
	srv := httptest.NewServer(fh)
	t.Cleanup(srv.Close)

	return fh, srv
}

func (fh *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if fh.token != "" && r.Header.Get("Authorization") != "Bearer "+fh.token {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	if rev, ok := strings.CutPrefix(r.URL.Path, "/api/models/"+fakeRepo+"/revision/"); ok {
		fh.mu.Lock()
		fh.apiCalls++
		fh.mu.Unlock()

		if rev != "main" && !strings.HasPrefix(fh.sha, rev) {
			http.NotFound(w, r)
			return
		}

		err := json.NewEncoder(w).Encode(map[string]string{"sha": fh.sha})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}

		return
	}

	path, ok := strings.CutPrefix(r.URL.Path, "/"+fakeRepo+"/resolve/"+fh.sha+"/")
	if !ok {
		http.NotFound(w, r)
		return
	}

	body, ok := fh.files[path]
	if !ok {
		http.NotFound(w, r)
		return
	}

	fh.mu.Lock()
	fh.gets++
	fh.rangeHdr = append(fh.rangeHdr, r.Header.Get("Range"))
	fh.mu.Unlock()

	if rng := r.Header.Get("Range"); fh.ranges && rng != "" {
		start, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rng, "bytes="), "-"))
		if err != nil {
			http.Error(w, "bad range", http.StatusBadRequest)
			return
		}

		if start >= len(body) {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", len(body)))
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)

			return
		}

		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
		w.Header().Set("Content-Length", strconv.Itoa(len(body)-start))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(body[start:])

		return
	}

	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	_, _ = w.Write(body)
}

func (fh *fakeHub) counts() (gets, apiCalls int) {
	fh.mu.Lock()
	defer fh.mu.Unlock()

	return fh.gets, fh.apiCalls
}

func sum(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func fakeContent() map[string][]byte {
	big := make([]byte, 200_000)
	for i := range big {
		big[i] = byte(i * 7)
	}

	return map[string][]byte{
		"weights/model.bin": big,
		"config.json":       []byte(`{"d_model": 512}`),
	}
}

func fakeManifest(files map[string][]byte) Manifest {
	return Manifest{
		Repo:     fakeRepo,
		Revision: fakeSHA,
		Files: []File{
			{
				Path:   "weights/model.bin",
				Local:  "model.bin",
				SHA256: sum(files["weights/model.bin"]),
				Size:   int64(len(files["weights/model.bin"])),
			},
			{
				Path:   "config.json",
				Local:  "config.json",
				SHA256: sum(files["config.json"]),
				Size:   int64(len(files["config.json"])),
			},
		},
	}
}

func testClient(srv *httptest.Server) *Client {
	return &Client{HTTP: srv.Client(), Endpoint: srv.URL}
}

func assertFiles(t *testing.T, dir string, m Manifest, files map[string][]byte) {
	t.Helper()

	for _, f := range m.Files {
		got, err := os.ReadFile(filepath.Join(dir, f.Local))
		if err != nil {
			t.Fatalf("read %s: %v", f.Local, err)
		}

		if sum(got) != sum(files[f.Path]) {
			t.Fatalf("%s: content mismatch", f.Local)
		}

		_, err = os.Stat(filepath.Join(dir, f.Local+partSuffix))
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s: leftover %s file (err=%v)", f.Local, partSuffix, err)
		}
	}
}

func TestPinned(t *testing.T) {
	m, err := Pinned()
	if err != nil {
		t.Fatalf("Pinned: %v", err)
	}

	if m.Repo != "Cactus-Compute/whistle" {
		t.Errorf("repo = %q", m.Repo)
	}

	if m.Revision != "b358ddadd89b7a713b5aa131f23032d3cca1b251" {
		t.Errorf("revision = %q", m.Revision)
	}

	want := map[string]string{
		"whistle.cact":                    "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb",
		"checkpoints/whistle.safetensors": "5fd58c246f522ecee2b568b306598a3befdc274c225e3a7376abce277ab3d107",
		"config.json":                     "ac2dbcb98a945eff2de98a03104c468dc9d68c06b9ff791539327f90298a3d43",
	}
	if len(m.Files) != len(want) {
		t.Fatalf("got %d files, want %d", len(m.Files), len(want))
	}

	for _, f := range m.Files {
		if want[f.Path] != f.SHA256 {
			t.Errorf("%s: sha256 = %s, want %s", f.Path, f.SHA256, want[f.Path])
		}
	}
}

func TestManifestValidate(t *testing.T) {
	good := fakeManifest(fakeContent())

	err := good.Validate()
	if err != nil {
		t.Fatalf("valid manifest rejected: %v", err)
	}

	cases := map[string]func(m *Manifest){
		"short revision": func(m *Manifest) { m.Revision = "b358ddad" },
		"branch":         func(m *Manifest) { m.Revision = "main" },
		"no repo":        func(m *Manifest) { m.Repo = "" },
		"no files":       func(m *Manifest) { m.Files = nil },
		"bad sha":        func(m *Manifest) { m.Files[0].SHA256 = "xyz" },
		"zero size":      func(m *Manifest) { m.Files[0].Size = 0 },
		"dup local":      func(m *Manifest) { m.Files[1].Local = m.Files[0].Local },
		"local escapes":  func(m *Manifest) { m.Files[0].Local = "../evil" },
		"local nested":   func(m *Manifest) { m.Files[0].Local = "a/b" },
	}

	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m := fakeManifest(fakeContent())
			mutate(&m)

			err := m.Validate()
			if err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestDownloadFreshThenSkip(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	dir := t.TempDir()

	err := testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	assertFiles(t, dir, m, files)

	if gets, api := fh.counts(); gets != 2 || api != 1 {
		t.Fatalf("first run: gets=%d api=%d, want 2 and 1", gets, api)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("second Download: %v", err)
	}

	if gets, _ := fh.counts(); gets != 2 {
		t.Fatalf("second run downloaded again: gets=%d", gets)
	}
}

func TestDownloadResume(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	dir := t.TempDir()

	const have = 123_456

	err := os.WriteFile(filepath.Join(dir, "model.bin"+partSuffix), files["weights/model.bin"][:have], 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	assertFiles(t, dir, m, files)

	fh.mu.Lock()
	defer fh.mu.Unlock()

	if fh.rangeHdr[0] != "bytes=123456-" {
		t.Fatalf("Range = %q, want bytes=123456-", fh.rangeHdr[0])
	}
}

func TestDownloadResumeServerIgnoresRange(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	fh.ranges = false
	m := fakeManifest(files)
	dir := t.TempDir()

	// A prefix that would corrupt the file if it were kept after a 200 response.
	err := os.WriteFile(filepath.Join(dir, "model.bin"+partSuffix), []byte("garbage"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	assertFiles(t, dir, m, files)
}

func TestDownloadCompletePart(t *testing.T) {
	files := fakeContent()
	_, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	dir := t.TempDir()

	// The whole file sits in .part (a crash between the last write and the rename): the server answers 416.
	err := os.WriteFile(filepath.Join(dir, "model.bin"+partSuffix), files["weights/model.bin"], 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	assertFiles(t, dir, m, files)
}

func TestDownloadCorruptCompletePart(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	dir := t.TempDir()

	// A full-size .part with a wrong byte: the server answers the range with 416, the digest fails, and the same
	// Download must start over from byte zero instead of failing.
	corrupt := bytes.Clone(files["weights/model.bin"])
	corrupt[len(corrupt)/2] ^= 0xff

	err := os.WriteFile(filepath.Join(dir, "model.bin"+partSuffix), corrupt, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	assertFiles(t, dir, m, files)

	fh.mu.Lock()
	defer fh.mu.Unlock()

	if !slices.Contains(fh.rangeHdr, "") {
		t.Errorf("Range headers %q: want a full re-download after the corrupt part", fh.rangeHdr)
	}
}

func TestDownloadChecksumMismatch(t *testing.T) {
	files := fakeContent()
	_, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	m.Files[1].SHA256 = strings.Repeat("0", 64)
	dir := t.TempDir()

	err := testClient(srv).Download(context.Background(), m, dir)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want sha256 mismatch", err)
	}

	for _, name := range []string{"config.json", "config.json" + partSuffix} {
		_, err := os.Stat(filepath.Join(dir, name))
		if !errors.Is(err, os.ErrNotExist) {
			t.Errorf("%s exists after a mismatch (err=%v)", name, err)
		}
	}
}

func TestDownloadReplacesCorruptFile(t *testing.T) {
	files := fakeContent()
	_, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	dir := t.TempDir()

	err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"d_model": 513}`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	assertFiles(t, dir, m, files)
}

func TestDownloadOffline(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	m := fakeManifest(files)
	dir := t.TempDir()

	offline := testClient(srv)
	offline.Offline = true

	err := offline.Download(context.Background(), m, dir)
	if err == nil || !strings.Contains(err.Error(), "HF_HUB_OFFLINE") {
		t.Fatalf("offline with missing files: err = %v, want a HF_HUB_OFFLINE hint", err)
	}

	if gets, api := fh.counts(); gets != 0 || api != 0 {
		t.Fatalf("offline run made requests: gets=%d api=%d", gets, api)
	}

	err = testClient(srv).Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("online Download: %v", err)
	}

	gets, api := fh.counts()

	err = offline.Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("offline with files present: %v", err)
	}

	if g, a := fh.counts(); g != gets || a != api {
		t.Fatalf("offline run made requests: gets %d->%d api %d->%d", gets, g, api, a)
	}
}

func TestDownloadToken(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	fh.token = "hf_secret"
	m := fakeManifest(files)

	err := testClient(srv).Download(context.Background(), m, t.TempDir())
	if _, ok := errors.AsType[*AccessDeniedError](err); !ok {
		t.Fatalf("without token: err = %v, want *AccessDeniedError", err)
	}

	if !strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("access error lacks the HF_TOKEN hint: %v", err)
	}

	c := testClient(srv)
	c.Token = "hf_secret"

	dir := t.TempDir()

	err = c.Download(context.Background(), m, dir)
	if err != nil {
		t.Fatalf("with token: %v", err)
	}

	assertFiles(t, dir, m, files)
}

func TestResolveRevision(t *testing.T) {
	_, srv := newFakeHub(t, fakeContent())
	c := testClient(srv)

	for _, rev := range []string{fakeSHA[:8], fakeSHA, "main"} {
		got, err := c.ResolveRevision(context.Background(), fakeRepo, rev)
		if err != nil {
			t.Fatalf("ResolveRevision(%s): %v", rev, err)
		}

		if got != fakeSHA {
			t.Fatalf("ResolveRevision(%s) = %s, want %s", rev, got, fakeSHA)
		}
	}

	_, err := c.ResolveRevision(context.Background(), fakeRepo, "deadbeef")
	if err == nil {
		t.Fatal("unknown revision resolved")
	}
}

func TestDownloadPinnedRevisionGone(t *testing.T) {
	files := fakeContent()
	fh, srv := newFakeHub(t, files)
	fh.sha = strings.Repeat("f", 40) // the hub no longer has the pinned commit
	m := fakeManifest(files)

	err := testClient(srv).Download(context.Background(), m, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "resolve revision") {
		t.Fatalf("err = %v, want a resolve revision error", err)
	}

	if gets, _ := fh.counts(); gets != 0 {
		t.Fatalf("downloaded %d files from an unresolved revision", gets)
	}
}

func TestNewClientFromEnv(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_x")
	t.Setenv("HF_ENDPOINT", "https://mirror.example/")

	for _, v := range []string{"1", "true", "YES", "on"} {
		t.Setenv("HF_HUB_OFFLINE", v)

		c := NewClientFromEnv()
		if !c.Offline || c.Token != "hf_x" || c.Endpoint != "https://mirror.example" {
			t.Fatalf("HF_HUB_OFFLINE=%s: got %+v", v, c)
		}
	}

	t.Setenv("HF_HUB_OFFLINE", "0")

	if NewClientFromEnv().Offline {
		t.Fatal("HF_HUB_OFFLINE=0 enabled offline mode")
	}

	t.Setenv("HF_ENDPOINT", "")

	if got := NewClientFromEnv().Endpoint; got != DefaultEndpoint {
		t.Fatalf("default endpoint = %q", got)
	}
}
