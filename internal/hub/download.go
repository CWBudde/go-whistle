package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
)

// partSuffix marks an incomplete download. It is kept on errors so the next run resumes from it.
const partSuffix = ".part"

// Download makes dir hold every file of m, verified by size and SHA-256.
//
// Files already present and verified are left alone. Missing or corrupt files are downloaded from m.Revision,
// resuming from a leftover <local>.part file when the server supports Range requests. In offline mode no request
// is made and a missing or corrupt file is an error.
func (c *Client) Download(ctx context.Context, m Manifest, dir string) error {
	err := m.Validate()
	if err != nil {
		return err
	}

	if !c.Offline {
		rev, err := c.ResolveRevision(ctx, m.Repo, m.Revision)
		if err != nil {
			return err
		}

		if rev != m.Revision {
			return fmt.Errorf("revision %s of %s resolved to %s", m.Revision, m.Repo, rev)
		}
	}

	err = os.MkdirAll(dir, 0o750)
	if err != nil {
		return fmt.Errorf("create download dir: %w", err)
	}

	for _, f := range m.Files {
		err := c.fetch(ctx, m, f, dir)
		if err != nil {
			return err
		}
	}

	return nil
}

// fetch makes dir/f.Local a verified copy of f.
func (c *Client) fetch(ctx context.Context, m Manifest, f File, dir string) error {
	dest := filepath.Join(dir, f.Local)

	ok, err := verified(dest, f)
	if err != nil {
		return err
	}

	if ok {
		c.logf("ok       %s (sha256 verified)", f.Local)
		return nil
	}

	if c.Offline {
		return fmt.Errorf("%s is missing or fails verification and HF_HUB_OFFLINE is set; unset it to download", dest)
	}

	u, err := url.JoinPath(c.endpoint(), m.Repo, "resolve", m.Revision, f.Path)
	if err != nil {
		return fmt.Errorf("build download URL: %w", err)
	}

	part := dest + partSuffix

	err = c.fetchPart(ctx, m.Repo, u, part, f)
	if errors.Is(err, errRestart) {
		// The server could not continue the .part file; start over once.
		_ = os.Remove(part)
		err = c.fetchPart(ctx, m.Repo, u, part, f)
	}

	if err != nil {
		return fmt.Errorf("download %s: %w", f.Path, err)
	}

	err = os.Rename(part, dest)
	if err != nil {
		return fmt.Errorf("move %s into place: %w", f.Local, err)
	}

	c.logf("ok       %s (sha256 verified)", f.Local)

	return nil
}

var errRestart = errors.New("server cannot resume the partial download")

// fetchPart completes part from u and verifies it against f. A part that fails verification is removed; any other
// error keeps it for the next run.
func (c *Client) fetchPart(ctx context.Context, repo, u, part string, f File) error {
	h := sha256.New()

	offset, err := hashPrefix(part, h, f.Size)
	if err != nil {
		return err
	}

	rangeHdr := ""
	if offset > 0 {
		rangeHdr = fmt.Sprintf("bytes=%d-", offset)
	}

	resp, err := c.get(ctx, u, rangeHdr)
	if err != nil {
		return err
	}

	defer func() { _ = resp.Body.Close() }()

	switch {
	case offset > 0 && resp.StatusCode == http.StatusPartialContent:
		c.logf("resume   %s at %d/%d bytes", f.Local, offset, f.Size)
	case offset == f.Size && resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		// Everything is in part already; only the rename was missing.
		return verifyPart(part, h, offset, f)
	case offset > 0 && resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		return errRestart
	default:
		err = checkStatus(resp, repo, http.StatusOK)
		if err != nil {
			return err
		}

		// A 200 carries the whole file: drop what the part file held.
		offset = 0

		h.Reset()

		c.logf("download %s (%d bytes)", f.Local, f.Size)
	}

	n, err := appendBody(part, offset, resp.Body, h)
	if err != nil {
		return err
	}

	return verifyPart(part, h, offset+n, f)
}

// hashPrefix feeds an existing part file into h and returns its length. A part longer than the file is discarded.
func hashPrefix(part string, h hash.Hash, size int64) (int64, error) {
	fh, err := os.Open(part) // #nosec G304 -- the path is dir + a validated local file name.
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}

	if err != nil {
		return 0, fmt.Errorf("open partial download: %w", err)
	}

	defer func() { _ = fh.Close() }()

	n, err := io.Copy(h, fh)
	if err != nil {
		return 0, fmt.Errorf("read partial download: %w", err)
	}

	if n > size {
		h.Reset()
		return 0, nil
	}

	return n, nil
}

// appendBody writes body to part at offset (truncating anything after it) and into h.
func appendBody(part string, offset int64, body io.Reader, h hash.Hash) (int64, error) {
	// #nosec G302 G304 -- model files are not secret; the path is dir + a validated local file name.
	fh, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, fmt.Errorf("open partial download: %w", err)
	}

	err = fh.Truncate(offset)
	if err == nil {
		_, err = fh.Seek(offset, io.SeekStart)
	}

	if err != nil {
		_ = fh.Close()
		return 0, fmt.Errorf("prepare partial download: %w", err)
	}

	n, copyErr := io.Copy(io.MultiWriter(fh, h), body)
	closeErr := fh.Close()

	if copyErr != nil {
		return n, fmt.Errorf("receive: %w", copyErr)
	}

	if closeErr != nil {
		return n, fmt.Errorf("close partial download: %w", closeErr)
	}

	return n, nil
}

// verifyPart checks the size and digest of a finished part file and removes it when they do not match.
func verifyPart(part string, h hash.Hash, size int64, f File) error {
	if size != f.Size {
		if size > f.Size {
			_ = os.Remove(part)
		}

		return fmt.Errorf("size mismatch: got %d bytes, want %d", size, f.Size)
	}

	got := hex.EncodeToString(h.Sum(nil))
	if got != f.SHA256 {
		_ = os.Remove(part)
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, f.SHA256)
	}

	return nil
}

// verified reports whether path exists with f's size and digest.
func verified(path string, f File) (bool, error) {
	fi, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}

	if err != nil {
		return false, fmt.Errorf("stat %s: %w", path, err)
	}

	if !fi.Mode().IsRegular() {
		return false, fmt.Errorf("%s is not a regular file", path)
	}

	if fi.Size() != f.Size {
		return false, nil
	}

	fh, err := os.Open(path) // #nosec G304 -- the path is dir + a validated local file name.
	if err != nil {
		return false, fmt.Errorf("open %s: %w", path, err)
	}

	defer func() { _ = fh.Close() }()

	h := sha256.New()

	_, err = io.Copy(h, fh)
	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}

	return hex.EncodeToString(h.Sum(nil)) == f.SHA256, nil
}
