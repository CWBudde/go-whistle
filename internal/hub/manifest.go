// Package hub is a minimal pure-Go Hugging Face Hub downloader for the pinned Whistle weights.
//
// It resolves the pinned revision, resumes interrupted downloads, verifies every file against the SHA-256 in the
// embedded manifest.lock.json and honours HF_TOKEN, HF_HUB_OFFLINE and HF_ENDPOINT.
package hub

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

//go:embed manifest.lock.json
var pinnedManifest []byte

var (
	commitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)
	sha256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

// File is one pinned file of a Hub repository.
type File struct {
	Path   string `json:"path"`   // path inside the repository
	Local  string `json:"local"`  // file name in the download directory
	SHA256 string `json:"sha256"` // lowercase hex digest of the content
	Size   int64  `json:"size"`   // content length in bytes
}

// Manifest pins a set of files of one repository to one commit.
type Manifest struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"` // full 40-hex commit sha
	Files    []File `json:"files"`
}

// Pinned returns the manifest embedded in the binary: Cactus-Compute/whistle at b358ddad.
func Pinned() (Manifest, error) {
	var m Manifest

	err := json.Unmarshal(pinnedManifest, &m)
	if err != nil {
		return Manifest{}, fmt.Errorf("parse embedded manifest: %w", err)
	}

	err = m.Validate()
	if err != nil {
		return Manifest{}, fmt.Errorf("embedded manifest: %w", err)
	}

	return m, nil
}

// Validate checks that the manifest pins a full commit and that every file has a digest, a size and a plain local
// file name, so a manifest cannot write outside the download directory.
func (m Manifest) Validate() error {
	if m.Repo == "" {
		return errors.New("manifest has no repo")
	}

	if !commitPattern.MatchString(m.Revision) {
		return fmt.Errorf("revision %q is not a full commit sha", m.Revision)
	}

	if len(m.Files) == 0 {
		return errors.New("manifest has no files")
	}

	seen := make(map[string]bool, len(m.Files))

	for _, f := range m.Files {
		switch {
		case f.Path == "" || strings.HasPrefix(f.Path, "/"):
			return fmt.Errorf("file %q: invalid repository path", f.Path)
		case f.Local == "" || f.Local == "." || f.Local == ".." || strings.ContainsAny(f.Local, `/\`):
			return fmt.Errorf("file %q: local name %q is not a plain file name", f.Path, f.Local)
		case !sha256Pattern.MatchString(f.SHA256):
			return fmt.Errorf("file %q: sha256 %q is not 64 lowercase hex digits", f.Path, f.SHA256)
		case f.Size <= 0:
			return fmt.Errorf("file %q: size must be positive", f.Path)
		case seen[f.Local]:
			return fmt.Errorf("file %q: duplicate local name %q", f.Path, f.Local)
		}

		seen[f.Local] = true
	}

	return nil
}
