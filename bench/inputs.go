package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Inputs fingerprints what a run transcribed, so that the report pairs
// the Go and the engine side only when both read the same model and clips
// (SameInputs).
type Inputs struct {
	// Model is the SHA-256 of the model file.
	Model string `json:"model_sha256"`
	// Clips is the SHA-256 over the clip manifest's ids and WAV digests, in
	// order. Both sides check every WAV against its manifest digest, and the
	// 30 s clip is the first clip tiled.
	Clips string `json:"clips_sha256"`
}

// InputsOf fingerprints the model at modelPath and the clips that
// audioDir/manifest.json lists.
func InputsOf(modelPath, audioDir string) (Inputs, error) {
	model, err := fileSHA256(modelPath)
	if err != nil {
		return Inputs{}, err
	}

	clips, err := clipsSHA256(filepath.Join(audioDir, "manifest.json"))
	if err != nil {
		return Inputs{}, err
	}

	return Inputs{Model: model, Clips: clips}, nil
}

// SameInputs reports whether a and b fingerprint the same model and clips.
// An unknown fingerprint matches nothing.
func SameInputs(a, b Inputs) bool {
	return a.Model != "" && a.Clips != "" && a == b
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path) // #nosec G304 -- the model path is the tool's flag.
	if err != nil {
		return "", fmt.Errorf("fingerprint: %w", err)
	}

	defer func() { _ = f.Close() }()

	h := sha256.New()

	_, err = io.Copy(h, f)
	if err != nil {
		return "", fmt.Errorf("fingerprint %s: %w", path, err)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// AcceptSubset is SubsetSHA256 of the committed testdata/oracle/librispeech:
// the 200 test-clean utterances of F-025 that the Phase 9 Accept is
// measured on. TestAcceptSubset fails when those fixtures change.
const AcceptSubset = "4f632cce6313e93954d74dcce04e4d41fddf1bfec2f89d0e975210733fef00a7"

// SubsetSHA256 fingerprints a LibriSpeech fixture set: the SHA-256 over
// dir/subset.json (ids, references, WAV digests) followed by
// dir/manifest.json (the engine's transcripts and error counts).
func SubsetSHA256(dir string) (string, error) {
	h := sha256.New()

	for _, name := range []string{"subset.json", "manifest.json"} {
		raw, err := os.ReadFile(filepath.Join(dir, name)) // #nosec G304 -- the fixture dir is the tool's flag.
		if err != nil {
			return "", fmt.Errorf("fingerprint: %w", err)
		}

		h.Write(raw)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

func clipsSHA256(manifest string) (string, error) {
	raw, err := os.ReadFile(manifest) // #nosec G304 -- the audio dir is the tool's flag.
	if err != nil {
		return "", fmt.Errorf("fingerprint: %w", err)
	}

	var m struct {
		Clips []struct {
			ID     string `json:"id"`
			SHA256 string `json:"sha256"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &m)
	if err != nil {
		return "", fmt.Errorf("fingerprint %s: %w", manifest, err)
	}

	if len(m.Clips) == 0 {
		return "", fmt.Errorf("fingerprint %s: no clips", manifest)
	}

	h := sha256.New()
	for _, c := range m.Clips {
		_, _ = fmt.Fprintf(h, "%s %s\n", c.ID, c.SHA256)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
