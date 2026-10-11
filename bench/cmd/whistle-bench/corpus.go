package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cwbudde/go-whistle/internal/eval"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
)

// clip is one fixture clip of testdata/audio with the engine's committed
// transcript (testdata/oracle), the set TestTranscribeMatchesEngine uses.
type clip struct {
	ID       string
	Text     string // the engine's
	Language string // the engine's
	Samples  []float32
}

// seconds is the clip's duration.
func (c clip) seconds() float64 { return float64(len(c.Samples)) / whistle.SampleRate }

// loadClips reads the corpus manifest, every clip (checked against its
// SHA-256) and the engine's transcript of it.
func loadClips(audioDir, oracleDir string) ([]clip, error) {
	var manifest struct {
		Clips []struct {
			ID     string `json:"id"`
			SHA256 string `json:"sha256"`
		} `json:"clips"`
	}

	err := readJSON(filepath.Join(audioDir, "manifest.json"), &manifest)
	if err != nil {
		return nil, err
	}

	if len(manifest.Clips) == 0 {
		return nil, fmt.Errorf("%s: no clips", audioDir)
	}

	clips := make([]clip, 0, len(manifest.Clips))

	for _, m := range manifest.Clips {
		pcm, err := readWAV(filepath.Join(audioDir, m.ID+".wav"), m.SHA256)
		if err != nil {
			return nil, err
		}

		var engine struct {
			Text     string `json:"text"`
			Language string `json:"language"`
		}

		err = readJSON(filepath.Join(oracleDir, m.ID+".json"), &engine)
		if err != nil {
			return nil, err
		}

		clips = append(clips, clip{ID: m.ID, Text: engine.Text, Language: engine.Language, Samples: pcm})
	}

	return clips, nil
}

// checkFixtureModel rejects engine fixtures recorded with another model
// than the one benchmarked: dir/manifest.json names the model's SHA-256,
// and only fixtures of that model are a reference for its transcripts.
func checkFixtureModel(dir, modelSHA256 string) error {
	var manifest struct {
		Model struct {
			SHA256 string `json:"sha256"`
		} `json:"model"`
	}

	err := readJSON(filepath.Join(dir, "manifest.json"), &manifest)
	if err != nil {
		return err
	}

	if manifest.Model.SHA256 != modelSHA256 {
		return fmt.Errorf("%s: the engine fixtures were recorded with model %q, the benchmarked model is %q",
			dir, manifest.Model.SHA256, modelSHA256)
	}

	return nil
}

// longClip is the 30 s clip: the first corpus clip tiled. It reads that
// clip alone (checked against its digest), not the whole corpus.
func longClip(audio string) ([]float32, error) {
	var manifest struct {
		Clips []struct {
			ID     string `json:"id"`
			SHA256 string `json:"sha256"`
		} `json:"clips"`
	}

	err := readJSON(filepath.Join(audio, "manifest.json"), &manifest)
	if err != nil {
		return nil, err
	}

	if len(manifest.Clips) == 0 {
		return nil, fmt.Errorf("%s: no clips", audio)
	}

	first := manifest.Clips[0]

	pcm, err := readWAV(filepath.Join(audio, first.ID+".wav"), first.SHA256)
	if err != nil {
		return nil, err
	}

	return tile30s(pcm), nil
}

// tile30s is pcm repeated to exactly 30 s, the 30 s clip of F-027, F-035
// and F-037 when pcm is the first corpus clip.
func tile30s(pcm []float32) []float32 {
	out := make([]float32, whistle.MaxSamples)
	if len(pcm) == 0 {
		return out
	}

	for i := 0; i < len(out); i += len(pcm) {
		copy(out[i:], pcm)
	}

	return out
}

// utterance is one utterance of the LibriSpeech subset (testdata/oracle/
// librispeech, docs/FINDINGS.md F-025) with the engine's transcript.
type utterance struct {
	ID         string `json:"id"`
	Samples    int    `json:"samples"`
	SHA256     string `json:"sha256"`
	Transcript string `json:"transcript"` // the LibriSpeech reference

	engine       string // the engine's text
	engineErrors int    // as manifest.json counts them
}

// librispeech is the subset with its audio directory and the manifest's
// totals over the engine's transcripts.
type librispeech struct {
	utterances          []utterance
	audioDir            string
	words, engineErrors int
}

// loadLibriSpeech reads subset.json and manifest.json of dir; the audio
// is in audioDir.
func loadLibriSpeech(dir, audioDir string) (*librispeech, error) {
	var subset struct {
		Utterances []utterance `json:"utterances"`
	}

	err := readJSON(filepath.Join(dir, "subset.json"), &subset)
	if err != nil {
		return nil, err
	}

	var manifest struct {
		Words      int `json:"words"`
		Errors     int `json:"errors"`
		Utterances []struct {
			ID     string `json:"id"`
			Text   string `json:"text"`
			Errors int    `json:"errors"`
		} `json:"utterances"`
	}

	err = readJSON(filepath.Join(dir, "manifest.json"), &manifest)
	if err != nil {
		return nil, err
	}

	if len(subset.Utterances) == 0 || manifest.Words <= 0 {
		return nil, fmt.Errorf("%s: %d utterances, %d reference words; the WER needs both", dir, len(subset.Utterances),
			manifest.Words)
	}

	if len(subset.Utterances) != len(manifest.Utterances) {
		return nil, fmt.Errorf("%s: %d utterances in subset.json, %d in manifest.json", dir, len(subset.Utterances),
			len(manifest.Utterances))
	}

	ls := &librispeech{audioDir: audioDir, words: manifest.Words, engineErrors: manifest.Errors}

	for i, m := range manifest.Utterances {
		u := subset.Utterances[i]
		if u.ID != m.ID {
			return nil, fmt.Errorf("%s: utterance %d is %s in subset.json, %s in manifest.json", dir, i, u.ID, m.ID)
		}

		u.engine, u.engineErrors = m.Text, m.Errors
		ls.utterances = append(ls.utterances, u)
	}

	return ls, nil
}

// wordErrors is the manifest's wer_rule, as whistle's TestEvalLibriSpeech
// applies it: the word edit distance of hyp against ref after eval.Words,
// and ref's word count.
func wordErrors(ref, hyp string) (errs, words int) {
	r := eval.Words(ref)

	return eval.EditDistance(r, eval.Words(hyp)), len(r)
}

// readWAV reads a WAV file, checks it against sha and decodes it to
// 16 kHz mono.
func readWAV(path, sha string) ([]float32, error) {
	raw, err := os.ReadFile(path) // #nosec G304 G703 -- corpus paths from the manifests.
	if err != nil {
		return nil, fmt.Errorf("read clip: %w", err)
	}

	if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != sha {
		return nil, fmt.Errorf("%s: SHA-256 differs from the manifest", path)
	}

	pcm, err := frontend.DecodeWAV(raw)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}

	return pcm, nil
}

func readJSON(path string, v any) error {
	raw, err := os.ReadFile(path) // #nosec G304 G703 -- fixture paths from flags.
	if err != nil {
		return fmt.Errorf("read: %w", err)
	}

	err = json.Unmarshal(raw, v)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}

	return nil
}

// exists reports whether path exists.
func exists(path string) bool {
	_, err := os.Stat(path)

	return !errors.Is(err, os.ErrNotExist)
}
