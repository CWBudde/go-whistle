//go:build oracle

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/cwbudde/go-whistle/internal/hub"
)

// embedWidth is the float count per needle_embed speech frame (docs/FINDINGS.md F-001).
const embedWidth = 512

// timingKeys are the needle CLI fields that vary from run to run; they are dropped from the fixtures.
var timingKeys = []string{"ttft_ms", "decode_tps"}

type runConfig struct {
	audioDir  string // corpus: manifest.json + <id>.wav
	outDir    string // testdata/oracle
	modelDir  string // whistle download --dir
	engineDir string // engine/<platform>
}

// oracleManifest is testdata/oracle/manifest.json: what produced the fixtures.
type oracleManifest struct {
	Engine     map[string]string `json:"engine"`
	Model      oracleModel       `json:"model"`
	EmbedWidth int               `json:"embed_width"`
	Clips      []string          `json:"clips"`
}

type oracleModel struct {
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	File     string `json:"file"`
	SHA256   string `json:"sha256"`
}

// cmdRun records the engine's transcript and speech embedding for every corpus clip.
func cmdRun(ctx context.Context, cfg runConfig, log io.Writer) error {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	model, err := pinnedModel(cfg.modelDir)
	if err != nil {
		return err
	}

	pin, err := readPin(filepath.Join(cfg.engineDir, "PIN"))
	if err != nil {
		return fmt.Errorf("%w (run `just fetch-engine`)", err)
	}

	cact, err := os.ReadFile(filepath.Join(cfg.modelDir, model.File)) // #nosec G304 -- modelDir is a flag.
	if err != nil {
		return fmt.Errorf("read model: %w", err)
	}

	eng, err := openEngine(cfg.engineDir, cact)
	if err != nil {
		return err
	}

	err = os.MkdirAll(cfg.outDir, 0o750)
	if err != nil {
		return fmt.Errorf("create %s: %w", cfg.outDir, err)
	}

	meta := map[string]string{
		"engine_version":  pin["engine_version"],
		"engine_revision": pin["revision"],
		"model_sha256":    model.SHA256,
	}

	ids := make([]string, 0, len(corpus.Clips))

	for _, c := range corpus.Clips {
		err := recordClip(ctx, cfg, eng, model, c, meta, log)
		if err != nil {
			return fmt.Errorf("%s: %w", c.ID, err)
		}

		ids = append(ids, c.ID)
	}

	js, err := json.MarshalIndent(oracleManifest{Engine: pin, Model: model, EmbedWidth: embedWidth, Clips: ids}, "", "  ")
	if err != nil {
		return fmt.Errorf("oracle manifest: %w", err)
	}

	return writeFile(filepath.Join(cfg.outDir, "manifest.json"), append(js, '\n'))
}

func recordClip(ctx context.Context, cfg runConfig, eng *engine, model oracleModel, c clip,
	meta map[string]string, log io.Writer,
) error {
	wav := filepath.Join(cfg.audioDir, c.ID+".wav")

	raw, err := readClip(wav, c.SHA256)
	if err != nil {
		return err
	}

	pcm, rate, err := readWAV(raw)
	if err != nil {
		return err
	}

	if rate != sampleRate {
		return fmt.Errorf("%d Hz, want %d", rate, sampleRate)
	}

	tr, err := transcribe(ctx, filepath.Join(cfg.engineDir, "needle"), filepath.Join(cfg.modelDir, model.File), wav)
	if err != nil {
		return err
	}

	err = writeFile(filepath.Join(cfg.outDir, c.ID+".json"), tr)
	if err != nil {
		return err
	}

	emb, err := eng.embedSpeech(pcm)
	if err != nil {
		return err
	}

	if len(emb)%embedWidth != 0 {
		return fmt.Errorf("needle_embed returned %d floats, not a multiple of %d", len(emb), embedWidth)
	}

	frames := len(emb) / embedWidth

	m := map[string]string{"clip_sha256": c.SHA256}
	maps.Copy(m, meta)

	var buf bytes.Buffer

	err = writeSafetensors(&buf, "embed", []int{frames, embedWidth}, emb, m)
	if err != nil {
		return err
	}

	err = writeFile(filepath.Join(cfg.outDir, c.ID+".embed.safetensors"), buf.Bytes())
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(log, "%-36s %3d frames  %s\n", c.ID, frames, oneLine(tr))

	return nil
}

// readClip reads a corpus clip and checks it against the digest in the corpus manifest, so that fixtures are never
// recorded from, and attributed to, audio other than the pinned clip.
func readClip(path, wantSHA256 string) ([]byte, error) {
	raw, err := os.ReadFile(path) // #nosec G304 G703 -- ids are checked by readCorpus.
	if err != nil {
		return nil, fmt.Errorf("read clip: %w", err)
	}

	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != wantSHA256 {
		return nil, fmt.Errorf("clip sha256 %s, corpus manifest pins %s (re-run `just oracle-clips`)", got, wantSHA256)
	}

	return raw, nil
}

// transcribe runs the needle CLI on wav and returns its JSON with the timing fields removed, keys sorted and
// indented, so that a re-run of the same engine diffs clean.
func transcribe(ctx context.Context, needle, model, wav string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, needle, "--model", model, "--audio", wav, "--audio-word-timestamps")

	cmd.Env = append(os.Environ(), "NEEDLE_TELEMETRY=0", "DO_NOT_TRACK=1")

	var stderr bytes.Buffer

	cmd.Stderr = &stderr

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("needle: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	var fields map[string]json.RawMessage

	err = json.Unmarshal(out, &fields)
	if err != nil {
		return nil, fmt.Errorf("needle output is not a JSON object: %w: %q", err, out)
	}

	if _, ok := fields["text"]; !ok {
		return nil, fmt.Errorf("needle output has no text: %q", out)
	}

	for _, k := range timingKeys {
		delete(fields, k)
	}

	js, err := json.MarshalIndent(fields, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("re-encode needle output: %w", err)
	}

	return append(js, '\n'), nil
}

func oneLine(js []byte) string {
	var v struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	}

	_ = json.Unmarshal(js, &v)

	return v.Language + " " + v.Text
}

// pinnedModel checks dir/whistle.cact against the pinned hub manifest.
func pinnedModel(dir string) (oracleModel, error) {
	m, err := hub.Pinned()
	if err != nil {
		return oracleModel{}, fmt.Errorf("pinned manifest: %w", err)
	}

	for _, f := range m.Files {
		if f.Local != "whistle.cact" {
			continue
		}

		sum, err := fileSHA256(filepath.Join(dir, f.Local))
		if err != nil {
			return oracleModel{}, fmt.Errorf("%w (run `whistle download`)", err)
		}

		if sum != f.SHA256 {
			return oracleModel{}, fmt.Errorf("%s: sha256 %s, pinned %s (run `whistle download`)", f.Local, sum, f.SHA256)
		}

		return oracleModel{Repo: m.Repo, Revision: m.Revision, File: f.Local, SHA256: sum}, nil
	}

	return oracleModel{}, errors.New("the pinned manifest has no whistle.cact")
}

// readPin parses engine/<platform>/PIN (key=value lines, written by scripts/fetch-engine.sh).
func readPin(p string) (map[string]string, error) {
	f, err := os.Open(p) // #nosec G304 -- the engine dir is a flag.
	if err != nil {
		return nil, fmt.Errorf("engine pin: %w", err)
	}

	defer func() { _ = f.Close() }()

	out := map[string]string{}
	sc := bufio.NewScanner(f)

	for sc.Scan() {
		k, v, ok := strings.Cut(strings.TrimSpace(sc.Text()), "=")
		if ok {
			out[k] = v
		}
	}

	err = sc.Err()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", p, err)
	}

	return out, nil
}

func writeFile(p string, b []byte) error {
	err := os.WriteFile(p, b, 0o600)
	if err != nil {
		return fmt.Errorf("write %s: %w", p, err)
	}

	return nil
}
