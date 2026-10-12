package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/hub"
	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
	"golang.org/x/sys/cpu"
)

// defaultClip is the fixture clip doctor transcribes when it is there,
// with the transcript the engine and the port give for it.
var (
	defaultClip     = filepath.Join("testdata", "audio", "ls-other-3080-5040-0018.wav")
	defaultClipText = "Directed for your master."
)

func newDoctorCmd() *cobra.Command {
	var model, clip string

	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check the model file, the build and a self-test transcription",
		Long: "Check the install: the model file (path, size, format, SHA-256 against the pinned Cactus-Compute/whistle\n" +
			"revision, and the other pinned files next to it), the weights mode, the Go version and platform, the\n" +
			"assembly kernels in use (NEON on arm64, AVX2 on amd64, none in a purego build) and the worker count.\n" +
			"Then it loads the model and runs a self-test: a built-in 2 s tone burst through Embed and Transcribe,\n" +
			"and --clip, by default the fixture clip " + defaultClip + " when it is there (that clip, wherever it\n" +
			"lies, must give \"" + defaultClipText + "\").\n\n" +
			"Exits with 1 when a check fails.",
		Args: usageArgs(cobra.NoArgs),
		RunE: func(cmd *cobra.Command, _ []string) error {
			m, err := hub.Pinned()
			if err != nil {
				return err
			}

			d := doctor{out: cmd.OutOrStdout(), pinned: m}
			if filepath.Base(clip) == filepath.Base(defaultClip) {
				d.clipText = defaultClipText
			}

			return d.run(model, clip)
		},
	}

	cmd.Flags().StringVar(&model, "model", defaultModelPath, "Model file: whistle.cact or whistle.safetensors")
	cmd.Flags().StringVar(&clip, "clip", defaultClip, "WAV file to transcribe as the last check; skipped when missing")

	return cmd
}

// doctor runs the checks and prints a line for each.
type doctor struct {
	out      io.Writer
	pinned   hub.Manifest
	clipText string // the transcript the clip must give; "" accepts any
	failed   int
	err      error // the first write error
}

// line prints one check; a non-nil err marks it failed.
func (d *doctor) line(name, value string, err error) {
	status := ""

	if err != nil {
		d.failed++
		status = "  FAIL: " + strings.TrimPrefix(err.Error(), "whistle: ")
	}

	_, werr := fmt.Fprintf(d.out, "%-10s %s%s\n", name, value, status)
	if werr != nil && d.err == nil {
		d.err = fmt.Errorf("write result: %w", werr)
	}
}

func (d *doctor) run(modelPath, clip string) error {
	d.line("model", modelPath, nil)

	ok, format := d.checkFile(modelPath)
	d.checkSiblings(modelPath)

	d.line("weights", weightsMode(format), nil)
	d.line("go", fmt.Sprintf("%s %s/%s", runtime.Version(), runtime.GOOS, runtime.GOARCH), nil)
	d.line("kernels", kernels(), nil)
	d.line("workers", fmt.Sprintf("GOMAXPROCS %d, %d CPUs", runtime.GOMAXPROCS(0), runtime.NumCPU()), nil)

	if ok {
		d.selfTest(modelPath, clip)
	}

	if d.err != nil {
		return d.err
	}

	if d.failed > 0 {
		return fmt.Errorf("doctor: %d check(s) failed", d.failed)
	}

	_, err := fmt.Fprintln(d.out, "all checks passed")
	if err != nil {
		return fmt.Errorf("write result: %w", err)
	}

	return nil
}

// checkFile reports the model's size, format and digest, and returns the
// format (modelFormat's); false when it is missing or not a model file.
func (d *doctor) checkFile(path string) (bool, string) {
	info, err := os.Stat(path)
	if err != nil {
		d.line("file", "missing", fmt.Errorf("%w; run `whistle download`", err))

		return false, ""
	}

	d.line("size", fmt.Sprintf("%d bytes (%.1f MB)", info.Size(), float64(info.Size())/1e6), nil)

	format, err := modelFormat(path)
	d.line("format", format, err)

	sum, err := fileSHA256(path)
	if err != nil {
		d.line("sha256", "", err)

		return false, format
	}

	f, found := d.pinnedFile(filepath.Base(path), sum)

	switch {
	case !found:
		d.line("sha256", sum, fmt.Errorf("not a file of %s@%.8s; run `whistle download`", d.pinned.Repo, d.pinned.Revision))
	case f.SHA256 != sum:
		d.line("sha256", sum, fmt.Errorf("the pinned %s is %s; run `whistle download`", f.Local, f.SHA256))
	default:
		d.line("sha256", fmt.Sprintf("%s = pinned %s@%.8s %s", sum, d.pinned.Repo, d.pinned.Revision, f.Path), nil)
	}

	return true, format
}

// pinnedFile is the pinned file named name, or the one whose digest is sum.
func (d *doctor) pinnedFile(name, sum string) (hub.File, bool) {
	for _, f := range d.pinned.Files {
		if f.SHA256 == sum {
			return f, true
		}
	}

	for _, f := range d.pinned.Files {
		if f.Local == name {
			return f, true
		}
	}

	return hub.File{}, false
}

// checkSiblings verifies the other pinned files in the model's directory;
// a missing one is fine, a different one is not.
func (d *doctor) checkSiblings(modelPath string) {
	dir := filepath.Dir(modelPath)

	var parts []string

	var bad []string

	for _, f := range d.pinned.Files {
		path := filepath.Join(dir, f.Local)
		if filepath.Clean(path) == filepath.Clean(modelPath) {
			continue
		}

		_, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			parts = append(parts, f.Local+" absent")

			continue
		}

		sum, err := fileSHA256(path)
		if err != nil || sum != f.SHA256 {
			bad = append(bad, f.Local)
			parts = append(parts, f.Local+" differs")

			continue
		}

		parts = append(parts, f.Local+" ok")
	}

	var err error
	if len(bad) > 0 {
		err = fmt.Errorf("%s not the pinned file(s); run `whistle download --dir %s`", strings.Join(bad, ", "), dir)
	}

	d.line("files", strings.Join(parts, ", "), err)
}

// selfTest loads the model, runs a tone burst through Embed and Transcribe
// and transcribes clip when it is there.
func (d *doctor) selfTest(modelPath, clip string) {
	t0 := time.Now()

	m, err := whistle.Load(modelPath)
	if err != nil {
		d.line("load", "", err)

		return
	}

	defer func() { _ = m.Close() }()

	d.line("load", fmt.Sprintf("ok in %v; languages %s", time.Since(t0).Round(time.Millisecond),
		strings.Join(m.Languages(), " ")), nil)

	t0 = time.Now()
	value, err := toneTest(m)
	d.line("self-test", fmt.Sprintf("2 s tone burst: %s in %v", value, time.Since(t0).Round(time.Millisecond)), err)

	_, err = os.Stat(clip)
	if errors.Is(err, os.ErrNotExist) {
		d.line("clip", clip+" not found, skipped", nil)

		return
	}

	t0 = time.Now()

	res, err := m.TranscribeWAV(clip, whistle.TranscribeOptions{})
	if err == nil && d.clipText != "" && res.Text != d.clipText {
		err = fmt.Errorf("want %q", d.clipText)
	}

	d.line("clip", fmt.Sprintf("%s: [%s] %q in %v", clip, res.Language, res.Text, time.Since(t0).Round(time.Millisecond)),
		err)
}

// toneTest embeds and transcribes toneBurst: the embedding must have the
// frame count of 2 s and finite values, the transcript a known language.
func toneTest(m *whistle.Model) (string, error) {
	pcm := toneBurst()

	e, err := m.Embed(pcm)
	if err != nil {
		return "", err
	}

	if want := 25; e.Frames != want || e.Dim != 512 || len(e.Values) != e.Frames*e.Dim {
		return "", fmt.Errorf("embedding %d×%d, want %d×512", e.Frames, e.Dim, want)
	}

	for _, v := range e.Values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return "", errors.New("embedding holds a non-finite value")
		}
	}

	res, err := m.Transcribe(pcm, whistle.TranscribeOptions{WordTimestamps: true})
	if err != nil {
		return "", err
	}

	_, err = res.MarshalJSON()
	if err != nil {
		return "", err
	}

	value := fmt.Sprintf("embedding 25×512, transcript [%s] %q", res.Language, res.Text)

	if res.Text != "" && !slices.Contains(m.Languages(), res.Language) {
		return value, fmt.Errorf("language %q", res.Language)
	}

	return value, nil
}

// toneBurst is 2 s of a 440 Hz tone at -12 dBFS, on for the first 0.25 s of
// every half second: level contrast in the speech band, so the silence gate
// lets it through and the decoder runs.
func toneBurst() []float32 {
	pcm := make([]float32, 2*whistle.SampleRate)

	for i := range pcm {
		if i%(whistle.SampleRate/2) < whistle.SampleRate/4 {
			pcm[i] = float32(0.25 * math.Sin(2*math.Pi*440*float64(i)/whistle.SampleRate))
		}
	}

	return pcm
}

// modelFormat names the file's format from its first bytes.
func modelFormat(path string) (string, error) {
	f, err := os.Open(path) // #nosec G304 -- the user names the file.
	if err != nil {
		return "", err
	}

	defer func() { _ = f.Close() }()

	var head [9]byte

	_, err = io.ReadFull(f, head[:])
	if err != nil {
		return "", fmt.Errorf("read header: %w", err)
	}

	switch {
	case binary.LittleEndian.Uint32(head[:4]) == cact.Tag:
		return ".cact (2/4-bit CQ weights with the tokenizer)", nil
	case head[8] == '{':
		return ".safetensors (float32; the tokenizer comes from whistle.cact next to it)", nil
	default:
		return "unknown", errors.New("neither .cact nor .safetensors")
	}
}

// weightsMode is how whistle.Load runs weights of the given format
// (modelFormat's): weights.Open tells the formats by their first bytes, not
// by the file name.
func weightsMode(format string) string {
	switch {
	case strings.HasPrefix(format, ".cact"):
		return "packed: CQ matrices straight from the file (whistle.Load's default; WithFloatWeights dequantises)"
	case strings.HasPrefix(format, ".safetensors"):
		return "float32 (a safetensors checkpoint is float32)"
	default:
		return "unknown (not a model file)"
	}
}

// kernels names the assembly kernels this build and CPU run, mirroring the
// build tags and CPU checks of github.com/cwbudde/algo-tensor/tensor,
// internal/cq and internal/needle.
func kernels() string {
	if puregoBuild {
		return "pure Go (purego build)"
	}

	switch runtime.GOARCH {
	case "arm64":
		return "NEON: GEMM, dot, axpy, CQ LUT GEMV, softmax exp"
	case "amd64":
		var parts []string

		if cpu.X86.HasAVX2 {
			parts = append(parts, "AVX2: CQ LUT GEMV")
		}

		// algo-tensor v0.4.0 runs MatMulTransB in assembly on the same
		// AVX2 + FMA check as dot and axpy (PR #60 review).
		if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
			return strings.Join(append(parts, "AVX2+FMA: GEMM, dot, axpy, softmax exp"), "; ")
		}

		if len(parts) == 0 {
			return "pure Go (no AVX2 on this CPU)"
		}

		return strings.Join(parts, "; ") + "; GEMM in Go"
	default:
		return "pure Go (no assembly for " + runtime.GOARCH + ")"
	}
}

// fileSHA256 is the hex SHA-256 of the file at path.
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path) // #nosec G304 -- the user names the file.
	if err != nil {
		return "", err
	}

	defer func() { _ = f.Close() }()

	h := sha256.New()

	_, err = io.Copy(h, f)
	if err != nil {
		return "", fmt.Errorf("hash %s: %w", path, err)
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
