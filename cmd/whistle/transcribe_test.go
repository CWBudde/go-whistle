package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/whistle"
)

const (
	cactPath = "../../models/whistle/whistle.cact"
	clipPath = "../../testdata/audio/ls-other-3080-5040-0018.wav"
	clipText = "Directed for your master."
)

func requireModel(t *testing.T) {
	t.Helper()

	_, err := os.Stat(cactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", cactPath)
	}
}

func TestRootHasTranscribe(t *testing.T) {
	cmd, _, err := NewRootCmd().Find([]string{"transcribe"})
	if err != nil || cmd.Name() != "transcribe" {
		t.Fatalf("root has no transcribe subcommand: %v", err)
	}

	for flag, def := range map[string]string{
		"model":    filepath.Join("models", "whistle", "whistle.cact"),
		"language": "",
		"json":     "false",
	} {
		f := cmd.Flags().Lookup(flag)
		if f == nil {
			t.Errorf("no --%s flag", flag)

			continue
		}

		if f.DefValue != def {
			t.Errorf("--%s default = %q, want %q", flag, f.DefValue, def)
		}
	}
}

func TestTranscribeCmdNeedsOneWAV(t *testing.T) {
	_, err := runCmd(t, "transcribe")
	if err == nil {
		t.Error("transcribe without a WAV: no error")
	}
}

func TestTranscribeCmdText(t *testing.T) {
	requireModel(t)

	out, err := runCmd(t, "transcribe", "--model", cactPath, clipPath)
	if err != nil {
		t.Fatal(err)
	}

	if out != clipText+"\n" {
		t.Errorf("stdout = %q, want %q", out, clipText+"\n")
	}
}

func TestTranscribeCmdJSON(t *testing.T) {
	requireModel(t)

	out, err := runCmd(t, "transcribe", "--model", cactPath, "--json", "--language", "en", clipPath)
	if err != nil {
		t.Fatal(err)
	}

	shape := regexp.MustCompile(`^\{"text":"[^"]*","language":"en","ttft_ms":\d+\.\d,"decode_tps":\d+\.\d\}\n$`)
	if !shape.MatchString(out) {
		t.Errorf("stdout %q does not have the engine's shape", out)
	}

	var res whistle.Result

	err = json.Unmarshal([]byte(out), &res)
	if err != nil {
		t.Fatal(err)
	}

	if res.Text != clipText || res.Language != "en" || res.TTFT <= 0 {
		t.Errorf("result = %+v", res)
	}
}

func TestTranscribeCmdLimit(t *testing.T) {
	requireModel(t)

	path := filepath.Join(t.TempDir(), "long.wav")
	writeSilentWAV(t, path, whistle.MaxSamples+1)

	out, err := runCmd(t, "transcribe", "--model", cactPath, path)
	if err == nil || !strings.Contains(err.Error(), "30 s") {
		t.Errorf("err = %v, want an error naming 30 s", err)
	}

	if out != "" {
		t.Errorf("stdout = %q, want empty on failure", out)
	}
}

func TestTranscribeCmdBadLanguage(t *testing.T) {
	requireModel(t)

	_, err := runCmd(t, "transcribe", "--model", cactPath, "--language", "xx", clipPath)
	if err == nil || !strings.Contains(err.Error(), `"xx"`) {
		t.Errorf("err = %v, want an error naming the language", err)
	}
}

// writeSilentWAV writes n zero samples as 16-bit mono 16 kHz PCM.
func writeSilentWAV(t *testing.T, path string, n int) {
	t.Helper()

	size := uint32(2 * n) // #nosec G115 -- a 30 s test clip.
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, 36+size)
	b = append(b, "WAVEfmt "...)

	// fmt: chunk size, PCM | mono, rate, byte rate, block align | bits per sample.
	for _, v := range []uint32{16, 1 | 1<<16, whistle.SampleRate, 2 * whistle.SampleRate, 2 | 16<<16} {
		b = binary.LittleEndian.AppendUint32(b, v)
	}

	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, size)
	b = append(b, make([]byte, size)...)

	err := os.WriteFile(path, b, 0o600)
	if err != nil {
		t.Fatal(err)
	}
}
