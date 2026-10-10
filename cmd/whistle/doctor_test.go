package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/hub"
	"github.com/cwbudde/go-whistle/whistle"
)

func TestRootHasDoctor(t *testing.T) {
	cmd, _, err := NewRootCmd().Find([]string{"doctor"})
	if err != nil || cmd.Name() != "doctor" {
		t.Fatalf("root has no doctor subcommand: %v", err)
	}

	for flag, def := range map[string]string{"model": defaultModelPath, "clip": defaultClip} {
		if f := cmd.Flags().Lookup(flag); f == nil || f.DefValue != def {
			t.Errorf("--%s: %v, want default %q", flag, f, def)
		}
	}
}

func TestDoctorMissingModel(t *testing.T) {
	out, err := runCmd(t, "doctor", "--model", filepath.Join(t.TempDir(), "whistle.cact"))
	if exitCode(err) != exitFailure {
		t.Errorf("err = %v, want exit code 1", err)
	}

	if !strings.Contains(out, "file       missing  FAIL") || !strings.Contains(out, "whistle download") ||
		strings.Contains(out, "\nload ") {
		t.Errorf("stdout:\n%s", out)
	}
}

// A file that is not the pinned model fails the format, digest and load
// checks; a corrupt sibling fails the files check.
func TestDoctorWrongFile(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "whistle.cact")

	err := os.WriteFile(model, []byte("not a model file at all"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(filepath.Join(dir, "config.json"), []byte("{}"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	out, err := runCmd(t, "doctor", "--model", model)
	if exitCode(err) != exitFailure || !strings.Contains(err.Error(), "4 check(s) failed") {
		t.Errorf("err = %v, want 4 failed checks", err)
	}

	for _, want := range []string{
		"format     unknown  FAIL",
		"sha256     ",
		"the pinned whistle.cact is b6e02f04",
		"files      whistle.safetensors absent, config.json differs  FAIL",
		"load       ",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout lacks %q:\n%s", want, out)
		}
	}
}

func TestDoctorPinnedFile(t *testing.T) {
	m, err := hub.Pinned()
	if err != nil {
		t.Fatal(err)
	}

	d := doctor{pinned: m}

	f, ok := d.pinnedFile("renamed.bin", m.Files[2].SHA256)
	if !ok || f.Local != "config.json" {
		t.Errorf("by digest: %v, %v", f, ok)
	}

	f, ok = d.pinnedFile("whistle.cact", "00")
	if !ok || f.Local != "whistle.cact" {
		t.Errorf("by name: %v, %v", f, ok)
	}

	_, ok = d.pinnedFile("other.bin", "00")
	if ok {
		t.Error("an unknown file matched")
	}
}

func TestFileSHA256(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x")

	err := os.WriteFile(path, []byte("whistle"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	sum := sha256.Sum256([]byte("whistle"))

	got, err := fileSHA256(path)
	if err != nil || got != hex.EncodeToString(sum[:]) {
		t.Errorf("fileSHA256 = %s, %v", got, err)
	}
}

func TestModelFormat(t *testing.T) {
	dir := t.TempDir()

	for name, c := range map[string]struct {
		data []byte
		want string
	}{
		"a.cact":        {[]byte{0x84, 0x2a, 0xe1, 0x05, 0, 0, 0, 0, 0}, ".cact"},
		"a.safetensors": {[]byte("P\x00\x00\x00\x00\x00\x00\x00{\"x\":1}"), ".safetensors"},
		"a.bin":         {[]byte("123456789"), "unknown"},
		"short":         {[]byte("123"), ""},
	} {
		path := filepath.Join(dir, name)

		err := os.WriteFile(path, c.data, 0o600)
		if err != nil {
			t.Fatal(err)
		}

		got, err := modelFormat(path)
		if !strings.HasPrefix(got, c.want) || (err != nil) != (c.want != ".cact" && c.want != ".safetensors") {
			t.Errorf("%s: %q, %v; want %q", name, got, err, c.want)
		}
	}
}

func TestKernels(t *testing.T) {
	got := kernels()

	switch {
	case puregoBuild:
		if got != "pure Go (purego build)" {
			t.Errorf("purego build: %q", got)
		}
	case runtime.GOARCH == "arm64":
		if !strings.HasPrefix(got, "NEON") {
			t.Errorf("arm64: %q", got)
		}
	case runtime.GOARCH == "amd64":
		if !strings.Contains(got, "AVX2") && !strings.Contains(got, "no AVX2") {
			t.Errorf("amd64: %q", got)
		}
	}
}

func TestWeightsMode(t *testing.T) {
	if got := weightsMode("m/whistle.cact"); !strings.HasPrefix(got, "packed") {
		t.Errorf(".cact: %q", got)
	}

	if got := weightsMode("m/whistle.safetensors"); !strings.HasPrefix(got, "float32") {
		t.Errorf(".safetensors: %q", got)
	}
}

// The tone burst has level contrast: silent quarters and a -12 dBFS tone.
func TestToneBurst(t *testing.T) {
	x := toneBurst()
	if len(x) != 2*whistle.SampleRate {
		t.Fatalf("%d samples", len(x))
	}

	var peak float32

	for i, v := range x {
		if i%8000 >= 4000 && v != 0 {
			t.Fatalf("sample %d = %v in a silent quarter", i, v)
		}

		peak = max(peak, v)
	}

	if peak < 0.24 || peak > 0.25 {
		t.Errorf("peak %v, want 0.25", peak)
	}
}

// doctor on the real model: every check passes, the fixture clip included.
func TestDoctorPasses(t *testing.T) {
	requireModel(t)

	out, err := runCmd(t, "doctor", "--model", cactPath, "--clip", clipPath)
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}

	for _, re := range []string{
		`(?m)^format     \.cact `,
		`(?m)^sha256     b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb = pinned Cactus-Compute/whistle@b358ddad whistle\.cact$`,
		`(?m)^weights    packed`,
		`(?m)^go         go1\.\d+`,
		`(?m)^kernels    `,
		`(?m)^workers    GOMAXPROCS \d+, \d+ CPUs$`,
		`(?m)^load       ok in .*; languages en de fr es it nl pl$`,
		`(?m)^self-test  2 s tone burst: embedding 25×512, transcript \[`,
		`(?m)^clip       .*ls-other-3080-5040-0018\.wav: \[en\] "Directed for your master\." in `,
		`(?m)^all checks passed$`,
	} {
		if !regexp.MustCompile(re).MatchString(out) {
			t.Errorf("stdout does not match %s:\n%s", re, out)
		}
	}

	if strings.Contains(out, "FAIL") {
		t.Errorf("a check failed:\n%s", out)
	}
}

// A clip named like the fixture must give its transcript; another clip
// only has to transcribe.
func TestDoctorClip(t *testing.T) {
	requireModel(t)

	other := "../../testdata/audio/ls-other-3080-5040-0011.wav"

	out, err := runCmd(t, "doctor", "--model", cactPath, "--clip", other)
	if err != nil || !strings.Contains(out, `"Here are some verses of`) {
		t.Errorf("another clip: %v\n%s", err, out)
	}

	wrong := filepath.Join(t.TempDir(), filepath.Base(defaultClip))

	raw, err := os.ReadFile(other)
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(wrong, raw, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	out, err = runCmd(t, "doctor", "--model", cactPath, "--clip", wrong)
	if exitCode(err) != exitFailure || !strings.Contains(out, `FAIL: want "Directed for your master."`) {
		t.Errorf("the fixture's name on other audio: %v\n%s", err, out)
	}

	out, err = runCmd(t, "doctor", "--model", cactPath, "--clip", filepath.Join(t.TempDir(), "none.wav"))
	if err != nil || !strings.Contains(out, "none.wav not found, skipped") {
		t.Errorf("no clip: %v\n%s", err, out)
	}
}
