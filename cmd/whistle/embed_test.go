package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/whistle"
)

// embedJSON is the --format json object.
type embedJSON struct {
	Frames    int         `json:"frames"`
	Dim       int         `json:"dim"`
	FrameMs   int         `json:"frame_ms"`
	Embedding [][]float32 `json:"embedding"`
}

func testEmbedding() whistle.Embedding {
	return whistle.Embedding{
		Values: []float32{0.1, -2.5e-7, 3, float32(math.Pi), -0, 1e30},
		Frames: 2,
		Dim:    3,
	}
}

func TestEmbeddingJSON(t *testing.T) {
	e := testEmbedding()

	var b bytes.Buffer

	err := writeEmbedding(&b, e, formatJSON)
	if err != nil {
		t.Fatal(err)
	}

	want := `{"frames":2,"dim":3,"frame_ms":80,"embedding":[[0.1,-2.5e-07,3],[3.1415927,0,1e+30]]}` + "\n"
	if b.String() != want {
		t.Errorf("json = %s, want %s", b.String(), want)
	}

	var got embedJSON

	err = json.Unmarshal(b.Bytes(), &got)
	if err != nil {
		t.Fatal(err)
	}

	for i := range e.Frames {
		for j, v := range e.Frame(i) {
			if got.Embedding[i][j] != v {
				t.Errorf("[%d][%d] = %v, want %v exactly", i, j, got.Embedding[i][j], v)
			}
		}
	}

	b.Reset()

	err = writeEmbedding(&b, whistle.Embedding{Dim: 512}, formatJSON)
	if err != nil || b.String() != `{"frames":0,"dim":512,"frame_ms":80,"embedding":[]}`+"\n" {
		t.Errorf("no frames: %q, %v", b.String(), err)
	}
}

// readNPY parses a .npy file of version 1.0 with a '<f4' C-order array.
func readNPY(t *testing.T, data []byte) (string, []float32) {
	t.Helper()

	if !bytes.HasPrefix(data, []byte("\x93NUMPY\x01\x00")) {
		t.Fatalf("npy magic: % x", data[:min(8, len(data))])
	}

	n := int(binary.LittleEndian.Uint16(data[8:10]))
	header := string(data[10 : 10+n])

	if (10+n)%64 != 0 || !strings.HasSuffix(header, "\n") {
		t.Errorf("header of %d bytes, %q: want a multiple of 64 ending in a newline", 10+n, header)
	}

	body := data[10+n:]
	values := make([]float32, len(body)/4)

	err := binary.Read(bytes.NewReader(body), binary.LittleEndian, values)
	if err != nil {
		t.Fatal(err)
	}

	return strings.TrimRight(header, " \n"), values
}

func TestEmbeddingNPY(t *testing.T) {
	e := testEmbedding()

	var b bytes.Buffer

	err := writeEmbedding(&b, e, formatNPY)
	if err != nil {
		t.Fatal(err)
	}

	header, values := readNPY(t, b.Bytes())
	if header != "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }" {
		t.Errorf("header %q", header)
	}

	if !equalPCM(values, e.Values) {
		t.Errorf("values %v, want %v", values, e.Values)
	}
}

func TestRootHasEmbed(t *testing.T) {
	cmd, _, err := NewRootCmd().Find([]string{"embed"})
	if err != nil || cmd.Name() != "embed" {
		t.Fatalf("root has no embed subcommand: %v", err)
	}

	for flag, def := range map[string]string{"model": defaultModelPath, "format": "json", "output": ""} {
		if f := cmd.Flags().Lookup(flag); f == nil || f.DefValue != def {
			t.Errorf("--%s: %v, want default %q", flag, f, def)
		}
	}

	_, err = runCmd(t, "embed", "--format", "csv", clipPath)
	if exitCode(err) != exitUsage {
		t.Errorf("--format csv: err = %v, want a usage error", err)
	}

	_, err = runCmd(t, "embed")
	if exitCode(err) != exitUsage {
		t.Errorf("no WAV: err = %v, want a usage error", err)
	}
}

// The embed command against the library's Embed (exactly, through both
// formats) and against the engine's needle_embed of the clip (cosine).
func TestEmbedCmd(t *testing.T) {
	requireModel(t)

	m, err := whistle.Load(cactPath)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = m.Close() })

	want, err := embedWAV(m, clipPath)
	if err != nil {
		t.Fatal(err)
	}

	out, err := runCmd(t, "embed", "--model", cactPath, clipPath)
	if err != nil {
		t.Fatal(err)
	}

	var got embedJSON

	err = json.Unmarshal([]byte(out), &got)
	if err != nil {
		t.Fatal(err)
	}

	if got.Frames != 33 || got.Dim != 512 || got.FrameMs != 80 || len(got.Embedding) != 33 {
		t.Fatalf("json: %d frames of %d, %d ms, %d rows; want 33 of 512, 80 ms", got.Frames, got.Dim, got.FrameMs,
			len(got.Embedding))
	}

	for i, row := range got.Embedding {
		if !equalPCM(row, want.Frame(i)) {
			t.Fatalf("json frame %d differs from Embed", i)
		}
	}

	npy := filepath.Join(t.TempDir(), "clip.npy")

	_, err = runCmd(t, "embed", "--model", cactPath, "--format", "npy", "-o", npy, clipPath)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(npy)
	if err != nil {
		t.Fatal(err)
	}

	header, values := readNPY(t, data)
	if !strings.Contains(header, "'shape': (33, 512)") || !equalPCM(values, want.Values) {
		t.Errorf("npy: header %q, %d values; want Embed's %d", header, len(values), len(want.Values))
	}

	store, err := safetensors.OpenStore("../../testdata/oracle/ls-other-3080-5040-0018.embed.safetensors",
		safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}

	defer store.Close()

	engine, err := store.Tensor("embed")
	if err != nil {
		t.Fatal(err)
	}

	if len(engine.Data) != len(values) {
		t.Fatalf("%d values, the engine's %d", len(values), len(engine.Data))
	}

	for i := range 33 {
		if c := cosine(values[i*512:(i+1)*512], engine.Data[i*512:(i+1)*512]); c < 0.99 {
			t.Errorf("frame %d: cosine %.6f against needle_embed, want ≥ 0.99", i, c)
		}
	}
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64

	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}

	return ab / math.Sqrt(aa*bb)
}

func TestEmbedCmdLimits(t *testing.T) {
	requireModel(t)

	long := filepath.Join(t.TempDir(), "long.wav")
	writeSilentWAV(t, long, whistle.MaxSamples+1)

	out, err := runCmd(t, "embed", "--model", cactPath, long)
	if exitCode(err) != exitFailure || !strings.Contains(err.Error(), "30 s") || out != "" {
		t.Errorf("31 s: err = %v, stdout %d bytes; want a failure naming 30 s and no output", err, len(out))
	}

	empty := filepath.Join(t.TempDir(), "empty.wav")
	writeSilentWAV(t, empty, 0)

	_, err = runCmd(t, "embed", "--model", cactPath, empty)
	if !errors.Is(err, whistle.ErrNoAudio) {
		t.Errorf("empty WAV: err = %v, want ErrNoAudio", err)
	}
}
