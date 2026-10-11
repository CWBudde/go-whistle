//go:build oracle

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"strconv"
)

// embedInputs are the texts needle_embed embeds; embed.safetensors holds their vectors as rows, in this order.
var embedInputs = []string{
	"set a timer for 10 minutes",
	"turn off the kitchen lights",
	"Wie wird das Wetter morgen in Berlin?",
}

// embedHead is how many leading values per vector embed.json lists, for reading without a safetensors reader.
const embedHead = 8

// embedFixture is testdata/text/engine/embed.json.
type embedFixture struct {
	header

	Vectors      string       `json:"vectors"`
	Dim          int          `json:"dim"`
	Inputs       []embedEntry `json:"inputs"`
	SameOnRepeat bool         `json:"same_on_repeat"`
	SameAfter    bool         `json:"same_after_init"`
}

// embedEntry summarises one vector: its text, L2 norm and first values.
type embedEntry struct {
	Text string        `json:"text"`
	Norm json.Number   `json:"norm"`
	Head []json.Number `json:"head"`
}

// cmdEmbed records needle_embed of the embedInputs right after needle_load, checks that a repeat and a call after
// needle_init (a tool set and the corpus's system text) give the same floats, and writes embed.json and
// embed.safetensors.
func cmdEmbed(_ context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	lib, err := openLibrary(cfg.engineDir, o.needle3Path())
	if err != nil {
		return err
	}

	first, err := embedAll(lib)
	if err != nil {
		return err
	}

	again, err := embedAll(lib)
	if err != nil {
		return err
	}

	_, err = lib.session(o.corpus.System, o.corpus.tools(o.corpus.toolsetNames()[0]))
	if err != nil {
		return err
	}

	after, err := embedAll(lib)
	if err != nil {
		return err
	}

	dim := len(first[0])
	fx := embedFixture{
		header:  o.header("libneedle3: needle_embed(text, NULL, 0, out, dim) after needle_load"),
		Vectors: "embed.safetensors: embed [inputs, dim] F32", Dim: dim,
		SameOnRepeat: equalVectors(first, again), SameAfter: equalVectors(first, after),
	}
	flat := make([]float32, 0, len(first)*dim)

	for i, v := range first {
		if len(v) != dim {
			return fmt.Errorf("needle_embed gave %d and %d floats", dim, len(v))
		}

		e := embedEntry{Text: embedInputs[i], Norm: number(norm(v))}
		for _, x := range v[:min(embedHead, dim)] {
			e.Head = append(e.Head, number(float64(x)))
		}

		fx.Inputs = append(fx.Inputs, e)
		flat = append(flat, v...)
		_, _ = fmt.Fprintf(log, "%-40q dim %d norm %s\n", e.Text, dim, e.Norm)
	}

	var buf bytes.Buffer

	err = writeSafetensors(&buf, "embed", []int{len(first), dim}, flat, map[string]string{
		"engine_version": o.pin["engine_version"], "model_sha256": o.needle3.SHA256,
	})
	if err != nil {
		return err
	}

	err = writeFile(filepath.Join(cfg.outDir, "embed.safetensors"), buf.Bytes())
	if err != nil {
		return err
	}

	return writeJSON(cfg.outDir, "embed.json", fx)
}

func embedAll(lib *library) ([][]float32, error) {
	out := make([][]float32, 0, len(embedInputs))

	for _, t := range embedInputs {
		v, err := lib.embedText(t)
		if err != nil {
			return nil, fmt.Errorf("%q: %w", t, err)
		}

		out = append(out, v)
	}

	return out, nil
}

func equalVectors(a, b [][]float32) bool {
	return slices.EqualFunc(a, b, slices.Equal[[]float32])
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}

	return math.Sqrt(s)
}

// number is x as the shortest float32 literal that reads back to the same float32.
func number(x float64) json.Number {
	return json.Number(strconv.FormatFloat(x, 'g', -1, 32))
}

// writeSafetensors writes a single F32 tensor in the safetensors format: a little-endian u64 header length, the
// JSON header padded with spaces to a multiple of 8, then the raw data.
func writeSafetensors(w io.Writer, name string, shape []int, data []float32, meta map[string]string) error {
	n := 1
	for _, d := range shape {
		n *= d
	}

	if n != len(data) {
		return fmt.Errorf("shape %v holds %d values, got %d", shape, n, len(data))
	}

	hdr := map[string]any{
		name: map[string]any{"dtype": "F32", "shape": shape, "data_offsets": []int{0, 4 * len(data)}},
	}
	if len(meta) > 0 {
		hdr["__metadata__"] = meta
	}

	js, err := json.Marshal(hdr)
	if err != nil {
		return fmt.Errorf("safetensors header: %w", err)
	}

	if pad := len(js) % 8; pad != 0 {
		js = append(js, bytes.Repeat([]byte(" "), 8-pad)...)
	}

	var b bytes.Buffer

	b.Grow(8 + len(js) + 4*len(data))
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(js)))
	b.Write(js)
	_ = binary.Write(&b, binary.LittleEndian, data)

	_, err = w.Write(b.Bytes())
	if err != nil {
		return fmt.Errorf("write safetensors: %w", err)
	}

	return nil
}
