//go:build oracle

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

// sampleRate is the only rate Whistle accepts (config.json "sample_rate").
const sampleRate = 16000

// writeWAV16 writes pcm as a canonical 44-byte-header, 16-bit mono 16 kHz RIFF WAV.
func writeWAV16(w io.Writer, pcm []int16) error {
	n := uint32(2 * len(pcm)) // #nosec G115 -- clips are a few seconds long.

	var b bytes.Buffer

	b.Grow(44 + int(n))
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, 36+n)
	b.WriteString("WAVEfmt ")
	// fmt: chunk size, PCM, mono, rate, byte rate, block align, bits per sample.
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(sampleRate), uint32(2 * sampleRate), uint16(2), uint16(16)} {
		_ = binary.Write(&b, binary.LittleEndian, v)
	}

	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, n)
	_ = binary.Write(&b, binary.LittleEndian, pcm)

	_, err := w.Write(b.Bytes())
	if err != nil {
		return fmt.Errorf("write wav: %w", err)
	}

	return nil
}

// wavInfo is the part of a RIFF WAV that readWAV needs.
type wavInfo struct {
	format, channels, bits uint16
	rate                   uint32
	data                   []byte
}

// parseWAV walks the RIFF chunks and returns the fmt fields and the data chunk.
func parseWAV(raw []byte) (wavInfo, error) {
	var (
		w       wavInfo
		haveFmt bool
	)

	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return w, errors.New("not a RIFF WAVE file")
	}

	for p := 12; p+8 <= len(raw); {
		id := string(raw[p : p+4])
		size := int(binary.LittleEndian.Uint32(raw[p+4:]))
		body := raw[p+8:]

		if size > len(body) {
			return w, fmt.Errorf("wav chunk %q overruns the file", id)
		}

		body = body[:size]

		switch id {
		case "fmt ":
			if size < 16 {
				return w, errors.New("wav fmt chunk too short")
			}

			w.format = binary.LittleEndian.Uint16(body)
			w.channels = binary.LittleEndian.Uint16(body[2:])
			w.rate = binary.LittleEndian.Uint32(body[4:])
			w.bits = binary.LittleEndian.Uint16(body[14:])
			haveFmt = true
		case "data":
			w.data = body
		}

		p += 8 + size + size%2
	}

	if !haveFmt || w.data == nil {
		return w, errors.New("wav lacks a fmt or data chunk")
	}

	if w.channels == 0 {
		return w, errors.New("wav has zero channels")
	}

	return w, nil
}

// readWAV decodes a RIFF WAV holding 16-bit PCM or 32-bit IEEE float samples into mono float32 in [-1, 1],
// averaging the channels. It returns the samples and the sample rate.
func readWAV(raw []byte) ([]float32, int, error) {
	w, err := parseWAV(raw)
	if err != nil {
		return nil, 0, err
	}

	var sample func([]byte) float32

	switch {
	case w.format == 1 && w.bits == 16:
		sample = func(b []byte) float32 { return float32(int16(binary.LittleEndian.Uint16(b))) / 32768 } // #nosec G115
	case w.format == 3 && w.bits == 32:
		sample = func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }
	default:
		return nil, 0, fmt.Errorf("unsupported wav encoding: format %d, %d bits", w.format, w.bits)
	}

	width := int(w.bits / 8)
	frame := width * int(w.channels)
	out := make([]float32, len(w.data)/frame)

	for i := range out {
		var sum float32

		for c := range int(w.channels) {
			sum += sample(w.data[i*frame+c*width:])
		}

		out[i] = sum / float32(w.channels)
	}

	return out, int(w.rate), nil
}

// toInt16 converts float samples to 16-bit PCM, clipping to [-1, 1]. Negative values scale by 32768 and positive
// ones by 32767, so both ends of the range are reachable.
func toInt16(pcm []float32) []int16 {
	out := make([]int16, len(pcm))

	for i, x := range pcm {
		v := float64(max(-1, min(1, x)))
		if v < 0 {
			v *= 32768
		} else {
			v *= 32767
		}

		out[i] = int16(math.Round(v))
	}

	return out
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
