//go:build oracle

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"
)

func TestWAVRoundTrip(t *testing.T) {
	pcm := []int16{0, 1, -1, 32767, -32768, 1234}

	var buf bytes.Buffer

	err := writeWAV16(&buf, pcm)
	if err != nil {
		t.Fatal(err)
	}

	if buf.Len() != 44+2*len(pcm) {
		t.Fatalf("wav size = %d, want %d", buf.Len(), 44+2*len(pcm))
	}

	got, rate, err := readWAV(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}

	if rate != sampleRate {
		t.Errorf("rate = %d, want %d", rate, sampleRate)
	}

	for i, s := range pcm {
		if want := float32(s) / 32768; got[i] != want {
			t.Errorf("sample %d = %v, want %v", i, got[i], want)
		}
	}
}

// floatWAV builds a WAVE_FORMAT_IEEE_FLOAT file the way FLEURS stores its clips (fmt size 18, a fact chunk).
func floatWAV(t *testing.T, channels int, frames [][]float32) []byte {
	t.Helper()

	var data bytes.Buffer

	for _, fr := range frames {
		for _, s := range fr {
			_ = binary.Write(&data, binary.LittleEndian, s)
		}
	}

	var b bytes.Buffer

	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }

	b.WriteString("RIFF")
	le(uint32(4 + 26 + 12 + 8 + data.Len()))
	b.WriteString("WAVEfmt ")
	le(uint32(18))
	le(uint16(3))
	le(uint16(channels))
	le(uint32(sampleRate))
	le(uint32(sampleRate * 4 * channels))
	le(uint16(4 * channels))
	le(uint16(32))
	le(uint16(0))
	b.WriteString("fact")
	le(uint32(4))
	le(uint32(len(frames)))
	b.WriteString("data")
	le(uint32(data.Len()))
	b.Write(data.Bytes())

	return b.Bytes()
}

func TestReadFloatWAVDownmix(t *testing.T) {
	raw := floatWAV(t, 2, [][]float32{{0.5, 0.25}, {-1, 1}, {0.1, 0.3}})

	got, rate, err := readWAV(raw)
	if err != nil {
		t.Fatal(err)
	}

	if rate != sampleRate {
		t.Errorf("rate = %d", rate)
	}

	want := []float32{0.375, 0, 0.2}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-7 {
			t.Errorf("sample %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestToInt16(t *testing.T) {
	got := toInt16([]float32{0, 1, -1, 2, -2, 0.5, -0.5})
	want := []int16{0, 32767, -32768, 32767, -32768, 16384, -16384}

	for i := range want {
		if got[i] != want[i] {
			t.Errorf("toInt16[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestReadWAVRejects(t *testing.T) {
	for name, raw := range map[string][]byte{
		"short":   []byte("RIFF"),
		"no wave": append([]byte("RIFF\x04\x00\x00\x00AVI "), make([]byte, 40)...),
	} {
		_, _, err := readWAV(raw)
		if err == nil {
			t.Errorf("%s: readWAV succeeded", name)
		}
	}
}

func TestWriteSafetensors(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}

	var buf bytes.Buffer

	err := writeSafetensors(&buf, "embed", []int{3, 2}, data, map[string]string{"engine": "3.2.0"})
	if err != nil {
		t.Fatal(err)
	}

	raw := buf.Bytes()
	n := binary.LittleEndian.Uint64(raw[:8])

	if n%8 != 0 {
		t.Errorf("header length %d is not 8-byte aligned", n)
	}

	var hdr map[string]json.RawMessage

	err = json.Unmarshal(raw[8:8+n], &hdr)
	if err != nil {
		t.Fatalf("header: %v", err)
	}

	var meta map[string]string

	err = json.Unmarshal(hdr["__metadata__"], &meta)
	if err != nil || meta["engine"] != "3.2.0" {
		t.Errorf("__metadata__ = %s (%v)", hdr["__metadata__"], err)
	}

	var info struct {
		Dtype   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}

	err = json.Unmarshal(hdr["embed"], &info)
	if err != nil {
		t.Fatal(err)
	}

	if info.Dtype != "F32" || len(info.Shape) != 2 || info.Shape[0] != 3 || info.Shape[1] != 2 ||
		info.Offsets != [2]int64{0, 24} {
		t.Errorf("tensor info = %+v", info)
	}

	body := raw[8+n:]
	if len(body) != 24 {
		t.Fatalf("data length = %d, want 24", len(body))
	}

	for i, want := range data {
		got := math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:]))
		if got != want {
			t.Errorf("data[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestWriteSafetensorsShapeMismatch(t *testing.T) {
	err := writeSafetensors(&bytes.Buffer{}, "embed", []int{4, 2}, make([]float32, 6), nil)
	if err == nil {
		t.Error("shape mismatch accepted")
	}
}
