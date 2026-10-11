package pcm

import (
	"bytes"
	"encoding/binary"
	"slices"
	"strings"
	"testing"
)

func TestWidth(t *testing.T) {
	for format, want := range map[string]int{S16LE: 2, F32LE: 4} {
		got, err := Width(format)
		if err != nil || got != want {
			t.Errorf("Width(%q) = %d, %v; want %d", format, got, err, want)
		}
	}

	_, err := Width("u8")
	if err == nil || !strings.Contains(err.Error(), "s16le or f32le") {
		t.Errorf("Width(u8): err = %v", err)
	}
}

func TestDecode(t *testing.T) {
	var s16 bytes.Buffer

	_ = binary.Write(&s16, binary.LittleEndian, []int16{0, 16384, -32768, 32767, -1})

	got, err := Decode(s16.Bytes(), 2)
	if err != nil || !slices.Equal(got, []float32{0, 0.5, -1, 32767.0 / 32768, -1.0 / 32768}) {
		t.Errorf("s16le = %v, %v", got, err)
	}

	var f32 bytes.Buffer

	_ = binary.Write(&f32, binary.LittleEndian, []float32{0.25, -0.75, 1})

	got, err = Decode(f32.Bytes(), 4)
	if err != nil || !slices.Equal(got, []float32{0.25, -0.75, 1}) {
		t.Errorf("f32le = %v, %v", got, err)
	}

	got, err = Decode(nil, 2)
	if err != nil || len(got) != 0 {
		t.Errorf("empty = %v, %v", got, err)
	}

	_, err = Decode([]byte{1, 2, 3}, 2)
	if err == nil || !strings.Contains(err.Error(), "inside a sample: 1 bytes") {
		t.Errorf("odd byte count: err = %v", err)
	}

	_, err = Decode([]byte{1, 2, 3}, 3)
	if err == nil {
		t.Error("width 3: no error")
	}
}
