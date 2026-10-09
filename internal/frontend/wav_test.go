package frontend

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// chunk is one RIFF sub-chunk for buildRIFF.
type chunk struct {
	id   string
	body []byte
	size int64 // declared size; -1 means len(body)
}

// buildRIFF lays out "RIFF" <size> "WAVE" and the chunks, each padded to an
// even length as wave.py expects. riffSize -1 declares the true size.
func buildRIFF(riffSize int64, chunks ...chunk) []byte {
	var body bytes.Buffer

	body.WriteString("WAVE")

	for _, c := range chunks {
		size := c.size
		if size < 0 {
			size = int64(len(c.body))
		}

		body.WriteString(c.id)
		_ = binary.Write(&body, binary.LittleEndian, uint32(size))
		body.Write(c.body)

		if len(c.body)%2 == 1 {
			body.WriteByte(0)
		}
	}

	if riffSize < 0 {
		riffSize = int64(body.Len())
	}

	out := []byte("RIFF")
	out = binary.LittleEndian.AppendUint32(out, uint32(riffSize))

	return append(out, body.Bytes()...)
}

// fmtPCM is a 16-byte fmt body with the given format tag. Byte rate and
// block align are filled in but ignored by the reader, as by wave.py.
func fmtPCM(tag uint16, channels, rate, bits int) []byte {
	b := binary.LittleEndian.AppendUint16(nil, tag)
	b = binary.LittleEndian.AppendUint16(b, uint16(channels))
	b = binary.LittleEndian.AppendUint32(b, uint32(rate))
	b = binary.LittleEndian.AppendUint32(b, uint32(rate*channels*((bits+7)/8)))
	b = binary.LittleEndian.AppendUint16(b, uint16(channels*((bits+7)/8)))

	return binary.LittleEndian.AppendUint16(b, uint16(bits))
}

// fmtExtensible is a 40-byte WAVE_FORMAT_EXTENSIBLE fmt body.
func fmtExtensible(channels, rate, bits int, sub [16]byte) []byte {
	b := fmtPCM(formatExtensible, channels, rate, bits)
	b = binary.LittleEndian.AppendUint16(b, 22)           // cbSize
	b = binary.LittleEndian.AppendUint16(b, uint16(bits)) // wValidBitsPerSample
	b = binary.LittleEndian.AppendUint32(b, 0)            // dwChannelMask

	return append(b, sub[:]...)
}

func simpleWAV(channels, rate, bits int, data []byte) []byte {
	return buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, channels, rate, bits), -1}, chunk{"data", data, -1})
}

func TestParseWAVFormat(t *testing.T) {
	for _, tc := range []struct {
		channels, bits, width int
	}{
		{1, 8, 1}, {2, 16, 2}, {3, 24, 3}, {1, 32, 4}, {1, 12, 2}, {2, 20, 3},
	} {
		w, err := ParseWAV(simpleWAV(tc.channels, 22050, tc.bits, make([]byte, 6*tc.channels*tc.width)))
		if err != nil {
			t.Fatalf("%d bits: %v", tc.bits, err)
		}

		want := Format{Channels: tc.channels, SampleWidth: tc.width, SampleRate: 22050}
		if w.Format != want || w.Frames() != 6 {
			t.Errorf("%d bits: got %+v with %d frames, want %+v with 6", tc.bits, w.Format, w.Frames(), want)
		}
	}
}

func TestMonoConversion(t *testing.T) {
	for _, tc := range []struct {
		name     string
		channels int
		bits     int
		data     []byte
		want     []float32
	}{
		{"8-bit unsigned, offset 128", 1, 8, []byte{0, 128, 255, 64}, []float32{-1, 0, 127.0 / 128, -0.5}},
		{"16-bit", 1, 16, le(2, 0, -32768, 32767, 16384), []float32{0, -1, 32767.0 / 32768, 0.5}},
		{"24-bit sign extension", 1, 24, le(3, -1, -8388608, 8388607, 1), []float32{
			-1.0 / 8388608, -1, 8388607.0 / 8388608, 1.0 / 8388608,
		}},
		{"32-bit", 1, 32, le(4, math.MinInt32, math.MaxInt32, -1073741824), []float32{
			-1, float32(float64(math.MaxInt32) / (1 << 31)), -0.5,
		}},
		{"stereo average", 2, 16, le(2, 32767, -32768, 16384, 16384, -100, 0), []float32{
			-1.0 / 65536, 0.5, -100.0 / 65536,
		}},
		{"8-bit stereo", 2, 8, []byte{255, 0, 200, 200}, []float32{-1.0 / 256, 144.0 / 256}},
		{"three channels", 3, 16, le(2, 3, 3, 3), []float32{float32(9.0 / (3 * 32768))}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, err := ParseWAV(simpleWAV(tc.channels, SampleRate, tc.bits, tc.data))
			if err != nil {
				t.Fatal(err)
			}

			got := w.Mono()
			if len(got) != len(tc.want) {
				t.Fatalf("got %d samples, want %d", len(got), len(tc.want))
			}

			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("sample %d: got %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// TestMonoFormula checks Mono against whistle.py's expression
// float32((sum(frame) - offset*channels) / (2^(8w-1)*channels)) on every
// width and channel count with pseudo-random data.
func TestMonoFormula(t *testing.T) {
	state := uint64(1)

	for width := 1; width <= 4; width++ {
		for channels := 1; channels <= 3; channels++ {
			data := make([]byte, 50*width*channels)
			for i := range data {
				state = state*6364136223846793005 + 1442695040888963407
				data[i] = byte(state >> 56)
			}

			w, err := ParseWAV(simpleWAV(channels, SampleRate, 8*width, data))
			if err != nil {
				t.Fatal(err)
			}

			got := w.Mono()
			scale := float64(int64(1)<<(8*width-1)) * float64(channels)

			for f := range 50 {
				var sum int64

				for c := range channels {
					sum += sampleAt(data[(f*channels+c)*width:], width)
				}

				want := float32(float64(sum) / scale)
				if got[f] != want {
					t.Fatalf("width %d, %d channels, frame %d: got %v, want %v", width, channels, f, got[f], want)
				}
			}
		}
	}
}

// sampleAt is an independent decoder for one sample: unsigned minus 128 for
// 8 bits, signed little-endian otherwise.
func sampleAt(b []byte, width int) int64 {
	if width == 1 {
		return int64(b[0]) - 128
	}

	var v int64
	for i := width - 1; i >= 0; i-- {
		v = v<<8 | int64(b[i])
	}

	shift := 64 - 8*width

	return v << shift >> shift
}

func TestParseWAVChunkWalk(t *testing.T) {
	fmtChunk := chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 16), -1}
	samples := le(2, 1, 2, 3, 4)

	for _, tc := range []struct {
		name string
		data []byte
		want int // frames
	}{
		{"odd chunk before fmt is padded", buildRIFF(-1, chunk{"LIST", []byte("abc"), -1}, fmtChunk, chunk{"data", samples, -1}), 4},
		{"chunks after data are ignored", buildRIFF(-1, fmtChunk, chunk{"data", samples, -1}, chunk{"fmt ", []byte{1}, -1}), 4},
		{"trailing partial frame is dropped", buildRIFF(-1, fmtChunk, chunk{"data", append(samples[:8:8], 9), -1}), 4},
		{"streamed data size past EOF", buildRIFF(0xffffffff, fmtChunk, chunk{"data", samples, 0xffffffff}), 4},
		{"data limited by the RIFF size", buildRIFF(4+8+16+8+4, fmtChunk, chunk{"data", samples, -1}), 2},
		{"empty data", buildRIFF(-1, fmtChunk, chunk{"data", nil, -1}), 0},
		{"long fmt chunk", buildRIFF(-1, chunk{"fmt ", append(fmtPCM(formatPCM, 1, SampleRate, 16), 0, 0), -1}, chunk{"data", samples, -1}), 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, err := ParseWAV(tc.data)
			if err != nil {
				t.Fatal(err)
			}

			if w.Frames() != tc.want {
				t.Fatalf("got %d frames, want %d", w.Frames(), tc.want)
			}

			mono := w.Mono()
			for i := range mono {
				if want := float32(i+1) / 32768; mono[i] != want {
					t.Errorf("sample %d: got %v, want %v", i, mono[i], want)
				}
			}
		})
	}
}

func TestParseWAVExtensible(t *testing.T) {
	data := buildRIFF(-1, chunk{"fmt ", fmtExtensible(2, 48000, 24, subtypePCM), -1}, chunk{"data", make([]byte, 12), -1})

	w, err := ParseWAV(data)
	if err != nil {
		t.Fatal(err)
	}

	if want := (Format{Channels: 2, SampleWidth: 3, SampleRate: 48000}); w.Format != want || w.Frames() != 2 {
		t.Fatalf("got %+v with %d frames", w.Format, w.Frames())
	}
}

func TestParseWAVErrors(t *testing.T) {
	pcm16 := chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 16), -1}
	data := chunk{"data", le(2, 1, 2), -1}
	floatSub := subtypePCM
	floatSub[0] = 3 // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT

	for _, tc := range []struct {
		name string
		data []byte
		want error
	}{
		{"empty file", nil, ErrTruncated},
		{"short header", []byte("RIFF\x04\x00"), ErrTruncated},
		{"not RIFF", append([]byte("RIFX"), buildRIFF(-1, pcm16, data)[4:]...), ErrNotRIFF},
		{"not WAVE", append(buildRIFF(-1)[:8:8], "AVI "...), ErrNotWAVE},
		{"RIFF size 0", buildRIFF(0, pcm16, data), ErrNotWAVE},
		{"no chunks", buildRIFF(-1), ErrMissingChunk},
		{"no data", buildRIFF(-1, pcm16), ErrMissingChunk},
		{"no fmt", buildRIFF(-1, chunk{"LIST", nil, -1}), ErrMissingChunk},
		{"data before fmt", buildRIFF(-1, data, pcm16), ErrDataBeforeFmt},
		{"fmt shorter than 16 bytes", buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 16)[:14], -1}, data), ErrTruncated},
		{"extensible fmt shorter than 40 bytes", buildRIFF(-1, chunk{"fmt ", fmtExtensible(1, SampleRate, 16, subtypePCM)[:30], -1}, data), ErrTruncated},
		{"chunk past the RIFF end", buildRIFF(-1, chunk{"LIST", nil, 100}), ErrTruncated},
		{"zero channels", buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 0, SampleRate, 16), -1}, data), ErrChannels},
		{"zero bits", buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 0), -1}, data), ErrSampleWidth},
		{"40 bits", buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 40), -1}, data), ErrSampleWidth},
		{"sample cut short", buildRIFF(4+8+16+8+3, pcm16, chunk{"data", []byte{1, 0, 2}, 4}), ErrTruncated},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseWAV(tc.data)
			if !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
		})
	}

	for _, tc := range []struct {
		name string
		fmt  []byte
		tag  uint16
	}{
		{"IEEE float", fmtPCM(3, 1, SampleRate, 32), 3},
		{"A-law", fmtPCM(6, 1, SampleRate, 8), 6},
		{"MPEG", fmtPCM(0x55, 1, SampleRate, 0), 0x55},
		{"extensible float", fmtExtensible(1, SampleRate, 32, floatSub), formatExtensible},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseWAV(buildRIFF(-1, chunk{"fmt ", tc.fmt, -1}, data))

			var fe *FormatError
			if !errors.As(err, &fe) || fe.Tag != tc.tag {
				t.Fatalf("got %v, want a FormatError with tag %#x", err, tc.tag)
			}
		})
	}
}

// le packs values as little-endian signed integers of width bytes.
func le(width int, values ...int64) []byte {
	out := make([]byte, 0, width*len(values))
	for _, v := range values {
		for i := range width {
			out = append(out, byte(v>>(8*i)))
		}
	}

	return out
}
