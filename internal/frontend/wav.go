// Package frontend turns audio into what Whistle's encoder consumes. wav.go
// and resample.go port whistle.py _read_wav: a WAV file becomes 16 kHz mono
// float32 samples in [-1, 1]. docs/FINDINGS.md F-010 records where the port
// differs from upstream and by how much.
package frontend

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

// SampleRate is the rate Whistle's front end runs at (whistle.py SAMPLE_RATE).
const SampleRate = 16000

// WAVE format tags accepted by Python's wave module.
const (
	formatPCM        = 0x0001
	formatExtensible = 0xFFFE
)

// subtypePCM is KSDATAFORMAT_SUBTYPE_PCM, the only WAVE_FORMAT_EXTENSIBLE
// sub-format wave.py reads.
var subtypePCM = [16]byte{0x01, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71}

// Errors for the files whistle.py's wave.open or _read_wav reject.
var (
	ErrTruncated     = errors.New("frontend: truncated WAV")
	ErrNotRIFF       = errors.New("frontend: file does not start with RIFF id")
	ErrNotWAVE       = errors.New("frontend: not a WAVE file")
	ErrMissingChunk  = errors.New("frontend: fmt chunk and/or data chunk missing")
	ErrDataBeforeFmt = errors.New("frontend: data chunk before fmt chunk")
	ErrChannels      = errors.New("frontend: bad # of channels")
	ErrSampleWidth   = errors.New("frontend: unsupported sample width")
)

// FormatError reports a WAVE format other than integer PCM, which wave.py
// rejects as "unknown format" (or "unknown extended format" for a
// WAVE_FORMAT_EXTENSIBLE sub-format other than PCM). IEEE float WAVs land
// here too.
type FormatError struct {
	Tag       uint16
	SubFormat [16]byte // set when Tag is WAVE_FORMAT_EXTENSIBLE
}

func (e *FormatError) Error() string {
	if e.Tag == formatExtensible {
		return fmt.Sprintf("frontend: unknown extended WAV format %x", e.SubFormat)
	}

	return fmt.Sprintf("frontend: unknown WAV format %#04x", e.Tag)
}

// Format is what wave.py reports for a file.
type Format struct {
	Channels    int
	SampleWidth int // bytes per sample: (bits + 7) / 8, 1 to 4
	SampleRate  int
}

// WAV is a parsed file: its format and the whole frames of its data chunk.
type WAV struct {
	Format

	Data []byte // little-endian PCM, len a multiple of Channels*SampleWidth
}

// Frames returns the number of sample frames.
func (w *WAV) Frames() int {
	return len(w.Data) / (w.Channels * w.SampleWidth)
}

// ParseWAV reads a RIFF/WAVE file the way Python's wave module does: every
// chunk is read through the RIFF chunk, so nothing past its declared size is
// seen; chunks are word-aligned; the last fmt chunk before the first data
// chunk wins and anything after the data chunk is ignored. The sample width
// comes from the bits per sample, never from the block align.
func ParseWAV(data []byte) (*WAV, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("%w: %d bytes, no RIFF header", ErrTruncated, len(data))
	}

	if string(data[:4]) != "RIFF" {
		return nil, ErrNotRIFF
	}

	riffSize := int64(binary.LittleEndian.Uint32(data[4:8]))
	body := data[8:]

	if int64(len(body)) > riffSize {
		body = body[:riffSize]
	}

	if len(body) < 4 || string(body[:4]) != "WAVE" {
		return nil, ErrNotWAVE
	}

	var format *Format

	for pos := int64(4); pos+8 <= int64(len(body)); {
		id := string(body[pos : pos+4])
		size := int64(binary.LittleEndian.Uint32(body[pos+4 : pos+8]))
		pos += 8
		contents := body[pos:min(pos+size, int64(len(body)))]

		switch id {
		case "fmt ":
			f, err := parseFmt(contents)
			if err != nil {
				return nil, err
			}

			format = &f
		case "data":
			if format == nil {
				return nil, ErrDataBeforeFmt
			}

			return newWAV(*format, size, contents)
		}

		// wave.py skips by seeking inside the RIFF chunk, which refuses to
		// move past the RIFF chunk's declared end.
		pos += size + size&1
		if pos > riffSize {
			return nil, fmt.Errorf("%w: chunk %q ends past the RIFF chunk", ErrTruncated, id)
		}
	}

	return nil, ErrMissingChunk
}

func parseFmt(b []byte) (Format, error) {
	if len(b) < 14 {
		return Format{}, fmt.Errorf("%w: %d-byte fmt chunk", ErrTruncated, len(b))
	}

	tag := binary.LittleEndian.Uint16(b[0:2])
	if tag != formatPCM && tag != formatExtensible {
		return Format{}, &FormatError{Tag: tag}
	}

	if len(b) < 16 {
		return Format{}, fmt.Errorf("%w: %d-byte fmt chunk", ErrTruncated, len(b))
	}

	if tag == formatExtensible {
		if len(b) < 40 {
			return Format{}, fmt.Errorf("%w: %d-byte extensible fmt chunk", ErrTruncated, len(b))
		}

		if sub := [16]byte(b[24:40]); sub != subtypePCM {
			return Format{}, &FormatError{Tag: tag, SubFormat: sub}
		}
	}

	f := Format{
		Channels:    int(binary.LittleEndian.Uint16(b[2:4])),
		SampleWidth: (int(binary.LittleEndian.Uint16(b[14:16])) + 7) / 8,
		SampleRate:  int(binary.LittleEndian.Uint32(b[4:8])),
	}

	// wave.py rejects width 0 and no channels; _read_wav has no decoder for
	// widths past 4 bytes (a KeyError).
	if f.SampleWidth == 0 || f.SampleWidth > 4 {
		return Format{}, fmt.Errorf("%w: %d bytes", ErrSampleWidth, f.SampleWidth)
	}

	if f.Channels == 0 {
		return Format{}, ErrChannels
	}

	return f, nil
}

// newWAV keeps the whole frames of a data chunk. wave.py takes the frame
// count from the declared size (a trailing partial frame is dropped) and
// reads what the file holds of it; a sample cut short by the end of the file
// is an error there for 16 and 32 bits, and here for every width above 8.
func newWAV(f Format, size int64, contents []byte) (*WAV, error) {
	frameSize := int64(f.Channels * f.SampleWidth)

	want := size / frameSize * frameSize
	if int64(len(contents)) > want {
		contents = contents[:want]
	}

	if f.SampleWidth > 1 && len(contents)%f.SampleWidth != 0 {
		return nil, fmt.Errorf("%w: data ends inside a %d-byte sample", ErrTruncated, f.SampleWidth)
	}

	return &WAV{Format: f, Data: contents[:int64(len(contents))/frameSize*frameSize]}, nil
}

// Mono averages the channels to float32 exactly as _read_wav does:
// (sum(frame) - offset*channels) / (2^(8*width-1) * channels) in float64,
// with offset 128 for unsigned 8-bit samples and 0 otherwise, then rounded
// to float32 (whistle.py _samples / numpy.float32).
func (w *WAV) Mono() []float32 {
	channels, width := w.Channels, w.SampleWidth
	frameSize := channels * width

	var offset int64
	if width == 1 {
		offset = 128
	}

	bias := offset * int64(channels)
	scale := float64(int64(1)<<(8*width-1)) * float64(channels)

	out := make([]float32, w.Frames())
	for f := range out {
		frame := w.Data[f*frameSize : (f+1)*frameSize]

		var sum int64
		for c := range channels {
			sum += sample(frame[c*width:], width)
		}

		out[f] = float32(float64(sum-bias) / scale)
	}

	return out
}

// sample decodes one little-endian sample: unsigned for 8 bits, signed
// otherwise.
func sample(b []byte, width int) int64 {
	switch width {
	case 1:
		return int64(b[0])
	case 2:
		return int64(int16(binary.LittleEndian.Uint16(b)))
	case 3:
		return int64(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24) >> 8)
	default:
		return int64(int32(binary.LittleEndian.Uint32(b)))
	}
}

// DecodeWAV is whistle.py _read_wav on file contents: parse, mix to mono and
// resample to SampleRate unless the file is already at that rate or empty.
func DecodeWAV(data []byte) ([]float32, error) {
	w, err := ParseWAV(data)
	if err != nil {
		return nil, err
	}

	mono := w.Mono()
	if w.SampleRate == SampleRate || len(mono) == 0 {
		return mono, nil
	}

	return Resample(mono, w.SampleRate)
}

// ReadWAV is DecodeWAV on a file.
func ReadWAV(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("frontend: %w", err)
	}

	return DecodeWAV(data)
}
