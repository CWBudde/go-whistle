// Package pcm decodes the raw PCM formats `whistle stream` reads from stdin
// and `whistle serve` takes over HTTP and WebSocket: 16 kHz mono, s16le or
// f32le.
package pcm

import (
	"encoding/binary"
	"fmt"
	"math"
)

// The raw formats.
const (
	S16LE = "s16le" // signed 16-bit little-endian integers, scaled by 1/32768 as the WAV reader does
	F32LE = "f32le" // IEEE float32 little-endian, in [-1, 1]
)

// Width is the size of one sample of format in bytes.
func Width(format string) (int, error) {
	switch format {
	case S16LE:
		return 2, nil
	case F32LE:
		return 4, nil
	default:
		return 0, fmt.Errorf("format %q: want %s or %s", format, S16LE, F32LE)
	}
}

// Decode converts b, whole samples of width bytes (2 for s16le, 4 for f32le),
// to float32 samples.
func Decode(b []byte, width int) ([]float32, error) {
	if width != 2 && width != 4 {
		return nil, fmt.Errorf("sample width %d: want 2 or 4", width)
	}

	if len(b)%width != 0 {
		return nil, fmt.Errorf("ends inside a sample: %d bytes after the last whole one", len(b)%width)
	}

	out := make([]float32, len(b)/width)

	for i := range out {
		s := b[i*width:]
		if width == 2 {
			out[i] = float32(int16(binary.LittleEndian.Uint16(s))) / 32768 // #nosec G115 -- reinterprets the bits.
		} else {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(s))
		}
	}

	return out, nil
}
