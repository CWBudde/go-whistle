package frontend

import (
	"bytes"
	"errors"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// readAll drains s in reads of n samples.
func readAll(t *testing.T, s *WAVStream, n int) []float32 {
	t.Helper()

	var out []float32

	for {
		pcm, err := s.Read(n)
		if errors.Is(err, io.EOF) {
			return out
		}

		if err != nil {
			t.Fatal(err)
		}

		if len(pcm) == 0 || len(pcm) > n {
			t.Fatalf("read of %d gave %d samples", n, len(pcm))
		}

		out = append(out, pcm...)
	}
}

// checkStream asserts that OpenWAV and Read give DecodeWAV's samples or its
// error, bit for bit, for every read size.
func checkStream(t *testing.T, data []byte) {
	t.Helper()

	want, wantErr := DecodeWAV(data)

	for _, n := range []int{1, 7, 1600, 16000, 1 << 20} {
		s, err := OpenWAV(bytes.NewReader(data), int64(len(data)))

		var got []float32
		if err == nil {
			got = readAll(t, s, n)
		}

		if (err == nil) != (wantErr == nil) || err != nil && err.Error() != wantErr.Error() {
			t.Fatalf("read %d: error %v, DecodeWAV %v", n, err, wantErr)
		}

		if !slices.Equal(got, want) {
			t.Fatalf("read %d: %d samples differ from DecodeWAV's %d", n, len(got), len(want))
		}
	}
}

func TestWAVStreamMatchesDecode(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	for _, rate := range []int{8000, 11025, 16000, 22050, 44100, 48000} {
		for _, channels := range []int{1, 2} {
			for _, bits := range []int{8, 16, 24, 32} {
				for _, frames := range []int{0, 1, 3, 1000, 37_301} {
					data := make([]byte, frames*channels*((bits+7)/8))
					for i := range data {
						data[i] = byte(rng.Uint32())
					}

					checkStream(t, simpleWAV(channels, rate, bits, data))
				}
			}
		}
	}
}

func TestWAVStreamHeaders(t *testing.T) {
	pcm16 := chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 16), -1}
	data := chunk{"data", le(2, 1, 2), -1}
	samples := le(2, 1, 2, 3, 4)

	for _, f := range [][]byte{
		// TestParseWAVChunkWalk's files.
		buildRIFF(-1, chunk{"LIST", []byte("abc"), -1}, pcm16, chunk{"data", samples, -1}),
		buildRIFF(-1, pcm16, chunk{"data", samples, -1}, chunk{"fmt ", []byte{1}, -1}),
		buildRIFF(-1, pcm16, chunk{"data", append(samples[:8:8], 9), -1}),
		buildRIFF(0xffffffff, pcm16, chunk{"data", samples, 0xffffffff}),
		buildRIFF(4+8+16+8+4, pcm16, chunk{"data", samples, -1}),
		buildRIFF(-1, chunk{"fmt ", append(fmtPCM(formatPCM, 1, SampleRate, 16), 0, 0), -1}, chunk{"data", samples, -1}),
		buildRIFF(-1, chunk{"fmt ", fmtExtensible(2, 48000, 24, subtypePCM), -1}, chunk{"data", make([]byte, 12), -1}),
		// TestParseWAVErrors's files.
		nil,
		[]byte("RIFF\x04\x00"),
		[]byte("RIFF\x04\x00\x00\x00WAV"),
		append([]byte("RIFX"), buildRIFF(-1, pcm16, data)[4:]...),
		append(buildRIFF(-1)[:8:8], "AVI "...),
		buildRIFF(0, pcm16, data),
		buildRIFF(-1),
		buildRIFF(-1, pcm16),
		buildRIFF(-1, chunk{"LIST", nil, -1}),
		buildRIFF(-1, data, pcm16),
		buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 16)[:14], -1}, data),
		buildRIFF(-1, chunk{"fmt ", fmtExtensible(1, SampleRate, 16, subtypePCM)[:30], -1}, data),
		buildRIFF(-1, chunk{"LIST", nil, 100}),
		buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 0, SampleRate, 16), -1}, data),
		buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 1, SampleRate, 40), -1}, data),
		buildRIFF(4+8+16+8+3, pcm16, chunk{"data", []byte{1, 0, 2}, 4}),
		buildRIFF(-1, chunk{"fmt ", fmtPCM(3, 1, SampleRate, 32), -1}, data),
		buildRIFF(-1, chunk{"fmt ", fmtPCM(formatPCM, 1, 4000, 16), -1}, data),
	} {
		checkStream(t, f)
	}
}

// TestWAVStreamClips streams the first fixture clips against DecodeWAV.
func TestWAVStreamClips(t *testing.T) {
	paths, _ := filepath.Glob("../../testdata/audio/*.wav")
	if len(paths) == 0 {
		t.Skip("no testdata/audio clips")
	}

	for _, p := range paths[:min(len(paths), 4)] {
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}

		checkStream(t, data)
	}
}
