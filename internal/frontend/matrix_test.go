package frontend

import (
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"testing"
)

// upstreamTone is test_whistle.py _tone: 0.3 * sin(2*pi*220*i/rate) for
// int(seconds*rate) samples.
func upstreamTone(seconds float64, rate int) []float64 {
	out := make([]float64, int(seconds*float64(rate)))
	for i := range out {
		out[i] = 0.3 * math.Sin(2*math.Pi*220*float64(i)/float64(rate))
	}

	return out
}

// upstreamWAV is test_whistle.py _write_wav: level = int(value * peak) with
// peak 2^(8w-1)-1, 8-bit stored as level+128, every channel the same frame.
func upstreamWAV(samples []float64, rate, channels, width int) []byte {
	peak := float64(int64(1)<<(8*width-1) - 1)

	var data []byte

	for _, v := range samples {
		level := int64(v * peak) // Python int() truncates toward zero

		var frame []byte
		if width == 1 {
			frame = []byte{byte(level + 128)}
		} else {
			frame = le(width, level)
		}

		for range channels {
			data = append(data, frame...)
		}
	}

	return simpleWAV(channels, rate, 8*width, data)
}

// TestUpstreamWAVMatrix is the PLAN.md Phase 3 Accept criterion, ported
// from test_whistle.py test_wav_files_become_16k_mono_floats: a 0.5 s 220 Hz
// tone at 0.3 in each (rate, channels, width) becomes 8000±1 samples at
// 16 kHz with peaks inside (0.28, 0.31) and 108 to 111 upward zero
// crossings.
func TestUpstreamWAVMatrix(t *testing.T) {
	for _, tc := range []struct{ rate, channels, width int }{
		{16000, 1, 2}, {8000, 1, 2}, {44100, 2, 2}, {16000, 1, 1}, {22050, 2, 3}, {48000, 1, 4},
	} {
		t.Run(fmt.Sprintf("%d/%d/%d", tc.rate, tc.channels, tc.width), func(t *testing.T) {
			samples, err := DecodeWAV(upstreamWAV(upstreamTone(0.5, tc.rate), tc.rate, tc.channels, tc.width))
			if err != nil {
				t.Fatal(err)
			}

			hi, lo := slices.Max(samples), slices.Min(samples)

			crossings := 0

			for i := 1; i < len(samples); i++ {
				if samples[i-1] < 0 && samples[i] >= 0 {
					crossings++
				}
			}

			t.Logf("%d samples, max %.5f, min %.5f, %d crossings", len(samples), hi, lo, crossings)

			if d := len(samples) - 8000; d < -1 || d > 1 {
				t.Errorf("got %d samples, want 8000±1", len(samples))
			}

			if !(0.28 < hi && hi < 0.31 && -0.31 < lo && lo < -0.28) {
				t.Errorf("peaks %v and %v, want inside (0.28, 0.31)", hi, lo)
			}

			if crossings < 108 || crossings > 111 {
				t.Errorf("got %d zero crossings, want 108 to 111", crossings)
			}
		})
	}
}

// TestUpstreamWAVBytes pins upstreamWAV to the files test_whistle.py
// _write_wav (Python 3.12's wave writer) produces for the first three
// samples of _tone(0.5, 8000), one per width. Python leaves an odd data
// chunk unpadded; buildRIFF adds the pad byte RIFF asks for, so widths 1 and
// 3 differ in the RIFF size (+1) and the trailing 00.
func TestUpstreamWAVBytes(t *testing.T) {
	tone := upstreamTone(0.5, 8000)[:3]

	for width, want := range map[int]string{
		1: "524946462800000057415645666d74201000000001000100401f0000401f0000010008006461746103000000" + "80868c" + "00",
		2: "524946462a00000057415645666d74201000000001000100401f0000803e0000020010006461746106000000" + "00009a06010d",
		3: "524946462e00000057415645666d74201000000001000100401f0000c05d0000030018006461746109000000" + "000000219a06ed010d" + "00",
		4: "524946463000000057415645666d74201000000001000100401f0000007d000004002000646174610c000000" + "00000000bf219a06e3ed010d",
	} {
		if got := hex.EncodeToString(upstreamWAV(tone, 8000, 1, width)); got != want {
			t.Errorf("width %d:\n got %s\nwant %s", width, got, want)
		}
	}
}
