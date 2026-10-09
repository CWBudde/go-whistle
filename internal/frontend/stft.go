package frontend

import (
	"errors"
	"fmt"
	"math"

	"github.com/cwbudde/algo-dsp/dsp/stft"
	"github.com/cwbudde/algo-dsp/dsp/window"
)

// STFT geometry: the config's fft, win and hop, the same in the .cact audio
// manifest (docs/FINDINGS.md F-003).
const (
	NFFT      = 512
	WinLength = 400
	Hop       = 160
	Bins      = NFFT/2 + 1
)

// ErrTooShort reports a signal PadReflect cannot frame: algo-dsp mirrors up
// to NFFT/2 samples, so it needs more than 256. Below 640 samples the engine
// returns no encoder frame at all (F-001), so nothing usable is lost (F-012).
var ErrTooShort = errors.New("frontend: signal too short for reflect padding")

// FrameCount is the number of STFT frames for n samples (F-012). Centred
// framing (PadReflect, PadZero) gives n / Hop, rounded down: frame i is
// centred on sample i·Hop, and the last one is dropped where torch.stft's
// n/Hop + 1 would add it, as Whisper does. That is the published 3000 frames
// for 30 s, and halved three times rounding up it gives needle_embed's frame
// count for every measured length, which ceil(n/Hop), n/Hop + 1 and PadNone's
// count do not. PadNone keeps the frames whose 400 samples lie inside the
// signal, 1 + (n − 400) / Hop: 2998 for 30 s.
func FrameCount(n int, p Padding) int {
	if p == PadNone {
		if n < WinLength {
			return 0
		}

		return 1 + (n-WinLength)/Hop
	}

	return max(n, 0) / Hop
}

// analysisWindow is the 400-sample Hann window.
func analysisWindow(w Window) []float64 {
	if w == HannPeriodic {
		return window.Generate(window.TypeHann, WinLength, window.WithPeriodic())
	}

	return window.Generate(window.TypeHann, WinLength)
}

// spectrogram frames a signal and runs the real FFT through algo-dsp's
// dsp/stft (F-012). algo-dsp centres an NFFT-sample frame on sample i·Hop
// and multiplies it by an NFFT-long window, so the 400-sample window goes
// into a 512-long one at an offset that puts its samples where the framing
// wants them:
//
//   - centred (PadReflect, PadZero): offset 56, i.e. the window is centred
//     in the FFT frame as torch.stft and librosa pad a short window, and
//     covers [i·Hop − 200, i·Hop + 200);
//   - PadNone: offset 96 in algo-dsp frame i+1, which covers
//     [i·Hop, i·Hop + 400). algo-dsp's own PadNone needs all 512 samples
//     inside the signal and would drop the last frame (2997 for 30 s).
//
// Zeros outside the window only change the phase, never |X|, so the power
// equals that of the 400 windowed samples zero-padded to 512.
type spectrogram struct {
	t       *stft.STFT32
	padding Padding
	shift   int // algo-dsp frame index of frame 0
}

func newSpectrogram(w Window, p Padding) (*spectrogram, error) {
	offset, shift, mode := (NFFT-WinLength)/2, 0, stft.PadZero

	switch p {
	case PadReflect:
		mode = stft.PadReflect
	case PadZero:
	case PadNone:
		offset, shift = NFFT/2-Hop, 1
	default:
		return nil, fmt.Errorf("%w: padding %d", ErrOptions, p)
	}

	coeffs := make([]float64, NFFT)
	copy(coeffs[offset:], analysisWindow(w))

	t, err := stft.New32(NFFT, Hop, stft.WithCenter(mode), stft.WithCustomWindow(coeffs))
	if err != nil {
		return nil, err
	}

	return &spectrogram{t: t, padding: p, shift: shift}, nil
}

// spectrum returns |X|² (or |X| when magnitude is set) for
// FrameCount(len(x)) frames of Bins values, frame-major. It is safe for
// concurrent use.
func (s *spectrogram) spectrum(x []float32, magnitude bool) ([]float32, int, error) {
	frames := FrameCount(len(x), s.padding)
	if frames == 0 {
		return []float32{}, 0, nil
	}

	if s.padding == PadReflect && len(x) <= NFFT/2 {
		return nil, 0, fmt.Errorf("%w: %d samples, need more than %d", ErrTooShort, len(x), NFFT/2)
	}

	t := s.t.Clone()
	bins := make([]complex64, Bins)
	out := make([]float32, frames*Bins)

	for i := range frames {
		err := t.FrameInto(bins, x, i+s.shift)
		if err != nil {
			return nil, 0, err
		}

		row := out[i*Bins : (i+1)*Bins]

		for k, c := range bins {
			re, im := float64(real(c)), float64(imag(c))

			p := re*re + im*im
			if magnitude {
				p = math.Sqrt(p)
			}

			row[k] = float32(p)
		}
	}

	return out, frames, nil
}
