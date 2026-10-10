package frontend

import (
	"errors"
	"fmt"
	"math"

	"github.com/cwbudde/algo-dsp/dsp/resample"
)

// ErrSampleRate reports a WAV whose rate cannot be resampled to SampleRate.
var ErrSampleRate = errors.New("frontend: unsupported sample rate")

// minRate is the lowest rate Resample accepts. It caps the expansion at 16
// output samples per input sample, so a small WAV with a near-zero rate field
// cannot ask for gigabytes; soxr takes any positive rate (F-010).
const minRate = 1000

// maxRatioTerm bounds down in the reduced conversion ratio up/down. It keeps
// every common rate exact, 44056 Hz (2000/5507) included, and the prototype
// at most about two million taps (35 MB, 70 ms to design). Rates that need a
// larger down, such as 47999 Hz, get algo-dsp's continued-fraction
// approximation instead (F-010).
const maxRatioTerm = 16384

// filterSpec sets the algo-dsp windowed-sinc prototype.
type filterSpec struct {
	// length is the span of the prototype in samples of the lower of the
	// two rates: taps per phase is ceil(length * max(up, down) / up). That
	// keeps the transition band equally narrow relative to the lower Nyquist
	// for every ratio. algo-dsp's presets fix the taps per phase instead,
	// which widens it by down/up when downsampling (QualityBest, 64 taps per
	// phase: up to -1.7 dB at 7 kHz, and a 9.5 kHz tone aliases at -52 dB).
	length float64
	cutoff float64 // -6 dB point relative to the lower Nyquist
	beta   float64 // Kaiser window beta
}

// hqSpec is the setting Resample uses. Against soxr HQ it matches the band
// edge (-2.8 dB at 7.6 kHz, soxr -2.9 dB) and leaves aliases at -87 dB from
// 8.1 kHz and below -110 dB from 8.2 kHz (soxr: -130 dB), for about 0.1 s
// per 30 s of 44.1 kHz audio. Length 64 takes less than half the time but
// lets 8.1 kHz alias at -29 dB; 160 gains 2 dB for 30 % more time (F-010).
var hqSpec = filterSpec{length: 128, cutoff: 0.96, beta: 11}

// Resample converts mono samples at rate Hz to SampleRate, where whistle.py
// calls soxr.resample(mono, rate, 16000, quality="HQ"). Like soxr's one-shot
// call it returns round(len(mono) * 16000 / rate) samples with the filter
// delay removed, so input time t maps to output time t. Samples at
// SampleRate are returned as a copy.
func Resample(mono []float32, rate int) ([]float32, error) {
	return resampleWith(mono, rate, hqSpec)
}

func resampleWith(mono []float32, rate int, spec filterSpec) ([]float32, error) {
	if rate < minRate {
		return nil, fmt.Errorf("%w: %d Hz is below %d Hz", ErrSampleRate, rate, minRate)
	}

	if rate == SampleRate {
		return append([]float32{}, mono...), nil
	}

	scaled := math.Round(float64(len(mono)) * SampleRate / float64(rate))
	if scaled > math.MaxInt32 {
		return nil, fmt.Errorf("frontend: %d samples at %d Hz resample to %.0f, more than %d", len(mono), rate, scaled, math.MaxInt32)
	}

	// The resampler validates the rate, so build it before the empty
	// output's early return: acceptance must not depend on the length.
	r, err := newResampler(rate, spec)
	if err != nil {
		return nil, err
	}

	outLen := int(scaled)
	if outLen == 0 {
		return []float32{}, nil
	}

	up, down := r.Ratio()
	lead, skip := alignment(r.TapsPerPhase()*up, up, down)

	// Raw output m needs input up to floor(m*down/up); pad with zeros so
	// that skip+outLen outputs exist.
	total := max(lead+len(mono), int(ceilDiv(int64(skip+outLen)*int64(down), int64(up)))+1)
	in := make([]float64, total)

	for i, v := range mono {
		in[lead+i] = float64(v)
	}

	raw := r.Process(in)

	out := make([]float32, outLen)
	for i := range out {
		out[i] = float32(raw[skip+i])
	}

	return out, nil
}

func newResampler(rate int, spec filterSpec) (*resample.Resampler, error) {
	g := gcd(SampleRate, rate)
	up, down := SampleRate/g, rate/g

	if down > maxRatioTerm {
		approx, err := resample.NewForRates(float64(rate), SampleRate, resample.WithMaxDenominator(maxRatioTerm))
		if err != nil {
			return nil, fmt.Errorf("%w: %d Hz: %w", ErrSampleRate, rate, err)
		}

		up, down = approx.Ratio()
	}

	taps := int(math.Ceil(spec.length * float64(max(up, down)) / float64(up)))
	if up%2 == 1 && taps%2 == 0 {
		taps++ // odd prototype length: integer group delay (F-010)
	}

	r, err := resample.NewRational(up, down,
		resample.WithTapsPerPhase(taps), resample.WithCutoffScale(spec.cutoff), resample.WithKaiserBeta(spec.beta))
	if err != nil {
		return nil, fmt.Errorf("%w: %d Hz: %w", ErrSampleRate, rate, err)
	}

	return r, nil
}

// alignment removes the group delay of an n-tap linear-phase prototype
// without interpolating. Raw output m of algo-dsp's causal polyphase filter
// sits at upsampled time m*down - (n-1)/2. Prepending lead zero samples
// moves the input by lead*up, so with (floor((n-1)/2) + lead*up) divisible
// by down, raw output skip+j lands on output time j exactly for odd n and
// half an upsampled step late for even n: 0.5/down output samples, which is
// half a sample from 8 kHz and 0.0011 from 22.05 or 44.1 kHz. algo-dsp's
// ProcessAligned interpolates the fraction linearly instead, which took
// another 14 dB off a 7 kHz tone (F-010).
func alignment(n, up, down int) (lead, skip int) {
	half := (n - 1) / 2
	for (half+lead*up)%down != 0 {
		lead++
	}

	return lead, (half + lead*up) / down
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}

	return a
}

func ceilDiv(a, b int64) int64 {
	return (a + b - 1) / b
}
