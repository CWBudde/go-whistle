// Package silence is the engine's silence gate (PLAN.md 6.6, U10). The
// engine returns the empty transcript, without decoding, for a clip whose
// speech-band level hardly changes over time: digital silence, steady noise
// at any level, speech buried in noise. The gate compares levels within
// the clip, so it does not depend on the clip's loudness.
//
// The measure was fitted to the engine's verdicts on the 342 signals of the
// 6.1 silence sweep and length limits (docs/FINDINGS.md F-031): the front
// end's power spectrogram (400-sample Hann windows centred every 160
// samples, zero padded), summed per frame over the mel channels 6 to 57
// (centres 261–3307 Hz, the "250-3500 Hz" band of cactuscompute.com/whistle,
// which the encoder's input does not have, F-018), in dB; then the spread
// between the 99th and the 10th percentile of those frame levels. A clip
// whose spread is at most 5 dB is silence.
package silence

import (
	"fmt"
	"math"
	"slices"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/frontend"
)

// The gate's constants (F-031).
const (
	// FirstChannel and LastChannel are the mel channels the gate sums,
	// both included. The sweep does not decide whether channel 5 (centre
	// 223 Hz) or 58 (3437 Hz) belongs to the band as well.
	FirstChannel = 6
	LastChannel  = 57
	// LowPercentile and HighPercentile are the nearest-rank percentiles
	// of the frame levels whose difference is the spread.
	LowPercentile  = 10
	HighPercentile = 99
	// ThresholdDB is the largest spread that is still silence.
	ThresholdDB = 5.0
)

// Framing is the front end whose power spectrogram the gate reads: the
// encoder's (F-020); only the window, padding, spectrum, pre-emphasis and
// dither matter here.
var Framing = frontend.Options{
	Window: frontend.HannSymmetric, Padding: frontend.PadZero, Spectrum: frontend.Power,
	Log: frontend.LogSoft76, Norm: frontend.NormChannel, Band: frontend.BandFull,
	Preemphasis: frontend.PreemphNone, Dither: frontend.DitherNone,
}

// Gate decides whether a clip is silence. It is safe for concurrent use.
type Gate struct {
	fe     *frontend.Frontend
	weight []float64 // per FFT bin, the summed weight of the in-band mel channels
}

// New builds the gate over librosa's Slaney mel bank, which equals the
// model's audio.mel_filters bit for bit (F-006).
func New() (*Gate, error) {
	values := cact.SlaneyMel(frontend.SampleRate, frontend.NFFT, frontend.Mels)

	bank, err := frontend.NewMelBank(values, []int{frontend.Bins, frontend.Mels})
	if err != nil {
		return nil, fmt.Errorf("silence: %w", err)
	}

	fe, err := frontend.New(bank, Framing)
	if err != nil {
		return nil, fmt.Errorf("silence: %w", err)
	}

	g := &Gate{fe: fe, weight: make([]float64, frontend.Bins)}

	for m := FirstChannel; m <= LastChannel; m++ {
		for k := range frontend.Bins {
			g.weight[k] += float64(values[k*frontend.Mels+m])
		}
	}

	return g, nil
}

// Levels returns the speech-band level of every frame in dB, 10·log10 of
// the in-band mel energy; a frame without energy is −Inf. A clip has
// len(pcm) / 160 frames.
func (g *Gate) Levels(pcm []float32) ([]float64, error) {
	if g.fe.FrameCount(len(pcm)) == 0 {
		return nil, nil
	}

	spec, frames, err := g.fe.Spectrogram(pcm)
	if err != nil {
		return nil, fmt.Errorf("silence: %w", err)
	}

	out := make([]float64, frames)

	for t := range frames {
		var e float64

		for k, w := range g.weight {
			e += float64(spec[t*frontend.Bins+k]) * w
		}

		out[t] = 10 * math.Log10(e)
	}

	return out, nil
}

// Spread is the clip's p99 − p10 spread of Levels in dB.
func (g *Gate) Spread(pcm []float32) (float64, error) {
	levels, err := g.Levels(pcm)
	if err != nil {
		return 0, err
	}

	return Spread(levels), nil
}

// Silent reports whether the engine treats the clip as silence: a spread
// of at most ThresholdDB.
func (g *Gate) Silent(pcm []float32) (bool, error) {
	s, err := g.Spread(pcm)
	if err != nil {
		return false, err
	}

	return s <= ThresholdDB, nil
}

// Spread is the difference between the HighPercentile and the
// LowPercentile of levels (dB, −Inf allowed). It is 0 without levels or
// when the high percentile is −Inf (all frames, or all but the top 1 %,
// without energy), and +Inf when only the low one is.
func Spread(levels []float64) float64 {
	if len(levels) == 0 {
		return 0
	}

	sorted := slices.Clone(levels)
	slices.Sort(sorted)

	hi := Percentile(sorted, HighPercentile)
	if math.IsInf(hi, -1) {
		return 0
	}

	return hi - Percentile(sorted, LowPercentile)
}

// Percentile is the nearest-rank p-th percentile (0 < p ≤ 100) of the
// ascending values sorted: the smallest value with at least p % of the
// values at or below it.
func Percentile(sorted []float64, p int) float64 {
	k := (p*len(sorted)+99)/100 - 1

	return sorted[max(k, 0)]
}
