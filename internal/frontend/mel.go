package frontend

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/cwbudde/go-whistle/internal/weights"
)

// Mels is the number of mel channels (config mels).
const Mels = 80

// MelBankName is the filterbank's name in a weights.Source (.cact tensor 679,
// docs/FINDINGS.md F-006). Both backends serve it bit-identically (F-007).
const MelBankName = "audio.mel_filters"

// Floors and constants of the log and normalisation hypotheses (F-012).
const (
	logFloor   = 1e-10           // LogLn, LogLog10, LogWhisper (Whisper's clamp)
	logEps     = 1.0 / (1 << 24) // LogAddEps (NeMo log_zero_guard_value 2⁻²⁴)
	whisperTop = 8.0             // LogWhisper keeps the top 8 decades (80 dB)
	soft76     = 7.6             // LogSoft76's knee in decades below the peak (F-020)
	soft80     = 8.0             // LogSoft80's
	normEps    = 1e-5            // NormChannel, NormGlobal (NeMo CONSTANT)
	preemph    = 0.97            // Preemph097
)

// MelBank is the (Bins, Mels) filterbank, row-major as .cact stores it:
// weight[b*Mels + m] maps FFT bin b to channel m.
type MelBank struct {
	weight []float32
	lo, hi []int // channel m's non-zero bins are [lo[m], hi[m])
}

// NewMelBank wraps a (Bins, Mels) filterbank. It keeps its own copy.
func NewMelBank(values []float32, shape []int) (*MelBank, error) {
	if !slices.Equal(shape, []int{Bins, Mels}) || len(values) != Bins*Mels {
		return nil, fmt.Errorf("frontend: mel bank shape %v (%d values), want [%d %d]", shape, len(values), Bins, Mels)
	}

	b := &MelBank{weight: slices.Clone(values), lo: make([]int, Mels), hi: make([]int, Mels)}

	for m := range Mels {
		b.lo[m], b.hi[m] = Bins, 0

		for k := range Bins {
			w := b.weight[k*Mels+m]
			if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) || w < 0 {
				return nil, fmt.Errorf("frontend: mel bank weight [%d, %d] = %v", k, m, w)
			}

			if w != 0 {
				b.lo[m], b.hi[m] = min(b.lo[m], k), k+1
			}
		}

		b.hi[m] = max(b.hi[m], b.lo[m])
	}

	return b, nil
}

// LoadMelBank reads MelBankName from a weights source (.cact or
// safetensors).
func LoadMelBank(src weights.Source) (*MelBank, error) {
	values, shape, err := src.Tensor(MelBankName)
	if err != nil {
		return nil, fmt.Errorf("frontend: %s: %w", MelBankName, err)
	}

	return NewMelBank(values, shape)
}

// apply returns the mel energies of frames spectra, frame-major (Mels per
// frame), accumulated in float64.
func (b *MelBank) apply(spec []float32, frames int) []float64 {
	out := make([]float64, frames*Mels)

	for t := range frames {
		row := spec[t*Bins : (t+1)*Bins]

		for m := range Mels {
			var sum float64
			for k := b.lo[m]; k < b.hi[m]; k++ {
				sum += float64(row[k]) * float64(b.weight[k*Mels+m])
			}

			out[t*Mels+m] = sum
		}
	}

	return out
}

// Slaney's mel scale (librosa htk=False), as internal/cact SlaneyMel builds
// tensor 679; only the channel centres are needed here (BandChannels).
func slaneyHzToMel(hz float64) float64 {
	if hz >= 1000 {
		return 15 + math.Log(hz/1000)/(math.Log(6.4)/27)
	}

	return hz * 3 / 200
}

func slaneyMelToHz(mel float64) float64 {
	if mel >= 15 {
		return 1000 * math.Exp(math.Log(6.4)/27*(mel-15))
	}

	return mel * 200 / 3
}

// ChannelCentres are the centre frequencies in Hz of tensor 679's channels:
// librosa mel_frequencies(Mels+2, 0, SampleRate/2)[1:-1].
func ChannelCentres() []float64 {
	step := slaneyHzToMel(SampleRate/2) / (Mels + 1)

	out := make([]float64, Mels)
	for m := range out {
		out[m] = slaneyMelToHz(float64(m+1) * step)
	}

	return out
}

// Features is a log-mel spectrogram: Frames rows of Mels values, row-major,
// time × mel as the conv stem consumes it.
type Features struct {
	Frames int
	Data   []float32 // Data[t*Mels + m]
}

// Frontend turns 16 kHz mono samples into log-mel features under one
// Options hypothesis. It is safe for concurrent use.
type Frontend struct {
	opts    Options
	bank    *MelBank
	spec    *spectrogram
	binKeep []bool // BandBins: FFT bins inside the band
	chKeep  []bool // BandChannels: channels whose centre is inside the band
}

// New builds a front end over bank for one hypothesis.
func New(bank *MelBank, opts Options) (*Frontend, error) {
	err := opts.Validate()
	if err != nil {
		return nil, err
	}

	if bank == nil {
		return nil, fmt.Errorf("%w: no mel bank", ErrOptions)
	}

	spec, err := newSpectrogram(opts.Window, opts.Padding)
	if err != nil {
		return nil, err
	}

	f := &Frontend{opts: opts, bank: bank, spec: spec}

	switch opts.Band {
	case BandBins:
		f.binKeep = make([]bool, Bins)
		for k := range f.binKeep {
			hz := float64(k) * SampleRate / NFFT
			f.binKeep[k] = hz >= BandLowHz && hz <= BandHighHz
		}
	case BandChannels:
		f.chKeep = make([]bool, Mels)
		for m, hz := range ChannelCentres() {
			f.chKeep[m] = hz >= BandLowHz && hz <= BandHighHz
		}
	}

	return f, nil
}

// Options returns the hypothesis the front end runs.
func (f *Frontend) Options() Options { return f.opts }

// FrameCount is the number of frames for n samples (FrameCount with the
// front end's padding).
func (f *Frontend) FrameCount(n int) int { return FrameCount(n, f.opts.Padding) }

// Spectrogram returns what the mel bank is applied to: the power or
// magnitude spectrum after dither, pre-emphasis and, for BandBins, the bin
// mask. It holds FrameCount(len(x)) frames of Bins values, frame-major.
func (f *Frontend) Spectrogram(x []float32) ([]float32, int, error) {
	if f.opts.Dither == Dither1e5 {
		x = dither(x)
	}

	if f.opts.Preemphasis == Preemph097 && len(x) > 1 {
		y := make([]float32, len(x))

		y[0] = x[0]
		for t := 1; t < len(x); t++ {
			y[t] = x[t] - preemph*x[t-1]
		}

		x = y
	}

	spec, frames, err := f.spec.spectrum(x, f.opts.Spectrum == Magnitude)
	if err != nil {
		return nil, 0, err
	}

	if f.binKeep != nil {
		for t := range frames {
			for k, keep := range f.binKeep {
				if !keep {
					spec[t*Bins+k] = 0
				}
			}
		}
	}

	return spec, frames, nil
}

// LogMel returns the log-mel features of x under the front end's
// hypothesis: spectrum, mel bank, band mask, log, normalisation.
func (f *Frontend) LogMel(x []float32) (Features, error) {
	spec, frames, err := f.Spectrogram(x)
	if err != nil {
		return Features{}, err
	}

	mel := f.bank.apply(spec, frames)

	if f.chKeep != nil {
		for i := range mel {
			if !f.chKeep[i%Mels] {
				mel[i] = 0
			}
		}
	}

	applyLog(mel, f.opts.Log)
	normalise(mel, frames, f.opts.Norm)

	out := make([]float32, len(mel))
	for i, v := range mel {
		out[i] = float32(v)
	}

	return Features{Frames: frames, Data: out}, nil
}

// dither returns x plus Gaussian noise of σ ditherSigma from a fixed seed
// (Dither1e5).
func dither(x []float32) []float32 {
	rng := rand.New(rand.NewPCG(ditherSeed, ditherSeed)) // #nosec G404 -- reproducible noise, not a secret.
	y := make([]float32, len(x))

	for t, v := range x {
		y[t] = v + float32(ditherSigma*rng.NormFloat64())
	}

	return y
}

// applyLog compresses mel energies in place (Log).
func applyLog(mel []float64, l Log) {
	switch l {
	case LogLn:
		for i, v := range mel {
			mel[i] = math.Log(max(v, logFloor))
		}
	case LogLog10:
		for i, v := range mel {
			mel[i] = math.Log10(max(v, logFloor))
		}
	case LogAddEps:
		for i, v := range mel {
			mel[i] = math.Log(v + logEps)
		}
	case LogWhisper:
		top := math.Inf(-1)

		for i, v := range mel {
			mel[i] = math.Log10(max(v, logFloor))
			top = max(top, mel[i])
		}

		for i, v := range mel {
			mel[i] = (max(v, top-whisperTop) + 4) / 4
		}
	case LogSoft76:
		softLog(mel, soft76)
	case LogSoft80:
		softLog(mel, soft80)
	}
}

// softLog is log10(x + ε) in place with ε = 10^−depth times the largest
// energy, a soft knee depth decades below the peak; an all-zero mel takes
// ε = logFloor (LogSoft76, LogSoft80).
func softLog(mel []float64, depth float64) {
	if len(mel) == 0 {
		return
	}

	eps := math.Pow(10, -depth) * slices.Max(mel)
	if eps == 0 {
		eps = logFloor
	}

	for i, v := range mel {
		mel[i] = math.Log10(v + eps)
	}
}

// normalise standardises log mel values in place (Norm): (x − mean) /
// (std + 1e-5) with the n−1 standard deviation, per channel over time or
// over everything. With fewer than two values the deviation counts as 0.
func normalise(mel []float64, frames int, n Norm) {
	switch n {
	case NormChannel:
		for m := range Mels {
			standardise(mel, m, Mels, frames)
		}
	case NormGlobal:
		standardise(mel, 0, 1, len(mel))
	}
}

// standardise standardises mel[start + i*stride] for i < count.
func standardise(mel []float64, start, stride, count int) {
	if count == 0 {
		return
	}

	var sum float64
	for i := range count {
		sum += mel[start+i*stride]
	}

	mean := sum / float64(count)

	var dev float64

	for i := range count {
		d := mel[start+i*stride] - mean
		dev += d * d
	}

	std := 0.0
	if count > 1 {
		std = math.Sqrt(dev / float64(count-1))
	}

	for i := range count {
		j := start + i*stride
		mel[j] = (mel[j] - mean) / (std + normEps)
	}
}
