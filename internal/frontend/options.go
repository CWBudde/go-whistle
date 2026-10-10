package frontend

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// The front end's open questions (PLAN.md Phase 3, U1) as typed options. The
// published facts (16 kHz, a 400-sample window zero-padded to 512, hop 160,
// 80 mels from .cact tensor 679, 3000 frames per 30 s) are fixed; everything
// the sources leave open is one axis below. Phase 4 resolves them against the
// encoder oracle by iterating Variants (docs/FINDINGS.md F-012).
//
// Every axis starts at 1 so that a zero Options is invalid: no hypothesis is
// a default until Phase 4 picks one.

// ErrOptions reports an Options value with an unset or unknown axis.
var ErrOptions = errors.New("frontend: invalid options")

// Window is the 400-sample analysis window.
type Window int

const (
	// HannPeriodic is 0.5 - 0.5·cos(2πk/400) (torch.hann_window default,
	// scipy get_window, Whisper).
	HannPeriodic Window = iota + 1
	// HannSymmetric is 0.5 - 0.5·cos(2πk/399) (np.hanning, NeMo's
	// torch.hann_window(periodic=False)).
	HannSymmetric
)

// Padding is the framing: where frame i sits and what fills samples outside
// the signal.
type Padding int

const (
	// PadReflect centres frame i on sample i·160 (window over
	// [i·160−200, i·160+200)) and mirrors the signal at its edges without
	// repeating the edge sample (numpy/torch "reflect", librosa, Whisper).
	PadReflect Padding = iota + 1
	// PadZero centres frame i like PadReflect but reads zeros outside the
	// signal (torch.stft pad_mode="constant", NeMo).
	PadZero
	// PadNone starts frame i at sample i·160 (window over
	// [i·160, i·160+400)) and keeps only frames inside the signal (Kaldi
	// snip_edges).
	PadNone
)

// Spectrum is what the mel bank is applied to.
type Spectrum int

const (
	// Power is |X|² (Whisper, NeMo mag_power 2).
	Power Spectrum = iota + 1
	// Magnitude is |X|.
	Magnitude
)

// Log is the compression applied to the mel energies.
type Log int

const (
	// LogLn is ln(max(x, 1e-10)).
	LogLn Log = iota + 1
	// LogLog10 is log10(max(x, 1e-10)).
	LogLog10
	// LogAddEps is ln(x + 2⁻²⁴) (NeMo log_zero_guard_type "add").
	LogAddEps
	// LogWhisper is Whisper's log_mel_spectrogram: y = log10(max(x, 1e-10)),
	// y = max(y, max(y) − 8) over the whole clip, then (y + 4) / 4.
	LogWhisper
	// LogSoft76 is a soft knee 76 dB below the clip's peak: y = log10(x + ε)
	// with ε = 10^−7.6 · max(x) over the whole clip, where LogWhisper clamps
	// hard at 80 dB. The depth is fitted on the corpus (docs/FINDINGS.md
	// F-020). It is scale invariant, as the engine is on quiet tones and
	// noise; an all-zero clip takes ε = 1e-10.
	LogSoft76
	// LogSoft80 is LogSoft76 with the knee at Whisper's 80 dB, ε = 10^−8 ·
	// max(x).
	LogSoft80
)

// Norm is the normalisation after the log.
type Norm int

const (
	// NormNone leaves the log mel as it is.
	NormNone Norm = iota + 1
	// NormChannel standardises each mel channel over time:
	// (x − mean) / (std + 1e-5), std with n−1 (NeMo "per_feature").
	NormChannel
	// NormGlobal standardises with one mean and std over all frames and
	// channels, same formula.
	NormGlobal
)

// Band is how "band-limited to 250-3500 Hz" (cactuscompute.com/whistle) is
// read. The shipped bank (tensor 679) covers the full 0–8 kHz, so a band
// limit, if any, is applied around it.
type Band int

const (
	// BandFull ignores the claim: all 257 bins and 80 channels.
	BandFull Band = iota + 1
	// BandBins zeroes the spectrum at FFT bins outside [250, 3500] Hz before
	// the mel bank: an ideal band-pass pre-filter in the frequency domain.
	BandBins
	// BandChannels zeroes the mel energy of the channels whose centre
	// frequency lies outside [250, 3500] Hz (channels 6–58 stay); after the
	// log they sit at the floor.
	BandChannels
)

// Band edges of the band-limit hypotheses, in Hz.
const (
	BandLowHz  = 250
	BandHighHz = 3500
)

// Preemphasis is an optional first-order high-pass before the STFT. Not in
// PLAN.md's list: added because NeMo's front end, which matches every
// published Whistle front-end constant, applies it by default (F-012).
type Preemphasis int

const (
	// PreemphNone feeds the samples unchanged.
	PreemphNone Preemphasis = iota + 1
	// Preemph097 is y[0] = x[0], y[t] = x[t] − 0.97·x[t−1] (NeMo preemph).
	Preemph097
)

// Dither is optional noise added to the samples before the STFT, as NeMo's
// preprocessor does in training (config "dither" 1e-5). Added to test
// whether the engine dithers; the silence probes rule it out (F-020).
type Dither int

const (
	// DitherNone feeds the samples unchanged.
	DitherNone Dither = iota + 1
	// Dither1e5 adds Gaussian noise with σ = 1e-5 from a fixed seed, so a
	// clip's features are reproducible.
	Dither1e5
)

// ditherSigma and ditherSeed define Dither1e5.
const (
	ditherSigma = 1e-5
	ditherSeed  = 0x5eed
)

// Options pick one hypothesis per axis.
type Options struct {
	Window      Window
	Padding     Padding
	Spectrum    Spectrum
	Log         Log
	Norm        Norm
	Band        Band
	Preemphasis Preemphasis
	Dither      Dither
}

// axis describes one Options field for Validate, String, ParseOptions and
// Variants.
type axis struct {
	key   string
	names []string // names[v-1] is value v
	field func(*Options) *int
}

// noneName is the "none" value of the pad, norm and pre axes.
const noneName = "none"

var axes = []axis{
	{"win", []string{"hann-periodic", "hann-symmetric"}, func(o *Options) *int { return (*int)(&o.Window) }},
	{"pad", []string{"reflect", "zero", noneName}, func(o *Options) *int { return (*int)(&o.Padding) }},
	{"spec", []string{"power", "magnitude"}, func(o *Options) *int { return (*int)(&o.Spectrum) }},
	{"log", []string{"ln", "log10", "ln-eps", "whisper", "soft-76db", "soft-80db"}, func(o *Options) *int { return (*int)(&o.Log) }},
	{"norm", []string{noneName, "channel", "global"}, func(o *Options) *int { return (*int)(&o.Norm) }},
	{"band", []string{"full", "bins", "channels"}, func(o *Options) *int { return (*int)(&o.Band) }},
	{"pre", []string{noneName, "0.97"}, func(o *Options) *int { return (*int)(&o.Preemphasis) }},
	{"dither", []string{noneName, "1e-5"}, func(o *Options) *int { return (*int)(&o.Dither) }},
}

// Validate reports an unset or unknown value on any axis.
func (o Options) Validate() error {
	for _, a := range axes {
		v := *a.field(&o)
		if v < 1 || v > len(a.names) {
			return fmt.Errorf("%w: %s = %d", ErrOptions, a.key, v)
		}
	}

	return nil
}

// String is the variant's stable key, e.g.
// "win=hann-periodic pad=reflect spec=power log=whisper norm=none band=full pre=none dither=none".
// ParseOptions reads it back.
func (o Options) String() string {
	parts := make([]string, len(axes))

	for i, a := range axes {
		v := *a.field(&o)

		name := strconv.Itoa(v)
		if v >= 1 && v <= len(a.names) {
			name = a.names[v-1]
		}

		parts[i] = a.key + "=" + name
	}

	return strings.Join(parts, " ")
}

// ParseOptions reads a key written by Options.String: every axis exactly
// once, in String's order.
func ParseOptions(key string) (Options, error) {
	var o Options

	fields := strings.Fields(key)
	if len(fields) != len(axes) {
		return Options{}, fmt.Errorf("%w: %q has %d fields, want %d", ErrOptions, key, len(fields), len(axes))
	}

	for i, f := range fields {
		a := axes[i]

		name, ok := strings.CutPrefix(f, a.key+"=")
		if !ok {
			return Options{}, fmt.Errorf("%w: field %d of %q is not %s=…", ErrOptions, i, key, a.key)
		}

		v := 0

		for j, n := range a.names {
			if n == name {
				v = j + 1
			}
		}

		if v == 0 {
			return Options{}, fmt.Errorf("%w: unknown %s %q", ErrOptions, a.key, name)
		}

		*a.field(&o) = v
	}

	return o, nil
}

// Variants enumerates every combination of the axes, 2592 in all, in a
// stable order: the last axis (dither) varies fastest and the first (win)
// slowest, each in its constant order. Phase 4's probe harness iterates it.
func Variants() []Options {
	n := 1
	for _, a := range axes {
		n *= len(a.names)
	}

	out := make([]Options, n)

	for i := range out {
		rest := i

		for _, a := range slices.Backward(axes) {
			k := len(a.names)
			*a.field(&out[i]) = rest%k + 1
			rest /= k
		}
	}

	return out
}

// NeMo is the hypothesis built from NeMo's AudioToMelSpectrogramPreprocessor
// in its Conformer configuration, which shares every published Whistle
// front-end constant (512/400/160, 80 Slaney mels with Slaney norm, dither
// 1e-5, "normalised per channel"); F-012.
func NeMo() Options {
	return Options{
		Window: HannSymmetric, Padding: PadZero, Spectrum: Power, Log: LogAddEps,
		Norm: NormChannel, Band: BandFull, Preemphasis: Preemph097, Dither: DitherNone,
	}
}

// Whisper is Whisper's log_mel_spectrogram framing and compression on the
// Whistle geometry (F-012).
func Whisper() Options {
	return Options{
		Window: HannPeriodic, Padding: PadReflect, Spectrum: Power, Log: LogWhisper,
		Norm: NormNone, Band: BandFull, Preemphasis: PreemphNone, Dither: DitherNone,
	}
}
