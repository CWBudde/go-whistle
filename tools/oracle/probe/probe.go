// Package probe synthesises the encoder probe signals: digital silence, DC, quiet tones and bursts, low-level noise
// and single-sample impulses, where the stem output is small or follows a known pattern, so that terms the speech
// clips bury (pe_gate, the log floor, the framing) dominate the engine's needle_embed rows (docs/FINDINGS.md F-019,
// F-020).
//
// It also synthesises the decoding oracle's silence sweep and length limits (SilenceSet, LimitSet; F-025): pink noise,
// noise with a louder section, and a corpus speech clip at a gain, padded, under noise or on the 16-bit grid.
//
// The oracle tool's probes subcommand records the engine's rows for Set; the encoder tests regenerate the same
// samples from the committed manifest's parameters and check them against its SHA-256. The package has no engine
// dependency, so it carries no build tag.
//
// Every sum of products is written with explicit float64 conversions, which the Go spec defines to round and so keep
// the compiler from fusing a multiply and an add on one architecture and not on another.
package probe

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand/v2"
)

// Geometry of the engine's framing (docs/FINDINGS.md F-012, F-016).
const (
	SampleRate = 16000
	Hop        = 160         // samples per mel frame
	Stride     = 8           // mel frames per encoder frame
	reach      = 7*Hop + 200 // samples an encoder frame's stem input reaches on either side of 1280·t
)

// Kinds of probe signal.
const (
	Silence = "silence" // all zeros
	DC      = "dc"      // a constant Amplitude
	Tone    = "tone"    // Amplitude·sin(2π·Hz·t)
	Burst   = "burst"   // the tone inside [From, To), zeros elsewhere
	Noise   = "noise"   // Gaussian, σ = Amplitude
	Decay   = "decay"   // Gaussian noise whose level falls linearly in dB from FromDB to ToDB
	Impulse = "impulse" // Amplitudes[i] at sample Offsets[i], zeros elsewhere
	Pink    = "pink"    // pink (1/f) noise, RMS = Amplitude
	Step    = "step"    // Gaussian noise, σ = Amplitude, times Gain inside [From, To)
	Speech  = "speech"  // Gain·Clip[ClipFrom:ClipTo] at PadBefore, zeros around it, plus Gaussian noise σ = Amplitude
)

// Spec is one probe: its signal parameters and why it discriminates.
type Spec struct {
	Name       string    `json:"name"`
	Kind       string    `json:"kind"`
	Samples    int       `json:"samples"`
	Amplitude  float64   `json:"amplitude,omitempty"`
	Hz         float64   `json:"hz,omitempty"`
	From       int       `json:"from,omitempty"`
	To         int       `json:"to,omitempty"`
	FromDB     float64   `json:"from_db,omitempty"`
	ToDB       float64   `json:"to_db,omitempty"`
	Offsets    []int     `json:"offsets,omitempty"`
	Amplitudes []float64 `json:"amplitudes,omitempty"`
	Seed       uint64    `json:"seed,omitempty"`

	// Gain is the exact linear gain of a Step's louder section or of a Speech clip; GainDB is it in dB, for reading.
	Gain   float64 `json:"gain,omitempty"`
	GainDB float64 `json:"gain_db,omitempty"`

	// Clip names the corpus clip a Speech signal reads (testdata/audio/<Clip>.wav, 16-bit, decoded as int16/32768),
	// and [ClipFrom, ClipTo) the samples taken from it; they start at PadBefore.
	Clip      string `json:"clip,omitempty"`
	ClipFrom  int    `json:"clip_from,omitempty"`
	ClipTo    int    `json:"clip_to,omitempty"`
	PadBefore int    `json:"pad_before,omitempty"`

	// SNRDB labels a Speech signal under noise: Amplitude is the clip excerpt's RMS times 10^(−SNRDB/20).
	SNRDB float64 `json:"snr_db,omitempty"`

	// Quantize16 rounds the result to the 16-bit grid, as a 16-bit WAV of it would hold it: round(32768·x) clamped to
	// [−32768, 32767], over 32768.
	Quantize16 bool `json:"quantize16,omitempty"`

	Purpose string `json:"purpose"`
}

// Frames is the engine's encoder frame count for n samples, ceil(floor(n/160)/8) (F-012).
func Frames(n int) int { return (n/Hop + Stride - 1) / Stride }

// InteriorFrames is the number of encoder frames whose stem input lies inside the clip: frame t reads the mel frames
// 8t−7 … 8t+7, whose windows span samples [1280t − 1320, 1280t + 1320). Under zero centring, a constant signal gives
// the same stem output on every interior frame. Every frame still sees the clip edges through the blocks' depthwise
// convs and the global attention.
func InteriorFrames(n int) int {
	k := 0

	for t := range Frames(n) {
		if Stride*Hop*t-reach >= 0 && Stride*Hop*t+reach <= n {
			k++
		}
	}

	return k
}

// Signal synthesises the probe's samples. A Speech probe needs its clip: use SignalWith.
func (s Spec) Signal() ([]float32, error) { return s.SignalWith(nil) }

// SignalWith synthesises the probe's samples; clip holds the samples of s.Clip and is read only by Speech probes.
func (s Spec) SignalWith(clip []float32) ([]float32, error) {
	if s.Samples <= 0 {
		return nil, fmt.Errorf("probe %s: %d samples", s.Name, s.Samples)
	}

	x := make([]float32, s.Samples)

	var err error

	switch s.Kind {
	case Silence:
	case DC:
		for i := range x {
			x[i] = float32(s.Amplitude)
		}
	case Tone, Burst:
		err = s.tone(x)
	case Noise, Decay, Step:
		err = s.noise(x)
	case Impulse:
		err = s.impulses(x)
	case Pink:
		s.pink(x)
	case Speech:
		err = s.speech(x, clip)
	default:
		err = fmt.Errorf("probe %s: unknown kind %q", s.Name, s.Kind)
	}

	if err != nil {
		return nil, err
	}

	if s.Quantize16 {
		quantize16(x)
	}

	return x, nil
}

func (s Spec) tone(x []float32) error {
	from, to := 0, s.Samples
	if s.Kind == Burst {
		from, to = s.From, s.To
	}

	if from < 0 || to > s.Samples || from >= to {
		return fmt.Errorf("probe %s: burst [%d, %d) outside %d samples", s.Name, from, to, s.Samples)
	}

	for i := from; i < to; i++ {
		x[i] = float32(s.Amplitude * math.Sin(2*math.Pi*s.Hz*float64(i)/SampleRate))
	}

	return nil
}

func (s Spec) noise(x []float32) error {
	if s.Kind == Step && (s.From < 0 || s.To > s.Samples || s.From >= s.To || s.Gain <= 0) {
		return fmt.Errorf("probe %s: step [%d, %d) × %g outside %d samples", s.Name, s.From, s.To, s.Gain, s.Samples)
	}

	g := newGauss(s.Seed)

	for i := range x {
		level := s.Amplitude

		switch {
		case s.Kind == Decay:
			level = math.Pow(10, (s.FromDB+(s.ToDB-s.FromDB)*float64(i)/float64(s.Samples))/20)
		case s.Kind == Step && i >= s.From && i < s.To:
			level = float64(s.Amplitude * s.Gain)
		}

		x[i] = float32(level * g.next())
	}

	return nil
}

// pinkWarmup is how many samples the pink filter runs before its output is kept, so that its slowest pole (time
// constant 877 samples) has settled and the noise starts at its steady level.
const pinkWarmup = SampleRate

// pink filters the Gaussian stream with Paul Kellet's refined 1/f filter (−3 dB per octave within ±0.05 dB from
// 9.2 Hz up) and scales the result to RMS Amplitude.
func (s Spec) pink(x []float32) {
	g := newGauss(s.Seed)
	y := make([]float64, len(x))

	var b0, b1, b2, b3, b4, b5, b6 float64

	for i := -pinkWarmup; i < len(x); i++ {
		w := g.next()
		b0 = float64(0.99886*b0) + float64(w*0.0555179)
		b1 = float64(0.99332*b1) + float64(w*0.0750759)
		b2 = float64(0.96900*b2) + float64(w*0.1538520)
		b3 = float64(0.86650*b3) + float64(w*0.3104856)
		b4 = float64(0.55000*b4) + float64(w*0.5329522)
		b5 = float64(-0.7616*b5) - float64(w*0.0168980)
		v := b0 + b1 + b2 + b3 + b4 + b5 + b6 + float64(w*0.5362)
		b6 = w * 0.115926

		if i >= 0 {
			y[i] = v
		}
	}

	scale := s.Amplitude / RMS(y)
	for i, v := range y {
		x[i] = float32(v * scale)
	}
}

// speech places Gain·clip[ClipFrom:ClipTo] at PadBefore and adds Gaussian noise of σ Amplitude to every sample.
func (s Spec) speech(x, clip []float32) error {
	if s.ClipFrom < 0 || s.ClipTo > len(clip) || s.ClipFrom >= s.ClipTo || s.Gain <= 0 {
		return fmt.Errorf("probe %s: clip %q [%d, %d) × %g of %d samples", s.Name, s.Clip, s.ClipFrom, s.ClipTo,
			s.Gain, len(clip))
	}

	if s.PadBefore < 0 || s.PadBefore+s.ClipTo-s.ClipFrom > s.Samples {
		return fmt.Errorf("probe %s: %d + %d clip samples exceed %d", s.Name, s.PadBefore, s.ClipTo-s.ClipFrom,
			s.Samples)
	}

	y := make([]float64, len(x))
	for i, v := range clip[s.ClipFrom:s.ClipTo] {
		y[s.PadBefore+i] = float64(float64(v) * s.Gain)
	}

	if s.Amplitude > 0 {
		g := newGauss(s.Seed)
		for i := range y {
			y[i] += float64(s.Amplitude * g.next())
		}
	}

	for i, v := range y {
		x[i] = float32(v)
	}

	return nil
}

// quantize16 rounds x to the 16-bit grid: round(32768·v) clamped to [−32768, 32767], over 32768.
func quantize16(x []float32) {
	for i, v := range x {
		x[i] = float32(max(-32768, min(32767, math.Round(float64(v)*32768))) / 32768)
	}
}

// RMS is the root mean square of x, summed in index order in float64.
func RMS[T float32 | float64](x []T) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(float64(v) * float64(v))
	}

	return math.Sqrt(sum / float64(len(x)))
}

func (s Spec) impulses(x []float32) error {
	if len(s.Offsets) != len(s.Amplitudes) {
		return fmt.Errorf("probe %s: %d offsets, %d amplitudes", s.Name, len(s.Offsets), len(s.Amplitudes))
	}

	for i, o := range s.Offsets {
		if o < 0 || o >= s.Samples {
			return fmt.Errorf("probe %s: impulse at %d outside %d samples", s.Name, o, s.Samples)
		}

		x[o] = float32(s.Amplitudes[i])
	}

	return nil
}

// SHA256 is the hex digest of x as little-endian float32, the manifest's check that a regenerated signal equals the
// one the engine embedded.
func SHA256(x []float32) string {
	b := make([]byte, 4*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}

	sum := sha256.Sum256(b)

	return hex.EncodeToString(sum[:])
}

// gauss is Box–Muller over a PCG stream: unlike rand.NormFloat64, whose algorithm the standard library may change,
// it depends only on PCG's specified output.
type gauss struct {
	rng   *rand.PCG
	spare float64
	have  bool
}

func newGauss(seed uint64) *gauss { return &gauss{rng: rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)} }

func (g *gauss) uniform() float64 { return (float64(g.rng.Uint64()>>11) + 0.5) / (1 << 53) }

func (g *gauss) next() float64 {
	if g.have {
		g.have = false

		return g.spare
	}

	r, a := math.Sqrt(-2*math.Log(g.uniform())), 2*math.Pi*g.uniform()
	g.spare, g.have = r*math.Sin(a), true

	return r * math.Cos(a)
}
