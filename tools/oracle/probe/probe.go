// Package probe synthesises the encoder probe signals: digital silence, DC, quiet tones and bursts, low-level noise
// and single-sample impulses, where the stem output is small or follows a known pattern, so that terms the speech
// clips bury (pe_gate, the log floor, the framing) dominate the engine's needle_embed rows (docs/FINDINGS.md F-019,
// F-020).
//
// The oracle tool's probes subcommand records the engine's rows for Set; the encoder tests regenerate the same
// samples from the committed manifest's parameters and check them against its SHA-256. The package has no engine
// dependency, so it carries no build tag.
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
	Purpose    string    `json:"purpose"`
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

// Signal synthesises the probe's samples.
func (s Spec) Signal() ([]float32, error) {
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
	case Noise, Decay:
		s.noise(x)
	case Impulse:
		err = s.impulses(x)
	default:
		err = fmt.Errorf("probe %s: unknown kind %q", s.Name, s.Kind)
	}

	if err != nil {
		return nil, err
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

func (s Spec) noise(x []float32) {
	g := newGauss(s.Seed)

	for i := range x {
		level := s.Amplitude
		if s.Kind == Decay {
			level = math.Pow(10, (s.FromDB+(s.ToDB-s.FromDB)*float64(i)/float64(s.Samples))/20)
		}

		x[i] = float32(level * g.next())
	}
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
