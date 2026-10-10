// Package encoder is Whistle's audio encoder: the log-mel front end, the
// 2-D conv stem, eight non-causal blocks wrapped in mHC and the final norm,
// mapping 16 kHz audio to one 512-wide row per 80 ms frame (needle_embed).
//
// The engine that runs it is closed, so the architecture is reverse
// engineered: every open choice is an axis of Hypothesis, and the probe
// harness (probe_test.go, build tag oracle) scores them against the
// engine's embeddings (docs/FINDINGS.md F-014).
package encoder

import (
	"fmt"
	"math"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/cq"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/needle"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Encoder runs one Hypothesis on one weight source. It is safe for
// concurrent use.
type Encoder struct {
	h      Hypothesis
	fe     *frontend.Frontend
	stem   *stem
	blocks []*block
	mhc    []*needle.MHC
	final  needle.ZCRMSNorm
	peGate float32 // σ(pe_gate)
	lanes  int
	dim    int
	qk     int
}

// Load builds the encoder for h from src, with the mel bank src holds. It
// returns an error wrapping ErrRuledOut when the tensors' shapes exclude
// h.
func Load(src weights.Source, h Hypothesis) (*Encoder, error) {
	err := h.Validate()
	if err != nil {
		return nil, err
	}

	bank, err := frontend.LoadMelBank(src)
	if err != nil {
		return nil, err
	}

	e := &Encoder{h: h}

	e.fe, err = frontend.New(bank, h.Frontend)
	if err != nil {
		return nil, err
	}

	e.stem, err = loadStem(src, h, frontend.Mels)
	if err != nil {
		return nil, err
	}

	e.dim = e.stem.dim

	err = e.loadStack(withPerms(src, h.HadaPerms), src.Config())
	if err != nil {
		return nil, err
	}

	gate, _, err := src.Tensor("pe_gate")
	if err != nil {
		return nil, fmt.Errorf("encoder: pe_gate: %w", err)
	}

	e.peGate = sigmoid(gate[0])
	e.stem.peGate = e.peGate

	e.final, err = needle.LoadZCRMSNorm(src, "enc.final_norm")
	if err != nil {
		return nil, err
	}

	return e, nil
}

// Hypothesis is the hypothesis the encoder runs.
func (e *Encoder) Hypothesis() Hypothesis { return e.h }

// Frames is the number of encoder frames for n samples.
func (e *Encoder) Frames(n int) int { return e.stem.length(e.fe.FrameCount(n)) }

// Embed runs the whole encoder on 16 kHz mono samples and returns T rows of
// 512 values, as needle_embed.
func (e *Encoder) Embed(samples []float32) ([]float32, int, error) {
	mel, err := e.fe.LogMel(samples)
	if err != nil {
		return nil, 0, err
	}

	return e.Forward(mel)
}

// Forward runs the stem and the stack on log-mel features.
func (e *Encoder) Forward(mel frontend.Features) ([]float32, int, error) {
	x, t, err := e.Stem(mel)
	if err != nil {
		return nil, 0, err
	}

	out, err := e.Stack(x, t)

	return out, t, err
}

// Stem runs the conv stem, the input scale and a stem-side pe_gate signal
// on log-mel features computed with the hypothesis' front end. The result,
// t rows of 512 values, is what Stack takes.
func (e *Encoder) Stem(mel frontend.Features) ([]float32, int, error) {
	if len(mel.Data) != mel.Frames*frontend.Mels {
		return nil, 0, fmt.Errorf("encoder: %d mel values for %d frames", len(mel.Data), mel.Frames)
	}

	in := mel.Data
	if e.h.PE == PEMelConcat || e.h.PE == PEMelInterleaved {
		in = append([]float32(nil), in...)
		addSinusoid(in, mel.Frames, frontend.Mels, e.peGate, e.h.PE.interleaved())
	}

	x, t, err := e.stem.forward(in, mel.Frames, frontend.Mels)
	if err != nil {
		return nil, 0, err
	}

	if e.h.InputScale == ScaleSqrtDim {
		s := float32(math.Sqrt(float64(e.dim)))
		for i := range x {
			x[i] *= s
		}
	}

	switch e.h.PE {
	case PEStemConcat, PEStemInterleaved:
		addSinusoid(x, t, e.dim, e.peGate, e.h.PE.interleaved())
	case PEStemUngated:
		addSinusoid(x, t, e.dim, 1, false)
	}

	return x, t, nil
}

// Stack runs the blocks (inside mHC unless MHCNone) and the output tap on
// x, t rows of 512 values; x is not written.
func (e *Encoder) Stack(x []float32, t int) ([]float32, error) {
	if len(x) != t*e.dim {
		return nil, fmt.Errorf("encoder: stack input has %d values, want %d rows of %d", len(x), t, e.dim)
	}

	rope := e.rope(t)

	// forward runs block b on u; a block-input pe_gate signal is added to
	// u first and taken out of the output again, so that the residual is
	// the block's change of u + PE.
	forward := func(b *block, u []float32) ([]float32, error) {
		if e.h.PE != PEBlockConcat && e.h.PE != PEBlockInterleaved {
			return b.forward(u, t, rope)
		}

		pe := make([]float32, len(u))
		addSinusoid(pe, t, e.dim, e.peGate, e.h.PE.interleaved())

		in := make([]float32, len(u))
		for i, v := range u {
			in[i] = v + pe[i]
		}

		out, err := b.forward(in, t, rope)
		if err != nil {
			return nil, err
		}

		for i, v := range pe {
			out[i] -= v
		}

		return out, nil
	}

	if e.h.MHC == MHCNone {
		var err error

		for _, b := range e.blocks {
			x, err = forward(b, x)
			if err != nil {
				return nil, err
			}
		}
	} else {
		stream := needle.ExpandLanes(x, e.lanes, e.dim)

		for l, b := range e.blocks {
			var err error

			stream, err = e.mhc[l].Forward(stream, t, func(u []float32) ([]float32, error) {
				return forward(b, u)
			})
			if err != nil {
				return nil, err
			}
		}

		x = needle.CollapseLanes(stream, e.lanes, e.dim)
	}

	if e.h.Output == OutFinalNorm {
		out := make([]float32, len(x))
		e.final.Apply(out, x)

		return out, nil
	}

	return append([]float32(nil), x...), nil
}

// loadStack loads the blocks and their mHC layers.
func (e *Encoder) loadStack(src weights.Source, cfg weights.Config) error {
	for l := range cfg.EncLayers {
		b, err := loadBlock(src, fmt.Sprintf("enc%02d.", l), e.h)
		if err != nil {
			return err
		}

		if b.dim != e.dim {
			return fmt.Errorf("encoder: enc%02d width %d, stem output %d", l, b.dim, e.dim)
		}

		m, err := needle.LoadMHC(src, "enc.", l)
		if err != nil {
			return err
		}

		e.blocks, e.mhc = append(e.blocks, b), append(e.mhc, m)
		e.lanes, e.qk = m.Lanes(), b.attn.QKDim()
	}

	if len(e.blocks) == 0 {
		return fmt.Errorf("encoder: config has %d encoder layers", cfg.EncLayers)
	}

	return nil
}

// rope builds the attention's rotation tables for t frames, nil for none.
func (e *Encoder) rope(t int) *needle.Rope {
	theta := 1e5
	if e.h.Rope == Rope1e4 {
		theta = 1e4
	}

	switch {
	case e.h.PE == PERopeMix:
		return needle.NewRope(e.qk, t, theta).Mix(e.peGate)
	case e.h.Rope == RopeNone:
		return nil
	default:
		return needle.NewRope(e.qk, t, theta)
	}
}

// addSinusoid adds g times the transformer sinusoid to t rows of dim
// values: frequency 10000^(−2i/dim) for pair i, laid out [sin | cos] or
// interleaved.
func addSinusoid(x []float32, t, dim int, g float32, interleaved bool) {
	half := dim / 2

	for p := range t {
		row := x[p*dim : (p+1)*dim]

		for i := range half {
			a := float64(p) * math.Pow(10000, -float64(2*i)/float64(dim))
			s, c := float32(math.Sin(a)), float32(math.Cos(a))

			if interleaved {
				row[2*i] += g * s
				row[2*i+1] += g * c
			} else {
				row[i] += g * s
				row[half+i] += g * c
			}
		}
	}
}

// permSource serves other Hadamard permutations under hada_p1/hada_p2.
type permSource struct {
	weights.Source

	p [2][]float32
}

func (s permSource) Tensor(name string) ([]float32, []int, error) {
	switch name {
	case "hada_p1":
		return s.p[0], []int{len(s.p[0])}, nil
	case "hada_p2":
		return s.p[1], []int{len(s.p[1])}, nil
	default:
		return s.Source.Tensor(name)
	}
}

// Matrix serves the packed matrices of the wrapped source, if any
// (weights.Packer): the wrapper only replaces the permutations.
func (s permSource) Matrix(name string) (*cq.Matrix, bool, error) {
	return weights.PackedMatrix(s.Source, name)
}

// withPerms wraps src so that LoadHadamardMLP reads the permutations of
// the PermsSplit or PermsIdentity hypothesis (U5).
func withPerms(src weights.Source, h HadaPerms) weights.Source {
	if h == PermsShared {
		return src
	}

	shape, err := src.Shape("hada_p1")
	if err != nil || len(shape) != 1 {
		return src // LoadHadamardMLP reports it
	}

	n := shape[0]
	ps := permSource{Source: src}

	for k, name := range []string{"hada_p1", "hada_p2"} {
		p := make([]float32, n)

		if h == PermsIdentity {
			for i := range p {
				p[i] = float32(i)
			}
		} else {
			seed, _ := cact.HadaPermSeed(name)
			lo, hi := cact.NumpyPermutation(seed, n/2), cact.NumpyPermutation(seed+977, n-n/2)

			for i, v := range lo {
				p[i] = float32(v)
			}

			for i, v := range hi {
				p[n/2+i] = float32(n/2 + v)
			}
		}

		ps.p[k] = p
	}

	return ps
}
