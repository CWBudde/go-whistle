package probe

import "fmt"

// Purposes, one per family, as recorded in the manifest (docs/FINDINGS.md F-019, F-020).
const (
	whySilence = "Digital silence. Per-channel normalisation maps a constant mel to 0 and the stem has no biases, so " +
		"the stack input is exactly 0; an additive position term (pe_gate) would be its only input, where RoPE, " +
		"the Hadamard MLPs' b2 and the zero-padded depthwise convs alone shape the rows. Under a clamp relative to " +
		"the clip's peak any dither would be normalised to full scale. The lengths sweep positions up to 374."
	whyDC = "Constant offset. Reflect centring makes every mel frame equal (the silence rows), zero centring changes " +
		"the frames that reach past the clip; two levels test scale invariance."
	whyTone = "Stationary 440 Hz tone. Interior frames normalise to 0 and the edges carry the partial-window spectrum; " +
		"the leakage spans 15 decades, so 1e-1 … 1e-5 show whether an absolute floor binds anywhere."
	whyBurst = "Tone in the middle second. Each channel is two-level, so the floor decides which leakage channels " +
		"carry the on/off pattern; the levels test scale invariance."
	whyNoise = "White noise: every channel at a similar level; σ 1e-5 is where the config's dither would sit."
	whyDecay = "Noise falling from −20 to −140 dB over 4 s: every channel crosses the floor, whose shape (hard " +
		"clamp, soft knee, absolute floor) bends each channel's trajectory."
	whyImpulseOffset = "One click 0.1 at 16000 + δ: frames i see it at window index 16000 + δ − 160i + 200, so 2–3 " +
		"frames sample the window at known offsets (centring and window shape); δ = 0 puts the side frames at " +
		"w(40) = 0.095, 2 decades below the centre."
	whyImpulseLevel = "The same click at a falling level: scale invariance, and the absolute floor for frames whose " +
		"energy is tiny."
	whyImpulsePair = "Clicks of 0.1 and 0.001 (40 dB apart) in one clip: the relative floor between them."
)

// Lengths and positions of the probe signals.
const (
	twoSeconds = 2 * SampleRate
	centre     = SampleRate // the impulses' base position, encoder frame 12.5
	burstFrom  = SampleRate / 2
	burstTo    = 3 * SampleRate / 2
	toneHz     = 440
)

// Set is the committed probe set: about 1200 encoder frames, 2.5 MB of needle_embed rows.
func Set() []Spec {
	var out []Spec

	for _, n := range []int{5120, 16000, 32000, 64000, 160000, 480000} {
		out = append(out, Spec{Name: fmt.Sprintf("silence-%d", n), Kind: Silence, Samples: n, Purpose: whySilence})
	}

	for _, a := range []float64{1e-2, 1e-4} {
		out = append(out, Spec{Name: "dc-" + level(a), Kind: DC, Samples: twoSeconds, Amplitude: a, Purpose: whyDC})
	}

	for _, a := range []float64{1e-1, 1e-3, 1e-5} {
		out = append(out,
			Spec{Name: "tone-" + level(a), Kind: Tone, Samples: twoSeconds, Amplitude: a, Hz: toneHz, Purpose: whyTone},
			Spec{
				Name: "burst-" + level(a), Kind: Burst, Samples: twoSeconds, Amplitude: a, Hz: toneHz,
				From: burstFrom, To: burstTo, Purpose: whyBurst,
			})
	}

	// One seed for both levels, so that the two differ only by gain.
	for _, a := range []float64{1e-1, 1e-5} {
		out = append(out, Spec{
			Name: "noise-" + level(a), Kind: Noise, Samples: twoSeconds, Amplitude: a, Seed: 1, Purpose: whyNoise,
		})
	}

	out = append(out, Spec{
		Name: "decay", Kind: Decay, Samples: 2 * twoSeconds, FromDB: -20, ToDB: -140, Seed: 3, Purpose: whyDecay,
	})

	for _, d := range []int{0, 20, 40, 80, 120, 140} {
		out = append(out, click(fmt.Sprintf("impulse-d%d-%s", d, level(1e-1)), d, 1e-1, whyImpulseOffset))
	}

	for _, c := range []struct {
		d int
		a float64
	}{{0, 1e-2}, {0, 1e-4}, {0, 1e-6}, {80, 1e-3}, {80, 1e-5}, {80, 1e-6}} {
		out = append(out, click(fmt.Sprintf("impulse-d%d-%s", c.d, level(c.a)), c.d, c.a, whyImpulseLevel))
	}

	return append(out, Spec{
		Name: "impulse-pair", Kind: Impulse, Samples: twoSeconds, Offsets: []int{burstFrom, burstTo},
		Amplitudes: []float64{1e-1, 1e-3}, Purpose: whyImpulsePair,
	})
}

func click(name string, d int, a float64, why string) Spec {
	return Spec{
		Name: name, Kind: Impulse, Samples: twoSeconds, Offsets: []int{centre + d}, Amplitudes: []float64{a},
		Purpose: why,
	}
}

// level names an amplitude in probe names: 1e-01, 1e-03, ….
func level(a float64) string { return fmt.Sprintf("%.0e", a) }
