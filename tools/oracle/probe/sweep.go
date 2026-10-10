package probe

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Corpus clips the silence sweep and the limits read (testdata/audio): 6.3 s of English, 6.9 s of German.
const (
	ClipEN = "ls-clean-1320-122617-0003"
	ClipDE = "fleurs-de_de-10308608197319017935"
)

// SweepClips lists the corpus clips SilenceSet and LimitSet read.
func SweepClips() []string { return []string{ClipEN, ClipDE} }

// MaxSamples is the engine's limit: 30 s at 16 kHz; one sample more is an error.
const MaxSamples = 30 * SampleRate

// Purposes of the silence-sweep families (docs/FINDINGS.md F-025).
const (
	whySweepSilence = "Digital silence of several lengths: the empty JSON at any length?"
	whyWhite        = "Steady white noise from −120 to 0 dBFS: does any absolute level open the gate?"
	whyPink         = "Steady pink noise from −120 to 0 dBFS: a spectrum closer to speech, still without dynamics."
	whyStep         = "White noise whose middle second is Δ dB louder (or quieter): the level contrast at which the " +
		"engine stops returning the empty JSON; three base levels test whether the threshold is relative."
	whyWidth = "White noise with a +10 dB section of growing length: how much of the clip must be loud " +
		"(a percentile of the frame levels?)."
	whyGain = "A speech clip scaled exactly in float: a scale-invariant gate never closes."
	whyI16  = "A speech clip scaled and rounded to the 16-bit grid, as a quiet 16-bit WAV holds it: the speech " +
		"sinks into the quantisation steps near −90 dBFS."
	whySNR = "A speech clip under white noise at a falling SNR (noise σ relative to the clip's RMS): the contrast " +
		"between speech and the noise floor closes the gate."
	whyPad = "A speech clip padded with long digital silence: the silent share of the clip."
)

// Lengths and positions of the sweep signals.
const (
	sweepLen    = 4 * SampleRate // noise families
	stepFrom    = 3 * SampleRate / 2
	stepTo      = 5 * SampleRate / 2
	widthCentre = sweepLen / 2
)

// Seeds of the sweep's noise; every family keeps one seed, so its members differ only by the swept parameter.
const (
	seedWhite = 1
	seedPink  = 2
	seedStep  = 3
	seedSNR   = 4
)

// SilenceSet is the decoding oracle's silence sweep. clips holds the samples of SweepClips (int16/32768); their
// lengths and RMS fix the speech probes' parameters, which the manifest records exactly.
func SilenceSet(clips map[string][]float32) ([]Spec, error) {
	out := make([]Spec, 0, 256)

	for _, n := range []int{160, 1280, 16000, 80000, 160000, 320000, MaxSamples} {
		out = append(out, Spec{Name: fmt.Sprintf("silence-%d", n), Kind: Silence, Samples: n, Purpose: whySweepSilence})
	}

	for db := -120.0; db <= 0; db += 10 {
		out = append(out, Spec{
			Name: "white-" + dbName(db), Kind: Noise, Samples: sweepLen, Amplitude: dbGain(db), Seed: seedWhite,
			Purpose: whyWhite,
		})
	}

	for db := -120.0; db <= 0; db += 20 {
		out = append(out, Spec{
			Name: "pink-" + dbName(db), Kind: Pink, Samples: sweepLen, Amplitude: dbGain(db), Seed: seedPink,
			Purpose: whyPink,
		})
	}

	out = append(out, stepFamily(-40, stepContrasts())...)
	out = append(out, stepFamily(-80, []float64{2, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 4, 6})...)
	out = append(out, stepFamily(-10, []float64{2, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 4, 6})...)

	for _, w := range []int{
		160, 240, 320, 400, 480, 560, 640, 1280, 2560, 5120, 10240, 20480, 40960, 51200, 57600, 58240, 58880, 59520,
		60160, 60800, 61440, 62720, 63360, 63840,
	} {
		out = append(out, Spec{
			Name: fmt.Sprintf("width-%d", w), Kind: Step, Samples: sweepLen, Amplitude: dbGain(-40), Seed: seedStep,
			From: widthCentre - w/2, To: widthCentre + w/2, Gain: dbGain(10), GainDB: 10, Purpose: whyWidth,
		})
	}

	for _, id := range SweepClips() {
		fam, err := speechFamilies(id, clips[id])
		if err != nil {
			return nil, err
		}

		out = append(out, fam...)
	}

	return out, nil
}

// stepContrasts are the contrasts of the −40 dBFS step family: quarter-dB steps up to 6 dB, twentieths of a dB
// across the measured transition between 2.5 and 2.75 dB, and dips in quarter-dB steps across −3 … −4 dB.
func stepContrasts() []float64 {
	out := []float64{-20, -10, -6, -5, -4, -3.75, -3.5, -3.25, -3, -2, -1}
	for q := range 25 {
		out = append(out, float64(q)/4)
		if q == 10 {
			out = append(out, 2.55, 2.6, 2.65, 2.7)
		}
	}

	return append(out, 7, 8, 10, 15, 20, 30, 40)
}

func stepFamily(baseDB float64, contrasts []float64) []Spec {
	out := make([]Spec, 0, len(contrasts))

	for _, d := range contrasts {
		out = append(out, Spec{
			Name: "step-" + dbName(baseDB) + "-" + dbName(d), Kind: Step, Samples: sweepLen, Amplitude: dbGain(baseDB),
			Seed: seedStep, From: stepFrom, To: stepTo, Gain: dbGain(d), GainDB: d, Purpose: whyStep,
		})
	}

	return out
}

// speechFamilies are the sweep's probes over one clip: exact gain, 16-bit gain, SNR and padding.
func speechFamilies(id string, clip []float32) ([]Spec, error) {
	if len(clip) == 0 {
		return nil, fmt.Errorf("probe: no samples for clip %s", id)
	}

	tag := "en"
	if rest, ok := strings.CutPrefix(id, "fleurs-"); ok {
		tag = rest[:2]
	}

	n := len(clip)
	base := Spec{Kind: Speech, Clip: id, ClipTo: n, Samples: n, Gain: 1}

	var out []Spec

	for _, db := range []float64{0, -20, -40, -60, -80, -90, -100, -110, -120} {
		s := base
		s.Name, s.Gain, s.GainDB, s.Purpose = tag+"-gain-"+dbName(db), dbGain(db), db, whyGain
		out = append(out, s)
	}

	i16 := []float64{0, -20, -40, -60, -70, -75}
	for q := 160; q <= 192; q++ {
		i16 = append(i16, -float64(q)/2)
	}

	for _, db := range i16 {
		s := base
		s.Name, s.Gain, s.GainDB, s.Quantize16, s.Purpose = tag+"-i16-"+dbName(db), dbGain(db), db, true, whyI16
		out = append(out, s)
	}

	snrs := []float64{30, 20, 10, 5, 0, -5}
	for q := 32; q <= 72; q++ {
		snrs = append(snrs, -float64(q)/4)
	}

	snrs = append(snrs, -20, -25)
	rms := RMS(clip)

	for _, snr := range snrs {
		s := base
		s.Name, s.SNRDB, s.Amplitude, s.Seed, s.Purpose = tag+"-snr-"+dbName(snr), snr, rms*dbGain(-snr), seedSNR, whySNR
		out = append(out, s)
	}

	for _, p := range []struct {
		name          string
		before, after int
	}{
		{"pad-1s", SampleRate, SampleRate},
		{"pad-5s", 5 * SampleRate, 5 * SampleRate},
		{"pad-10s", 10 * SampleRate, 10 * SampleRate},
		{"pad-before-23s", 23 * SampleRate, 0},
		{"pad-after-23s", 0, 23 * SampleRate},
	} {
		s := base
		s.Name, s.PadBefore, s.Samples, s.Purpose = tag+"-"+p.name, p.before, p.before+n+p.after, whyPad
		out = append(out, s)
	}

	return out, nil
}

// Purposes of the limit probes.
const (
	whyLimitShort = "A speech excerpt of a handful of samples up to 2 s: the shortest input, the frame edges at 160 " +
		"and 1280 samples, and when a short excerpt starts to transcribe."
	whyLimitMax = "Exactly 30 s and one sample more: the limit is inclusive, and the exact error text."
	whyLimitNil = "No samples at all, and one."
)

// limitExcerpt is where LimitSet's short speech excerpts start in ClipEN: inside "something in his air".
const limitExcerpt = 2 * SampleRate

// LimitSet is the decoding oracle's length-limit set. A probe of 0 samples has no signal; the oracle passes the
// engine an empty clip.
func LimitSet(clips map[string][]float32) ([]Spec, error) {
	en := clips[ClipEN]
	if len(en) < limitExcerpt+2*SampleRate {
		return nil, fmt.Errorf("probe: clip %s has %d samples", ClipEN, len(en))
	}

	out := []Spec{
		{Name: "silence-0", Kind: Silence, Purpose: whyLimitNil},
		{Name: "silence-1", Kind: Silence, Samples: 1, Purpose: whyLimitNil},
	}

	for _, n := range []int{1, 2, 3, 10, 100, 159, 160, 161, 1279, 1280, 1281, 2560, 4000, 8000, 16000, 32000} {
		out = append(out, Spec{
			Name: fmt.Sprintf("speech-%d", n), Kind: Speech, Samples: n, Clip: ClipEN, ClipFrom: limitExcerpt,
			ClipTo: limitExcerpt + n, Gain: 1, Purpose: whyLimitShort,
		})
	}

	for _, n := range []int{MaxSamples, MaxSamples + 1} {
		out = append(out,
			Spec{Name: fmt.Sprintf("silence-%d", n), Kind: Silence, Samples: n, Purpose: whyLimitMax},
			Spec{
				Name: fmt.Sprintf("speech-%d", n), Kind: Speech, Samples: n, Clip: ClipEN, ClipTo: len(en), Gain: 1,
				Purpose: whyLimitMax,
			})
	}

	return out, nil
}

// dbGain is 10^(db/20).
func dbGain(db float64) float64 { return math.Pow(10, db/20) }

// dbName writes a level or contrast for a probe name: −84 → m84, 2.25 → 2.25.
func dbName(db float64) string {
	return strings.Replace(strconv.FormatFloat(db, 'f', -1, 64), "-", "m", 1)
}
