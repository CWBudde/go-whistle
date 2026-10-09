package frontend

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/cmplx"
	"testing"

	algofft "github.com/cwbudde/algo-fft"
	"github.com/cwbudde/go-whistle/internal/safetensors"
)

const soxrFixture = "../../testdata/frontend/soxr_hq.safetensors"

// soxrCase is one entry of the fixture's case list
// (scripts/resample_fixture.py).
type soxrCase struct {
	Key         string  `json:"key"`
	Rate        int     `json:"rate"`
	Band        string  `json:"band"`
	Amplitude   float64 `json:"amplitude"`
	Samples     int     `json:"samples"`
	InputSHA256 string  `json:"input_sha256"`
	Kind        string  `json:"kind"`
	Freq        float64 `json:"freq"`
	F0          float64 `json:"f0"`
	F1          float64 `json:"f1"`
	Seed        uint64  `json:"seed"`

	input []float32 // regenerated, as _read_wav reads the 16-bit WAV
	soxr  []float32 // soxr.resample(input, rate, 16000, quality="HQ")
}

// aliasCase is a float32 tone above 8 kHz and the level soxr leaves of it
// (scripts/resample_fixture.py alias_level).
type aliasCase struct {
	Rate        int     `json:"rate"`
	Freq        float64 `json:"freq"`
	Amplitude   float64 `json:"amplitude"`
	Samples     int     `json:"samples"`
	SoxrLevelDB float64 `json:"soxr_level_db"`
}

// input is float32(a * sin(2*pi*f*i/rate)), as the fixture script feeds soxr.
func (c aliasCase) input() []float32 {
	out := make([]float32, c.Samples)
	for i := range out {
		out[i] = float32(c.Amplitude * math.Sin(2*math.Pi*c.Freq*float64(i)/float64(c.Rate)))
	}

	return out
}

type soxrFixtureData struct {
	Quality    string      `json:"quality"`
	TargetRate int         `json:"target_rate"`
	Cases      []soxrCase  `json:"cases"`
	Alias      []aliasCase `json:"alias"`
}

func loadSoxrFixture(t *testing.T) soxrFixtureData {
	t.Helper()

	store, err := safetensors.OpenStore(soxrFixture, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("%v (regenerate with just resample-fixture)", err)
	}
	defer store.Close()

	var fixture soxrFixtureData

	err = json.Unmarshal([]byte(store.Metadata()["fixture"]), &fixture)
	if err != nil {
		t.Fatalf("fixture metadata: %v", err)
	}

	if fixture.Quality != "HQ" || fixture.TargetRate != SampleRate || len(fixture.Cases) == 0 || len(fixture.Alias) == 0 {
		t.Fatalf("fixture is %s at %d Hz with %d cases and %d alias cases",
			fixture.Quality, fixture.TargetRate, len(fixture.Cases), len(fixture.Alias))
	}

	for i := range fixture.Cases {
		c := &fixture.Cases[i]

		pcm := c.generate()

		raw := make([]byte, 2*len(pcm))
		for j, v := range pcm {
			binary.LittleEndian.PutUint16(raw[2*j:], uint16(v))
		}

		if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != c.InputSHA256 {
			t.Fatalf("%s: regenerated input does not match the fixture's", c.Key)
		}

		c.input = make([]float32, len(pcm))
		for j, v := range pcm {
			c.input[j] = float32(v) / 32768
		}

		tensor, err := store.Tensor(c.Key)
		if err != nil {
			t.Fatal(err)
		}

		c.soxr = tensor.Data
	}

	return fixture
}

// generate rebuilds the 16-bit input with the formulas of
// scripts/resample_fixture.py, operation for operation.
func (c *soxrCase) generate() []int16 {
	n := c.Samples
	rate := float64(c.Rate)
	out := make([]int16, n)

	pcm := func(v float64) int16 { return int16(math.RoundToEven(32767 * v)) }

	switch c.Kind {
	case "tone":
		for i := range out {
			out[i] = pcm(c.Amplitude * math.Sin(2*math.Pi*c.Freq*float64(i)/rate))
		}
	case "chirp":
		duration := float64(n) / rate
		for i := range out {
			t := float64(i) / rate
			out[i] = pcm(c.Amplitude * math.Sin(2*math.Pi*(c.F0*t+(c.F1-c.F0)*t*t/(2*duration))))
		}
	case "noise":
		state := c.Seed
		for i := range out {
			state += 0x9E3779B97F4A7C15
			z := state
			z = (z ^ z>>30) * 0xBF58476D1CE4E5B9
			z = (z ^ z>>27) * 0x94D049BB133111EB
			z ^= z >> 31
			out[i] = pcm(c.Amplitude * (2*(float64(z>>11)*0x1p-53) - 1))
		}
	}

	return out
}

// deviation compares a resampled signal with soxr's over the interior
// (edge samples dropped at each end). Index 0 of the band pairs is the
// passband, index 1 the near-Nyquist band (see nearNyquist).
type deviation struct {
	SNR    float64 // dB, soxr energy over difference energy
	MaxAbs float64
	Cosine float64
	Lag    float64 // output samples by which got trails soxr (cross-correlation peak)
	GainDB float64 // 10 log10(energy(got) / energy(soxr))
	Level  float64 // got's rms in dB relative to the input's (stopband leakage)

	// Aligned is the SNR per band after got is moved back by the expected
	// alignment residual (residualLag), so it shows the filter-shape
	// difference alone.
	Aligned [2]float64

	// STFT is the mean |log-power difference| in dB per band over
	// 400-sample Hann frames, hop 160, in bins within 60 dB of their frame's
	// peak: what a log-mel front end would see.
	STFT [2]float64
}

const (
	// edge is the number of output samples dropped at each end: both
	// resamplers see zeros outside the signal, and their different filter
	// lengths give different start and end transients.
	edge = 400

	// nearNyquist is where the near-Nyquist band starts, as a fraction of
	// the lower of the two Nyquist rates: 6 kHz when downsampling to
	// 16 kHz, 3 kHz when upsampling from 8 kHz. soxr HQ is flat to 0.913
	// of it, algo-dsp's -6 dB point is at 0.96 (hqSpec).
	nearNyquist = 0.75 // scripts/resample_fixture.py NEAR_NYQUIST
)

// splitHz is the passband / near-Nyquist boundary for an input rate.
func splitHz(rate int) float64 {
	return nearNyquist * float64(min(rate, SampleRate)) / 2
}

func measure(got, ref []float32, c soxrCase, residual float64) deviation {
	g, r := toF64(got[edge:len(got)-edge]), toF64(ref[edge:len(ref)-edge])

	var d deviation

	var eRef, eGot, eDiff, dot float64

	for i := range r {
		diff := g[i] - r[i]
		eRef += r[i] * r[i]
		eGot += g[i] * g[i]
		eDiff += diff * diff
		dot += g[i] * r[i]
		d.MaxAbs = math.Max(d.MaxAbs, math.Abs(diff))
	}

	d.SNR = 10 * math.Log10(eRef/eDiff)
	d.Cosine = dot / math.Sqrt(eRef*eGot)
	d.GainDB = 10 * math.Log10(eGot/eRef)
	d.Level = level(got, c.Amplitude)
	d.Lag = lag(g, r)
	d.Aligned = alignedSNR(g, r, residual, splitHz(c.Rate))
	d.STFT = stftDeviation(g, r, splitHz(c.Rate))

	return d
}

// level is the rms of x's interior in dB relative to a sine of the given
// amplitude.
func level(x []float32, amplitude float64) float64 {
	var e float64
	for _, v := range x[edge : len(x)-edge] {
		e += float64(v) * float64(v)
	}

	return 10 * math.Log10(e/float64(len(x)-2*edge)/(amplitude*amplitude/2))
}

// lag finds the cross-correlation peak over ±3 samples and refines it with
// a parabola through the neighbours.
func lag(g, r []float64) float64 {
	const span = 3

	var xc [2*span + 1]float64

	for k := -span; k <= span; k++ {
		for i := span; i < len(r)-span; i++ {
			xc[k+span] += g[i+k] * r[i]
		}
	}

	best := 1
	for k := 1; k < len(xc)-1; k++ {
		if xc[k] > xc[best] {
			best = k
		}
	}

	a, b, c := xc[best-1], xc[best], xc[best+1]

	return float64(best-span) + 0.5*(a-c)/(a-2*b+c)
}

// alignedSNR compares the central 4096 samples, Hann-windowed, after a
// fractional advance of got by shift samples (a linear phase in the
// frequency domain), separately below and above split Hz.
func alignedSNR(g, r []float64, shift, split float64) [2]float64 {
	const n = 4096

	start := (len(r) - n) / 2
	sg, sr := spectrum(hann(g[start:start+n])), spectrum(hann(r[start:start+n]))

	var ref, noise [2]float64

	for k := range sr {
		band := 0
		if float64(k)*SampleRate/n >= split {
			band = 1
		}

		advanced := sg[k] * cmplx.Rect(1, 2*math.Pi*float64(k)*shift/n)
		ref[band] += sqAbs(sr[k])
		noise[band] += sqAbs(advanced - sr[k])
	}

	return [2]float64{10 * math.Log10(ref[0]/noise[0]), 10 * math.Log10(ref[1]/noise[1])}
}

// stftDeviation returns the mean |log-power difference| below and above
// split Hz over 400-sample Hann frames zero-padded to 512, hop 160.
func stftDeviation(g, r []float64, split float64) [2]float64 {
	const (
		win   = 400
		nfft  = 512
		hop   = 160
		floor = 60.0 // dB below the frame peak that still counts
	)

	var (
		sum   [2]float64
		count [2]int
	)

	for start := 0; start+win <= len(r); start += hop {
		fg, fr := make([]float64, nfft), make([]float64, nfft)
		copy(fg, hann(g[start:start+win]))
		copy(fr, hann(r[start:start+win]))

		sg, sr := spectrum(fg), spectrum(fr)

		peak := 0.0
		for _, c := range sr {
			peak = math.Max(peak, sqAbs(c))
		}

		for k := range sr {
			pr := sqAbs(sr[k])
			if pr < peak*math.Pow(10, -floor/10) {
				continue
			}

			band := 0
			if float64(k)*SampleRate/nfft >= split {
				band = 1
			}

			sum[band] += math.Abs(10 * math.Log10(sqAbs(sg[k])/pr))
			count[band]++
		}
	}

	var mean [2]float64
	for b := range mean {
		if count[b] > 0 {
			mean[b] = sum[b] / float64(count[b])
		}
	}

	return mean
}

// spectrum is the real FFT of x.
func spectrum(x []float64) []complex128 {
	plan, err := algofft.NewPlanReal64(len(x))
	if err != nil {
		panic(err)
	}

	out := make([]complex128, len(x)/2+1)

	err = plan.Forward(out, x)
	if err != nil {
		panic(err)
	}

	return out
}

// hann returns x times a periodic Hann window of its length.
func hann(x []float64) []float64 {
	out := make([]float64, len(x))
	for i := range out {
		out[i] = x[i] * 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(len(x))))
	}

	return out
}

func sqAbs(c complex128) float64 {
	re, im := real(c), imag(c)

	return re*re + im*im
}

func toF64(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}

	return out
}
