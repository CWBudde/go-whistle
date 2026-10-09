package frontend

import (
	"errors"
	"math"
	"slices"
	"strconv"
	"testing"
)

// residualLag is the alignment residual of Resample at rate in output
// samples: half an upsampled step, 0.5/down, when the prototype length is
// even, and 0 when it is odd (see alignment).
func residualLag(t *testing.T, rate int) float64 {
	t.Helper()

	r, err := newResampler(rate, hqSpec)
	if err != nil {
		t.Fatal(err)
	}

	up, down := r.Ratio()
	if r.TapsPerPhase()*up%2 == 1 {
		return 0
	}

	return 0.5 / float64(down)
}

// TestResampleMatchesSoxr measures Resample against soxr HQ, which
// whistle.py uses, on the fixture's tones, chirps and noise
// (docs/FINDINGS.md F-010). The bounds sit a few dB below what was measured
// at the time of writing.
func TestResampleMatchesSoxr(t *testing.T) {
	t.Log("case            band          SNR dB  max abs   cosine     lag      resid   gain dB  aligned dB pass/near  STFT dB pass/near  level dB")

	for _, c := range loadSoxrFixture(t).Cases {
		got, err := Resample(c.input, c.Rate)
		if err != nil {
			t.Fatal(err)
		}

		if len(got) != len(c.soxr) {
			t.Fatalf("%s: got %d samples, soxr %d", c.Key, len(got), len(c.soxr))
		}

		residual := residualLag(t, c.Rate)
		d := measure(got, c.soxr, c, residual)

		t.Logf("%-15s %-12s %7.1f  %.2e  %.7f  %+.4f  %.4f  %+.4f  %6.1f / %6.1f       %.4f / %.4f   %6.1f",
			c.Key, c.Band, d.SNR, d.MaxAbs, d.Cosine, d.Lag, residual, d.GainDB,
			d.Aligned[0], d.Aligned[1], d.STFT[0], d.STFT[1], d.Level)

		checkDeviation(t, c, d, residual)
	}
}

// checkDeviation pins the measurements. Measured minima and maxima at the
// time of writing are in the comments.
func checkDeviation(t *testing.T, c soxrCase, d deviation, residual float64) {
	t.Helper()

	atLeast := func(what string, got, bound float64) {
		if !(got >= bound) {
			t.Errorf("%s: %s %.4g, want >= %.4g", c.Key, what, got, bound)
		}
	}

	atMost := func(what string, got, bound float64) {
		if !(got <= bound) {
			t.Errorf("%s: %s %.4g, want <= %.4g", c.Key, what, got, bound)
		}
	}

	if c.Band == "stopband" {
		// The 16-bit input's quantisation noise is the floor here, for soxr
		// (-93.6 to -97.2 dB) and Resample alike; TestResampleAliasRejection
		// measures below it.
		atMost("alias level dB", d.Level, -90) // -93.6

		return
	}

	atMost("|lag - residual|", math.Abs(d.Lag-residual), 0.005) // 0.0011

	// The raw comparison carries the alignment residual, which at 8 kHz is
	// half a sample; Aligned and STFT do not.
	raw := c.Rate != 8000

	switch c.Band {
	case "passband":
		atMost("|gain dB|", math.Abs(d.GainDB), 0.02)        // 0.0000
		atLeast("aligned passband SNR dB", d.Aligned[0], 62) // 67.1 at 8 kHz, 116.6 otherwise
		atMost("STFT passband dB", d.STFT[0], 0.1)           // 0.063

		if raw {
			atLeast("SNR dB", d.SNR, 53)        // 56.1
			atMost("max abs", d.MaxAbs, 1.5e-3) // 7.8e-4
		}
	case "near-nyquist":
		atMost("|gain dB|", math.Abs(d.GainDB), 0.02)            // 0.0065
		atLeast("aligned near-Nyquist SNR dB", d.Aligned[1], 58) // 62.5
		atMost("STFT near-Nyquist dB", d.STFT[1], 0.02)          // 0.0065

		if raw {
			atLeast("SNR dB", d.SNR, 47)      // 49.9
			atMost("max abs", d.MaxAbs, 3e-3) // 1.6e-3
		}
	case "broadband":
		atMost("|gain dB|", math.Abs(d.GainDB), 0.02)            // 0.0040
		atLeast("aligned passband SNR dB", d.Aligned[0], 62)     // 66.9
		atLeast("aligned near-Nyquist SNR dB", d.Aligned[1], 22) // 25.4, the transition band
		atMost("STFT passband dB", d.STFT[0], 0.1)               // 0.052
		atMost("STFT near-Nyquist dB", d.STFT[1], 2)             // 1.49

		if raw {
			atLeast("SNR dB", d.SNR, 30)        // 32.9
			atLeast("cosine", d.Cosine, 0.9995) // 0.99974
		}
	default:
		t.Fatalf("%s: unknown band %q", c.Key, c.Band)
	}
}

// TestResampleAliasRejection feeds float32 tones just above 8 kHz, where
// the 16-bit fixture cases only reach the quantisation floor, and compares
// what is left of them with what soxr leaves.
func TestResampleAliasRejection(t *testing.T) {
	for _, c := range loadSoxrFixture(t).Alias {
		got, err := Resample(c.input(), c.Rate)
		if err != nil {
			t.Fatal(err)
		}

		lvl := level(got, c.Amplitude)
		t.Logf("%d Hz, %.0f Hz tone: %7.1f dB (soxr %7.1f dB)", c.Rate, c.Freq, lvl, c.SoxrLevelDB)

		// Measured: -87 dB at 8.1 kHz, <= -110 dB from 8.2 kHz.
		bound := -105.0
		if c.Freq < 8200 {
			bound = -82
		}

		if lvl > bound {
			t.Errorf("%d Hz, %.0f Hz tone: alias level %.1f dB, want <= %.0f", c.Rate, c.Freq, lvl, bound)
		}
	}
}

func TestResampleLength(t *testing.T) {
	for _, rate := range []int{minRate, 8000, 11025, 22050, 32000, 44100, 44101, 48000, 96000, 192000} {
		for _, n := range []int{1, 2, 3, 7, 100, 1001} {
			got, err := Resample(make([]float32, n), rate)
			if err != nil {
				t.Fatalf("%d Hz: %v", rate, err)
			}

			if want := int(math.Round(float64(n) * SampleRate / float64(rate))); len(got) != want {
				t.Errorf("%d Hz, %d samples: got %d, want %d", rate, n, len(got), want)
			}
		}
	}
}

func TestResampleSameRateCopies(t *testing.T) {
	in := []float32{0.25, -0.5}

	got, err := Resample(in, SampleRate)
	if err != nil {
		t.Fatal(err)
	}

	got[0] = 1
	if in[0] != 0.25 || got[1] != -0.5 {
		t.Fatalf("got %v from %v", got, in)
	}
}

func TestResampleRejectsRates(t *testing.T) {
	for _, rate := range []int{0, -8000, 1, minRate - 1, 1_000_000_000} {
		_, err := Resample(make([]float32, 1<<20), rate)
		if !errors.Is(err, ErrSampleRate) {
			t.Errorf("%d Hz: got %v, want ErrSampleRate", rate, err)
		}
	}
}

// TestResampleImpulseAlignment checks the delay removal: an impulse at
// input frame i peaks at output frame i*16000/rate, odd and even prototypes
// alike.
func TestResampleImpulseAlignment(t *testing.T) {
	for _, rate := range []int{8000, 11025, 22050, 24000, 32000, 44100, 48000, 96000} {
		g := gcd(SampleRate, rate)
		at := 50 * (rate / g) // a frame whose output time is an integer

		in := make([]float32, 3*at)
		in[at] = 1

		got, err := Resample(in, rate)
		if err != nil {
			t.Fatal(err)
		}

		peak := 0
		for i, v := range got {
			if v > got[peak] {
				peak = i
			}
		}

		if want := at * SampleRate / rate; peak != want {
			t.Errorf("%d Hz: impulse at %d peaks at %d, want %d", rate, at, peak, want)
		}
	}
}

func TestResamplerRatio(t *testing.T) {
	for _, tc := range []struct{ rate, up, down int }{
		{8000, 2, 1},
		{11025, 640, 441},
		{22050, 320, 441},
		{44100, 160, 441},
		{48000, 1, 3},
		{192000, 1, 12},
		{44056, 2000, 5507},
		{16001, 16000, 16001},
		{47999, 1, 3}, // 16000/47999 needs down > maxRatioTerm
	} {
		r, err := newResampler(tc.rate, hqSpec)
		if err != nil {
			t.Fatal(err)
		}

		if up, down := r.Ratio(); up != tc.up || down != tc.down {
			t.Errorf("%d Hz: ratio %d/%d, want %d/%d", tc.rate, up, down, tc.up, tc.down)
		}
	}
}

// TestResampleTone checks gain and frequency on rates with exact and with
// approximated ratios (maxRatioTerm).
func TestResampleTone(t *testing.T) {
	for _, rate := range []int{11025, 44100, 44101, 47999, 7999} {
		in := make([]float32, rate) // 1 s
		for i := range in {
			in[i] = float32(0.5 * math.Sin(2*math.Pi*1000*float64(i)/float64(rate)))
		}

		got, err := Resample(in, rate)
		if err != nil {
			t.Fatal(err)
		}

		inner := got[1000 : len(got)-1000]

		crossings := 0

		for i := 1; i < len(inner); i++ {
			if inner[i-1] < 0 && inner[i] >= 0 {
				crossings++
			}
		}

		if crossings < 873 || crossings > 875 { // 1 kHz over 14000 samples
			t.Errorf("%d Hz: %d upward zero crossings, want 874±1", rate, crossings)
		}

		if hi := slices.Max(inner); math.Abs(float64(hi)-0.5) > 0.001 {
			t.Errorf("%d Hz: peak %v, want 0.5", rate, hi)
		}
	}
}

// BenchmarkResample converts 30 s, Whistle's input limit, from common rates.
func BenchmarkResample(b *testing.B) {
	for _, rate := range []int{8000, 22050, 44100, 48000} {
		in := make([]float32, 30*rate)
		for i := range in {
			in[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/float64(rate)))
		}

		b.Run(strconv.Itoa(rate), func(b *testing.B) {
			for b.Loop() {
				_, err := Resample(in, rate)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
