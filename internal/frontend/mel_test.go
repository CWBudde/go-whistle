package frontend

import (
	"errors"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/weights"
)

const (
	whistleCactPath        = "../../models/whistle/whistle.cact"
	whistleSafetensorsPath = "../../models/whistle/whistle.safetensors"
)

// testBank is tensor 679 as internal/cact recomputes it: librosa's Slaney
// bank, equal to the shipped record bit for bit (F-006, F-007;
// TestMelBankBackends checks it against both files).
func testBank(t *testing.T) *MelBank {
	t.Helper()

	b, err := NewMelBank(cact.SlaneyMel(SampleRate, NFFT, Mels), []int{Bins, Mels})
	if err != nil {
		t.Fatal(err)
	}

	return b
}

// TestMelBankBackends loads audio.mel_filters from whistle.cact and
// whistle.safetensors: both serve the same (257, 80) bank bit for bit, equal
// to the Slaney recomputation the other tests use.
func TestMelBankBackends(t *testing.T) {
	want := cact.SlaneyMel(SampleRate, NFFT, Mels)

	for _, path := range []string{whistleCactPath, whistleSafetensorsPath} {
		_, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", path)
		}

		src, err := weights.Open(path, weights.Options{})
		if err != nil {
			t.Fatal(err)
		}

		values, shape, err := src.Tensor(MelBankName)
		if err != nil {
			t.Fatal(err)
		}

		if len(shape) != 2 || shape[0] != Bins || shape[1] != Mels {
			t.Fatalf("%s: shape %v", path, shape)
		}

		for i, v := range values {
			if math.Float32bits(v) != math.Float32bits(want[i]) {
				t.Fatalf("%s: value %d = %v, Slaney %v", path, i, v, want[i])
			}
		}

		bank, err := LoadMelBank(src)
		if err != nil {
			t.Fatal(err)
		}

		for i := range want {
			if bank.weight[i] != want[i] {
				t.Fatalf("%s: bank value %d differs", path, i)
			}
		}

		t.Logf("%s: %s (257, 80) equals the Slaney bank bit for bit", path, MelBankName)

		err = src.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestNewMelBankErrors(t *testing.T) {
	good := cact.SlaneyMel(SampleRate, NFFT, Mels)

	_, err := NewMelBank(good, []int{Mels, Bins})
	if err == nil {
		t.Error("transposed shape accepted")
	}

	_, err = NewMelBank(good[:len(good)-1], []int{Bins, Mels})
	if err == nil {
		t.Error("short bank accepted")
	}

	bad := append([]float32{}, good...)
	bad[5] = float32(math.NaN())

	_, err = NewMelBank(bad, []int{Bins, Mels})
	if err == nil {
		t.Error("NaN weight accepted")
	}

	_, err = New(nil, Whisper())
	if !errors.Is(err, ErrOptions) {
		t.Errorf("nil bank: %v", err)
	}
}

// TestMelBankApply checks the sparse product against the dense one.
func TestMelBankApply(t *testing.T) {
	b := testBank(t)
	spec := noise(3*Bins, 4, 1)

	for i := range spec {
		spec[i] *= spec[i]
	}

	got := b.apply(spec, 3)

	for f := range 3 {
		for m := range Mels {
			var want float64
			for k := range Bins {
				want += float64(spec[f*Bins+k]) * float64(b.weight[k*Mels+m])
			}

			if d := math.Abs(got[f*Mels+m] - want); d > 1e-12*max(1, want) {
				t.Fatalf("frame %d mel %d: %v, dense %v", f, m, got[f*Mels+m], want)
			}
		}
	}
}

// TestSpectrumMagnitude: Magnitude is the square root of Power.
func TestSpectrumMagnitude(t *testing.T) {
	x := noise(3000, 5, 0.3)
	opts := Whisper()

	power, frames := spectrogramOf(t, x, opts)

	opts.Spectrum = Magnitude
	mag, _ := spectrogramOf(t, x, opts)

	for i := range frames * Bins {
		want := float32(math.Sqrt(float64(power[i])))
		if math.Abs(float64(mag[i]-want)) > 1e-6*float64(want)+1e-30 {
			t.Fatalf("value %d: magnitude %v, sqrt(power) %v", i, mag[i], want)
		}
	}
}

// TestLogVariants applies each log flavour to known energies.
func TestLogVariants(t *testing.T) {
	in := []float64{1, math.E, 100, 1e-3, 1e-12, 0}

	cases := []struct {
		log  Log
		want []float64
	}{
		{LogLn, []float64{0, 1, math.Log(100), math.Log(1e-3), math.Log(1e-10), math.Log(1e-10)}},
		{LogLog10, []float64{0, math.Log10(math.E), 2, -3, -10, -10}},
		{LogAddEps, []float64{
			math.Log(1 + 0x1p-24), math.Log(math.E + 0x1p-24), math.Log(100 + 0x1p-24),
			math.Log(1e-3 + 0x1p-24), math.Log(1e-12 + 0x1p-24), -24 * math.Ln2,
		}},
		// Whisper: log10 with the 1e-10 floor, then max(y, max(y) − 8) with
		// max(y) = 2, then (y + 4) / 4.
		{LogWhisper, []float64{1, (math.Log10(math.E) + 4) / 4, 1.5, 0.25, -0.5, -0.5}},
		// Soft knees: log10(x + ε) with ε = 10^−7.6 or 10^−8 times the
		// largest energy, 100.
		{LogSoft76, softWant(in, 100*math.Pow(10, -7.6))},
		{LogSoft80, softWant(in, 100*1e-8)},
	}

	for _, c := range cases {
		got := append([]float64{}, in...)
		applyLog(got, c.log)

		for i := range got {
			if math.Abs(got[i]-c.want[i]) > 1e-12 {
				t.Errorf("log %d: f(%v) = %v, want %v", c.log, in[i], got[i], c.want[i])
			}
		}
	}

	// The Whisper clamp keeps exactly the top 8 decades: log10 gives
	// 4, −3, −5, the clamp lifts −5 to 4 − 8 = −4.
	got := []float64{1e4, 1e-3, 1e-5}
	applyLog(got, LogWhisper)

	if want := []float64{2, 0.25, 0}; got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Errorf("Whisper clamp: %v, want %v", got, want)
	}
}

func softWant(in []float64, eps float64) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = math.Log10(v + eps)
	}

	return out
}

// TestSoftLog checks what makes the soft knees differ from the Whisper
// clamp (docs/FINDINGS.md F-020): a gain shifts every value by the same
// amount, energies far below the knee all land on log10(ε), an energy at the
// knee lies log10(2) above it, and digital silence stays finite.
func TestSoftLog(t *testing.T) {
	in := []float64{3, 0.5, 2e-6, 3e-8, 1e-14, 0}

	for _, l := range []Log{LogSoft76, LogSoft80} {
		ref := append([]float64{}, in...)
		applyLog(ref, l)

		for _, gain := range []float64{1e-9, 1e-4, 1e3} {
			got := make([]float64, len(in))
			for i, v := range in {
				got[i] = v * gain
			}

			applyLog(got, l)

			for i := range got {
				if d := got[i] - ref[i] - math.Log10(gain); math.Abs(d) > 1e-9 {
					t.Errorf("log %d gain %g: value %d moves by %v more than log10(gain)", l, gain, i, d)
				}
			}
		}

		depth := soft76
		if l == LogSoft80 {
			depth = soft80
		}

		floor := math.Log10(3) - depth
		if math.Abs(ref[5]-floor) > 1e-12 || math.Abs(ref[4]-floor) > 1e-6 {
			t.Errorf("log %d: zero → %v, 1e-14 → %v, want ≈ %v", l, ref[5], ref[4], floor)
		}

		knee := []float64{1, math.Pow(10, -depth)}
		applyLog(knee, l)

		if d := knee[1] - (-depth); math.Abs(d-math.Log10(2)) > 1e-12 {
			t.Errorf("log %d: the knee lies %v above the floor, want log10(2)", l, d)
		}

		zero := make([]float64, 4)
		applyLog(zero, l)

		for _, v := range zero {
			if v != math.Log10(logFloor) {
				t.Errorf("log %d: silence → %v, want %v", l, v, math.Log10(logFloor))
			}
		}

		applyLog(nil, l)
	}

	// The Whisper clamp, unlike the knee, is not scale invariant once its
	// 1e-10 floor binds.
	quiet := []float64{1e-6, 1e-11, 1e-13}
	applyLog(quiet, LogWhisper)

	if quiet[1] != quiet[2] {
		t.Errorf("Whisper: 1e-11 → %v and 1e-13 → %v, want both at the floor", quiet[1], quiet[2])
	}
}

// TestDither checks Dither1e5: σ = 1e-5 noise, the same on every call, and
// DitherNone leaves the samples alone.
func TestDither(t *testing.T) {
	x := make([]float32, 20000)

	a, b := dither(x), dither(x)

	var ss float64

	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("sample %d differs between calls: %v, %v", i, a[i], b[i])
		}

		ss += float64(a[i]) * float64(a[i])
	}

	if sigma := math.Sqrt(ss / float64(len(a))); math.Abs(sigma-1e-5) > 2e-7 {
		t.Errorf("dither σ = %.3g, want 1e-5", sigma)
	}

	for _, d := range []Dither{DitherNone, Dither1e5} {
		o := Whisper()
		o.Dither = d

		fe, err := New(testBank(t), o)
		if err != nil {
			t.Fatal(err)
		}

		spec, _, err := fe.Spectrogram(x)
		if err != nil {
			t.Fatal(err)
		}

		if nonzero := slices.ContainsFunc(spec, func(v float32) bool { return v != 0 }); nonzero != (d == Dither1e5) {
			t.Errorf("dither %d: non-zero spectrum of silence = %v", d, nonzero)
		}
	}
}

// TestNormVariants checks the standardisation formulas on a small matrix.
func TestNormVariants(t *testing.T) {
	const frames = 4

	in := make([]float64, frames*Mels)
	for i := range in {
		in[i] = float64(i%7) + 0.5*float64(i/Mels) + float64(i%Mels)
	}

	none := append([]float64{}, in...)
	normalise(none, frames, NormNone)

	for i := range in {
		if none[i] != in[i] {
			t.Fatal("NormNone changed the input")
		}
	}

	channel := append([]float64{}, in...)
	normalise(channel, frames, NormChannel)

	for m := range Mels {
		col := make([]float64, frames)
		for f := range frames {
			col[f] = in[f*Mels+m]
		}

		mean, std := meanStd(col)
		for f := range frames {
			want := (col[f] - mean) / (std + 1e-5)
			if math.Abs(channel[f*Mels+m]-want) > 1e-12 {
				t.Fatalf("channel %d frame %d: %v, want %v", m, f, channel[f*Mels+m], want)
			}
		}
	}

	global := append([]float64{}, in...)
	normalise(global, frames, NormGlobal)

	mean, std := meanStd(in)
	for i := range in {
		want := (in[i] - mean) / (std + 1e-5)
		if math.Abs(global[i]-want) > 1e-12 {
			t.Fatalf("value %d: %v, want %v", i, global[i], want)
		}
	}

	// One frame: the deviation counts as 0, so every channel becomes 0.
	one := append([]float64{}, in[:Mels]...)
	normalise(one, 1, NormChannel)

	for m, v := range one {
		if v != 0 {
			t.Fatalf("single frame channel %d = %v, want 0", m, v)
		}
	}
}

// meanStd is the mean and the n−1 standard deviation, two-pass.
func meanStd(x []float64) (float64, float64) {
	var sum float64
	for _, v := range x {
		sum += v
	}

	mean := sum / float64(len(x))

	var dev float64
	for _, v := range x {
		dev += (v - mean) * (v - mean)
	}

	return mean, math.Sqrt(dev / float64(len(x)-1))
}

func tones(n int, hz ...float64) []float32 {
	out := make([]float32, n)

	for i := range out {
		for _, f := range hz {
			out[i] += float32(0.2 * math.Sin(2*math.Pi*f*float64(i)/SampleRate))
		}
	}

	return out
}

// TestBandBins: BandBins zeroes exactly the bins outside [250, 3500] Hz and
// leaves the others as BandFull computes them.
func TestBandBins(t *testing.T) {
	x := tones(4000, 100, 1000, 5000)

	full, frames := spectrogramOf(t, x, Whisper())

	opts := Whisper()
	opts.Band = BandBins
	banded, _ := spectrogramOf(t, x, opts)

	kept := 0

	for k := range Bins {
		hz := float64(k) * SampleRate / NFFT
		inside := hz >= BandLowHz && hz <= BandHighHz

		if inside {
			kept++
		}

		for f := range frames {
			i := f*Bins + k
			if inside && banded[i] != full[i] || !inside && banded[i] != 0 {
				t.Fatalf("bin %d (%.2f Hz) frame %d: banded %v, full %v", k, hz, f, banded[i], full[i])
			}
		}
	}

	// Bins 8 (250 Hz) to 112 (3500 Hz).
	if kept != 105 {
		t.Errorf("%d bins kept, want 105", kept)
	}

	// The 5 kHz tone vanishes from the mel energies, 1 kHz stays.
	fe, err := newFrontend(t, opts).LogMel(x)
	if err != nil {
		t.Fatal(err)
	}

	hi := fe.Data[10*Mels+argNearest(ChannelCentres(), 5000)]
	mid := fe.Data[10*Mels+argNearest(ChannelCentres(), 1000)]

	if mid-hi < 2 { // Whisper scale: 2 = 8 decades
		t.Errorf("BandBins: 1 kHz channel %v, 5 kHz channel %v", mid, hi)
	}
}

func argNearest(xs []float64, v float64) int {
	best := 0

	for i, x := range xs {
		if math.Abs(x-v) < math.Abs(xs[best]-v) {
			best = i
		}
	}

	return best
}

// TestBandChannels: BandChannels keeps channels 6–58, whose centres lie in
// [250, 3500] Hz, and puts the others at the log floor.
func TestBandChannels(t *testing.T) {
	centres := ChannelCentres()

	// librosa mel_frequencies(82, fmax=8000)[1:-1] (Slaney) at the band
	// edges, from numpy's linspace in float64.
	for _, c := range []struct {
		m  int
		hz float64
	}{
		{0, 37.23921026495882},
		{5, 223.43526158975294},
		{6, 260.6744718547118},
		{58, 3436.8491819322394},
		{59, 3571.4048886299915},
		{79, 7698.593218313329},
	} {
		if math.Abs(centres[c.m]-c.hz) > 1e-9 {
			t.Errorf("centre %d = %.12f Hz, want %.12f", c.m, centres[c.m], c.hz)
		}
	}

	// Each centre is where the bank's triangle peaks, to within a bin.
	b := testBank(t)

	for m, hz := range centres {
		peak := b.lo[m]
		for k := b.lo[m]; k < b.hi[m]; k++ {
			if b.weight[k*Mels+m] > b.weight[peak*Mels+m] {
				peak = k
			}
		}

		if d := math.Abs(float64(peak)*SampleRate/NFFT - hz); d > SampleRate/NFFT {
			t.Errorf("channel %d: centre %.1f Hz, bank peak at bin %d", m, hz, peak)
		}
	}

	x := noise(4000, 6, 0.1)

	opts := stftOptions(HannPeriodic, PadZero) // LogLn, NormNone

	full, err := newFrontend(t, opts).LogMel(x)
	if err != nil {
		t.Fatal(err)
	}

	opts.Band = BandChannels

	banded, err := newFrontend(t, opts).LogMel(x)
	if err != nil {
		t.Fatal(err)
	}

	floor := float32(math.Log(1e-10))

	for i := range full.Data {
		m := i % Mels
		if inside := m >= 6 && m <= 58; inside && banded.Data[i] != full.Data[i] || !inside && banded.Data[i] != floor {
			t.Fatalf("channel %d: banded %v, full %v", m, banded.Data[i], full.Data[i])
		}
	}
}

// TestPreemphasis: Preemph097 equals feeding y[t] = x[t] − 0.97·x[t−1].
func TestPreemphasis(t *testing.T) {
	x := noise(3000, 7, 0.2)

	y := make([]float32, len(x))

	y[0] = x[0]
	for i := 1; i < len(x); i++ {
		y[i] = x[i] - 0.97*x[i-1]
	}

	opts := Whisper()
	want, _ := spectrogramOf(t, y, opts)

	opts.Preemphasis = Preemph097
	got, _ := spectrogramOf(t, x, opts)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("value %d: %v, want %v", i, got[i], want[i])
		}
	}

	// The high-pass tilts bin 3 (93.75 Hz) against bin 160 (5 kHz) by
	// |H(f)|² = 1.9409 − 1.94·cos(2πf/16000): 10·log10(2.6834/0.0022) = 30.8 dB.
	tone := tones(4000, 100, 5000)

	opts.Preemphasis = PreemphNone
	plain, _ := spectrogramOf(t, tone, opts)

	opts.Preemphasis = Preemph097
	pre, _ := spectrogramOf(t, tone, opts)

	lo, hi := 10*Bins+3, 10*Bins+160 // 93.75 Hz and 5 kHz
	shift := 10 * math.Log10(float64(plain[lo]/plain[hi])/float64(pre[lo]/pre[hi]))

	h := func(k int) float64 { return 1.9409 - 1.94*math.Cos(2*math.Pi*float64(k)/NFFT) }
	if want := 10 * math.Log10(h(160)/h(3)); math.Abs(shift-want) > 0.5 {
		t.Errorf("pre-emphasis tilt 100 Hz vs 5 kHz: %.2f dB, |H|² ratio %.2f dB", shift, want)
	}
}

// TestLogMelPipeline checks the stage order by hand: power spectrum, mel
// bank, log, normalisation.
func TestLogMelPipeline(t *testing.T) {
	x := noise(1600, 8, 0.2)
	opts := Options{
		Window: HannSymmetric, Padding: PadNone, Spectrum: Power, Log: LogAddEps,
		Norm: NormGlobal, Band: BandFull, Preemphasis: PreemphNone, Dither: DitherNone,
	}

	f := newFrontend(t, opts)

	spec, frames, err := f.Spectrogram(x)
	if err != nil {
		t.Fatal(err)
	}

	got, err := f.LogMel(x)
	if err != nil {
		t.Fatal(err)
	}

	if got.Frames != frames || frames != 8 || len(got.Data) != frames*Mels {
		t.Fatalf("%d frames (%d values), spectrogram %d", got.Frames, len(got.Data), frames)
	}

	want := testBank(t).apply(spec, frames)
	for i, v := range want {
		want[i] = math.Log(v + 0x1p-24)
	}

	mean, std := meanStd(want)
	for i, v := range want {
		w := float32((v - mean) / (std + 1e-5))
		if math.Abs(float64(got.Data[i]-w)) > 1e-6 {
			t.Fatalf("value %d: %v, want %v", i, got.Data[i], w)
		}
	}
}
