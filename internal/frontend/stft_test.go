package frontend

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-whistle/internal/safetensors"
)

const stftFixture = "../../testdata/frontend/stft.safetensors"

var (
	allWindows  = []Window{HannPeriodic, HannSymmetric}
	allPaddings = []Padding{PadReflect, PadZero, PadNone}
)

// stftOptions is a valid Options with the given framing; the other axes do
// not touch the plain power spectrum.
func stftOptions(w Window, p Padding) Options {
	return Options{
		Window: w, Padding: p, Spectrum: Power, Log: LogLn,
		Norm: NormNone, Band: BandFull, Preemphasis: PreemphNone,
	}
}

func newFrontend(t *testing.T, opts Options) *Frontend {
	t.Helper()

	f, err := New(testBank(t), opts)
	if err != nil {
		t.Fatal(err)
	}

	return f
}

func spectrogramOf(t *testing.T, x []float32, opts Options) ([]float32, int) {
	t.Helper()

	spec, frames, err := newFrontend(t, opts).Spectrogram(x)
	if err != nil {
		t.Fatal(err)
	}

	return spec, frames
}

// TestFrameCount pins the frame-count rule of each padding (F-012).
func TestFrameCount(t *testing.T) {
	cases := []struct {
		n              int
		centred, plain int
	}{
		{480000, 3000, 2998}, // 30 s: the published 3000 frames
		{0, 0, 0},
		{159, 0, 0},
		{160, 1, 0},
		{399, 2, 0},
		{400, 2, 1},
		{559, 3, 1},
		{560, 3, 2},
		{4321, 27, 25},
		{16000, 100, 98},
	}

	for _, c := range cases {
		for _, p := range []Padding{PadReflect, PadZero} {
			if got := FrameCount(c.n, p); got != c.centred {
				t.Errorf("FrameCount(%d, %d) = %d, want %d", c.n, p, got, c.centred)
			}
		}

		if got := FrameCount(c.n, PadNone); got != c.plain {
			t.Errorf("FrameCount(%d, PadNone) = %d, want %d", c.n, got, c.plain)
		}
	}
}

// TestAnalysisWindow checks algo-dsp's Hann windows against their formulas.
func TestAnalysisWindow(t *testing.T) {
	for _, w := range allWindows {
		got := analysisWindow(w)
		if len(got) != WinLength {
			t.Fatalf("window %d: %d coefficients", w, len(got))
		}

		den := float64(WinLength)
		if w == HannSymmetric {
			den = WinLength - 1
		}

		for k, v := range got {
			want := 0.5 - 0.5*math.Cos(2*math.Pi*float64(k)/den)
			if math.Abs(v-want) > 1e-15 {
				t.Fatalf("window %d [%d] = %v, want %v", w, k, v, want)
			}
		}
	}

	periodic, symmetric := analysisWindow(HannPeriodic), analysisWindow(HannSymmetric)
	if periodic[0] != 0 || periodic[WinLength/2] != 1 || symmetric[0] != 0 || symmetric[WinLength-1] != 0 {
		t.Errorf("window ends: periodic %v, %v; symmetric %v, %v",
			periodic[0], periodic[WinLength/2], symmetric[0], symmetric[WinLength-1])
	}
}

// refSample is x[i] under a padding mode, written independently of algo-dsp:
// numpy "reflect" mirrors without repeating the edge sample, "constant"
// reads zeros.
func refSample(x []float32, i int, p Padding) float64 {
	n := len(x)
	if i >= 0 && i < n {
		return float64(x[i])
	}

	if p != PadReflect {
		return 0
	}

	period := 2 * (n - 1)

	i %= period
	if i < 0 {
		i += period
	}

	if i >= n {
		i = period - i
	}

	return float64(x[i])
}

// refPower is the float64 direct DFT of every frame: the 400 windowed samples
// of frame t, zero-padded to 512 (F-012).
func refPower(x []float32, w Window, p Padding) ([]float64, int) {
	den := float64(WinLength)
	if w == HannSymmetric {
		den = WinLength - 1
	}

	win := make([]float64, WinLength)
	for k := range win {
		win[k] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(k)/den)
	}

	frames := FrameCount(len(x), p)
	out := make([]float64, frames*Bins)
	seg := make([]float64, WinLength)

	for t := range frames {
		start := t*Hop - WinLength/2
		if p == PadNone {
			start = t * Hop
		}

		for j := range seg {
			seg[j] = refSample(x, start+j, p) * win[j]
		}

		for k := range Bins {
			var re, im float64

			for j, v := range seg {
				phase := -2 * math.Pi * float64(j*k%NFFT) / NFFT
				re += v * math.Cos(phase)
				im += v * math.Sin(phase)
			}

			out[t*Bins+k] = re*re + im*im
		}
	}

	return out, frames
}

// spectrumError is the largest |got − want| relative to the spectrogram's
// peak, and the largest relative error over bins within 60 dB of their
// frame's peak.
func spectrumError(got []float32, want []float64, frames int) (peakRel, binRel float64) {
	var peak float64
	for _, v := range want {
		peak = max(peak, v)
	}

	for t := range frames {
		var framePeak float64
		for k := range Bins {
			framePeak = max(framePeak, want[t*Bins+k])
		}

		for k := range Bins {
			i := t*Bins + k
			d := math.Abs(float64(got[i]) - want[i])

			peakRel = max(peakRel, d/peak)
			if want[i] >= 1e-6*framePeak {
				binRel = max(binRel, d/want[i])
			}
		}
	}

	return peakRel, binRel
}

func noise(n int, seed uint64, amp float64) []float32 {
	rng := rand.New(rand.NewPCG(seed, 0))

	out := make([]float32, n)
	for i := range out {
		out[i] = float32(amp * rng.NormFloat64())
	}

	return out
}

// TestSTFTDirectDFT compares the algo-dsp framing (offset windows, frame
// shift for PadNone) with a float64 direct DFT of explicitly framed samples,
// for every window and padding (F-012).
func TestSTFTDirectDFT(t *testing.T) {
	x := noise(2037, 1, 0.1)
	for i := range x {
		x[i] += float32(0.4 * math.Sin(2*math.Pi*1000*float64(i)/SampleRate))
	}

	for _, w := range allWindows {
		for _, p := range allPaddings {
			got, frames := spectrogramOf(t, x, stftOptions(w, p))
			want, wantFrames := refPower(x, w, p)

			if frames != wantFrames || len(got) != len(want) {
				t.Fatalf("win %d pad %d: %d frames (%d values), want %d (%d)", w, p, frames, len(got), wantFrames, len(want))
			}

			peakRel, binRel := spectrumError(got, want, frames)
			t.Logf("win %d pad %d: %d frames, max |Δ|/peak %.2e, max rel (60 dB) %.2e", w, p, frames, peakRel, binRel)

			if peakRel > 1e-6 || binRel > 2e-4 {
				t.Errorf("win %d pad %d: max |Δ|/peak %.2e (pin 1e-6), max rel %.2e (pin 2e-4)", w, p, peakRel, binRel)
			}
		}
	}
}

type stftCase struct {
	Key     string `json:"key"`
	Signal  string `json:"signal"`
	Window  string `json:"window"`
	Padding string `json:"padding"`
	Frames  int    `json:"frames"`
}

// TestSTFTNumpyFixture compares the power spectrum with numpy's rfft over the
// same framing (scripts/mel_fixture.py), for every window and padding.
func TestSTFTNumpyFixture(t *testing.T) {
	store, err := safetensors.OpenStore(stftFixture, safetensors.StoreOptions{})
	if err != nil {
		t.Fatalf("%v (regenerate with just mel-fixture)", err)
	}
	defer store.Close()

	var fixture struct {
		NFFT  int        `json:"nfft"`
		Win   int        `json:"win"`
		Hop   int        `json:"hop"`
		Cases []stftCase `json:"cases"`
	}

	err = json.Unmarshal([]byte(store.Metadata()["fixture"]), &fixture)
	if err != nil {
		t.Fatal(err)
	}

	if fixture.NFFT != NFFT || fixture.Win != WinLength || fixture.Hop != Hop || len(fixture.Cases) != 10 {
		t.Fatalf("fixture geometry %d/%d/%d with %d cases", fixture.NFFT, fixture.Win, fixture.Hop, len(fixture.Cases))
	}

	windows := map[string]Window{"hann-periodic": HannPeriodic, "hann-symmetric": HannSymmetric}
	paddings := map[string]Padding{"reflect": PadReflect, "zero": PadZero, "none": PadNone}

	for _, c := range fixture.Cases {
		x, err := store.Tensor(c.Signal + "/input")
		if err != nil {
			t.Fatal(err)
		}

		ref, err := store.TensorWithShape(c.Key, []int64{int64(c.Frames), Bins})
		if err != nil {
			t.Fatal(err)
		}

		want := make([]float64, len(ref.Data))
		for i, v := range ref.Data {
			want[i] = float64(v)
		}

		got, frames := spectrogramOf(t, x.Data, stftOptions(windows[c.Window], paddings[c.Padding]))
		if frames != c.Frames || len(got) != len(want) {
			t.Fatalf("%s: %d frames, fixture %d", c.Key, frames, c.Frames)
		}

		peakRel, binRel := spectrumError(got, want, frames)
		t.Logf("%-28s %2d frames, max |Δ|/peak %.2e, max rel (60 dB) %.2e", c.Key, frames, peakRel, binRel)

		if peakRel > 1e-6 || binRel > 2e-4 {
			t.Errorf("%s: max |Δ|/peak %.2e (pin 1e-6), max rel %.2e (pin 2e-4)", c.Key, peakRel, binRel)
		}
	}
}

// TestPaddingEdges checks that reflect and zero padding differ only in the
// edge frames that reach outside the signal, and the reflect length limit.
func TestPaddingEdges(t *testing.T) {
	x := noise(4321, 2, 0.2)

	reflect, frames := spectrogramOf(t, x, stftOptions(HannPeriodic, PadReflect))
	zero, _ := spectrogramOf(t, x, stftOptions(HannPeriodic, PadZero))

	for f := range frames {
		// Frame f covers [f·160 − 200, f·160 + 200).
		inside := f*Hop-WinLength/2 >= 0 && f*Hop+WinLength/2 <= len(x)

		same := true

		for k := range Bins {
			if reflect[f*Bins+k] != zero[f*Bins+k] {
				same = false
			}
		}

		if same != inside {
			t.Errorf("frame %d: reflect == zero is %v, frame inside the signal is %v", f, same, inside)
		}
	}

	f := newFrontend(t, stftOptions(HannPeriodic, PadReflect))

	_, _, err := f.Spectrogram(noise(256, 3, 0.1))
	if !errors.Is(err, ErrTooShort) {
		t.Errorf("reflect on 256 samples: %v, want ErrTooShort", err)
	}

	for _, n := range []int{0, 159, 257} {
		_, frames, err := f.Spectrogram(noise(n, 3, 0.1))
		if err != nil || frames != n/Hop {
			t.Errorf("reflect on %d samples: %d frames, %v", n, frames, err)
		}
	}
}

// TestFrameCountMatchesEngine checks the centred frame-count rule against the
// engine (F-012): three stride-2 halvings that round up (SAME), applied to
// FrameCount, give needle_embed's frame count for all 32 corpus clips
// (testdata/oracle) and the seven synthetic lengths of F-001. PadNone's count
// misses three clips under the same stem rule.
func TestFrameCountMatchesEngine(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(clipDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Clips []struct {
			ID      string `json:"id"`
			Samples int    `json:"samples"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	type length struct{ samples, frames int }

	// F-001's synthetic lengths, then the corpus.
	lengths := make([]length, 0, 7+len(manifest.Clips))
	lengths = append(lengths, length{1280, 1}, length{1281, 1}, length{2560, 2}, length{16000, 13},
		length{32000, 25}, length{48000, 38}, length{160000, 125})

	for _, c := range manifest.Clips {
		store, err := safetensors.OpenStore(filepath.Join("../../testdata/oracle", c.ID+".embed.safetensors"), safetensors.StoreOptions{})
		if err != nil {
			t.Fatal(err)
		}

		shape, err := store.Shape("embed")
		if err != nil {
			t.Fatal(err)
		}

		store.Close()

		lengths = append(lengths, length{c.Samples, int(shape[0])})
	}

	stem := func(m int) int { return (((m+1)/2+1)/2 + 1) / 2 }

	noneMisses := 0

	for _, l := range lengths {
		for _, p := range []Padding{PadReflect, PadZero} {
			if got := stem(FrameCount(l.samples, p)); got != l.frames {
				t.Errorf("%d samples: %d mel frames give %d encoder frames, engine %d", l.samples, FrameCount(l.samples, p), got, l.frames)
			}
		}

		if stem(FrameCount(l.samples, PadNone)) != l.frames {
			noneMisses++
		}
	}

	if len(lengths) != 39 || noneMisses != 3 {
		t.Errorf("%d lengths, PadNone misses %d (want 39 and 3)", len(lengths), noneMisses)
	}
}
