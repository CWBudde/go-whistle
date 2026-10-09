package cact_test

import (
	"math"
	"slices"
	"testing"
)

// Recomputations of the derived .cact tensors (docs/FINDINGS.md F-006), so
// TestWhistleNameMap can check them by equality. Phase 2 (safetensors-only
// loads) and Phase 3 (front end) may promote them to library code.

// mt19937 is the Mersenne Twister as numpy's legacy RandomState seeds it
// from an integer: init_genrand(seed), 32-bit outputs.
type mt19937 struct {
	state [624]uint32
	next  int
}

func newMT19937(seed uint32) *mt19937 {
	m := &mt19937{next: 624}

	m.state[0] = seed
	for i := 1; i < 624; i++ {
		prev := m.state[i-1]
		m.state[i] = 1812433253*(prev^(prev>>30)) + uint32(i)
	}

	return m
}

func (m *mt19937) uint32() uint32 {
	if m.next == 624 {
		for k := range 624 {
			y := m.state[k]&0x80000000 | m.state[(k+1)%624]&0x7fffffff

			v := m.state[(k+397)%624] ^ y>>1
			if y&1 == 1 {
				v ^= 0x9908b0df
			}

			m.state[k] = v
		}

		m.next = 0
	}

	y := m.state[m.next]
	m.next++

	y ^= y >> 11
	y ^= y << 7 & 0x9d2c5680
	y ^= y << 15 & 0xefc60000

	return y ^ y>>18
}

// numpyPermutation is np.random.RandomState(seed).permutation(n), which
// architecture.py _hada_perms draws with seeds 11 and 13: the legacy
// shuffle of arange(n) swaps i with j = random_interval(i) for i from n-1
// down to 1, and random_interval draws 32-bit words masked to the smallest
// covering power of two minus one until one is <= i.
func numpyPermutation(seed uint32, n int) []int {
	m := newMT19937(seed)

	p := make([]int, n)
	for i := range p {
		p[i] = i
	}

	for i := n - 1; i >= 1; i-- {
		limit := uint32(i)

		mask := limit
		for s := 1; s < 32; s *= 2 {
			mask |= mask >> s
		}

		j := m.uint32() & mask
		for j > limit {
			j = m.uint32() & mask
		}

		p[i], p[j] = p[j], p[i]
	}

	return p
}

// Slaney's mel scale as librosa implements it (htk=False): linear below
// 1 kHz at 200/3 Hz per mel, logarithmic above at log(6.4)/27 per mel.
const (
	slaneyHzPerMel = 200.0 / 3
	slaneyBreakHz  = 1000.0
	slaneyBreakMel = slaneyBreakHz / slaneyHzPerMel
)

var slaneyLogStep = math.Log(6.4) / 27

func hzToMel(hz float64) float64 {
	if hz >= slaneyBreakHz {
		return slaneyBreakMel + math.Log(hz/slaneyBreakHz)/slaneyLogStep
	}

	return hz / slaneyHzPerMel
}

func melToHz(mel float64) float64 {
	if mel >= slaneyBreakMel {
		return slaneyBreakHz * math.Exp(slaneyLogStep*(mel-slaneyBreakMel))
	}

	return slaneyHzPerMel * mel
}

// slaneyMel is librosa.filters.mel(sr=rate, n_fft=nfft, n_mels=mels) with its
// defaults fmin 0, fmax rate/2, htk False and norm "slaney", computed in
// float64 and rounded once to float32, laid out [nfft/2+1][mels] as .cact
// tensor 679 stores it. The bin and mel grids use numpy's rfftfreq and
// linspace arithmetic.
func slaneyMel(rate, nfft, mels int) []float32 {
	bins := nfft/2 + 1

	freq := make([]float64, bins)
	for b := range freq {
		freq[b] = float64(b) * (1 / (float64(nfft) * (1 / float64(rate))))
	}

	lo, hi := hzToMel(0), hzToMel(float64(rate)/2)
	step := (hi - lo) / float64(mels+1)

	edge := make([]float64, mels+2)
	for k := range edge {
		edge[k] = melToHz(float64(k)*step + lo)
	}

	edge[mels+1] = melToHz(hi) // linspace pins the endpoint

	out := make([]float32, bins*mels)

	for m := range mels {
		enorm := 2 / (edge[m+2] - edge[m])

		for b, f := range freq {
			lower := -(edge[m] - f) / (edge[m+1] - edge[m])
			upper := (edge[m+2] - f) / (edge[m+2] - edge[m+1])
			out[b*mels+m] = float32(max(0, min(lower, upper)) * enorm)
		}
	}

	return out
}

func TestMT19937(t *testing.T) {
	// The C++ standard fixes the 10000th output of mt19937 seeded 5489.
	m := newMT19937(5489)
	for range 9999 {
		m.uint32()
	}

	if got := m.uint32(); got != 4123659995 {
		t.Errorf("10000th output = %d, want 4123659995", got)
	}
}

func TestNumpyPermutation(t *testing.T) {
	// numpy 2.5.3: RandomState(s).permutation(512)[:8] and [-4:].
	for seed, want := range map[uint32][2][]int{
		11: {{396, 445, 60, 115, 423, 146, 410, 329}, {91, 80, 191, 409}},
		13: {{458, 437, 29, 0, 58, 152, 269, 482}, {16, 74, 176, 338}},
	} {
		p := numpyPermutation(seed, 512)
		if !slices.Equal(p[:8], want[0]) || !slices.Equal(p[508:], want[1]) {
			t.Errorf("seed %d: %v ... %v, want %v ... %v", seed, p[:8], p[508:], want[0], want[1])
		}

		sorted := slices.Sorted(slices.Values(p))
		for k, v := range sorted {
			if v != k {
				t.Fatalf("seed %d: not a permutation of 0..511", seed)
			}
		}
	}
}

func TestSlaneyMel(t *testing.T) {
	for hz, mel := range map[float64]float64{0: 0, 200: 3, 1000: 15, 6400: 42} {
		if got := hzToMel(hz); math.Abs(got-mel) > 1e-12 {
			t.Errorf("hzToMel(%v) = %v, want %v", hz, got, mel)
		}

		if got := melToHz(mel); math.Abs(got-hz) > 1e-9 {
			t.Errorf("melToHz(%v) = %v, want %v", mel, got, hz)
		}
	}

	fb := slaneyMel(16000, 512, 80)
	if len(fb) != 257*80 {
		t.Fatalf("%d weights, want 257*80", len(fb))
	}

	// Each filter is a triangle between its neighbours' centres: never
	// negative and positive somewhere.
	for m := range 80 {
		peak := float32(0)

		for b := range 257 {
			v := fb[b*80+m]
			if v < 0 {
				t.Fatalf("filter %d bin %d = %v", m, b, v)
			}

			peak = max(peak, v)
		}

		if peak == 0 {
			t.Errorf("filter %d is empty", m)
		}
	}
}
