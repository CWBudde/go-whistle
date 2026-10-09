package cact

import (
	"math"
	"slices"
	"testing"
)

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
		p := NumpyPermutation(seed, 512)
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

	fb := SlaneyMel(16000, 512, 80)
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

func TestHadaPermSeed(t *testing.T) {
	for name, want := range map[string]uint32{"hada_p1": 11, "hada_p2": 13} {
		if got, ok := HadaPermSeed(name); !ok || got != want {
			t.Errorf("HadaPermSeed(%s) = %d, %v, want %d", name, got, ok, want)
		}
	}

	if _, ok := HadaPermSeed("hada_p3"); ok {
		t.Error("HadaPermSeed(hada_p3) found a seed")
	}
}
