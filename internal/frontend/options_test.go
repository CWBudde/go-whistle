package frontend

import (
	"errors"
	"testing"
)

// TestVariants pins the enumeration Phase 4's probe harness iterates: its
// size, order, uniqueness and the String/ParseOptions round trip.
func TestVariants(t *testing.T) {
	vs := Variants()
	if len(vs) != 2*3*2*6*3*3*2*2 {
		t.Fatalf("%d variants, want 2592", len(vs))
	}

	seen := map[string]int{}

	for i, v := range vs {
		err := v.Validate()
		if err != nil {
			t.Fatalf("variant %d: %v", i, err)
		}

		key := v.String()
		if j, ok := seen[key]; ok {
			t.Fatalf("variants %d and %d share key %q", j, i, key)
		}

		seen[key] = i

		back, err := ParseOptions(key)
		if err != nil || back != v {
			t.Fatalf("ParseOptions(%q) = %+v, %v; want %+v", key, back, err, v)
		}
	}

	pinned := map[int]string{
		0:    "win=hann-periodic pad=reflect spec=power log=ln norm=none band=full pre=none dither=none",
		1:    "win=hann-periodic pad=reflect spec=power log=ln norm=none band=full pre=none dither=1e-5",
		2:    "win=hann-periodic pad=reflect spec=power log=ln norm=none band=full pre=0.97 dither=none",
		4:    "win=hann-periodic pad=reflect spec=power log=ln norm=none band=bins pre=none dither=none",
		1295: "win=hann-periodic pad=none spec=magnitude log=soft-80db norm=global band=channels pre=0.97 dither=1e-5",
		1296: "win=hann-symmetric pad=reflect spec=power log=ln norm=none band=full pre=none dither=none",
		2591: "win=hann-symmetric pad=none spec=magnitude log=soft-80db norm=global band=channels pre=0.97 dither=1e-5",
	}

	for i, want := range pinned {
		if got := vs[i].String(); got != want {
			t.Errorf("variant %d = %q, want %q", i, got, want)
		}
	}

	for _, preset := range []Options{NeMo(), Whisper()} {
		if _, ok := seen[preset.String()]; !ok {
			t.Errorf("preset %q is not a variant", preset)
		}
	}

	if got := NeMo().String(); got != "win=hann-symmetric pad=zero spec=power log=ln-eps norm=channel band=full pre=0.97 dither=none" {
		t.Errorf("NeMo() = %q", got)
	}

	if got := Whisper().String(); got != "win=hann-periodic pad=reflect spec=power log=whisper norm=none band=full pre=none dither=none" {
		t.Errorf("Whisper() = %q", got)
	}
}

func TestOptionsInvalid(t *testing.T) {
	bad := make([]Options, 1, 1+2*len(axes))

	for i := range axes {
		o := Whisper()
		*axes[i].field(&o) = 0
		bad = append(bad, o)

		o = Whisper()
		*axes[i].field(&o) = len(axes[i].names) + 1
		bad = append(bad, o)
	}

	for _, o := range bad {
		err := o.Validate()
		if !errors.Is(err, ErrOptions) {
			t.Errorf("%v: Validate = %v", o, err)
		}

		_, err = New(testBank(t), o)
		if !errors.Is(err, ErrOptions) {
			t.Errorf("%v: New = %v", o, err)
		}
	}

	for _, key := range []string{
		"",
		"win=hann-periodic",
		"pad=reflect win=hann-periodic spec=power log=ln norm=none band=full pre=none",
		"win=hann pad=reflect spec=power log=ln norm=none band=full pre=none",
		"win=hann-periodic pad=reflect spec=power log=ln norm=none band=full pre=none",
		"win=hann-periodic pad=reflect spec=power log=ln norm=none band=full pre=none dither=none extra=1",
	} {
		_, err := ParseOptions(key)
		if !errors.Is(err, ErrOptions) {
			t.Errorf("ParseOptions(%q) = %v", key, err)
		}
	}
}
