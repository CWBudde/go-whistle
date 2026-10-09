package cact

import (
	"math"
	"testing"
)

// Copied from internal/safetensors/store_unit_test.go.
func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		name string
		h    uint16
		want float32
	}{
		{name: "positive zero", h: 0x0000, want: 0.0},
		{name: "negative zero", h: 0x8000, want: float32(math.Copysign(0, -1))},
		{name: "one", h: 0x3c00, want: 1.0},
		{name: "negative one", h: 0xbc00, want: -1.0},
		{name: "half", h: 0x3800, want: 0.5},
		{name: "two", h: 0x4000, want: 2.0},
		{name: "max normal", h: 0x7bff, want: 65504.0},
		{name: "smallest positive normal", h: 0x0400, want: float32(math.Ldexp(1, -14))},       // 2^-14
		{name: "smallest positive subnormal", h: 0x0001, want: float32(math.Ldexp(1, -14-10))}, // 2^-24
		{name: "positive infinity", h: 0x7c00, want: float32(math.Inf(1))},
		{name: "negative infinity", h: 0xfc00, want: float32(math.Inf(-1))},
		{name: "NaN", h: 0x7e00, want: float32(math.NaN())},
		{name: "subnormal half of smallest normal", h: 0x0200, want: float32(math.Ldexp(1, -15))}, // 2^-15
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float16ToFloat32(tt.h)
			if math.IsNaN(float64(tt.want)) {
				if !math.IsNaN(float64(got)) {
					t.Fatalf("float16ToFloat32(0x%04x) = %v; want NaN", tt.h, got)
				}

				return
			}

			if got != tt.want {
				t.Fatalf("float16ToFloat32(0x%04x) = %v; want %v", tt.h, got, tt.want)
			}
		})
	}
}
