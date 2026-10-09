package needle

import (
	"math"
	"testing"
)

// TestRopeHandComputed rotates one 4-wide head by hand: half split, so the
// pairs are (x0, x2) at frequency 1 and (x1, x3) at 1/θ^(2/4).
func TestRopeHandComputed(t *testing.T) {
	const theta = 1e5

	r := NewRope(4, 3, theta)
	f1 := 1 / math.Sqrt(theta)

	for pos := range 3 {
		p := float64(pos)
		wantCos := []float64{math.Cos(p), math.Cos(p * f1)}
		wantSin := []float64{math.Sin(p), math.Sin(p * f1)}

		if d := maxAbsDiff64(r.cos[2*pos:2*pos+2], wantCos); d > 1e-7 {
			t.Errorf("cos[%d] = %v, want %v", pos, r.cos[2*pos:2*pos+2], wantCos)
		}

		if d := maxAbsDiff64(r.sin[2*pos:2*pos+2], wantSin); d > 1e-7 {
			t.Errorf("sin[%d] = %v, want %v", pos, r.sin[2*pos:2*pos+2], wantSin)
		}
	}

	x := []float32{1, 2, 3, 4}
	r.apply(x, 2)

	c0, s0 := math.Cos(2), math.Sin(2)
	c1, s1 := math.Cos(2*f1), math.Sin(2*f1)
	want := []float64{
		1*c0 - 3*s0, // x1·cos − x2·sin
		2*c1 - 4*s1,
		3*c0 + 1*s0, // x2·cos + x1·sin
		4*c1 + 2*s1,
	}

	if d := maxAbsDiff64(x, want); d > 1e-6 {
		t.Errorf("rope(pos 2) = %v, want %v", x, want)
	}

	// Position 0 is the identity.
	y := []float32{1, 2, 3, 4}
	r.apply(y, 0)

	if y[0] != 1 || y[1] != 2 || y[2] != 3 || y[3] != 4 {
		t.Errorf("rope(pos 0) = %v, want identity", y)
	}
}

func TestRopeFixtureTables(t *testing.T) {
	fx := openFixture(t, "attention_decoder")
	wantCos, shape := fx.get("rope_cos")
	wantSin, _ := fx.get("rope_sin")

	r := NewRope(2*shape[1], shape[0], 1e5)
	checkContract(t, "rope_cos", r.cos, wantCos)
	checkContract(t, "rope_sin", r.sin, wantSin)
}
