package needle

import (
	"math"
	"testing"
)

func TestZCRMSNormTiny(t *testing.T) {
	n := NewZCRMSNorm([]float32{0, 1, -0.5})
	x := []float32{
		3, 4, 0, // rms² = 25/3
		0, 0, 0, // all zeros: eps keeps it finite
		-1, 2, 2, // rms² = 3
	}
	got := make([]float32, len(x))
	n.Apply(got, x)

	want := make([]float64, 0, len(x))

	for r := range 3 {
		row := x[3*r : 3*r+3]
		ms := (float64(row[0])*float64(row[0]) + float64(row[1])*float64(row[1]) + float64(row[2])*float64(row[2])) / 3
		rms := math.Sqrt(ms + 1e-6)
		want = append(want, float64(row[0])/rms, 2*float64(row[1])/rms, 0.5*float64(row[2])/rms)
	}

	if d := maxAbsDiff64(got, want); d > 1e-6 {
		t.Errorf("ZCRMSNorm = %v, want %v (max |Δ| %.3g)", got, want, d)
	}

	if got[3] != 0 || got[4] != 0 || got[5] != 0 {
		t.Errorf("zero row = %v, want zeros", got[3:6])
	}

	// In place.
	n.Apply(x, x)

	for i := range x {
		if x[i] != got[i] {
			t.Fatalf("in-place Apply differs at %d: %v vs %v", i, x[i], got[i])
		}
	}
}

func TestRMSUnitTiny(t *testing.T) {
	x := []float32{1, 1, 1, 1, 2, 0, 0, 0}
	got := make([]float32, len(x))
	RMSUnit(got, x, 4)

	one := 1 / math.Sqrt(1+1e-6)
	want := []float64{one, one, one, one, 2 / math.Sqrt(1+1e-6), 0, 0, 0}

	if d := maxAbsDiff64(got, want); d > 1e-7 {
		t.Errorf("RMSUnit = %v, want %v", got, want)
	}
}

func TestZCRMSNormPanicsOnRagged(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Apply on 5 values with dim 2 did not panic")
		}
	}()

	NewZCRMSNorm([]float32{0, 0}).Apply(make([]float32, 5), make([]float32, 5))
}

// TestZCRMSNormFixture checks ZCRMSNorm and rms_unit against
// testdata/needle/zcrmsnorm.safetensors with layer00.norm_in.
func TestZCRMSNormFixture(t *testing.T) {
	src := openWhistle(t)
	fx := openFixture(t, "zcrmsnorm")

	n, err := LoadZCRMSNorm(src, "layer00.norm_in")
	if err != nil {
		t.Fatal(err)
	}

	x, _ := fx.get("x")
	wantRMS, _ := fx.get("rms")
	wantY, _ := fx.get("y")
	wantUnit, _ := fx.get("rms_unit")

	rows := len(x) / n.Dim()
	rms := make([]float32, rows)

	for r := range rows {
		rms[r] = rowRMS(x[r*n.Dim() : (r+1)*n.Dim()])
	}

	y := make([]float32, len(x))
	n.Apply(y, x)

	unit := make([]float32, len(x))
	RMSUnit(unit, x, n.Dim())

	checkContract(t, "rms", rms, wantRMS)
	checkContract(t, "y", y, wantY)
	checkContract(t, "rms_unit", unit, wantUnit)
}
