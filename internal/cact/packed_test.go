package cact

import (
	"bytes"
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// streamIndex reads index k of an LSB-first bitstream bit by bit, apart
// from unpackLSB's word-wise reading.
func streamIndex(stream []byte, k, width int) uint8 {
	var v uint8

	for b := range width {
		bit := k*width + b
		v |= (stream[bit/8] >> (bit % 8) & 1) << b
	}

	return v
}

// TestCQPacked packs random matrices with the _cq_pack port and checks that
// CQPacked returns the shape, the codebook, the indices and the norms the
// port wrote, over every padding case and record width.
func TestCQPacked(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 9))

	for _, s := range cqShapes {
		for _, width := range cqWidths {
			w := make([]float32, s.out*s.in)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
			}

			cb := portCodebook(width, s.group)
			pp := portPack(w, s.out, s.in, width, s.group, cb)
			f := cqFile(t, s.group, cqTensor(s.out, s.in, s.group, width, pp.blob))

			p, err := f.CQPacked(0)
			if err != nil {
				t.Fatalf("%v bits %d: CQPacked: %v", s, width, err)
			}

			groups := (s.in + s.group - 1) / s.group
			inPad := groups * s.group

			idxBits := width
			if width == ternaryRecordBits {
				idxBits = 2
			}

			if p.Out != s.out || p.In != s.in || p.Group != s.group || p.Bits != width || p.Width != idxBits ||
				p.Groups() != groups || p.RowBytes() != inPad*idxBits/8 {
				t.Fatalf("%v bits %d: Packed %+v, Groups %d, RowBytes %d", s, width, p, p.Groups(), p.RowBytes())
			}

			if !slices.Equal(p.Codebook, cb) {
				t.Errorf("%v bits %d: Codebook = %v, want %v", s, width, p.Codebook, cb)
			}

			n := s.out * p.RowBytes()
			if !bytes.Equal(p.Index, pp.blob[:n]) || !bytes.Equal(p.Norms, pp.blob[n:]) {
				t.Fatalf("%v bits %d: Index/Norms do not split the blob at %d", s, width, n)
			}

			for r := range s.out {
				for k := range inPad {
					got := streamIndex(p.Index[r*p.RowBytes():], k, p.Width)
					if width == ternaryRecordBits {
						got = uint8(slices.Index(tritCrumb[:], got))
					}

					if want := pp.idx[r*inPad+k]; got != want {
						t.Fatalf("%v bits %d: row %d index %d = %d, want %d", s, width, r, k, got, want)
					}
				}

				for g := range groups {
					if got, want := p.Norm(r, g), pp.norm16[r*groups+g]; got != want {
						t.Fatalf("%v bits %d: Norm(%d, %d) = %v, want %v", s, width, r, g, got, want)
					}
				}
			}
		}
	}
}

// TestCQPacked_Aliases checks that the packed views share the file's bytes
// and that Index cannot grow into the norms.
func TestCQPacked_Aliases(t *testing.T) {
	f := cqFile(t, 8, cqTensor(2, 8, 8, 2, make([]byte, 2*(8*2/8+2))))

	p, err := f.CQPacked(0)
	if err != nil {
		t.Fatalf("CQPacked: %v", err)
	}

	blob := f.Blob(0)
	if &p.Index[0] != &blob[0] || &p.Norms[0] != &blob[len(p.Index)] {
		t.Error("Index/Norms do not alias the tensor blob")
	}

	if cap(p.Index) != len(p.Index) {
		t.Errorf("cap(Index) = %d, want %d", cap(p.Index), len(p.Index))
	}
}

func TestCQPacked_Errors(t *testing.T) {
	f := cqFile(t, 8,
		testTensor{dtype: FP32, shape: []uint32{1}, blob: f32Blob(1)},
		cqTensor(1, 24, 24, 2, make([]byte, 24*2/8+2)),
	)

	_, err := f.CQPacked(0)
	if !errors.Is(err, ErrDType) {
		t.Errorf("CQPacked(FP32): err = %v, want ErrDType", err)
	}

	_, err = f.CQPacked(1)
	if err == nil || !strings.Contains(err.Error(), "power of two") {
		t.Errorf("CQPacked(group 24): err = %v, want one mentioning %q", err, "power of two")
	}

	_, err = f.CQPacked(2)
	if err == nil || errors.Is(err, ErrDType) {
		t.Errorf("CQPacked(2): err = %v, want an index error", err)
	}
}

// dequantPacked runs dequantCQ's arithmetic from a Packed: the same
// unpack, scale and FWHT, so the result must equal CQ bit for bit.
func dequantPacked(p Packed) []float32 {
	scale := float32(1 / math.Sqrt(float64(p.Group)))
	w := make([]float32, p.Out*p.In)
	idx := make([]uint8, p.Groups()*p.Group)
	buf := make([]float32, p.Group)

	for r := range p.Out {
		unpackLSB(idx, p.Index[r*p.RowBytes():(r+1)*p.RowBytes()], p.Width)

		for g := range p.Groups() {
			s := p.Norm(r, g) * scale
			for k, j := range idx[g*p.Group : (g+1)*p.Group] {
				buf[k] = p.Codebook[j] * s
			}

			fwht(buf)

			col := g * p.Group
			copy(w[r*p.In+col:r*p.In+min(col+p.Group, p.In)], buf)
		}
	}

	return w
}

// TestWhistleCQPacked checks every CQ record of whistle.cact: the packed
// view covers the whole blob, uses the header codebook, and decodes to
// exactly what CQ returns.
func TestWhistleCQPacked(t *testing.T) {
	f := openWhistle(t)

	n := 0

	for i := range f.Len() {
		rec, _ := f.Record(i)
		if rec.DType != CQ {
			continue
		}

		p, err := f.CQPacked(i)
		if err != nil {
			t.Fatalf("CQPacked(%d): %v", i, err)
		}

		cb, _ := f.Codebook(int(rec.Bits))
		if !slices.Equal(rec.Dims(), []int{p.Out, p.In}) || p.Group != 128 || p.Width != int(rec.Bits) ||
			!slices.Equal(p.Codebook, cb) || uint64(len(p.Index)+len(p.Norms)) != rec.NBytes ||
			len(p.Norms) != 2*p.Out*p.Groups() {
			t.Fatalf("CQPacked(%d) = {Out %d In %d Group %d Bits %d Width %d}, %d+%d bytes; record %+v",
				i, p.Out, p.In, p.Group, p.Bits, p.Width, len(p.Index), len(p.Norms), rec)
		}

		want, err := f.CQ(i)
		if err != nil {
			t.Fatalf("CQ(%d): %v", i, err)
		}

		if !slices.Equal(dequantPacked(p), want) {
			t.Errorf("CQPacked(%d) decodes differently from CQ(%d)", i, i)
		}

		n++
	}

	if n != 152 {
		t.Errorf("checked %d CQ records, want 152", n)
	}
}
