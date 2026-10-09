package cact

import (
	"encoding/binary"
	"errors"
	"math"
	"math/bits"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// lloydMaxUnit is quantize.py _lloyd_max_gaussian(bits) @ af2654dffa for 2, 3
// and 4 bits: the unit-variance Gaussian Lloyd-Max levels in float64,
// recomputed with numpy (docs/FINDINGS.md F-004).
var lloydMaxUnit = map[int][]float64{
	2: {-1.5066708475715247, -0.45144060939344055, 0.4529474939511613, 1.5099326242547164},
	3: {
		-2.141247862845774, -1.3337164609389895, -0.7483982925010272, -0.23774016363616235,
		0.2500446195658646, 0.7555868210708537, 1.3379826350451236, 2.1476362705161423,
	},
	4: {
		-2.709990028558534, -2.0444556051349227, -1.5947570221842273, -1.233336191082712,
		-0.9222139662739897, -0.6395373737252813, -0.37266394934485414, -0.11445072030649898,
		0.14380628677643076, 0.4039889754802918, 0.6726933686864701, 0.9614053009735016,
		1.2755257831234952, 1.637058796059282, 2.0844263722579237, 2.7410141261982846,
	},
}

// portCodebook ports quantize.py _cq_codebook_np(bits, group): the unit
// levels over sqrt(group), rounded to float32. Width 5 is the ternary record
// width, whose levels _TERNARY_CB fixes; 1 bit is ±sqrt(2/pi).
func portCodebook(width, group int) []float32 {
	unit := lloydMaxUnit[width]

	switch width {
	case 1:
		c := math.Sqrt(2 / math.Pi)
		unit = []float64{-c, c}
	case ternaryRecordBits:
		unit = []float64{-1.2240064, 0, 1.2240064}
	}

	cb := make([]float32, len(unit))
	for i, c := range unit {
		cb[i] = float32(c / math.Sqrt(float64(group)))
	}

	return cb
}

// portStoredCodebook is the 28-float header codebook _pack_cact writes for
// a file exported with group size group.
func portStoredCodebook(group int) []float32 {
	return slices.Concat(portCodebook(2, group), portCodebook(3, group), portCodebook(4, group))
}

// sylvester is _cq_hadamard_np(n) before the 1/sqrt(n): the ±1 Hadamard
// matrix grown by np.block([[H, H], [H, -H]]).
func sylvester(n int) [][]float64 {
	h := [][]float64{{1}}
	for len(h) < n {
		m := len(h)
		next := make([][]float64, 2*m)

		for i := range next {
			next[i] = make([]float64, 2*m)
			for j := range next[i] {
				v := h[i%m][j%m]
				if i >= m && j >= m {
					v = -v
				}

				next[i][j] = v
			}
		}

		h = next
	}

	return h
}

// packLSB ports export.py _pack_lsb on a flat index stream: every run of 8
// indices is OR'd into a word at offsets i*width and written as width
// little-endian bytes.
func packLSB(idx []uint8, width int) []byte {
	out := make([]byte, 0, len(idx)*width/8)

	for c := 0; c < len(idx); c += 8 {
		var word uint64
		for i := range 8 {
			word |= uint64(idx[c+i]) << (i * width)
		}

		for b := range width {
			out = append(out, byte(word>>(8*b)))
		}
	}

	return out
}

// tritCrumb inverts export.py _unpack_ternary_crumbs: trits 0, 1, 2 are
// stored as the crumbs 3, 0, 1.
var tritCrumb = [3]uint8{3, 0, 1}

// float32ToFloat16 rounds to the nearest binary16 value, ties to even, like
// numpy's astype(np.float16). It searches the monotonic positive codes, so
// it only has to trust float16ToFloat32.
func float32ToFloat16(f float32) uint16 {
	var sign uint16
	if math.Signbit(float64(f)) {
		sign, f = 0x8000, -f
	}

	if f >= 65520 { // half an ulp past the largest finite half
		return sign | 0x7c00
	}

	lo, hi := uint16(0), uint16(0x7bff)
	for lo < hi { // largest code whose value is <= f
		mid := lo + (hi-lo+1)/2
		if float16ToFloat32(mid) <= f {
			lo = mid
		} else {
			hi = mid - 1
		}
	}

	if lo == 0x7bff {
		return sign | lo
	}

	below, above := f-float16ToFloat32(lo), float16ToFloat32(lo+1)-f
	if above < below || (above == below && lo&1 == 1) {
		lo++
	}

	return sign | lo
}

// cqPacked is what portPack produces: the record blob and, for the error
// bound, the intermediate values _cq_pack computes on the way.
type cqPacked struct {
	blob   []byte    // packed indices, then the FP16 norms, as _q concatenates them
	unit   []float64 // rotated groups over their norm, [out][inPad]
	idx    []uint8   // the chosen levels, [out][inPad]
	norm   []float64 // per-group L2 norms before the FP16 rounding, [out][groups]
	norm16 []float32 // the same norms after it
}

// portPack ports export.py _cq_pack, generalised from WEIGHT_BITS to every
// record width (1-4 bits and ternary): pad the input to the group, rotate
// each group by H = sylvester/sqrt(group), split off its L2 norm, pick the
// nearest level of cb for every coordinate of the unit vector (ties to the
// lower one, as _nearest_idx), and pack. It works in float64 where
// _cq_pack uses float32, which only moves exact ties.
func portPack(w []float32, out, in, width, group int, cb []float32) cqPacked {
	groups := (in + group - 1) / group
	inPad := groups * group

	h := sylvester(group)
	invSqrt := 1 / math.Sqrt(float64(group))

	p := cqPacked{
		unit:   make([]float64, out*inPad),
		idx:    make([]uint8, out*inPad),
		norm:   make([]float64, out*groups),
		norm16: make([]float32, out*groups),
	}
	codes := make([]uint8, out*inPad)
	norms := make([]byte, 0, 2*out*groups)
	g := make([]float64, group)
	rot := make([]float64, group)

	for r := range out {
		for gi := range groups {
			for k := range g {
				g[k] = 0
				if col := gi*group + k; col < in {
					g[k] = float64(w[r*in+col])
				}
			}

			var sq float64

			for j := range rot {
				rot[j] = 0
				for k := range g {
					rot[j] += g[k] * h[k][j] * invSqrt
				}

				sq += rot[j] * rot[j]
			}

			n := math.Sqrt(sq)
			n16 := float32ToFloat16(float32(n))
			p.norm[r*groups+gi] = n
			p.norm16[r*groups+gi] = float16ToFloat32(n16)
			norms = binary.LittleEndian.AppendUint16(norms, n16)

			for j, v := range rot {
				at := r*inPad + gi*group + j
				u := v / math.Max(n, 1e-12)
				k := nearestLevel(u, cb)

				p.unit[at], p.idx[at], codes[at] = u, k, k
				if width == ternaryRecordBits {
					codes[at] = tritCrumb[k]
				}
			}
		}
	}

	idxBits := width
	if width == ternaryRecordBits {
		idxBits = 2
	}

	p.blob = append(packLSB(codes, idxBits), norms...)

	return p
}

// nearestLevel ports export.py _nearest_idx for one value: searchsorted
// (side left), clipped to [1, len-1], then the closer neighbour, ties low.
func nearestLevel(x float64, cb []float32) uint8 {
	pos := 0
	for pos < len(cb) && float64(cb[pos]) < x {
		pos++
	}

	pos = min(max(pos, 1), len(cb)-1)
	if math.Abs(x-float64(cb[pos-1])) <= math.Abs(x-float64(cb[pos])) {
		return uint8(pos - 1)
	}

	return uint8(pos)
}

// levelBound bounds |u - q| for the nearest level q of the sorted codebook
// cb: half the gap of the cell u falls in, or the overshoot past an end
// level.
func levelBound(u float64, cb []float32) float64 {
	first, last := float64(cb[0]), float64(cb[len(cb)-1])

	switch {
	case u <= first:
		return first - u
	case u >= last:
		return u - last
	}

	k := 1
	for float64(cb[k]) < u {
		k++
	}

	return (float64(cb[k]) - float64(cb[k-1])) / 2
}

// fp32Gamma bounds the relative float32 error of the dequant of a group:
// the FWHT rounds once per stage and the input x = cb * (norm *
// float32(1/sqrt(group))) three times, so output j is off by at most
// γ(log2(group) + 3) * sum_k |x_k| / sqrt(group), with γ(k) = ku / (1 - ku)
// and u = 2^-24.
func fp32Gamma(group int) float64 {
	k := float64(bits.Len(uint(group))-1) + 3
	u := 0x1p-24

	return k * u / (1 - k*u)
}

// cqFile encodes a .cact holding the tensors, with the header codebook
// _pack_cact writes for group.
func cqFile(t *testing.T, group int, tensors ...testTensor) *File {
	t.Helper()

	f, err := Parse(encodeTestFile(t, testHeader(), portStoredCodebook(group), tensors))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	return f
}

func cqTensor(out, in, group, width int, blob []byte) testTensor {
	return testTensor{
		dtype: CQ, shape: []uint32{uint32(out), uint32(in)}, blob: blob,
		group: uint32(group), bits: uint32(width),
	}
}

// TestUnpackLSB pins the bitstream by hand: index k sits in bits
// [k*width, (k+1)*width) of the row, LSB first, across byte boundaries.
func TestUnpackLSB(t *testing.T) {
	for _, tc := range []struct {
		name   string
		width  int
		packed []byte
		want   []uint8
	}{
		// 0xa5 = 0b1010_0101, read from bit 0 up.
		{"1 bit", 1, []byte{0xa5}, []uint8{1, 0, 1, 0, 0, 1, 0, 1}},
		// 0xe4 = 0b11_10_01_00, 0x1b = 0b00_01_10_11.
		{"2 bits", 2, []byte{0xe4, 0x1b}, []uint8{0, 1, 2, 3, 3, 2, 1, 0}},
		// sum(k << 3k) = 0xfac688; indices 2 and 5 straddle a byte boundary.
		{"3 bits", 3, []byte{0x88, 0xc6, 0xfa}, []uint8{0, 1, 2, 3, 4, 5, 6, 7}},
		// 7 << 3k for k = 0..7 sets all 24 bits; the second chunk holds 1 at k = 2 only (1 << 6).
		{"3 bits, two chunks", 3, []byte{0xff, 0xff, 0xff, 0x40, 0x00, 0x00}, []uint8{
			7, 7, 7, 7, 7, 7, 7, 7,
			0, 0, 1, 0, 0, 0, 0, 0,
		}},
		// Nibbles: the low one is the even column.
		{"4 bits", 4, []byte{0x21, 0x43, 0x65, 0xf7}, []uint8{1, 2, 3, 4, 5, 6, 7, 15}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint8, len(tc.want))
			unpackLSB(got, tc.packed, tc.width)

			if !slices.Equal(got, tc.want) {
				t.Errorf("unpackLSB(% x, %d) = %v, want %v", tc.packed, tc.width, got, tc.want)
			}

			if packed := packLSB(tc.want, tc.width); !slices.Equal(packed, tc.packed) {
				t.Errorf("packLSB(%v, %d) = % x, want % x", tc.want, tc.width, packed, tc.packed)
			}
		})
	}
}

// TestUnpackLSB_RoundTrip runs packLSB → unpackLSB over every width on
// three chunks of random indices.
func TestUnpackLSB_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	for width := 1; width <= 4; width++ {
		idx := make([]uint8, 24)
		for i := range idx {
			idx[i] = uint8(rng.IntN(1 << width))
		}

		got := make([]uint8, len(idx))
		unpackLSB(got, packLSB(idx, width), width)

		if !slices.Equal(got, idx) {
			t.Errorf("%d bits: round trip %v, want %v", width, got, idx)
		}
	}
}

func TestFWHT(t *testing.T) {
	for _, n := range []int{1, 2, 8, 128} {
		h := sylvester(n)
		rng := rand.New(rand.NewPCG(uint64(n), 3))

		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.IntN(17) - 8) // small integers keep every sum exact
		}

		got := slices.Clone(x)
		fwht(got)

		for j := range n {
			var want float64
			for k := range n {
				want += float64(x[k]) * h[k][j]
			}

			if float64(got[j]) != want {
				t.Fatalf("n=%d: fwht[%d] = %v, want x @ H = %v", n, j, got[j], want)
			}
		}
	}
}

// TestCQ_Hand checks dequantised values computed by hand on group 8, where
// H/sqrt(8) maps a constant group onto its first column and a one-hot group
// onto one ± row of the Sylvester matrix.
func TestCQ_Hand(t *testing.T) {
	const half, one = 0x3800, 0x3c00 // FP16 0.5 and 1.0

	ternaryLevel := float32(1.2240064 / math.Sqrt(8))
	binaryLevel := float64(float32(math.Sqrt(2/math.Pi) / math.Sqrt(8)))

	for _, tc := range []struct {
		name  string
		width int
		blob  []byte
		want  []float64
	}{
		// All eight indices 3 → the top 2-bit level 1.5099.../sqrt(8); with
		// norm 0.5 the group sums onto column 0 as 8 * level * 0.5 / sqrt(8).
		{"CQ2 constant", 2, []byte{0xff, 0xff, half & 0xff, half >> 8}, []float64{
			8 * float64(portCodebook(2, 8)[3]) * 0.5 / math.Sqrt(8), 0, 0, 0, 0, 0, 0, 0,
		}},
		// Crumb 1 → trit 2 → +c at column 0, crumb 0 → trit 1 → 0 elsewhere:
		// +c times row 0 of H/sqrt(8).
		{"ternary one-hot 0", ternaryRecordBits, []byte{0x01, 0x00, one & 0xff, one >> 8}, rowOf(0, ternaryLevel)},
		// Crumb 3 → trit 0 → -c at column 1: -c times row 1 = [+ - + - + - + -].
		{"ternary one-hot 1", ternaryRecordBits, []byte{0x0c, 0x00, one & 0xff, one >> 8}, rowOf(1, -ternaryLevel)},
		// +c at column 2: row 2 = [+ + - - + + - -].
		{"ternary one-hot 2", ternaryRecordBits, []byte{0x10, 0x00, one & 0xff, one >> 8}, rowOf(2, ternaryLevel)},
		// +c at column 5 (crumb 1 in bits 10-11, i.e. byte 1 bit 2): row 5 = [+ - + - - + - +].
		{"ternary one-hot 5", ternaryRecordBits, []byte{0x00, 0x04, one & 0xff, one >> 8}, rowOf(5, ternaryLevel)},
		// 1-bit indices 1,0,0,0,0,0,0,0 (0x01): column 0 is +b, the rest -b,
		// so the group is -b*[1..1] + 2b*e0 → -b*sqrt(8)*e0 + 2b/sqrt(8) everywhere.
		{"CQ1", 1, []byte{0x01, one & 0xff, one >> 8}, func() []float64 {
			want := make([]float64, 8)
			for j := range want {
				want[j] = 2 * binaryLevel / math.Sqrt(8)
			}

			want[0] -= binaryLevel * math.Sqrt(8)

			return want
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := cqFile(t, 8, cqTensor(1, 8, 8, tc.width, tc.blob))

			got, err := f.CQ(0)
			if err != nil {
				t.Fatalf("CQ: %v", err)
			}

			for j, want := range tc.want {
				if math.Abs(float64(got[j])-want) > 1e-6*math.Max(1, math.Abs(want)) {
					t.Errorf("w[%d] = %v, want %v (all %v)", j, got[j], want, got)
				}
			}
		})
	}
}

// rowOf is level times row r of the 8-point Sylvester matrix over sqrt(8).
func rowOf(r int, level float32) []float64 {
	h := sylvester(8)

	want := make([]float64, 8)
	for j := range want {
		want[j] = float64(level) * h[r][j] / math.Sqrt(8)
	}

	return want
}

// cqShapes cover one and several groups, every padding case (none, a few
// columns, a single real column in the last group) and groups 8 to 128.
var cqShapes = []struct{ out, in, group int }{
	{1, 8, 8},
	{3, 16, 8},
	{4, 13, 8},
	{2, 40, 32},
	{2, 200, 128},
	{3, 384, 128},
	{5, 129, 128},
}

// cqWidths are the record widths, ternary included.
var cqWidths = []int{1, 2, 3, 4, ternaryRecordBits}

// TestCQ_MatrixReference dequantises random indices and norms and compares
// with _cq_unpack's arithmetic in float64: (cb[idx] * norm) @ H with H the
// explicit Sylvester matrix over sqrt(group), cut to [out][in]. The
// tolerance is the float32 rounding bound of fp32Gamma.
func TestCQ_MatrixReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 5))

	for _, s := range cqShapes {
		for _, width := range cqWidths {
			groups := (s.in + s.group - 1) / s.group
			inPad := groups * s.group
			cb := portCodebook(width, s.group)

			idxBits := width
			if width == ternaryRecordBits {
				idxBits = 2
			}

			idx := make([]uint8, s.out*inPad)
			codes := make([]uint8, len(idx))

			for i := range idx {
				idx[i] = uint8(rng.IntN(len(cb)))

				codes[i] = idx[i]
				if width == ternaryRecordBits {
					codes[i] = tritCrumb[idx[i]]
				}
			}

			blob := packLSB(codes, idxBits)
			norm := make([]float32, s.out*groups)

			for i := range norm {
				h := float32ToFloat16(float32(rng.ExpFloat64() * 4))
				norm[i] = float16ToFloat32(h)
				blob = binary.LittleEndian.AppendUint16(blob, h)
			}

			f := cqFile(t, s.group, cqTensor(s.out, s.in, s.group, width, blob))

			got, err := f.CQ(0)
			if err != nil {
				t.Fatalf("%v bits %d: CQ: %v", s, width, err)
			}

			if len(got) != s.out*s.in {
				t.Fatalf("%v bits %d: %d values, want %d", s, width, len(got), s.out*s.in)
			}

			h := sylvester(s.group)
			sqrtG := math.Sqrt(float64(s.group))
			gamma := fp32Gamma(s.group)

			for r := range s.out {
				for col := range s.in {
					gi, j := col/s.group, col%s.group

					var want, abs float64

					for k := range s.group {
						x := float64(cb[idx[r*inPad+gi*s.group+k]]) * float64(norm[r*groups+gi])
						want += x * h[k][j] / sqrtG
						abs += math.Abs(x)
					}

					tol := gamma * abs / sqrtG
					if diff := math.Abs(float64(got[r*s.in+col]) - want); diff > tol {
						t.Fatalf("%v bits %d: w[%d][%d] = %v, want %v (diff %.3g > tol %.3g)",
							s, width, r, col, got[r*s.in+col], want, diff, tol)
					}
				}
			}
		}
	}
}

// TestCQ_RoundTrip packs random matrices with the _cq_pack port and checks
// that dequantising restores them as closely as the codebook allows.
//
// H is orthonormal, so per group the error in w equals the error of the
// rotated group, u*n - q*n16 (u the unit vector, n its norm, q the chosen
// levels, n16 the FP16 norm):
//
//	|w - ŵ| <= n * |u - q| + |n - n16| * |q| + fp32
//
// Every |u_k - q_k| is at most half the gap of the codebook cell u_k falls
// in, or the overshoot past an end level (levelBound). fp32 is the float32
// rounding of fp32Gamma summed over the group: γ * n16 * sum_k |q_k|.
func TestCQ_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 7))
	rowScale := []float64{0.1, 1, 10, 0} // every fourth row is all zero

	for _, s := range cqShapes {
		for _, width := range cqWidths {
			w := make([]float32, s.out*s.in)
			for i := range w {
				w[i] = float32(rng.NormFloat64() * rowScale[i/s.in%len(rowScale)])
			}

			cb := portCodebook(width, s.group)
			p := portPack(w, s.out, s.in, width, s.group, cb)
			f := cqFile(t, s.group, cqTensor(s.out, s.in, s.group, width, p.blob))

			got, err := f.CQ(0)
			if err != nil {
				t.Fatalf("%v bits %d: CQ: %v", s, width, err)
			}

			groups := (s.in + s.group - 1) / s.group
			inPad := groups * s.group
			gamma := fp32Gamma(s.group)

			var errSum, normSum float64

			for r := range s.out {
				for gi := range groups {
					at := r*inPad + gi*s.group

					var levelSq, qSq, qAbs float64

					for k := range s.group {
						b := levelBound(p.unit[at+k], cb)
						q := float64(cb[p.idx[at+k]])
						levelSq += b * b
						qSq += q * q
						qAbs += math.Abs(q)
					}

					n, n16 := p.norm[r*groups+gi], float64(p.norm16[r*groups+gi])
					bound := n*math.Sqrt(levelSq) + math.Abs(n-n16)*math.Sqrt(qSq) + gamma*n16*qAbs

					var diffSq, wSq float64

					for col := gi * s.group; col < min((gi+1)*s.group, s.in); col++ {
						d := float64(got[r*s.in+col]) - float64(w[r*s.in+col])
						diffSq += d * d
						wSq += float64(w[r*s.in+col]) * float64(w[r*s.in+col])
					}

					if diff := math.Sqrt(diffSq); diff > bound {
						t.Errorf("%v bits %d: row %d group %d: |w - ŵ| = %.6g > bound %.6g (norm %.6g)",
							s, width, r, gi, diff, bound, n)
					}

					errSum += math.Sqrt(diffSq)
					normSum += math.Sqrt(wSq)
				}
			}

			t.Logf("%v bits %d: sum |w - ŵ| / sum |w| = %.3f", s, width, errSum/normSum)
		}
	}
}

func TestCQ_Errors(t *testing.T) {
	f := cqFile(t, 8,
		testTensor{dtype: FP32, shape: []uint32{1}, blob: f32Blob(1)},
		// Crumb code 2 has no trit; _cq_unpack would index past the 3-level codebook.
		cqTensor(1, 8, 8, ternaryRecordBits, []byte{0x00, 0x20, 0x00, 0x3c}),
		// Group 24 passes Parse (a multiple of 8) but has no Sylvester Hadamard.
		cqTensor(1, 24, 24, 2, make([]byte, 24*2/8+2)),
	)

	_, err := f.CQ(0)
	if !errors.Is(err, ErrDType) {
		t.Errorf("CQ(FP32): err = %v, want ErrDType", err)
	}

	_, err = f.CQ(3)
	if err == nil || errors.Is(err, ErrDType) {
		t.Errorf("CQ(3): err = %v, want an index error", err)
	}

	for i, want := range map[int]string{1: "crumb", 2: "power of two"} {
		_, err := f.CQ(i)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("CQ(%d): err = %v, want one mentioning %q", i, err, want)
		}
	}
}

// TestCQ_ZeroBlobs reads the all-zero CQ2 and ternary tensors of the
// container fixture: zero norms and the zero crumb both give zeros.
func TestCQ_ZeroBlobs(t *testing.T) {
	f := parseTestFile(t)

	for i, n := range map[int]int{3: 400, 4: 128} {
		got, err := f.CQ(i)
		if err != nil {
			t.Fatalf("CQ(%d): %v", i, err)
		}

		if len(got) != n || slices.ContainsFunc(got, func(v float32) bool { return v != 0 }) {
			t.Errorf("CQ(%d) = %d values %v, want %d zeros", i, len(got), got[:min(len(got), 8)], n)
		}
	}
}

func TestFloat32ToFloat16(t *testing.T) {
	for _, tc := range []struct {
		f    float32
		want uint16
	}{
		{0, 0},
		{1, 0x3c00},
		{-2, 0xc000},
		{0.5, 0x3800},
		{65504, 0x7bff},
		{65519, 0x7bff},
		{1e6, 0x7c00},
		{1 + 0x1p-11, 0x3c00},           // tie between 1 and 1+2^-10: to even
		{1 + 3*0x1p-11, 0x3c02},         // tie between 1+2^-10 and 1+2^-9: to even
		{1 + 0x1p-11 + 0x1p-20, 0x3c01}, // just above the tie
		{0x1p-24, 0x0001},               // smallest subnormal
		{0x1p-26, 0x0000},               // below half of it
	} {
		if got := float32ToFloat16(tc.f); got != tc.want {
			t.Errorf("float32ToFloat16(%v) = 0x%04x, want 0x%04x", tc.f, got, tc.want)
		}
	}
}
