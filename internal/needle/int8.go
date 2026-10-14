package needle

import (
	"fmt"
	"math"
	"slices"
)

// Int8 selects the engine's int8 numerics, the deploy settings act_bits 8
// and kv_bits 8 of whistle.cact, at the places where the engine applies
// them (docs/FINDINGS.md F-041): upstream's fake_quant
// (needle/model/quantize.py) on the mHC heads' input and on the
// attention's queries, keys and values. The other CQ products keep float32
// inputs, as the engine's outputs show. The zero value is float32
// throughout.
type Int8 struct {
	// Act quantises the input rows of the mHC φ products (upstream's
	// _aq(_rms_unit(stream))) and the attention's queries per head
	// (maybe_quant_query) to int8.
	Act bool
	// KV quantises the attention's keys and values per head and position
	// (maybe_quant_kv with kv_bits 8), and keeps a decoder's caches in
	// int8: Int8Rows instead of float32.
	KV bool
}

// int8Max is the largest int8 magnitude a scale maps to, 2^(8−1) − 1.
const int8Max = 127

// QuantizeRow sets q to the symmetric int8 code of x and returns its
// scale s, so that x ≈ s · q: upstream's fake_quant with group_size =
// len(x) and bits 8 (needle/model/quantize.py): s = max|x| / 127 (1 for a
// zero row), q = clip(round(x / s), −128, 127), rounding half to even as
// jnp.round, all in float32 (docs/FINDINGS.md F-041). q must hold len(x)
// values.
func QuantizeRow(q []int8, x []float32) float32 {
	q = q[:len(x)]

	var m float32
	for _, v := range x {
		m = max(m, abs32(v))
	}

	s := float32(1)
	if m > 0 {
		s = m / int8Max
	}

	for i, v := range x {
		r := math.RoundToEven(float64(v / s))
		q[i] = int8(min(max(r, -int8Max-1), int8Max))
	}

	return s
}

// fakeQuant sets dst to x quantised to int8 in chunks of width values
// (QuantizeRow per chunk) and dequantised again, s · q in float32: the
// values an int8 operand stands for. dst may be x. Every chunk is its own,
// so the chunks run on tensor.Parallel in pieces of at least
// quantPieceValues.
func fakeQuant(dst, x []float32, width int) {
	chunks := len(x) / width

	forRows(chunks, (quantPieceValues+width-1)/width, func(lo, hi int) {
		q := make([]int8, width)

		for c := lo; c < hi; c++ {
			s := QuantizeRow(q, x[c*width:(c+1)*width])
			for i, v := range q {
				dst[c*width+i] = float32(v) * s
			}
		}
	})
}

// quantPieceValues is the fewest values of one fakeQuant piece, some
// 15 µs on one core (docs/FINDINGS.md F-038's piece sizes).
const quantPieceValues = 1 << 14

// quantInput returns a copy of x, rows of width values, quantised per row
// (fakeQuant): the input of a Linear WithInt8Input.
func quantInput(x []float32, width int) []float32 {
	dst := make([]float32, len(x))
	fakeQuant(dst, x, width)

	return dst
}

func abs32(x float32) float32 { return math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) }

// Int8Rows is a matrix in int8: rows of Cols codes, each chunk of Width
// codes (one head) with its own float32 scale, as QuantizeRow gives them.
// Value c of row r stands for S[r·Cols/Width + c/Width] · Q[r·Cols + c].
// It holds a decoder's keys or values in the KV cache and in the
// cross-attention memory, a quarter of their float32 size plus one scale
// per head and row (docs/FINDINGS.md F-041). Append only appends, so
// copies of an Int8Rows share their rows like slices.
type Int8Rows struct {
	Q           []int8
	S           []float32
	Cols, Width int
}

// NewInt8Rows returns an empty matrix of rows of cols codes in chunks of
// width; width must divide cols.
func NewInt8Rows(cols, width int) Int8Rows {
	if width <= 0 || cols%width != 0 {
		panic(fmt.Sprintf("needle: int8 rows of %d in chunks of %d", cols, width))
	}

	return Int8Rows{Cols: cols, Width: width}
}

// QuantizeRows quantises x, rows of cols values, per chunk of width.
func QuantizeRows(x []float32, cols, width int) Int8Rows {
	r := NewInt8Rows(cols, width)
	r.Q, r.S = make([]int8, 0, len(x)), make([]float32, 0, len(x)/width)

	return r.Append(x)
}

// Append quantises x, whole rows of Cols values, and returns r with them
// appended.
func (r Int8Rows) Append(x []float32) Int8Rows {
	if len(x)%r.Cols != 0 {
		panic(fmt.Sprintf("needle: %d values are not rows of %d", len(x), r.Cols))
	}

	n, ns, w := len(r.Q), len(r.S), r.Width
	r.Q, r.S = append(r.Q, make([]int8, len(x))...), append(r.S, make([]float32, len(x)/w)...)
	q, sc := r.Q[n:], r.S[ns:]

	forRows(len(sc), (quantPieceValues+w-1)/w, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			sc[c] = QuantizeRow(q[c*w:(c+1)*w], x[c*w:(c+1)*w])
		}
	})

	return r
}

// Len is the number of rows.
func (r Int8Rows) Len() int {
	if r.Cols == 0 {
		return 0
	}

	return len(r.Q) / r.Cols
}

// Clip returns r with its slices capped at their length, so that an
// Append to it copies rather than writes into memory r's owner may still
// append to (KVCache.Fork).
func (r Int8Rows) Clip() Int8Rows {
	r.Q, r.S = slices.Clip(r.Q), slices.Clip(r.S)

	return r
}

// Floats returns the values the codes stand for, s · q in float32, rows of
// Cols.
func (r Int8Rows) Floats() []float32 {
	out := make([]float32, len(r.Q))

	for i, c := range r.Q {
		out[i] = float32(c) * r.S[i/r.Width]
	}

	return out
}

// Bytes is the memory the codes and scales take.
func (r Int8Rows) Bytes() int { return len(r.Q) + 4*len(r.S) }

// AllocatedBytes is the memory the codes' and scales' backing arrays
// hold, Append's spare capacity included: at least Bytes.
func (r Int8Rows) AllocatedBytes() int { return cap(r.Q) + 4*cap(r.S) }

// AttendInt8 is softmax(q kᵀ / scale) v for t query rows against the int8
// keys k and values v of the same rows (docs/FINDINGS.md F-041). q holds
// t rows of heads chunks of k.Width values; query head h reads chunk
// h / (heads / kvHeads) of k and v, kvHeads = k.Cols / k.Width, as the
// self-attention's GQA does; query row i sees the first keys(i) rows. The
// result is t rows of heads · v.Width values. A non-nil probs receives
// the attention weights, (i·heads + h)·k.Len() + j for row i, head h and
// key j; it is not written past keys(i).
//
// With quantQ, each query head is quantised to int8 (QuantizeRow), so
// that each logit is an exact int32 dot product times the two scales,
// float32(q·k) · (s_q · s_k) / scale; otherwise the float32 query meets
// the key codes, (Σ q · code) · s_k / scale. The weights times the value
// scales then add the value codes, one key after the other per output.
func AttendInt8(q []float32, t, heads int, k, v Int8Rows, keys func(i int) int, scale float32, quantQ bool,
	probs []float32,
) []float32 {
	spans := func(i int) [2]Span { return [2]Span{{}, {0, keys(i)}} }

	return AttendInt8Spans(q, t, heads, k, v, spans, scale, quantQ, probs)
}

// AttendInt8Spans is AttendInt8 with query row i seeing the key rows of
// the two ascending, disjoint spans(i) instead of the first keys(i): the
// band of a sliding window and of a KV window with its sink on the int8 KV
// cache (Attention.SetWindow, SetKVWindow, docs/FINDINGS.md F-060). probs
// is written at the spans' keys only. With the first span empty and the
// second from 0 it is AttendInt8 bit for bit.
func AttendInt8Spans(q []float32, t, heads int, k, v Int8Rows, spans func(i int) [2]Span, scale float32,
	quantQ bool, probs []float32,
) []float32 {
	qk, vd := k.Width, v.Width
	kvHeads := k.Cols / qk
	n := k.Len()
	out := make([]float32, t*heads*vd)
	q8, dots := make([]int8, qk), make([]int32, n)
	w := make([]float32, n)

	for h := range heads {
		kv := h / (heads / kvHeads)

		for i := range t {
			sp := spans(i)
			qi, s := q[(i*heads+h)*qk:(i*heads+h+1)*qk], w[:sp[0].Len()+sp[1].Len()]

			var sq float32
			if quantQ {
				sq = QuantizeRow(q8, qi)
			}

			// The logits of each span's keys, s[o:] from key sp[p].Lo on.
			o := 0

			for _, p := range sp {
				sj := s[o : o+p.Len()]
				o += p.Len()

				if quantQ {
					int8Logits(sj, dots, q8, sq, k, p.Lo, kv, scale)

					continue
				}

				dotsF32I8(sj, qi, k.Q[p.Lo*k.Cols+kv*qk:], k.Cols)

				for j := range sj {
					sj[j] = sj[j] * k.S[(p.Lo+j)*kvHeads+kv] / scale
				}
			}

			softmaxInPlace(s)

			o = 0

			for _, p := range sp {
				sj, vs := s[o:o+p.Len()], v.S[p.Lo*kvHeads:]

				if probs != nil {
					copy(probs[(i*heads+h)*n+p.Lo:], sj)
				}

				for j := range sj {
					sj[j] *= vs[j*kvHeads+kv]
				}

				axpysI8(out[(i*heads+h)*vd:(i*heads+h+1)*vd], sj, v.Q[p.Lo*v.Cols+kv*vd:], v.Cols)
				o += p.Len()
			}
		}
	}

	return out
}

// int8Logits sets s[j] to the logit of the int8 query q8 with scale sq
// against key row lo + j of k's KV head kv: the exact int32 dot product
// times the two scales, over scale (AttendInt8Spans). dots is scratch of
// at least len(s).
func int8Logits(s []float32, dots []int32, q8 []int8, sq float32, k Int8Rows, lo, kv int, scale float32) {
	qk, kvHeads := k.Width, k.Cols/k.Width
	dotsI8(dots[:len(s)], q8, k.Q[lo*k.Cols+kv*qk:], k.Cols)

	for j := range s {
		s[j] = float32(dots[j]) * (sq * k.S[(lo+j)*kvHeads+kv]) / scale
	}
}

// dotsF32I8 sets dst[j] = Σ_i q[i] · k[j·stride + i], the float32 query
// against int8 codes, summed in order.
func dotsF32I8(dst, q []float32, k []int8, stride int) {
	for j := range dst {
		row := k[j*stride : j*stride+len(q)]

		var s float32
		for i, x := range q {
			s += x * float32(row[i])
		}

		dst[j] = s
	}
}

// FakeQuant replaces x, chunks of width values, by the values of its int8
// codes per chunk (QuantizeRow): what an int8 operand stands for in a
// float32 product, such as a query per head that meets float32 keys.
func FakeQuant(x []float32, width int) { fakeQuant(x, x, width) }
