package needle

import "fmt"

// dotsI8 sets dst[j] = Σ_i q[i] · k[j·stride + i] over the len(q) codes of
// q, for j < len(dst): the int8 logits of AttendInt8. The sums are exact
// in int32 (|q[i] · k| ≤ 2¹⁴, so up to 2¹⁷ terms), so every kernel gives
// the same result. On arm64 and on amd64 with AVX2 an assembly kernel runs
// when len(q) is a multiple of 16 (docs/FINDINGS.md F-041).
func dotsI8(dst []int32, q, k []int8, stride int) {
	if len(dst) == 0 {
		return
	}

	if need := (len(dst)-1)*stride + len(q); len(k) < need || stride < len(q) {
		panic(fmt.Sprintf("needle: %d int8 dot products of %d at stride %d over %d codes", len(dst), len(q), stride,
			len(k)))
	}

	dotsI8Kernel(dst, q, k, stride)
}

// axpysI8 adds Σ_j w[j] · v[j·stride + d] to out[d] for d < len(out), j <
// len(w): the weighted int8 values of AttendInt8. For each d it adds one
// key after the other, each product rounded to float32 before the add
// (no FMA), so the assembly kernels (arm64, amd64 with AVX2, for len(out)
// a multiple of 16) equal axpysI8Generic bit for bit.
func axpysI8(out, w []float32, v []int8, stride int) {
	if len(w) == 0 {
		return
	}

	if need := (len(w)-1)*stride + len(out); len(v) < need || stride < len(out) {
		panic(fmt.Sprintf("needle: %d int8 axpys of %d at stride %d over %d codes", len(w), len(out), stride, len(v)))
	}

	axpysI8Kernel(out, w, v, stride)
}

// dotsI8Generic is dotsI8 in Go.
func dotsI8Generic(dst []int32, q, k []int8, stride int) {
	for j := range dst {
		row := k[j*stride : j*stride+len(q)]

		var s int32
		for i, x := range q {
			s += int32(x) * int32(row[i])
		}

		dst[j] = s
	}
}

// axpysI8Generic is axpysI8 in Go.
func axpysI8Generic(out, w []float32, v []int8, stride int) {
	for j, wj := range w {
		row := v[j*stride : j*stride+len(out)]

		for d, c := range row {
			out[d] += float32(wj * float32(c))
		}
	}
}
