//go:build arm64 && !purego

package needle

// dotsI8Kernel runs the NEON kernel when the row width is a multiple of
// 16, the Go loop otherwise.
func dotsI8Kernel(dst []int32, q, k []int8, stride int) {
	if len(q) == 0 || len(q)%16 != 0 {
		dotsI8Generic(dst, q, k, stride)

		return
	}

	dotsI8NEON(&dst[0], &q[0], &k[0], len(dst), len(q), stride)
}

// axpysI8Kernel runs the NEON kernel when the output width is a multiple
// of 16, the Go loop otherwise.
func axpysI8Kernel(out, w []float32, v []int8, stride int) {
	if len(out) == 0 || len(out)%16 != 0 {
		axpysI8Generic(out, w, v, stride)

		return
	}

	axpysI8NEON(&out[0], &w[0], &v[0], len(w), len(out), stride)
}

// dotsI8NEON is dotsI8 for n ≥ 1 keys and a width that is a positive
// multiple of 16.
//
//go:noescape
func dotsI8NEON(dst *int32, q, k *int8, n, width, stride int)

// axpysI8NEON is axpysI8 for n ≥ 1 keys and a width that is a positive
// multiple of 16.
//
//go:noescape
func axpysI8NEON(out, w *float32, v *int8, n, width, stride int)
