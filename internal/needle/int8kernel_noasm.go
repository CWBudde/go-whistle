//go:build (!amd64 && !arm64) || purego

package needle

func dotsI8Kernel(dst []int32, q, k []int8, stride int) { dotsI8Generic(dst, q, k, stride) }

func axpysI8Kernel(out, w []float32, v []int8, stride int) { axpysI8Generic(out, w, v, stride) }
