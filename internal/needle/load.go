package needle

import (
	"fmt"
	"math"

	"github.com/cwbudde/go-whistle/internal/tensor"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// loader reads one block's tensors, prefix+name, from a Source. The first
// error sticks: later reads return nil and zero sizes, and err reports it.
type loader struct {
	src    weights.Source
	prefix string
	err    error
}

// failf records an error unless one is already recorded.
func (l *loader) failf(format string, args ...any) {
	if l.err == nil {
		l.err = fmt.Errorf("needle: "+format, args...)
	}
}

// tensor returns a tensor of the given rank and its shape.
func (l *loader) tensor(name string, rank int) ([]float32, []int) {
	if l.err != nil {
		return nil, nil
	}

	data, shape, err := l.src.Tensor(l.prefix + name)
	if err != nil {
		l.failf("%s%s: %w", l.prefix, name, err)

		return nil, nil
	}

	if len(shape) != rank {
		l.failf("%s%s has shape %v, want rank %d", l.prefix, name, shape, rank)

		return nil, nil
	}

	return data, shape
}

// matrix returns a rank-2 tensor and its dimensions, which must equal rows
// and cols where those are positive.
func (l *loader) matrix(name string, rows, cols int) ([]float32, int, int) {
	data, shape := l.tensor(name, 2)
	if data == nil {
		return nil, 0, 0
	}

	if (rows > 0 && shape[0] != rows) || (cols > 0 && shape[1] != cols) {
		l.failf("%s%s has shape %v, want (%d, %d)", l.prefix, name, shape, rows, cols)

		return nil, 0, 0
	}

	return data, shape[0], shape[1]
}

// vector returns a rank-1 tensor of length n, any length if n is 0.
func (l *loader) vector(name string, n int) []float32 {
	data, shape := l.tensor(name, 1)
	if data == nil {
		return nil
	}

	if n > 0 && shape[0] != n {
		l.failf("%s%s has length %d, want %d", l.prefix, name, shape[0], n)

		return nil
	}

	return data
}

// linear computes dst = x · wᵀ for rows of x of width in and w [out, in],
// the .cact layout of a Flax Dense kernel (upstream x @ kernel).
func linear(dst, x []float32, rows int, w []float32, out, in int) {
	tensor.MatMulTransB(dst, out, x, w, rows, out, in)
}

// sigmoid is jax.nn.sigmoid in float32.
func sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// softmaxInPlace is jax.nn.softmax over s: exp(s − max) / Σ.
func softmaxInPlace(s []float32) {
	m := float32(math.Inf(-1))
	for _, v := range s {
		m = max(m, v)
	}

	var sum float32

	for i, v := range s {
		e := float32(math.Exp(float64(v - m)))
		s[i] = e
		sum += e
	}

	for i := range s {
		s[i] /= sum
	}
}
