package align

import "math"

// Step is a DTW step pattern.
type Step int

const (
	_ Step = iota
	// StepWhisper is Whisper's dtw: from (i−1, j−1), (i−1, j) or (i, j−1),
	// so a row may share a column with the next one and stretch over many.
	// Ties prefer the column move, then the row move.
	StepWhisper
	// StepNoShare forbids two rows on one column: from (i−1, j−1) or
	// (i, j−1) only, so every row holds at least one column of its own.
	StepNoShare
)

// DTW is the minimum-cost monotonic path through cost, rows × cols values
// row-major, from (0, 0) to (rows−1, cols−1) under step. It returns, per
// row, the first and the last column the path visits on it; the first are
// the jump times of Whisper's find_alignment. Under StepNoShare more rows
// than columns have no path; the results are then nil.
func DTW(cost []float64, rows, cols int, step Step) (starts, ends []int) {
	if rows == 0 || cols == 0 || (step == StepNoShare && rows > cols) {
		return nil, nil
	}

	trace, ok := accumulate(cost, rows, cols, step)
	if !ok {
		return nil, nil
	}

	return backtrace(trace, rows, cols)
}

// Moves of the trace: from (i−1, j−1), from (i−1, j), from (i, j−1).
const (
	moveDiag int8 = iota
	moveRow
	moveCol
)

// accumulate fills the accumulated cost and the trace, both (rows+1) ×
// (cols+1), and reports whether the end is reachable.
func accumulate(cost []float64, rows, cols int, step Step) ([]int8, bool) {
	inf := math.Inf(1)
	w := cols + 1
	acc := make([]float64, (rows+1)*w)
	trace := make([]int8, (rows+1)*w)

	for i := range acc {
		acc[i] = inf
	}

	acc[0] = 0

	for j := 1; j <= cols; j++ {
		for i := 1; i <= rows; i++ {
			c0, c1, c2 := acc[(i-1)*w+j-1], acc[(i-1)*w+j], acc[i*w+j-1]
			if step == StepNoShare {
				c1 = inf
			}

			c, t := c2, moveCol

			switch {
			case c0 < c1 && c0 < c2:
				c, t = c0, moveDiag
			case c1 < c0 && c1 < c2:
				c, t = c1, moveRow
			}

			acc[i*w+j] = cost[(i-1)*cols+j-1] + c
			trace[i*w+j] = t
		}
	}

	return trace, !math.IsInf(acc[rows*w+cols], 1)
}

// backtrace follows the trace from (rows, cols) to (0, 0) and returns the
// first and last column of every row.
func backtrace(trace []int8, rows, cols int) (starts, ends []int) {
	w := cols + 1
	starts, ends = make([]int, rows), make([]int, rows)

	for i := range ends {
		ends[i] = -1
	}

	for i, j := rows, cols; i > 0 || j > 0; {
		if i > 0 && j > 0 {
			starts[i-1] = j - 1
			if ends[i-1] < 0 {
				ends[i-1] = j - 1
			}
		}

		t := trace[i*w+j]

		switch {
		case i == 0:
			t = moveCol
		case j == 0:
			t = moveRow
		}

		switch t {
		case moveDiag:
			i, j = i-1, j-1
		case moveRow:
			i--
		default:
			j--
		}
	}

	return starts, ends
}
