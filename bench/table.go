package bench

import (
	"strings"
	"unicode/utf8"
)

// table is a markdown table laid out the way prettier lays it out, so the
// generated RESULTS.md passes `treefmt --fail-on-change`: every cell padded
// to its column's width (at least 3), the first column left-aligned, the
// others right-aligned unless left is set.
type table struct {
	head []string
	rows [][]string
	left bool // every column left-aligned
}

func newTable(head ...string) *table { return &table{head: head} }

func (t *table) String() string {
	width := make([]int, len(t.head))
	for _, r := range append([][]string{t.head}, t.rows...) {
		for i, c := range r {
			width[i] = max(width[i], utf8.RuneCountInString(c), 3)
		}
	}

	var b strings.Builder

	line := func(cells []string) {
		for i := range width {
			c := ""
			if i < len(cells) {
				c = cells[i]
			}

			pad := strings.Repeat(" ", width[i]-utf8.RuneCountInString(c))
			if i == 0 || t.left {
				c += pad
			} else {
				c = pad + c
			}

			b.WriteString("| ")
			b.WriteString(c)
			b.WriteString(" ")
		}

		b.WriteString("|\n")
	}

	line(t.head)

	for i, w := range width {
		sep := strings.Repeat("-", w)
		if i > 0 && !t.left {
			sep = sep[:w-1] + ":"
		}

		b.WriteString("| ")
		b.WriteString(sep)
		b.WriteString(" ")
	}

	b.WriteString("|\n")

	for _, r := range t.rows {
		line(r)
	}

	return b.String()
}

func (t *table) row(cells ...string) { t.rows = append(t.rows, cells) }
