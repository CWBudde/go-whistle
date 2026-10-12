// Package toolcall is the Needle 3 text model's tool-calling interface
// (PLAN.md 11.3, 11.4): the prompt as upstream's finetune.render_example
// renders it, the static prefix the engine's needle_init counts, and the
// date fact of the Python client (docs/FINDINGS.md F-048).
package toolcall

import (
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/cwbudde/go-whistle/internal/spm"
)

// The chat markers of upstream's needle/model/tokenizer.py, user-defined
// pieces of the tokenizer.
const (
	imStart  = "<|im_start|>"
	imEnd    = "<|im_end|>"
	toolsBeg = "<tools>"
	toolsEnd = "</tools>"
)

// Render is finetune.render_example's prompt for system facts, tools (the
// JSON text ToolsJSON returns) and query:
//
//	<|im_start|>system\n{system}<|im_end|>\n        (only with system facts)
//	<|im_start|>user\n<tools>{tools}</tools>\n{query}<|im_end|>\n
//	<|im_start|>assistant\n
//
// system is stripped of white space as Python's str.strip does; the query
// is used as is.
func Render(system, tools, query string) string {
	return staticText(system, tools) + "\n" + query + imEnd + "\n" + imStart + "assistant\n"
}

// staticText is the part of Render that does not depend on the query, up
// to </tools>.
func staticText(system, tools string) string {
	var prefix string
	if s := strings.TrimFunc(system, pyIsSpace); s != "" {
		prefix = imStart + "system\n" + s + imEnd + "\n"
	}

	return prefix + imStart + "user\n" + toolsBeg + tools + toolsEnd
}

// Encode is the model input for a prompt, as upstream's run.py builds it:
// [BOS] and sentencepiece's ids of prompt.
func Encode(p *spm.Processor, prompt string) []int {
	return append([]int{p.BOSID()}, p.Encode(prompt)...)
}

// StaticPrefix is the start of Encode(p, Render(system, tools, query))
// that every query shares: [BOS] and the rendering up to </tools>. Its
// length is needle_init's count on every case of the engine corpus; so is
// that of the rendering up to the query without [BOS] (docs/FINDINGS.md
// F-048).
func StaticPrefix(p *spm.Processor, system, tools string) []int {
	return Encode(p, staticText(system, tools))
}

// pyIsSpace is Python's str.isspace for one character: Unicode white space
// and the separators U+001C–U+001F, which unicode.IsSpace leaves out.
func pyIsSpace(r rune) bool { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }

var (
	// dateKey is upstream's _DATE_KEY, (?<![A-Za-z])date:, without the
	// look-behind.
	dateKey = regexp.MustCompile(`(?:^|[^A-Za-z])date:`)
	// isoStamp is _ISO_STAMP, \d{4}-\d{2}-\d{2}; Python's \d is any
	// Unicode decimal digit.
	isoStamp = regexp.MustCompile(`\p{Nd}{4}-\p{Nd}{2}-\p{Nd}{2}`)
)

// WithDateFact is the Python client's _with_date_fact at the time now: the
// system facts unchanged if they hold a date: key or an ISO date or are a
// JSON object, else "date: YYYY-MM-DD Day HH:MM" before them, joined by
// "; ". Neither the engine's CLI nor its library adds a date
// (docs/FINDINGS.md F-047); callers that want the client's behaviour pass
// the system facts through this.
func WithDateFact(system string, now time.Time) string {
	if dateKey.MatchString(system) || isoStamp.MatchString(system) {
		return system
	}

	fact := now.Format("date: 2006-01-02 Mon 15:04")

	if strings.TrimFunc(system, pyIsSpace) == "" {
		return fact
	}

	if strings.HasPrefix(strings.TrimLeftFunc(system, pyIsSpace), "{") {
		return system
	}

	return fact + "; " + system
}
