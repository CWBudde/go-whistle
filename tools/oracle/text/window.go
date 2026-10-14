//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// windowProbe is the engine's answer to one conversation that places the fact its answer needs far behind the
// answer: what it covers, the filler sentences between the fact and the answer, the tools JSON as sent, the turns
// and the answer to each, without its timings.
type windowProbe struct {
	ID      string            `json:"id"`
	Covers  string            `json:"covers"`
	Filler  int               `json:"filler"`
	Tools   string            `json:"tools"`
	Turns   []string          `json:"turns"`
	Answers []json.RawMessage `json:"answers"`
}

// windowFixture is testdata/text/engine/window.json.
type windowFixture struct {
	header

	System string        `json:"system"`
	Probes []windowProbe `json:"probes"`
}

// windowSentences are the neutral filler of the window probes: no numbers, times, names, negation cues or words a
// tool of the assistant set answers to. Each is about 13 tokens.
var windowSentences = []string{
	"The garden behind the house was quiet in the early light.",
	"A few birds moved between the branches of the old apple tree.",
	"The neighbours were still asleep, and the street looked empty.",
	"Someone had left a blue bicycle leaning against the fence.",
	"The air smelled of wet grass and freshly turned soil.",
	"Swallows circled high above the rooftops to the east.",
	"On the windowsill, a row of small pots held herbs and flowers.",
	"The cat watched the birds from the porch with great patience.",
	"Further down the road, a bakery was opening its shutters.",
	"The church bell rang, and the sound carried over the roofs.",
	"A delivery van stopped briefly and drove on toward the market.",
	"Chalk drawings still covered the pavement near the school.",
	"The river ran high after the long autumn.",
	"Along the path, the hedges had grown thick and green.",
	"An elderly man walked his dog along the riverbank.",
	"The library across the square had put out a sign for its book sale.",
	"Leaves gathered in small heaps by the side of the road.",
	"Somewhere a door creaked open and closed again.",
	"The café on the corner was setting out chairs and tables.",
	"A pair of ducks paddled along the edge of the pond.",
	"Light slowly reached the top of the tall chestnut trees.",
	"The postman greeted everyone he passed with a cheerful wave.",
	"In the distance, a train crossed the bridge over the valley.",
	"The smell of fresh bread drifted from the house next door.",
}

// windowFiller is n filler sentences, cycling through windowSentences.
func windowFiller(n int) string {
	s := make([]string, n)
	for i := range s {
		s[i] = windowSentences[i%len(windowSentences)]
	}

	return strings.Join(s, " ")
}

// windowSpec is one probe to record; a probe with more than one turn runs over `needle --serve`.
type windowSpec struct {
	id, covers string
	filler     int
	turns      []string
}

// windowFillers are the filler lengths in sentences: about 90 tokens (inside a window of 256), 290, 520 and 1100
// (past the local layers' band of 1024).
var windowFillers = []int{6, 22, 40, 85}

// windowProbes are the recorded probes, over the corpus's assistant tools: requests whose fact comes first and a
// filler after it, the same with the fact last, and conversations with the fact in an earlier turn (the first
// prompt) or at the start of a later long turn. A KV window of 256 positions without sinks, with the static prefix
// as a sink, or with the first prompt as a sink cuts different facts from view, and a window of 256 on the global
// layers only cuts the facts past the local band (docs/FINDINGS.md F-060).
func windowProbes() []windowSpec {
	const (
		timer    = "Set a timer for 25 minutes for the bread."
		message  = "Text Anna that dinner is ready at eight."
		weather  = "What's the weather in Lisbon today?"
		started  = `{"status": "started"}`
		forecast = `{"forecast": "sunny", "high_c": 24}`
	)

	ps := make([]windowSpec, 0, 2*len(windowFillers)+6)

	for _, n := range windowFillers {
		f := windowFiller(n)
		ps = append(ps,
			windowSpec{
				fmt.Sprintf("timer_first_%d", n), "the timer's length and label first, then the filler", n,
				[]string{timer + " " + f + " That is all for now."},
			},
			windowSpec{
				fmt.Sprintf("message_first_%d", n), "the message's recipient and text first, then the filler", n,
				[]string{message + " " + f + " Thanks."},
			},
		)
	}

	for _, n := range []int{22, 85} {
		ps = append(ps, windowSpec{
			fmt.Sprintf("timer_last_%d", n), "the filler, then the timer's length and label (the control)", n,
			[]string{windowFiller(n) + " " + timer},
		})
	}

	for _, n := range []int{22, 40} {
		f := windowFiller(n)
		ps = append(ps,
			windowSpec{
				fmt.Sprintf("earlier_turn_%d", n), "the timer's length in the first turn, the filler in the third", n,
				[]string{timer, started, f + " Now set a second timer of the same length for the pasta."},
			},
			windowSpec{
				fmt.Sprintf("later_turn_%d", n), "the timer's length at the start of a long third turn", n,
				[]string{weather, forecast, timer + " " + f + " That is all for now."},
			},
		)
	}

	return ps
}

// cmdWindow records the CLI's answers to the window probes, to set the port's hypotheses about the engine's KV
// window of 256 positions (needle3.cact's kv_window) against.
func cmdWindow(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	fx := windowFixture{
		header: o.header("needle --model needle3.cact --tools <tools> --system <system file> --threads 4 " +
			"--prompt <query>; with more than one turn --serve, POST /complete per turn"),
		System: o.corpus.System,
	}

	tools := o.corpus.tools(assistantSet)

	for _, p := range windowProbes() {
		answers, err := o.validationAnswers(ctx, "window-"+p.id, tools, o.corpus.System, p.turns)
		if err != nil {
			return fmt.Errorf("%s: %w", p.id, err)
		}

		rec := windowProbe{ID: p.id, Covers: p.covers, Filler: p.filler, Tools: tools, Turns: p.turns}

		for _, a := range answers {
			rec.Answers = append(rec.Answers, compact(a))
		}

		fx.Probes = append(fx.Probes, rec)
		_, _ = fmt.Fprintf(log, "%-20s %s\n", p.id, summary(answers[len(answers)-1]))
	}

	return writeJSON(cfg.outDir, "window.json", fx)
}
