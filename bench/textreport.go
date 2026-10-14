package bench

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// RenderText writes the text bench's report (PLAN.md 11.5's Accept,
// docs/FINDINGS.md F-061) from the Go and the engine results, read from
// p.Go and p.Engine; either may be nil when that side has not run.
func RenderText(w io.Writer, g *TextGoResults, e *TextEngineResults, p Paths) error {
	var b strings.Builder

	b.WriteString("# Benchmarks: the Needle 3 text model against the engine\n\n" +
		"PLAN.md 11.5's **Accept**: speed and memory against the engine, and the packed weights. This file is\n" +
		"generated from `" + p.Go + "` (`just bench-text`) and `" + p.Engine + "`\n" +
		"(`just oracle-text bench`); do not edit it by hand. Method and reading: `docs/FINDINGS.md` F-061.\n\n")

	// The sides are compared only when they measured the same model, cases
	// and recorded answers on the same machine.
	same := g != nil && e != nil && SameMachine(g.Run.Machine, e.Run.Machine) && SameTextInputs(g, e)

	renderTextRuns(&b, g, e, same)
	renderTextSpeed(&b, g, e, same)
	renderTextLoad(&b, g, e)
	renderTextMemory(&b, g, e)
	renderTextMethod(&b, p, g)

	_, err := io.WriteString(w, b.String())
	if err != nil {
		return fmt.Errorf("write report: %w", err)
	}

	return nil
}

// notComparableText is the text report's warning when the Go and the
// engine results come from different machines or inputs.
const notComparableText = "**The Go and the engine results are not comparable**: they come from different " +
	"machines (CPU, cores, OS or architecture) or answered different inputs (model, cases or recorded answers), " +
	"so this report does not compare them. Re-run `just bench-text` and `just oracle-text bench` on the same " +
	"machine and inputs for the ratios."

func renderTextRuns(b *strings.Builder, g *TextGoResults, e *TextEngineResults, same bool) {
	var gr, er Run
	if g != nil {
		gr = g.Run
	}

	if e != nil {
		er = e.Run
	}

	t := newTable("run", "Go (`just bench-text`)", "engine (`just oracle-text bench`)")
	t.left = true
	t.row("date", or(gr.Date), or(er.Date))
	t.row("commit", or(gr.Commit), or(er.Commit))
	t.row("CPU", or(gr.Machine.CPU), or(er.Machine.CPU))
	t.row("cores", cores(gr.Machine), cores(er.Machine))

	if gr.Machine.Governor != "" || er.Machine.Governor != "" {
		t.row("governor", or(gr.Machine.Governor), or(er.Machine.Governor))
	}

	t.row("memory", gibString(gr.Machine.MemoryGiB), gibString(er.Machine.MemoryGiB))
	t.row("OS", or(gr.Machine.OS), or(er.Machine.OS))
	t.row("Go", or(gr.Machine.GoVersion), or(er.Machine.GoVersion))
	t.row("needle3.cact SHA-256", or(short(gr.Inputs.Model)), or(short(er.Inputs.Model)))

	st := missing
	if g != nil {
		st = or(short(g.Safetensors))
	}

	t.row("needle3.safetensors SHA-256", st, missing)
	t.row("cases.json SHA-256", or(short(gr.Inputs.Clips)), or(short(er.Inputs.Clips)))

	textRunRows(t, g, e)
	t.row("load average, start", or(gr.LoadAvgStart), or(er.LoadAvgStart))
	t.row("load average, end", or(gr.LoadAvgEnd), or(er.LoadAvgEnd))
	t.row("run time", seconds(gr.Seconds), seconds(er.Seconds))

	b.WriteString("## Runs\n\n")
	b.WriteString(t.String())
	b.WriteString("\n")

	if g != nil && e != nil && !same {
		b.WriteString(notComparableText + "\n\n")
	}
}

// textRunRows adds the runs table's engine, workers and passes rows.
func textRunRows(t *table, g *TextGoResults, e *TextEngineResults) {
	workersGo, workersEngine, engine, passes, passesEngine := missing, missing, missing, missing, missing
	if g != nil {
		workersGo = fmt.Sprintf("GOMAXPROCS %d", g.GOMAXPROCS)

		if len(g.Modes) > 0 {
			passes = fmt.Sprintf("%d warm-up + %d timed", g.WarmUp, g.Modes[0].Answers.Passes)
		}
	}

	if e != nil {
		workersEngine = "CLI `--threads " + e.Threads + "`, library default (fast cores, at most 4)"
		engine = e.Version + " (" + short(e.Revision) + ")"
		passesEngine = fmt.Sprintf("%d warm-up + %d timed", e.WarmUp, e.CLI.Answers.Passes)
	}

	t.row("engine", missing, engine)
	t.row("workers", workersGo, workersEngine)
	t.row("passes over the cases", passes, passesEngine)
}

// textColumns are the speed table's columns: the Go modes in TextModes'
// order, then the engine library cold and with the static prefix cached,
// and the engine CLI.
func textColumns(g *TextGoResults, e *TextEngineResults) ([]string, []*TextSet) {
	heads := make([]string, 0, len(TextModes)+3)
	sets := make([]*TextSet, 0, len(TextModes)+3)

	for _, m := range TextModes {
		heads = append(heads, "Go "+m.Label)

		var s *TextSet

		if g != nil {
			if gm := g.mode(m.Name); gm != nil {
				s = &gm.Answers
			}
		}

		sets = append(sets, s)
	}

	heads = append(heads, "engine library", "engine library, prefix cached", "engine CLI (process)")

	if e != nil {
		sets = append(sets, &e.Library.Answers, &e.Library.Cached, &e.CLI.Answers)
	} else {
		sets = append(sets, nil, nil, nil)
	}

	return heads, sets
}

// caseCount is the number of cases the runs answered and a space, for the
// labels: the Go side's, else the engine's; without either it is empty.
func caseCount(g *TextGoResults, e *TextEngineResults) string {
	_, sets := textColumns(g, e)

	for _, s := range sets {
		if s != nil && s.Cases > 0 {
			return strconv.Itoa(s.Cases) + " "
		}
	}

	return ""
}

// summaryOf formats one field of a TextSet: its median, or its p90.
func summaryOf(f func(*TextSet) Summary, format func(float64) string, p90 bool) func(*TextSet) string {
	return func(s *TextSet) string {
		v := f(s)

		switch {
		case v.N == 0:
			return missing
		case p90:
			return format(v.P90)
		default:
			return format(v.Median)
		}
	}
}

func renderTextSpeed(b *strings.Builder, g *TextGoResults, e *TextEngineResults, same bool) {
	heads, sets := textColumns(g, e)
	title := "the " + caseCount(g, e) + "first turns"
	t := newTable(append([]string{title}, heads...)...)

	add := func(label string, f func(*TextSet) string) {
		cells := make([]string, 0, 1+len(sets))
		cells = append(cells, label)

		for _, s := range sets {
			c := missing
			if s != nil && s.Wall.N > 0 {
				c = f(s)
			}

			cells = append(cells, c)
		}

		t.row(cells...)
	}

	ttft := func(s *TextSet) Summary { return s.TTFT }
	decode := func(s *TextSet) Summary { return s.DecodeTPS }
	wall := func(s *TextSet) Summary { return s.Wall }
	count := func(v float64) string { return strconv.Itoa(int(v)) }

	add("prefill tokens, median", summaryOf(func(s *TextSet) Summary { return s.PrefillTokens }, count, false))
	add("generated tokens, median", summaryOf(func(s *TextSet) Summary { return s.Generated }, count, false))
	add("TTFT, median (ms)", summaryOf(ttft, ms, false))
	add("TTFT, p90 (ms)", summaryOf(ttft, ms, true))
	add("prefill, median (tok/s)", summaryOf(func(s *TextSet) Summary { return s.PrefillTPS }, tps, false))
	add("engine's `prefill_tps` (the turn), median", summaryOf(func(s *TextSet) Summary { return s.ReportedPrefillTPS },
		tps, false))
	add("decode, median (tok/s)", summaryOf(decode, tps, false))
	add("decode, p90 (tok/s)", summaryOf(decode, tps, true))
	add("answer, median (ms)", summaryOf(wall, ms, false))
	add("answer, p90 (ms)", summaryOf(wall, ms, true))
	add("answers timed (cases × passes)", func(s *TextSet) string { return fmt.Sprintf("%d × %d", s.Cases, s.Passes) })

	t.row(append([]string{"`function_calls` equal to the recording"}, callsCells(g, e, len(sets))...)...)

	fmt.Fprintf(b, "## Speed: %s\n\n%s\n", title, t)

	if g != nil && e != nil && !same {
		b.WriteString("The Go and the engine columns are not comparable (machine or inputs); compare within a side " +
			"only.\n\n")
	}

	if same {
		renderTextRatios(b, g, e)
	}
}

// callsCells are the speed table's function_calls row: each side's
// warm-up answers against the engine's recorded ones.
func callsCells(g *TextGoResults, e *TextEngineResults, n int) []string {
	cells := make([]string, 0, n)

	for _, m := range TextModes {
		c := missing

		if g != nil {
			if gm := g.mode(m.Name); gm != nil && gm.Answers.Cases > 0 {
				c = frac(gm.FunctionCallsEqual, gm.Answers.Cases)
			}
		}

		cells = append(cells, c)
	}

	if e == nil {
		return append(cells, missing, missing, missing)
	}

	return append(cells, frac(e.Library.FunctionCallsEqual, e.Library.Answers.Cases),
		frac(e.Library.CachedFunctionCallsEqual, e.Library.Cached.Cases),
		frac(e.CLI.FunctionCallsEqual, e.CLI.Answers.Cases))
}

// renderTextRatios divides each Go mode's medians by the engine library's.
func renderTextRatios(b *strings.Builder, g *TextGoResults, e *TextEngineResults) {
	ref := &e.Library.Answers
	t := newTable("median, Go / engine library", "TTFT", "prefill", "decode", "answer")

	for _, m := range TextModes {
		gm := g.mode(m.Name)
		if gm == nil || gm.Answers.Wall.N == 0 {
			continue
		}

		s := &gm.Answers
		t.row("Go "+m.Label, quotient(s.TTFT, ref.TTFT), quotient(s.PrefillTPS, ref.PrefillTPS),
			quotient(s.DecodeTPS, ref.DecodeTPS), quotient(s.Wall, ref.Wall))
	}

	fmt.Fprintf(b, "%s\nA TTFT or answer ratio above 1 is slower than the engine, a prefill or decode ratio below 1 "+
		"too.\n\n", t)
}

// quotient is the ratio of two medians, missing without either.
func quotient(a, b Summary) string {
	if a.N == 0 || b.N == 0 || b.Median == 0 {
		return missing
	}

	return ratio(a.Median / b.Median)
}

func renderTextLoad(b *strings.Builder, g *TextGoResults, e *TextEngineResults) {
	heads := make([]string, 0, len(TextModes)+2)
	heads = append(heads, "model load (ms)")

	for _, m := range TextModes {
		heads = append(heads, "Go "+m.Label)
	}

	heads = append(heads, "engine library")
	t := newTable(heads...)

	for _, p90 := range []bool{false, true} {
		cells := make([]string, 0, len(TextModes)+2)
		cells = append(cells, "median")

		if p90 {
			cells[0] = "p90"
		}

		for _, m := range TextModes {
			var s Summary

			if g != nil {
				if gm := g.mode(m.Name); gm != nil {
					s = gm.Load
				}
			}

			cells = append(cells, summaryCell(s, p90))
		}

		var s Summary
		if e != nil {
			s = e.Library.Load
		}

		t.row(append(cells, summaryCell(s, p90))...)
	}

	fmt.Fprintf(b, "## Speed: model load\n\n%s\nThe Go load reads the file and builds the tokenizer, the model and "+
		"the decoder; `needle_load` takes the file's bytes already read into memory. The CLI's load is inside its "+
		"process time.\n\n", t)
}

func summaryCell(s Summary, p90 bool) string {
	switch {
	case s.N == 0:
		return missing
	case p90:
		return ms(s.P90)
	default:
		return ms(s.Median)
	}
}

func renderTextMemory(b *strings.Builder, g *TextGoResults, e *TextEngineResults) {
	heads := make([]string, 0, len(TextModes)+2)
	heads = append(heads, "memory (MiB)")

	for _, m := range TextModes {
		heads = append(heads, "Go "+m.Label)
	}

	heads = append(heads, "engine CLI")
	t := newTable(heads...)

	row := func(label string, goCell func(*TextGoMode) string, engine string) {
		cells := make([]string, 0, len(TextModes)+2)
		cells = append(cells, label)

		for _, m := range TextModes {
			c := missing

			if g != nil {
				if gm := g.mode(m.Name); gm != nil {
					c = goCell(gm)
				}
			}

			cells = append(cells, c)
		}

		t.row(append(cells, engine)...)
	}

	engineRSS, engineRAM, engineRAMMax := missing, missing, missing
	if e != nil {
		engineRSS = rssMiB(e.CLI.MaxRSS.Max)
		engineRAM, engineRAMMax = rssMiB(e.CLI.PeakRAM.Median), rssMiB(e.CLI.PeakRAM.Max)
	}

	none := func(*TextGoMode) string { return missing }

	row("peak RSS after the load", func(m *TextGoMode) string { return rssMiB(m.MaxRSSLoadMiB) }, missing)
	row("peak RSS after the "+caseCount(g, e)+"answers, load included", func(m *TextGoMode) string {
		return rssMiB(m.MaxRSSPeakMiB)
	}, engineRSS)
	row("Go heap after the load", func(m *TextGoMode) string { return rssMiB(m.HeapLoadMiB) }, missing)
	row("engine's `peak_ram_mb`, median", none, engineRAM)
	row("engine's `peak_ram_mb`, largest", none, engineRAMMax)

	weights := float64(PackedWeightsBytes) / (1 << 20)

	verdict := "under"
	if weights >= PackedWeightsBoundMiB {
		verdict = "**not** under"
	}

	fmt.Fprintf(b, "## Memory\n\n%s\nThe Go figures come from a fresh `needle-bench` child per mode (the load, "+
		"then one pass over the cases); peak RSS is `ru_maxrss`, a high-water mark. The engine's come from the CLI "+
		"processes of the speed runs: the largest `ru_maxrss` over them, and the `peak_ram_mb` it reports.\n\n"+
		"**Packed weights:** %.3f MiB (%s bytes), as `TestPackedMemory` (`internal/text`) counts the packed "+
		"`needle3.cact` on exact kernels (F-059): %s the Accept's %d MiB, its \"40 MB\" read as MiB.\n\n",
		t, weights, thousands(PackedWeightsBytes), verdict, PackedWeightsBoundMiB)
}

// thousands is n with commas between groups of three digits.
func thousands(n int) string {
	s := strconv.Itoa(n)

	var b strings.Builder

	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}

		b.WriteRune(c)
	}

	return b.String()
}

func renderTextMethod(b *strings.Builder, p Paths, g *TextGoResults) {
	b.WriteString("## What is measured\n\n" +
		"- **The cases** are the first turns of `testdata/text/engine/cases.json`: their tools, the corpus's system\n" +
		"  text with its date fact, and the `--max` a case sets (512 otherwise), as the corpus was recorded.\n" +
		"- **TTFT** runs from the turn's tokens in hand to the first generated token, the static prefix's prefill\n" +
		"  included. Go times the first `Extend` over the static prefix, the turn and the forced `<think>\\n`, and\n" +
		"  the pick of the next token. The engine reports no TTFT for text: the library's column is\n" +
		"  `needle_init`, which prefills the static prefix, and the `needle_complete` call less (n − 1) /\n" +
		"  `decode_tps`, an upper bound that also holds the confidence and the envelope. The CLI's process time\n" +
		"  cannot be split.\n" +
		"- **Prefix cached**: libneedle3 keeps the static prefix's prefill while `needle_init` gets the same system\n" +
		"  text and tools. The library column answers every case cold (another prefix answered before it), as the\n" +
		"  CLI and Go do; the cached column answers it again at once.\n" +
		"- **Prefill** is the prefill tokens over the TTFT on both sides. The engine's own `prefill_tps` counts the\n" +
		"  turn's tokens only, the static prefix being prefilled by `needle_init`.\n" +
		"- **Decode** is the tokens after the first over their steps: Go replays its answer one `Extend` per\n" +
		"  token; the engine reports `decode_tps`.\n" +
		"- **Answer** is one turn. For Go: the tools' normalisation, the prompt, the decode under the grammar, the\n" +
		"  confidence and the envelope, the model loaded. For the engine library: `needle_init` and\n" +
		"  `needle_complete`. For the engine CLI: the whole process, start-up and model load included.\n" +
		"- **Like with like** is Go against the engine library: in-process, the model loaded. Go runs its four\n" +
		"  modes in one process, alternating which goes first per case and pass, so all see the same machine load.\n" +
		"- **Modes**: safetensors float32 is `needle3.safetensors`; packed exact is `needle3.cact`'s CQ records\n" +
		"  packed on exact kernels (F-059); packed + int8 + window adds the engine's numerics (F-060); packed LUT\n" +
		"  runs the lut kernel (`OpenOptions.LUT`). The bench changes no numerics.\n\n" +
		"## Reproduce\n\n" + reproduceText(p, g) +
		"`just bench-text` needs `models/needle3` (`just fetch-needle3`); `just oracle-text bench` needs\n" +
		"`engine/<platform>` (`just fetch-engine`).\n")
}

// reproduceText is the Reproduce section's commands, with the flags that
// select p where it is not DefaultTextPaths, and -procs where the Go side
// ran with fewer workers than the machine's cores.
func reproduceText(p Paths, g *TextGoResults) string {
	procs := ""
	if g != nil && g.GOMAXPROCS > 0 && g.GOMAXPROCS != g.Run.Machine.Cores {
		procs = " -procs " + strconv.Itoa(g.GOMAXPROCS)
	}

	if p == DefaultTextPaths && procs == "" {
		return "```bash\n" +
			"just bench-text          # the Go side: bench/results/text-go.json, then this file\n" +
			"just oracle-text bench   # the engine side: bench/results/text-engine.json, then this file\n" +
			"go run ./bench/cmd/needle-bench report   # re-render this file from both JSON files\n" +
			"```\n\n"
	}

	flags := " -go " + p.Go + " -engine " + p.Engine + " -report " + p.Report

	return "```bash\n" +
		"# the Go side: " + p.Go + ", then this file\n" +
		"just bench-text" + procs + flags + "\n" +
		"# the engine side: " + p.Engine + ", then this file\n" +
		"just oracle-text bench -out " + p.Engine + " -go " + p.Go + " -report " + p.Report + "\n" +
		"# re-render this file from both JSON files\n" +
		"go run ./bench/cmd/needle-bench report" + flags + "\n" +
		"```\n\n"
}
