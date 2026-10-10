package bench

import (
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// The engine's published figures (PLAN.md 9.6): an M4 Pro, 5 beams.
const (
	PublishedMachine = "Apple M4 Pro"
	PublishedTTFTMS  = 11.1
	PublishedTPS     = 1319.0
)

// AcceptBoundPP is the Phase 9 Accept's bound on the packed WER minus the
// float WER, in percentage points.
const AcceptBoundPP = 0.1

const missing = "—"

// Render writes bench/RESULTS.md from the Go and the engine results;
// either may be nil when that side has not run.
func Render(w io.Writer, g *GoResults, e *EngineResults) error {
	var b strings.Builder

	b.WriteString("# Benchmarks: go-whistle against the Needle 3 engine\n\n" +
		"PLAN.md 9.6 and the Phase 9 **Accept**. This file is generated from `" + GoResultsPath + "` (`just bench`)\n" +
		"and `" + EngineResultsPath + "` (`just oracle-bench`); do not edit it by hand. Method and reading:\n" +
		"`docs/FINDINGS.md` F-040.\n\n")

	same := g != nil && e != nil && SameMachine(g.Run.Machine, e.Run.Machine)

	renderRuns(&b, g, e, same)
	renderAccept(&b, g)
	renderSpeed(&b, g, e, same)
	renderPublished(&b, g, e, same)
	renderMemory(&b, g, e)
	renderMethod(&b)

	_, err := io.WriteString(w, b.String())
	if err != nil {
		return fmt.Errorf("write report: %w", err)
	}

	return nil
}

func renderRuns(b *strings.Builder, g *GoResults, e *EngineResults, same bool) {
	var gr, er Run
	if g != nil {
		gr = g.Run
	}

	if e != nil {
		er = e.Run
	}

	t := newTable("run", "Go (`just bench`)", "engine (`just oracle-bench`)")
	t.left = true
	t.row("date", or(gr.Date), or(er.Date))
	t.row("commit", or(gr.Commit), or(er.Commit))
	t.row("CPU", or(gr.Machine.CPU), or(er.Machine.CPU))
	t.row("cores", cores(gr.Machine), cores(er.Machine))
	t.row("memory", gibString(gr.Machine.MemoryGiB), gibString(er.Machine.MemoryGiB))
	t.row("OS", or(gr.Machine.OS), or(er.Machine.OS))
	t.row("Go", or(gr.Machine.GoVersion), or(er.Machine.GoVersion))

	workersGo, workersEngine, engine, passes, passesEngine := missing, missing, missing, missing, missing
	if g != nil {
		workersGo = fmt.Sprintf("GOMAXPROCS %d", g.GOMAXPROCS)

		if m := g.mode("packed"); m != nil {
			passes = fmt.Sprintf("%d warm-up + %d timed; 30 s: %d + %d", g.WarmUp, m.Clips.Passes, g.WarmUp, m.Long.Passes)
		}
	}

	if e != nil {
		workersEngine = "CLI `--threads " + e.Threads + "`, library default (fast cores, at most 4)"
		engine = e.Version + " (" + short(e.Revision) + ")"
		passesEngine = fmt.Sprintf("%d warm-up + %d timed; 30 s: %d + %d", e.WarmUp, e.Library.Clips.Passes, e.WarmUp,
			e.Library.Long.Passes)
	}

	t.row("engine", missing, engine)
	t.row("workers", workersGo, workersEngine)
	t.row("passes per clip", passes, passesEngine)
	t.row("load average, start", or(gr.LoadAvgStart), or(er.LoadAvgStart))
	t.row("load average, end", or(gr.LoadAvgEnd), or(er.LoadAvgEnd))
	t.row("run time", seconds(gr.Seconds), seconds(er.Seconds))

	b.WriteString("## Runs\n\n")
	b.WriteString(t.String())
	b.WriteString("\n")

	if g != nil && e != nil && !same {
		b.WriteString(differentMachines + "\n\n")
	}
}

// differentMachines is the report's warning when the Go and the engine
// results were measured on different machines (SameMachine).
const differentMachines = "**The Go and the engine results come from different machines** (CPU, cores, OS or " +
	"architecture differ), so this report does not compare them: re-run `just bench` and `just oracle-bench` on " +
	"the same machine for the ratios."

func renderAccept(b *strings.Builder, g *GoResults) {
	b.WriteString("## WER: packed against float (Phase 9 Accept)\n\n")

	if g == nil || g.WER == nil {
		b.WriteString("Not measured: run `just bench` with the LibriSpeech audio (`just oracle-librispeech-audio`).\n\n")

		return
	}

	r := g.WER
	t := newTable("LibriSpeech test-clean subset", "Go packed", "Go float", "engine")
	t.row("errors / reference words", frac(r.PackedErrors, r.Words), frac(r.FloatErrors, r.Words),
		frac(r.EngineErrors, r.Words))
	t.row("WER", pct(r.Percent(r.PackedErrors)), pct(r.Percent(r.FloatErrors)), pct(r.Percent(r.EngineErrors)))
	t.row("text equal to the engine's", frac(r.PackedEqualsEngine, r.Utterances),
		frac(r.FloatEqualsEngine, r.Utterances), missing)
	t.row("real-time factor (Σ Transcribe / Σ audio)", rtf(r.PackedRTF), rtf(r.FloatRTF), missing)

	diff := r.Percent(r.PackedErrors) - r.Percent(r.FloatErrors)
	verdict := "holds"

	if math.Abs(diff) > AcceptBoundPP {
		verdict = "fails"
	}

	fmt.Fprintf(b, "%s\n\"The float path\" is `whistle.Load` with `WithFloatWeights()`; packed is `whistle.Load`'s "+
		"default on `whistle.cact` (F-037). %d utterances, %d reference words, %.1f s of audio, the manifest's "+
		"`wer_rule` on all three. The engine's transcripts are the committed `testdata/oracle/librispeech`.\n\n"+
		"- Packed − float: **%+.2f** percentage points; the Accept allows %.1f, so it **%s**.\n"+
		"- Packed and float transcripts equal (text and language): %s.\n"+
		"- Packed − engine: %+.2f percentage points.\n\n",
		t, r.Utterances, r.Words, r.AudioSeconds, diff, AcceptBoundPP, verdict,
		frac(r.PackedEqualsFloat, r.Utterances), r.Percent(r.PackedErrors)-r.Percent(r.EngineErrors))
}

// speedRows adds the rows of a speed table, one column per set; a nil set
// is a missing column.
func speedRows(t *table, sets [4]*Set) {
	add := func(label string, f func(*Set) string) {
		cells := make([]string, 0, 1+len(sets))
		cells = append(cells, label)

		for _, s := range sets {
			c := missing
			if s != nil && s.TTFT.N > 0 {
				c = f(s)
			}

			cells = append(cells, c)
		}

		t.row(cells...)
	}

	add("TTFT, median (ms)", func(s *Set) string { return ms(s.TTFT.Median) })
	add("TTFT, p90 (ms)", func(s *Set) string { return ms(s.TTFT.P90) })
	add("decode, median (tok/s)", func(s *Set) string { return tps(s.DecodeTPS.Median) })
	add("decode, p90 (tok/s)", func(s *Set) string { return tps(s.DecodeTPS.P90) })
	add("wall per transcription, median (ms)", func(s *Set) string { return ms(s.Wall.Median) })
	add("wall per transcription, p90 (ms)", func(s *Set) string { return ms(s.Wall.P90) })
	add("real-time factor, median over passes", func(s *Set) string { return rtf(s.RTF.Median) })
	add("transcriptions timed (clips × passes)", func(s *Set) string { return fmt.Sprintf("%d × %d", s.Clips, s.Passes) })
}

func renderSpeed(b *strings.Builder, g *GoResults, e *EngineResults, same bool) {
	var pk, fl *GoMode
	if g != nil {
		pk, fl = g.mode("packed"), g.mode("float")
	}

	for k, title := range []string{"the 32 fixture clips", "the 30 s clip (the first clip tiled)"} {
		var sets [4]*Set

		if pk != nil && fl != nil {
			sets[0], sets[1] = pick(&pk.Clips, &pk.Long, k), pick(&fl.Clips, &fl.Long, k)
		}

		if e != nil {
			sets[2], sets[3] = pick(&e.Library.Clips, &e.Library.Long, k), pick(&e.CLI.Clips, &e.CLI.Long, k)
		}

		t := newTable(title, "Go packed", "Go float", "engine library", "engine CLI (process)")
		speedRows(t, sets)

		if k == 0 {
			t.row("text and language equal to the engine's fixtures", equalCell(pk), equalCell(fl), missing,
				engineEqual(e))
		}

		fmt.Fprintf(b, "## Speed: %s\n\n%s\n", title, t)

		if g != nil && e != nil && !same {
			b.WriteString("The Go and the engine columns come from different machines; compare within a side only.\n\n")
		}

		if k == 0 && e != nil && e.Library.OutsideClock.N > 0 {
			fmt.Fprintf(b, "A `needle_transcribe` call takes its `ttft_ms` plus (n − 1) / `decode_tps` for the n tokens "+
				"it returns, plus a median %.2f ms (p90 %.2f ms) that its own clock does not explain.\n\n",
				e.Library.OutsideClock.Median, e.Library.OutsideClock.P90)
		}
	}

	t := newTable("model load (ms)", "Go packed", "Go float", "engine library")
	t.row("median", loadCell(pk, false), loadCell(fl, false), engineLoad(e, false))
	t.row("p90", loadCell(pk, true), loadCell(fl, true), engineLoad(e, true))

	fmt.Fprintf(b, "## Speed: model load\n\n%s\n`whistle.Load` maps `whistle.cact` and builds the model; "+
		"`needle_load` takes the file's bytes already read into memory. The CLI's load is inside its process wall "+
		"time.\n\n", t)
}

func renderPublished(b *strings.Builder, g *GoResults, e *EngineResults, same bool) {
	t := newTable("median of the 32 clips", "TTFT (ms)", "decode (tok/s)", "TTFT / engine here",
		"decode / engine here")
	t.row("engine, "+PublishedMachine+", 5 beams (published)", ms(PublishedTTFTMS), tps(PublishedTPS), missing, missing)

	// The engine here is the yardstick only when both sides ran on the same
	// machine; otherwise each row names its own machine and has no ratio.
	where := func(m Machine) string {
		if same {
			return "this machine"
		}

		return or(m.CPU)
	}

	var ref *Set

	if e != nil && e.Library.Clips.TTFT.N > 0 {
		engineRatio := missing

		if same {
			ref = &e.Library.Clips
			engineRatio = ratio(1)
		}

		t.row("engine library, "+where(e.Run.Machine), ms(e.Library.Clips.TTFT.Median),
			tps(e.Library.Clips.DecodeTPS.Median), engineRatio, engineRatio)
	}

	if g != nil {
		for _, m := range g.Modes {
			ttftRatio, tpsRatio := missing, missing
			if ref != nil {
				ttftRatio = ratio(m.Clips.TTFT.Median / ref.TTFT.Median)
				tpsRatio = ratio(m.Clips.DecodeTPS.Median / ref.DecodeTPS.Median)
			}

			t.row("Go "+m.Mode+", "+where(g.Run.Machine), ms(m.Clips.TTFT.Median), tps(m.Clips.DecodeTPS.Median),
				ttftRatio, tpsRatio)
		}
	}

	note := "The published figures come from another machine and clip set; the engine on the same machine is the " +
		"yardstick. A TTFT ratio above 1 is slower than the engine, a decode ratio below 1 too."
	if g != nil && e != nil && !same {
		note = "The published figures come from another machine and clip set. The Go and the engine rows come from " +
			"different machines too, so there are no ratios."
	}

	fmt.Fprintf(b, "## Against the published figures\n\n%s\n%s\n\n", t, note)
}

func renderMemory(b *strings.Builder, g *GoResults, e *EngineResults) {
	var pk, fl *GoMode
	if g != nil {
		pk, fl = g.mode("packed"), g.mode("float")
	}

	cli := func(v float64) string {
		if e == nil {
			return missing
		}

		return mib(v)
	}

	rss := func(m *GoMode, load bool) string {
		switch {
		case m == nil:
			return missing
		case load:
			return mib(m.MaxRSSLoadMiB)
		default:
			return mib(m.MaxRSSPeakMiB)
		}
	}

	var long, clips float64
	if e != nil {
		long, clips = e.CLI.MaxRSSLongMiB, e.CLI.MaxRSSClipsMiB
	}

	t := newTable("peak RSS (MiB, ru_maxrss of a fresh process)", "Go packed", "Go float", "engine CLI")
	t.row("after the model load", rss(pk, true), rss(fl, true), missing)
	t.row("one 30 s transcription, load included", rss(pk, false), rss(fl, false), cli(long))
	t.row("largest over the 32 clips, load included", missing, missing, cli(clips))

	fmt.Fprintf(b, "## Memory\n\n%s\nThe Go figures come from a fresh `whistle-bench` child per mode (`Load`, then "+
		"one `Transcribe` of the 30 s clip, as `just mem-probe`); the engine's from the CLI processes of the speed "+
		"runs.\n\n", t)
}

func renderMethod(b *strings.Builder) {
	b.WriteString("## What is measured\n\n" +
		"- **TTFT** is the engine's `ttft_ms` and `Result.TTFT`: from the PCM in hand to the first token after the\n" +
		"  language token. It covers the front end, the encoder, the cross-attention memory and the first decoder\n" +
		"  steps, not the process start, the model load or the WAV decoding (F-027, F-040).\n" +
		"- **Decode** is the engine's `decode_tps` and `Result.DecodeTPS`: the tokens after that one over their time,\n" +
		"  the beam search's steps at 5 beams.\n" +
		"- **Wall** is one transcription from PCM in memory to the result: `Transcribe` for Go and\n" +
		"  `needle_transcribe` for the engine library, both without word timestamps, the model loaded, after a\n" +
		"  warm-up. For the engine CLI it is the whole process: start-up, model load, WAV reading and the\n" +
		"  transcription.\n" +
		"- **Real-time factor** is the wall time over the audio duration, per pass over a clip set, median over the\n" +
		"  passes.\n" +
		"- **Like with like** is Go against the engine library: the same PCM, in-process, the model loaded. Go runs\n" +
		"  packed and float alternately in one process, clip by clip, so both see the same machine load.\n\n" +
		"## Reproduce\n\n" +
		"```bash\n" +
		"just bench          # the Go side: bench/results/go.json, then this file\n" +
		"just oracle-bench   # the engine side: bench/results/engine.json, then this file\n" +
		"go run ./bench/cmd/whistle-bench report   # re-render this file from both JSON files\n" +
		"```\n\n" +
		"`just bench` needs `models/whistle/whistle.cact` (`whistle download`) and, for the WER,\n" +
		"`models/corpus/librispeech` (`just oracle-librispeech-audio`); `just oracle-bench` needs `engine/<platform>`\n" +
		"(`just fetch-engine`).\n")
}

func (g *GoResults) mode(name string) *GoMode {
	for i := range g.Modes {
		if g.Modes[i].Mode == name {
			return &g.Modes[i]
		}
	}

	return nil
}

func pick(clips, long *Set, k int) *Set {
	if k == 0 {
		return clips
	}

	return long
}

func loadCell(m *GoMode, p90 bool) string {
	switch {
	case m == nil || m.Load.N == 0:
		return missing
	case p90:
		return ms(m.Load.P90)
	default:
		return ms(m.Load.Median)
	}
}

func engineLoad(e *EngineResults, p90 bool) string {
	switch {
	case e == nil || e.Library.Load.N == 0:
		return missing
	case p90:
		return ms(e.Library.Load.P90)
	default:
		return ms(e.Library.Load.Median)
	}
}

func equalCell(m *GoMode) string {
	if m == nil {
		return missing
	}

	return frac(m.EqualEngine, m.Clips.Clips)
}

func engineEqual(e *EngineResults) string {
	if e == nil {
		return missing
	}

	return frac(e.CLI.EqualFixtures, e.CLI.Clips.Clips)
}

func or(s string) string {
	if s == "" {
		return missing
	}

	return s
}

func cores(m Machine) string {
	switch {
	case m.Cores == 0:
		return missing
	case m.CoreTypes == "":
		return strconv.Itoa(m.Cores)
	default:
		return fmt.Sprintf("%d (%s)", m.Cores, m.CoreTypes)
	}
}

func short(rev string) string {
	if len(rev) > 10 {
		return rev[:10]
	}

	return rev
}

func gibString(v float64) string {
	if v == 0 {
		return missing
	}

	return fmt.Sprintf("%.0f GiB", v)
}

func seconds(v float64) string {
	if v == 0 {
		return missing
	}

	return fmt.Sprintf("%.0f s", v)
}

func ms(v float64) string    { return fmt.Sprintf("%.1f", v) }
func tps(v float64) string   { return fmt.Sprintf("%.0f", v) }
func rtf(v float64) string   { return fmt.Sprintf("%.4f", v) }
func mib(v float64) string   { return fmt.Sprintf("%.1f", v) }
func pct(v float64) string   { return fmt.Sprintf("%.2f %%", v) }
func ratio(v float64) string { return fmt.Sprintf("%.2f×", v) }

func frac(n, of int) string { return fmt.Sprintf("%d/%d", n, of) }
