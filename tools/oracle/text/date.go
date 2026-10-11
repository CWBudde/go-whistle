//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

// dateCases are the corpus cases whose answers hang on the date: fact: a relative day ("tomorrow") and a weekday.
var dateCases = []string{"event_tomorrow", "weather_friday"}

// probe is one CLI or library run of the date probe: how it was run, how it ended and its answer.
type probe struct {
	Run    string          `json:"run"`
	Exit   int             `json:"exit"`
	Stderr string          `json:"stderr,omitempty"`
	Answer json.RawMessage `json:"answer,omitempty"`
}

// dateFixture is testdata/text/engine/date.json.
type dateFixture struct {
	header

	Variants []string                    `json:"variants"`
	Probes   map[string]map[string]probe `json:"probes"`
	NoSystem map[string]map[string]bool  `json:"same_as_no_system"`
	Fixed    map[string]map[string]bool  `json:"same_as_fixed_file"`
}

// dateVariant is one way to hand the engine (or not) the system facts; args builds the --system flags in the scratch
// directory, and lib, when set, runs needle_init with that system text instead of the CLI.
type dateVariant struct {
	name string
	run  string
	args func(o *oracle, fixed string) ([]string, error)
	lib  func(fixed string) string
}

var dateVariants = []dateVariant{
	{"fixed_file", "--system <file holding the case's system text>", func(o *oracle, fixed string) ([]string, error) {
		p, err := o.systemFile(fixed)

		return []string{flagSystem, p}, err
	}, nil},
	{"no_system", "no --system", func(*oracle, string) ([]string, error) { return nil, nil }, nil},
	{"empty_file", "--system <empty file>", func(o *oracle, _ string) ([]string, error) {
		p, err := o.systemFile("")

		return []string{flagSystem, p}, err
	}, nil},
	{"trailing_newline", "--system <file holding the system text and a newline>", func(o *oracle, fixed string) ([]string, error) {
		p, err := o.systemFile(fixed + "\n")

		return []string{flagSystem, p}, err
	}, nil},
	{"literal_text", "--system '<the system text itself, not a file>'", func(_ *oracle, fixed string) ([]string, error) {
		return []string{flagSystem, fixed}, nil
	}, nil},
	{"library_fixed", "needle_init(<the case's system text>, tools, NULL)", nil, func(fixed string) string { return fixed }},
	{"library_empty", "needle_init(\"\", tools, NULL)", nil, func(string) string { return "" }},
}

// cmdDate records how the engine answers the date-dependent cases with the corpus's date: fact, without any system
// text, with an empty one, with a trailing newline, with the text passed literally where the CLI wants a file, and
// through needle_init with and without the fact. Upstream's Python adds the local date itself (_with_date_fact);
// this shows whether the engine does.
func cmdDate(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	lib, err := openLibrary(cfg.engineDir, o.needle3Path())
	if err != nil {
		return err
	}

	fx := dateFixture{
		header: o.header("needle --model needle3.cact --tools <toolset> --threads 4 <variant> --prompt <query>"),
		Probes: map[string]map[string]probe{}, NoSystem: map[string]map[string]bool{},
		Fixed: map[string]map[string]bool{},
	}

	for _, v := range dateVariants {
		fx.Variants = append(fx.Variants, v.name+": "+v.run)
	}

	for _, id := range dateCases {
		k, err := o.corpus.find(id)
		if err != nil {
			return err
		}

		probes := map[string]probe{}

		for _, v := range dateVariants {
			p, err := o.dateProbe(ctx, lib, k, v)
			if err != nil {
				return fmt.Errorf("%s %s: %w", id, v.name, err)
			}

			probes[v.name] = p
			_, _ = fmt.Fprintf(log, "%-16s %-18s exit %d %s\n", id, v.name, p.Exit, p.Answer)
		}

		fx.Probes[id] = probes
		fx.NoSystem[id] = sameAs(probes, "no_system")
		fx.Fixed[id] = sameAs(probes, "fixed_file")
	}

	return writeJSON(cfg.outDir, "date.json", fx)
}

// sameAs reports for every other probe whether its answer equals the reference probe's.
func sameAs(probes map[string]probe, ref string) map[string]bool {
	out := map[string]bool{}

	for name, p := range probes {
		if name != ref {
			out[name] = p.Exit == probes[ref].Exit && string(p.Answer) == string(probes[ref].Answer)
		}
	}

	return out
}

// dateProbe runs one variant of case k.
func (o *oracle) dateProbe(ctx context.Context, lib *library, k textCase, v dateVariant) (probe, error) {
	fixed := o.corpus.system(k)

	if v.lib != nil {
		_, err := lib.session(v.lib(fixed), o.corpus.tools(k.Toolset))
		if err != nil {
			return probe{}, err
		}

		lib.reset()

		c, err := lib.completeText(k.Query, k.max())
		if err != nil {
			return probe{}, err
		}

		a, err := normaliseResponse(c.out)
		if err != nil {
			return probe{}, err
		}

		return probe{Run: v.run, Answer: compact(a)}, nil
	}

	sys, err := v.args(o, fixed)
	if err != nil {
		return probe{}, err
	}

	args := append([]string{"--model", o.needle3Path(), "--tools", o.toolsPath(k.Toolset), "--threads", cliThreads},
		sys...)

	res, err := runCLI(ctx, o.needle(), maxArgs(k, append(args, "--prompt", k.Query)...))
	if err != nil {
		return probe{}, err
	}

	p := probe{Run: v.run, Exit: res.Exit, Stderr: res.Stderr}

	if res.Exit == 0 {
		a, err := normaliseResponse(res.stdout)
		if err != nil {
			return probe{}, err
		}

		p.Answer = compact(a)
	}

	return p, nil
}

// find returns the corpus case with the given id.
func (c corpus) find(id string) (textCase, error) {
	for _, k := range c.Cases {
		if k.ID == id {
			return k, nil
		}
	}

	return textCase{}, fmt.Errorf("the corpus has no case %q", id)
}
