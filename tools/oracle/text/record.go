//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strings"
)

// Fixture file names in testdata/text/engine.
const (
	responsesFile = "responses.json"
	libraryFile   = "library.json"
	manifestFile  = "manifest.json"
)

// fixtureFiles lists every fixture file and the subcommand that writes it, for manifest.json.
var fixtureFiles = []string{
	"cases.json: the case corpus (hand-written)",
	responsesFile + ": cases, the CLI's answers per case and turn",
	libraryFile + ": cases, libneedle3's needle_init and needle_complete results",
	manifestFile + ": cases, this file",
	"date.json: date, answers with and without the date: fact",
	"depth.json: depth, the --depth ladder",
	"audio.json: audio, answers to speech (needle3.cact + whistle.cact)",
	"embed.json, embed.safetensors: embed, needle_embed of short texts",
	"retrieval.json: retrieval, needle_init past 5 tools and up to the context budget, the tool index",
	"normalise.json: normalise, needle_init on OpenAI-style, camelCase and colliding tools, answers naming the caller's tools",
	"turns.json: turns, conversations over needle --serve with --fail-input-overflow probes",
}

// textManifest is testdata/text/engine/manifest.json: what produced responses.json, what the checks found, and the
// speech model the audio subcommand loads beside the text model.
type textManifest struct {
	header

	Speech    modelPin       `json:"speech_model"`
	Files     []string       `json:"files"`
	Cases     int            `json:"cases"`
	Turns     int            `json:"turns"`
	Responses string         `json:"responses"`
	Dropped   []string       `json:"dropped_fields"`
	Types     map[string]int `json:"types"`
	Calls     map[string]int `json:"function_calls"`
	MinConf   json.Number    `json:"min_confidence"`
	MaxConf   json.Number    `json:"max_confidence"`
	Threads   []sameCheck    `json:"threads"`
	Library   sameCheck      `json:"library"`
	Serve     sameCheck      `json:"serve_first_turn"`
}

// sameCheck is the outcome of answering every case a second way: which cases gave other JSON and, per top-level
// key, in how many of them that key's value differed.
type sameCheck struct {
	Compared  string         `json:"compared"`
	Identical bool           `json:"identical"`
	Differing []string       `json:"differing"`
	Fields    map[string]int `json:"differing_fields"`
}

func newCheck(compared string) sameCheck {
	return sameCheck{Compared: compared, Identical: true, Differing: []string{}, Fields: map[string]int{}}
}

// note records case id's answers a against the reference ref.
func (s *sameCheck) note(id string, ref, a []node) {
	if equalAll(ref, a) {
		return
	}

	s.Differing = append(s.Differing, id)
	s.Identical = false

	for _, k := range diffKeys(ref, a) {
		s.Fields[k]++
	}
}

// diffKeys are the top-level keys whose values differ in some turn of a and b, in first-seen order; a missing turn
// counts as "turns".
func diffKeys(a, b []node) []string {
	if len(a) != len(b) {
		return []string{"turns"}
	}

	var keys []string

	for i := range a {
		x, y := a[i], b[i]

		for _, k := range append(append([]string{}, x.keys...), y.keys...) {
			if slices.Contains(keys, k) {
				continue
			}

			vx, vy := x.get(k), y.get(k)
			if vx == nil || vy == nil || !equalNodes(*vx, *vy) {
				keys = append(keys, k)
			}
		}
	}

	return keys
}

// libraryCase is one case's libneedle3 run in library.json: needle_init's return (the static-prefix token count),
// needle_complete's return per turn, whether the JSON equals the CLI's, and whether a second pass in the same process
// gave the same JSON. Answers is set only when the library differs from the CLI.
type libraryCase struct {
	Init      int32             `json:"init"`
	Complete  []int32           `json:"complete"`
	SameAsCLI bool              `json:"same_as_cli"`
	Repeat    bool              `json:"same_on_repeat"`
	Answers   []json.RawMessage `json:"answers,omitempty"`
}

// otherThreads are the thread counts every case is also answered with and compared against cliThreads; "" is the
// engine's default.
var otherThreads = []string{"", "1", "2", "3", "8"}

// cmdCases answers every corpus case with the needle CLI at --threads 4 (recorded), again at the otherThreads, with
// `needle --serve` for the first turn of multi-turn cases, and with libneedle3's needle_complete (twice), and writes
// responses.json, library.json and manifest.json.
func cmdCases(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, true)
	if err != nil {
		return err
	}
	defer o.close()

	man := textManifest{
		Speech: *o.whistle, Files: fixtureFiles,
		header: o.header("needle --model needle3.cact --tools <toolset> --system <system file> --threads 4 " +
			"--prompt <query> [--max N]; needle --serve on the same flags for cases with tool results"),
		Cases: len(o.corpus.Cases), Responses: responsesFile,
		Dropped: []string{"*_tps", "*_ms", "*_mb"}, Types: map[string]int{}, Calls: map[string]int{},
		Serve: newCheck("needle --serve's answer to the query against --prompt's, for cases with tool results"),
	}

	for _, t := range otherThreads {
		what := "--threads " + t
		if t == "" {
			what = "the default thread count (no --threads)"
		}

		man.Threads = append(man.Threads, newCheck(what+" against --threads "+cliThreads))
	}

	responses := node{kind: kindObject}
	answers := make(map[string][]node, len(o.corpus.Cases))

	for _, k := range o.corpus.Cases {
		got, err := o.recordCase(ctx, k, &man)
		if err != nil {
			return fmt.Errorf("%s: %w", k.ID, err)
		}

		answers[k.ID] = got
		responses.keys = append(responses.keys, k.ID)
		responses.elems = append(responses.elems, node{kind: kindArray, elems: got})
		man.Turns += len(got)

		_, _ = fmt.Fprintf(log, "%-24s %s\n", k.ID, summary(got[len(got)-1]))
	}

	err = writeFile(filepath.Join(cfg.outDir, responsesFile), marshalPretty(responses))
	if err != nil {
		return err
	}

	lib, err := o.libraryCases(answers, &man)
	if err != nil {
		return err
	}

	err = writeJSON(cfg.outDir, libraryFile, struct {
		header

		Cases map[string]libraryCase `json:"cases"`
	}{o.header("libneedle3: needle_init(system, tools, NULL), needle_reset, needle_complete per turn"), lib})
	if err != nil {
		return err
	}

	return writeJSON(cfg.outDir, manifestFile, man)
}

// recordCase answers case k with the CLI at the pinned thread count and at the otherThreads, and returns the pinned
// answers.
func (o *oracle) recordCase(ctx context.Context, k textCase, man *textManifest) ([]node, error) {
	got, err := o.cliCase(ctx, k, cliThreads)
	if err != nil {
		return nil, err
	}

	for i, t := range otherThreads {
		other, err := o.cliCase(ctx, k, t)
		if err != nil {
			return nil, fmt.Errorf("--threads %q: %w", t, err)
		}

		man.Threads[i].note(k.ID, got, other)
	}

	if len(k.Results) > 0 {
		args, err := o.baseArgs(k.Toolset, o.corpus.system(k))
		if err != nil {
			return nil, err
		}

		first, err := answer(ctx, o.needle(), append(args, maxArgs(k, "--prompt", k.Query)...))
		if err != nil {
			return nil, err
		}

		man.Serve.note(k.ID, got[:1], []node{first})
	}

	for _, a := range got {
		tally(man, a)
	}

	return got, nil
}

// cliCase answers every turn of case k with the CLI at the given thread count ("" for the default): --prompt for a
// single turn, --serve for a conversation.
func (o *oracle) cliCase(ctx context.Context, k textCase, threads string) ([]node, error) {
	args, err := o.threadArgs(k.Toolset, o.corpus.system(k), threads)
	if err != nil {
		return nil, err
	}

	if len(k.Results) == 0 {
		a, err := answer(ctx, o.needle(), append(args, maxArgs(k, "--prompt", k.Query)...))
		if err != nil {
			return nil, err
		}

		return []node{a}, nil
	}

	return serveTurns(ctx, o.needle(), append(args, maxArgs(k)...), k.turns())
}

// maxArgs appends --max to args when case k sets a token limit.
func maxArgs(k textCase, args ...string) []string {
	if k.Max > 0 {
		return append(args, "--max", itoa(k.Max))
	}

	return args
}

// libraryCases answers every case with libneedle3 twice and compares each answer with the CLI's.
func (o *oracle) libraryCases(cli map[string][]node, man *textManifest) (map[string]libraryCase, error) {
	lib, err := openLibrary(o.cfg.engineDir, o.needle3Path())
	if err != nil {
		return nil, err
	}

	man.Library = newCheck("libneedle3 needle_complete, twice, against the CLI")
	out := make(map[string]libraryCase, len(o.corpus.Cases))

	for _, k := range o.corpus.Cases {
		first, lc, err := o.libraryCase(lib, k)
		if err != nil {
			return nil, fmt.Errorf("%s: library: %w", k.ID, err)
		}

		again, _, err := o.libraryCase(lib, k)
		if err != nil {
			return nil, fmt.Errorf("%s: library repeat: %w", k.ID, err)
		}

		lc.Repeat = equalAll(first, again)
		lc.SameAsCLI = equalAll(first, cli[k.ID])

		if !lc.SameAsCLI {
			for _, a := range first {
				lc.Answers = append(lc.Answers, compact(a))
			}
		}

		man.Library.note(k.ID, cli[k.ID], first)

		if !lc.Repeat {
			man.Library.note(k.ID+" (repeat)", first, again)
		}

		out[k.ID] = lc
	}

	return out, nil
}

// libraryCase runs case k through needle_init, needle_reset and needle_complete per turn.
func (o *oracle) libraryCase(lib *library, k textCase) ([]node, libraryCase, error) {
	rc, err := lib.session(o.corpus.system(k), o.corpus.tools(k.Toolset))
	if err != nil {
		return nil, libraryCase{}, err
	}

	lib.reset()

	lc := libraryCase{Init: rc}

	var got []node

	for _, t := range k.turns() {
		c, err := lib.completeText(t, k.max())
		if err != nil {
			return nil, libraryCase{}, err
		}

		a, err := normaliseResponse(c.out)
		if err != nil {
			return nil, libraryCase{}, err
		}

		got = append(got, a)
		lc.Complete = append(lc.Complete, c.rc)
	}

	return got, lc, nil
}

func equalAll(a, b []node) bool {
	return slices.EqualFunc(a, b, equalNodes)
}

// tally counts an answer's type and calls into the manifest and widens its confidence range.
func tally(man *textManifest, a node) {
	if t := a.get("type"); t != nil && t.kind == kindString {
		man.Types[t.str]++
	}

	if fc := a.get("function_calls"); fc != nil {
		man.Calls[itoa(len(fc.elems))]++
	}

	c := a.get("confidence")
	if c == nil || c.kind != kindNumber {
		return
	}

	v := json.Number(canonicalNumber(c.str))

	if man.MinConf == "" || less(v, man.MinConf) {
		man.MinConf = v
	}

	if man.MaxConf == "" || less(man.MaxConf, v) {
		man.MaxConf = v
	}
}

func less(a, b json.Number) bool {
	x, _ := a.Float64()
	y, _ := b.Float64()

	return x < y
}

// summary is one log line for an answer: its type, calls and confidence.
func summary(a node) string {
	var b strings.Builder

	if t := a.get("type"); t != nil {
		b.WriteString(t.str)
	}

	for _, key := range []string{"function_calls", "suppressed_calls"} {
		fc := a.get(key)
		if fc == nil || len(fc.elems) == 0 {
			continue
		}

		b.WriteString(" " + key + "=")

		for i, c := range fc.elems {
			if i > 0 {
				b.WriteByte(',')
			}

			b.Write(compact(c))
		}
	}

	if c := a.get("confidence"); c != nil {
		b.WriteString(" confidence=" + c.str)
	}

	if e := a.get("error_code"); e != nil && e.kind == kindString {
		b.WriteString(" error_code=" + e.str)
	}

	return b.String()
}
