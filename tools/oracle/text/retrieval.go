//go:build oracle

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// retrievalToolset is the corpus tool set the retrieval probes grow: 17 tools, more than the 5 a turn renders when
// the engine retrieves.
const retrievalToolset = "catalogue"

// retrievalCase is the corpus case the CLI answers with --tool-index: its tool is the catalogue's last.
const retrievalCase = "catalogue_reminder"

// syntheticSizes are the catalogue sizes past the corpus's tools that needle_init is run on.
var syntheticSizes = []int{32, 64, 128, 200}

// syntheticTool is the template of the tools appended to the catalogue: tool i of them (from 1), and pad " a" words
// after the description of the last one, which moves the static prefix one token at a time.
const syntheticTool = `{"name":"tool_%03d","description":"Do thing number %d%s",` +
	`"parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}`

// syntheticTools is the catalogue's compact JSON with n synthetic tools appended, the last one padded.
func syntheticTools(catalogue string, n, pad int) string {
	if n == 0 {
		return catalogue
	}

	parts := make([]string, n)
	for i := range parts {
		tail := ""
		if i == n-1 {
			tail = strings.Repeat(" a", pad)
		}

		parts[i] = fmt.Sprintf(syntheticTool, i+1, i+1, tail)
	}

	return strings.TrimSuffix(catalogue, "]") + "," + strings.Join(parts, ",") + "]"
}

// initProbe is one needle_init run with a tool index path: the tool set (the first Tools of the catalogue, or the
// catalogue and Synthetic tools of syntheticTool with Pad words), needle_init's result and error, and whether the
// engine wrote the index file.
type initProbe struct {
	Tools     int    `json:"tools"`
	Synthetic int    `json:"synthetic"`
	Pad       int    `json:"pad"`
	SHA256    string `json:"tools_sha256"`
	Init      int    `json:"init"`
	Error     string `json:"error,omitempty"`
	Index     bool   `json:"index_written"`
}

// cliIndex is the CLI's answer to retrievalCase with --tool-index.
type cliIndex struct {
	Run    string          `json:"run"`
	Exit   int             `json:"exit"`
	Stderr string          `json:"stderr,omitempty"`
	Index  bool            `json:"index_written"`
	Answer json.RawMessage `json:"answer,omitempty"`
}

// retrievalFixture is testdata/text/engine/retrieval.json.
type retrievalFixture struct {
	header

	Heads     []int       `json:"heads"`
	Context   int         `json:"context"`
	Toolset   string      `json:"toolset"`
	Synthetic string      `json:"synthetic_tool"`
	Probes    []initProbe `json:"probes"`
	CLI       cliIndex    `json:"cli"`
}

// cmdRetrieval records what the engine does with more tools than a turn renders under retrieval: the head codes of
// needle3.cact's heads.manifest (retrieval needs an embedding head), needle_init's result with a tool index path for
// every prefix of the catalogue and for synthetic catalogues up to and past the context budget, whether the index file
// appears, and the CLI's answer with --tool-index.
func cmdRetrieval(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	heads, budget, err := cactHeads(o.needle3Path())
	if err != nil {
		return err
	}

	lib, err := openLibrary(cfg.engineDir, o.needle3Path())
	if err != nil {
		return err
	}

	fx := retrievalFixture{
		header: o.header("libneedle3: needle_init(system, tools, <scratch index path>); " +
			"needle --model needle3.cact --tools <catalogue> --system <system file> --threads 4 --tool-index <path> --prompt <query>"),
		Heads: heads, Context: budget, Toolset: retrievalToolset, Synthetic: fmt.Sprintf(syntheticTool, 0, 0, ""),
	}

	sets, err := o.retrievalSets(budget)
	if err != nil {
		return err
	}

	for i, s := range sets {
		p, err := o.initProbe(lib, s, i)
		if err != nil {
			return err
		}

		fx.Probes = append(fx.Probes, p)
		_, _ = fmt.Fprintf(log, "%3d tools (%3d synthetic, pad %3d): init %5d index %v %s\n",
			p.Tools, p.Synthetic, p.Pad, p.Init, p.Index, p.Error)
	}

	fx.CLI, err = o.cliIndex(ctx)
	if err != nil {
		return err
	}

	_, _ = fmt.Fprintf(log, "cli --tool-index: exit %d index %v %s\n", fx.CLI.Exit, fx.CLI.Index, fx.CLI.Answer)

	return writeJSON(cfg.outDir, "retrieval.json", fx)
}

// toolSet is one tool set the probes run: its JSON and how it was built.
type toolSet struct {
	json                  string
	tools, synthetic, pad int
}

// retrievalSets are the catalogue's prefixes, the synthetic sizes, and the two sets around the context budget: a
// static prefix of context−1 tokens and one of context tokens, found with the Go rendering (the engine judges them).
func (o *oracle) retrievalSets(budget int) ([]toolSet, error) {
	var all []json.RawMessage

	err := json.Unmarshal([]byte(o.corpus.tools(retrievalToolset)), &all)
	if err != nil {
		return nil, fmt.Errorf("toolset %s: %w", retrievalToolset, err)
	}

	sets := make([]toolSet, 0, len(all)+len(syntheticSizes)+2)

	for n := 1; n <= len(all); n++ {
		js, err := json.Marshal(all[:n])
		if err != nil {
			return nil, fmt.Errorf("toolset %s: %w", retrievalToolset, err)
		}

		sets = append(sets, toolSet{json: string(js), tools: n})
	}

	catalogue := o.corpus.tools(retrievalToolset)

	for _, size := range syntheticSizes {
		n := size - len(all)
		sets = append(sets, toolSet{json: syntheticTools(catalogue, n, 0), tools: size, synthetic: n})
	}

	edge, err := o.budgetSets(catalogue, len(all), budget)
	if err != nil {
		return nil, err
	}

	return append(sets, edge...), nil
}

// budgetSets finds the synthetic catalogues whose static prefix is context−1 and context tokens.
func (o *oracle) budgetSets(catalogue string, base, budget int) ([]toolSet, error) {
	p, err := spm.Load(filepath.Join(o.cfg.needle3Dir, "tokenizer.model"))
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	count := func(n, pad int) (int, error) {
		tools, err := toolcall.ToolsJSON([]byte(syntheticTools(catalogue, n, pad)))
		if err != nil {
			return 0, fmt.Errorf("%d synthetic tools: %w", n, err)
		}

		return len(toolcall.StaticPrefix(p, o.corpus.System, tools)), nil
	}

	// The largest synthetic count below the budget, then pad its last tool up to it.
	n := 0

	for {
		c, err := count(n+1, 0)
		if err != nil {
			return nil, err
		}

		if c >= budget {
			break
		}

		n++
	}

	var sets []toolSet

	for pad := 0; len(sets) < 2 && pad < budget; pad++ {
		c, err := count(n, pad)
		if err != nil {
			return nil, err
		}

		if c == budget-1 || c == budget {
			sets = append(sets, toolSet{json: syntheticTools(catalogue, n, pad), tools: base + n, synthetic: n, pad: pad})
		}
	}

	if len(sets) != 2 {
		return nil, fmt.Errorf("no padding of %d synthetic tools gives %d and %d tokens", n, budget-1, budget)
	}

	return sets, nil
}

// initProbe runs needle_init on one tool set with a fresh index path in the scratch directory.
func (o *oracle) initProbe(lib *library, s toolSet, i int) (initProbe, error) {
	sum := sha256.Sum256([]byte(s.json))
	idx := filepath.Join(o.scratch, fmt.Sprintf("tools-%03d.idx", i))

	rc, err := lib.sessionIndex(o.corpus.System, s.json, idx)

	p := initProbe{
		Tools: s.tools, Synthetic: s.synthetic, Pad: s.pad, SHA256: hex.EncodeToString(sum[:]), Init: int(rc),
	}
	if err != nil {
		if rc >= 0 {
			return initProbe{}, err
		}

		p.Error = lib.lastError()
	}

	p.Index, err = exists(idx)

	return p, err
}

// cliIndex answers retrievalCase with the CLI and --tool-index.
func (o *oracle) cliIndex(ctx context.Context) (cliIndex, error) {
	k, err := o.corpus.find(retrievalCase)
	if err != nil {
		return cliIndex{}, err
	}

	args, err := o.baseArgs(k.Toolset, o.corpus.system(k))
	if err != nil {
		return cliIndex{}, err
	}

	idx := filepath.Join(o.scratch, "cli.idx")

	res, err := runCLI(ctx, o.needle(), append(args, "--tool-index", idx, "--prompt", k.Query))
	if err != nil {
		return cliIndex{}, err
	}

	out := cliIndex{
		Run:  "--tool-index <scratch path> --prompt <" + retrievalCase + ">",
		Exit: res.Exit, Stderr: res.Stderr,
	}

	if res.Exit == 0 {
		a, err := normaliseResponse(res.stdout)
		if err != nil {
			return cliIndex{}, err
		}

		out.Answer = compact(a)
	}

	out.Index, err = exists(idx)

	return out, err
}

// cactHeads reads the head codes of a text .cact's heads.manifest (none without one) and its context length.
func cactHeads(p string) ([]int, int, error) {
	f, err := cact.Open(p)
	if err != nil {
		return nil, 0, fmt.Errorf("heads: %w", err)
	}

	names, err := cact.TextNames(f)
	if err != nil {
		return nil, 0, fmt.Errorf("heads: %w", err)
	}

	heads := []int{}

	for i, e := range names {
		if e.Name != "heads.manifest" {
			continue
		}

		codes, err := f.FP16(i)
		if err != nil {
			return nil, 0, fmt.Errorf("heads.manifest: %w", err)
		}

		for _, c := range codes {
			heads = append(heads, int(math.Round(float64(c))))
		}
	}

	return heads, int(f.Header().MaxSeqLen), nil
}

func exists(p string) (bool, error) {
	_, err := os.Stat(p)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}

	if err != nil {
		return false, fmt.Errorf("stat %s: %w", p, err)
	}

	return true, nil
}
