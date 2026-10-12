package toolcall

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/text"
)

// The fixture of `just oracle-text retrieval` and the pinned text model.
const (
	engineRetrieval = "../../testdata/text/engine/retrieval.json"
	needle3Cact     = "../../models/needle3/needle3.cact"
)

// retrievalFile is testdata/text/engine/retrieval.json (tools/oracle/text
// retrieval.go).
type retrievalFile struct {
	Heads     []int  `json:"heads"`
	Context   int    `json:"context"`
	Toolset   string `json:"toolset"`
	Synthetic string `json:"synthetic_tool"`
	Probes    []struct {
		Tools     int    `json:"tools"`
		Synthetic int    `json:"synthetic"`
		Pad       int    `json:"pad"`
		SHA256    string `json:"tools_sha256"`
		Init      int    `json:"init"`
		Error     string `json:"error"`
		Index     bool   `json:"index_written"`
	} `json:"probes"`
	CLI struct {
		Exit   int  `json:"exit"`
		Index  bool `json:"index_written"`
		Answer struct {
			Calls []call `json:"function_calls"`
		} `json:"answer"`
	} `json:"cli"`
}

// syntheticTool is the oracle's template for the tools appended to the
// catalogue: tool i (from 1), and pad " a" words after the last one's
// description. The fixture's synthetic_tool and every probe's SHA-256 pin
// it to the oracle's.
const syntheticTool = `{"name":"tool_%03d","description":"Do thing number %d%s",` +
	`"parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}`

// probeTools rebuilds a probe's tools JSON as the oracle sent it: the first
// n tools of the catalogue, or the whole catalogue and synthetic tools.
func probeTools(catalogue []json.RawMessage, tools, synthetic, pad int) (string, error) {
	if synthetic == 0 {
		js, err := json.Marshal(catalogue[:tools])

		return string(js), err
	}

	js, err := json.Marshal(catalogue)
	if err != nil {
		return "", err
	}

	parts := make([]string, synthetic)
	for i := range parts {
		tail := ""
		if i == synthetic-1 {
			tail = strings.Repeat(" a", pad)
		}

		parts[i] = fmt.Sprintf(syntheticTool, i+1, i+1, tail)
	}

	return strings.TrimSuffix(string(js), "]") + "," + strings.Join(parts, ",") + "]", nil
}

// budgetError is the counts in needle_init's budget error.
var budgetError = regexp.MustCompile(`static prefix is (\d+) tokens, model context is (\d+) tokens`)

// TestStaticPrefixPastFiveTools: from 1 to 239 tools, and at the context
// budget, the static prefix StaticPrefix renders has needle_init's count, so
// the engine renders every tool, not 5; CheckPrefix accepts what needle_init
// accepts and refuses the rest with the engine's numbers; and no run writes a
// tool index (docs/FINDINGS.md F-049).
func TestStaticPrefixPastFiveTools(t *testing.T) {
	p := loadTokenizer(t)

	var (
		fx     retrievalFile
		corpus struct {
			System   string                     `json:"system"`
			Toolsets map[string]json.RawMessage `json:"toolsets"`
		}
		catalogue []json.RawMessage
	)

	readJSON(t, engineRetrieval, &fx)
	readJSON(t, engineCases, &corpus)

	err := json.Unmarshal(corpus.Toolsets[fx.Toolset], &catalogue)
	if err != nil {
		t.Fatal(err)
	}

	if want := fmt.Sprintf(syntheticTool, 0, 0, ""); fx.Synthetic != want {
		t.Fatalf("synthetic tool %s, the test builds %s", fx.Synthetic, want)
	}

	var past, refused int

	for _, pr := range fx.Probes {
		name := fmt.Sprintf("%d tools (%d synthetic, pad %d)", pr.Tools, pr.Synthetic, pr.Pad)

		raw, err := probeTools(catalogue, pr.Tools-pr.Synthetic, pr.Synthetic, pr.Pad)
		if err != nil {
			t.Fatal(err)
		}

		if sum := sha256.Sum256([]byte(raw)); hex.EncodeToString(sum[:]) != pr.SHA256 {
			t.Fatalf("%s: the rebuilt tools are not the oracle's", name)
		}

		tools, err := ToolsJSON([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}

		prefix := StaticPrefix(p, corpus.System, tools)
		budget := CheckPrefix(prefix, fx.Context)

		if pr.Index || Retrieves(slices.Contains(fx.Heads, cact.HeadEmbedding), pr.Tools) {
			t.Errorf("%s: index written %v, Retrieves true", name, pr.Index)
		}

		if pr.Error == "" {
			if len(prefix) != pr.Init || budget != nil {
				t.Errorf("%s: static prefix %d tokens (%v), needle_init %d", name, len(prefix), budget, pr.Init)
			}

			if pr.Tools > RetrievalTools {
				past++
			}

			continue
		}

		refused++

		m := budgetError.FindStringSubmatch(pr.Error)
		if m == nil || !errors.Is(budget, ErrContextBudget) {
			t.Errorf("%s: needle_init %q, CheckPrefix %v", name, pr.Error, budget)

			continue
		}

		if n, _ := strconv.Atoi(m[1]); n != len(prefix) || m[2] != strconv.Itoa(fx.Context) {
			t.Errorf("%s: needle_init counted %s of %s tokens, StaticPrefix %d of %d", name, m[1], m[2], len(prefix), fx.Context)
		}

		if !strings.Contains(budget.Error(), m[0]) {
			t.Errorf("%s: CheckPrefix says %q, needle_init %q", name, budget, m[0])
		}
	}

	if past == 0 || refused == 0 {
		t.Errorf("%d accepted probes past %d tools, %d refused; the fixture covers neither side", past, RetrievalTools, refused)
	}

	t.Logf("%d probes: %d accepted past %d tools, %d refused at %d tokens", len(fx.Probes), past, RetrievalTools, refused, fx.Context)
}

// TestRetrievalOff: the pinned needle3.cact has no embedding head, so the
// engine does not retrieve; the CLI with --tool-index writes no index and
// calls the catalogue's 17th tool.
func TestRetrievalOff(t *testing.T) {
	var fx retrievalFile

	readJSON(t, engineRetrieval, &fx)

	if !slices.Equal(fx.Heads, []int{cact.HeadConfidence}) {
		t.Errorf("heads.manifest %v, want only the confidence head", fx.Heads)
	}

	if fx.CLI.Exit != 0 || fx.CLI.Index || len(fx.CLI.Answer.Calls) != 1 || fx.CLI.Answer.Calls[0].Name != "set_reminder" {
		t.Errorf("CLI with --tool-index: exit %d, index %v, calls %v", fx.CLI.Exit, fx.CLI.Index, fx.CLI.Answer.Calls)
	}

	_, err := os.Stat(needle3Cact)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3`", needle3Cact)
	}

	src, err := text.Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = src.Close() }()

	if src.HasHead(cact.HeadEmbedding) || !src.HasHead(cact.HeadConfidence) || src.HasHead(cact.HeadRouter) {
		t.Errorf("heads: embedding %v, confidence %v, router %v; the fixture says %v", src.HasHead(cact.HeadEmbedding),
			src.HasHead(cact.HeadConfidence), src.HasHead(cact.HeadRouter), fx.Heads)
	}

	if got := src.TextConfig().MaxSeqLen; got != fx.Context {
		t.Errorf("max_seq_len %d, needle_init's context %d", got, fx.Context)
	}
}

// TestCheckPrefix: the budget refuses context tokens and more.
func TestCheckPrefix(t *testing.T) {
	for _, c := range []struct {
		n, context int
		ok         bool
	}{
		{0, 8, true}, {7, 8, true}, {8, 8, false}, {9, 8, false},
	} {
		err := CheckPrefix(make([]int, c.n), c.context)
		if (err == nil) != c.ok || err != nil && !errors.Is(err, ErrContextBudget) {
			t.Errorf("CheckPrefix(%d tokens, %d) = %v", c.n, c.context, err)
		}
	}

	if !Retrieves(true, RetrievalTools+1) || Retrieves(true, RetrievalTools) || Retrieves(false, 100) {
		t.Error("Retrieves: an embedding head and more than RetrievalTools tools")
	}
}
