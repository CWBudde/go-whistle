package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// weightsFile is the model the weights-gated tests run, relative to the
// package.
const weightsFile = "../../models/needle3/needle3.cact"

var (
	weightsOnce sync.Once
	weights     model
	errWeights  error
)

// shared is the loaded model with a Close that keeps it open, so that
// every run of the CLI in the tests reuses one load.
type shared struct{ model }

func (shared) Close() error { return nil }

// weightsCLI is the CLI on the model in the engine's numerics, loaded once
// for the package's tests and never closed. It skips without the weights.
func weightsCLI(t *testing.T, stdout, stderr *bytes.Buffer) *cli {
	t.Helper()

	if testing.Short() {
		t.Skip("decodes with the weights")
	}

	// The corpus alone ran past 10 minutes under -race. The CLI's own
	// concurrency, the server's lock, is checked without the weights
	// (TestServeSerialises).
	if raceDetector {
		t.Skip("decodes with the weights, too slow under the race detector")
	}

	_, err := os.Stat(weightsFile)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just fetch-needle3`", weightsFile)
	}

	weightsOnce.Do(func() { weights, errWeights = openNeedle(weightsFile) })

	if errWeights != nil {
		t.Fatal(errWeights)
	}

	return &cli{
		stdout: stdout, stderr: stderr,
		open: func(path string) (model, error) {
			if path != weightsFile {
				return nil, fmt.Errorf("the tests open %s, not %s", weightsFile, path)
			}

			return shared{weights}, nil
		},
		threads: func(int) {},
	}
}

// engineCorpus is testdata/text/engine/cases.json.
type engineCorpus struct {
	System   string                     `json:"system"`
	Toolsets map[string]json.RawMessage `json:"toolsets"`
	Cases    []struct {
		ID      string            `json:"id"`
		Toolset string            `json:"toolset"`
		System  string            `json:"system"`
		Query   string            `json:"query"`
		Max     int               `json:"max"`
		Results []json.RawMessage `json:"results"`
	} `json:"cases"`
}

type engineCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// engineEnvelope is an envelope as recorded or as the CLI prints it,
// parsed.
type engineEnvelope struct {
	Type       string       `json:"type"`
	Calls      []engineCall `json:"function_calls"`
	Suppressed []engineCall `json:"suppressed_calls"`
	Reasoning  *string      `json:"reasoning"`
	Confidence float64      `json:"confidence"`
}

func sameCalls(got, want []engineCall) bool {
	if len(got) != len(want) {
		return false
	}

	for i := range got {
		if got[i].Name != want[i].Name || !reflect.DeepEqual(got[i].Arguments, want[i].Arguments) {
			return false
		}
	}

	return true
}

// tally counts envelopes against the engine's as the needle package's
// TestCompleteMatchesPipeline does: type and function_calls, reasoning,
// and the confidence within 0.02 on the turns decoded as the engine's
// (the calls, suppressed ones included, and the reasoning).
type tally struct {
	total, typeAndFC, reasoning, decoded, within int
}

func (c *tally) add(t *testing.T, at string, js []byte, want engineEnvelope) {
	t.Helper()

	var got engineEnvelope

	readAnswer(t, js, &got)

	typeAndFC := got.Type == want.Type && sameCalls(got.Calls, want.Calls)
	reasoning := reflect.DeepEqual(got.Reasoning, want.Reasoning)
	same := reasoning && got.Type == want.Type &&
		sameCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...))
	off := math.Abs(got.Confidence - want.Confidence)

	c.total++

	for _, f := range []struct {
		ok bool
		n  *int
	}{{typeAndFC, &c.typeAndFC}, {reasoning, &c.reasoning}, {same, &c.decoded}, {same && off <= 0.02, &c.within}} {
		if f.ok {
			*f.n++
		}
	}

	t.Logf("%-28s type+function_calls %-5v reasoning %-5v confidence off %.4f", at, typeAndFC, reasoning, off)
}

func (c *tally) log(t *testing.T) {
	t.Helper()

	t.Logf("type and function_calls equal on %d of %d", c.typeAndFC, c.total)
	t.Logf("reasoning equal on %d of %d", c.reasoning, c.total)
	t.Logf("confidence within 0.02 on %d of %d turns decoded as the engine's", c.within, c.decoded)
}

func readAnswer(t *testing.T, js []byte, v any) {
	t.Helper()

	err := json.Unmarshal(js, v)
	if err != nil {
		t.Fatalf("%v: %s", err, js)
	}
}

// TestPromptMatchesEngine runs the CLI as the oracle ran the engine's on
// the corpus (cases.json): --prompt for each single-turn case, --serve for
// the case with a tool result, the tools and the system facts from files,
// and counts the answers against responses.json, parsed and without the
// timings, as the API's TestCompleteMatchesPipeline does. On arm64 (M5
// Pro) the API's counts: type and function_calls 24 of 25, reasoning 22 of
// 25, the confidence within 0.02 on 20 of the 22 turns decoded as the
// engine's (docs/FINDINGS.md F-066).
func TestPromptMatchesEngine(t *testing.T) {
	var stdout, stderr bytes.Buffer

	c := weightsCLI(t, &stdout, &stderr)

	var (
		corpus    engineCorpus
		responses map[string][]engineEnvelope
		counts    tally
	)

	readJSON(t, engineCases, &corpus)
	readJSON(t, engineAnswers, &responses)

	for _, cs := range corpus.Cases {
		system := corpus.System
		if cs.System != "" {
			system = cs.System
		}

		args := []string{
			"--model", weightsFile, "--tools", writeFile(t, "tools.json", string(corpus.Toolsets[cs.Toolset])),
			"--system", writeFile(t, "system.txt", system),
		}
		if cs.Max > 0 {
			args = append(args, "--max", strconv.Itoa(cs.Max))
		}

		if len(cs.Results) > 0 {
			t.Run(cs.ID, func(t *testing.T) {
				url := serveCLI(t, weightsCLI(t, &bytes.Buffer{}, &bytes.Buffer{}), args...)

				inputs := make([]string, 0, 1+len(cs.Results))

				inputs = append(inputs, cs.Query)
				for _, r := range cs.Results {
					inputs = append(inputs, string(r))
				}

				for i, in := range inputs {
					counts.add(t, cs.ID+"/"+strconv.Itoa(i), postTurn(t, url, in), responses[cs.ID][i])
				}
			})

			continue
		}

		stdout.Reset()
		stderr.Reset()

		if code := c.run(t.Context(), append(args, "--prompt", cs.Query)); code != 0 || stderr.Len() != 0 {
			t.Fatalf("%s: exit %d, stderr %q", cs.ID, code, stderr.String())
		}

		counts.add(t, cs.ID, stdout.Bytes(), responses[cs.ID][0])
	}

	counts.log(t)

	if counts.total != 25 {
		t.Errorf("%d turns, want the corpus's 25", counts.total)
	}

	// The API's floors: its int8 counts less one turn for another
	// architecture's rounding at the near-ties.
	if counts.typeAndFC < 23 || counts.reasoning < 21 || counts.within < 19 {
		t.Errorf("below the floors (23, 21 and 19)")
	}
}

// postTurn posts input as the next turn of the --serve at url and returns
// the envelope.
func postTurn(t *testing.T, url, input string) []byte {
	t.Helper()

	body, err := json.Marshal(map[string]string{"input": input})
	if err != nil {
		t.Fatal(err)
	}

	status, _, js := post(t, url+"/complete", string(body))
	if status != http.StatusOK {
		t.Fatalf("complete: %d %s", status, js)
	}

	return []byte(js)
}

// turnsFile is testdata/text/engine/turns.json.
type turnsFile struct {
	System    string `json:"system"`
	Toolset   string `json:"toolset"`
	Scenarios []struct {
		ID     string `json:"id"`
		Max    int    `json:"max"`
		Refuse bool   `json:"fail_input_overflow"`
		Steps  []struct {
			Input struct {
				Head   string `json:"head"`
				Unit   string `json:"unit"`
				Repeat int    `json:"repeat"`
				Tail   string `json:"tail"`
				Reset  bool   `json:"reset"`
			} `json:"input"`
			SHA256 string          `json:"sha256"`
			Answer json.RawMessage `json:"answer"`
		} `json:"steps"`
	} `json:"scenarios"`
}

// overflowError is the engine's --fail-input-overflow error.
var overflowError = regexp.MustCompile(`^input exceeds the context window \(\d+ tokens, \d+ available\)$`)

// goEscaped are the runes json.Marshal writes as \u escapes in a string:
// the oracle of turns.json sent them so, and the engine read each as the
// escape without its backslash (docs/FINDINGS.md F-063).
var goEscaped = "<&>" + string(rune(0x2028)) + string(rune(0x2029))

// asRead is input as the engine read it from the oracle's request body.
func asRead(input string) string {
	var b strings.Builder

	for _, r := range input {
		if strings.ContainsRune(goEscaped, r) {
			fmt.Fprintf(&b, "u%04x", r)

			continue
		}

		b.WriteRune(r)
	}

	return b.String()
}

// TestServeMatchesTurns replays turns.json through --serve, one server per
// scenario at its --max and --fail-input-overflow, each input as the
// engine read it: every refusal is the engine's envelope (the error with
// its two counts, the empty fields), and the answered turns are counted
// against the engine's. On arm64 (M5 Pro): all 53 refusals, and of the 51
// answered turns type and function_calls 50, reasoning 50, the confidence
// within 0.02 on 47 of the 50 decoded as the engine's (docs/FINDINGS.md
// F-066).
func TestServeMatchesTurns(t *testing.T) {
	weightsCLI(t, &bytes.Buffer{}, &bytes.Buffer{})

	var (
		fx     turnsFile
		corpus engineCorpus
		counts tally
	)

	readJSON(t, engineTurns, &fx)
	readJSON(t, engineCases, &corpus)

	tools := writeFile(t, "tools.json", string(corpus.Toolsets[fx.Toolset]))
	system := writeFile(t, "system.txt", fx.System)
	refused := 0

	for _, sc := range fx.Scenarios {
		t.Run(sc.ID, func(t *testing.T) {
			args := []string{"--model", weightsFile, "--tools", tools, "--system", system, "--max", strconv.Itoa(sc.Max)}
			if sc.Refuse {
				args = append(args, "--fail-input-overflow")
			}

			url := serveCLI(t, weightsCLI(t, &bytes.Buffer{}, &bytes.Buffer{}), args...)

			for i, st := range sc.Steps {
				if st.Input.Reset {
					if status, _, body := post(t, url+"/reset", `{}`); status != http.StatusOK {
						t.Fatalf("reset: %d %s", status, body)
					}

					continue
				}

				text := st.Input.Head + strings.Repeat(st.Input.Unit, st.Input.Repeat) + st.Input.Tail
				if sum := sha256.Sum256([]byte(text)); hex.EncodeToString(sum[:]) != st.SHA256 {
					t.Fatalf("step %d: the rebuilt input is not the engine's", i)
				}

				js := postTurn(t, url, asRead(text))

				var want struct {
					Error string `json:"error"`
				}

				readAnswer(t, st.Answer, &want)

				if overflowError.MatchString(want.Error) {
					refused++

					if !reflect.DeepEqual(parsed(t, js), parsed(t, st.Answer)) {
						t.Errorf("step %d: refusal\n%s\nthe engine's\n%s", i, js, st.Answer)
					}

					continue
				}

				var e engineEnvelope

				readAnswer(t, st.Answer, &e)
				counts.add(t, sc.ID+"/"+strconv.Itoa(i), js, e)
			}
		})
	}

	counts.log(t)
	t.Logf("%d refusals with the engine's envelope", refused)

	if refused != 53 || counts.total != 51 {
		t.Errorf("%d refused and %d answered turns, want turns.json's 53 and 51", refused, counts.total)
	}

	// The measured counts less one turn for another architecture's
	// rounding at the near-ties.
	if counts.typeAndFC < 49 || counts.reasoning < 49 || counts.within < 46 {
		t.Errorf("below the floors (49, 49 and 46)")
	}
}

// TestToolIndexMatchesEngine runs the CLI with --tool-index as the oracle
// ran the engine's (retrieval.json): it exits 0 with the engine's calls
// and writes no index.
func TestToolIndexMatchesEngine(t *testing.T) {
	var stdout, stderr bytes.Buffer

	c := weightsCLI(t, &stdout, &stderr)

	var (
		fx struct {
			Toolset string `json:"toolset"`
			CLI     struct {
				Exit         int             `json:"exit"`
				IndexWritten bool            `json:"index_written"`
				Answer       json.RawMessage `json:"answer"`
			} `json:"cli"`
		}
		corpus engineCorpus
	)

	readJSON(t, engineIndex, &fx)
	readJSON(t, engineCases, &corpus)

	query := ""

	for _, cs := range corpus.Cases {
		if cs.ID == "catalogue_reminder" {
			query = cs.Query
		}
	}

	index := filepath.Join(t.TempDir(), "tools.index")

	code := c.run(t.Context(), []string{
		"--model", weightsFile, "--tools", writeFile(t, "tools.json", string(corpus.Toolsets[fx.Toolset])),
		"--system", writeFile(t, "system.txt", corpus.System), "--tool-index", index, "--prompt", query,
	})

	_, err := os.Stat(index)
	if written := err == nil; code != fx.CLI.Exit || written != fx.CLI.IndexWritten {
		t.Fatalf("exit %d, index written %v; the engine: exit %d, written %v (stderr %q)", code, written,
			fx.CLI.Exit, fx.CLI.IndexWritten, stderr.String())
	}

	var got, want engineEnvelope

	readAnswer(t, stdout.Bytes(), &got)
	readAnswer(t, fx.CLI.Answer, &want)

	if got.Type != want.Type || !sameCalls(got.Calls, want.Calls) {
		t.Errorf("answer %s\nthe engine's %s", stdout.Bytes(), fx.CLI.Answer)
	}
}
