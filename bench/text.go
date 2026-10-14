package bench

import (
	"bytes"
	"encoding/json"
	"io"
	"slices"
)

// The text bench's default paths (PLAN.md 11.5's Accept, docs/FINDINGS.md
// F-061), relative to the repository root where the just recipes run.
const (
	TextGoResultsPath     = "bench/results/text-go.json"
	TextEngineResultsPath = "bench/results/text-engine.json"
	TextReportPath        = "bench/RESULTS-text.md"
)

// DefaultTextPaths are the files `just bench-text` and
// `just oracle-text bench` use without flags.
var DefaultTextPaths = Paths{Go: TextGoResultsPath, Engine: TextEngineResultsPath, Report: TextReportPath}

// PackedWeightsBytes is what the packed needle3.cact's weights hold on
// exact kernels, as internal/text's TestPackedMemory counts them
// (docs/FINDINGS.md F-059): 38.252 MiB. The bench does not count it
// itself; the report quotes it against PackedWeightsBoundMiB.
const PackedWeightsBytes = 40_109_936

// PackedWeightsBoundMiB is 11.5's Accept bound on the packed weights, its
// "40 MB" read as 40 MiB by the maintainer's decision (F-059, F-061).
const PackedWeightsBoundMiB = 40

// TextAnswer is one timed answer to a first turn.
type TextAnswer struct {
	// PrefillTokens is what the first forward pass feeds: the static
	// prefix, the turn and the forced "<think>\n".
	PrefillTokens int
	// Generated is the answer's tokens as needle_complete counts them, the
	// forced ones included.
	Generated int
	// TTFTMS is the time to the first generated token; 0 is unknown.
	TTFTMS float64
	// PrefillTPS and DecodeTPS are the prefill and decode rates; 0 is
	// unknown (an answer of one token has no decode rate). PrefillTPS is
	// PrefillTokens over TTFTMS on both sides.
	PrefillTPS, DecodeTPS float64
	// ReportedPrefillTPS is the engine's own prefill_tps, which counts the
	// turn's tokens only (docs/FINDINGS.md F-061); 0 on the Go side.
	ReportedPrefillTPS float64
	// WallMS is the answer's wall time, whatever the side times as one
	// answer (TextSet.Wall).
	WallMS float64
}

// TextSamples collects the answers of one way of running the corpus, over
// the passes.
type TextSamples struct {
	ttft, prefill, decode, wall []float64
	reported                    []float64
	prefillTokens, generated    []float64
	passes                      int
	perCase                     map[string]*textCaseSamples
	order                       []string
}

type textCaseSamples struct {
	prefillTokens, generated int
	ttft, decode, wall       []float64
}

// Add records case id's answer a in pass.
func (s *TextSamples) Add(pass int, id string, a TextAnswer) {
	s.passes = max(s.passes, pass+1)
	s.wall = append(s.wall, a.WallMS)
	s.prefillTokens = append(s.prefillTokens, float64(a.PrefillTokens))
	s.generated = append(s.generated, float64(a.Generated))

	if s.perCase == nil {
		s.perCase = map[string]*textCaseSamples{}
	}

	c, ok := s.perCase[id]
	if !ok {
		c = &textCaseSamples{}
		s.perCase[id] = c
		s.order = append(s.order, id)
	}

	c.prefillTokens, c.generated = a.PrefillTokens, a.Generated
	c.wall = append(c.wall, a.WallMS)

	if a.TTFTMS > 0 {
		s.ttft = append(s.ttft, a.TTFTMS)
		c.ttft = append(c.ttft, a.TTFTMS)
	}

	if a.PrefillTPS > 0 {
		s.prefill = append(s.prefill, a.PrefillTPS)
	}

	if a.ReportedPrefillTPS > 0 {
		s.reported = append(s.reported, a.ReportedPrefillTPS)
	}

	if a.DecodeTPS > 0 {
		s.decode = append(s.decode, a.DecodeTPS)
		c.decode = append(c.decode, a.DecodeTPS)
	}
}

// Set summarises the answers, pooled over the passes.
func (s *TextSamples) Set() TextSet {
	return TextSet{
		Cases: len(s.order), Passes: s.passes,
		PrefillTokens: Summarise(s.prefillTokens), Generated: Summarise(s.generated),
		TTFT: Summarise(s.ttft), PrefillTPS: Summarise(s.prefill), ReportedPrefillTPS: Summarise(s.reported),
		DecodeTPS: Summarise(s.decode), Wall: Summarise(s.wall),
	}
}

// Cases is each case's medians over the passes, in the order first added.
func (s *TextSamples) Cases() []TextCase {
	out := make([]TextCase, 0, len(s.order))

	for _, id := range s.order {
		c := s.perCase[id]
		out = append(out, TextCase{
			ID: id, PrefillTokens: c.prefillTokens, Generated: c.generated,
			TTFTMS: Summarise(c.ttft).Median, DecodeTPS: Summarise(c.decode).Median, WallMS: Summarise(c.wall).Median,
		})
	}

	return out
}

// TextSet is the corpus's first turns answered one way, every answer
// pooled over the passes.
type TextSet struct {
	Cases         int     `json:"cases"`
	Passes        int     `json:"passes"`
	PrefillTokens Summary `json:"prefill_tokens"`
	Generated     Summary `json:"generated_tokens"`
	// TTFT is the time to the first generated token, the static prefix's
	// prefill included. Go measures it. For the engine library it is
	// needle_init, which prefills the static prefix, and the
	// needle_complete call less (n − 1) / decode_tps: an upper bound that
	// also holds what follows the last token. The CLI has none
	// (docs/FINDINGS.md F-061).
	TTFT Summary `json:"ttft_ms"`
	// PrefillTPS is PrefillTokens over TTFT; ReportedPrefillTPS the
	// engine's own prefill_tps, over the turn's tokens alone.
	PrefillTPS         Summary `json:"prefill_tps"`
	ReportedPrefillTPS Summary `json:"reported_prefill_tps"`
	DecodeTPS          Summary `json:"decode_tps"`
	// Wall is one answer: on the Go side the turn from the tools to the
	// envelope, the model loaded; for the engine library needle_init and
	// needle_complete; for the engine CLI the whole process.
	Wall Summary `json:"wall_ms"`
}

// TextCase is one case's medians over the passes.
type TextCase struct {
	ID            string  `json:"id"`
	PrefillTokens int     `json:"prefill_tokens"`
	Generated     int     `json:"generated_tokens"`
	TTFTMS        float64 `json:"ttft_ms"`
	DecodeTPS     float64 `json:"decode_tps"`
	WallMS        float64 `json:"wall_ms"`
}

// TextGoResults is the Go side of the text bench, written by needle-bench.
// Run.Inputs fingerprints needle3.cact as the model and the case corpus
// (cases.json) in place of the clips.
type TextGoResults struct {
	Run Run `json:"run"`
	// Safetensors is the SHA-256 of needle3.safetensors, which the
	// safetensors mode loads; Tokenizer that of the tokenizer the Go side
	// encodes with (the engine's is inside needle3.cact, Run.Inputs.Model).
	Safetensors string `json:"safetensors_sha256"`
	Tokenizer   string `json:"tokenizer_sha256"`
	// Responses is the SHA-256 of the engine's recorded answers the
	// function_calls are counted against (responses.json).
	Responses  string       `json:"responses_sha256"`
	GOMAXPROCS int          `json:"gomaxprocs"`
	WarmUp     int          `json:"warm_up"` // untimed passes over the cases
	Modes      []TextGoMode `json:"modes"`
}

// The Go side's modes (needle-bench).
const (
	TextModeSafetensors = "safetensors"
	TextModePacked      = "packed"
	TextModeInt8        = "int8"
	TextModeInt8LUT     = "int8-lut"
)

// TextModes are the Go side's modes in the report's column order, with
// their labels.
var TextModes = []struct{ Name, Label string }{
	{TextModeSafetensors, "safetensors float32"},
	{TextModePacked, "packed exact"},
	{TextModeInt8, "packed + int8 + window"},
	{TextModeInt8LUT, "packed LUT + int8 + window"},
}

// TextGoMode is one way of running the text model.
type TextGoMode struct {
	Mode string `json:"mode"`
	// Load is the tokenizer, the weights and the decoder, timed in the
	// timing process.
	Load    Summary    `json:"load_ms"`
	Answers TextSet    `json:"answers"`
	Cases   []TextCase `json:"cases"`
	// FunctionCallsEqual counts the first turns whose function_calls equal
	// the engine's recorded ones (responses.json), in the warm-up pass.
	FunctionCallsEqual int `json:"function_calls_equal"`
	// MaxRSSLoadMiB and MaxRSSPeakMiB are ru_maxrss of a fresh process
	// after the load and after one pass over the cases; HeapLoadMiB is its
	// live Go heap after the load.
	MaxRSSLoadMiB float64 `json:"max_rss_load_mib"`
	MaxRSSPeakMiB float64 `json:"max_rss_peak_mib"`
	HeapLoadMiB   float64 `json:"heap_load_mib"`
}

// TextEngineResults is the engine side, written by `oracle/text bench`.
type TextEngineResults struct {
	Run      Run    `json:"run"`
	Version  string `json:"version"`
	Revision string `json:"revision"`
	Threads  string `json:"threads"` // the CLI's --threads
	// Responses is the SHA-256 of the recorded answers the engine's
	// function_calls are checked against (responses.json), as on the Go
	// side.
	Responses string `json:"responses_sha256"`
	WarmUp    int    `json:"warm_up"` // untimed passes over the cases
	// CLI is the needle binary, one process per answer.
	CLI TextEngineCLI `json:"cli"`
	// Library is libneedle3 in the oracle's process, on its default
	// threads.
	Library TextEngineLibrary `json:"library"`
}

// TextEngineCLI is the needle binary's numbers: Answers.Wall is the whole
// process, start-up and model load included.
type TextEngineCLI struct {
	Answers TextSet    `json:"answers"`
	Cases   []TextCase `json:"cases"`
	// PeakRAM is the engine's own peak_ram_mb per answer, MaxRSS the
	// process's ru_maxrss in MiB.
	PeakRAM Summary `json:"peak_ram_mb"`
	MaxRSS  Summary `json:"max_rss_mib"`
	// FunctionCallsEqual counts the warm-up pass's answers whose
	// function_calls equal responses.json's.
	FunctionCallsEqual int `json:"function_calls_equal"`
}

// TextEngineLibrary is libneedle3's numbers: Answers.Wall is needle_init
// and needle_complete. Answers are cold, the static prefix prefilled in
// the call as the CLI and the Go side do; Cached answers the same turn
// again on the same system text and tools, whose prefill the library
// keeps (docs/FINDINGS.md F-061).
type TextEngineLibrary struct {
	Load     Summary    `json:"load_ms"`     // needle_load on the .cact bytes in memory
	Init     Summary    `json:"init_ms"`     // needle_init: the system text and the tools, cold
	Complete Summary    `json:"complete_ms"` // needle_complete: the turn, cold
	Answers  TextSet    `json:"answers"`
	Cases    []TextCase `json:"cases"`
	Cached   TextSet    `json:"cached"`
	// FunctionCallsEqual and CachedFunctionCallsEqual count the warm-up
	// pass's answers whose function_calls equal responses.json's.
	FunctionCallsEqual       int `json:"function_calls_equal"`
	CachedFunctionCallsEqual int `json:"cached_function_calls_equal"`
}

// mode is the Go mode named name, or nil.
func (g *TextGoResults) mode(name string) *TextGoMode {
	i := slices.IndexFunc(g.Modes, func(m TextGoMode) bool { return m.Mode == name })
	if i < 0 {
		return nil
	}

	return &g.Modes[i]
}

// WriteTextReport renders reportPath from the text results at goPath and
// enginePath; a side whose file does not exist is left out. needle-bench
// and `oracle/text bench` call it with the paths they wrote.
func WriteTextReport(goPath, enginePath, reportPath string) error {
	g, e, err := readSides[TextGoResults, TextEngineResults](goPath, enginePath)
	if err != nil {
		return err
	}

	return writeRendered(reportPath, func(w io.Writer) error {
		return RenderText(w, g, e, Paths{Go: goPath, Engine: enginePath, Report: reportPath})
	})
}

// FunctionCallsEqual reports whether two envelopes' function_calls are the
// same JSON values: names, arguments and their values, in order. An absent
// list equals an empty one; JSON that does not parse equals nothing.
func FunctionCallsEqual(a, b json.RawMessage) bool {
	x, okA := canonicalCalls(a)
	y, okB := canonicalCalls(b)

	return okA && okB && bytes.Equal(x, y)
}

// canonicalCalls re-encodes a function_calls list with sorted object keys
// and normalised numbers, "[]" for an absent one.
func canonicalCalls(raw json.RawMessage) ([]byte, bool) {
	var v any

	if len(bytes.TrimSpace(raw)) > 0 {
		err := json.Unmarshal(raw, &v)
		if err != nil {
			return nil, false
		}
	}

	if v == nil {
		v = []any{}
	}

	out, err := json.Marshal(v)

	return out, err == nil
}

// FileSHA256 is the SHA-256 of the file at path, in hex: the text bench's
// fingerprints of the model files and the corpus.
func FileSHA256(path string) (string, error) { return fileSHA256(path) }

// SameTextInputs reports whether the text bench's two sides answered the
// same inputs: the model, the cases and the recorded answers both count
// function_calls against (docs/FINDINGS.md F-061).
func SameTextInputs(g *TextGoResults, e *TextEngineResults) bool {
	return SameInputs(g.Run.Inputs, e.Run.Inputs) && g.Responses != "" && g.Responses == e.Responses
}
