package bench

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Default paths, relative to the repository root where the just recipes
// run.
const (
	GoResultsPath     = "bench/results/go.json"
	EngineResultsPath = "bench/results/engine.json"
	ReportPath        = "bench/RESULTS.md"
)

// Run describes one benchmark run: when, on which commit and machine, and
// how busy the machine was.
type Run struct {
	Date         string  `json:"date"`   // RFC 3339, local time
	Commit       string  `json:"commit"` // git describe of the working tree
	Machine      Machine `json:"machine"`
	Inputs       Inputs  `json:"inputs"`         // the model and the clips the run transcribed
	LoadAvgStart string  `json:"load_avg_start"` // 1, 5 and 15 min load average before the run
	LoadAvgEnd   string  `json:"load_avg_end"`   // and after it
	Seconds      float64 `json:"seconds"`        // the run's wall time
}

// GoResults is the Go side, written by whistle-bench.
type GoResults struct {
	Run        Run      `json:"run"`
	GOMAXPROCS int      `json:"gomaxprocs"` // the worker limit of the timing process
	WarmUp     int      `json:"warm_up"`    // untimed passes over the clips, and untimed 30 s runs
	Modes      []GoMode `json:"modes"`      // packed, float
	// PackedEqualsFloat counts the clips whose packed and float
	// transcripts (text and language) are equal, in the warm-up pass.
	PackedEqualsFloat int  `json:"packed_equals_float"`
	WER               *WER `json:"wer,omitempty"`
}

// GoMode is one weight mode: packed, whistle.Load's default on a .cact, or
// float, WithFloatWeights().
type GoMode struct {
	Mode string  `json:"mode"`
	Load Summary `json:"load_ms"` // whistle.Load, in the timing process
	// Clips is the 32 fixture clips, Long the first one tiled to 30 s.
	Clips Set `json:"clips"`
	Long  Set `json:"long"`
	// EqualEngine counts the clips whose transcript (text and language)
	// equals the engine's committed one (testdata/oracle).
	EqualEngine int `json:"equal_engine"`
	// MaxRSSLoadMiB and MaxRSSPeakMiB are ru_maxrss of a fresh process
	// after Load and after one 30 s Transcribe.
	MaxRSSLoadMiB float64 `json:"max_rss_load_mib"`
	MaxRSSPeakMiB float64 `json:"max_rss_peak_mib"`
}

// WER is the LibriSpeech test-clean subset (docs/FINDINGS.md F-025, F-031)
// under the manifest's wer_rule, packed against float, with the engine's
// recorded transcripts.
type WER struct {
	// Subset is SubsetSHA256 of the fixtures the run scored; only
	// AcceptSubset gives the Phase 9 Accept's verdict.
	Subset       string  `json:"subset_sha256"`
	Utterances   int     `json:"utterances"`
	Words        int     `json:"words"` // reference words
	AudioSeconds float64 `json:"audio_seconds"`
	PackedErrors int     `json:"packed_errors"`
	FloatErrors  int     `json:"float_errors"`
	EngineErrors int     `json:"engine_errors"`
	// PackedEqualsFloat counts the utterances whose packed and float
	// transcripts (text and language) are equal; *EqualsEngine the texts
	// equal to the engine's.
	PackedEqualsFloat  int     `json:"packed_equals_float"`
	PackedEqualsEngine int     `json:"packed_equals_engine"`
	FloatEqualsEngine  int     `json:"float_equals_engine"`
	PackedRTF          float64 `json:"packed_rtf"` // Σ Transcribe wall / Σ audio
	FloatRTF           float64 `json:"float_rtf"`
}

// Percent is errors over the reference words in percent.
func (w WER) Percent(errors int) float64 {
	if w.Words == 0 {
		return 0
	}

	return 100 * float64(errors) / float64(w.Words)
}

// EngineResults is the engine side, written by `oracle bench`.
type EngineResults struct {
	Run      Run    `json:"run"`
	Version  string `json:"version"`  // engine/<platform>/PIN engine_version
	Revision string `json:"revision"` // and revision
	Threads  string `json:"threads"`  // the CLI's --threads
	WarmUp   int    `json:"warm_up"`  // untimed runs per clip and mode
	// CLI is the needle binary, one process per transcription: Set.Wall
	// is the process's wall time, start-up and model load included.
	CLI EngineCLI `json:"cli"`
	// Library is libneedle3 in this process (default threads): Set.Wall is
	// the needle_transcribe call.
	Library EngineLibrary `json:"library"`
}

// EngineCLI is the needle binary's numbers.
type EngineCLI struct {
	Clips Set `json:"clips"`
	Long  Set `json:"long"`
	// EqualFixtures counts the clips whose text and language equal the
	// committed fixtures, a check that the run transcribed the same thing.
	EqualFixtures int `json:"equal_fixtures"`
	// MaxRSSClipsMiB and MaxRSSLongMiB are the largest ru_maxrss of a CLI
	// process over the clips and over the 30 s runs.
	MaxRSSClipsMiB float64 `json:"max_rss_clips_mib"`
	MaxRSSLongMiB  float64 `json:"max_rss_long_mib"`
}

// EngineLibrary is libneedle3's numbers in one process.
type EngineLibrary struct {
	Load  Summary `json:"load_ms"` // needle_load on the .cact bytes already in memory
	Clips Set     `json:"clips"`
	Long  Set     `json:"long"`
	// OutsideClock is, per timed call over the clips, the call's wall time
	// minus ttft_ms minus (n − 1) / decode_tps for the n tokens that
	// needle_transcribe returns: the time its own clock does not explain.
	OutsideClock Summary `json:"outside_clock_ms"`
}

// WriteJSON writes v to path as indented JSON with a final newline,
// creating the directory.
func WriteJSON(path string, v any) error {
	var b bytes.Buffer

	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")

	err := enc.Encode(v)
	if err != nil {
		return fmt.Errorf("encode %s: %w", path, err)
	}

	err = os.MkdirAll(filepath.Dir(path), 0o750)
	if err != nil {
		return fmt.Errorf("create %s: %w", filepath.Dir(path), err)
	}

	err = os.WriteFile(path, b.Bytes(), 0o600)
	if err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}

	return nil
}

// ReadJSON reads path into v; a missing file reports false without error.
func ReadJSON(path string, v any) (bool, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- a flag or a default path.
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}

	if err != nil {
		return false, fmt.Errorf("read %s: %w", path, err)
	}

	err = json.Unmarshal(raw, v)
	if err != nil {
		return false, fmt.Errorf("%s: %w", path, err)
	}

	return true, nil
}

// WriteReport renders reportPath from the results at goPath and
// enginePath; a side whose file does not exist is left out. Both
// whistle-bench and `oracle bench` call it with the paths they wrote.
func WriteReport(goPath, enginePath, reportPath string) error {
	g, e := new(GoResults), new(EngineResults)

	ok, err := ReadJSON(goPath, g)
	if err != nil {
		return err
	}

	if !ok {
		g = nil
	}

	ok, err = ReadJSON(enginePath, e)
	if err != nil {
		return err
	}

	if !ok {
		e = nil
	}

	var b bytes.Buffer

	err = Render(&b, g, e)
	if err != nil {
		return err
	}

	err = os.MkdirAll(filepath.Dir(reportPath), 0o750)
	if err != nil {
		return fmt.Errorf("create %s: %w", filepath.Dir(reportPath), err)
	}

	err = os.WriteFile(reportPath, b.Bytes(), 0o600) // #nosec G306 G703 -- a flag or the default path.
	if err != nil {
		return fmt.Errorf("write %s: %w", reportPath, err)
	}

	return nil
}
