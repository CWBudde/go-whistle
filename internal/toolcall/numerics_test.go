package toolcall

import (
	"os"
	"testing"

	"github.com/cwbudde/go-whistle/internal/text"
)

// numericsEnv selects the numerics the engine tests run the text model in
// (docs/FINDINGS.md F-060):
//
//   - unset or "float": the model each test names, the dequantised
//     needle3.cact or the float needle3.safetensors, in float32; every
//     test holds the expectations it lists turn by turn.
//   - "packed": needle3.cact's CQ records packed (text.OpenOptions.Packed,
//     F-059), which give the dequantised model's numerics; the tests hold
//     the same expectations.
//   - "int8": the packed needle3.cact with the engine's numerics
//     (text.LoadOptions: Int8, the KV window of 256 and the static prefix
//     up to </tools> as its sink). The per-turn lists were set on float32,
//     so a turn that differs from them is logged, not an error (listedf),
//     and each test holds floors on its counts instead (int8Floor), as
//     Whistle's WHISTLE_TEST_INT8 does (F-041).
//   - "safetensors": every test on the float needle3.safetensors in
//     float32, the window test with the KV window and its sink added as
//     under "float". This is the float weights' side of 11.5's Accept
//     (docs/FINDINGS.md F-061): a turn that differs from the lists is
//     logged as under int8, and no floor applies, since the run measures
//     the float weights against the int8 run rather than holding a bound.
const numericsEnv = "NEEDLE_TEST_NUMERICS"

const (
	numericsFloat       = "float"
	numericsPacked      = "packed"
	numericsInt8        = "int8"
	numericsSafetensors = "safetensors"
)

// testNumerics is the numerics numericsEnv selects.
func testNumerics(t *testing.T) string {
	t.Helper()

	switch v := os.Getenv(numericsEnv); v {
	case "", numericsFloat:
		return numericsFloat
	case numericsPacked, numericsInt8, numericsSafetensors:
		return v
	default:
		t.Fatalf("%s=%q: want float, packed, int8 or safetensors", numericsEnv, v)

		return ""
	}
}

// loadNumerics loads the packed needle3.cact for the packed and int8
// numerics and needle3.safetensors for the safetensors numerics; the float
// numerics keep loadModel's path.
func loadNumerics(t *testing.T, numerics string) *text.Model {
	t.Helper()

	if numerics == numericsSafetensors {
		return loadFloat(t, needle3Safetensors)
	}

	requireModel(t, needle3Cact)

	src, err := text.OpenWith(needle3Cact, text.OpenOptions{Packed: true})
	if err != nil {
		t.Fatal(err)
	}

	cfg := src.TextConfig()

	var opts text.LoadOptions

	if numerics == numericsInt8 {
		end, ok := loadTokenizer(t).PieceID(toolsEnd)
		if !ok {
			t.Fatalf("no piece %s", toolsEnd)
		}

		opts = text.LoadOptions{Int8: true, KVWindow: cfg.KVWindow, SinkToken: end}
	}

	m, err := text.LoadWith(src, cfg, opts)
	if err != nil {
		t.Fatal(err)
	}

	return m
}

// listedf reports a turn whose outcome differs from what a test lists for
// it: an error, but a log line under the int8 and the safetensors
// numerics, whose tests hold floors on their counts or only report them.
func listedf(t *testing.T, format string, args ...any) {
	t.Helper()

	if n := testNumerics(t); n == numericsInt8 || n == numericsSafetensors {
		t.Logf(n+", unlike the float32 list: "+format, args...)

		return
	}

	t.Errorf(format, args...)
}

// logCount logs a count of total under the numerics of the run, the line
// the numerics comparisons are read from (docs/FINDINGS.md F-060, F-061).
func logCount(t *testing.T, name string, got, total int) {
	t.Helper()

	t.Logf("%s numerics: %s on %d of %d", testNumerics(t), name, got, total)
}

// functionCallsEqual reports whether an envelope dispatches the engine's
// function_calls, names and arguments, whatever it suppresses: the count
// 11.5's Accept compares (docs/FINDINGS.md F-061). The tests' own "calls
// equal" counts compare the dispatched and the suppressed calls together.
func functionCallsEqual(got, want []call) bool {
	return diffCalls(got, want) == ""
}

// int8Floor logs a count and, under the int8 numerics, requires at least
// floor of total.
func int8Floor(t *testing.T, name string, got, total, floor int) {
	t.Helper()

	logCount(t, name, got, total)

	if testNumerics(t) == numericsInt8 && got < floor {
		t.Errorf("int8: %s on %d of %d, want at least %d", name, got, total, floor)
	}
}

// The floors of the engine tests under the int8 numerics: the counts
// measured on arm64 (M5 Pro; docs/FINDINGS.md F-060), less one or two
// turns (about 2 %) for another architecture's rounding at the near-ties.
// The float32 numerics give, as listed turn by turn: reasoning 21 of 25
// (TestDecodeMatchesEngine); calls 36, reasoning 29 and confidence 27 of
// 36 (TestEnvelopeMatchesEngine); calls 159 and the split 158 of 171
// (TestValidationMatchesEngine); 194 of 204 (TestGatesMatchesEngine, 193
// before docs/FINDINGS.md F-063); calls 85 and reasoning 77 of 90
// (TestRepairMatchesEngine); 10 of 15 (TestGrammarMatchesEngine);
// confidence 106 of the 135 turns decoded as the engine's
// (TestConfidenceMatchesEngine). The confidence counts are the min with
// the decode probability's (docs/FINDINGS.md F-062).
const (
	int8DecodeReasoning    = 20  // 21 of 25
	int8EnvelopeCalls      = 34  // 35 of 36
	int8EnvelopeReasoning  = 32  // 33 of 36
	int8EnvelopeConfidence = 31  // 32 of 36
	int8ValidationCalls    = 160 // 162 of 171
	int8ValidationSplit    = 159 // 161 of 171
	int8GatesEqual         = 192 // 194 of 204
	int8RepairCalls        = 84  // 85 of 90
	int8RepairReasoning    = 76  // 78 of 90
	int8GrammarCalls       = 8   // 9 of 15
	int8ConfidenceWithin   = 126 // 128 of the 148 turns decoded as the engine's
)
