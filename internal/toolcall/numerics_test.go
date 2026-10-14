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
const numericsEnv = "NEEDLE_TEST_NUMERICS"

const (
	numericsFloat  = "float"
	numericsPacked = "packed"
	numericsInt8   = "int8"
)

// testNumerics is the numerics numericsEnv selects.
func testNumerics(t *testing.T) string {
	t.Helper()

	switch v := os.Getenv(numericsEnv); v {
	case "", numericsFloat:
		return numericsFloat
	case numericsPacked, numericsInt8:
		return v
	default:
		t.Fatalf("%s=%q: want float, packed or int8", numericsEnv, v)

		return ""
	}
}

// loadNumerics loads the packed needle3.cact for the packed and int8
// numerics; the float numerics keep loadModel's path.
func loadNumerics(t *testing.T, numerics string) *text.Model {
	t.Helper()

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
// it: an error, but a log line under the int8 numerics, whose tests hold
// floors on their counts instead.
func listedf(t *testing.T, format string, args ...any) {
	t.Helper()

	if testNumerics(t) == numericsInt8 {
		t.Logf("int8, unlike the float32 list: "+format, args...)

		return
	}

	t.Errorf(format, args...)
}

// int8Floor logs a count and, under the int8 numerics, requires at least
// floor of total.
func int8Floor(t *testing.T, name string, got, total, floor int) {
	t.Helper()

	t.Logf("%s numerics: %s on %d of %d", testNumerics(t), name, got, total)

	if testNumerics(t) == numericsInt8 && got < floor {
		t.Errorf("int8: %s on %d of %d, want at least %d", name, got, total, floor)
	}
}

// The floors of the engine tests under the int8 numerics: the counts
// measured on arm64 (M5 Pro; docs/FINDINGS.md F-060), less one or two
// turns (about 2 %) for another architecture's rounding at the near-ties.
// The float32 numerics give, as listed turn by turn: reasoning 21 of 25
// (TestDecodeMatchesEngine); calls 36, reasoning 29 and confidence 26 of
// 36 (TestEnvelopeMatchesEngine); calls 159 and the split 158 of 171
// (TestValidationMatchesEngine); 193 of 204 (TestGatesMatchesEngine);
// calls 85 and reasoning 77 of 90 (TestRepairMatchesEngine); 10 of 15
// (TestGrammarMatchesEngine).
const (
	int8DecodeReasoning    = 20  // 21 of 25
	int8EnvelopeCalls      = 34  // 35 of 36
	int8EnvelopeReasoning  = 32  // 33 of 36
	int8EnvelopeConfidence = 30  // 31 of 36
	int8ValidationCalls    = 160 // 162 of 171
	int8ValidationSplit    = 159 // 161 of 171
	int8GatesEqual         = 192 // 194 of 204
	int8RepairCalls        = 84  // 85 of 90
	int8RepairReasoning    = 76  // 78 of 90
	int8GrammarCalls       = 8   // 9 of 15
)
