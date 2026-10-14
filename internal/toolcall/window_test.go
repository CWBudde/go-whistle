package toolcall

import (
	"encoding/json"
	"math"
	"strconv"
	"testing"

	"github.com/cwbudde/go-whistle/internal/text"
)

const engineWindow = "../../testdata/text/engine/window.json"

// loadWindowModel is the model the window probes run on: loadModel's
// under the int8 numerics, which carry the KV window; otherwise the
// dequantised (float) or packed needle3.cact with the KV window of 256
// and the static prefix as its sink added, in float32 (docs/FINDINGS.md
// F-060).
func loadWindowModel(t *testing.T) *text.Model {
	t.Helper()

	n := testNumerics(t)
	if n == numericsInt8 {
		return loadModel(t, needle3Cact)
	}

	requireModel(t, needle3Cact)

	src, err := text.OpenWith(needle3Cact, text.OpenOptions{Packed: n == numericsPacked})
	if err != nil {
		t.Fatal(err)
	}

	end, ok := loadTokenizer(t).PieceID(toolsEnd)
	if !ok {
		t.Fatalf("no piece %s", toolsEnd)
	}

	cfg := src.TextConfig()

	m, err := text.LoadWith(src, cfg, text.LoadOptions{KVWindow: cfg.KVWindow, SinkToken: end})
	if err != nil {
		t.Fatal(err)
	}

	return m
}

// windowCut are the window probes whose fact the engine's KV window cuts
// from view: the timer or the message comes first in a query some 290 and
// 520 tokens long, and the engine answers the filler with a weather call
// it makes up, the same for both. With the window and the sink the model
// writes the same calls.
var windowCut = []string{"timer_first_22/0", "message_first_22/0", "timer_first_40/0", "message_first_40/0"}

// TestWindowMatchesEngine decodes the window probes
// (testdata/text/engine/window.json, `just oracle-text window`), whose
// decisive fact lies 90 to 1100 filler tokens before the answer, turn by
// turn with the engine's KV window of 256 and the static prefix up to
// </tools> as its sink (docs/FINDINGS.md F-060): the probes whose fact
// the window cuts have the engine's calls, and the counts of calls,
// reasoning and confidence (within 0.02) equal to the engine's hold their
// floors. Without the window, or without the sink, the calls of the cut
// probes differ from the engine's (F-060's table).
func TestWindowMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the window probes")
	}

	p := loadTokenizer(t)
	m := loadWindowModel(t)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var fx struct {
		System string `json:"system"`
		Probes []struct {
			ID      string           `json:"id"`
			Tools   string           `json:"tools"`
			Turns   []string         `json:"turns"`
			Answers []engineEnvelope `json:"answers"`
		} `json:"probes"`
	}

	readJSON(t, engineWindow, &fx)

	equal := map[string]bool{}

	var calls, reasoning, conf, total int

	for _, pr := range fx.Probes {
		for i, turn := range decodeTurns(t, d, m, p, fx.System, []byte(pr.Tools), pr.Turns, 512) {
			key := pr.ID + "/" + strconv.Itoa(i)
			want := pr.Answers[i]
			total++

			js, err := json.Marshal(turn.env)
			if err != nil {
				t.Fatal(err)
			}

			var got engineEnvelope

			err = json.Unmarshal(js, &got)
			if err != nil {
				t.Fatal(err)
			}

			diff := diffCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...))
			same := diff == ""
			think := thinkOf(turn.a.Text) == derefString(want.Reasoning)
			near := math.Abs(got.Confidence-want.Confidence) <= 0.02
			equal[key] = same

			for _, c := range []struct {
				ok bool
				n  *int
			}{{same, &calls}, {think, &reasoning}, {near, &conf}} {
				if c.ok {
					*c.n++
				}
			}

			t.Logf("%-20s reasoning equal %-5v confidence %.4f engine %.4f  %q", key, think, got.Confidence,
				want.Confidence, diff)
		}
	}

	for _, key := range windowCut {
		if !equal[key] {
			t.Errorf("%s: the calls differ from the engine's, which the window explains", key)
		}
	}

	floors := windowFloors[testNumerics(t)]
	for _, c := range []struct {
		name       string
		got, floor int
	}{{"calls equal", calls, floors[0]}, {"reasoning equal", reasoning, floors[1]}, {"confidence within 0.02", conf, floors[2]}} {
		t.Logf("%s numerics: %s on %d of %d", testNumerics(t), c.name, c.got, total)

		if c.got < c.floor {
			t.Errorf("%s on %d of %d, want at least %d", c.name, c.got, total, c.floor)
		}
	}
}

// windowFloors are TestWindowMatchesEngine's floors on the calls, the
// reasoning and the confidence equal to the engine's, by numerics: the
// counts of 22 measured on arm64 (docs/FINDINGS.md F-060), less one under
// int8 (numericsInt8's floors).
var windowFloors = map[string][3]int{
	numericsFloat:  {17, 11, 13},
	numericsPacked: {17, 11, 13},
	numericsInt8:   {16, 10, 11}, // 17, 11, 12
}

func derefString(s *string) string {
	if s == nil {
		return ""
	}

	return *s
}
