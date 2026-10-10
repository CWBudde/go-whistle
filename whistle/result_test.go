package whistle

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// engineDir holds raw stdout of the engine's CLI (docs/FINDINGS.md F-027).
const engineDir = "testdata/needle-cli"

const silenceJSON = `{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}`

func engineOutputs(t *testing.T) map[string][]byte {
	t.Helper()

	paths, err := filepath.Glob(filepath.Join(engineDir, "*.stdout"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no engine outputs in %s: %v", engineDir, err)
	}

	out := make(map[string][]byte, len(paths))

	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}

		line, ok := bytes.CutSuffix(raw, []byte("\n"))
		if !ok || bytes.ContainsRune(line, '\n') {
			t.Fatalf("%s: want one JSON line ending in a newline", p)
		}

		out[filepath.Base(p)] = line
	}

	return out
}

func TestResultRoundTripsEngineOutput(t *testing.T) {
	for name, raw := range engineOutputs(t) {
		var r Result

		err := json.Unmarshal(raw, &r)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		got, err := r.MarshalJSON()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		if !bytes.Equal(got, raw) {
			t.Errorf("%s: round trip differs\n got %s\nwant %s", name, got, raw)
		}

		if strings.Contains(name, ".words.") != (r.Words != nil) {
			t.Errorf("%s: Words nil = %v", name, r.Words == nil)
		}
	}
}

func TestSilenceJSON(t *testing.T) {
	got, err := Silence().MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	if string(got) != silenceJSON {
		t.Errorf("Silence() = %s, want %s", got, silenceJSON)
	}

	// The engine's own silence outputs, with and without timestamps.
	outs := engineOutputs(t)
	if string(outs["silence-2s.words.stdout"]) != silenceJSON {
		t.Errorf("recorded silence = %s, want %s", outs["silence-2s.words.stdout"], silenceJSON)
	}

	got, err = Result{}.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(got, outs["silence-2s.stdout"]) {
		t.Errorf("Result{} = %s, want %s", got, outs["silence-2s.stdout"])
	}
}

func TestResultMarshalFormat(t *testing.T) {
	r := Result{
		Text:     "a \"b\" \\ <c> & é\n\t\x01\xff",
		Language: "en",
		Words: []Word{
			{Text: "a", Start: 400 * time.Millisecond, End: 560 * time.Millisecond, Probability: 0.9936},
			{Text: "b", Start: 0, End: 30 * time.Second, Probability: 1},
		},
		TTFT:      7449 * time.Microsecond,
		DecodeTPS: 1433.66,
	}

	got, err := r.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	want := `{"text":"a \"b\" \\ <c> & é\n\t\u0001` + "�" + `","language":"en",` +
		`"words":[{"word":"a","start":0.40,"end":0.56,"probability":0.994},` +
		`{"word":"b","start":0.00,"end":30.00,"probability":1.000}],"ttft_ms":7.4,"decode_tps":1433.7}`
	if string(got) != want {
		t.Errorf("MarshalJSON\n got %s\nwant %s", got, want)
	}

	if !json.Valid(got) {
		t.Error("output is not valid JSON")
	}

	var back Result

	err = json.Unmarshal(got, &back)
	if err != nil {
		t.Fatal(err)
	}

	if back.Text != strings.ToValidUTF8(r.Text, "�") || back.Language != r.Language || len(back.Words) != 2 {
		t.Errorf("unmarshal = %+v", back)
	}
}

func TestResultMarshalRejectsNaN(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1)} {
		_, err := Result{DecodeTPS: v}.MarshalJSON()
		if err == nil {
			t.Errorf("DecodeTPS %v: no error", v)
		}
	}
}

func TestResultUnmarshalRejectsUnknownField(t *testing.T) {
	var r Result

	err := json.Unmarshal([]byte(`{"text":"","language":"","ttft_ms":0.0,"decode_tps":0.0,"x":1}`), &r)
	if err == nil {
		t.Error("unknown field accepted")
	}
}

func TestClock(t *testing.T) {
	base := time.Unix(0, 0)
	ticks := []time.Duration{0, 30 * time.Millisecond, 40 * time.Millisecond, 50 * time.Millisecond, 60 * time.Millisecond}
	i := 0
	now := func() time.Time {
		v := base.Add(ticks[i])
		i++

		return v
	}

	c := newClock(now)
	if c.ttft() != 0 || c.tps() != 0 {
		t.Errorf("before any token: ttft %v, tps %v", c.ttft(), c.tps())
	}

	c.token() // the first token, at 30 ms

	if c.ttft() != 30*time.Millisecond || c.tps() != 0 {
		t.Errorf("after one token: ttft %v, tps %v", c.ttft(), c.tps())
	}

	c.token()
	c.token()
	c.token() // three more in 30 ms

	if c.ttft() != 30*time.Millisecond || math.Abs(c.tps()-100) > 1e-9 {
		t.Errorf("ttft %v, tps %v; want 30ms, 100", c.ttft(), c.tps())
	}
}

// The engine's CLI on 480,001 samples prints "needle: audio limit is 30 s".
func TestTooLongIsTheEngineText(t *testing.T) {
	stderr, err := os.ReadFile(filepath.Join(engineDir, "limit-480001.stderr"))
	if err != nil {
		t.Fatal(err)
	}

	if !strings.HasSuffix(string(stderr), ": "+ErrTooLong.Error()+"\n") {
		t.Errorf("engine stderr %q does not end in %q", stderr, ErrTooLong.Error())
	}
}

func TestCheckLength(t *testing.T) {
	err := checkLength(MaxSamples)
	if err != nil {
		t.Errorf("%d samples: %v", MaxSamples, err)
	}

	err = checkLength(MaxSamples + 1)
	if err == nil || !strings.Contains(err.Error(), "30 s") {
		t.Errorf("%d samples: %v, want an error naming 30 s", MaxSamples+1, err)
	}
}
