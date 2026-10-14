//go:build oracle

package main

import (
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/bench"
)

// TestParseTiming reads an engine answer's timing fields, which the
// fixtures drop and the bench keeps.
func TestParseTiming(t *testing.T) {
	out := []byte(`{"type":"call","success":true,"error":null,"error_code":null,"reason":null,` +
		`"function_calls":[{"name":"set_timer","arguments":{"minutes":10,"label":"Pasta"}}],"suppressed_calls":[],` +
		`"reasoning":"r","confidence":0.8173,"prefill_tps":3946.6,"decode_tps":1764.3,"peak_ram_mb":94.3,` +
		`"validation":{"ungrounded":[],"negation":false}}` + "\n")

	got, err := parseTiming(out)
	if err != nil {
		t.Fatal(err)
	}

	if got.PrefillTPS != 3946.6 || got.DecodeTPS != 1764.3 || got.PeakRAMMB != 94.3 ||
		!bench.FunctionCallsEqual(got.Calls, []byte(`[{"name":"set_timer","arguments":{"label":"Pasta","minutes":10}}]`)) {
		t.Errorf("%+v", got)
	}

	for _, bad := range []string{
		`{"function_calls":[]}`, // no timings
		`{"function_calls":[],"prefill_tps":0,"decode_tps":0,"peak_ram_mb":90}`,
		`not json`,
	} {
		_, err := parseTiming([]byte(bad))
		if err == nil {
			t.Errorf("%s: no error", bad)
		}
	}

	// A one-token answer has no decode rate; that is not an error.
	_, err = parseTiming([]byte(`{"function_calls":[],"prefill_tps":900,"decode_tps":0,"peak_ram_mb":90}`))
	if err != nil {
		t.Errorf("no decode rate: %v", err)
	}
}

// TestBenchAnswer takes the engine's TTFT as needle_init and the
// needle_complete call less its decode clock's share.
func TestBenchAnswer(t *testing.T) {
	c := benchCase{prefillTokens: 330, generated: 31}
	tm := engineTimingText{PrefillTPS: 3300, DecodeTPS: 1500}

	// needle_init 40 ms, needle_complete 80 ms of which 20 ms decode.
	a := c.answer(tm, 40+outsideDecode(80, c.generated, tm.DecodeTPS), 120, true)
	if a.TTFTMS != 100 || a.PrefillTokens != 330 || a.Generated != 31 || a.PrefillTPS != 3300 ||
		a.ReportedPrefillTPS != 3300 || a.DecodeTPS != 1500 || a.WallMS != 120 {
		t.Errorf("%+v", a)
	}

	// A cached prefix was not prefilled: no prefill rate of the answer's
	// own, the engine's kept.
	a = c.answer(tm, 0.2+outsideDecode(80, c.generated, tm.DecodeTPS), 80.2, false)
	if a.PrefillTPS != 0 || a.ReportedPrefillTPS != 3300 {
		t.Errorf("cached: %+v", a)
	}

	// Without a decode rate or a second token the call is all there is.
	for _, c := range []struct {
		n   int
		tps float64
	}{{1, 1500}, {31, 0}} {
		if got := outsideDecode(80, c.n, c.tps); got != 80 {
			t.Errorf("outsideDecode(80, %d, %v) = %v", c.n, c.tps, got)
		}
	}
}

func TestBenchConfigCheck(t *testing.T) {
	ok := benchConfig{warmUp: 1, passes: 5, loads: 5}

	err := ok.check()
	if err != nil {
		t.Errorf("defaults: %v", err)
	}

	for _, c := range []benchConfig{
		{warmUp: -1, passes: 1, loads: 1}, {warmUp: 0, passes: 0, loads: 1}, {warmUp: 0, passes: 1, loads: 0},
	} {
		err := c.check()
		if err == nil || !strings.Contains(err.Error(), "want") {
			t.Errorf("%+v: %v", c, err)
		}
	}
}
