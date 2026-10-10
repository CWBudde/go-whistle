//go:build oracle

package main

import (
	"strings"
	"testing"
)

func TestBenchConfigCheck(t *testing.T) {
	ok := benchConfig{warmUp: 0, runs: 1, long: 1, loads: 1}
	if err := ok.check(); err != nil {
		t.Errorf("the smallest valid run: %v", err)
	}

	for name, c := range map[string]benchConfig{
		"no runs":         {warmUp: 1, runs: 0, long: 1, loads: 1},
		"negative runs":   {warmUp: 1, runs: -1, long: 1, loads: 1},
		"no 30 s runs":    {warmUp: 1, runs: 1, long: 0, loads: 1},
		"no loads":        {warmUp: 1, runs: 1, long: 1, loads: 0},
		"negative warmup": {warmUp: -1, runs: 1, long: 1, loads: 1},
	} {
		err := c.check()
		if err == nil || !strings.Contains(err.Error(), "want") {
			t.Errorf("%s: %v", name, err)
		}
	}
}
