//go:build oracle

package main

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func TestBenchConfigCheck(t *testing.T) {
	ok := benchConfig{warmUp: 0, runs: 1, long: 1, loads: 1}

	err := ok.check()
	if err != nil {
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

func TestBenchPCMRate(t *testing.T) {
	var b bytes.Buffer

	err := writeWAV16(&b, make([]int16, 160))
	if err != nil {
		t.Fatal(err)
	}

	pcm, err := benchPCM(b.Bytes())
	if err != nil || len(pcm) != 160 {
		t.Fatalf("16 kHz: %d samples, %v", len(pcm), err)
	}

	raw := b.Bytes()
	binary.LittleEndian.PutUint32(raw[24:], 8000)   // the fmt chunk's sample rate
	binary.LittleEndian.PutUint32(raw[28:], 8000*2) // and byte rate

	_, err = benchPCM(raw)
	if err == nil || !strings.Contains(err.Error(), "8000 Hz") {
		t.Errorf("8 kHz: %v", err)
	}

	b.Reset()

	err = writeWAV16(&b, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = benchPCM(b.Bytes())
	if err == nil || !strings.Contains(err.Error(), "no samples") {
		t.Errorf("empty clip: %v", err)
	}
}
