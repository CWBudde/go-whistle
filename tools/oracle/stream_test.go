//go:build oracle

package main

import (
	"encoding/json"
	"slices"
	"testing"
)

// TestReceivedText pins the engine's "received" for sample counts where n · (1/16000) and n / 16000 print
// differently, as needle_stream_transcribe_process returned them.
func TestReceivedText(t *testing.T) {
	for n, want := range map[int]string{
		0: "0.00", 48080: "3.00", 50160: "3.14", 50480: "3.16", 51120: "3.20", 54160: "3.39", 56080: "3.50",
		114240: "7.14", 576080: "36.01", 754480: "47.16",
	} {
		if got := receivedText(n); got != want {
			t.Errorf("receivedText(%d) = %s, want %s", n, got, want)
		}
	}
}

func TestCloseReceived(t *testing.T) {
	for _, c := range []struct {
		received string
		n        int
		want     bool
	}{
		{"42.15", 674480, true},
		{"42.16", 674480, true},
		{"42.17", 674480, false},
		{"3.00", 48000, true},
		{"2.99", 48000, false},
		{"x", 0, false},
	} {
		if got := closeReceived(json.Number(c.received), c.n); got != c.want {
			t.Errorf("closeReceived(%s, %d) = %v, want %v", c.received, c.n, got, c.want)
		}
	}
}

func TestCallLengths(t *testing.T) {
	for _, c := range []struct {
		spec streamSpec
		n    int
		want []int
	}{
		{streamSpec{ChunkSamples: 16000}, 40000, []int{16000, 16000, 8000}},
		{streamSpec{ChunkSamples: 16000}, 32000, []int{16000, 16000}},
		{streamSpec{ChunkSamples: 16000}, 0, nil},
		{streamSpec{CallSamples: []int{}}, 0, []int{}},
		{streamSpec{CallSamples: []int{48000, 0, 0}}, 48000, []int{48000, 0, 0}},
	} {
		if got := c.spec.callLengths(c.n); !slices.Equal(got, c.want) {
			t.Errorf("%+v.callLengths(%d) = %v, want %v", c.spec, c.n, got, c.want)
		}
	}
}

func TestStreamAudio(t *testing.T) {
	pcm := map[string][]float32{"a": {1, 2}, "b": {3}}

	got, err := streamAudio(streamSpec{Clips: []string{"a", "b", "a"}, LeadSamples: 2, GapSamples: 1}, pcm)
	if err != nil {
		t.Fatal(err)
	}

	if want := []float32{0, 0, 1, 2, 0, 3, 0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("streamAudio = %v, want %v", got, want)
	}

	_, err = streamAudio(streamSpec{Clips: []string{"c"}}, pcm)
	if err == nil {
		t.Error("streamAudio accepted a clip outside the corpus")
	}
}

// TestStreamSpecs checks the recording plan: unique names, every clip at 1, 0.5 and 2 s, and long streams that
// exceed the engine's 30 s even with clips of 4 s.
func TestStreamSpecs(t *testing.T) {
	clips := append(append([]string{}, streamLongClean...), streamLongOther...)
	clips = append(clips, streamForced[0].clip, streamForced[2].clip)

	names := map[string]bool{}
	perSet := map[string]int{}

	for _, s := range streamSpecs(clips) {
		if names[s.Name] {
			t.Errorf("duplicate run %s", s.Name)
		}

		names[s.Name] = true
		perSet[s.Set]++

		if s.Set != streamSetLong {
			continue
		}

		n := s.LeadSamples + s.GapSamples*max(0, len(s.Clips)-1) + 4*sampleRate*len(s.Clips)
		if n <= streamLimit {
			t.Errorf("%s: %d samples, not longer than 30 s", s.Name, n)
		}
	}

	want := map[string]int{
		streamSetClips: len(clips), streamSetChunks: 2 * len(clips), streamSetLanguage: len(streamForced),
		streamSetLong: 3, streamSetEdge: 5,
	}
	for set, n := range want {
		if perSet[set] != n {
			t.Errorf("set %s: %d runs, want %d", set, perSet[set], n)
		}
	}
}

func TestStreamCallParse(t *testing.T) {
	out := `{"text":"a b","words":[{"word":"a","start":0.00,"end":0.10,"probability":0.900},` +
		`{"word":"b","start":0.10,"end":0.20,"probability":0.800}],"pending":"c","language":"en",` +
		`"received":1.00,"pass_ms":5.1}`

	var c streamCall

	err := c.parse(libResult{RC: 2, out: []byte(out)}, sampleRate, true, true)
	if err != nil {
		t.Fatal(err)
	}

	want := `{"language":"en","pending":"c","received":1.00,"text":"a b","words":[{"word":"a","start":0.00,` +
		`"end":0.10,"probability":0.900},{"word":"b","start":0.10,"end":0.20,"probability":0.800}]}`
	if string(c.Out) != want {
		t.Errorf("out = %s\nwant %s", c.Out, want)
	}

	for name, bad := range map[string]struct {
		rc      int32
		fed     int
		process bool
	}{
		"rc is not the word count": {1, sampleRate, true},
		"received is off":          {2, 2 * sampleRate, true},
		"pass_ms on a stop":        {2, sampleRate, false},
	} {
		var c streamCall

		err := c.parse(libResult{RC: bad.rc, out: []byte(out)}, bad.fed, true, bad.process)
		if err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}
