//go:build oracle

package main

import (
	"archive/tar"
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type tarEntry struct {
	name string
	body []byte
}

func buildTar(t *testing.T, entries []tarEntry) io.Reader {
	t.Helper()

	var buf bytes.Buffer

	tw := tar.NewWriter(&buf)

	for _, e := range entries {
		err := tw.WriteHeader(&tar.Header{Name: e.name, Mode: 0o644, Size: int64(len(e.body))})
		if err != nil {
			t.Fatal(err)
		}

		_, err = tw.Write(e.body)
		if err != nil {
			t.Fatal(err)
		}
	}

	err := tw.Close()
	if err != nil {
		t.Fatal(err)
	}

	return &buf
}

// fakeFLAC stands in for a FLAC decoder: the body is the clip length in seconds.
func fakeFLAC(r io.Reader) ([]int16, int, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, 0, err
	}

	sec, err := strconv.ParseFloat(string(b), 64)
	if err != nil {
		return nil, 0, err
	}

	return make([]int16, int(sec*sampleRate)), sampleRate, nil
}

func TestParseTrans(t *testing.T) {
	got, err := parseTrans(strings.NewReader("1-2-0000 HELLO WORLD\n1-2-0001 AGAIN\n\n"))
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != 2 || got["1-2-0000"] != "HELLO WORLD" || got["1-2-0001"] != "AGAIN" {
		t.Errorf("parseTrans = %v", got)
	}

	_, err = parseTrans(strings.NewReader("no-transcript\n"))
	if err == nil {
		t.Error("line without transcript accepted")
	}
}

func lsEntries() []tarEntry {
	const dir = "LibriSpeech/test-clean/"

	return []tarEntry{
		{dir + "README.TXT", []byte("x")},
		{dir + "11/22/11-22-0000.flac", []byte("3")},
		{dir + "11/22/11-22-0001.flac", []byte("1.5")}, // too short
		{dir + "11/22/11-22-0002.flac", []byte("9")},   // too long
		{dir + "11/22/11-22-0003.flac", []byte("4")},
		{dir + "11/22/11-22-0004.flac", []byte("5")}, // third clip of speaker 11
		{dir + "11/22/11-22.trans.txt", []byte("11-22-0000 A B\n11-22-0003 C D\n11-22-0004 E\n")},
		{dir + "33/44/33-44-0000.flac", []byte("8")},
		{dir + "33/44/33-44.trans.txt", []byte("33-44-0000 F G\n")},
		{dir + "55/66/55-66-0000.flac", []byte("2")},
	}
}

func TestSelectLibriSpeech(t *testing.T) {
	got, err := selectLibriSpeech(buildTar(t, lsEntries()), "test-clean", 3, fakeFLAC)
	if err != nil {
		t.Fatal(err)
	}

	want := []struct{ id, text string }{
		{"ls-clean-11-22-0000", "A B"},
		{"ls-clean-11-22-0003", "C D"},
		{"ls-clean-33-44-0000", "F G"},
	}

	if len(got) != len(want) {
		t.Fatalf("selected %d clips, want %d", len(got), len(want))
	}

	for i, w := range want {
		c := got[i]
		if c.ID != w.id || c.Transcript != w.text || c.Language != "en" || c.Split != "test-clean" {
			t.Errorf("clip %d = %+v, want id %s text %q", i, c, w.id, w.text)
		}

		if c.Samples != len(c.pcm) || c.Samples < minSamples || c.Samples > maxSamples {
			t.Errorf("clip %d: %d samples", i, c.Samples)
		}
	}
}

func TestSelectLibriSpeechShortStream(t *testing.T) {
	_, err := selectLibriSpeech(buildTar(t, lsEntries()), "test-clean", 5, fakeFLAC)
	if err == nil {
		t.Error("too few clips accepted")
	}

	// 55-66-0000 qualifies, but its chapter never brings a trans.txt.
	_, err = selectLibriSpeech(buildTar(t, lsEntries()), "test-clean", 4, fakeFLAC)
	if err == nil || !strings.Contains(err.Error(), "55-66-0000") {
		t.Errorf("err = %v, want the clip without a transcript", err)
	}
}

func TestParseFleursTSV(t *testing.T) {
	tsv := "1855\ta.wav\tLöwen, Rudel.\tlöwen rudel\tx\t160000\tFEMALE\n" +
		"1856\tb.wav\tZweiter Satz.\tzweiter satz\tx\t96000\tMALE\n"

	got, err := parseFleursTSV(strings.NewReader(tsv))
	if err != nil {
		t.Fatal(err)
	}

	if got["a.wav"] != (fleursRow{sentence: "1855", transcript: "Löwen, Rudel."}) {
		t.Errorf("a.wav = %+v", got["a.wav"])
	}

	if len(got) != 2 {
		t.Errorf("rows = %d, want 2", len(got))
	}
}

func monoSeconds(t *testing.T, sec float64) []byte {
	t.Helper()

	frames := make([][]float32, int(sec*sampleRate))
	for i := range frames {
		frames[i] = []float32{0.25}
	}

	return floatWAV(t, 1, frames)
}

func TestSelectFleurs(t *testing.T) {
	rows := map[string]fleursRow{
		"a.wav": {"1", "Eins."},
		"b.wav": {"1", "Eins, nochmal."}, // same sentence as a.wav
		"c.wav": {"2", "Zu lang."},
		"d.wav": {"3", "Drei."},
		"e.wav": {"4", "Vier."},
	}

	stream := buildTar(t, []tarEntry{
		{"test/a.wav", monoSeconds(t, 3)},
		{"test/b.wav", monoSeconds(t, 3)},
		{"test/c.wav", monoSeconds(t, 9)},
		{"test/x.wav", monoSeconds(t, 3)}, // not in the tsv
		{"test/d.wav", monoSeconds(t, 2.5)},
		{"test/e.wav", monoSeconds(t, 3)},
	})

	got, err := selectFleurs(stream, "de_de", 2, rows)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != 2 || got[0].ID != "fleurs-de_de-a" || got[1].ID != "fleurs-de_de-d" {
		t.Fatalf("selected %+v", got)
	}

	c := got[1]
	if c.Language != "de" || c.Transcript != "Drei." || c.Samples != 40000 || c.pcm[0] != 8192 {
		t.Errorf("clip = %+v (first sample %d)", c, c.pcm[0])
	}
}

func TestSelectFleursShortStream(t *testing.T) {
	stream := buildTar(t, []tarEntry{{"test/a.wav", monoSeconds(t, 3)}})

	_, err := selectFleurs(stream, "de_de", 2, map[string]fleursRow{"a.wav": {"1", "Eins."}})
	if err == nil {
		t.Error("too few clips accepted")
	}
}

func TestValidID(t *testing.T) {
	for id, want := range map[string]bool{
		"ls-clean-6930-75918-0000":           true,
		"fleurs-es_419-10004059102127011984": true,
		"":                                   false,
		"../escape":                          false,
		"a/b":                                false,
		"Upper":                              false,
	} {
		if got := validID(id); got != want {
			t.Errorf("validID(%q) = %v, want %v", id, got, want)
		}
	}
}

func TestReadClipVerifiesDigest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "clip.wav")

	err := os.WriteFile(path, []byte("RIFF....WAVE"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	want, err := fileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}

	raw, err := readClip(path, want)
	if err != nil || string(raw) != "RIFF....WAVE" {
		t.Fatalf("readClip with the pinned digest = %q, %v", raw, err)
	}

	_, err = readClip(path, strings.Repeat("0", 64))
	if err == nil || !strings.Contains(err.Error(), "corpus manifest pins") {
		t.Fatalf("readClip with a stale digest: err = %v, want a digest mismatch", err)
	}
}
