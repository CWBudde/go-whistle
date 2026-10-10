//go:build oracle

package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestWAVFloatRoundTrip(t *testing.T) {
	pcm := []float32{0, 1e-30, -1, 0.123456789, 3.5}

	var buf bytes.Buffer

	err := writeWAVFloat(&buf, pcm)
	if err != nil {
		t.Fatal(err)
	}

	if buf.Len() != 44+4*len(pcm) {
		t.Fatalf("wav size = %d, want %d", buf.Len(), 44+4*len(pcm))
	}

	got, rate, err := readWAV(buf.Bytes())
	if err != nil || rate != sampleRate || !slices.Equal(got, pcm) {
		t.Fatalf("readWAV = %v, %d, %v; want %v exactly", got, rate, err, pcm)
	}

	buf.Reset()

	err = writeWAVFloat(&buf, nil)
	if err != nil || buf.Len() != 44 {
		t.Fatalf("empty wav: %d bytes, %v", buf.Len(), err)
	}
}

func TestNormaliseJSON(t *testing.T) {
	got, err := normaliseJSON([]byte(`{"text":"Hi.","language":"en","words":[{"word":"Hi.","start":0.40,` +
		`"end":0.56,"probability":0.990}],"ttft_ms":9.1,"decode_tps":2018.6}`))
	if err != nil {
		t.Fatal(err)
	}

	want := `{
  "language": "en",
  "text": "Hi.",
  "words": [
    {
      "word": "Hi.",
      "start": 0.40,
      "end": 0.56,
      "probability": 0.990
    }
  ]
}
`
	if string(got) != want {
		t.Errorf("normaliseJSON =\n%s\nwant\n%s", got, want)
	}

	for _, bad := range []string{`[1]`, `{"language":"en"}`, `not json`} {
		_, err := normaliseJSON([]byte(bad))
		if err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestBundleWrite(t *testing.T) {
	dir := t.TempDir()
	b := bundle{}
	b.add("b", []byte("{\n  \"text\": \"x\",\n  \"start\": 0.40\n}\n"))
	b.add("a", []byte(`{"text":""}`))

	err := b.write(dir)
	if err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(filepath.Join(dir, transcriptsFile))
	if err != nil {
		t.Fatal(err)
	}

	want := "{\n  \"a\": {\n    \"text\": \"\"\n  },\n  \"b\": {\n    \"text\": \"x\",\n    \"start\": 0.40\n  }\n}\n"
	if string(got) != want {
		t.Errorf("transcripts.json =\n%s\nwant\n%s", got, want)
	}
}

func TestHits(t *testing.T) {
	for _, c := range []struct {
		keywords []string
		text     string
		want     int
	}{
		{[]string{"Cowley's"}, "HERE ARE SOME VERSES OF COWLEY'S TELL ME", 1},
		{[]string{"verses of Cowley's"}, "some verses of Cowley's.", 1},
		{[]string{"door"}, "She found the doors.", 0},
		{[]string{"the door"}, "She found a door, but", 0},
		{[]string{"rénale", "ont déclaré"}, "la fonction rénale, ont déclaré des", 2},
		{[]string{"Francją"}, "z Francją i", 1},
		{[]string{"Fast jeder"}, "Transportsysteme, fast jeder beklagt", 1},
	} {
		if got := hits(c.keywords, c.text); got != c.want {
			t.Errorf("hits(%q, %q) = %d, want %d", c.keywords, c.text, got, c.want)
		}
	}
}

// TestKeywordCases checks the case table against the committed corpus: the PLAN.md 6.1 spread and every kind's
// relation to the reference transcript (cmdKeywords also checks the engine side).
func TestKeywordCases(t *testing.T) {
	corpus, err := readCorpus(filepath.Join("..", "..", "testdata", "audio"))
	if err != nil {
		t.Skipf("corpus manifest: %v", err)
	}

	clips := map[string]clip{}
	for _, c := range corpus.Clips {
		clips[c.ID] = c
	}

	names, used, langs, kinds := map[string]bool{}, map[string]bool{}, map[string]bool{}, map[string]int{}

	cases := keywordCases()

	for _, kc := range cases {
		c, ok := clips[kc.clip]
		if !ok || names[kc.name] || !validID(kc.name) || !strings.HasPrefix(kc.name, c.Language+"-") {
			t.Fatalf("%s: unknown clip, duplicate or invalid name", kc.name)
		}

		names[kc.name], used[kc.clip], langs[c.Language] = true, true, true
		kinds[kc.kind]++

		rec := keywordRecord{HitsReference: hits(kc.keywords, c.Transcript)}
		if kc.kind == kwPresent {
			rec.HitsWithout = 1 // the engine side is checked by cmdKeywords
		}

		err := checkKind(kc, rec)
		if err != nil {
			t.Errorf("%s: %v", kc.name, err)
		}
	}

	if len(cases) < 24 || len(used) < 8 || len(langs) != 7 {
		t.Errorf("%d cases over %d clips in %d languages, want ≥ 24, ≥ 8, 7", len(cases), len(used), len(langs))
	}

	for _, k := range []string{kwPresent, kwMissed, kwDistractor, kwPhrase} {
		if kinds[k] == 0 {
			t.Errorf("no %s case", k)
		}
	}
}

func TestCheckKind(t *testing.T) {
	one := func(kind, kw string) keywordCase { return keywordCase{name: "x", kind: kind, keywords: []string{kw}} }

	for _, c := range []struct {
		kc        keywordCase
		ref, base int
		ok        bool
	}{
		{one(kwPresent, "a"), 1, 1, true},
		{one(kwPresent, "a"), 1, 0, false},
		{one(kwMissed, "a"), 1, 0, true},
		{one(kwMissed, "a"), 0, 0, false},
		{one(kwDistractor, "a"), 0, 0, true},
		{one(kwDistractor, "a"), 0, 1, false},
		{one(kwPhrase, "a b"), 1, 1, true},
		{one(kwPhrase, "a"), 1, 0, false},
		{keywordCase{name: "x", kind: kwList, keywords: []string{"a", "b c"}}, 2, 0, true},
		{keywordCase{name: "x", kind: kwList, keywords: []string{"a", "b c"}}, 1, 0, false},
		{one("other", "a"), 1, 1, false},
	} {
		err := checkKind(c.kc, keywordRecord{HitsReference: c.ref, HitsWithout: c.base})
		if (err == nil) != c.ok {
			t.Errorf("%s %q, ref %d, base %d: err = %v", c.kc.kind, c.kc.keywords, c.ref, c.base, err)
		}
	}
}

// subsetEntries builds a test-clean-like stream: lsSpeakers speakers with 7 utterances each, one of them over the
// engine's limit, and every chapter's transcript after its audio.
func subsetEntries() []tarEntry {
	out := make([]tarEntry, 0, 8*lsSpeakers)

	for s := range lsSpeakers {
		spk := strconv.Itoa(100 + s)

		var trans strings.Builder

		for u := range 7 {
			utt := fmt.Sprintf("%s-9-%04d", spk, u)
			sec := "2"

			if u == 3 {
				sec = "31" // over 30 s
			}

			out = append(out, tarEntry{"LibriSpeech/test-clean/" + spk + "/9/" + utt + ".flac", []byte(sec)})
			trans.WriteString(utt + " WORDS OF " + utt + "\n")
		}

		out = append(out, tarEntry{"LibriSpeech/test-clean/" + spk + "/9/" + spk + "-9.trans.txt", []byte(trans.String())})
	}

	return out
}

func TestSelectSubset(t *testing.T) {
	entries := subsetEntries()

	got, err := selectSubset(buildTar(t, entries), fakeFLAC)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != lsSpeakers*lsPerSpeaker {
		t.Fatalf("%d utterances, want %d", len(got), lsSpeakers*lsPerSpeaker)
	}

	// Per speaker, the lsPerSpeaker smallest keys among the utterances within the limit.
	for s := range lsSpeakers {
		spk := strconv.Itoa(100 + s)

		var keys []string

		for u := range 7 {
			if u != 3 {
				keys = append(keys, subsetKey(fmt.Sprintf("%s-9-%04d", spk, u)))
			}
		}

		slices.Sort(keys)

		var picked []string

		for _, u := range got {
			if u.Speaker == spk {
				picked = append(picked, subsetKey(strings.TrimPrefix(u.ID, "ls-clean-")))

				if u.Transcript != "WORDS OF "+strings.TrimPrefix(u.ID, "ls-clean-") || u.Samples != 2*sampleRate {
					t.Errorf("%s: %+v", u.ID, u)
				}
			}
		}

		slices.Sort(picked)

		if !slices.Equal(picked, keys[:lsPerSpeaker]) {
			t.Errorf("speaker %s: picked %v, want %v", spk, picked, keys[:lsPerSpeaker])
		}
	}

	// The selection does not depend on the tar order.
	slices.Reverse(entries)

	again, err := selectSubset(buildTar(t, entries), fakeFLAC)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.EqualFunc(got, again, func(a, b lsUtterance) bool { return a.ID == b.ID && a.Transcript == b.Transcript }) {
		t.Error("reversing the tar changed the subset")
	}
}

func TestSelectSubsetRejects(t *testing.T) {
	entries := subsetEntries()

	_, err := selectSubset(buildTar(t, entries[:len(entries)-8]), fakeFLAC)
	if err == nil || !strings.Contains(err.Error(), "speakers") {
		t.Errorf("one speaker short: err = %v", err)
	}

	_, err = selectSubset(buildTar(t, entries[:len(entries)-1]), fakeFLAC)
	if err == nil || !strings.Contains(err.Error(), "no transcript") {
		t.Errorf("a chapter without transcript: err = %v", err)
	}

	failing := func(io.Reader) ([]int16, int, error) { return nil, 0, io.ErrUnexpectedEOF }

	_, err = selectSubset(buildTar(t, entries), failing)
	if err == nil {
		t.Error("a decode error was swallowed")
	}
}

func TestWordErrors(t *testing.T) {
	for _, c := range []struct {
		ref, hyp string
		want     int
	}{
		{"A B C", "A B C", 0},
		{"A B C", "A X C", 1},
		{"A B C", "A C", 1},
		{"A B C", "A B C D", 1},
		{"A B C", "", 3},
		{"", "A", 1},
		{"THE CAT SAT", "cat sat on", 2},
	} {
		if got := wordErrors(upperWords(c.ref), upperWords(c.hyp)); got != c.want {
			t.Errorf("wordErrors(%q, %q) = %d, want %d", c.ref, c.hyp, got, c.want)
		}
	}

	if got := upperWords("Oh, mouse!\" St. James's"); !slices.Equal(got, []string{"OH", "MOUSE", "ST", "JAMES'S"}) {
		t.Errorf("upperWords = %q", got)
	}
}

// TestWriteWAV16MatchesCorpus re-encodes every committed corpus clip: writeWAV16 must reproduce the file, and so
// its pinned SHA-256, byte for byte.
func TestWriteWAV16MatchesCorpus(t *testing.T) {
	dir := filepath.Join("..", "..", "testdata", "audio")

	corpus, err := readCorpus(dir)
	if err != nil {
		t.Skipf("corpus manifest: %v", err)
	}

	for _, c := range corpus.Clips {
		raw, err := readClip(filepath.Join(dir, c.ID+".wav"), c.SHA256)
		if err != nil {
			t.Fatal(err)
		}

		x, _, err := readWAV(raw)
		if err != nil {
			t.Fatal(err)
		}

		pcm := make([]int16, len(x))
		for i, v := range x {
			pcm[i] = int16(v * 32768)
		}

		var buf bytes.Buffer

		err = writeWAV16(&buf, pcm)
		if err != nil || !bytes.Equal(buf.Bytes(), raw) {
			t.Fatalf("%s: re-encoded WAV differs (%v)", c.ID, err)
		}
	}
}
