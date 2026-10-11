package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
)

// Pinned inputs as laid out by `whistle download` and `just oracle-run`.
const (
	cactPath  = "../../models/whistle/whistle.cact"
	clipDir   = "../../testdata/audio"
	oracleDir = "../../testdata/oracle"
)

// equalClips are four corpus clips (English test-other and test-clean,
// German, French) on which Transcribe gives the engine's text on every
// platform (docs/FINDINGS.md F-026).
var equalClips = []string{
	"ls-other-3080-5040-0018", "ls-clean-5639-40744-0011",
	"fleurs-de_de-10308608197319017935", "fleurs-fr_fr-10174413140250646001",
}

// fixture is a clip's engine transcript and its words' spelling.
type fixture struct {
	Language string `json:"language"`
	Text     string `json:"text"`
	Words    []struct {
		Word string `json:"word"`
	} `json:"words"`
}

func readFixture(t *testing.T, id string) fixture {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(oracleDir, id+".json"))
	if err != nil {
		t.Fatal(err)
	}

	var f fixture

	err = json.Unmarshal(raw, &f)
	if err != nil {
		t.Fatalf("%s: %v", id, err)
	}

	return f
}

// modelBytes reads whistle.cact or skips the test.
func modelBytes(t *testing.T) []byte {
	t.Helper()

	data, err := os.ReadFile(cactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", cactPath)
	}

	if err != nil {
		t.Fatal(err)
	}

	return data
}

// result is the engine JSON the page reads.
type result struct {
	Text     string `json:"text"`
	Language string `json:"language"`
	Words    []struct {
		Word  string  `json:"word"`
		Start float64 `json:"start"`
		End   float64 `json:"end"`
	} `json:"words"`
	TTFT      float64 `json:"ttft_ms"`
	DecodeTPS float64 `json:"decode_tps"`
}

// checkResult compares a transcribe result with the engine's transcript of
// clip id: the text, the language and the words that spell it.
func checkResult(t *testing.T, id string, out []byte) result {
	t.Helper()

	var got result

	err := json.Unmarshal(out, &got)
	if err != nil {
		t.Fatalf("%s: %v in %s", id, err, out)
	}

	want := readFixture(t, id)
	if got.Text != want.Text || got.Language != want.Language {
		t.Errorf("%s: [%s] %q, the engine [%s] %q", id, got.Language, got.Text, want.Language, want.Text)
	}

	spelled := make([]string, len(got.Words))
	for i, w := range got.Words {
		spelled[i] = w.Word
	}

	if strings.Join(spelled, " ") != got.Text || len(got.Words) != len(want.Words) {
		t.Errorf("%s: %d words %q spell %q; the engine has %d", id, len(got.Words), spelled, got.Text, len(want.Words))
	}

	return got
}

func TestDecodeOptions(t *testing.T) {
	for _, js := range []string{"", "null", "undefined", " {} "} {
		var o options

		err := decodeOptions(js, &o)
		if err != nil || o.Language != "" || o.Keywords != nil || o.WordTimestamps || o.Depth != 0 {
			t.Errorf("%q: %+v, %v; want the zero options", js, o, err)
		}
	}

	var o options

	err := decodeOptions(`{"language":"de","keywords":["Bahre"],"word_timestamps":true,"depth":4}`, &o)
	if err != nil || o.Language != "de" || !slices.Equal(o.Keywords, []string{"Bahre"}) || !o.WordTimestamps || o.Depth != 4 {
		t.Errorf("full options: %+v, %v", o, err)
	}

	for _, js := range []string{`{"wordTimestamps":true}`, `{"depth":"4"}`, `[1]`, `{`} {
		var o options

		err := decodeOptions(js, &o)
		if err == nil || !strings.HasPrefix(err.Error(), "options: ") {
			t.Errorf("%s: %v, want an options error", js, err)
		}
	}

	var s streamOptions

	err = decodeOptions(`{"word_timestamps":true}`, &s)
	if err == nil {
		t.Error("streamStart takes no word_timestamps, got no error")
	}
}

// Every call needs a model and the stream calls an open stream; the
// errors say which call comes first.
func TestEngineOrder(t *testing.T) {
	var e engine

	_, err := e.transcribe(make([]float32, 1600), "")
	if !errors.Is(err, errNoModel) {
		t.Errorf("transcribe before load: %v", err)
	}

	err = e.streamStart("")
	if !errors.Is(err, errNoModel) {
		t.Errorf("streamStart before load: %v", err)
	}

	_, err = e.streamProcess(make([]float32, 1600))
	if !errors.Is(err, errNoStream) {
		t.Errorf("streamProcess before streamStart: %v", err)
	}

	_, err = e.streamStop()
	if !errors.Is(err, errNoStream) {
		t.Errorf("streamStop before streamStart: %v", err)
	}

	_, err = e.load([]byte("not a weight file"))
	if err == nil || e.model != nil {
		t.Errorf("load of garbage: %v, model %v", err, e.model)
	}
}

// decodeWAV reads a WAV as the CLI does (frontend.ReadWAV); an empty or
// non-WAV file fails.
func TestDecodeWAV(t *testing.T) {
	path := filepath.Join(clipDir, equalClips[0]+".wav")

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	got, err := decodeWAV(data)
	if err != nil {
		t.Fatal(err)
	}

	want, err := frontend.ReadWAV(path)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got, want) {
		t.Errorf("decodeWAV: %d samples differ from ReadWAV's %d", len(got), len(want))
	}

	_, err = decodeWAV([]byte("RIFF"))
	if err == nil {
		t.Error("decodeWAV of a truncated header: no error")
	}

	empty := append([]byte(nil), data[:44]...) // the header alone
	empty[40], empty[41], empty[42], empty[43] = 0, 0, 0, 0

	_, err = decodeWAV(empty)
	if !errors.Is(err, whistle.ErrNoAudio) {
		t.Errorf("decodeWAV of an empty WAV: %v, want ErrNoAudio", err)
	}
}

// The page's path: the model from bytes, each WAV through decodeWAV, then
// transcribe with word timestamps gives the engine's transcript.
func TestEngineTranscribe(t *testing.T) {
	var e engine

	info, err := e.load(modelBytes(t))
	if err != nil {
		t.Fatal(err)
	}

	if len(info.Languages) != 7 || info.Bytes == 0 {
		t.Errorf("load: %+v", info)
	}

	for _, id := range equalClips {
		data, err := os.ReadFile(filepath.Join(clipDir, id+".wav"))
		if err != nil {
			t.Fatal(err)
		}

		pcm, err := decodeWAV(data)
		if err != nil {
			t.Fatal(err)
		}

		out, err := e.transcribe(pcm, `{"word_timestamps":true}`)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		res := checkResult(t, id, out)
		t.Logf("%s: [%s] ttft %.1f ms, %.1f tok/s", id, res.Language, res.TTFT, res.DecodeTPS)
	}

	_, err = e.transcribe(make([]float32, whistle.MaxSamples+1), "")
	if !errors.Is(err, whistle.ErrTooLong) {
		t.Errorf("over 30 s: %v, want ErrTooLong", err)
	}

	_, err = e.transcribe(make([]float32, 1600), `{"language":"xx"}`)
	if err == nil || !strings.Contains(err.Error(), `"xx"`) {
		t.Errorf("unknown language: %v", err)
	}
}

// A load while a transcribe runs (two promises on globalThis.whistle) must
// not close the model under the call: every transcribe still gives the
// engine's text.
func TestEngineLoadDuringTranscribe(t *testing.T) {
	var e engine

	data := modelBytes(t)

	_, err := e.load(data)
	if err != nil {
		t.Fatal(err)
	}

	id := equalClips[0]

	wav, err := os.ReadFile(filepath.Join(clipDir, id+".wav"))
	if err != nil {
		t.Fatal(err)
	}

	pcm, err := decodeWAV(wav)
	if err != nil {
		t.Fatal(err)
	}

	const runs = 3

	type call struct {
		out []byte
		err error
	}

	done := make(chan call, runs)

	for range runs {
		go func() {
			out, err := e.transcribe(pcm, `{"word_timestamps":true}`)
			done <- call{out, err}
		}()
	}

	for range runs {
		_, err := e.load(data)
		if err != nil {
			t.Fatal(err)
		}
	}

	for range runs {
		c := <-done
		if c.err != nil {
			t.Fatalf("transcribe with a load beside it: %v", c.err)
		}

		checkResult(t, id, c.out)
	}
}

// streamRun is the engine stream ls-clean-6930-75918-0000.c1: the clip in
// 1 s calls, then the stop (testdata/oracle/stream).
const (
	streamClip = "ls-clean-6930-75918-0000"
	streamText = "Concord returned to its place amidst the tents."
)

// The microphone's path: 1 s chunks through streamStart, streamProcess and
// streamStop commit the engine stream's text.
func TestEngineStream(t *testing.T) {
	var e engine

	_, err := e.load(modelBytes(t))
	if err != nil {
		t.Fatal(err)
	}

	pcm, err := frontend.ReadWAV(filepath.Join(clipDir, streamClip+".wav"))
	if err != nil {
		t.Fatal(err)
	}

	err = e.streamStart(`{"language":""}`)
	if err != nil {
		t.Fatal(err)
	}

	err = e.streamStart("")
	if !errors.Is(err, errStreamOn) {
		t.Errorf("a second streamStart: %v, want errStreamOn", err)
	}

	var text []string

	commit := func(out []byte) {
		t.Helper()

		var r struct {
			Text     string  `json:"text"`
			Received float64 `json:"received"`
		}

		err := json.Unmarshal(out, &r)
		if err != nil {
			t.Fatalf("%v in %s", err, out)
		}

		if r.Text != "" {
			text = append(text, r.Text)
		}
	}

	for i := 0; i < len(pcm); i += whistle.SampleRate {
		out, err := e.streamProcess(pcm[i:min(i+whistle.SampleRate, len(pcm))])
		if err != nil {
			t.Fatal(err)
		}

		commit(out)
	}

	out, err := e.streamStop()
	if err != nil {
		t.Fatal(err)
	}

	commit(out)

	if got := strings.Join(text, " "); got != streamText {
		t.Errorf("stream text %q, the engine's stream %q", got, streamText)
	}

	_, err = e.streamProcess(pcm[:1600])
	if !errors.Is(err, errNoStream) {
		t.Errorf("streamProcess after streamStop: %v, want errNoStream", err)
	}
}
