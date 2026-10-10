//go:build oracle

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/cwbudde/go-whistle/tools/oracle/probe"
)

// Stream sets: the "set" of every run in testdata/oracle/stream/manifest.json.
const (
	streamSetClips    = "clips"    // every corpus clip in 1 s chunks, language detected
	streamSetChunks   = "chunks"   // every corpus clip in 0.5 s and 2 s chunks
	streamSetLanguage = "language" // 1 s chunks with a forced language
	streamSetLong     = "long"     // concatenated clips longer than 30 s
	streamSetEdge     = "edge"     // calls at the API's edges: no call, empty calls, 30 s and 30 s + 1 sample
)

// streamPassKey is the timing field of the stream JSON; it is dropped from the fixtures.
const streamPassKey = "pass_ms"

// streamLimit is the most samples one call (and needle_transcribe) takes: 30 s (needle.h, F-025).
const streamLimit = 30 * sampleRate

// streamSpec is one stream: lead zeros, then the clips with gap zeros between consecutive ones, cut into calls of
// chunk samples (the last one shorter) or of the explicit call lengths, with a forced language ("" detects).
type streamSpec struct {
	Name         string   `json:"name"`
	Set          string   `json:"set"`
	Clips        []string `json:"clips"`
	LeadSamples  int      `json:"lead_samples,omitempty"`
	GapSamples   int      `json:"gap_samples,omitempty"`
	ChunkSamples int      `json:"chunk_samples,omitempty"`
	CallSamples  []int    `json:"call_samples,omitempty"`
	Language     string   `json:"language,omitempty"`
}

// callLengths cuts n samples into the spec's calls.
func (s streamSpec) callLengths(n int) []int {
	if s.ChunkSamples == 0 {
		return s.CallSamples
	}

	var out []int

	for i := 0; i < n; i += s.ChunkSamples {
		out = append(out, min(s.ChunkSamples, n-i))
	}

	return out
}

// The long streams exceed 30 s: nine LibriSpeech test-clean clips back to back (44.91 s), nine test-other clips with
// 0.5 s of silence between them (42.16 s), and 32.5 s of silence before a clip (36.01 s).
var (
	streamLongClean = []string{
		"ls-clean-6930-75918-0000", "ls-clean-6930-75918-0007", "ls-clean-1320-122617-0003",
		"ls-clean-1320-122617-0012", "ls-clean-5639-40744-0011", "ls-clean-5639-40744-0014",
		"ls-clean-260-123440-0018", "ls-clean-260-123440-0007", "ls-clean-7729-102255-0045",
	}
	streamLongOther = []string{
		"ls-other-7902-96591-0023", "ls-other-7902-96591-0005", "ls-other-7018-75788-0016",
		"ls-other-7018-75788-0019", "ls-other-4198-12281-0004", "ls-other-4198-12281-0014",
		"ls-other-3080-5040-0018", "ls-other-3080-5040-0011", "ls-other-8280-266249-0032",
	}
)

// streamGerman is the German clip whose first pass detects English ("Eden.").
const streamGerman = "fleurs-de_de-10229344228128634115"

// streamForced are the forced-language runs: clips in their own language whose first passes detect another one,
// an English clip, and the German clip forced to English.
var streamForced = []struct{ clip, language string }{
	{streamGerman, "de"},
	{streamGerman, "en"},
	{"fleurs-fr_fr-10174413140250646001", "fr"},
	{"ls-clean-6930-75918-0000", "en"},
}

// streamSpecs lists every stream in recording order.
func streamSpecs(clips []string) []streamSpec {
	out := make([]streamSpec, 0, 3*len(clips)+len(streamForced)+8)

	for _, ch := range []struct {
		set, suffix string
		samples     int
	}{
		{streamSetClips, "c1", sampleRate},
		{streamSetChunks, "c0.5", sampleRate / 2},
		{streamSetChunks, "c2", 2 * sampleRate},
	} {
		for _, c := range clips {
			out = append(out, streamSpec{
				Name: c + "." + ch.suffix, Set: ch.set, Clips: []string{c}, ChunkSamples: ch.samples,
			})
		}
	}

	for _, f := range streamForced {
		out = append(out, streamSpec{
			Name: f.clip + ".c1." + f.language, Set: streamSetLanguage, Clips: []string{f.clip},
			ChunkSamples: sampleRate, Language: f.language,
		})
	}

	return append(out,
		streamSpec{Name: "long-clean", Set: streamSetLong, Clips: streamLongClean, ChunkSamples: sampleRate},
		streamSpec{
			Name: "long-other-gaps", Set: streamSetLong, Clips: streamLongOther, GapSamples: sampleRate / 2,
			ChunkSamples: sampleRate,
		},
		streamSpec{
			Name: "long-silence-lead", Set: streamSetLong, Clips: []string{streamLongClean[0]},
			LeadSamples: 65 * sampleRate / 2, ChunkSamples: sampleRate,
		},
		streamSpec{Name: "edge-stop-only", Set: streamSetEdge, CallSamples: []int{}},
		streamSpec{Name: "edge-empty-call", Set: streamSetEdge, CallSamples: []int{0}},
		streamSpec{
			Name: "edge-empty-calls-after-speech", Set: streamSetEdge, Clips: []string{streamGerman},
			CallSamples: []int{3 * sampleRate, 0, 0},
		},
		streamSpec{Name: "edge-30s-call", Set: streamSetEdge, LeadSamples: streamLimit, CallSamples: []int{streamLimit}},
		streamSpec{
			Name: "edge-30s-plus-1-call", Set: streamSetEdge, LeadSamples: streamLimit + 1,
			CallSamples: []int{streamLimit + 1},
		},
	)
}

// streamManifest is testdata/oracle/stream/manifest.json.
type streamManifest struct {
	fixtureHeader

	Calls       string            `json:"calls"`
	Audio       string            `json:"audio"`
	Output      string            `json:"output"`
	Checks      string            `json:"checks"`
	BufferStart string            `json:"buffer_start"`
	CLICheck    string            `json:"cli_check"`
	Offline     string            `json:"offline"`
	ClipSHA256  map[string]string `json:"clip_sha256"`
	Sets        []*streamSetCount `json:"sets"`
	Runs        []streamRecord    `json:"runs"`
}

// streamSetCount sums one set: its runs, how many have an offline text and end with exactly it, and the word-level
// edit distance from the offline texts.
type streamSetCount struct {
	Set           string `json:"set"`
	Runs          int    `json:"runs"`
	Compared      int    `json:"compared"`
	SameAsOffline int    `json:"same_as_offline"`
	WordErrors    int    `json:"offline_word_errors"`
	OfflineWords  int    `json:"offline_words"`
	CallsWithText int    `json:"calls_with_words"`
	BufferStarts  int    `json:"buffer_starts"`
	ReceivedOff   int    `json:"received_not_n_over_16000"`
}

// streamRecord is one run's summary.
type streamRecord struct {
	streamSpec

	Samples      int    `json:"samples"`
	PCMSHA256    string `json:"pcm_sha256"`
	Calls        int    `json:"calls"`
	Committed    int    `json:"committed_words"`
	Text         string `json:"text"`
	Lang         string `json:"stop_language"`
	Offline      string `json:"offline,omitempty"`
	Same         *bool  `json:"same_as_offline,omitempty"`
	WordErrors   *int   `json:"offline_word_errors,omitempty"`
	CallsWords   int    `json:"calls_with_words"`
	BufferStarts int    `json:"buffer_starts"`
	ReceivedOff  int    `json:"received_not_n_over_16000,omitempty"`
	CLI          string `json:"cli,omitempty"`
}

// streamRun is one run's calls in calls.json.
type streamRun struct {
	Calls []streamCall `json:"calls"`
	Stop  streamCall   `json:"stop"`
}

// streamCall is one needle_stream_transcribe_process call (start, end: the samples fed) or the stop.
type streamCall struct {
	Index       *int            `json:"index,omitempty"`
	Start       *int            `json:"start,omitempty"`
	End         *int            `json:"end,omitempty"`
	RC          int32           `json:"rc"`
	Error       string          `json:"error,omitempty"`
	BufferStart *int            `json:"buffer_start,omitempty"`
	Dropped     *int            `json:"dropped,omitempty"`
	Out         json.RawMessage `json:"out,omitempty"`

	parsed streamOut
	passMS string
}

// streamOut is the stream JSON without pass_ms; it also reads needle_transcribe's JSON.
type streamOut struct {
	Text     string       `json:"text"`
	Words    []streamWord `json:"words"`
	Pending  string       `json:"pending"`
	Language string       `json:"language"`
	Received json.Number  `json:"received"`
}

// streamWord keeps the numbers as printed.
type streamWord struct {
	Word        string      `json:"word"`
	Start       json.Number `json:"start"`
	End         json.Number `json:"end"`
	Probability json.Number `json:"probability"`
}

const (
	streamCallsDoc = "calls.json: <run> → {calls: [{index, start, end, rc, error, buffer_start, dropped, out}], " +
		"stop: {rc, error, out}}; start and end delimit the samples fed (from the stream start), rc is the return code"
	streamAudioDoc = "lead_samples zeros, then the clips (testdata/audio, int16 / 32768) with gap_samples zeros " +
		"between consecutive clips; pcm_sha256 is the SHA-256 of those float32 samples, little-endian. The calls " +
		"cut them into chunk_samples (the last call shorter) or into call_samples. Every run is one stream that " +
		"ends with a stop, so the next run starts a new one"
	streamOutputDoc = "out is the engine's JSON with pass_ms dropped (a timing), keys sorted, values verbatim. " +
		"text is the committed text of all calls and the stop joined by spaces, stop_language the stop's language"
	streamChecksDoc = "outside the edge set the run fails unless every call returns rc ≥ 0, rc equals the number " +
		"of words, text is the words joined by spaces, received is the samples fed so far / 16000 rounded " +
		"to two decimals either way, and pass_ms is > 0 on every process call and 0.0 on every stop"
	streamBufferDoc = "buffer_start (samples from the stream start), on calls that committed words: the start t0 " +
		"of the audio [t0, end) whose needle_transcribe JSON (in the run's language) ends with this call's words (the same text, start " +
		"and end once shifted by t0, and probability) followed by the words of pending, in the same language. " +
		"Candidates in order: the previous call's t0 (0 at first), then the ends of earlier committed words after " +
		"it, newest first, as long as [t0, end) is at most 30 s. dropped is the number of words of that JSON " +
		"before the call's words, which the engine left out as already committed. Both absent when the call " +
		"committed nothing or no candidate matched"
	streamCLIDoc = "\"same\": needle --audio-stream --audio-chunk <chunk_samples / 16000> [--audio-language] " +
		"--audio-word-timestamps --threads 4 on the run's samples as a float WAV prints, after dropping pass_ms, " +
		"exactly the outputs of the calls with rc > 0 followed by the stop's; the run fails otherwise. Not run " +
		"on the edge set"
	streamOfflineDoc = "offline is testdata/oracle/<clip>.json's text, for the long runs the clips' texts joined " +
		"by spaces; offline_word_errors is the word-level edit distance from it to text (words split at spaces)"
)

// cmdStream records needle_stream_transcribe_process and _stop on every stream of streamSpecs into
// testdata/oracle/stream, cross-checks them against the CLI's --audio-stream and compares the committed text with
// the offline transcripts.
func cmdStream(ctx context.Context, cfg runConfig, log io.Writer) error {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	o, err := openOracle(cfg, true)
	if err != nil {
		return err
	}

	tmp, err := os.MkdirTemp("", "oracle-stream-")
	if err != nil {
		return fmt.Errorf("temp dir: %w", err)
	}

	defer func() { _ = os.RemoveAll(tmp) }()

	man := streamManifest{
		fixtureHeader: o.header("needle_stream_transcribe_process, needle_stream_transcribe_stop (libneedle3)"),
		Calls:         streamCallsDoc, Audio: streamAudioDoc, Output: streamOutputDoc, Checks: streamChecksDoc,
		BufferStart: streamBufferDoc, CLICheck: streamCLIDoc, Offline: streamOfflineDoc,
		ClipSHA256: map[string]string{},
	}
	man.Threads = "the library's default (the device's fast cores, at most 4); the CLI check runs --threads " +
		cliThreads

	pcm, offline, ids, err := loadStreamCorpus(cfg, corpus, man.ClipSHA256)
	if err != nil {
		return err
	}

	runs := map[string]streamRun{}
	sets := map[string]*streamSetCount{}

	for _, s := range streamSpecs(ids) {
		rec, run, err := recordStream(ctx, o, tmp, s, pcm, offline)
		if err != nil {
			return fmt.Errorf("%s: %w", s.Name, err)
		}

		runs[s.Name] = run

		man.Runs = append(man.Runs, rec)

		sc := sets[s.Set]
		if sc == nil {
			sc = &streamSetCount{Set: s.Set}
			sets[s.Set] = sc
			man.Sets = append(man.Sets, sc)
		}

		sc.add(rec)
		rec.log(log)
	}

	js, err := json.MarshalIndent(runs, "", "  ")
	if err != nil {
		return fmt.Errorf("calls: %w", err)
	}

	err = writeFile(filepath.Join(cfg.outDir, "calls.json"), append(js, '\n'))
	if err != nil {
		return err
	}

	return writeManifest(cfg.outDir, man)
}

// log prints a run's summary line: = or ≠ marks whether its text equals the offline one.
func (r *streamRecord) log(w io.Writer) {
	mark := "="
	if r.Same == nil {
		mark = " "
	} else if !*r.Same {
		mark = "≠"
	}

	_, _ = fmt.Fprintf(w, "%-44s %3d calls %2d/%2d t0  %s [%s] %s\n", r.Name, r.Calls, r.BufferStarts, r.CallsWords,
		mark, r.Lang, r.Text)
}

// loadStreamCorpus reads every corpus clip and its offline text, and records the clips' digests in sums.
func loadStreamCorpus(cfg runConfig, corpus corpusManifest, sums map[string]string,
) (map[string][]float32, map[string]string, []string, error) {
	pcm, offline := map[string][]float32{}, map[string]string{}
	ids := make([]string, 0, len(corpus.Clips))

	for _, c := range corpus.Clips {
		x, text, err := loadStreamClip(cfg, c)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("%s: %w", c.ID, err)
		}

		pcm[c.ID], offline[c.ID], sums[c.ID] = x, text, c.SHA256
		ids = append(ids, c.ID)
	}

	return pcm, offline, ids, nil
}

func (sc *streamSetCount) add(r streamRecord) {
	sc.Runs++
	sc.CallsWithText += r.CallsWords
	sc.BufferStarts += r.BufferStarts
	sc.ReceivedOff += r.ReceivedOff

	if r.Same == nil {
		return
	}

	sc.Compared++
	sc.WordErrors += *r.WordErrors
	sc.OfflineWords += len(strings.Fields(r.Offline))

	if *r.Same {
		sc.SameAsOffline++
	}
}

// loadStreamClip reads a corpus clip, checked against its digest, and the text of its offline fixture.
func loadStreamClip(cfg runConfig, c clip) ([]float32, string, error) {
	raw, err := readClip(filepath.Join(cfg.audioDir, c.ID+".wav"), c.SHA256)
	if err != nil {
		return nil, "", err
	}

	x, rate, err := readWAV(raw)
	if err != nil {
		return nil, "", err
	}

	if rate != sampleRate {
		return nil, "", fmt.Errorf("%d Hz, want %d", rate, sampleRate)
	}

	js, err := readFixture(filepath.Join(cfg.oracleDir, c.ID+".json"))
	if err != nil {
		return nil, "", fmt.Errorf("%w (run `just oracle-run`)", err)
	}

	t, err := parseTranscript(js)
	if err != nil {
		return nil, "", err
	}

	return x, t.Text, nil
}

// streamAudio builds a spec's samples (streamAudioDoc).
func streamAudio(s streamSpec, pcm map[string][]float32) ([]float32, error) {
	parts := [][]float32{make([]float32, s.LeadSamples)}

	for i, id := range s.Clips {
		x, ok := pcm[id]
		if !ok {
			return nil, fmt.Errorf("clip %s is not in the corpus", id)
		}

		if i > 0 {
			parts = append(parts, make([]float32, s.GapSamples))
		}

		parts = append(parts, x)
	}

	return slices.Concat(parts...), nil
}

// recordStream runs one stream through the library, then infers the buffer starts and, outside the edge set,
// compares the text with the offline one and cross-checks the CLI.
func recordStream(ctx context.Context, o *oracle, tmp string, s streamSpec, pcm map[string][]float32,
	offline map[string]string,
) (streamRecord, streamRun, error) {
	x, err := streamAudio(s, pcm)
	if err != nil {
		return streamRecord{}, streamRun{}, err
	}

	rec := streamRecord{streamSpec: s, Samples: len(x), PCMSHA256: probe.SHA256(x)}

	run, err := rec.feed(o.eng, x)
	if err != nil {
		return rec, run, err
	}

	rec.BufferStarts, err = inferBufferStarts(o.eng, x, s.Language, run.Calls)
	if err != nil {
		return rec, run, err
	}

	if s.Set == streamSetEdge {
		return rec, run, nil
	}

	off := make([]string, len(s.Clips))
	for i, id := range s.Clips {
		off[i] = offline[id]
	}

	rec.Offline = strings.Join(off, " ")
	same, errs := rec.Text == rec.Offline, wordErrors(strings.Fields(rec.Offline), strings.Fields(rec.Text))
	rec.Same, rec.WordErrors = &same, &errs

	rec.CLI, err = streamCLI(ctx, o, tmp, s, x, run)

	return rec, run, err
}

// feed sends x to needle_stream_transcribe_process in the spec's calls, stops the stream and fills in the record's
// counts and text. Outside the edge set every call is checked (streamChecksDoc).
func (rec *streamRecord) feed(e *engine, x []float32) (streamRun, error) {
	strict := rec.Set != streamSetEdge
	run := streamRun{Calls: []streamCall{}}
	fed := 0

	var texts []string

	note := func(c streamCall) {
		if c.parsed.Text != "" {
			texts = append(texts, c.parsed.Text)
		}

		if strict && string(c.parsed.Received) != receivedText(fed) {
			rec.ReceivedOff++
		}

		rec.Committed += len(c.parsed.Words)
	}

	for i, n := range rec.callLengths(len(x)) {
		res := e.streamProcess(x[fed:fed+n], rec.Language)
		c := streamCall{Index: ptr(i), Start: ptr(fed), End: ptr(fed + n)}
		fed += n

		err := c.parse(res, fed, strict, true)
		if err != nil {
			return run, fmt.Errorf("call %d: %w", i, err)
		}

		if c.parsed.Text != "" {
			rec.CallsWords++
		}

		note(c)
		run.Calls = append(run.Calls, c)
	}

	err := run.Stop.parse(e.streamStop(), fed, strict, false)
	if err != nil {
		return run, fmt.Errorf("stop: %w", err)
	}

	note(run.Stop)
	rec.Calls, rec.Text, rec.Lang = len(run.Calls), strings.Join(texts, " "), run.Stop.parsed.Language

	return run, nil
}

func ptr(v int) *int { return &v }

// parse keeps a call's rc and error, normalises its JSON into c.Out and, when strict, checks it (streamChecksDoc). fed is the samples fed
// so far; process is false for the stop. Outside strict mode a failed call is kept as its rc and error.
func (c *streamCall) parse(res libResult, fed int, strict, process bool) error {
	c.RC, c.Error = res.RC, res.Error

	if res.RC < 0 {
		if strict {
			return fmt.Errorf("rc %d: %s", res.RC, res.Error)
		}

		return nil
	}

	var fields map[string]json.RawMessage

	err := json.Unmarshal(res.out, &fields)
	if err != nil {
		return fmt.Errorf("stream output is not a JSON object: %w: %q", err, res.out)
	}

	c.passMS = string(fields[streamPassKey])
	delete(fields, streamPassKey)

	c.Out, err = json.Marshal(fields)
	if err != nil {
		return fmt.Errorf("re-encode stream output: %w", err)
	}

	c.parsed, err = decodeStreamOut(res.out)
	if err != nil {
		return err
	}

	if !strict {
		return nil
	}

	return c.check(fed, process)
}

func decodeStreamOut(js []byte) (streamOut, error) {
	var out streamOut

	dec := json.NewDecoder(bytes.NewReader(js))
	dec.UseNumber()

	err := dec.Decode(&out)
	if err != nil {
		return out, fmt.Errorf("engine JSON: %w", err)
	}

	return out, nil
}

// check applies streamChecksDoc to a parsed call.
func (c *streamCall) check(fed int, process bool) error {
	words := make([]string, len(c.parsed.Words))
	for i, w := range c.parsed.Words {
		words[i] = w.Word
	}

	pass, err := strconv.ParseFloat(c.passMS, 64)
	if err != nil {
		return fmt.Errorf("pass_ms %q: %w", c.passMS, err)
	}

	switch {
	case int(c.RC) != len(words):
		return fmt.Errorf("rc %d, %d words", c.RC, len(words))
	case c.parsed.Text != strings.Join(words, " "):
		return fmt.Errorf("text %q is not the words %q", c.parsed.Text, words)
	case !closeReceived(c.parsed.Received, fed):
		return fmt.Errorf("received %s after %d samples", c.parsed.Received, fed)
	case process && pass <= 0:
		return fmt.Errorf("pass_ms %s on a process call", c.passMS)
	case !process && pass != 0:
		return fmt.Errorf("pass_ms %s on the stop", c.passMS)
	}

	return nil
}

// receivedText is how the engine prints n samples as seconds until its buffer is first trimmed: n times the float64
// 1/16000, to two decimals. Dividing by 16000 instead prints 50,160 samples as "3.13" where the engine prints "3.14".
// After a trim it can be one hundredth off (674,480 samples, 42.155 s, print as "42.15"); streamRecord counts those.
func receivedText(n int) string { return strconv.FormatFloat(float64(n)*(1.0/sampleRate), 'f', 2, 64) }

// closeReceived reports whether received is n samples rounded to a hundredth either way.
func closeReceived(received json.Number, n int) bool {
	v, err := received.Float64()

	return err == nil && math.Abs(v-float64(n)/sampleRate) <= 0.005+1e-9
}

// inferBufferStarts sets buffer_start on the calls that committed words (streamBufferDoc) and returns how many got
// one. It runs after the stream has stopped, so needle_transcribe cannot disturb the stream.
func inferBufferStarts(e *engine, x []float32, language string, calls []streamCall) (int, error) {
	var (
		start, found int
		ends         []int
	)

	for i := range calls {
		c := &calls[i]
		if c.RC <= 0 {
			continue
		}

		end := *c.End
		cands := []int{start}

		for j := len(ends) - 1; j >= 0 && ends[j] > start; j-- {
			cands = append(cands, ends[j])
		}

		for _, t0 := range cands {
			if end-t0 > streamLimit {
				continue
			}

			dropped, err := explains(e, x[t0:end], t0, language, c.parsed)
			if err != nil {
				return found, fmt.Errorf("call %d: %w", i, err)
			}

			if dropped >= 0 {
				start = t0
				c.BufferStart, c.Dropped = ptr(t0), ptr(dropped)
				found++

				break
			}
		}

		for _, w := range c.parsed.Words {
			f, err := w.End.Float64()
			if err != nil {
				return found, fmt.Errorf("call %d: word end: %w", i, err)
			}

			ends = append(ends, int(math.Round(f*sampleRate)))
		}
	}

	return found, nil
}

// explains returns how many leading words of needle_transcribe's output on buf (which starts t0 samples into the
// stream, in the run's language) precede the call's committed words and then its pending words (streamBufferDoc),
// or -1 when the output does not end with them.
func explains(e *engine, buf []float32, t0 int, language string, c streamOut) (int, error) {
	res := e.transcribeIn(buf, language)
	if res.RC < 0 {
		return -1, fmt.Errorf("needle_transcribe: %s (%d)", res.Error, res.RC)
	}

	off, err := decodeStreamOut(res.out)
	if err != nil {
		return -1, err
	}

	pending := strings.Fields(c.Pending)

	k := len(off.Words) - len(c.Words) - len(pending)
	if off.Language != c.Language || k < 0 {
		return -1, nil
	}

	for i, p := range pending {
		if off.Words[k+len(c.Words)+i].Word != p {
			return -1, nil
		}
	}

	shift := float64(t0) / sampleRate

	for i, w := range c.Words {
		o := off.Words[k+i]
		if o.Word != w.Word || o.Probability != w.Probability || !sameTime(o.Start, w.Start, shift) ||
			!sameTime(o.End, w.End, shift) {
			return -1, nil
		}
	}

	return k, nil
}

// sameTime reports whether the offline time off, shifted by shift seconds, rounds to the stream time s.
func sameTime(off, s json.Number, shift float64) bool {
	a, err := off.Float64()
	if err != nil {
		return false
	}

	b, err := s.Float64()
	if err != nil {
		return false
	}

	return math.Abs(a+shift-b) < 0.006
}

// streamCLI runs the CLI's --audio-stream on the run's samples and requires streamCLIDoc.
func streamCLI(ctx context.Context, o *oracle, tmp string, s streamSpec, x []float32, run streamRun) (string, error) {
	wav, err := writeTempWAV(tmp, "stream", x)
	if err != nil {
		return "", err
	}

	flags := []string{
		"--audio-stream", "--audio-chunk", strconv.FormatFloat(float64(s.ChunkSamples)/sampleRate, 'g', -1, 64),
	}
	if s.Language != "" {
		flags = append(flags, "--audio-language", s.Language)
	}

	res, err := o.runIn(ctx, tmp, wav, flags...)
	if err != nil {
		return "", err
	}

	if res.Exit != 0 {
		return "", fmt.Errorf("needle --audio-stream: exit %d: %s", res.Exit, res.Stderr)
	}

	var want []json.RawMessage

	for _, c := range run.Calls {
		if c.RC > 0 {
			want = append(want, c.Out)
		}
	}

	want = append(want, run.Stop.Out)

	lines := bytes.Split(bytes.TrimSpace(res.stdout), []byte("\n"))
	if len(lines) != len(want) {
		return "", fmt.Errorf("needle --audio-stream printed %d lines, the library committed in %d calls",
			len(lines), len(want)-1)
	}

	for i, l := range lines {
		var c streamCall

		err := c.parse(libResult{out: l}, 0, false, false)
		if err != nil {
			return "", fmt.Errorf("needle --audio-stream line %d: %w", i+1, err)
		}

		if !bytes.Equal(c.Out, want[i]) {
			return "", fmt.Errorf("needle --audio-stream line %d differs:\n%s\n%s", i+1, c.Out, want[i])
		}
	}

	return "same", nil
}
