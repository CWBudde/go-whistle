package whistle

import (
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// The stream's rules (docs/FINDINGS.md F-032, F-036): ufal whisper_streaming's
// OnlineASRProcessor with Transcribe as the ASR, no prompt, LocalAgreement-2
// and buffer_trimming="sentence" at 15 s.
const (
	// keepMargin: a pass keeps the words that start after the end of the
	// last committed word minus this margin (F-032 item 3).
	keepMargin = 100 * time.Millisecond
	// ngramWindow: when the first kept word starts within this time of the
	// last committed end, a repeat of the committed tail is dropped.
	ngramWindow = time.Second
	// maxNgram is the longest committed tail the repeat check compares.
	maxNgram = 5
	// trimAfter: a call that committed words trims the buffer at a
	// completed sentence once it holds more than this (F-032 item 5).
	trimAfter = 15 * SampleRate
)

// StreamOptions are the settings of a stream, the options of the engine's
// needle_stream_transcribe_process. The zero value detects the language of
// every pass, without keywords.
type StreamOptions struct {
	// Language forces the language of every pass; empty detects it per
	// pass, so it can change between calls (F-032 item 2).
	Language string
	// Keywords bias every pass, as TranscribeOptions.Keywords.
	Keywords []string
}

// Stream is live transcription, the engine's needle_stream_transcribe_process
// and _stop (docs/FINDINGS.md F-032, F-036). Process appends a chunk of
// 16 kHz mono samples to the buffer and re-transcribes the buffer in one pass,
// with Transcribe's beam search, silence gate and word timestamps. A word is
// committed once two consecutive passes agree on it: each call commits the
// longest prefix that its pass and the previous call's Pending have in
// common, compared as exact strings. Once the buffer holds more than 15 s
// after a commit, the audio up to the end of the second-to-last committed
// sentence is dropped. Stop commits Pending without a pass and ends the
// stream; the next Process starts a new one.
//
// A Stream is used from one goroutine. Several Streams, and Transcribe
// calls, can share one Model concurrently.
type Stream struct {
	m    *Model
	opts TranscribeOptions
	now  func() time.Time

	open     bool          // Process has fed audio since the last Stop
	buf      []float32     // the samples from offset on: the next pass' audio
	offset   int           // samples dropped from the front, the buffer start t0
	at       float64       // offset in seconds as the engine holds it (F-036)
	received int           // samples fed since the stream started
	inBuffer []streamWord  // committed words that end after offset
	pending  []streamWord  // the last pass' words after its commit
	last     time.Duration // the end of the last committed word
	lang     string        // the last pass' language
}

// streamWord is a word of a pass with its end as the engine holds it: the
// pass' buffer offset plus float32(frames) · 0.08f seconds, in float64
// (F-036). The trim cuts there, and the offset it leaves shows in Received.
type streamWord struct {
	Word

	end float64
}

// StreamResult is one Process or Stop call, the engine's stream JSON object.
//
// Its JSON form is the engine's byte for byte (F-036): text, words, pending,
// language, received and pass_ms in that order, received with two decimals
// and pass_ms with one; the words as in Result.
type StreamResult struct {
	Text     string // the words committed by this call, joined by spaces
	Words    []Word // the words committed by this call, times from the stream start; never nil
	Pending  string // the rest of the last pass, not yet agreed on; "" after Stop
	Language string // the last pass' language, "" when it heard silence
	// Received is the audio fed since the stream started, in seconds, as
	// the engine computes it (F-036).
	Received float64
	// Pass is the time the call took; 0 for Stop, which runs no pass.
	Pass time.Duration
}

// MarshalJSON writes r in the engine's form.
func (r StreamResult) MarshalJSON() ([]byte, error) {
	b := make([]byte, 0, 96+len(r.Text)+len(r.Pending)+64*len(r.Words))

	b = append(b, `{"text":`...)
	b = appendString(b, r.Text)
	b = append(b, `,"words":`...)
	b = appendWords(b, r.Words)
	b = append(b, `,"pending":`...)
	b = appendString(b, r.Pending)
	b = append(b, `,"language":`...)
	b = appendString(b, r.Language)
	b = append(b, `,"received":`...)
	b = strconv.AppendFloat(b, r.Received, 'f', 2, 64)
	b = append(b, `,"pass_ms":`...)
	b = strconv.AppendFloat(b, float64(r.Pass)/float64(time.Millisecond), 'f', 1, 64)

	return append(b, '}'), nil
}

// NewStream starts a stream on m. It checks opts once; every pass uses them.
func (m *Model) NewStream(opts StreamOptions) (*Stream, error) {
	return m.newStream(opts, time.Now)
}

func (m *Model) newStream(opts StreamOptions, now func() time.Time) (*Stream, error) {
	t := TranscribeOptions{Language: opts.Language, Keywords: opts.Keywords, WordTimestamps: true}

	err := m.check(t)
	if err != nil {
		return nil, err
	}

	_, err = m.request(t)
	if err != nil {
		return nil, err
	}

	return &Stream{m: m, opts: t, now: now}, nil
}

// Process feeds pcm, 16 kHz mono samples in [-1, 1] and at most MaxSamples
// per call, and runs one pass over the buffer, whatever the chunk length
// (F-032 item 1). About a second per call is the engine's advice. A call over
// MaxSamples fails with ErrTooLong and feeds nothing. An empty first call
// returns the empty result without starting the stream; an empty later call
// re-transcribes the unchanged buffer, which agrees with the previous pass
// and so commits all of Pending (F-032 item 9).
func (s *Stream) Process(pcm []float32) (StreamResult, error) {
	start := s.now()

	err := checkLength(len(pcm))
	if err != nil {
		return StreamResult{}, err
	}

	if !s.open && len(pcm) == 0 {
		return emptyStream(), nil
	}

	s.open = true
	s.buf = append(s.buf, pcm...)
	s.received += len(pcm)
	s.limit()

	words, lang, err := s.pass()
	if err != nil {
		return StreamResult{}, err
	}

	s.lang = lang
	commit := s.agree(s.fresh(words))

	if len(commit) > 0 && len(s.buf) > trimAfter {
		s.trimSentence()
	}

	return StreamResult{
		Text:     joinWords(commit),
		Words:    commit,
		Pending:  joinWords(plainWords(s.pending)),
		Language: lang,
		Received: s.receivedSeconds(),
		Pass:     s.now().Sub(start),
	}, nil
}

// Stop ends the stream: it commits Pending verbatim, with the times and
// probabilities of the last pass, and runs no pass (F-032 item 6). A Stop
// without a running stream returns the empty result. The next Process starts
// a new stream.
func (s *Stream) Stop() StreamResult {
	if !s.open {
		return emptyStream()
	}

	words := plainWords(s.pending)
	res := StreamResult{
		Text:     joinWords(words),
		Words:    words,
		Language: s.lang,
		Received: s.receivedSeconds(),
	}

	*s = Stream{m: s.m, opts: s.opts, now: s.now}

	return res
}

// emptyStream is the engine's result of a Stop without a stream:
// {"text":"","words":[],"pending":"","language":"","received":0.00,"pass_ms":0.0}.
func emptyStream() StreamResult {
	return StreamResult{Words: []Word{}}
}

// pass transcribes the buffer and returns its words with times from the
// stream start, and its language.
func (s *Stream) pass() ([]streamWord, string, error) {
	res, err := s.m.transcribe(s.buf, s.opts, s.now)
	if err != nil {
		return nil, "", err
	}

	shift := samplesDuration(s.offset)

	words := make([]streamWord, len(res.Words))
	for i, w := range res.Words {
		end := s.at + frameSeconds(int(w.End/FrameDuration))
		w.Start += shift
		w.End += shift
		words[i] = streamWord{Word: w, end: end}
	}

	return words, res.Language, nil
}

// limit keeps a pass within MaxSamples (PLAN.md 8.4, docs/FINDINGS.md
// F-036): a buffer over 30 s, which the sentence trim could not shorten,
// loses its oldest audio in whole 80 ms frames, so it holds the last 30 s
// or up to one frame less and its word times stay on the frame grid. The
// committed words in the dropped audio go with it, as in the trim. The
// offset moves by the exact sample time, which keeps Received at the
// engine's (F-036).
func (s *Stream) limit() {
	over := len(s.buf) - MaxSamples
	if over <= 0 {
		return
	}

	drop := (over + frameSamples - 1) / frameSamples * frameSamples
	s.chunkAt(s.offset+drop, s.at+float64(drop)*(1.0/SampleRate))
}

// fresh drops the words of a pass that the stream has committed already
// (F-032 item 3, whisper_streaming's HypothesisBuffer.insert): it keeps the
// words that start after the last committed end minus keepMargin; when the
// first of them starts within ngramWindow of that end, it also drops the
// shortest n-gram, n ≤ maxNgram, that repeats the committed tail.
func (s *Stream) fresh(words []streamWord) []streamWord {
	k := 0
	for k < len(words) && words[k].Start <= s.last-keepMargin {
		k++
	}

	words = words[k:]

	if len(words) == 0 || len(s.inBuffer) == 0 || (words[0].Start-s.last).Abs() >= ngramWindow {
		return words
	}

	for n := 1; n <= min(len(s.inBuffer), len(words), maxNgram); n++ {
		if sameWords(s.inBuffer[len(s.inBuffer)-n:], words[:n]) {
			return words[n:]
		}
	}

	return words
}

// agree commits the longest common prefix of words and the previous call's
// pending words, compared as exact strings (LocalAgreement-2, F-032 item 4),
// keeps the rest as the new pending words and returns the commit.
func (s *Stream) agree(words []streamWord) []Word {
	k := 0
	for k < len(words) && k < len(s.pending) && words[k].Text == s.pending[k].Text {
		k++
	}

	s.pending = append([]streamWord(nil), words[k:]...)

	if k > 0 {
		s.last = words[k-1].End
		s.inBuffer = append(s.inBuffer, words[:k]...)
	}

	return plainWords(words[:k])
}

// trimSentence drops the buffer up to the end of the second-to-last sentence
// of the committed words (F-032 item 5), when that end lies inside it.
func (s *Stream) trimSentence() {
	i := len(s.inBuffer) - 2
	for i >= 0 && !endsSentence(s.inBuffer[i].Text, s.inBuffer[i+1].Text) {
		i--
	}

	if i < 0 || durationSamples(s.inBuffer[i].End) <= s.offset {
		return
	}

	s.chunkAt(durationSamples(s.inBuffer[i].End), s.inBuffer[i].end)
}

// chunkAt drops the buffer before sample cut, at seconds as the engine
// holds it, and the committed words that end there or earlier
// (whisper_streaming's chunk_at).
func (s *Stream) chunkAt(cut int, at float64) {
	s.buf = s.buf[min(cut-s.offset, len(s.buf)):]
	s.offset, s.at = cut, at

	k := 0
	for k < len(s.inBuffer) && durationSamples(s.inBuffer[k].End) <= cut {
		k++
	}

	s.inBuffer = s.inBuffer[k:]
}

// endsSentence reports whether a sentence ends between word and next: word
// ends in '.', '?' or '!', closing quotes allowed, and next starts with an
// upper-case letter or a digit, after opening quotes (F-032 item 5).
func endsSentence(word, next string) bool {
	w := strings.TrimRight(word, `"'”’»)`)

	if w == "" || !strings.ContainsRune(".?!", rune(w[len(w)-1])) {
		return false
	}

	r, _ := utf8.DecodeRuneInString(strings.TrimLeft(next, `"'“‘«(`))

	return unicode.IsUpper(r) || unicode.IsDigit(r)
}

// receivedSeconds is Received as the engine computes it: the buffer offset
// plus the buffer length times 1/16000, in float64 (F-036). Before the first
// trim it is n · (1/16000), which prints 50,160 samples as 3.14, not 3.13.
func (s *Stream) receivedSeconds() float64 {
	return s.at + float64(s.received-s.offset)*(1.0/SampleRate)
}

// plainWords is the words without the engine's float ends, never nil.
func plainWords(words []streamWord) []Word {
	out := make([]Word, len(words))
	for i, w := range words {
		out[i] = w.Word
	}

	return out
}

// sameWords reports whether a and b spell the same words.
func sameWords(a, b []streamWord) bool {
	return slices.EqualFunc(a, b, func(x, y streamWord) bool { return x.Text == y.Text })
}

// joinWords is the words' text joined by spaces.
func joinWords(words []Word) string {
	parts := make([]string, len(words))
	for i, w := range words {
		parts[i] = w.Text
	}

	return strings.Join(parts, " ")
}

// frameSamples is the hop of one encoder frame in samples, 1280.
const frameSamples = SampleRate * int(FrameDuration/time.Millisecond) / 1000

// frameSeconds is the time of n frames as the engine computes a word time:
// float32(n) · 0.08f (F-036).
func frameSeconds(n int) float64 {
	return float64(float32(n) * float32(FrameDuration.Seconds()))
}

// sampleDuration is the time of one sample at SampleRate, exactly 62.5 µs.
const sampleDuration = time.Second / SampleRate

// samplesDuration is the time of n samples.
func samplesDuration(n int) time.Duration { return time.Duration(n) * sampleDuration }

// durationSamples is the sample at time d, rounded.
func durationSamples(d time.Duration) int {
	return int((d + sampleDuration/2) / sampleDuration)
}
