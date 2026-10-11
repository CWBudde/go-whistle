package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
)

// fakeModel answers with the sample count; "xx" is an unknown language.
type fakeModel struct {
	mu    sync.Mutex
	opts  []whistle.TranscribeOptions // the options of every call with audio
	block chan struct{}               // when set, a call with audio waits for it
	run   chan struct{}               // when set, a call with audio reports here first
}

func (f *fakeModel) Transcribe(pcm []float32, opts whistle.TranscribeOptions) (whistle.Result, error) {
	if opts.Language == "xx" {
		return whistle.Result{}, errors.New(`whistle: unknown language "xx"`)
	}

	if opts.Depth == 1 {
		return whistle.Result{}, errors.New("whistle: depth 1 outside 2..8")
	}

	if len(pcm) > whistle.MaxSamples {
		return whistle.Result{}, fmt.Errorf("whistle: %w", whistle.ErrTooLong)
	}

	if len(pcm) == 0 {
		return whistle.Result{}, nil
	}

	f.mu.Lock()
	f.opts = append(f.opts, opts)
	f.mu.Unlock()

	if f.run != nil {
		f.run <- struct{}{}
	}

	if f.block != nil {
		<-f.block
	}

	return whistle.Result{Text: fmt.Sprintf("%d samples, first %.4f", len(pcm), pcm[0]), Language: "en"}, nil
}

func (f *fakeModel) NewStream(opts whistle.StreamOptions) (Stream, error) {
	if opts.Language == "xx" {
		return nil, errors.New(`whistle: unknown language "xx"`)
	}

	return &fakeStream{m: f}, nil
}

func (f *fakeModel) lastOpts() whistle.TranscribeOptions {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.opts[len(f.opts)-1]
}

// fakeStream commits a word per call: the call's sample count.
type fakeStream struct {
	m        *fakeModel
	received int
}

func (s *fakeStream) Process(pcm []float32) (whistle.StreamResult, error) {
	if s.m.run != nil {
		s.m.run <- struct{}{}
	}

	if s.m.block != nil {
		<-s.m.block
	}

	s.received += len(pcm)
	w := strconv.Itoa(len(pcm))

	return whistle.StreamResult{
		Text: w, Words: []whistle.Word{{Text: w, Probability: 1}}, Pending: "next", Language: "en",
		Received: float64(s.received) / whistle.SampleRate,
	}, nil
}

func (s *fakeStream) Stop() whistle.StreamResult {
	return whistle.StreamResult{
		Text: "next", Words: []whistle.Word{{Text: "next"}}, Language: "en", Received: float64(s.received) / whistle.SampleRate,
	}
}

func testConfig() Config {
	return Config{
		MaxBody: 1 << 20, Concurrency: 1, Queue: 1, MaxStreams: 2, StreamIdle: time.Minute, ReadTimeout: time.Minute,
	}
}

// newTestServer serves a fakeModel with cfg.
func newTestServer(t *testing.T, f *fakeModel, cfg Config) (*Server, *httptest.Server) {
	t.Helper()

	s, err := New(f, cfg)
	if err != nil {
		t.Fatal(err)
	}

	hs := httptest.NewServer(s)
	t.Cleanup(hs.Close)

	return s, hs
}

// wav is x as a 16-bit mono 16 kHz WAV file.
func wav(x []int16) []byte {
	const rate = whistle.SampleRate

	size := uint32(2 * len(x)) // #nosec G115 -- a test clip.
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, 36+size)
	b = append(b, "WAVEfmt "...)

	for _, v := range []uint32{16, 1 | 1<<16, rate, 2 * rate, 2 | 16<<16} {
		b = binary.LittleEndian.AppendUint32(b, v)
	}

	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, size)

	for _, v := range x {
		b = binary.LittleEndian.AppendUint16(b, uint16(v))
	}

	return b
}

// wav8 is an 8-bit mono WAV of n frames at rate, a ramp.
func wav8(rate, n int) []byte {
	size := uint32(n) // #nosec G115 -- a test clip.
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, 36+size)
	b = append(b, "WAVEfmt "...)

	for _, v := range []int{16, 1 | 1<<16, rate, rate, 1 | 8<<16} {
		b = binary.LittleEndian.AppendUint32(b, uint32(v)) // #nosec G115 -- a test clip.
	}

	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, size)

	for i := range n {
		b = append(b, byte(i))
	}

	return b
}

func post(t *testing.T, url string, body io.Reader) (int, string, http.Header) {
	t.Helper()

	resp, err := http.Post(url, "application/octet-stream", body)
	if err != nil {
		t.Error(err)

		return 0, "", nil
	}

	defer func() { _ = resp.Body.Close() }()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Error(err)
	}

	return resp.StatusCode, string(b), resp.Header
}

func TestNewChecksConfig(t *testing.T) {
	for _, c := range []struct {
		edit func(*Config)
		want string
	}{
		{func(c *Config) { c.MaxBody = 0 }, "max body"},
		{func(c *Config) { c.Concurrency = 0 }, "concurrency"},
		{func(c *Config) { c.Queue = -1 }, "queue"},
		{func(c *Config) { c.MaxStreams = 0 }, "max streams"},
		{func(c *Config) { c.StreamIdle = 0 }, "stream idle"},
		{func(c *Config) { c.ReadTimeout = 0 }, "read timeout"},
	} {
		cfg := testConfig()
		c.edit(&cfg)

		_, err := New(&fakeModel{}, cfg)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v", c.want, err)
		}
	}
}

func TestTranscribe(t *testing.T) {
	f := &fakeModel{}
	_, hs := newTestServer(t, f, testConfig())

	var f32 bytes.Buffer

	_ = binary.Write(&f32, binary.LittleEndian, []float32{0.25, 0, 0})

	for _, c := range []struct {
		query string
		body  []byte
		want  string
		opts  whistle.TranscribeOptions
	}{
		{"", wav([]int16{16384, 0, 0, 0}), "4 samples, first 0.5000", whistle.TranscribeOptions{}},
		{
			"?format=wav&word_timestamps", wav([]int16{-16384, 0}), "2 samples, first -0.5000",
			whistle.TranscribeOptions{WordTimestamps: true},
		},
		{
			"?format=s16le&language=de&depth=4&word_timestamps=false",
			[]byte{0, 0x40, 0, 0},
			"2 samples, first 0.5000",
			whistle.TranscribeOptions{Language: "de", Depth: 4},
		},
		{
			"?format=f32le&keywords=Concord&keywords=New+York%0Atents&word_timestamps=1", f32.Bytes(),
			"3 samples, first 0.2500",
			whistle.TranscribeOptions{
				Keywords: []string{"Concord", "New York\ntents"}, WordTimestamps: true,
			},
		},
	} {
		status, body, h := post(t, hs.URL+"/transcribe"+c.query, bytes.NewReader(c.body))

		want := `{"text":"` + c.want + `","language":"en","ttft_ms":0.0,"decode_tps":0.0}` + "\n"
		if status != http.StatusOK || body != want || h.Get("Content-Type") != "application/json" {
			t.Errorf("%q: %d %q (%s), want 200 %q", c.query, status, body, h.Get("Content-Type"), want)

			continue
		}

		if got := f.lastOpts(); fmt.Sprint(got) != fmt.Sprint(c.opts) {
			t.Errorf("%q: options %+v, want %+v", c.query, got, c.opts)
		}
	}
}

func TestTranscribeErrors(t *testing.T) {
	cfg := testConfig()
	cfg.MaxBody = 4*whistle.MaxSamples + 64
	_, hs := newTestServer(t, &fakeModel{}, cfg)

	tooLong := make([]byte, 2*(whistle.MaxSamples+1))
	oddTooLong := make([]byte, 4*(whistle.MaxSamples+1)+1)
	floatWAV := wav([]int16{1, 2})
	floatWAV[20] = 3 // IEEE float

	for _, c := range []struct {
		name, query string
		body        []byte
		status      int
		want        string
	}{
		{"unknown parameter", "?lang=en", nil, 400, `unknown parameter \"lang\"`},
		{"repeated parameter", "?depth=2&depth=3", nil, 400, `\"depth\" given 2 times`},
		{"bad format", "?format=u8", nil, 400, "s16le or f32le, or wav"},
		{"bad depth", "?depth=x", nil, 400, `depth \"x\"`},
		{"model's depth check", "?depth=1", nil, 400, `"depth 1 outside 2..8"`},
		{"bad language", "?language=xx", wav([]int16{1}), 400, `"unknown language \"xx\""`},
		{"bad word_timestamps", "?word_timestamps=maybe", nil, 400, "want true or false"},
		{"not a WAV", "", []byte("hello world"), 400, "wav body: "},
		{"float WAV", "", floatWAV, 400, "wav body: "},
		{"empty WAV", "", wav(nil), 400, `"the body holds no audio"`},
		{"empty PCM", "?format=s16le", nil, 400, `"the body holds no audio"`},
		{"odd PCM", "?format=f32le", []byte{1, 2, 3, 4, 5}, 400, "ends inside a sample"},
		{"over 30 s", "?format=s16le", tooLong, 422, "audio limit is 30 s: 480001 samples, at most 480000"},
		{"over 30 s f32le", "?format=f32le", oddTooLong[1:], 422, "480001 samples, at most 480000"},
		{"odd PCM over 30 s", "?format=f32le", oddTooLong, 400, "ends inside a sample"},
		{"over 30 s WAV", "", wav(make([]int16, whistle.MaxSamples+1)), 422, "/stream takes longer audio"},
		// 40 s at 1 kHz: 40 kB under the body limit, 640,000 samples at 16 kHz.
		{"over 30 s at 1 kHz", "", wav8(1000, 40000), 422, "640000 samples, at most 480000"},
		// 1,323,002 frames at 44.1 kHz resample to 480,000.7, rounded to 480,001.
		{"over 30 s at 44.1 kHz", "", wav8(44100, 1323002), 422, "480001 samples, at most 480000"},
	} {
		status, body, _ := post(t, hs.URL+"/transcribe"+c.query, bytes.NewReader(c.body))
		if status != c.status || !strings.HasPrefix(body, `{`) || !strings.Contains(body, c.want) {
			t.Errorf("%s: %d %s; want %d with %s", c.name, status, body, c.status, c.want)
		}
	}
}

// TestTranscribeLowRateWAV resamples a WAV under 30 s as DecodeWAV does,
// though the server decodes it a second at a time.
func TestTranscribeLowRateWAV(t *testing.T) {
	_, hs := newTestServer(t, &fakeModel{}, testConfig())

	for _, rate := range []int{1000, 8000, 44100} {
		data := wav8(rate, 2*rate+7)

		want, err := frontend.DecodeWAV(data)
		if err != nil {
			t.Fatal(err)
		}

		status, body, _ := post(t, hs.URL+"/transcribe", bytes.NewReader(data))
		if prefix := fmt.Sprintf(`{"text":"%d samples, first %.4f"`, len(want), want[0]); status != http.StatusOK ||
			!strings.HasPrefix(body, prefix) {
			t.Errorf("%d Hz: %d %s, want 200 %s", rate, status, body, prefix)
		}
	}
}

func TestTranscribeBodyLimit(t *testing.T) {
	cfg := testConfig()
	cfg.MaxBody = 100
	_, hs := newTestServer(t, &fakeModel{}, cfg)

	// With a Content-Length and without one (chunked).
	for _, body := range []io.Reader{bytes.NewReader(make([]byte, 101)), io.MultiReader(bytes.NewReader(make([]byte, 101)))} {
		status, msg, _ := post(t, hs.URL+"/transcribe?format=s16le", body)
		if status != http.StatusRequestEntityTooLarge || !strings.Contains(msg, "over the limit of 100 bytes") {
			t.Errorf("101 bytes: %d %s", status, msg)
		}
	}

	status, _, _ := post(t, hs.URL+"/transcribe?format=s16le", bytes.NewReader(make([]byte, 100)))
	if status != http.StatusOK {
		t.Errorf("100 bytes: %d", status)
	}
}

// A body slower than ReadTimeout gets 408.
func TestTranscribeReadTimeout(t *testing.T) {
	cfg := testConfig()
	cfg.ReadTimeout = 100 * time.Millisecond
	_, hs := newTestServer(t, &fakeModel{}, cfg)

	body, w := io.Pipe()
	defer func() { _ = w.Close() }()

	go func() { _, _ = w.Write([]byte{0, 0}) }() // then nothing

	status, msg, _ := post(t, hs.URL+"/transcribe?format=s16le", body)
	if status != http.StatusRequestTimeout || !strings.Contains(msg, "the body took over 100ms") {
		t.Errorf("slow body: %d %s, want 408", status, msg)
	}
}

// With one pass running and one request queued, the next gets 503; the
// queued one runs once the pass ends.
func TestTranscribeQueue(t *testing.T) {
	f := &fakeModel{block: make(chan struct{}), run: make(chan struct{})}
	s, hs := newTestServer(t, f, testConfig()) // 1 pass, a queue of 1

	url := hs.URL + "/transcribe?format=s16le"
	results := make(chan int, 2)

	go func() {
		status, _, _ := post(t, url, bytes.NewReader([]byte{1, 0}))
		results <- status
	}()

	<-f.run // the first request runs

	go func() {
		status, _, _ := post(t, url, bytes.NewReader([]byte{2, 0}))
		results <- status
	}()

	for deadline := time.Now().Add(5 * time.Second); s.gate.waiting.Load() < 1; time.Sleep(time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the second request never queued")
		}
	}

	status, body, h := post(t, url, bytes.NewReader([]byte{3, 0}))
	if status != http.StatusServiceUnavailable || h.Get("Retry-After") != "1" ||
		!strings.Contains(body, "1 passes run and 1 requests wait") {
		t.Errorf("third request: %d %s, Retry-After %q; want 503", status, body, h.Get("Retry-After"))
	}

	f.block <- struct{}{} // the first finishes

	<-f.run // the second runs

	f.block <- struct{}{}

	for range 2 {
		if status := <-results; status != http.StatusOK {
			t.Errorf("queued request: %d", status)
		}
	}
}

func TestRoutes(t *testing.T) {
	var logged syncBuffer

	cfg := testConfig()
	cfg.Log = log.New(&logged, "", 0)
	cfg.Info = Info{Model: "whistle.cact", SHA256: "abc", Format: ".cact", Weights: "packed", Languages: []string{"en"}}
	s, hs := newTestServer(t, &fakeModel{}, cfg)

	for _, c := range []struct {
		method, path string
		status       int
		allow, want  string
	}{
		{"GET", "/transcribe", 405, "POST", `"/transcribe takes POST, not GET"`},
		{"POST", "/stream", 405, "GET", "takes GET"},
		{"POST", "/healthz", 405, "GET", "takes GET"},
		{"DELETE", "/v1/info", 405, "GET", "takes GET"},
		{"GET", "/nothing", 404, "", `"no endpoint /nothing"`},
		{"GET", "/healthz", 200, "", "ok\n"},
		{"HEAD", "/healthz", 200, "", ""},
		{"GET", "/v1/info", 200, "", `"sha256": "abc"`},
		{"GET", "/v1/info", 200, "", `"max_body": 1048576`},
		{"GET", "/v1/info", 200, "", `"max_audio_seconds": 30`},
	} {
		req, _ := http.NewRequest(c.method, hs.URL+c.path, nil)

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}

		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode != c.status || resp.Header.Get("Allow") != c.allow || !strings.Contains(string(body), c.want) {
			t.Errorf("%s %s: %d (Allow %q) %q; want %d (Allow %q) with %q", c.method, c.path, resp.StatusCode,
				resp.Header.Get("Allow"), body, c.status, c.allow, c.want)
		}
	}

	var info struct {
		Info

		Limits limits `json:"limits"`
	}

	resp, err := http.Get(hs.URL + "/v1/info")
	if err != nil {
		t.Fatal(err)
	}

	err = json.NewDecoder(resp.Body).Decode(&info)
	_ = resp.Body.Close()

	if err != nil || fmt.Sprint(info.Info) != fmt.Sprint(cfg.Info) || info.Limits.StreamIdle != 60 {
		t.Errorf("info = %+v, %v", info, err)
	}

	err = s.Shutdown(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	resp, err = http.Get(hs.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}

	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("healthz while shutting down: %d", resp.StatusCode)
	}

	for _, line := range []string{"GET /transcribe 405 ", "GET /nothing 404 ", "GET /v1/info 200 ", "GET /healthz 503 "} {
		if !strings.Contains(logged.String(), line) {
			t.Errorf("log has no %q:\n%s", line, logged.String())
		}
	}
}

// syncBuffer is a bytes.Buffer for concurrent writers.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.b.String()
}

// wsURL is the /stream URL of hs with query.
func wsURL(hs *httptest.Server, query string) string {
	return "ws" + strings.TrimPrefix(hs.URL, "http") + "/stream" + query
}

// dial opens /stream with query.
func dial(t *testing.T, hs *httptest.Server, query string) *websocket.Conn {
	t.Helper()

	c, resp, err := websocket.Dial(context.Background(), wsURL(hs, query), nil) //nolint:bodyclose // Dial closes it
	if err != nil {
		t.Fatalf("dial: %v (%v)", err, resp)
	}

	t.Cleanup(func() { _ = c.CloseNow() })

	return c
}

// readText reads one text message.
func readText(t *testing.T, c *websocket.Conn) string {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	typ, b, err := c.Read(ctx)
	if err != nil || typ != websocket.MessageText {
		t.Fatalf("read: %v, %v %q", err, typ, b)
	}

	return string(b)
}

// readClose reads until the close and returns its status and reason.
func readClose(t *testing.T, c *websocket.Conn) (websocket.StatusCode, string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, b, err := c.Read(ctx)

	var ce websocket.CloseError
	if !errors.As(err, &ce) {
		t.Fatalf("read %q, %v; want a close", b, err)
	}

	return ce.Code, ce.Reason
}

func write(t *testing.T, c *websocket.Conn, typ websocket.MessageType, b []byte) {
	t.Helper()

	err := c.Write(context.Background(), typ, b)
	if err != nil {
		t.Fatal(err)
	}
}

func TestStream(t *testing.T) {
	_, hs := newTestServer(t, &fakeModel{}, testConfig())

	for _, c := range []struct {
		query string
		chunk []byte
	}{
		{"", make([]byte, 2*16000)},
		{"?format=s16le&language=en&keywords=a&keywords=b", make([]byte, 2*16000)},
		{"?format=f32le", make([]byte, 4*16000)},
	} {
		ws := dial(t, hs, c.query)

		for i := 1; i <= 2; i++ {
			write(t, ws, websocket.MessageBinary, c.chunk)

			want := fmt.Sprintf(`{"text":"16000","words":[{"word":"16000","start":0.00,"end":0.00,"probability":1.000}],`+
				`"pending":"next","language":"en","received":%d.00,"pass_ms":0.0}`, i)
			if got := readText(t, ws); got != want {
				t.Errorf("%q call %d: %s, want %s", c.query, i, got, want)
			}
		}

		write(t, ws, websocket.MessageText, []byte(`{"type": "stop"}`))

		want := `{"text":"next","words":[{"word":"next","start":0.00,"end":0.00,"probability":0.000}],` +
			`"pending":"","language":"en","received":2.00,"pass_ms":0.0}`
		if got := readText(t, ws); got != want {
			t.Errorf("%q stop: %s, want %s", c.query, got, want)
		}

		if code, _ := readClose(t, ws); code != websocket.StatusNormalClosure {
			t.Errorf("%q: close %v, want 1000", c.query, code)
		}
	}
}

func TestStreamProtocolErrors(t *testing.T) {
	_, hs := newTestServer(t, &fakeModel{}, testConfig())

	for _, c := range []struct {
		name, query string
		typ         websocket.MessageType
		msg         []byte
		code        websocket.StatusCode
	}{
		{"other text", "", websocket.MessageText, []byte(`{"type":"start"}`), websocket.StatusUnsupportedData},
		{"odd bytes", "", websocket.MessageBinary, []byte{1, 2, 3}, websocket.StatusInvalidFramePayloadData},
		{"f32 inside a sample", "?format=f32le", websocket.MessageBinary, []byte{1, 2}, websocket.StatusInvalidFramePayloadData},
		{"30 s and a byte", "", websocket.MessageBinary, make([]byte, 2*whistle.MaxSamples+1), websocket.StatusInvalidFramePayloadData},
		{"f32 30 s and 3 bytes", "?format=f32le", websocket.MessageBinary, make([]byte, 4*whistle.MaxSamples+3), websocket.StatusInvalidFramePayloadData},
		{"over 30 s", "", websocket.MessageBinary, make([]byte, 2*whistle.MaxSamples+2), websocket.StatusMessageTooBig},
		{"f32 over 30 s", "?format=f32le", websocket.MessageBinary, make([]byte, 4*whistle.MaxSamples+4), websocket.StatusMessageTooBig},
	} {
		ws := dial(t, hs, c.query)

		// The server may close while a long message is still being written.
		_ = ws.Write(context.Background(), c.typ, c.msg)

		if code, reason := readClose(t, ws); code != c.code {
			t.Errorf("%s: close %v %q, want %v", c.name, code, reason, c.code)
		}
	}
}

// 30 s in one message is one Process call, not a close.
func TestStreamThirtySeconds(t *testing.T) {
	_, hs := newTestServer(t, &fakeModel{}, testConfig())
	ws := dial(t, hs, "?format=f32le")

	write(t, ws, websocket.MessageBinary, make([]byte, 4*whistle.MaxSamples))

	if got := readText(t, ws); !strings.Contains(got, `"text":"480000"`) {
		t.Errorf("30 s: %s", got)
	}
}

func TestStreamBeforeUpgrade(t *testing.T) {
	cfg := testConfig()
	cfg.MaxStreams = 1
	_, hs := newTestServer(t, &fakeModel{}, cfg)

	for query, want := range map[string]string{
		"?format=wav":    "s16le or f32le",
		"?language=xx":   "unknown language",
		"?depth=4":       `unknown parameter \"depth\"`,
		"?format=a&lang": "unknown parameter",
	} {
		resp, err := http.Get(hs.URL + "/stream" + query)
		if err != nil {
			t.Fatal(err)
		}

		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode != http.StatusBadRequest || !strings.Contains(string(body), want) {
			t.Errorf("%s: %d %s, want 400 with %s", query, resp.StatusCode, body, want)
		}
	}

	// A plain GET is no WebSocket handshake.
	resp, err := http.Get(hs.URL + "/stream")
	if err != nil {
		t.Fatal(err)
	}

	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusUpgradeRequired {
		t.Errorf("plain GET: %d, want 426", resp.StatusCode)
	}

	// The second session over MaxStreams.
	dial(t, hs, "")

	_, resp, err = websocket.Dial(context.Background(), wsURL(hs, ""), nil) //nolint:bodyclose // Dial closes it
	if err == nil || resp == nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("second stream: %v, %v; want 503", err, resp)
	}
}

// A session silent for StreamIdle gets the Stop result and 1008.
func TestStreamIdle(t *testing.T) {
	cfg := testConfig()
	cfg.StreamIdle = 100 * time.Millisecond
	_, hs := newTestServer(t, &fakeModel{}, cfg)
	ws := dial(t, hs, "")

	write(t, ws, websocket.MessageBinary, make([]byte, 3200))
	readText(t, ws)

	if got := readText(t, ws); !strings.Contains(got, `"text":"next"`) || !strings.Contains(got, `"received":0.10`) {
		t.Errorf("idle stop: %s", got)
	}

	if code, reason := readClose(t, ws); code != websocket.StatusPolicyViolation || reason != "idle for 100ms" {
		t.Errorf("close %v %q, want 1008", code, reason)
	}
}

// Shutdown sends each session its Stop result and closes with 1001.
func TestStreamShutdown(t *testing.T) {
	s, hs := newTestServer(t, &fakeModel{}, testConfig())
	ws := dial(t, hs, "")

	write(t, ws, websocket.MessageBinary, make([]byte, 2*16000))
	readText(t, ws)

	done := make(chan error, 1)

	go func() { done <- s.Shutdown(context.Background()) }()

	if got := readText(t, ws); !strings.Contains(got, `"text":"next"`) {
		t.Errorf("shutdown stop: %s", got)
	}

	if code, _ := readClose(t, ws); code != websocket.StatusGoingAway {
		t.Errorf("close %v, want 1001", code)
	}

	err := <-done
	if err != nil {
		t.Errorf("Shutdown: %v", err)
	}

	_, resp, err := websocket.Dial(context.Background(), wsURL(hs, ""), nil) //nolint:bodyclose // Dial closes it
	if err == nil || resp == nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("stream after shutdown: %v, %v; want 503", err, resp)
	}
}

// A client that leaves while its chunk waits for a pass ends its session at
// once, also with more chunks queued behind it: the session frees its place
// under MaxStreams and never takes the pass (PR #35, PR #37 review).
func TestStreamLeaveWhileWaiting(t *testing.T) {
	for _, chunks := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("%d chunks", chunks), func(t *testing.T) {
			f := &fakeModel{block: make(chan struct{}), run: make(chan struct{})}
			s, hs := newTestServer(t, f, testConfig()) // one pass, two streams

			a := dial(t, hs, "")
			write(t, a, websocket.MessageBinary, make([]byte, 2))
			<-f.run // a holds the pass

			b := dial(t, hs, "")
			for range chunks {
				write(t, b, websocket.MessageBinary, make([]byte, 2))
			}

			time.Sleep(50 * time.Millisecond) // b's first chunk waits for the pass, the rest behind it

			_ = b.CloseNow()

			for deadline := time.Now().Add(5 * time.Second); s.openSessions() > 1; time.Sleep(10 * time.Millisecond) {
				if time.Now().After(deadline) {
					t.Fatalf("%d sessions open after b left, want 1", s.openSessions())
				}
			}

			close(f.block)

			if got := readText(t, a); !strings.Contains(got, `"text":"1"`) {
				t.Errorf("a: %s", got)
			}

			select {
			case <-f.run:
				t.Error("b's chunk took the pass after b left")
			case <-time.After(100 * time.Millisecond):
			}
		})
	}
}

// A client more than the backlog ahead of the passes is closed with 1008
// and its chunks never take the pass (PR #37 review).
func TestStreamAhead(t *testing.T) {
	f := &fakeModel{block: make(chan struct{}), run: make(chan struct{})}
	_, hs := newTestServer(t, f, testConfig())

	a := dial(t, hs, "")
	write(t, a, websocket.MessageBinary, make([]byte, 2))
	<-f.run // a holds the pass

	// b's first 30 s chunk waits at the gate; the next two fill the 60 s
	// backlog, the fourth is over it.
	b := dial(t, hs, "")
	chunk := make([]byte, 2*whistle.MaxSamples)

	for range 4 {
		write(t, b, websocket.MessageBinary, chunk)
	}

	if code, reason := readClose(t, b); code != websocket.StatusPolicyViolation || !strings.Contains(reason, "wait for a pass") {
		t.Errorf("b closed with %v %q, want 1008", code, reason)
	}

	close(f.block)

	if got := readText(t, a); !strings.Contains(got, `"text":"1"`) {
		t.Errorf("a: %s", got)
	}

	select {
	case <-f.run:
		t.Error("b's chunk took the pass after its close")
	case <-time.After(100 * time.Millisecond):
	}
}

// openSessions is the number of open /stream sessions.
func (s *Server) openSessions() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.sessions
}

// A session still in a pass at Shutdown's deadline is dropped, and Shutdown
// says so.
func TestStreamShutdownDeadline(t *testing.T) {
	f := &fakeModel{block: make(chan struct{}), run: make(chan struct{})}
	s, hs := newTestServer(t, f, testConfig())
	ws := dial(t, hs, "")

	write(t, ws, websocket.MessageBinary, make([]byte, 2))
	<-f.run

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	err := s.Shutdown(ctx)
	if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "streams still open") {
		t.Errorf("Shutdown: %v", err)
	}

	close(f.block)

	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, _, err = ws.Read(ctx)
	if err == nil || errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("read after the drop: %v, want the closed connection", err)
	}
}

func TestGate(t *testing.T) {
	g := newGate(1, 0)
	ctx := context.Background()

	err := g.enter(ctx, true)
	if err != nil {
		t.Fatal(err)
	}

	err = g.enter(ctx, true)
	if !errors.Is(err, errBusy) {
		t.Errorf("full, no queue: %v, want errBusy", err)
	}

	short, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()

	err = g.enter(short, false)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("unqueued wait: %v, want the deadline", err)
	}

	g.leave()

	// An ended context gets no slot, though one is free, and leaves it free.
	gone, stop := context.WithCancel(ctx)
	stop()

	for _, queued := range []bool{true, false} {
		err = g.enter(gone, queued)
		if !errors.Is(err, context.Canceled) {
			t.Errorf("ended context, queued %v: %v, want context.Canceled", queued, err)
		}
	}

	err = g.enter(ctx, true)
	if err != nil {
		t.Errorf("after leave: %v", err)
	}
}

func TestIsStop(t *testing.T) {
	for _, c := range []struct {
		msg  string
		want bool
	}{
		{`{"type":"stop"}`, true},
		{"{ \"type\" : \"stop\" }\n", true},
		{`{"type":"Stop"}`, false},
		{`stop`, false},
		{`{}`, false},
	} {
		if isStop([]byte(c.msg)) != c.want {
			t.Errorf("isStop(%s) = %v", c.msg, !c.want)
		}
	}
}

func TestTooLongMessage(t *testing.T) {
	got := tooLong(math.MaxInt32)
	if !strings.HasPrefix(got, "audio limit is 30 s: 2147483647 samples, at most 480000") {
		t.Errorf("tooLong = %q", got)
	}
}
