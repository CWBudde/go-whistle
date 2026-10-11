package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cwbudde/go-whistle/whistle"
)

// Model is the part of *whistle.Model the server runs; FromModel adapts one.
type Model interface {
	Transcribe(pcm []float32, opts whistle.TranscribeOptions) (whistle.Result, error)
	NewStream(opts whistle.StreamOptions) (Stream, error)
}

// Stream is the part of *whistle.Stream a /stream session drives.
type Stream interface {
	Process(pcm []float32) (whistle.StreamResult, error)
	Stop() whistle.StreamResult
}

// FromModel is m as a Model.
func FromModel(m *whistle.Model) Model { return model{m} }

type model struct{ *whistle.Model }

func (m model) NewStream(opts whistle.StreamOptions) (Stream, error) {
	s, err := m.Model.NewStream(opts)
	if err != nil {
		return nil, err
	}

	return s, nil
}

// Config are the server's limits and what /v1/info reports.
type Config struct {
	MaxBody        int64         // bytes of a /transcribe body, at least 1
	Concurrency    int           // passes that run at once, at least 1
	Queue          int           // /transcribe requests that wait for a pass, at least 0
	MaxStreams     int           // open /stream sessions, at least 1
	StreamIdle     time.Duration // a session without a message for this long ends, > 0
	ReadTimeout    time.Duration // the longest a /transcribe body may take to arrive, > 0
	OriginPatterns []string      // origins besides the host's that may open a /stream (websocket.AcceptOptions)
	Info           Info          // the model as /v1/info names it
	Log            *log.Logger   // a line per request; nil logs nothing
}

// Info describes the loaded model in /v1/info.
type Info struct {
	Model     string   `json:"model"`     // the file's name
	SHA256    string   `json:"sha256"`    // its hex SHA-256
	Format    string   `json:"format"`    // .cact or .safetensors
	Weights   string   `json:"weights"`   // how Load runs them: packed or float32
	Languages []string `json:"languages"` // the codes language takes
}

// Server is the http.Handler of the endpoints on one model.
type Server struct {
	m    Model
	cfg  Config
	mux  *http.ServeMux
	gate *gate

	admitted atomic.Int64 // /transcribe requests past the option checks

	mu       sync.Mutex
	closed   bool           // Shutdown has begun: no new session
	sessions int            // open /stream sessions
	wg       sync.WaitGroup // one per open session

	stopping chan struct{} // closed when Shutdown begins: sessions stop
	dropping chan struct{} // closed when Shutdown's deadline passes: sessions drop
	drop     sync.Once
}

// Check rejects limits out of their ranges.
func (c Config) Check() error {
	switch {
	case c.MaxBody < 1:
		return fmt.Errorf("max body %d bytes, want at least 1", c.MaxBody)
	case c.Concurrency < 1:
		return fmt.Errorf("concurrency %d, want at least 1", c.Concurrency)
	case c.Queue < 0:
		return fmt.Errorf("queue %d, want at least 0", c.Queue)
	case c.MaxStreams < 1:
		return fmt.Errorf("max streams %d, want at least 1", c.MaxStreams)
	case c.StreamIdle <= 0:
		return fmt.Errorf("stream idle %v, want more than 0", c.StreamIdle)
	case c.ReadTimeout <= 0:
		return fmt.Errorf("read timeout %v, want more than 0", c.ReadTimeout)
	}

	return nil
}

// New builds the server on m. It does not close m.
func New(m Model, cfg Config) (*Server, error) {
	err := cfg.Check()
	if err != nil {
		return nil, fmt.Errorf("server: %w", err)
	}

	s := &Server{
		m: m, cfg: cfg, mux: http.NewServeMux(), gate: newGate(cfg.Concurrency, cfg.Queue),
		stopping: make(chan struct{}), dropping: make(chan struct{}),
	}

	s.route("/transcribe", http.MethodPost, s.transcribe)
	s.route("/stream", http.MethodGet, s.stream)
	s.route("/healthz", http.MethodGet, s.healthz)
	s.route("/v1/info", http.MethodGet, s.info)
	s.mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "no endpoint "+r.URL.Path)
	})

	return s, nil
}

// ServeHTTP serves the endpoints and logs a line per request: method, path,
// status and duration (for /stream the whole session).
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if s.cfg.Log == nil {
		s.mux.ServeHTTP(w, r)

		return
	}

	start := time.Now()
	rec := &recorder{ResponseWriter: w}

	s.mux.ServeHTTP(rec, r)

	if rec.status == 0 {
		rec.status = http.StatusOK
	}

	ms := float64(time.Since(start)) / float64(time.Millisecond)
	s.cfg.Log.Printf("%s %s %d %.1fms", r.Method, r.URL.Path, rec.status, ms)
}

// Shutdown ends the /stream sessions, each with its Stop result, and waits
// for them until ctx ends; then it drops the rest and returns ctx's error.
// The http.Server's Shutdown, which does not track hijacked connections,
// handles the /transcribe requests.
func (s *Server) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		close(s.stopping)
	}
	s.mu.Unlock()

	done := make(chan struct{})

	go func() {
		s.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		s.drop.Do(func() { close(s.dropping) })

		return fmt.Errorf("server: streams still open: %w", ctx.Err())
	}
}

// route serves path with h for method (and HEAD for GET), 405 otherwise.
func (s *Server) route(path, method string, h http.HandlerFunc) {
	s.mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method && (method != http.MethodGet || r.Method != http.MethodHead) {
			w.Header().Set("Allow", method)
			writeError(w, http.StatusMethodNotAllowed, fmt.Sprintf("%s takes %s, not %s", path, method, r.Method))

			return
		}

		h(w, r)
	})
}

// open registers a session; false when MaxStreams are open or the server
// shuts down, with the reason.
func (s *Server) open() (bool, string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case s.closed:
		return false, "the server is shutting down"
	case s.sessions >= s.cfg.MaxStreams:
		return false, fmt.Sprintf("%d streams are open, the limit", s.sessions)
	}

	s.sessions++
	s.wg.Add(1)

	return true, ""
}

func (s *Server) close() {
	s.mu.Lock()
	s.sessions--
	s.mu.Unlock()
	s.wg.Done()
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")

	if closed {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("shutting down\n"))

		return
	}

	_, _ = w.Write([]byte("ok\n"))
}

// limits is the Config part of /v1/info.
type limits struct {
	MaxBody         int64   `json:"max_body"`
	MaxAudioSeconds float64 `json:"max_audio_seconds"`
	Concurrency     int     `json:"concurrency"`
	Queue           int     `json:"queue"`
	MaxStreams      int     `json:"max_streams"`
	StreamIdle      float64 `json:"stream_idle_seconds"`
	ReadTimeout     float64 `json:"read_timeout_seconds"`
}

func (s *Server) info(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, struct {
		Info

		Limits limits `json:"limits"`
	}{s.cfg.Info, limits{
		MaxBody: s.cfg.MaxBody, MaxAudioSeconds: float64(whistle.MaxSamples) / whistle.SampleRate,
		Concurrency: s.cfg.Concurrency, Queue: s.cfg.Queue, MaxStreams: s.cfg.MaxStreams,
		StreamIdle: s.cfg.StreamIdle.Seconds(), ReadTimeout: s.cfg.ReadTimeout.Seconds(),
	}})
}

// writeJSON writes v as indented JSON with status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		status, b = http.StatusInternalServerError, []byte(`{"error":"encode the response"}`)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(append(b, '\n'))
}

// writeError writes {"error":msg} with status.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, struct {
		Error string `json:"error"`
	}{strings.TrimPrefix(msg, "whistle: ")})
}

// recorder notes the status a handler writes, for the log.
type recorder struct {
	http.ResponseWriter

	status int
}

func (r *recorder) WriteHeader(code int) {
	if r.status == 0 {
		r.status = code
	}

	r.ResponseWriter.WriteHeader(code)
}

func (r *recorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}

	return r.ResponseWriter.Write(b)
}

// Unwrap lets http.ResponseController and the WebSocket upgrade reach the
// connection.
func (r *recorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// errBusy reports a full queue.
var errBusy = errors.New("busy")

// gate bounds the passes that run at once and the requests that wait.
type gate struct {
	slots   chan struct{}
	queue   int64
	waiting atomic.Int64
}

func newGate(concurrency, queue int) *gate {
	return &gate{slots: make(chan struct{}, concurrency), queue: int64(queue)}
}

// enter takes a slot, waiting until ctx ends. A queued caller fails with
// errBusy when the queue is full; the others wait without a bound. A caller
// whose ctx has ended never gets a slot, even when one is free or comes
// free as ctx ends, so a client that left takes no pass (PR #37 review).
func (g *gate) enter(ctx context.Context, queued bool) error {
	err := g.take(ctx, queued)
	if err == nil && ctx.Err() != nil {
		g.leave()

		return ctx.Err()
	}

	return err
}

// take is enter without the final check of ctx.
func (g *gate) take(ctx context.Context, queued bool) error {
	select {
	case g.slots <- struct{}{}:
		return nil
	default:
	}

	if queued {
		if g.waiting.Add(1) > g.queue {
			g.waiting.Add(-1)

			return errBusy
		}

		defer g.waiting.Add(-1)
	}

	select {
	case g.slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// leave frees a slot.
func (g *gate) leave() { <-g.slots }
