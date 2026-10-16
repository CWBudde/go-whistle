package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

const (
	// maxBody bounds a request body. The context holds 8192 tokens, a few
	// tens of KiB of text; a refused turn must still arrive to be counted.
	maxBody = 4 << 20
	// readHeaderTimeout bounds the request line and headers of a request.
	readHeaderTimeout = 10 * time.Second
	// shutdownTimeout bounds the wait for the open connections at exit.
	shutdownTimeout = 30 * time.Second
)

// server answers one conversation over HTTP, one request at a time.
type server struct {
	mu      sync.Mutex // held while a request uses the agent
	a       agent
	maxNew  int
	stopped bool // the model closes: refuse what still waits for mu
}

func newServer(a agent, maxNew int) *server { return &server{a: a, maxNew: maxNew} }

// handler routes POST /complete and POST /reset; another method on them
// gets 405, another path 404.
func (s *server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /complete", s.complete)
	mux.HandleFunc("POST /reset", s.reset)

	return mux
}

// complete answers {"input": "…"} as the next turn with the envelope, a
// refused turn with the engine's refusal, both with 200, as the engine's
// --serve does (turns.json, history.json).
func (s *server) complete(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		if mbe, ok := errors.AsType[*http.MaxBytesError](err); ok {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("the body is over %d bytes", mbe.Limit))

			return
		}

		writeError(w, http.StatusBadRequest, "reading the body: "+err.Error())

		return
	}

	var req struct {
		Input *string `json:"input"`
	}

	err = json.Unmarshal(body, &req)
	if err != nil || req.Input == nil {
		writeError(w, http.StatusBadRequest, `the body is not {"input": "…"}`)

		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		writeError(w, http.StatusServiceUnavailable, "shutting down")

		return
	}

	resp, err := complete(s.a, *req.Input, s.maxNew)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())

		return
	}

	js, err := encode(resp)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())

		return
	}

	writeJSON(w, http.StatusOK, js)
}

// reset starts a new conversation; the body is not read.
func (s *server) reset(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		writeError(w, http.StatusServiceUnavailable, "shutting down")

		return
	}

	s.a.Reset()
	writeJSON(w, http.StatusOK, []byte("{\"ok\":true}\n"))
}

// stop waits for the running request and refuses the ones after it.
func (s *server) stop() {
	s.mu.Lock()
	s.stopped = true
	s.mu.Unlock()
}

func writeJSON(w http.ResponseWriter, status int, js []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(js)
}

// errorBody is the body of an error answer.
type errorBody struct {
	Error string `json:"error"`
}

// writeError answers {"error": msg}.
func writeError(w http.ResponseWriter, status int, msg string) {
	js, err := json.Marshal(errorBody{msg})
	if err != nil {
		http.Error(w, msg, status)

		return
	}

	writeJSON(w, status, append(js, '\n'))
}

// serve answers the conversation of a on 127.0.0.1:port until ctx ends,
// after a line on stderr that it serves.
func (c *cli) serve(ctx context.Context, a agent, maxNew, port int) int {
	l, err := (&net.ListenConfig{}).Listen(ctx, "tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		return c.fail(err)
	}

	s := newServer(a, maxNew)
	srv := &http.Server{Handler: s.handler(), ReadHeaderTimeout: readHeaderTimeout}

	done := make(chan error, 1)

	go func() { done <- srv.Serve(l) }()

	// The oracle waits for "serving on" (tools/oracle/text/cli.go).
	c.warn(fmt.Sprintf("serving on http://%s (POST /complete, POST /reset)", l.Addr()))

	select {
	case err := <-done:
		return c.fail(err)
	case <-ctx.Done():
	}

	// The running request finishes, the ones after it get 503; then no
	// call runs in the agent, and the model may close whatever Shutdown
	// returns.
	s.stop()

	sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), shutdownTimeout)
	defer cancel()

	_ = srv.Shutdown(sctx)

	return 0
}
