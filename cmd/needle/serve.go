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
	"sync/atomic"
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
	mu     sync.Mutex // held while a request uses the agent
	a      agent
	maxNew int
	// refusing is set at once when shutdown begins, without waiting for mu:
	// a request that gets mu after it, queued or new, answers 503 and does
	// not reach the agent.
	refusing atomic.Bool
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

	if s.refusing.Load() {
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

	if s.refusing.Load() {
		writeError(w, http.StatusServiceUnavailable, "shutting down")

		return
	}

	s.a.Reset()
	writeJSON(w, http.StatusOK, []byte("{\"ok\":true}\n"))
}

// refuse makes every request that has not yet reached the agent answer
// 503, at once.
func (s *server) refuse() { s.refusing.Store(true) }

// stop refuses the requests after the running one and waits for that one
// to leave the agent; then no call runs in it and the model may close.
func (s *server) stop() {
	s.refuse()
	s.mu.Lock()
	s.mu.Unlock() //nolint:staticcheck // SA2001: the lock only waits for the running request.
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

// serve answers the conversation of a on 127.0.0.1:port until ctx ends
// or a signal of c.signals comes, after a line on stderr that it serves.
func (c *cli) serve(ctx context.Context, a agent, maxNew, port int) int {
	stopSignals := func() {}
	if c.signals != nil {
		ctx, stopSignals = c.signals(ctx)
	}

	defer stopSignals()

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
		s.stop()

		return c.fail(err)
	case <-ctx.Done():
	}

	// A second signal ends the process at once.
	stopSignals()

	// Refuse first, so that no queued request reaches the agent, then
	// close the listener and wait for the open connections within
	// shutdownTimeout. The running request finishes on its own (a turn is
	// bounded by --max); stop waits for it to leave the agent, and then
	// the model may close.
	s.refuse()

	sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), shutdownTimeout)
	defer cancel()

	_ = srv.Shutdown(sctx)

	s.stop()

	return 0
}
