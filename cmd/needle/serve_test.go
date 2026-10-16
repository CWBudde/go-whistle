package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/needle"
)

// post sends body to url and returns the status, the content type and the
// response body.
func post(t *testing.T, url, body string) (int, string, string) {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = resp.Body.Close() }()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	return resp.StatusCode, resp.Header.Get("Content-Type"), string(b)
}

// TestServeProtocol: POST /complete answers {"input": "…"} with the
// envelope as --prompt prints it, a refused turn with the engine's
// refusal and status 200 (turns.json, history.json), POST /reset starts
// over; a body that is not {"input": string} gets 400, an error of the
// model 500, each with {"error": "…"}, other methods and paths 405 and
// 404, and every request after stop 503.
func TestServeProtocol(t *testing.T) {
	a := &fakeAgent{}
	s := newServer(a, 77)
	srv := httptest.NewServer(s.handler())

	defer srv.Close()

	status, ctype, body := post(t, srv.URL+"/complete", `{"input": "set a timer for 10 minutes for the pasta"}`)
	if status != http.StatusOK || ctype != "application/json" || body != timerJSON {
		t.Errorf("complete: %d %s %q", status, ctype, body)
	}

	// JSON escapes are read as JSON reads them, unlike the engine (F-063).
	status, _, _ = post(t, srv.URL+"/complete", `{"input": "a < b ü \"q\"", "max": 3}`)
	if status != http.StatusOK {
		t.Errorf("complete with escapes: %d", status)
	}

	status, ctype, body = post(t, srv.URL+"/reset", `{}`)
	if status != http.StatusOK || ctype != "application/json" || body != "{\"ok\":true}\n" {
		t.Errorf("reset: %d %s %q", status, ctype, body)
	}

	inputs, resets := a.asked()
	if want := []string{"set a timer for 10 minutes for the pasta", `a < b ü "q"`}; !reflect.DeepEqual(inputs, want) ||
		resets != 1 || !reflect.DeepEqual(a.maxNew, []int{77, 77}) {
		t.Errorf("asked %q at %v, %d resets; want %q at 77, 1 reset", inputs, a.maxNew, resets, want)
	}

	for _, body := range []string{``, `not json`, `{}`, `{"input": 5}`, `{"input": null}`, `["x"]`, `{"input": "a"} {}`} {
		status, ctype, got := post(t, srv.URL+"/complete", body)
		if status != http.StatusBadRequest || ctype != "application/json" || !strings.HasPrefix(got, `{"error":"`) {
			t.Errorf("complete %q: %d %s %q, want 400 and an error", body, status, ctype, got)
		}
	}

	if inputs, _ := a.asked(); len(inputs) != 2 {
		t.Errorf("a bad request reached the agent: %q", inputs)
	}

	// Through the handler: over a socket the server may close before the
	// client has sent the whole body, and the client sees a reset.
	rec := httptest.NewRecorder()
	s.handler().ServeHTTP(rec, httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/complete",
		strings.NewReader(`{"input": "`+strings.Repeat("x", maxBody)+`"}`)))

	if got := rec.Body.String(); rec.Code != http.StatusRequestEntityTooLarge || !strings.HasPrefix(got, `{"error":"`) {
		t.Errorf("oversized body: %d %q, want 413", rec.Code, got)
	}

	a.reply = func(string) (*needle.Response, error) {
		return nil, &needle.OverflowError{Tokens: 1407, Available: 878}
	}

	status, _, got := post(t, srv.URL+"/complete", `{"input": "long"}`)
	if want := `{"type":"call","success":false,"error":"input exceeds the context window (1407 tokens, 878 available)",` +
		`"error_code":"context_overflow","reason":"runtime_failure","function_calls":[],"suppressed_calls":[],` +
		`"reasoning":null,"confidence":0.0000}` + "\n"; status != http.StatusOK || got != want {
		t.Errorf("refusal: %d %q, want 200 %q", status, got, want)
	}

	a.reply = func(string) (*needle.Response, error) { return nil, errBoom }

	status, _, got = post(t, srv.URL+"/complete", `{"input": "x"}`)
	if status != http.StatusInternalServerError || got != "{\"error\":\"needle: boom\"}\n" {
		t.Errorf("model error: %d %q, want 500", status, got)
	}

	for _, c := range []struct {
		method, path string
		want         int
	}{
		{http.MethodGet, "/complete", http.StatusMethodNotAllowed},
		{http.MethodGet, "/reset", http.StatusMethodNotAllowed},
		{http.MethodPost, "/nope", http.StatusNotFound},
	} {
		req, err := http.NewRequestWithContext(t.Context(), c.method, srv.URL+c.path, nil)
		if err != nil {
			t.Fatal(err)
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}

		_ = resp.Body.Close()

		if resp.StatusCode != c.want {
			t.Errorf("%s %s: %d, want %d", c.method, c.path, resp.StatusCode, c.want)
		}
	}

	// Once stopped for the model to close, nothing reaches the agent.
	s.stop()

	for _, path := range []string{"/complete", "/reset"} {
		if status, _, got := post(t, srv.URL+path, `{"input": "x"}`); status != http.StatusServiceUnavailable {
			t.Errorf("%s after stop: %d %q, want 503", path, status, got)
		}
	}

	if inputs, resets := a.asked(); len(inputs) != 4 || resets != 1 {
		t.Errorf("after stop the agent was asked %q, reset %d times", inputs, resets)
	}
}

// TestServeSerialises: concurrent requests reach the one agent one at a
// time, as an Agent is not safe for concurrent use.
func TestServeSerialises(t *testing.T) {
	a := &fakeAgent{delay: 2 * time.Millisecond}
	srv := httptest.NewServer(newServer(a, 0).handler())

	defer srv.Close()

	var wg sync.WaitGroup

	for i := range 16 {
		wg.Go(func() {
			path, body := "/complete", `{"input": "x"}`
			if i%4 == 3 {
				path, body = "/reset", `{}`
			}

			if status, _, _ := post(t, srv.URL+path, body); status != http.StatusOK {
				t.Errorf("%s: %d", path, status)
			}
		})
	}

	wg.Wait()

	if inputs, resets := a.asked(); a.overlap.Load() || len(inputs) != 12 || resets != 4 {
		t.Errorf("overlap %v, %d turns, %d resets; want none, 12 and 4", a.overlap.Load(), len(inputs), resets)
	}
}

// banner is the line --serve writes on stderr once it listens.
var banner = regexp.MustCompile(`^needle: serving on (http://127\.0\.0\.1:\d+)`)

// serveCLI runs c with --serve on a free port and the flags args until the
// test ends, and returns the server's address.
func serveCLI(t *testing.T, c *cli, args ...string) string {
	t.Helper()

	pr, pw := io.Pipe()
	c.stderr = pw

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan int, 1)

	go func() {
		done <- c.run(ctx, append(args, "--serve", "--port", "0"))

		_ = pw.Close()
	}()

	var stderr []string

	lines := bufio.NewScanner(pr)
	for lines.Scan() {
		stderr = append(stderr, lines.Text())

		if m := banner.FindStringSubmatch(lines.Text()); m != nil {
			go func() { _, _ = io.Copy(io.Discard, pr) }()

			t.Cleanup(func() {
				cancel()

				if code := <-done; code != 0 {
					t.Errorf("--serve exited %d", code)
				}
			})

			return m[1]
		}
	}

	cancel()
	t.Fatalf("--serve exited %d before serving: %q", <-done, stderr)

	return ""
}

// TestServeRun: --serve announces "serving on" with its address on stderr,
// answers there with the flags' agent, and exits 0 when its context ends,
// with the model closed.
func TestServeRun(t *testing.T) {
	h := newHarness()

	t.Run("serve", func(t *testing.T) {
		url := serveCLI(t, h.cli(), "--max", "33", "--fail-input-overflow")

		status, _, body := post(t, url+"/complete", `{"input": "hi"}`)
		if status != http.StatusOK || body != timerJSON {
			t.Errorf("complete: %d %q", status, body)
		}
	})

	if !h.model.closed || !h.model.opts.FailInputOverflow || !reflect.DeepEqual(h.model.agent.maxNew, []int{33}) {
		t.Errorf("closed %v, options %+v, --max %v", h.model.closed, h.model.opts, h.model.agent.maxNew)
	}
}

// TestServeSignal: a signal of c.signals ends --serve with exit 0 and the
// model closed; serve stops listening for signals once one came.
func TestServeSignal(t *testing.T) {
	h := newHarness()
	c := h.cli()

	signal := make(chan context.CancelFunc, 1)
	stopped := 0

	var mu sync.Mutex

	c.signals = func(ctx context.Context) (context.Context, context.CancelFunc) {
		ctx, cancel := context.WithCancel(ctx)
		signal <- cancel

		return ctx, func() {
			mu.Lock()
			stopped++
			mu.Unlock()
			cancel()
		}
	}

	pr, pw := io.Pipe()
	c.stderr = pw
	done := make(chan int, 1)

	go func() {
		done <- c.run(t.Context(), []string{"--serve", "--port", "0"})

		_ = pw.Close()
	}()

	lines := bufio.NewScanner(pr)
	for lines.Scan() && !banner.MatchString(lines.Text()) {
	}

	go func() { _, _ = io.Copy(io.Discard, pr) }()

	(<-signal)()

	code := <-done

	mu.Lock()
	defer mu.Unlock()

	if code != 0 || !h.model.closed || stopped == 0 {
		t.Errorf("exit %d, model closed %v, signals stopped %d times", code, h.model.closed, stopped)
	}
}
