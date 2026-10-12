//go:build oracle

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// telemetryEnv opts every engine run out of usage reporting.
var telemetryEnv = map[string]string{"NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1"}

// cliResult is one needle CLI run: its exit code and stderr (with the program name the engine prefixes reduced to
// "needle"), and its stdout.
type cliResult struct {
	Exit   int    `json:"exit"`
	Stderr string `json:"stderr,omitempty"`
	stdout []byte
}

// command builds a needle CLI command with the telemetry opt-outs. The engine prefixes its errors with argv[0], so
// argv[0] is "needle" whatever the path, and the recorded errors do not name the checkout.
func command(ctx context.Context, needle string, args []string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, needle, args...) // #nosec G204 -- the engine dir and the flags are the tool's own.
	cmd.Args[0] = "needle"

	cmd.Env = os.Environ()
	for k, v := range telemetryEnv {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	return cmd
}

// runCLI runs the needle CLI with args and returns its outcome; err is set only when it could not run.
func runCLI(ctx context.Context, needle string, args []string) (cliResult, error) {
	cmd := command(ctx, needle, args)

	var stdout, stderr bytes.Buffer

	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	err := cmd.Run()

	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return cliResult{Exit: exit.ExitCode(), Stderr: strings.TrimSpace(stderr.String())}, nil
	}

	if err != nil {
		return cliResult{}, fmt.Errorf("needle: %w", err)
	}

	return cliResult{Stderr: strings.TrimSpace(stderr.String()), stdout: stdout.Bytes()}, nil
}

// answer runs the needle CLI with args and returns its answer with the timing fields dropped; a non-zero exit is
// an error.
func answer(ctx context.Context, needle string, args []string) (node, error) {
	res, err := runCLI(ctx, needle, args)
	if err != nil {
		return node{}, err
	}

	if res.Exit != 0 {
		return node{}, fmt.Errorf("needle %s: exit %d: %s", strings.Join(args, " "), res.Exit, res.Stderr)
	}

	return normaliseResponse(res.stdout)
}

// serveTimeout bounds the wait for `needle --serve` to listen.
const serveTimeout = 30 * time.Second

// serveTurns answers a conversation with `needle --serve`: args select the model, tools, system and engine flags,
// each turn is one POST /complete on the same session, and the answers come back with the timing fields dropped.
// The single-shot --prompt mode answers one turn only.
func serveTurns(ctx context.Context, needle string, args, turns []string) ([]node, error) {
	s, err := startServe(ctx, needle, args)
	if err != nil {
		return nil, err
	}
	defer s.close()

	answers := make([]node, 0, len(turns))

	for i, t := range turns {
		n, err := s.complete(ctx, t)
		if err != nil {
			return nil, fmt.Errorf("turn %d: %w", i+1, err)
		}

		answers = append(answers, n)
	}

	return answers, nil
}

// serveSession is one running `needle --serve`: a conversation that POST /complete extends and POST /reset clears.
type serveSession struct {
	base   string
	cancel context.CancelFunc
	cmd    *exec.Cmd
}

// startServe starts `needle --serve` with args on a free port and waits until it listens.
func startServe(ctx context.Context, needle string, args []string) (*serveSession, error) {
	port, err := freePort(ctx)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(ctx)

	cmd := command(ctx, needle, append(append([]string{}, args...), "--serve", "--port", strconv.Itoa(port)))

	// The server announces itself on stderr; its errors go there too, so the banner wait keeps what it read.
	errPipe, err := cmd.StderrPipe()
	if err != nil {
		cancel()

		return nil, fmt.Errorf("needle --serve: %w", err)
	}

	err = cmd.Start()
	if err != nil {
		cancel()

		return nil, fmt.Errorf("needle --serve: %w", err)
	}

	s := &serveSession{base: "http://127.0.0.1:" + strconv.Itoa(port), cancel: cancel, cmd: cmd}

	var stderr strings.Builder

	err = awaitServing(errPipe, &stderr)
	if err != nil {
		s.close()

		return nil, fmt.Errorf("needle --serve: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	go func() { _, _ = io.Copy(io.Discard, errPipe) }()

	return s, nil
}

// complete answers one turn of the session with the timing fields dropped.
func (s *serveSession) complete(ctx context.Context, input string) (node, error) {
	js, err := post(ctx, s.base+"/complete", map[string]string{"input": input})
	if err != nil {
		return node{}, err
	}

	return normaliseResponse(js)
}

// reset starts a new conversation on the session.
func (s *serveSession) reset(ctx context.Context) error {
	_, err := post(ctx, s.base+"/reset", map[string]string{})

	return err
}

// close stops the server.
func (s *serveSession) close() {
	s.cancel()

	_ = s.cmd.Wait()
}

// freePort asks the kernel for a free local TCP port.
func freePort(ctx context.Context) (int, error) {
	l, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("free port: %w", err)
	}

	addr, ok := l.Addr().(*net.TCPAddr)
	_ = l.Close()

	if !ok {
		return 0, errors.New("free port: not a TCP address")
	}

	return addr.Port, nil
}

// awaitServing reads the server's stderr until it reports that it listens, copying the lines before into log.
func awaitServing(stderr io.Reader, log *strings.Builder) error {
	ready := make(chan error, 1)
	lines := make(chan string, 64)

	go func() {
		defer close(lines)

		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			if strings.Contains(sc.Text(), "serving on") {
				ready <- nil

				return
			}

			lines <- sc.Text()
		}

		ready <- errors.New("exited before serving")
	}()

	timeout := time.After(serveTimeout)

	for {
		select {
		case l, ok := <-lines:
			if ok {
				log.WriteString(l + "\n")
			} else {
				lines = nil
			}
		case err := <-ready:
			if lines != nil {
				for l := range lines { // the lines read before the banner or the exit
					log.WriteString(l + "\n")
				}
			}

			return err
		case <-timeout:
			return fmt.Errorf("not serving after %s", serveTimeout)
		}
	}
}

// post sends body as JSON and returns the response body; a status other than 200 is an error.
func post(ctx context.Context, url string, body any) ([]byte, error) {
	js, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(js))
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req) // #nosec G704 -- a loopback URL built from a free port.
	if err != nil {
		return nil, fmt.Errorf("post %s: %w", url, err)
	}

	defer func() { _ = resp.Body.Close() }()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", url, err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("post %s: %s: %s", url, resp.Status, b)
	}

	return b, nil
}
