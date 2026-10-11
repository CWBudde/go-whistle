package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/cwbudde/go-whistle/internal/server"
	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
)

func TestRootHasServe(t *testing.T) {
	cmd, _, err := NewRootCmd().Find([]string{"serve"})
	if err != nil || cmd.Name() != "serve" {
		t.Fatalf("root has no serve subcommand: %v", err)
	}

	for flag, def := range map[string]string{
		"addr": "127.0.0.1:8080", "model": defaultModelPath, "max-body-mib": "16", "concurrency": "2", "queue": "8",
		"max-streams": "8", "stream-idle": "1m0s", "read-timeout": "1m0s", "origin": "[]", "shutdown-timeout": "30s",
	} {
		f := cmd.Flags().Lookup(flag)
		if f == nil {
			t.Errorf("no --%s flag", flag)

			continue
		}

		if f.DefValue != def {
			t.Errorf("--%s default = %q, want %q", flag, f.DefValue, def)
		}
	}
}

func TestServeCmdUsage(t *testing.T) {
	for _, c := range []struct {
		args []string
		want string
	}{
		{[]string{"serve", "extra"}, "unknown command"},
		{[]string{"serve", "--concurrency", "0"}, "concurrency 0"},
		{[]string{"serve", "--queue", "-1"}, "queue -1"},
		{[]string{"serve", "--max-streams", "0"}, "max streams 0"},
		{[]string{"serve", "--stream-idle", "0s"}, "stream idle 0s"},
		{[]string{"serve", "--read-timeout", "-1s"}, "read timeout -1s"},
		{[]string{"serve", "--max-body-mib", "0"}, "max body 0 bytes"},
		{[]string{"serve", "--max-body-mib", "2000000"}, "--max-body-mib 2000000"},
		{[]string{"serve", "--shutdown-timeout", "0s"}, "--shutdown-timeout 0s"},
		{[]string{"serve", "--shutdown-timeout", "-1s"}, "--shutdown-timeout -1s"},
		{[]string{"serve", "--addr"}, "flag needs an argument"},
	} {
		_, err := runCmd(t, c.args...)
		if exitCode(err) != exitUsage || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%v: err = %v, want a usage error with %q", c.args, err, c.want)
		}
	}

	_, err := runCmd(t, "serve", "--model", filepath.Join(t.TempDir(), "missing.cact"))
	if exitCode(err) != exitFailure || !strings.Contains(err.Error(), "missing.cact") {
		t.Errorf("missing model: err = %v, want a failure naming the file", err)
	}
}

// silentModel hears silence.
type silentModel struct{}

func (silentModel) Transcribe([]float32, whistle.TranscribeOptions) (whistle.Result, error) {
	return whistle.Result{}, nil
}

func (silentModel) NewStream(whistle.StreamOptions) (server.Stream, error) {
	return nil, errors.New("no streams")
}

// syncBuffer collects the log of a running server.
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

// serve logs where it listens and every request, and returns nil once ctx
// ends and it has shut down.
func TestServeLifecycle(t *testing.T) {
	h, err := server.New(silentModel{}, server.Config{
		MaxBody: 1 << 10, Concurrency: 1, MaxStreams: 1, StreamIdle: time.Second, ReadTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	var logw syncBuffer

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)

	go func() { done <- serve(ctx, ln, h, time.Second, &logw) }()

	resp, err := http.Get("http://" + ln.Addr().String() + "/healthz")
	if err != nil {
		t.Fatal(err)
	}

	_ = resp.Body.Close()

	cancel()

	err = <-done
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Errorf("serve: %v, healthz %d", err, resp.StatusCode)
	}

	want := "whistle: listening on http://" + ln.Addr().String() + "\n"
	if log := logw.String(); !strings.HasPrefix(log, want) || !strings.Contains(log, "whistle: shutting down") {
		t.Errorf("log =\n%s", log)
	}
}

// startServe runs `whistle serve` with args on a free port and returns its
// URL; the test's cleanup shuts it down and checks it exits with nil.
func startServe(t *testing.T, args ...string) string {
	t.Helper()

	var logw syncBuffer

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)

	root := serveRoot(&logw, args)

	go func() { done <- root.ExecuteContext(ctx) }()

	t.Cleanup(func() {
		cancel()

		err := <-done
		if err != nil {
			t.Errorf("serve: %v\n%s", err, logw.String())
		}
	})

	listening := regexp.MustCompile(`whistle: listening on (http://\S+)`)

	for deadline := time.Now().Add(30 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if m := listening.FindStringSubmatch(logw.String()); m != nil {
			return m[1]
		}

		select {
		case err := <-done:
			t.Fatalf("serve ended: %v\n%s", err, logw.String())
		default:
		}
	}

	t.Fatalf("serve did not listen:\n%s", logw.String())

	return ""
}

// serveRoot is the command line `whistle serve --addr 127.0.0.1:0` with
// args, logging to logw.
func serveRoot(logw io.Writer, args []string) *cobra.Command {
	root := NewRootCmd()
	root.SetOut(io.Discard)
	root.SetErr(logw)
	root.SetArgs(append([]string{"serve", "--addr", "127.0.0.1:0"}, args...))

	return root
}

// timing is the part of the engine's JSON that differs between runs.
var timing = regexp.MustCompile(`"ttft_ms":\d+\.\d,"decode_tps":\d+\.\d}\n$`)

// untimed is a /transcribe answer or `transcribe --json` line without its
// timing, which must be there.
func untimed(t *testing.T, line string) string {
	t.Helper()

	if !timing.MatchString(line) {
		t.Errorf("%q does not end with the engine's timing", line)
	}

	return timing.ReplaceAllString(line, `"ttft_ms":_,"decode_tps":_}`)
}

// POST /transcribe answers with the line `whistle transcribe --json` prints
// for the same file and options, but for ttft_ms and decode_tps.
func TestServeMatchesTranscribeCmd(t *testing.T) {
	requireModel(t)

	base := startServe(t, "--model", cactPath)

	kw := filepath.Join(t.TempDir(), "keywords.txt")

	err := os.WriteFile(kw, []byte("cowley's\nNew  York\n"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	cowley := "../../testdata/audio/ls-other-3080-5040-0011.wav"

	for _, c := range []struct {
		path  string
		args  []string
		query url.Values
	}{
		{clipPath, nil, nil},
		{clipPath, []string{"--language", "en", "--word-timestamps"}, url.Values{"language": {"en"}, "word_timestamps": {"true"}}},
		{cowley, []string{"--keywords", kw}, url.Values{"keywords": {"cowley's", "New  York"}}},
		{cowley, []string{"--depth", "4", "--word-timestamps"}, url.Values{"depth": {"4"}, "word_timestamps": {""}}},
		{
			"../../testdata/audio/fleurs-de_de-10229344228128634115.wav",
			[]string{"--word-timestamps"},
			url.Values{"word_timestamps": {"1"}},
		},
	} {
		want, err := runCmd(t, append(append([]string{"transcribe", "--model", cactPath, "--json"}, c.args...), c.path)...)
		if err != nil {
			t.Fatal(err)
		}

		body, err := os.ReadFile(c.path)
		if err != nil {
			t.Fatal(err)
		}

		status, got := postAudio(t, base+"/transcribe?"+c.query.Encode(), body)
		if status != http.StatusOK || untimed(t, got) != untimed(t, want) {
			t.Errorf("%s %v: %d %q\nwant %q", c.path, c.args, status, got, want)
		}
	}

	// The clip is 16-bit mono 16 kHz PCM, so its data is the same audio as
	// s16le.
	wav, err := os.ReadFile(clipPath)
	if err != nil {
		t.Fatal(err)
	}

	want, err := runCmd(t, "transcribe", "--model", cactPath, "--json", clipPath)
	if err != nil {
		t.Fatal(err)
	}

	status, got := postAudio(t, base+"/transcribe?format=s16le", wav[44:])
	if status != http.StatusOK || untimed(t, got) != untimed(t, want) {
		t.Errorf("s16le: %d %q, want %q", status, got, want)
	}

	long := filepath.Join(t.TempDir(), "long.wav")
	writeSilentWAV(t, long, whistle.MaxSamples+1)

	body, err := os.ReadFile(long)
	if err != nil {
		t.Fatal(err)
	}

	status, got = postAudio(t, base+"/transcribe", body)
	if status != http.StatusUnprocessableEntity || !strings.Contains(got, "audio limit is 30 s") {
		t.Errorf("31 s: %d %s, want 422 naming the limit", status, got)
	}
}

// Concurrent requests on one model, as --concurrency lets them run, give
// what one at a time gives; under -race this checks that Transcribe is safe
// for concurrent use.
func TestServeConcurrentTranscribe(t *testing.T) {
	requireModel(t)

	base := startServe(t, "--model", cactPath, "--concurrency", "4")
	clips := []string{clipPath, "../../testdata/audio/ls-other-3080-5040-0011.wav"}
	bodies := make([][]byte, len(clips))
	want := make([]string, len(clips))

	for i, path := range clips {
		var err error

		bodies[i], err = os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}

		status, got := postAudio(t, base+"/transcribe?word_timestamps", bodies[i])
		if status != http.StatusOK {
			t.Fatalf("%s: %d %s", path, status, got)
		}

		want[i] = untimed(t, got)
	}

	var wg sync.WaitGroup

	for k := range 8 {
		wg.Go(func() {
			i := k % len(clips)

			status, got := postAudio(t, base+"/transcribe?word_timestamps", bodies[i])
			if status != http.StatusOK || untimed(t, got) != want[i] {
				t.Errorf("request %d: %d %q, want %q", k, status, got, want[i])
			}
		})
	}

	wg.Wait()
}

func postAudio(t *testing.T, url string, body []byte) (int, string) {
	t.Helper()

	resp, err := http.Post(url, "audio/wav", bytes.NewReader(body))
	if err != nil {
		t.Error(err)

		return 0, ""
	}

	defer func() { _ = resp.Body.Close() }()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Error(err)
	}

	return resp.StatusCode, string(b)
}

// A /stream session fed a manifest run in 1 s chunks sends the lines
// `whistle stream --json` prints for the run's WAV, without pass_ms: one per
// chunk, then the Stop result after {"type":"stop"}, then a 1000 close.
func TestServeStreamMatchesStreamCmd(t *testing.T) {
	requireModel(t)

	base := startServe(t, "--model", cactPath)

	for _, name := range []string{"ls-other-3080-5040-0018.c1", "ls-clean-6930-75918-0000.c1"} {
		t.Run(name, func(t *testing.T) {
			r, x := loadStreamRun(t, name)

			wav := filepath.Join(t.TempDir(), name+".wav")
			writeWAV(t, wav, x)

			args := []string{"stream", "--model", cactPath, "--json"}
			query := url.Values{}

			if r.Language != "" {
				args = append(args, "--language", r.Language)
				query.Set("language", r.Language)
			}

			out, err := runCmd(t, append(args, wav)...)
			if err != nil {
				t.Fatal(err)
			}

			want, wantText := cliLines(t, out)
			if len(want) < 4 || wantText == "" {
				t.Fatalf("stream --json printed %d lines, text %q", len(want), wantText)
			}

			var f32 bytes.Buffer

			_ = binary.Write(&f32, binary.LittleEndian, x)

			for format, pcm := range map[string][]byte{"s16le": s16le(x), "f32le": f32.Bytes()} {
				query.Set("format", format)

				got, text := streamSession(t, "ws"+strings.TrimPrefix(base, "http")+"/stream?"+query.Encode(), pcm,
					len(pcm)/len(x)*whistle.SampleRate)
				if text != wantText || strings.Join(got, "\n") != strings.Join(want, "\n") {
					t.Errorf("%s: text %q, want %q; %d lines, want %d:\n%s\nwant\n%s", format, text, wantText,
						len(got), len(want), strings.Join(got, "\n"), strings.Join(want, "\n"))
				}
			}
		})
	}
}

// streamSession sends pcm to the /stream at url in chunks of chunk bytes,
// then the stop, and returns the stream JSON lines it got and their text,
// as cliLines does.
func streamSession(t *testing.T, url string, pcm []byte, chunk int) ([]string, string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	c, _, err := websocket.Dial(ctx, url, nil) //nolint:bodyclose // Dial closes it
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = c.CloseNow() }()

	var out strings.Builder

	read := func() {
		typ, msg, err := c.Read(ctx)
		if err != nil || typ != websocket.MessageText {
			t.Fatalf("read: %v %v", err, typ)
		}

		out.Write(msg)
		out.WriteByte('\n')
	}

	for i := 0; i < len(pcm); i += chunk {
		err = c.Write(ctx, websocket.MessageBinary, pcm[i:min(i+chunk, len(pcm))])
		if err != nil {
			t.Fatal(err)
		}

		read()
	}

	err = c.Write(ctx, websocket.MessageText, []byte(`{"type":"stop"}`))
	if err != nil {
		t.Fatal(err)
	}

	read()

	_, _, err = c.Read(ctx)
	if websocket.CloseStatus(err) != websocket.StatusNormalClosure {
		t.Errorf("after the stop: %v, want a 1000 close", err)
	}

	return cliLines(t, out.String())
}
