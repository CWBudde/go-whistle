package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"path/filepath"
	"time"

	"github.com/cwbudde/go-whistle/internal/server"
	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
)

const (
	// readHeaderTimeout bounds the request line and headers of a request.
	readHeaderTimeout = 10 * time.Second
	// maxBodyMiB is the largest --max-body-mib, 1 TiB in bytes.
	maxBodyMiB = 1 << 20
)

func newServeCmd() *cobra.Command {
	var (
		model, addr string
		bodyMiB     int64
		shutdown    time.Duration
		cfg         server.Config
	)

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Serve transcription over HTTP: POST /transcribe and a WebSocket at /stream",
		Long: "Load the model once and serve it over HTTP:\n\n" +
			"  POST /transcribe  the body is a WAV file (or with ?format=s16le|f32le raw 16 kHz mono PCM), the\n" +
			"                    options are query parameters: language, keywords (repeatable), depth,\n" +
			"                    word_timestamps. The answer is the JSON `whistle transcribe --json` prints.\n" +
			"  GET  /stream      a WebSocket (?format=s16le|f32le, language, keywords): each binary message is a\n" +
			"                    PCM chunk for one Process call, answered by its stream JSON as `whistle stream\n" +
			"                    --json` prints it; the text message {\"type\":\"stop\"} runs Stop, sends its\n" +
			"                    result and closes.\n" +
			"  GET  /healthz     ok\n" +
			"  GET  /v1/info     the model's file, SHA-256, format, weights mode, languages and the limits\n\n" +
			"Passes run --concurrency at a time; /transcribe requests beyond them wait in a queue of --queue and get\n" +
			"503 when it is full. A body over --max-body-mib gets 413, one slower than --read-timeout 408, audio\n" +
			"over 30 s 422. Every request logs a line on stderr: method, path, status, duration. SIGINT or\n" +
			"SIGTERM shuts down: open streams get their Stop result and close with 1001, running requests finish,\n" +
			"all within --shutdown-timeout.",
		Args: usageArgs(cobra.NoArgs),
		RunE: func(cmd *cobra.Command, _ []string) error {
			if bodyMiB > maxBodyMiB {
				return usageError{fmt.Errorf("--max-body-mib %d: want at most %d", bodyMiB, maxBodyMiB)}
			}

			// An expired shutdown clock can only fail the shutdown, so it is
			// refused before the model loads (PR #35 review).
			if shutdown <= 0 {
				return usageError{fmt.Errorf("--shutdown-timeout %v: want a positive duration", shutdown)}
			}

			cfg.MaxBody = bodyMiB << 20

			err := cfg.Check()
			if err != nil {
				return usageError{fmt.Errorf("serve: %w", err)}
			}

			m, err := whistle.Load(model)
			if err != nil {
				return err
			}

			// A failed shutdown can leave a pass running in the model, which
			// must not be closed under it; the exit releases it then.
			running := false

			defer func() {
				if !running {
					_ = m.Close()
				}
			}()

			cfg.Info, err = modelInfo(model, m)
			if err != nil {
				return err
			}

			cfg.Log = log.New(cmd.ErrOrStderr(), "", log.LstdFlags)

			h, err := server.New(server.FromModel(m), cfg)
			if err != nil {
				return err
			}

			var lc net.ListenConfig

			ln, err := lc.Listen(cmd.Context(), "tcp", addr)
			if err != nil {
				return err
			}

			err = serve(cmd.Context(), ln, h, shutdown, cmd.ErrOrStderr())
			running = err != nil

			return err
		},
	}

	cmd.Flags().StringVar(&addr, "addr", "127.0.0.1:8080", "Address to listen on")
	cmd.Flags().StringVar(&model, "model", defaultModelPath, "Model file: whistle.cact or whistle.safetensors")
	cmd.Flags().Int64Var(&bodyMiB, "max-body-mib", 16, "Largest /transcribe body, in MiB")
	cmd.Flags().IntVar(&cfg.Concurrency, "concurrency", 2, "Passes (requests and stream chunks) that run at once")
	cmd.Flags().IntVar(&cfg.Queue, "queue", 8, "/transcribe requests that wait for a pass before 503")
	cmd.Flags().IntVar(&cfg.MaxStreams, "max-streams", 8, "Open /stream sessions")
	cmd.Flags().DurationVar(&cfg.StreamIdle, "stream-idle", time.Minute, "End a stream after this long without a message")
	cmd.Flags().DurationVar(&cfg.ReadTimeout, "read-timeout", time.Minute, "Longest a /transcribe body may take to arrive (408)")
	cmd.Flags().StringSliceVar(&cfg.OriginPatterns, "origin", nil,
		"Origins besides the server's own that may open /stream, as host patterns (example.com, *.example.com)")
	cmd.Flags().DurationVar(&shutdown, "shutdown-timeout", 30*time.Second, "Time to finish requests and streams on shutdown")

	return cmd
}

// modelInfo is what /v1/info reports on the model at path: doctor's format,
// weights mode and SHA-256.
func modelInfo(path string, m *whistle.Model) (server.Info, error) {
	format, err := modelFormat(path)
	if err != nil {
		return server.Info{}, err
	}

	sum, err := fileSHA256(path)
	if err != nil {
		return server.Info{}, err
	}

	return server.Info{
		Model: filepath.Base(path), SHA256: sum, Format: format, Weights: weightsMode(format), Languages: m.Languages(),
	}, nil
}

// serve serves h on ln until ctx ends, then shuts down within timeout:
// running requests finish and the streams end with their Stop result.
func serve(ctx context.Context, ln net.Listener, h *server.Server, timeout time.Duration, logw io.Writer) error {
	hs := &http.Server{
		Handler:           h,
		ReadHeaderTimeout: readHeaderTimeout,
		ErrorLog:          log.New(logw, "whistle: http: ", 0),
	}

	_, _ = fmt.Fprintf(logw, "whistle: listening on http://%s\n", ln.Addr())

	served := make(chan error, 1)

	go func() { served <- hs.Serve(ln) }()

	select {
	case err := <-served:
		return err
	case <-ctx.Done():
	}

	_, _ = fmt.Fprintf(logw, "whistle: shutting down (at most %v)\n", timeout)

	// The cancelled ctx would end the shutdown at once; it gets its own clock.
	sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), timeout)
	defer cancel()

	streams := make(chan error, 1)

	go func() { streams <- h.Shutdown(sctx) }()

	err := hs.Shutdown(sctx)
	if err != nil {
		_ = hs.Close()
	}

	err = errors.Join(err, <-streams)

	<-served // http.ErrServerClosed

	if err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}

	return nil
}
