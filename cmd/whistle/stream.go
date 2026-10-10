package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
)

// PCM formats whistle stream reads from stdin.
const (
	formatS16 = "s16le" // signed 16-bit little-endian integers, scaled by 1/32768 as the WAV reader does
	formatF32 = "f32le" // IEEE float32 little-endian, in [-1, 1]
)

// streamConfig is how runStream feeds and prints a stream.
type streamConfig struct {
	chunk       int  // samples per Process call
	realtime    bool // feed no faster than the audio plays
	asJSON      bool // the engine's stream JSON lines instead of the view
	commitsOnly bool // print only the calls that commit words, and the stop
}

func newStreamCmd() *cobra.Command {
	var (
		model, language, keywords, format string
		chunk                             float64
		cfg                               streamConfig
	)

	cmd := &cobra.Command{
		Use:   "stream [wav]",
		Short: "Transcribe audio of any length live, from a WAV file or raw PCM on stdin",
		Long: "Transcribe audio live, as the engine's needle_stream_transcribe_process and _stop do: every --chunk\n" +
			"seconds of audio are fed to one Process call, which re-transcribes the buffer and commits the words two\n" +
			"consecutive passes agree on; at the end of the input (or on Ctrl-C) Stop commits the rest. There is no\n" +
			"length limit; the stream trims committed audio once its buffer holds more than 15 s.\n\n" +
			"Input: a RIFF WAV file (integer PCM, any channels and rate, played in chunks), or without an argument or\n" +
			"with \"-\" raw 16 kHz mono PCM on stdin in --format s16le (signed 16-bit little-endian, the default) or\n" +
			"f32le (float32 little-endian in [-1, 1]), for example from a microphone:\n" +
			"  arecord -q -f S16_LE -r 16000 -c 1 -t raw | whistle stream\n" +
			"  ffmpeg -loglevel error -i talk.mp3 -f s16le -ac 1 -ar 16000 - | whistle stream\n\n" +
			"Output: a line per call with the received seconds, the words it committed and the pending ones, the\n" +
			"stop's line, then the whole committed text. With --json, each call's result as the engine's stream JSON:\n" +
			`  {"text":"...","words":[...],"pending":"...","language":"en","received":1.00,"pass_ms":0.0}` + "\n" +
			"--commits-only prints only the calls that commit words and the stop, as the engine's --audio-stream does.",
		Args: usageArgs(cobra.MaximumNArgs(1)),
		RunE: func(cmd *cobra.Command, args []string) error {
			n, err := chunkSamples(chunk)
			if err != nil {
				return err
			}

			cfg.chunk = n

			path := "-"
			if len(args) == 1 {
				path = args[0]
			}

			if path != "-" && cmd.Flags().Changed("format") {
				return usageError{errors.New("--format is for raw PCM on stdin; a WAV file names its own")}
			}

			src, err := openSource(path, format, n, cmd.InOrStdin())
			if err != nil {
				return err
			}

			opts := whistle.StreamOptions{Language: language}

			opts.Keywords, err = readKeywords(keywords)
			if err != nil {
				return err
			}

			m, err := whistle.Load(model)
			if err != nil {
				return err
			}

			defer func() { _ = m.Close() }()

			s, err := m.NewStream(opts)
			if err != nil {
				return err
			}

			return runStream(cmd.Context(), s, src, cfg, cmd.OutOrStdout())
		},
	}

	cmd.Flags().StringVar(&model, "model", defaultModelPath, "Model file: whistle.cact or whistle.safetensors")
	cmd.Flags().StringVar(&language, "language", "", "Force the language of every pass; default: detect it per pass")
	cmd.Flags().StringVar(&keywords, "keywords", "", "File of words and phrases to favour, one per line")
	cmd.Flags().Float64Var(&chunk, "chunk", 1, "Seconds of audio per call, in (0, 30]")
	cmd.Flags().BoolVar(&cfg.realtime, "realtime", false, "Feed the audio no faster than it plays")
	cmd.Flags().BoolVar(&cfg.asJSON, "json", false, "Print each call's result as the engine's stream JSON")
	cmd.Flags().BoolVar(&cfg.commitsOnly, "commits-only", false, "Print only the calls that commit words, and the stop")
	cmd.Flags().StringVar(&format, "format", formatS16, "Raw PCM on stdin: s16le or f32le, 16 kHz mono")

	return cmd
}

// chunkSamples is --chunk in samples. It takes the engine's range,
// (0, 30] seconds (--audio-chunk, docs/FINDINGS.md F-032).
func chunkSamples(seconds float64) (int, error) {
	err := usageError{fmt.Errorf("--chunk %v: takes seconds between 0 and 30", seconds)}

	if !(seconds > 0 && seconds <= float64(whistle.MaxSamples)/whistle.SampleRate) { // NaN fails too
		return 0, err
	}

	n := int(math.Round(seconds * whistle.SampleRate))
	if n < 1 {
		return 0, err
	}

	return n, nil
}

// chunkSource yields the audio in chunks; io.EOF ends it.
type chunkSource interface {
	next() ([]float32, error)
}

// openSource opens a WAV file, or stdin for "-".
func openSource(path, format string, chunk int, stdin io.Reader) (chunkSource, error) {
	if path == "-" {
		switch format {
		case formatS16:
			return &pcmSource{r: stdin, width: 2, chunk: chunk}, nil
		case formatF32:
			return &pcmSource{r: stdin, width: 4, chunk: chunk}, nil
		default:
			return nil, usageError{fmt.Errorf("--format %q: want %s or %s", format, formatS16, formatF32)}
		}
	}

	pcm, err := frontend.ReadWAV(path)
	if err != nil {
		return nil, fileError(path, err)
	}

	if len(pcm) == 0 {
		return nil, fmt.Errorf("%s %w", path, whistle.ErrNoAudio)
	}

	return &sliceSource{pcm: pcm, chunk: chunk}, nil
}

// sliceSource plays decoded samples in chunks.
type sliceSource struct {
	pcm   []float32
	chunk int
}

func (s *sliceSource) next() ([]float32, error) {
	if len(s.pcm) == 0 {
		return nil, io.EOF
	}

	n := min(s.chunk, len(s.pcm))
	c := s.pcm[:n]
	s.pcm = s.pcm[n:]

	return c, nil
}

// pcmSource reads raw little-endian PCM, width bytes per sample.
type pcmSource struct {
	r     io.Reader
	width int
	chunk int
	buf   []byte
}

func (p *pcmSource) next() ([]float32, error) {
	if p.buf == nil {
		p.buf = make([]byte, p.chunk*p.width)
	}

	n, err := io.ReadFull(p.r, p.buf)

	switch {
	case errors.Is(err, io.EOF):
		return nil, io.EOF
	case errors.Is(err, io.ErrUnexpectedEOF):
		if n%p.width != 0 {
			return nil, fmt.Errorf("stdin ends inside a sample: %d bytes after the last whole one", n%p.width)
		}
	case err != nil:
		return nil, fmt.Errorf("read stdin: %w", err)
	}

	pcm := make([]float32, n/p.width)

	for i := range pcm {
		b := p.buf[i*p.width:]
		if p.width == 2 {
			pcm[i] = float32(int16(binary.LittleEndian.Uint16(b))) / 32768 // #nosec G115 -- reinterprets the bits.
		} else {
			pcm[i] = math.Float32frombits(binary.LittleEndian.Uint32(b))
		}
	}

	return pcm, nil
}

// read is one chunk or the error that ends the input.
type read struct {
	pcm []float32
	err error
}

// streamer is the part of *whistle.Stream that runStream drives.
type streamer interface {
	Process(pcm []float32) (whistle.StreamResult, error)
	Stop() whistle.StreamResult
}

// runStream feeds src to s chunk by chunk and prints every call, then
// stops the stream at the end of the input or when ctx ends.
func runStream(ctx context.Context, s streamer, src chunkSource, cfg streamConfig, out io.Writer) error {
	// The reader runs apart so that Ctrl-C ends a stream whose stdin is blocked.
	chunks := make(chan read)

	go func() {
		defer close(chunks)

		for {
			pcm, err := src.next()

			select {
			case chunks <- read{pcm, err}:
			case <-ctx.Done():
				return
			}

			if err != nil {
				return
			}
		}
	}()

	p := streamPrinter{cfg: cfg, out: out}
	start, fed := time.Now(), 0

	for {
		if ctx.Err() != nil {
			return p.stop(s.Stop())
		}

		var c read

		select {
		case <-ctx.Done():
			return p.stop(s.Stop())
		case c = <-chunks:
		}

		if errors.Is(c.err, io.EOF) {
			return p.stop(s.Stop())
		}

		if c.err != nil {
			return c.err
		}

		fed += len(c.pcm)

		if cfg.realtime && !waitUntil(ctx, start.Add(time.Duration(fed)*time.Second/whistle.SampleRate)) {
			return p.stop(s.Stop())
		}

		res, err := s.Process(c.pcm)
		if err != nil {
			return err
		}

		err = p.call(res)
		if err != nil {
			return err
		}
	}
}

// waitUntil sleeps until t; false when ctx ends first.
func waitUntil(ctx context.Context, t time.Time) bool {
	d := time.Until(t)
	if d <= 0 {
		return true
	}

	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// streamPrinter writes the calls of one stream.
type streamPrinter struct {
	cfg       streamConfig
	out       io.Writer
	committed []string
}

// call prints a Process result.
func (p *streamPrinter) call(r whistle.StreamResult) error {
	if r.Text != "" {
		p.committed = append(p.committed, r.Text)
	}

	if p.cfg.commitsOnly && len(r.Words) == 0 {
		return nil
	}

	if p.cfg.asJSON {
		return p.json(r)
	}

	return p.printf("%7.2f s  commit %q  pending %q\n", r.Received, r.Text, r.Pending)
}

// stop prints the Stop result and, in the view, the whole committed text.
func (p *streamPrinter) stop(r whistle.StreamResult) error {
	if r.Text != "" {
		p.committed = append(p.committed, r.Text)
	}

	if p.cfg.asJSON {
		return p.json(r)
	}

	return p.printf("%7.2f s  stop   %q\n%s\n", r.Received, r.Text, strings.Join(p.committed, " "))
}

func (p *streamPrinter) json(r whistle.StreamResult) error {
	line, err := r.MarshalJSON()
	if err != nil {
		return err
	}

	return p.printf("%s\n", line)
}

func (p *streamPrinter) printf(format string, args ...any) error {
	_, err := fmt.Fprintf(p.out, format, args...)
	if err != nil {
		return fmt.Errorf("write result: %w", err)
	}

	return nil
}
