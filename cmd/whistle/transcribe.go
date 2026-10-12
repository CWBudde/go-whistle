package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
)

// defaultModelPath is the .cact `whistle download` puts into defaultModelDir.
var defaultModelPath = filepath.Join(defaultModelDir, "whistle.cact")

func newTranscribeCmd() *cobra.Command {
	var (
		model    string
		language string
		keywords string
		depth    int
		asJSON   bool
		words    bool
	)

	cmd := &cobra.Command{
		Use:   "transcribe <wav>...",
		Short: "Transcribe WAV files of at most 30 s each",
		Long: "Transcribe RIFF WAV files (8/16/24/32-bit integer PCM, any channels and rate) of at most 30 s each and\n" +
			"print the text, or with --json the engine's JSON object:\n" +
			`  {"text":"...","language":"en","ttft_ms":0.0,"decode_tps":0.0}` + "\n\n" +
			"Decoding is the engine's beam search (5 beams); silence gives the empty text. --keywords names a file of\n" +
			"words and phrases to favour, one per line, and --depth runs fewer decoder layers (2..8), as the engine's\n" +
			"--audio-keywords and --audio-depth do. --word-timestamps adds every word's start, end and probability\n" +
			"(the engine's --audio-word-timestamps): a line per word after the text, or the JSON's \"words\".\n\n" +
			"With several files the model loads once. The text of each is prefixed with its path; with --json each\n" +
			"gives one line, in the order given. A file that fails (unreadable, not PCM, over 30 s: use `whistle\n" +
			"stream` for longer audio) prints nothing on stdout and its error on stderr, the others go on, and the\n" +
			"command exits with 1.",
		Args: usageArgs(cobra.MinimumNArgs(1)),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts := whistle.TranscribeOptions{Language: language, Depth: depth, WordTimestamps: words}

			var err error

			opts.Keywords, err = readKeywords(keywords)
			if err != nil {
				return err
			}

			m, err := whistle.Load(model)
			if err != nil {
				return err
			}

			defer func() { _ = m.Close() }()

			return transcribeFiles(m, args, opts, asJSON, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}

	cmd.Flags().StringVar(&model, "model", defaultModelPath, "Model file: whistle.cact or whistle.safetensors")
	cmd.Flags().StringVar(&language, "language", "", "Force the language (en, de, fr, es, it, nl, pl); default: detect it")
	cmd.Flags().StringVar(&keywords, "keywords", "", "File of words and phrases to favour, one per line")
	cmd.Flags().IntVar(&depth, "depth", 0, "Decoder layers to run, 2..8 (the engine's ladder); default: all")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Print the result as the engine's JSON object")
	cmd.Flags().BoolVar(&words, "word-timestamps", false, "Add each word's start, end and probability")

	return cmd
}

// readKeywords reads the --keywords file, one entry per line; "" reads none.
func readKeywords(path string) ([]string, error) {
	if path == "" {
		return nil, nil
	}

	raw, err := os.ReadFile(path) // #nosec G304 -- the user names the file.
	if err != nil {
		return nil, fmt.Errorf("keywords: %w", err)
	}

	return strings.Split(string(raw), "\n"), nil
}

// transcribeFiles transcribes every path with m. One file returns its error;
// with several, each failure is reported on errOut and the rest go on.
func transcribeFiles(m *whistle.Model, paths []string, opts whistle.TranscribeOptions, asJSON bool,
	out, errOut io.Writer,
) error {
	if len(paths) == 1 {
		return runTranscribe(m, paths[0], opts, asJSON, "", out)
	}

	failed := 0

	for _, path := range paths {
		err := runTranscribe(m, path, opts, asJSON, path+": ", out)
		if errors.Is(err, errWrite) {
			return err
		}

		if err != nil {
			failed++

			_, err = fmt.Fprintln(errOut, "whistle:", err)
			if err != nil {
				return fmt.Errorf("write error: %w", err)
			}
		}
	}

	if failed > 0 {
		return fmt.Errorf("%d of %d files failed", failed, len(paths))
	}

	return nil
}

// errWrite marks a failed write of the output, which ends a multi-file run.
var errWrite = errors.New("write result")

// runTranscribe transcribes the WAV file at path with m and writes the text,
// after prefix, or the JSON line to out. With word timestamps the text is
// followed by a line per word, as upstream's playground prints them.
func runTranscribe(m *whistle.Model, path string, opts whistle.TranscribeOptions, asJSON bool, prefix string,
	out io.Writer,
) error {
	res, err := m.TranscribeWAV(path, opts)
	if err != nil {
		return fileError(path, err)
	}

	var b bytes.Buffer

	if asJSON {
		line, err := res.MarshalJSON()
		if err != nil {
			return err
		}

		b.Write(line)
		b.WriteByte('\n')
	} else {
		b.WriteString(prefix + res.Text + "\n")

		for _, w := range res.Words {
			fmt.Fprintf(&b, "  %6.2f - %5.2f  %-20s %.2f\n", w.Start.Seconds(), w.End.Seconds(), w.Text, w.Probability)
		}
	}

	_, err = out.Write(b.Bytes())
	if err != nil {
		return fmt.Errorf("%w: %w", errWrite, err)
	}

	return nil
}

// fileError starts err's message with path, unless it does already, and
// points a clip over the 30 s limit to `whistle stream`.
func fileError(path string, err error) error {
	msg := strings.TrimPrefix(err.Error(), "whistle: ")

	if pe, ok := errors.AsType[*fs.PathError](err); ok {
		msg = pe.Err.Error() // "no such file or directory" without the reader's prefixes
	}

	if !strings.HasPrefix(msg, path) {
		msg = path + ": " + msg
	}

	if errors.Is(err, whistle.ErrTooLong) {
		msg += " (transcribe takes one pass of at most 30 s; whistle stream takes longer audio)"
	}

	return &pathError{msg: msg, err: err}
}

// pathError is an error with the file it concerns in its message.
type pathError struct {
	msg string
	err error
}

func (e *pathError) Error() string { return e.msg }

func (e *pathError) Unwrap() error { return e.err }
