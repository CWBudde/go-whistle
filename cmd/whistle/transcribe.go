package main

import (
	"bytes"
	"fmt"
	"io"
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
		Use:   "transcribe <wav>",
		Short: "Transcribe a WAV file of at most 30 s",
		Long: "Transcribe a RIFF WAV file (8/16/24/32-bit PCM or float, any channels and rate) of at most 30 s and\n" +
			"print the text, or with --json the engine's JSON object:\n" +
			`  {"text":"...","language":"en","ttft_ms":0.0,"decode_tps":0.0}` + "\n\n" +
			"Decoding is the engine's beam search (5 beams); silence gives the empty text. --keywords names a file of\n" +
			"words and phrases to favour, one per line, and --depth runs fewer decoder layers (2..8), as the engine's\n" +
			"--audio-keywords and --audio-depth do. --word-timestamps adds every word's start, end and probability\n" +
			"(the engine's --audio-word-timestamps): a line per word after the text, or the JSON's \"words\".",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts := whistle.TranscribeOptions{Language: language, Depth: depth, WordTimestamps: words}

			if keywords != "" {
				raw, err := os.ReadFile(keywords) // #nosec G304 -- the user names the file.
				if err != nil {
					return fmt.Errorf("keywords: %w", err)
				}

				opts.Keywords = strings.Split(string(raw), "\n")
			}

			m, err := whistle.Load(model)
			if err != nil {
				return err
			}

			defer func() { _ = m.Close() }()

			return runTranscribe(m, args[0], opts, asJSON, cmd.OutOrStdout())
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

// runTranscribe transcribes the WAV file at path with m and writes the text
// or the JSON line to out. With word timestamps the text is followed by a
// line per word, as upstream's playground prints them.
func runTranscribe(m *whistle.Model, path string, opts whistle.TranscribeOptions, asJSON bool, out io.Writer) error {
	res, err := m.TranscribeWAV(path, opts)
	if err != nil {
		return err
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
		b.WriteString(res.Text + "\n")

		for _, w := range res.Words {
			fmt.Fprintf(&b, "  %6.2f - %5.2f  %-20s %.2f\n", w.Start.Seconds(), w.End.Seconds(), w.Text, w.Probability)
		}
	}

	_, err = out.Write(b.Bytes())
	if err != nil {
		return fmt.Errorf("write result: %w", err)
	}

	return nil
}
