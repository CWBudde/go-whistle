package main

import (
	"fmt"
	"io"
	"path/filepath"

	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
)

// defaultModelPath is the .cact `whistle download` puts into defaultModelDir.
var defaultModelPath = filepath.Join(defaultModelDir, "whistle.cact")

func newTranscribeCmd() *cobra.Command {
	var (
		model    string
		language string
		asJSON   bool
	)

	cmd := &cobra.Command{
		Use:   "transcribe <wav>",
		Short: "Transcribe a WAV file of at most 30 s",
		Long: "Transcribe a RIFF WAV file (8/16/24/32-bit PCM or float, any channels and rate) of at most 30 s and\n" +
			"print the text, or with --json the engine's JSON object:\n" +
			`  {"text":"...","language":"en","ttft_ms":0.0,"decode_tps":0.0}` + "\n\n" +
			"Decoding is the engine's beam search (5 beams); silence gives the empty text.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := whistle.Load(model)
			if err != nil {
				return err
			}

			defer func() { _ = m.Close() }()

			return runTranscribe(m, args[0], whistle.TranscribeOptions{Language: language}, asJSON, cmd.OutOrStdout())
		},
	}

	cmd.Flags().StringVar(&model, "model", defaultModelPath, "Model file: whistle.cact or whistle.safetensors")
	cmd.Flags().StringVar(&language, "language", "", "Force the language (en, de, fr, es, it, nl, pl); default: detect it")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Print the result as the engine's JSON object")

	return cmd
}

// runTranscribe transcribes the WAV file at path with m and writes the text
// or the JSON line to out.
func runTranscribe(m *whistle.Model, path string, opts whistle.TranscribeOptions, asJSON bool, out io.Writer) error {
	res, err := m.TranscribeWAV(path, opts)
	if err != nil {
		return err
	}

	line := []byte(res.Text)

	if asJSON {
		line, err = res.MarshalJSON()
		if err != nil {
			return err
		}
	}

	_, err = fmt.Fprintf(out, "%s\n", line)
	if err != nil {
		return fmt.Errorf("write result: %w", err)
	}

	return nil
}
