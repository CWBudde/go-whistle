package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/whistle"
	"github.com/spf13/cobra"
)

// Output formats of whistle embed.
const (
	formatJSON = "json"
	formatNPY  = "npy"
)

func newEmbedCmd() *cobra.Command {
	var model, format, output string

	cmd := &cobra.Command{
		Use:   "embed <wav>",
		Short: "Write the encoder embedding of a WAV file of at most 30 s",
		Long: "Run the front end and the encoder on a RIFF WAV file (integer PCM, any channels and rate) of at most\n" +
			"30 s and write what the engine's needle_embed returns: one row of 512 float32 values per 80 ms frame,\n" +
			"ceil(floor(n / 160) / 8) rows for n samples at 16 kHz.\n\n" +
			"--format json (the default) writes one object,\n" +
			`  {"frames":34,"dim":512,"frame_ms":80,"embedding":[[...512 values...],...]}` + "\n" +
			"with each value the shortest decimal that reads back as the same float32. --format npy writes a NumPy\n" +
			"array file (format 1.0, '<f4', shape (frames, 512), C order), for numpy.load. -o names the output file;\n" +
			"the default is stdout.",
		Args: usageArgs(cobra.ExactArgs(1)),
		RunE: func(cmd *cobra.Command, args []string) error {
			if format != formatJSON && format != formatNPY {
				return usageError{fmt.Errorf("--format %q: want %s or %s", format, formatJSON, formatNPY)}
			}

			m, err := whistle.Load(model)
			if err != nil {
				return err
			}

			defer func() { _ = m.Close() }()

			e, err := embedWAV(m, args[0])
			if err != nil {
				return err
			}

			if output == "" || output == "-" {
				return writeEmbedding(cmd.OutOrStdout(), e, format)
			}

			return writeEmbeddingFile(output, e, format)
		},
	}

	cmd.Flags().StringVar(&model, "model", defaultModelPath, "Model file: whistle.cact or whistle.safetensors")
	cmd.Flags().StringVar(&format, "format", formatJSON, "Output format: json or npy")
	cmd.Flags().StringVarP(&output, "output", "o", "", "Output file; default: stdout")

	return cmd
}

// embedWAV is Embed on a WAV file, with TranscribeWAV's checks.
func embedWAV(m *whistle.Model, path string) (whistle.Embedding, error) {
	pcm, err := frontend.ReadWAV(path)
	if err != nil {
		return whistle.Embedding{}, fileError(path, err)
	}

	if len(pcm) == 0 {
		return whistle.Embedding{}, fmt.Errorf("%s %w", path, whistle.ErrNoAudio)
	}

	e, err := m.Embed(pcm)
	if err != nil {
		return whistle.Embedding{}, fileError(path, err)
	}

	return e, nil
}

// writeEmbeddingFile writes e to the file at path.
func writeEmbeddingFile(path string, e whistle.Embedding, format string) error {
	f, err := os.Create(path) // #nosec G304 -- the user names the file.
	if err != nil {
		return fmt.Errorf("output: %w", err)
	}

	err = writeEmbedding(f, e, format)

	cerr := f.Close()
	if err == nil && cerr != nil {
		err = fmt.Errorf("output: %w", cerr)
	}

	return err
}

// writeEmbedding writes e to w as JSON or as a .npy array.
func writeEmbedding(w io.Writer, e whistle.Embedding, format string) error {
	bw := bufio.NewWriter(w)

	if format == formatNPY {
		writeNPY(bw, e)
	} else {
		writeEmbeddingJSON(bw, e)
	}

	err := bw.Flush()
	if err != nil {
		return fmt.Errorf("write embedding: %w", err)
	}

	return nil
}

// writeEmbeddingJSON writes e as one JSON object. bufio.Writer keeps the
// first write error for Flush.
func writeEmbeddingJSON(w *bufio.Writer, e whistle.Embedding) {
	fmt.Fprintf(w, `{"frames":%d,"dim":%d,"frame_ms":%d,"embedding":[`, e.Frames, e.Dim,
		whistle.FrameDuration.Milliseconds())

	var num []byte

	for i := range e.Frames {
		if i > 0 {
			_ = w.WriteByte(',')
		}

		_ = w.WriteByte('[')

		for j, v := range e.Frame(i) {
			if j > 0 {
				_ = w.WriteByte(',')
			}

			num = strconv.AppendFloat(num[:0], float64(v), 'g', -1, 32)
			_, _ = w.Write(num)
		}

		_ = w.WriteByte(']')
	}

	_, _ = w.WriteString("]}\n")
}

// writeNPY writes e as a NumPy .npy file, format version 1.0: the magic,
// the header length, a header dict padded with spaces to a multiple of 64
// bytes and ending in '\n', then the float32 values little-endian.
func writeNPY(w *bufio.Writer, e whistle.Embedding) {
	header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", e.Frames, e.Dim)

	const prefix = 10 // magic (6), version (2), header length (2)

	pad := 63 - (prefix+len(header))%64 // spaces so that prefix + header + '\n' is a multiple of 64
	header += strings.Repeat(" ", pad) + "\n"

	_, _ = w.WriteString("\x93NUMPY\x01\x00")
	_ = binary.Write(w, binary.LittleEndian, uint16(len(header))) // #nosec G115 -- under 128 bytes.
	_, _ = w.WriteString(header)

	var b [4]byte

	for _, v := range e.Values[:e.Frames*e.Dim] {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		_, _ = w.Write(b[:])
	}
}
