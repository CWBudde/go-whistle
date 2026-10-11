//go:build oracle

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// sampleRate is the only rate needle_complete's pcm takes.
const sampleRate = 16000

// audioFixture is testdata/text/engine/audio.json.
type audioFixture struct {
	header

	Models int32                     `json:"needle_models"`
	Cases  map[string]audioRecording `json:"cases"`
}

// audioRecording is one speech case: the clip, the CLI's fused answer (its audio_* fields are the transcription),
// whether libneedle3's needle_complete on the clip's pcm gives the same JSON, and whether the CLI's text answer to
// the transcript (audio_text) equals the fused answer without its audio_* fields.
type audioRecording struct {
	Clip       string          `json:"clip"`
	ClipSHA256 string          `json:"clip_sha256"`
	Toolset    string          `json:"toolset"`
	Answer     json.RawMessage `json:"answer"`
	LibraryRC  int32           `json:"library_complete"`
	SameAsCLI  bool            `json:"library_same_as_cli"`
	SameAsText bool            `json:"same_as_text_prompt"`
	Library    json.RawMessage `json:"library_answer,omitempty"`
	TextAnswer json.RawMessage `json:"text_answer,omitempty"`
}

// cmdAudio answers every speech case of the corpus with both models loaded: the CLI on the clip's WAV, libneedle3 on
// its pcm, and the CLI on the transcript as text.
func cmdAudio(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, true)
	if err != nil {
		return err
	}
	defer o.close()

	if len(o.corpus.Audio) == 0 {
		return errors.New("the corpus has no audio cases")
	}

	clips, err := readClipDigests(filepath.Join(cfg.audioDir, "manifest.json"))
	if err != nil {
		return err
	}

	whistle := filepath.Join(cfg.whistleDir, o.whistle.File)

	lib, err := openLibrary(cfg.engineDir, o.needle3Path(), whistle)
	if err != nil {
		return err
	}

	if m := lib.models(); m != needleText|needleSpeech {
		return fmt.Errorf("needle_models is %d after loading both models, want %d", m, needleText|needleSpeech)
	}

	lib.setAudio(nil, nil, 0) // the defaults: detect the language, no keywords, no word times

	fx := audioFixture{
		header: o.speechHeader("needle --model needle3.cact --model whistle.cact --tools <toolset> --system <system file> " +
			"--threads 4 --audio <clip>.wav"),
		Models: lib.models(), Cases: map[string]audioRecording{},
	}

	for _, a := range o.corpus.Audio {
		rec, err := o.recordAudio(ctx, lib, whistle, a, clips[a.Clip])
		if err != nil {
			return fmt.Errorf("%s: %w", a.ID, err)
		}

		fx.Cases[a.ID] = rec
		_, _ = fmt.Fprintf(log, "%-20s library=%t text=%t %s\n", a.ID, rec.SameAsCLI, rec.SameAsText, rec.Answer)
	}

	return writeJSON(cfg.outDir, "audio.json", fx)
}

func (o *oracle) recordAudio(ctx context.Context, lib *library, whistle string, a audioCase,
	digest string,
) (audioRecording, error) {
	if digest == "" {
		return audioRecording{}, fmt.Errorf("clip %s is not in the clip manifest", a.Clip)
	}

	wav := filepath.Join(o.cfg.audioDir, a.Clip+".wav")

	raw, err := os.ReadFile(wav) // #nosec G304 -- clip ids come from the corpus and the clip manifest.
	if err != nil {
		return audioRecording{}, fmt.Errorf("read clip: %w", err)
	}

	if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != digest {
		return audioRecording{}, fmt.Errorf("clip %s does not match the clip manifest's sha256", a.Clip)
	}

	pcm, err := readPCM16(raw)
	if err != nil {
		return audioRecording{}, fmt.Errorf("%s: %w", a.Clip, err)
	}

	args, err := o.baseArgs(a.Toolset, o.corpus.System)
	if err != nil {
		return audioRecording{}, err
	}

	cli, err := answer(ctx, o.needle(), append(slicesWith(args, "--model", whistle), "--audio", wav))
	if err != nil {
		return audioRecording{}, err
	}

	rec := audioRecording{Clip: a.Clip, ClipSHA256: digest, Toolset: a.Toolset, Answer: compact(cli)}

	got, rc, err := o.speechTurn(lib, a.Toolset, pcm)
	if err != nil {
		return audioRecording{}, err
	}

	rec.LibraryRC, rec.SameAsCLI = rc, equalNodes(got, cli)
	if !rec.SameAsCLI {
		rec.Library = compact(got)
	}

	text := cli.get("audio_text")
	if text == nil || text.str == "" {
		return rec, nil
	}

	ta, err := answer(ctx, o.needle(), append(args, "--prompt", text.str))
	if err != nil {
		return audioRecording{}, err
	}

	rec.SameAsText = equalNodes(ta, cli.without(func(k string) bool { return strings.HasPrefix(k, "audio_") }))
	if !rec.SameAsText {
		rec.TextAnswer = compact(ta)
	}

	return rec, nil
}

// speechTurn answers pcm with libneedle3: needle_init with the corpus's system text and a tool set, needle_reset,
// needle_complete on the samples. It returns the answer and needle_complete's return.
func (o *oracle) speechTurn(lib *library, toolset string, pcm []float32) (node, int32, error) {
	_, err := lib.session(o.corpus.System, o.corpus.tools(toolset))
	if err != nil {
		return node{}, 0, err
	}

	lib.reset()

	c, err := lib.completeSpeech(pcm, defaultMax)
	if err != nil {
		return node{}, 0, err
	}

	got, err := normaliseResponse(c.out)
	if err != nil {
		return node{}, 0, err
	}

	return got, c.rc, nil
}

// slicesWith is args with the extra flags inserted after the first two (--model needle3.cact), so the text model
// stays the first --model.
func slicesWith(args []string, extra ...string) []string {
	out := append([]string{}, args[:2]...)
	out = append(out, extra...)

	return append(out, args[2:]...)
}

// readClipDigests maps every clip id of testdata/audio/manifest.json to its sha256.
func readClipDigests(p string) (map[string]string, error) {
	raw, err := os.ReadFile(p) // #nosec G304 -- the audio dir is a flag.
	if err != nil {
		return nil, fmt.Errorf("read clip manifest: %w", err)
	}

	var m struct {
		Clips []struct {
			ID     string `json:"id"`
			SHA256 string `json:"sha256"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &m)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", p, err)
	}

	out := make(map[string]string, len(m.Clips))
	for _, c := range m.Clips {
		out[c.ID] = c.SHA256
	}

	return out, nil
}

// readPCM16 decodes a 16 kHz mono 16-bit PCM RIFF WAV, the format of every corpus clip, into float32 in [-1, 1).
func readPCM16(raw []byte) ([]float32, error) {
	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, errors.New("not a RIFF WAVE file")
	}

	var (
		fmtOK bool
		data  []byte
	)

	for p := 12; p+8 <= len(raw); {
		id := string(raw[p : p+4])
		size := int(binary.LittleEndian.Uint32(raw[p+4:]))
		body := raw[p+8:]

		if size > len(body) {
			return nil, fmt.Errorf("wav chunk %q overruns the file", id)
		}

		body = body[:size]

		switch id {
		case "fmt ":
			fmtOK = size >= 16 &&
				binary.LittleEndian.Uint16(body) == 1 && // PCM
				binary.LittleEndian.Uint16(body[2:]) == 1 && // mono
				binary.LittleEndian.Uint32(body[4:]) == sampleRate &&
				binary.LittleEndian.Uint16(body[14:]) == 16
		case "data":
			data = body
		}

		p += 8 + size + size%2
	}

	if !fmtOK || data == nil {
		return nil, errors.New("not a 16 kHz mono 16-bit PCM WAV")
	}

	pcm := make([]float32, len(data)/2)
	for i := range pcm {
		pcm[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768 // #nosec G115 -- a bit cast.
	}

	return pcm, nil
}
