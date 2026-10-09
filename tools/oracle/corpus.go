//go:build oracle

package main

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/mewkiz/flac"
)

// Clips are 2–8 s long: long enough for a sentence, short enough to keep testdata/audio small.
const (
	minSamples = 2 * sampleRate
	maxSamples = 8 * sampleRate

	// maxPerSpeaker spreads the LibriSpeech clips over speakers.
	maxPerSpeaker = 2

	// streamCap bounds how much of one tarball is read before giving up; the selection normally needs a few MB.
	streamCap = 256 << 20

	librispeechURL = "https://www.openslr.org/resources/12/%s.tar.gz"
	fleursRevision = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
	fleursURL      = "https://huggingface.co/datasets/google/fleurs/resolve/" + fleursRevision + "/data/%s/%s"

	datasetLibriSpeech = "librispeech"
	datasetFleurs      = "fleurs"
	splitTest          = "test"
)

// clip is one corpus entry of testdata/audio/manifest.json.
type clip struct {
	ID         string `json:"id"`
	Language   string `json:"language"`
	Locale     string `json:"locale,omitempty"`
	Dataset    string `json:"dataset"`
	Split      string `json:"split"`
	Archive    string `json:"archive"`
	Member     string `json:"member"`
	Transcript string `json:"transcript"`
	Samples    int    `json:"samples"`
	SHA256     string `json:"sha256"`

	pcm []int16
}

func (c clip) key() string { return c.Dataset + "/" + c.Split + "/" + c.Locale }

type corpusManifest struct {
	Clips []clip `json:"clips"`
}

// corpusSpec is one group of clips taken from one tarball.
type corpusSpec struct {
	dataset string // "librispeech" or "fleurs"
	split   string // LibriSpeech split, or the FLEURS split "test"
	locale  string // FLEURS locale such as "de_de"; empty for LibriSpeech
	want    int
}

func (s corpusSpec) key() string { return s.dataset + "/" + s.split + "/" + s.locale }

// corpus is the oracle fixture set: 20 English clips (PLAN.md Phase 0) and 2 per other Whistle language.
var corpus = []corpusSpec{
	{dataset: datasetLibriSpeech, split: "test-clean", want: 10},
	{dataset: datasetLibriSpeech, split: "test-other", want: 10},
	{dataset: datasetFleurs, split: splitTest, locale: "de_de", want: 2},
	{dataset: datasetFleurs, split: splitTest, locale: "fr_fr", want: 2},
	{dataset: datasetFleurs, split: splitTest, locale: "es_419", want: 2},
	{dataset: datasetFleurs, split: splitTest, locale: "it_it", want: 2},
	{dataset: datasetFleurs, split: splitTest, locale: "nl_nl", want: 2},
	{dataset: datasetFleurs, split: splitTest, locale: "pl_pl", want: 2},
}

func inWindow(n int) bool { return n >= minSamples && n <= maxSamples }

// validID reports whether id is safe to use as a file name: lower-case letters, digits, '-' and '_'.
func validID(id string) bool {
	if id == "" {
		return false
	}

	for _, r := range id {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '-' && r != '_' {
			return false
		}
	}

	return true
}

// parseTrans reads a LibriSpeech <speaker>-<chapter>.trans.txt: one "<utterance-id> <TRANSCRIPT>" per line.
func parseTrans(r io.Reader) (map[string]string, error) {
	out := map[string]string{}
	sc := bufio.NewScanner(r)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}

		id, text, ok := strings.Cut(line, " ")
		if !ok || strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("transcript line without text: %q", line)
		}

		out[id] = strings.TrimSpace(text)
	}

	err := sc.Err()
	if err != nil {
		return nil, fmt.Errorf("read transcripts: %w", err)
	}

	return out, nil
}

// lsSelection is the state of selectLibriSpeech while it walks one split's tar stream.
type lsSelection struct {
	split      string
	want       int
	decode     func(io.Reader) ([]int16, int, error)
	perSpeaker map[string]int
	trans      map[string]string
	got        []clip
	utts       []string
}

// missing lists the chosen utterances whose transcript has not been seen yet.
func (s *lsSelection) missing() []string {
	var m []string

	for _, u := range s.utts {
		if _, ok := s.trans[u]; !ok {
			m = append(m, u)
		}
	}

	return m
}

// flac considers one .flac member for the selection.
func (s *lsSelection) flac(name string, r io.Reader) error {
	utt := strings.TrimSuffix(path.Base(name), ".flac")
	spk, _, _ := strings.Cut(utt, "-")

	if s.perSpeaker[spk] >= maxPerSpeaker {
		return nil
	}

	pcm, rate, err := s.decode(r)
	if err != nil {
		return fmt.Errorf("decode %s: %w", name, err)
	}

	if rate != sampleRate {
		return fmt.Errorf("%s: %d Hz, want %d", name, rate, sampleRate)
	}

	if !inWindow(len(pcm)) {
		return nil
	}

	s.perSpeaker[spk]++
	s.utts = append(s.utts, utt)
	s.got = append(s.got, clip{
		ID:       "ls-" + strings.TrimPrefix(s.split, "test-") + "-" + utt,
		Language: "en",
		Dataset:  datasetLibriSpeech,
		Split:    s.split,
		Member:   name,
		Samples:  len(pcm),
		pcm:      pcm,
	})

	return nil
}

// selectLibriSpeech takes the first want clips of a LibriSpeech split tar stream, in tar order, that last 2–8 s,
// with at most maxPerSpeaker per speaker. It reads on until every chosen clip's chapter transcript has been seen.
func selectLibriSpeech(r io.Reader, split string, want int,
	decode func(io.Reader) ([]int16, int, error),
) ([]clip, error) {
	tr := tar.NewReader(r)
	s := &lsSelection{
		split: split, want: want, decode: decode, perSpeaker: map[string]int{}, trans: map[string]string{},
	}

	for len(s.got) < want || len(s.missing()) > 0 {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			if len(s.got) < want {
				return nil, fmt.Errorf("%s: found %d of %d clips", split, len(s.got), want)
			}

			return nil, fmt.Errorf("%s: no transcript for %s", split, strings.Join(s.missing(), ", "))
		}

		if err != nil {
			return nil, fmt.Errorf("read %s tar: %w", split, err)
		}

		switch {
		case strings.HasSuffix(hdr.Name, ".trans.txt"):
			m, err := parseTrans(tr)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", hdr.Name, err)
			}

			maps.Copy(s.trans, m)
		case strings.HasSuffix(hdr.Name, ".flac") && len(s.got) < want:
			err := s.flac(hdr.Name, tr)
			if err != nil {
				return nil, err
			}
		}
	}

	for i := range s.got {
		s.got[i].Transcript = s.trans[s.utts[i]]
	}

	return s.got, nil
}

// decodeFLAC decodes a 16-bit FLAC stream to mono samples, averaging the channels.
func decodeFLAC(r io.Reader) ([]int16, int, error) {
	stream, err := flac.New(r)
	if err != nil {
		return nil, 0, fmt.Errorf("flac: %w", err)
	}

	if stream.Info.BitsPerSample != 16 {
		return nil, 0, fmt.Errorf("flac: %d bits per sample, want 16", stream.Info.BitsPerSample)
	}

	pcm := make([]int16, 0, stream.Info.NSamples)

	for {
		fr, err := stream.ParseNext()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			return nil, 0, fmt.Errorf("flac frame: %w", err)
		}

		ch := fr.Subframes
		for i := range ch[0].Samples {
			var sum int32
			for _, sf := range ch {
				sum += sf.Samples[i]
			}

			pcm = append(pcm, int16(sum/int32(len(ch)))) // #nosec G115 -- the mean of 16-bit samples fits.
		}
	}

	return pcm, int(stream.Info.SampleRate), nil
}

type fleursRow struct {
	sentence   string
	transcript string
}

// parseFleursTSV reads a FLEURS <split>.tsv: sentence id, wav file name, raw transcription, normalised
// transcription, … per line. It returns the rows by file name.
func parseFleursTSV(r io.Reader) (map[string]fleursRow, error) {
	out := map[string]fleursRow{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)

	for sc.Scan() {
		if sc.Text() == "" {
			continue
		}

		f := strings.Split(sc.Text(), "\t")
		if len(f) < 3 {
			return nil, fmt.Errorf("fleurs tsv: %d fields in %q", len(f), sc.Text())
		}

		out[f[1]] = fleursRow{sentence: f[0], transcript: f[2]}
	}

	err := sc.Err()
	if err != nil {
		return nil, fmt.Errorf("read fleurs tsv: %w", err)
	}

	return out, nil
}

// selectFleurs takes the first want clips of a FLEURS audio tar stream, in tar order, that last 2–8 s, have a row
// in the tsv and read a sentence no earlier clip read.
func selectFleurs(r io.Reader, locale string, want int, rows map[string]fleursRow) ([]clip, error) {
	tr := tar.NewReader(r)
	used := map[string]bool{}

	var got []clip

	for len(got) < want {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("fleurs %s: found %d of %d clips", locale, len(got), want)
		}

		if err != nil {
			return nil, fmt.Errorf("read fleurs %s tar: %w", locale, err)
		}

		name := path.Base(hdr.Name)

		row, ok := rows[name]
		if !ok || !strings.HasSuffix(name, ".wav") || used[row.sentence] {
			continue
		}

		raw, err := io.ReadAll(tr)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", hdr.Name, err)
		}

		pcm, rate, err := readWAV(raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", hdr.Name, err)
		}

		if rate != sampleRate {
			return nil, fmt.Errorf("%s: %d Hz, want %d", hdr.Name, rate, sampleRate)
		}

		if !inWindow(len(pcm)) {
			continue
		}

		used[row.sentence] = true

		got = append(got, clip{
			ID:         "fleurs-" + locale + "-" + strings.TrimSuffix(name, ".wav"),
			Language:   locale[:2],
			Locale:     locale,
			Dataset:    datasetFleurs,
			Split:      splitTest,
			Member:     hdr.Name,
			Transcript: row.transcript,
			Samples:    len(pcm),
			pcm:        toInt16(pcm),
		})
	}

	return got, nil
}

// get opens url for streaming. The caller closes the body.
func get(ctx context.Context, url string) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req) // #nosec G107 G704 -- URLs are the pinned dataset constants.
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", url, err)
	}

	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	return resp.Body, nil
}

// fetchSpec streams one tarball and selects its clips.
func fetchSpec(ctx context.Context, s corpusSpec) ([]clip, error) {
	var (
		archive string
		rows    map[string]fleursRow
	)

	if s.dataset == datasetLibriSpeech {
		archive = fmt.Sprintf(librispeechURL, s.split)
	} else {
		archive = fmt.Sprintf(fleursURL, s.locale, "audio/"+s.split+".tar.gz")

		body, err := get(ctx, fmt.Sprintf(fleursURL, s.locale, s.split+".tsv"))
		if err != nil {
			return nil, err
		}

		rows, err = parseFleursTSV(body)
		_ = body.Close()

		if err != nil {
			return nil, err
		}
	}

	body, err := get(ctx, archive)
	if err != nil {
		return nil, err
	}

	defer func() { _ = body.Close() }()

	gz, err := gzip.NewReader(io.LimitReader(body, streamCap))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", archive, err)
	}

	var clips []clip

	if s.dataset == datasetLibriSpeech {
		clips, err = selectLibriSpeech(gz, s.split, s.want, decodeFLAC)
	} else {
		clips, err = selectFleurs(gz, s.locale, s.want, rows)
	}

	if err != nil {
		return nil, fmt.Errorf("%w (read at most %d MB of %s)", err, streamCap>>20, archive)
	}

	for i := range clips {
		clips[i].Archive = archive
	}

	return clips, nil
}

func fileSHA256(p string) (string, error) {
	b, err := os.ReadFile(p) // #nosec G304 G703 -- dir is a flag, ids are checked by validID.
	if err != nil {
		return "", fmt.Errorf("hash: %w", err)
	}

	sum := sha256.Sum256(b)

	return hex.EncodeToString(sum[:]), nil
}

func readCorpus(dir string) (corpusManifest, error) {
	var m corpusManifest

	b, err := os.ReadFile(filepath.Join(dir, "manifest.json")) // #nosec G304 G703 -- dir is a flag.
	if err != nil {
		return m, fmt.Errorf("read corpus manifest: %w", err)
	}

	err = json.Unmarshal(b, &m)
	if err != nil {
		return m, fmt.Errorf("corpus manifest: %w", err)
	}

	for _, c := range m.Clips {
		if !validID(c.ID) {
			return m, fmt.Errorf("corpus manifest: invalid clip id %q", c.ID)
		}
	}

	return m, nil
}

// cmdClips fills dir with the corpus WAVs, manifest.json and ATTRIBUTION.md. A group whose clips are all present
// with the manifest's digest is not downloaded again.
func cmdClips(ctx context.Context, dir string, log io.Writer) error {
	old, err := readCorpus(dir)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}

	err = os.MkdirAll(dir, 0o750) // #nosec G703 -- dir is the operator's -audio flag.
	if err != nil {
		return fmt.Errorf("create %s: %w", dir, err)
	}

	byKey := map[string][]clip{}
	for _, c := range old.Clips {
		byKey[c.key()] = append(byKey[c.key()], c)
	}

	var all []clip

	for _, s := range corpus {
		if prev := byKey[s.key()]; len(prev) == s.want && verifiedClips(dir, prev) {
			_, _ = fmt.Fprintf(log, "ok       %-28s %d clips (sha256 verified)\n", s.key(), len(prev))
			all = append(all, prev...)

			continue
		}

		clips, err := fetchSpec(ctx, s)
		if err != nil {
			return err
		}

		for i := range clips {
			err := writeClip(dir, &clips[i])
			if err != nil {
				return err
			}
		}

		_, _ = fmt.Fprintf(log, "fetched  %-28s %d clips\n", s.key(), len(clips))
		all = append(all, clips...)
	}

	js, err := json.MarshalIndent(corpusManifest{Clips: all}, "", "  ")
	if err != nil {
		return fmt.Errorf("corpus manifest: %w", err)
	}

	err = os.WriteFile(filepath.Join(dir, "manifest.json"), append(js, '\n'), 0o600) // #nosec G703 -- see MkdirAll.
	if err != nil {
		return fmt.Errorf("write corpus manifest: %w", err)
	}

	err = os.WriteFile(filepath.Join(dir, "ATTRIBUTION.md"), []byte(attribution), 0o600) // #nosec G703 -- see MkdirAll.
	if err != nil {
		return fmt.Errorf("write attribution: %w", err)
	}

	return nil
}

func verifiedClips(dir string, clips []clip) bool {
	for _, c := range clips {
		sum, err := fileSHA256(filepath.Join(dir, c.ID+".wav"))
		if err != nil || sum != c.SHA256 {
			return false
		}
	}

	return true
}

func writeClip(dir string, c *clip) error {
	f, err := os.CreateTemp(dir, c.ID+".*.tmp")
	if err != nil {
		return fmt.Errorf("create %s: %w", c.ID, err)
	}

	h := sha256.New()

	err = writeWAV16(io.MultiWriter(f, h), c.pcm)
	closeErr := f.Close()

	if err == nil {
		err = closeErr
	}

	if err == nil {
		err = os.Rename(f.Name(), filepath.Join(dir, c.ID+".wav")) // #nosec G703 -- ids are checked by validID.
	}

	if err != nil {
		_ = os.Remove(f.Name()) // #nosec G703 -- a temp file this function created.
		return fmt.Errorf("write %s: %w", c.ID, err)
	}

	c.SHA256 = hex.EncodeToString(h.Sum(nil))

	return nil
}

const attribution = `# Audio fixture attribution

The clips in this directory are short excerpts, converted to 16 kHz 16-bit mono WAV, of two datasets published under
the Creative Commons Attribution 4.0 International licence (https://creativecommons.org/licenses/by/4.0/).
` + "`manifest.json`" + ` names the archive and member every clip was taken from. They are regenerated by
` + "`just oracle-clips`" + ` (tools/oracle).

- **LibriSpeech ASR corpus** (` + "`ls-*.wav`" + `), test-clean and test-other. V. Panayotov, G. Chen, D. Povey and
  S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015.
  https://www.openslr.org/12
- **FLEURS** (` + "`fleurs-*.wav`" + `), test split, ` + "`google/fleurs`" + ` revision ` + fleursRevision + `.
  A. Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", SLT 2022.
  https://huggingface.co/datasets/google/fleurs

Changes: decoded, down-mixed to mono and re-quantised to 16-bit PCM; nothing else.
`
