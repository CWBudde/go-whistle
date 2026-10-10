//go:build oracle

package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"unicode"
)

// The LibriSpeech test-clean subset (PLAN.md 6.1, 6.6): 5 utterances of each of the split's 40 speakers.
const (
	lsSplit      = "test-clean"
	lsSpeakers   = 40
	lsPerSpeaker = 5
	lsSeed       = "go-whistle 6.1"
	lsMaxSamples = 30 * sampleRate // the engine's limit

	// lsArchiveSHA256 pins test-clean.tar.gz (346,663,984 bytes); the subset reads all of it.
	lsArchiveSHA256 = "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
	lsArchiveCap    = 512 << 20

	lsSelectionRule = "per speaker, the 5 utterances of at most 480000 samples with the smallest " +
		"hex SHA-256 of \"<seed>/<utterance id>\""
)

// lsUtterance is one subset entry of testdata/oracle/librispeech/subset.json; its audio is
// models/corpus/librispeech/<id>.wav (16-bit mono 16 kHz, the corpus's canonical WAV).
type lsUtterance struct {
	ID         string `json:"id"`
	Speaker    string `json:"speaker"`
	Chapter    string `json:"chapter"`
	Member     string `json:"member"`
	Samples    int    `json:"samples"`
	SHA256     string `json:"sha256"`
	Transcript string `json:"transcript"`

	key string
	pcm []int16
}

// lsSubset is testdata/oracle/librispeech/subset.json, written by `oracle librispeech-audio`.
type lsSubset struct {
	Dataset       string        `json:"dataset"`
	Archive       string        `json:"archive"`
	ArchiveSHA256 string        `json:"archive_sha256"`
	Seed          string        `json:"seed"`
	Selection     string        `json:"selection"`
	AudioDir      string        `json:"audio_dir"`
	Utterances    []lsUtterance `json:"utterances"`
}

// lsManifest is testdata/oracle/librispeech/manifest.json, written by `oracle librispeech`.
type lsManifest struct {
	fixtureHeader

	Subset      string     `json:"subset"`
	Transcripts string     `json:"transcripts"`
	WERRule     string     `json:"wer_rule"`
	Words       int        `json:"words"`
	Errors      int        `json:"errors"`
	WER         float64    `json:"wer"`
	Utterances  []lsResult `json:"utterances"`
}

type lsResult struct {
	ID       string `json:"id"`
	Text     string `json:"text"`
	Language string `json:"language"`
	Errors   int    `json:"errors"`
}

const werRule = "word edit distance over the reference's words; both sides upper-cased and split at every " +
	"character other than a letter, digit or apostrophe; no other normalisation (numbers, abbreviations)"

// subsetKey ranks an utterance for the seeded selection.
func subsetKey(utt string) string {
	sum := sha256.Sum256([]byte(lsSeed + "/" + utt))

	return hex.EncodeToString(sum[:])
}

// subsetSelection is the state of selectSubset while it walks the split's tar stream.
type subsetSelection struct {
	decode func(io.Reader) ([]int16, int, error)
	picks  map[string][]lsUtterance // by speaker, sorted by key, at most lsPerSpeaker
	trans  map[string]string
}

// flac considers one .flac member: decode it only if its key would enter its speaker's picks.
func (s *subsetSelection) flac(name string, r io.Reader) error {
	utt := strings.TrimSuffix(path.Base(name), ".flac")

	parts := strings.Split(utt, "-")
	if len(parts) != 3 {
		return fmt.Errorf("unexpected utterance id %q", utt)
	}

	key := subsetKey(utt)
	picks := s.picks[parts[0]]

	if len(picks) == lsPerSpeaker && key > picks[len(picks)-1].key {
		return nil
	}

	pcm, rate, err := s.decode(r)
	if err != nil {
		return fmt.Errorf("decode %s: %w", name, err)
	}

	if rate != sampleRate {
		return fmt.Errorf("%s: %d Hz, want %d", name, rate, sampleRate)
	}

	if len(pcm) > lsMaxSamples {
		return nil
	}

	u := lsUtterance{
		ID: "ls-clean-" + utt, Speaker: parts[0], Chapter: parts[1], Member: name, Samples: len(pcm), key: key, pcm: pcm,
	}
	i, _ := slices.BinarySearchFunc(picks, key, func(p lsUtterance, k string) int { return strings.Compare(p.key, k) })
	picks = slices.Insert(picks, i, u)
	s.picks[parts[0]] = picks[:min(len(picks), lsPerSpeaker)]

	return nil
}

// selectSubset walks a LibriSpeech split's tar stream to its end and returns the seeded subset in id order, with
// transcripts.
func selectSubset(r io.Reader, decode func(io.Reader) ([]int16, int, error)) ([]lsUtterance, error) {
	tr := tar.NewReader(r)
	s := &subsetSelection{decode: decode, picks: map[string][]lsUtterance{}, trans: map[string]string{}}

	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			return nil, fmt.Errorf("read %s tar: %w", lsSplit, err)
		}

		switch {
		case strings.HasSuffix(hdr.Name, ".trans.txt"):
			m, err := parseTrans(tr)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", hdr.Name, err)
			}

			maps.Copy(s.trans, m)
		case strings.HasSuffix(hdr.Name, ".flac"):
			err := s.flac(hdr.Name, tr)
			if err != nil {
				return nil, err
			}
		}
	}

	if len(s.picks) != lsSpeakers {
		return nil, fmt.Errorf("%s: %d speakers, want %d", lsSplit, len(s.picks), lsSpeakers)
	}

	var out []lsUtterance

	for _, spk := range slices.Sorted(maps.Keys(s.picks)) {
		picks := s.picks[spk]
		if len(picks) != lsPerSpeaker {
			return nil, fmt.Errorf("%s: speaker %s has %d utterances, want %d", lsSplit, spk, len(picks), lsPerSpeaker)
		}

		for _, u := range picks {
			utt := strings.TrimPrefix(u.ID, "ls-clean-")

			u.Transcript = s.trans[utt]
			if u.Transcript == "" {
				return nil, fmt.Errorf("%s: no transcript for %s", lsSplit, utt)
			}

			out = append(out, u)
		}
	}

	slices.SortFunc(out, func(a, b lsUtterance) int { return strings.Compare(a.ID, b.ID) })

	return out, nil
}

// ensureLibriSpeech makes sure corpusDir holds the subset of outDir/subset.json. When it is missing, or any WAV is
// missing or differs, it streams test-clean.tar.gz from openslr, checks its SHA-256 against lsArchiveSHA256,
// selects the subset and writes the WAVs and subset.json. A committed subset.json must be reproduced exactly.
func ensureLibriSpeech(ctx context.Context, corpusDir, outDir string, log io.Writer) (lsSubset, error) {
	p := filepath.Join(outDir, "subset.json")

	var old lsSubset

	b, err := os.ReadFile(p) // #nosec G304 G703 -- outDir is a flag.
	if err == nil {
		err = json.Unmarshal(b, &old)
		if err != nil {
			return old, fmt.Errorf("%s: %w", p, err)
		}

		err = checkSubset(old)
		if err != nil {
			_, _ = fmt.Fprintf(log, "refetch  %s: %v\n", p, err)
		} else if verifiedUtterances(corpusDir, old.Utterances) {
			_, _ = fmt.Fprintf(log, "ok       %s: %d utterances (sha256 verified)\n", corpusDir, len(old.Utterances))

			return old, nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return old, fmt.Errorf("read %s: %w", p, err)
	}

	sub, err := fetchSubset(ctx, corpusDir)
	if err != nil {
		return sub, err
	}

	js, err := json.MarshalIndent(sub, "", "  ")
	if err != nil {
		return sub, fmt.Errorf("subset: %w", err)
	}

	js = append(js, '\n')

	if len(old.Utterances) > 0 && !bytes.Equal(js, b) {
		return sub, fmt.Errorf("the fetched subset differs from the committed %s", p)
	}

	err = os.MkdirAll(outDir, 0o750) // #nosec G703 -- outDir is the operator's -out flag.
	if err != nil {
		return sub, fmt.Errorf("create %s: %w", outDir, err)
	}

	_, _ = fmt.Fprintf(log, "fetched  %s: %d utterances\n", corpusDir, len(sub.Utterances))

	return sub, writeFile(p, js)
}

// checkSubset checks what a subset.json must hold before its WAVs alone vouch for it: the header fetchSubset
// writes, lsPerSpeaker utterances of each of lsSpeakers speakers, sorted unique IDs that agree with their speaker,
// chapter and archive member, lengths within the engine's limit, a transcript and a SHA-256 each. The selection
// itself needs the archive; a subset that passes but differs from it fails when it is fetched again.
func checkSubset(s lsSubset) error {
	want := lsSubset{
		Dataset: "LibriSpeech " + lsSplit, Archive: fmt.Sprintf(librispeechURL, lsSplit), ArchiveSHA256: lsArchiveSHA256,
		Seed: lsSeed, Selection: lsSelectionRule,
		AudioDir: filepath.ToSlash(filepath.Join("models", "corpus", datasetLibriSpeech)),
	}

	for _, f := range []struct{ name, got, want string }{
		{"dataset", s.Dataset, want.Dataset},
		{"archive", s.Archive, want.Archive},
		{"archive_sha256", s.ArchiveSHA256, want.ArchiveSHA256},
		{"seed", s.Seed, want.Seed},
		{"selection", s.Selection, want.Selection},
		{"audio_dir", s.AudioDir, want.AudioDir},
	} {
		if f.got != f.want {
			return fmt.Errorf("subset %s is %q, want %q", f.name, f.got, f.want)
		}
	}

	if len(s.Utterances) != lsSpeakers*lsPerSpeaker {
		return fmt.Errorf("subset has %d utterances, want %d", len(s.Utterances), lsSpeakers*lsPerSpeaker)
	}

	perSpeaker := map[string]int{}

	for i, u := range s.Utterances {
		err := checkUtterance(u)
		if err != nil {
			return err
		}

		if i > 0 && u.ID <= s.Utterances[i-1].ID {
			return fmt.Errorf("subset utterance %s is not after %s", u.ID, s.Utterances[i-1].ID)
		}

		perSpeaker[u.Speaker]++
	}

	if len(perSpeaker) != lsSpeakers {
		return fmt.Errorf("subset has %d speakers, want %d", len(perSpeaker), lsSpeakers)
	}

	for spk, n := range perSpeaker {
		if n != lsPerSpeaker {
			return fmt.Errorf("subset speaker %s has %d utterances, want %d", spk, n, lsPerSpeaker)
		}
	}

	return nil
}

// checkUtterance checks one subset entry against itself.
func checkUtterance(u lsUtterance) error {
	parts := strings.Split(strings.TrimPrefix(u.ID, "ls-clean-"), "-")

	switch {
	case !validID(u.ID) || !strings.HasPrefix(u.ID, "ls-clean-") || len(parts) != 3:
		return fmt.Errorf("subset utterance id %q is not ls-clean-<speaker>-<chapter>-<n>", u.ID)
	case parts[0] != u.Speaker || parts[1] != u.Chapter:
		return fmt.Errorf("subset utterance %s: speaker %q, chapter %q", u.ID, u.Speaker, u.Chapter)
	case u.Member != path.Join("LibriSpeech", lsSplit, u.Speaker, u.Chapter, strings.Join(parts, "-")+".flac"):
		return fmt.Errorf("subset utterance %s: member %q", u.ID, u.Member)
	case u.Samples < 1 || u.Samples > lsMaxSamples:
		return fmt.Errorf("subset utterance %s: %d samples, want 1 to %d", u.ID, u.Samples, lsMaxSamples)
	case strings.TrimSpace(u.Transcript) == "":
		return fmt.Errorf("subset utterance %s: no transcript", u.ID)
	case !isSHA256Hex(u.SHA256):
		return fmt.Errorf("subset utterance %s: sha256 %q is not 64 lowercase hex digits", u.ID, u.SHA256)
	}

	return nil
}

// isSHA256Hex reports whether s is a SHA-256 digest in lowercase hex.
func isSHA256Hex(s string) bool {
	if len(s) != 2*sha256.Size {
		return false
	}

	for _, r := range s {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return false
		}
	}

	return true
}

func verifiedUtterances(dir string, us []lsUtterance) bool {
	if len(us) == 0 {
		return false
	}

	for _, u := range us {
		if !validID(u.ID) {
			return false
		}

		sum, err := fileSHA256(filepath.Join(dir, u.ID+".wav"))
		if err != nil || sum != u.SHA256 {
			return false
		}
	}

	return true
}

// fetchSubset streams the whole split, selects the subset and writes its WAVs into dir.
func fetchSubset(ctx context.Context, dir string) (lsSubset, error) {
	archive := fmt.Sprintf(librispeechURL, lsSplit)
	sub := lsSubset{
		Dataset: "LibriSpeech " + lsSplit, Archive: archive, ArchiveSHA256: lsArchiveSHA256, Seed: lsSeed,
		Selection: lsSelectionRule, AudioDir: filepath.ToSlash(filepath.Join("models", "corpus", datasetLibriSpeech)),
	}

	body, err := get(ctx, archive)
	if err != nil {
		return sub, err
	}

	defer func() { _ = body.Close() }()

	h := sha256.New()
	in := io.TeeReader(io.LimitReader(body, lsArchiveCap), h)

	gz, err := gzip.NewReader(in)
	if err != nil {
		return sub, fmt.Errorf("%s: %w", archive, err)
	}

	us, err := selectSubset(gz, decodeFLAC)
	if err != nil {
		return sub, err
	}

	_, err = io.Copy(io.Discard, in)
	if err != nil {
		return sub, fmt.Errorf("%s: %w", archive, err)
	}

	if got := hex.EncodeToString(h.Sum(nil)); got != lsArchiveSHA256 {
		return sub, fmt.Errorf("%s: sha256 %s, pinned %s", archive, got, lsArchiveSHA256)
	}

	err = os.MkdirAll(dir, 0o750) // #nosec G703 -- dir is the operator's -corpus flag.
	if err != nil {
		return sub, fmt.Errorf("create %s: %w", dir, err)
	}

	for i := range us {
		c := clip{ID: us[i].ID, pcm: us[i].pcm}

		err := writeClip(dir, &c)
		if err != nil {
			return sub, err
		}

		us[i].SHA256 = c.SHA256
	}

	sub.Utterances = us

	return sub, nil
}

// cmdLibriSpeech records the engine's transcript of every subset utterance as <id> in transcripts.json, fetching the
// audio first when it is missing.
func cmdLibriSpeech(ctx context.Context, cfg runConfig, log io.Writer) error {
	sub, err := ensureLibriSpeech(ctx, cfg.corpusDir, cfg.outDir, log)
	if err != nil {
		return err
	}

	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}

	man := lsManifest{
		fixtureHeader: o.header("needle --audio-word-timestamps"), Subset: "subset.json",
		Transcripts: transcriptsFile + ": <id>", WERRule: werRule,
	}
	tr := bundle{}

	for _, u := range sub.Utterances {
		wav := filepath.Join(cfg.corpusDir, u.ID+".wav")

		_, err := readClip(wav, u.SHA256)
		if err != nil {
			return fmt.Errorf("%s: %w", u.ID, err)
		}

		js, err := o.cli(ctx, wav)
		if err != nil {
			return fmt.Errorf("%s: %w", u.ID, err)
		}

		r := lsResult{ID: u.ID}
		tr.add(u.ID, js)

		t, err := parseTranscript(js)
		if err != nil {
			return err
		}

		ref := upperWords(u.Transcript)
		r.Text, r.Language, r.Errors = t.Text, t.Language, wordErrors(ref, upperWords(t.Text))
		man.Words += len(ref)
		man.Errors += r.Errors
		man.Utterances = append(man.Utterances, r)

		_, _ = fmt.Fprintf(log, "%-28s %2d  %s\n", u.ID, r.Errors, t.Text)
	}

	man.WER = float64(man.Errors) / float64(man.Words)
	_, _ = fmt.Fprintf(log, "WER %d/%d = %.4f\n", man.Errors, man.Words, man.WER)

	err = tr.write(cfg.outDir)
	if err != nil {
		return err
	}

	return writeManifest(cfg.outDir, man)
}

// upperWords upper-cases s and splits it at every character other than a letter, digit or apostrophe.
func upperWords(s string) []string {
	return strings.FieldsFunc(strings.ToUpper(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
}

// wordErrors is the edit distance between two word sequences.
func wordErrors(ref, hyp []string) int {
	prev := make([]int, len(hyp)+1)
	cur := make([]int, len(hyp)+1)

	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(ref); i++ {
		cur[0] = i

		for j := 1; j <= len(hyp); j++ {
			sub := prev[j-1]
			if ref[i-1] != hyp[j-1] {
				sub++
			}

			cur[j] = min(sub, prev[j]+1, cur[j-1]+1)
		}

		prev, cur = cur, prev
	}

	return prev[len(hyp)]
}
