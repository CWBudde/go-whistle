//go:build oracle

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func committedSubset(t *testing.T) lsSubset {
	t.Helper()

	b, err := os.ReadFile(filepath.Join("..", "..", "testdata", "oracle", datasetLibriSpeech, "subset.json"))
	if err != nil {
		t.Fatal(err)
	}

	var s lsSubset

	err = json.Unmarshal(b, &s)
	if err != nil {
		t.Fatal(err)
	}

	return s
}

func TestCheckSubset(t *testing.T) {
	s := committedSubset(t)

	err := checkSubset(s)
	if err != nil {
		t.Fatalf("committed subset: %v", err)
	}

	for _, tc := range []struct {
		name  string
		alter func(s *lsSubset)
	}{
		{"one utterance", func(s *lsSubset) { s.Utterances = s.Utterances[:1] }},
		{"seed", func(s *lsSubset) { s.Seed = "other" }},
		{"archive digest", func(s *lsSubset) { s.ArchiveSHA256 = "" }},
		{"selection", func(s *lsSubset) { s.Selection = "any 200" }},
		{"duplicate id", func(s *lsSubset) { s.Utterances[1] = s.Utterances[0] }},
		{"unsorted", func(s *lsSubset) { s.Utterances[0], s.Utterances[1] = s.Utterances[1], s.Utterances[0] }},
		{"speaker", func(s *lsSubset) { s.Utterances[0].Speaker = "1" }},
		{"member", func(s *lsSubset) { s.Utterances[0].Member = "x.flac" }},
		{"no samples", func(s *lsSubset) { s.Utterances[0].Samples = 0 }},
		{"too long", func(s *lsSubset) { s.Utterances[0].Samples = lsMaxSamples + 1 }},
		{"no transcript", func(s *lsSubset) { s.Utterances[0].Transcript = " " }},
		{"digest", func(s *lsSubset) { s.Utterances[0].SHA256 = "ABC" }},
		{"six of one speaker", func(s *lsSubset) {
			u := s.Utterances[5] // the next speaker's first utterance, moved to the first speaker
			u.Speaker, u.Chapter = s.Utterances[4].Speaker, s.Utterances[4].Chapter
			u.ID = s.Utterances[4].ID + "9"
			u.Member = "LibriSpeech/" + lsSplit + "/" + u.Speaker + "/" + u.Chapter + "/" +
				u.ID[len("ls-clean-"):] + ".flac"
			s.Utterances[5] = u
		}},
	} {
		c := s
		c.Utterances = slices.Clone(s.Utterances)
		tc.alter(&c)

		err := checkSubset(c)
		if err == nil {
			t.Errorf("%s: accepted", tc.name)
		} else {
			t.Logf("%s: %v", tc.name, err)
		}
	}
}

// TestEnsureLibriSpeechRejectsStaleSubset: a one-utterance subset.json whose WAV is present is not taken as the
// subset; with the context cancelled the refetch fails instead of returning it.
func TestEnsureLibriSpeechRejectsStaleSubset(t *testing.T) {
	s := committedSubset(t)
	s.Utterances = s.Utterances[:1]

	corpus, out := t.TempDir(), t.TempDir()
	wav := []byte("RIFF stand-in")
	sum := sha256.Sum256(wav)
	s.Utterances[0].SHA256 = hex.EncodeToString(sum[:])

	err := os.WriteFile(filepath.Join(corpus, s.Utterances[0].ID+".wav"), wav, 0o600)
	if err != nil {
		t.Fatal(err)
	}

	js, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile(filepath.Join(out, "subset.json"), append(js, '\n'), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var log bytes.Buffer

	_, err = ensureLibriSpeech(ctx, corpus, out, &log)
	if err == nil {
		t.Fatalf("stale subset accepted; log:\n%s", log.String())
	}

	if !bytes.Contains(log.Bytes(), []byte("refetch")) {
		t.Errorf("log does not say why it refetches:\n%s", log.String())
	}
}
