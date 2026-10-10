//go:build oracle

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
)

// fullDepth is Whistle's decoder layer count: --audio-depth 8 gives byte-identical JSON to the default on all 32
// clips, so the default is the full ladder (docs/FINDINGS.md F-025).
const fullDepth = 8

// depths are the truncated ladders recorded; the engine refuses 1 ("a layer count of 2 or more").
var depths = []int{2, 3, 4, 5, 6, 7}

// depthManifest is testdata/oracle/depth/manifest.json.
type depthManifest struct {
	fixtureHeader

	Transcripts string       `json:"transcripts"`
	FullDepth   int          `json:"full_depth"`
	Depths      []int        `json:"depths"`
	Clips       []string     `json:"clips"`
	SameText    []depthCount `json:"same_text_as_full"`
}

// depthCount is how many clips keep the full-depth text at one depth.
type depthCount struct {
	Depth int `json:"depth"`
	Clips int `json:"clips"`
}

// cmdDepth records the engine's transcript of every corpus clip at --audio-depth 2..7 as <clip>.d<N> in
// transcripts.json. It first checks that --audio-depth 8 reproduces the committed full-depth fixture byte for byte.
func cmdDepth(ctx context.Context, cfg runConfig, log io.Writer) error {
	corpus, err := readCorpus(cfg.audioDir)
	if err != nil {
		return fmt.Errorf("%w (run `just oracle-clips`)", err)
	}

	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}

	man := depthManifest{
		fixtureHeader: o.header("needle --audio-word-timestamps --audio-depth N"),
		Transcripts:   transcriptsFile + ": <clip>.d<N>", FullDepth: fullDepth, Depths: depths,
	}
	same := make([]int, len(depths))
	tr := bundle{}

	for _, c := range corpus.Clips {
		wav := filepath.Join(cfg.audioDir, c.ID+".wav")

		_, err := readClip(wav, c.SHA256)
		if err != nil {
			return fmt.Errorf("%s: %w", c.ID, err)
		}

		full, err := checkFullDepth(ctx, o, c.ID, wav)
		if err != nil {
			return err
		}

		_, _ = fmt.Fprintf(log, "%-36s full  %s\n", c.ID, full.Text)

		for i, d := range depths {
			js, err := o.cli(ctx, wav, "--audio-depth", strconv.Itoa(d))
			if err != nil {
				return fmt.Errorf("%s depth %d: %w", c.ID, d, err)
			}

			tr.add(fmt.Sprintf("%s.d%d", c.ID, d), js)

			t, err := parseTranscript(js)
			if err != nil {
				return err
			}

			if t.Text == full.Text {
				same[i]++
			}

			_, _ = fmt.Fprintf(log, "%-36s d%d    %s\n", "", d, t.Text)
		}

		man.Clips = append(man.Clips, c.ID)
	}

	for i, d := range depths {
		man.SameText = append(man.SameText, depthCount{Depth: d, Clips: same[i]})
	}

	err = tr.write(cfg.outDir)
	if err != nil {
		return err
	}

	return writeManifest(cfg.outDir, man)
}

// checkFullDepth runs --audio-depth 8 on a clip and requires the committed testdata/oracle/<id>.json byte for byte.
func checkFullDepth(ctx context.Context, o *oracle, id, wav string) (transcript, error) {
	js, err := o.cli(ctx, wav, "--audio-depth", strconv.Itoa(fullDepth))
	if err != nil {
		return transcript{}, fmt.Errorf("%s depth %d: %w", id, fullDepth, err)
	}

	want, err := readFixture(filepath.Join(o.cfg.oracleDir, id+".json"))
	if err != nil {
		return transcript{}, fmt.Errorf("%w (run `just oracle-run`)", err)
	}

	if !bytes.Equal(js, want) {
		return transcript{}, fmt.Errorf("%s: --audio-depth %d differs from %s/%s.json", id, fullDepth,
			o.cfg.oracleDir, id)
	}

	return parseTranscript(js)
}
