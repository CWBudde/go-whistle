//go:build oracle

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
)

// depthCases are the corpus cases answered along the whole --depth ladder: one call, two calls and a refusal.
var depthCases = []string{"timer_pasta", "lights_and_lock", "offtopic_capital"}

// depthRecorded are the depths whose answers depth.json keeps in full; the others are summarised by same_as_full.
var depthRecorded = []int{2, 5, 10, 15, 19}

// depthFixture is testdata/text/engine/depth.json.
type depthFixture struct {
	header

	FullDepth int                        `json:"full_depth"`
	Loads     []int                      `json:"loads"`
	Refused   map[string]cliResult       `json:"refused"`
	SameFull  map[string][]int           `json:"same_as_full"`
	FullSame  map[string]bool            `json:"full_same_as_default"`
	Answers   map[string]json.RawMessage `json:"answers"`
}

// cmdDepth runs --depth 0 … full+1 on the depth cases: which depths load, which give the default answer, and the
// answers at depthRecorded. needle --help calls it the "endpoint-preserving ladder depth (2..full; default full)";
// full is config.json's num_hidden_layers.
func cmdDepth(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	full, err := layerCount(filepath.Join(cfg.needle3Dir, "config.json"))
	if err != nil {
		return err
	}

	fx := depthFixture{
		header:    o.header("needle --model needle3.cact --tools <toolset> --system <system file> --threads 4 --prompt <query> --depth N"),
		FullDepth: full, Loads: []int{}, Refused: map[string]cliResult{}, SameFull: map[string][]int{},
		FullSame: map[string]bool{}, Answers: map[string]json.RawMessage{},
	}

	for _, id := range depthCases {
		k, err := o.corpus.find(id)
		if err != nil {
			return err
		}

		err = o.depthLadder(ctx, k, full, &fx, id == depthCases[0], log)
		if err != nil {
			return fmt.Errorf("%s: %w", id, err)
		}
	}

	return writeJSON(cfg.outDir, "depth.json", fx)
}

// depthLadder answers case k at the default depth and at --depth 0 … full+1; with probe set it records which depths
// load into fx.
func (o *oracle) depthLadder(ctx context.Context, k textCase, full int, fx *depthFixture, probe bool,
	log io.Writer,
) error {
	args, err := o.baseArgs(k.Toolset, o.corpus.system(k))
	if err != nil {
		return err
	}

	args = maxArgs(k, append(args, "--prompt", k.Query)...)

	def, err := answer(ctx, o.needle(), args)
	if err != nil {
		return err
	}

	fx.SameFull[k.ID] = []int{}

	for d := 0; d <= full+1; d++ {
		res, err := runCLI(ctx, o.needle(), append(slices.Clone(args), "--depth", itoa(d)))
		if err != nil {
			return err
		}

		if res.Exit != 0 {
			if probe {
				fx.Refused[itoa(d)] = res
			}

			_, _ = fmt.Fprintf(log, "%-18s d%-2d exit %d: %s\n", k.ID, d, res.Exit, res.Stderr)

			continue
		}

		if probe {
			fx.Loads = append(fx.Loads, d)
		}

		a, err := normaliseResponse(res.stdout)
		if err != nil {
			return fmt.Errorf("depth %d: %w", d, err)
		}

		same := equalNodes(a, def)
		if same {
			fx.SameFull[k.ID] = append(fx.SameFull[k.ID], d)
		}

		if d == full {
			fx.FullSame[k.ID] = same
		}

		if slices.Contains(depthRecorded, d) {
			fx.Answers[fmt.Sprintf("%s.d%d", k.ID, d)] = compact(a)
		}

		_, _ = fmt.Fprintf(log, "%-18s d%-2d same=%-5t %s\n", k.ID, d, same, summary(a))
	}

	return nil
}

// layerCount reads num_hidden_layers from the model's config.json.
func layerCount(p string) (int, error) {
	raw, err := os.ReadFile(p) // #nosec G304 -- the model dir is a flag.
	if err != nil {
		return 0, fmt.Errorf("read config: %w", err)
	}

	var c struct {
		Layers int `json:"num_hidden_layers"`
	}

	err = json.Unmarshal(raw, &c)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", p, err)
	}

	if c.Layers <= 0 {
		return 0, fmt.Errorf("%s has no num_hidden_layers", p)
	}

	return c.Layers, nil
}
