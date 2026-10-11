//go:build oracle

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"

	"github.com/cwbudde/go-whistle/internal/hub"
)

// cmdFetch downloads the pinned Needle 3 text model (hub.PinnedNeedle3: needle3.cact, needle3.safetensors,
// config.json, tokenizer.model) into dir, verified by size and SHA-256 and resumed from a .part file. HF_TOKEN,
// HF_HUB_OFFLINE and HF_ENDPOINT apply as for `whistle download`. It needs no engine.
func cmdFetch(ctx context.Context, args []string, log io.Writer) error {
	fs := flag.NewFlagSet("fetch", flag.ContinueOnError)
	fs.SetOutput(log)

	dir := fs.String("dir", filepath.Join("models", "needle3"), "download directory")

	err := fs.Parse(args)
	if err != nil {
		return fmt.Errorf("fetch: %w", err)
	}

	m, err := hub.PinnedNeedle3()
	if err != nil {
		return fmt.Errorf("fetch: %w", err)
	}

	c := hub.NewClientFromEnv()
	c.Progress = log

	err = c.Download(ctx, m, *dir)
	if err != nil {
		return fmt.Errorf("fetch: %w", err)
	}

	_, err = fmt.Fprintf(log, "needle3 %.8s ready in %s\n", m.Revision, *dir)
	if err != nil {
		return fmt.Errorf("fetch: %w", err)
	}

	return nil
}
