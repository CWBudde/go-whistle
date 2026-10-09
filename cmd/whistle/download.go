package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"

	"github.com/cwbudde/go-whistle/internal/hub"
	"github.com/spf13/cobra"
)

// defaultModelDir is where the other commands look for the model files.
var defaultModelDir = filepath.Join("models", "whistle")

func newDownloadCmd() *cobra.Command {
	var (
		dir     string
		offline bool
	)

	cmd := &cobra.Command{
		Use:   "download",
		Short: "Download the pinned Whistle model files from Hugging Face",
		Long: "Download whistle.cact, whistle.safetensors and config.json of the pinned Cactus-Compute/whistle\n" +
			"revision into --dir and verify each by size and SHA-256. Verified files are skipped, and an\n" +
			"interrupted download resumes from its .part file.\n\n" +
			"Environment:\n" +
			"  HF_TOKEN        Hugging Face token, sent as a Bearer token\n" +
			"  HF_HUB_OFFLINE  1/true/yes/on: no network, only verify the files on disk (same as --offline)\n" +
			"  HF_ENDPOINT     Hub base URL (default " + hub.DefaultEndpoint + ")",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			m, err := hub.Pinned()
			if err != nil {
				return err
			}

			c := hub.NewClientFromEnv()
			c.Offline = c.Offline || offline
			c.Progress = cmd.ErrOrStderr()

			return runDownload(cmd.Context(), c, m, dir, cmd.OutOrStdout())
		},
	}

	cmd.Flags().StringVar(&dir, "dir", defaultModelDir, "Directory that receives the model files")
	cmd.Flags().BoolVar(&offline, "offline", false, "Do not use the network; only verify the files already in --dir")

	return cmd
}

// runDownload makes dir hold the verified files of m and reports the result on out.
func runDownload(ctx context.Context, c *hub.Client, m hub.Manifest, dir string, out io.Writer) error {
	err := c.Download(ctx, m, dir)
	if err != nil {
		return fmt.Errorf("download whistle model: %w", err)
	}

	_, err = fmt.Fprintf(out, "whistle %s ready in %s\n", m.Revision[:8], dir)
	if err != nil {
		return fmt.Errorf("write result: %w", err)
	}

	return nil
}
