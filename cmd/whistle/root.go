package main

import "github.com/spf13/cobra"

// NewRootCmd builds the whistle command tree.
func NewRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "whistle",
		Short: "Pure-Go speech-to-text with the Whistle model",
		// main prints the error once; usage is only useful for flag errors, which cobra reports itself.
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	cmd.AddCommand(newDownloadCmd())

	return cmd
}
