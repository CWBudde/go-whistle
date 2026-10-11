package main

import (
	"errors"
	"fmt"

	"github.com/spf13/cobra"
)

// Exit codes of the whistle command.
const (
	exitOK      = 0 // the command did its work
	exitFailure = 1 // it failed: a missing model, an unreadable file, a failed doctor check
	exitUsage   = 2 // the command line is wrong: an unknown command or flag, a missing argument
)

// usageError marks an error in the command line, which exits with exitUsage.
type usageError struct{ err error }

func (e usageError) Error() string { return e.err.Error() }

func (e usageError) Unwrap() error { return e.err }

// usageArgs makes the argument check v report a usageError.
func usageArgs(v cobra.PositionalArgs) cobra.PositionalArgs {
	return func(cmd *cobra.Command, args []string) error {
		err := v(cmd, args)
		if err != nil {
			return usageError{err}
		}

		return nil
	}
}

// exitCode is the process exit code for the error a command returned.
func exitCode(err error) int {
	if err == nil {
		return exitOK
	}

	if errors.As(err, new(usageError)) {
		return exitUsage
	}

	return exitFailure
}

// NewRootCmd builds the whistle command tree.
func NewRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "whistle",
		Short: "Pure-Go speech-to-text with the Whistle model",
		Long: "Pure-Go speech-to-text with the Whistle model.\n\n" +
			"Exit codes: 0 on success, 1 when a command fails, 2 for a wrong command line.",
		// main prints the error once, and the error names what is wrong.
		SilenceUsage:  true,
		SilenceErrors: true,
		// Without Args cobra reports an unknown command as a plain error; this makes it a usage error.
		Args: func(cmd *cobra.Command, args []string) error {
			if len(args) > 0 {
				return usageError{fmt.Errorf("unknown command %q for %q", args[0], cmd.CommandPath())}
			}

			return nil
		},
		RunE: func(cmd *cobra.Command, _ []string) error { return cmd.Help() },
	}

	cmd.SetFlagErrorFunc(func(_ *cobra.Command, err error) error { return usageError{err} })
	cmd.AddCommand(newDownloadCmd(), newTranscribeCmd(), newStreamCmd(), newEmbedCmd(), newDoctorCmd(), newServeCmd())

	return cmd
}
