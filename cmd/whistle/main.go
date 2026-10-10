// Command whistle is the CLI of the pure-Go Whistle port. It fetches the pinned model files and transcribes WAV files;
// stream, embed and doctor follow in later phases (see PLAN.md).
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
)

func main() {
	// Ctrl-C cancels a running download; the .part file stays and the next run resumes from it.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	err := NewRootCmd().ExecuteContext(ctx)

	stop()

	if err != nil {
		// The whistle package prefixes its errors with "whistle: " too; print the name once.
		_, _ = fmt.Fprintln(os.Stderr, "whistle:", strings.TrimPrefix(err.Error(), "whistle: "))

		os.Exit(1)
	}
}
