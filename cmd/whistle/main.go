// Command whistle is the CLI of the pure-Go Whistle port. For now it only fetches the pinned model files; transcribe,
// stream, embed and doctor follow in later phases (see PLAN.md).
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// Ctrl-C cancels a running download; the .part file stays and the next run resumes from it.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	err := NewRootCmd().ExecuteContext(ctx)

	stop()

	if err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "whistle:", err)

		os.Exit(1)
	}
}
