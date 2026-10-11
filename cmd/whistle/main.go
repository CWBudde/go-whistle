// Command whistle is the CLI of the pure-Go Whistle port: download fetches the pinned model files, transcribe turns
// WAV files of up to 30 s into text, stream transcribes audio of any length live from a WAV file or raw PCM on stdin,
// embed writes the encoder output, doctor checks the install, and serve offers transcription and streaming over
// HTTP and WebSocket.
//
// It exits with 0 on success, 1 when a command fails and 2 for a wrong command line.
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
	// Ctrl-C cancels a running download, whose .part file stays for the next run to resume from, and ends a stream,
	// which then commits its pending words.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	err := NewRootCmd().ExecuteContext(ctx)

	stop()

	if err != nil {
		// The whistle package prefixes its errors with "whistle: " too; print the name once.
		_, _ = fmt.Fprintln(os.Stderr, "whistle:", strings.TrimPrefix(err.Error(), "whistle: "))

		os.Exit(exitCode(err))
	}
}
