// Command needle is the CLI of the pure-Go Needle 3 port, a drop-in for the flags of the engine's needle binary: it
// answers a request with tool calls in the engine's JSON envelope.
//
//	needle --model models/needle3/needle3.cact --tools tools.json --system system.txt --prompt "set a timer for 10 minutes"
//	needle --model models/needle3/needle3.cact --tools tools.json --serve --port 8080
//
// --prompt prints the envelope of one query on stdout as one line of JSON. --serve keeps one conversation and answers
// its turns over HTTP on 127.0.0.1: POST /complete with {"input": "…"} answers the next turn with the envelope (after
// a call, a JSON input is that call's result), POST /reset starts over; requests run one at a time. --tools and
// --system name files: the tools as a JSON array, the system facts as written. --max is the answer's token limit
// (512), --fail-input-overflow refuses a turn that does not fit the context instead of cutting it, --threads sets
// GOMAXPROCS. --depth runs only the full model, 20 layers, until the depth ladder lands (PLAN.md 11.6); a second
// --model and --audio wait for Whistle's audio in (PLAN.md 11.8). --tool-index is accepted and writes nothing, as the
// engine does on needle3.cact, which carries no embedding head.
//
// It renders the system facts as given and writes the engine's validation: no date fact and no year check, which
// the Python client adds. It exits 0 when it answered, a refused or truncated turn included, and 1 on any error, with
// the message on stderr prefixed "needle: ". docs/FINDINGS.md F-066 records the port's choices.
package main

import (
	"context"
	"os"
	"os/signal"
	"runtime"
	"syscall"
)

func main() {
	// SIGINT or SIGTERM ends --serve: the running request finishes, the ones after it get 503, the model closes.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)

	c := &cli{
		stdout: os.Stdout, stderr: os.Stderr, open: openNeedle,
		threads: func(n int) { runtime.GOMAXPROCS(n) },
	}

	code := c.run(ctx, os.Args[1:])

	stop()
	os.Exit(code)
}
