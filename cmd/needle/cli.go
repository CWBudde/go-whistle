package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cwbudde/go-whistle/internal/toolcall"
	"github.com/cwbudde/go-whistle/needle"
)

const (
	// defaultModel is where `just fetch-needle3` puts the weights.
	defaultModel = "models/needle3/needle3.cact"
	// needleFile is the file needle.Load reads beside tokenizer.model.
	needleFile = "needle3.cact"
	// fullDepth is needle3's layer count, the only --depth that runs
	// until PLAN.md 11.6 (depth.json's full_depth).
	fullDepth = 20
	// defaultPort is --serve's port without --port.
	defaultPort = 8080
)

// model is what the CLI needs of a loaded needle.Model; the tests put one
// without weights in its place.
type model interface {
	NewAgent(tools []byte, o needle.AgentOptions) (agent, error)
	Close() error
}

// agent is what the CLI needs of a needle.Agent.
type agent interface {
	Complete(text string, o needle.CompleteOptions) (*needle.Response, error)
	Reset()
}

// needleModel is a needle.Model as a model.
type needleModel struct{ m *needle.Model }

func (n needleModel) NewAgent(tools []byte, o needle.AgentOptions) (agent, error) {
	a, err := n.m.NewAgent(tools, o)
	if err != nil {
		return nil, err
	}

	return a, nil
}

func (n needleModel) Close() error { return n.m.Close() }

// openNeedle loads the model that --model names, in the engine's numerics.
func openNeedle(path string) (model, error) {
	dir, err := modelDir(path)
	if err != nil {
		return nil, err
	}

	m, err := needle.Load(dir)
	if err != nil {
		return nil, err
	}

	return needleModel{m}, nil
}

// modelDir is the directory needle.Load opens for --model path: that of
// needle3.cact, or path itself when it is a directory.
func modelDir(path string) (string, error) {
	fi, err := os.Stat(path)
	if err == nil && fi.IsDir() {
		return path, nil
	}

	if filepath.Base(path) != needleFile {
		return "", fmt.Errorf("want %s or its directory, which holds tokenizer.model too", needleFile)
	}

	return filepath.Dir(path), nil
}

// cli is the needle command: where it writes, how it opens the model and
// sets the thread count, and the signals that end --serve (nil: none, only
// the context).
type cli struct {
	stdout, stderr io.Writer
	open           func(path string) (model, error)
	threads        func(n int)
	signals        func(ctx context.Context) (context.Context, context.CancelFunc)
}

// value is a flag's text as given, checked after parsing so that every
// error has the CLI's own message.
type value struct {
	text string
	set  bool
}

func (v *value) String() string { return v.text }

func (v *value) Set(s string) error {
	v.text, v.set = s, true

	return nil
}

// boolValue is a flag without an argument.
type boolValue struct{ value }

func (*boolValue) IsBoolFlag() bool { return true }

func (b *boolValue) Set(s string) error {
	on, err := strconv.ParseBool(s)
	if err != nil {
		return err
	}

	b.set = on

	return nil
}

// list is a flag that may repeat.
type list []string

func (l *list) String() string { return strings.Join(*l, ", ") }

func (l *list) Set(s string) error {
	*l = append(*l, s)

	return nil
}

// options are the command line as given.
type options struct {
	models                                     list
	tools, system, prompt, toolIndex, audio    value
	maxNew, depth, threads, port               value
	serve, refuse                              boolValue
	flags                                      *flag.FlagSet
	order                                      []string // the flags in the help's order
	maxTokens, threadCount, portNumber, layers int      // the numbers, once checked
}

func newOptions() *options {
	o := &options{
		maxNew: value{text: strconv.Itoa(needle.DefaultMaxNewTokens)},
		port:   value{text: strconv.Itoa(defaultPort)},
	}

	fs := flag.NewFlagSet("needle", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	add := func(v flag.Value, name, usage string) {
		fs.Var(v, name, usage)
		o.order = append(o.order, name)
	}

	add(&o.models, "model", "the weights `file`, needle3.cact with tokenizer.model beside it, or their directory")
	add(&o.tools, "tools", "the tools, a JSON array of schemas, from `file`; none: every request gets the empty call")
	add(&o.system, "system", "the system facts from `file`, as written; a text that names no file is ignored")
	add(&o.prompt, "prompt", "answer `query` and print the envelope as one line of JSON")
	add(&o.serve, "serve", "answer one conversation over HTTP: POST /complete {\"input\": \"…\"}, POST /reset")
	add(&o.port, "port", "the `port` --serve listens on at 127.0.0.1; 0 picks a free one")
	add(&o.maxNew, "max", "the answer's token limit in `tokens`")
	add(&o.refuse, "fail-input-overflow", "refuse a turn that does not fit the context window instead of cutting it")
	add(&o.threads, "threads", "the `count` of threads, GOMAXPROCS (default all cores)")
	add(&o.depth, "depth", "the `layers` to run: only the full model, 20, until PLAN.md 11.6")
	add(&o.toolIndex, "tool-index", "the tool index `file`; needle3.cact has no embedding head, so none is written")
	add(&o.audio, "audio", "speech in from a WAV `file`: not ported yet (PLAN.md 11.8)")

	o.flags = fs

	return o
}

// usage writes the help.
func (o *options) usage(w io.Writer) {
	_, _ = fmt.Fprint(w, `Usage: needle [flags] --prompt <query>
       needle [flags] --serve [--port N]

Answers a request with tool calls in the JSON envelope of the Needle 3 engine, in pure Go: --prompt for one
query, --serve for a conversation over HTTP. Errors go to stderr prefixed "needle: " and exit 1.

Flags:
`)

	for _, name := range o.order {
		f := o.flags.Lookup(name)
		arg, usage := flag.UnquoteUsage(f)

		head := "--" + name
		if arg != "" {
			head += " <" + arg + ">"
		}

		def := f.DefValue
		if name == "model" {
			def = defaultModel
		}

		if def != "" {
			def = " (default " + def + ")"
		}

		_, _ = fmt.Fprintf(w, "  %-28s %s%s\n", head, usage, def)
	}
}

// count parses v as an integer of at least lowest.
func count(v value, lowest int) (int, bool) {
	n, err := strconv.Atoi(strings.TrimSpace(v.text))

	return n, err == nil && n >= lowest
}

// check checks the parsed command line and returns the error to report.
func (o *options) check() error {
	if o.flags.NArg() > 0 {
		return fmt.Errorf("unexpected argument %q", o.flags.Arg(0))
	}

	if len(o.models) > 1 {
		return errors.New("a second --model, Whistle's audio in for the text model, is not ported yet (PLAN.md 11.8)")
	}

	if o.audio.set {
		return errors.New("--audio, Whistle's transcript as the request, is not ported yet (PLAN.md 11.8)")
	}

	err := o.checkNumbers()
	if err != nil {
		return err
	}

	switch {
	case o.prompt.set && o.serve.set:
		return errors.New("--prompt and --serve exclude each other")
	case !o.prompt.set && !o.serve.set:
		return errors.New("give --prompt <query> or --serve (see needle --help)")
	}

	return nil
}

// checkNumbers checks --depth, --max, --threads and, with --serve, --port.
func (o *options) checkNumbers() error {
	var ok bool

	if o.depth.set {
		// The engine's refusal of 0 and 1, byte for byte (depth.json).
		o.layers, ok = count(o.depth, 2)

		switch {
		case !ok:
			return fmt.Errorf("--depth takes a layer count of 2 or more, not %q", o.depth.text)
		case o.layers > fullDepth:
			// The engine fails to load past the model's layers (depth.json).
			return fmt.Errorf("cannot load %s: --depth %d is past the model's %d layers", o.model(), o.layers, fullDepth)
		case o.layers < fullDepth:
			return fmt.Errorf("--depth %d: only the full model, %d layers, runs until the depth ladder lands (PLAN.md 11.6)",
				o.layers, fullDepth)
		}
	}

	o.maxTokens, ok = count(o.maxNew, 1)
	if !ok {
		return fmt.Errorf("--max takes a token count of 1 or more, not %q", o.maxNew.text)
	}

	if o.threads.set {
		o.threadCount, ok = count(o.threads, 1)
		if !ok {
			return fmt.Errorf("--threads takes a thread count of 1 or more, not %q", o.threads.text)
		}
	}

	// --port without --serve is ignored (F-066), so only --serve checks it.
	if o.serve.set {
		o.portNumber, ok = count(o.port, 0)
		if !ok || o.portNumber > 65535 {
			return fmt.Errorf("--port takes a port number of 0 to 65535, not %q", o.port.text)
		}
	}

	return nil
}

// model is the path --model names.
func (o *options) model() string {
	if len(o.models) == 0 {
		return defaultModel
	}

	return o.models[0]
}

// run runs the command line args and returns the exit code.
func (c *cli) run(ctx context.Context, args []string) int {
	o := newOptions()

	err := o.flags.Parse(args)
	if errors.Is(err, flag.ErrHelp) {
		o.usage(c.stdout)

		return 0
	}

	if err == nil {
		err = o.check()
	}

	if err != nil {
		return c.fail(err)
	}

	var tools []byte

	if o.tools.set {
		tools, err = os.ReadFile(o.tools.text)
		if err != nil {
			return c.fail(fmt.Errorf("cannot read --tools: %w", err))
		}
	}

	// The engine takes a --system that names no file for no system facts
	// (docs/FINDINGS.md F-047); the CLI says so.
	var system string

	if o.system.set {
		data, err := os.ReadFile(o.system.text)
		if err != nil {
			c.warn(fmt.Sprintf("ignoring --system %q: not a readable file", o.system.text))
		}

		system = string(data)
	}

	if o.threads.set {
		c.threads(o.threadCount)
	}

	m, err := c.open(o.model())
	if err != nil {
		return c.fail(fmt.Errorf("cannot load %s: %s", o.model(), strings.TrimPrefix(err.Error(), "needle: ")))
	}

	defer func() { _ = m.Close() }()

	a, err := m.NewAgent(tools, needle.AgentOptions{
		System: system, NoAutoDate: true, NoYearCheck: true, FailInputOverflow: o.refuse.set,
	})
	if err != nil {
		return c.fail(err)
	}

	c.warnTriggers(tools)

	if o.serve.set {
		return c.serve(ctx, a, o.maxTokens, o.portNumber)
	}

	return c.prompt(a, o.prompt.text, o.maxTokens)
}

// prompt answers query and prints the envelope.
func (c *cli) prompt(a agent, query string, maxNew int) int {
	r, err := complete(a, query, maxNew)
	if err != nil {
		return c.fail(err)
	}

	js, err := encode(r)
	if err != nil {
		return c.fail(err)
	}

	_, err = c.stdout.Write(js)
	if err != nil {
		return c.fail(err)
	}

	return 0
}

// complete answers text as the next turn; a turn the context refuses is
// the engine's refusal envelope, not an error.
func complete(a agent, text string, maxNew int) (*needle.Response, error) {
	r, err := a.Complete(text, needle.CompleteOptions{MaxNewTokens: maxNew})
	if oe, ok := errors.AsType[*needle.OverflowError](err); ok {
		return refusal(oe), nil
	}

	return r, err
}

// refusal is the engine's envelope for a turn that --fail-input-overflow
// refuses, as turns.json and history.json record it: no calls, no
// reasoning, no validation and a confidence of 0.
func refusal(oe *needle.OverflowError) *needle.Response {
	msg := fmt.Sprintf("input exceeds the context window (%d tokens, %d available)", oe.Tokens, oe.Available)

	return &needle.Response{Type: "call", Error: &msg, ErrorCode: new("context_overflow"), Reason: new("runtime_failure")}
}

// encode is the envelope as the CLI writes it: one line of JSON, with '<',
// '>' and '&' as themselves (docs/FINDINGS.md F-066).
func encode(r *needle.Response) ([]byte, error) {
	js, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}

	return append(unescapeHTML(js), '\n'), nil
}

// htmlEscapes are the escapes encoding/json writes for '<', '>' and '&'.
var htmlEscapes = map[string]byte{"\\u003c": '<', "\\u003e": '>', "\\u0026": '&'}

// unescapeHTML writes encoding/json's htmlEscapes as the characters. In
// its output a backslash only starts an escape, so the escapes pair up
// from the left and the second backslash of \\ never starts one.
func unescapeHTML(js []byte) []byte {
	out := make([]byte, 0, len(js))

	for i := 0; i < len(js); i++ {
		if js[i] != '\\' || i+1 == len(js) {
			out = append(out, js[i])

			continue
		}

		if i+6 <= len(js) {
			if c, ok := htmlEscapes[string(js[i:i+6])]; ok {
				out = append(out, c)
				i += 5

				continue
			}
		}

		out = append(out, js[i], js[i+1])
		i++
	}

	return out
}

// warnTriggers warns of the tools' triggers that do not compile, as the
// engine does when it skips them (triggers.json; docs/FINDINGS.md F-054).
func (c *cli) warnTriggers(tools []byte) {
	if tools == nil {
		return
	}

	n, err := toolcall.Normalise(tools)
	if err != nil {
		return
	}

	for _, t := range n.Ignored {
		c.warn("ignoring invalid trigger for " + t)
	}
}

func (c *cli) warn(msg string) { _, _ = fmt.Fprintln(c.stderr, "needle: "+msg) }

// fail reports err on stderr, the package's "needle: " prefix once, and
// returns the exit code of an error.
func (c *cli) fail(err error) int {
	c.warn(strings.TrimPrefix(err.Error(), "needle: "))

	return 1
}
