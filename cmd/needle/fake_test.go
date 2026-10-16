package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/needle"
)

// fakeModel is a model without weights: it records how the CLI opened it
// and made its agent, and its agent answers from a script.
type fakeModel struct {
	path   string
	tools  []byte
	opts   needle.AgentOptions
	agent  *fakeAgent
	closed bool
	err    error // NewAgent's error
}

func (m *fakeModel) NewAgent(tools []byte, o needle.AgentOptions) (agent, error) {
	if m.err != nil {
		return nil, m.err
	}

	m.tools, m.opts = tools, o

	return m.agent, nil
}

func (m *fakeModel) Close() error {
	m.closed = true

	return nil
}

// fakeAgent answers each turn with reply and records what it was asked.
// It notes when two calls overlap, which the server must not let happen.
type fakeAgent struct {
	mu      sync.Mutex
	inputs  []string
	maxNew  []int
	resets  int
	reply   func(input string) (*needle.Response, error)
	delay   time.Duration
	active  atomic.Int32
	overlap atomic.Bool
}

func (a *fakeAgent) Complete(text string, o needle.CompleteOptions) (*needle.Response, error) {
	defer a.enter()()

	a.mu.Lock()
	a.inputs, a.maxNew = append(a.inputs, text), append(a.maxNew, o.MaxNewTokens)
	a.mu.Unlock()

	if a.reply == nil {
		return timerResponse(), nil
	}

	return a.reply(text)
}

func (a *fakeAgent) Reset() {
	defer a.enter()()

	a.mu.Lock()
	a.resets++
	a.mu.Unlock()
}

// enter notes a call into the agent until the returned func runs.
func (a *fakeAgent) enter() func() {
	if a.active.Add(1) > 1 {
		a.overlap.Store(true)
	}

	time.Sleep(a.delay)

	return func() { a.active.Add(-1) }
}

func (a *fakeAgent) asked() ([]string, int) {
	a.mu.Lock()
	defer a.mu.Unlock()

	return append([]string(nil), a.inputs...), a.resets
}

// timerResponse is an envelope with a call, a reasoning with "->" and a
// validation, as the engine answers "set a timer for 10 minutes for the
// pasta" (responses.json, timer_pasta).
func timerResponse() *needle.Response {
	reasoning := "'10 minutes' -> minutes 10; 'pasta' -> label 'pasta'."

	return &needle.Response{
		Type: "call", Success: true,
		FunctionCalls:   []needle.Call{{Name: "set_timer", Arguments: json.RawMessage(`{"minutes":10,"label":"Pasta"}`)}},
		SuppressedCalls: []needle.Call{},
		Reasoning:       &reasoning, Confidence: 0.9324,
		Validation: &needle.Validation{Ungrounded: []string{}},
	}
}

// timerJSON is timerResponse as the CLI prints it.
const timerJSON = `{"type":"call","success":true,"error":null,"error_code":null,"reason":null,` +
	`"function_calls":[{"name":"set_timer","arguments":{"minutes":10,"label":"Pasta"}}],"suppressed_calls":[],` +
	`"reasoning":"'10 minutes' -> minutes 10; 'pasta' -> label 'pasta'.","confidence":0.9324,` +
	`"validation":{"ungrounded":[],"negation":false}}` + "\n"

// harness runs the CLI on a fake model and keeps what it wrote.
type harness struct {
	model          *fakeModel
	opened         int
	threads        []int
	stdout, stderr *bytes.Buffer
}

func newHarness() *harness {
	return &harness{
		model:  &fakeModel{agent: &fakeAgent{}},
		stdout: &bytes.Buffer{}, stderr: &bytes.Buffer{},
	}
}

func (h *harness) cli() *cli {
	return &cli{
		stdout: h.stdout, stderr: h.stderr,
		open: func(path string) (model, error) {
			h.opened++
			h.model.path = path

			return h.model, nil
		},
		threads: func(n int) { h.threads = append(h.threads, n) },
	}
}

func (h *harness) run(t *testing.T, args ...string) int {
	t.Helper()

	return h.cli().run(context.Background(), args)
}

// writeFile writes data to a new file in a test's temporary directory and
// returns its path.
func writeFile(t *testing.T, name, data string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), name)

	err := os.WriteFile(path, []byte(data), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	return path
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	err = json.Unmarshal(data, v)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

// parsed is a JSON text decoded into Go values, for comparisons that do
// not depend on the bytes.
func parsed(t *testing.T, js []byte) any {
	t.Helper()

	var v any

	err := json.Unmarshal(js, &v)
	if err != nil {
		t.Fatalf("%s: %v", js, err)
	}

	return v
}

var errBoom = errors.New("needle: boom")
