package needle

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/cwbudde/go-whistle/internal/text"
	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// DefaultMaxNewTokens is the answer's token limit when an options struct
// leaves it at 0: upstream's max_new_tokens and the engine's --max.
const DefaultMaxNewTokens = 512

// AgentOptions configure NewAgent; the zero value is the engine's
// defaults.
type AgentOptions struct {
	// System are the system facts, rendered before the tools.
	System string
	// NoAutoDate leaves the system facts as given. By default the date
	// fact "date: YYYY-MM-DD Day HH:MM" of Now goes in front of them,
	// unless they hold a date already or are a JSON object, as upstream's
	// Python client does (auto_date=True). The engine itself adds no date
	// (docs/FINDINGS.md F-047), so NoAutoDate renders what the engine's
	// CLI renders.
	NoAutoDate bool
	// Now is the clock of the date fact; nil is time.Now.
	Now func() time.Time
	// Stateless resets the conversation before every Complete and Run, as
	// upstream's stateless=True does.
	Stateless bool
	// FailInputOverflow refuses a turn that does not fit the context
	// window with an *OverflowError, as the engine's
	// --fail-input-overflow does. Without it the turn is cut to fit and
	// the conversation before it dropped (docs/FINDINGS.md F-063).
	FailInputOverflow bool
	// NoYearCheck leaves validation.ungrounded as the engine writes it,
	// without the Python client's year check (_annotate_ungrounded,
	// docs/FINDINGS.md F-064), which the engine's CLI and library do not
	// make. Run's strict check then reads the engine's fields alone;
	// Extract (also Agent.Extract, which keeps this option) reads them plus
	// its own temporal check (_temporal_grounding), which Lenient turns off.
	NoYearCheck bool
}

// CompleteOptions configure Complete.
type CompleteOptions struct {
	// MaxNewTokens is the answer's token limit, the forced "<think>\n"
	// included; 0 is DefaultMaxNewTokens.
	MaxNewTokens int
}

// Agent is one conversation over a tool set: what upstream's needle.Needle
// holds. Each Complete or Run adds its turns to the conversation until
// Reset. An Agent is not safe for concurrent use; the agents of one Model
// are independent of each other.
type Agent struct {
	m         *Model               // nil for a test's agent
	n         *toolcall.Normalised // nil without tools
	c         completer            // nil without tools
	system    string               // the system facts as rendered, the date fact included
	schemas   map[string]any       // the tools as given, by name (_tool_schemas)
	years     map[int]bool         // the years the turns since Reset wrote (_seen_years)
	stateless bool
	noYears   bool // AgentOptions.NoYearCheck
}

// completer answers one turn of a conversation with the engine's envelope:
// the model (session) or a test's script.
type completer interface {
	complete(input string, maxNew int) (*toolcall.Envelope, error)
	reset()
}

// NewAgent starts a conversation over tools, a JSON array of tool schemas
// in upstream's form ({"name", "description", "parameters"}, or OpenAI's
// {"type": "function", "function": {…}}); "triggers" route a request to a
// tool. An empty list (or nil) answers every request with the engine's
// empty call, without decoding.
func (m *Model) NewAgent(tools []byte, o AgentOptions) (*Agent, error) {
	if m.closed.Load() {
		return nil, ErrClosed
	}

	a, err := newAgent(tools, o)
	if err != nil {
		return nil, err
	}

	a.m = m

	if a.n != nil {
		a.c, err = m.newSession(a.n, a.system, o.FailInputOverflow)
		if err != nil {
			return nil, err
		}
	}

	return a, nil
}

// newAgent is NewAgent without the model: the tools, the system facts and
// the state of the Python client's checks.
func newAgent(tools []byte, o AgentOptions) (*Agent, error) {
	if len(bytes.TrimSpace(tools)) == 0 {
		tools = []byte("[]")
	}

	n, err := toolcall.Normalise(tools)
	if err != nil {
		return nil, wrap(err)
	}

	system := o.System
	if !o.NoAutoDate {
		system = withDate(system, o.Now)
	}

	a := &Agent{
		system: system, schemas: map[string]any{}, years: map[int]bool{}, stateless: o.Stateless,
		noYears: o.NoYearCheck,
	}

	list, _ := decodeJSON(tools).([]any)
	for _, entry := range list {
		tool := unwrapTool(entry)
		if name, ok := asObject(tool)["name"].(string); ok {
			a.schemas[name] = tool
		}
	}

	if len(list) > 0 {
		a.n = n
	}

	return a, nil
}

// withDate is upstream's _with_date_fact: the date fact of now (nil is
// time.Now) in front of system, unless it holds a date or is a JSON object.
func withDate(system string, now func() time.Time) string {
	if now == nil {
		now = time.Now
	}

	return toolcall.WithDateFact(system, now())
}

// System are the system facts the agent renders: AgentOptions.System with
// the date fact unless NoAutoDate.
func (a *Agent) System() string { return a.system }

// Complete answers text as the next turn of the conversation and returns
// the engine's envelope with the Python client's year check in its
// validation, as needle.Needle.complete does (annotate). A turn after an answer that called a tool is the
// tool's result when text is JSON, a user turn otherwise
// (docs/FINDINGS.md F-051). A refused turn (ErrContextOverflow) leaves the
// conversation as it was; any other error starts it anew, as Reset does.
func (a *Agent) Complete(text string, o CompleteOptions) (*Response, error) {
	if a.stateless {
		a.Reset()
	}

	return a.complete(text, maxNewTokens(o.MaxNewTokens), true)
}

// Reset starts a new conversation over the same tools and system facts.
func (a *Agent) Reset() {
	clear(a.years)

	if a.c != nil {
		a.c.reset()
	}
}

// complete is upstream's _complete: one turn, its years noted for the
// checks and, when ground, the answer annotated with the dates whose year
// they do not license (annotate).
func (a *Agent) complete(input string, maxNew int, ground bool) (*Response, error) {
	if a.m != nil && a.m.closed.Load() {
		return nil, ErrClosed
	}

	for y := range sourceYears(input) {
		a.years[y] = true
	}

	if a.n == nil {
		return newResponse(toolcall.NoTools(), true), nil
	}

	e, err := a.c.complete(input, maxNew)
	if err != nil {
		// Past a refusal the conversation is left half fed: start anew.
		if !errors.Is(err, ErrContextOverflow) {
			a.Reset()
		}

		return nil, err
	}

	r := newResponse(e, false)
	if ground && !a.noYears {
		a.annotate(r, input)
	}

	return r, nil
}

func maxNewTokens(n int) int {
	if n <= 0 {
		return DefaultMaxNewTokens
	}

	return n
}

// session is the model's side of a conversation: the turns as the engine
// renders them and the state that holds them.
type session struct {
	m        *Model
	n        *toolcall.Normalised
	system   string
	prefix   []int // the static prefix
	conv     *toolcall.Conversation
	s        *text.State
	refuse   bool
	closings []int // <|im_end|> and </tool_call>, which close the answer before a turn
	dropped  int   // turns answered alone after the conversation outgrew the context
}

// newSession starts the model's side of a conversation over the tools n
// and the system facts system.
func (m *Model) newSession(n *toolcall.Normalised, system string, refuse bool) (*session, error) {
	prefix := toolcall.StaticPrefix(m.p, system, n.JSON)
	if len(prefix) >= m.context() {
		return nil, fmt.Errorf("%w: static prefix is %d tokens, model context is %d tokens", ErrContextBudget,
			len(prefix), m.context())
	}

	s := &session{
		m: m, n: n, system: system, prefix: prefix, refuse: refuse,
		conv: toolcall.NewConversation(m.p, system, n.JSON, m.context()), s: m.tm.NewState(),
	}

	for _, piece := range []string{"<|im_end|>", "</tool_call>"} {
		id, ok := m.p.PieceID(piece)
		if !ok {
			return nil, fmt.Errorf("needle: the tokenizer has no %s", piece)
		}

		s.closings = append(s.closings, id)
	}

	return s, nil
}

// complete renders input as the next turn, decodes the answer under the
// grammar of input's tools and builds the envelope, as the engine does
// (docs/FINDINGS.md F-051 to F-063). A turn that the engine cuts to fit
// restarts the state from the static prefix (F-063). A conversation that
// outgrows the context, which the engine's answers do not pin, is dropped
// and the turn answered alone after the static prefix, as a cut turn is
// (docs/FINDINGS.md F-064).
func (s *session) complete(input string, maxNew int) (*toolcall.Envelope, error) {
	ids, err := s.conv.Turn(input, maxNew, s.refuse)
	if err != nil {
		if oe, ok := errors.AsType[*toolcall.OverflowError](err); ok {
			return nil, &OverflowError{Tokens: oe.Tokens, Available: oe.Available}
		}

		return nil, wrap(err)
	}

	if s.conv.Restarted() {
		s.s = s.m.tm.NewState()
	}

	g := s.n.GrammarFor(input)

	var a *toolcall.Decoded

	switch {
	case s.s.Len() == 0:
		a, err = s.m.d.Decode(s.s, slices.Concat(s.prefix, ids), maxNew, g)
	case s.s.Len()+len(ids) > s.m.context():
		a, err = s.m.d.Decode(s.s, s.alone(ids), maxNew, g)
	default:
		a, err = s.m.d.Decode(s.s, ids, maxNew, g)
		if errors.Is(err, text.ErrTooLong) {
			a, err = s.m.d.Decode(s.s, s.alone(ids), maxNew, g)
		}
	}

	if err != nil {
		return nil, wrap(err)
	}

	conf, err := s.m.d.Confidence(s.s)
	if err != nil {
		return nil, wrap(err)
	}

	said := s.conv.Inputs()

	e, err := s.n.Envelope(a, input, s.system, said[:len(said)-1], conf)
	if err != nil {
		return nil, wrap(err)
	}

	s.conv.Answered(a, len(e.FunctionCalls))

	return e, nil
}

// alone drops the conversation before the turn ids for a new state and
// returns the turn to feed it: the static prefix and the turn without the
// closing of the answer before it, as for a cut turn.
func (s *session) alone(ids []int) []int {
	for len(ids) > 0 && slices.Contains(s.closings, ids[0]) {
		ids = ids[1:]
	}

	s.s = s.m.tm.NewState()
	s.dropped++

	return slices.Concat(s.prefix, ids)
}

func (s *session) reset() {
	s.conv.Reset()
	s.s = s.m.tm.NewState()
}

// Response is the engine's envelope for one turn (docs/FINDINGS.md F-054,
// F-062) with the Python client's year check in Validation (F-064), and for
// Run the results of the tools it called.
type Response struct {
	// Type is "call" (an empty call is a refusal) or "respond".
	Type    string
	Success bool
	// Error, ErrorCode and Reason are nil but on a failed answer, which
	// the engine writes as "tool call truncated: token budget exhausted",
	// "truncated" and "runtime_failure".
	Error, ErrorCode, Reason *string
	// FunctionCalls are the calls to make, under the caller's names;
	// SuppressedCalls the calls the engine withheld (a confidence below
	// 0.1, a contradiction or a guess).
	FunctionCalls, SuppressedCalls []Call
	// Reasoning is the model's reasoning, nil for a truncated answer.
	Reasoning *string
	// Confidence is the min of the confidence head and the answer's
	// decode probability, in [0, 1].
	Confidence float64
	// Validation is set when the answer has calls, suppressed ones
	// included.
	Validation *Validation
	// Results are the values the tools of Run returned, in call order
	// across its steps (an error as {"error": message}); nil for Complete.
	Results []any

	noTools   bool        // the engine's answer without tools, which has no error_code and reason
	engine    int         // how many of Validation.Ungrounded are the engine's, the rest annotate's
	annotated []grounding // the dates annotate added, under the caller's names
}

// Call is one function call.
type Call struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Validation is the engine's check of the calls against the request.
type Validation struct {
	// Ungrounded are the fields, "tool.field" under the model's
	// snake_case names, whose value the conversation's turns do not
	// contain.
	Ungrounded []string `json:"ungrounded"`
	// Negation reports a negated request.
	Negation bool `json:"negation"`
}

func newResponse(e *toolcall.Envelope, noTools bool) *Response {
	r := &Response{
		Type: e.Type, Success: e.Success, Error: e.Error, ErrorCode: e.ErrorCode, Reason: e.Reason,
		FunctionCalls: calls(e.FunctionCalls), SuppressedCalls: calls(e.SuppressedCalls), Reasoning: e.Reasoning,
		Confidence: e.Confidence, noTools: noTools,
	}

	if e.Validation != nil {
		r.Validation = &Validation{Ungrounded: e.Validation.Ungrounded, Negation: e.Validation.Negation}
		r.engine = len(e.Validation.Ungrounded)
	}

	return r
}

func calls(cs []toolcall.Call) []Call {
	if cs == nil {
		return nil
	}

	out := make([]Call, len(cs))
	for i, c := range cs {
		out[i] = Call(c)
	}

	return out
}

// MarshalJSON writes the envelope as the engine does, byte for byte: its
// field order and the confidence to four decimals, then "results" for Run.
func (r Response) MarshalJSON() ([]byte, error) {
	e := &toolcall.Envelope{}
	if r.noTools {
		e = toolcall.NoTools()
	}

	e.Type, e.Success, e.Error, e.ErrorCode, e.Reason = r.Type, r.Success, r.Error, r.ErrorCode, r.Reason
	e.Reasoning, e.Confidence = r.Reasoning, r.Confidence

	for _, c := range r.FunctionCalls {
		e.FunctionCalls = append(e.FunctionCalls, toolcall.Call(c))
	}

	for _, c := range r.SuppressedCalls {
		e.SuppressedCalls = append(e.SuppressedCalls, toolcall.Call(c))
	}

	if r.Validation != nil {
		e.Validation = &toolcall.Validation{Ungrounded: r.Validation.Ungrounded, Negation: r.Validation.Negation}
	}

	js, err := e.MarshalJSON()
	if err != nil {
		return nil, wrap(err)
	}

	if r.Results == nil {
		return js, nil
	}

	js = append(js[:len(js)-1], `,"results":[`...)

	for i, v := range r.Results {
		if i > 0 {
			js = append(js, ',')
		}

		js = append(js, resultJSON(v)...)
	}

	return append(js, "]}"...), nil
}
