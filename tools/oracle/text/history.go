//go:build oracle

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/cwbudde/go-whistle/internal/spm"
	"github.com/cwbudde/go-whistle/internal/toolcall"
)

// historyFixture is testdata/text/engine/history.json: conversations over `needle --serve` that read what the engine
// keeps of a conversation (docs/FINDINGS.md F-063).
type historyFixture struct {
	header

	System  string `json:"system"`
	Context int    `json:"context"`
	// Wire are probes whose inputs reach the server with chosen characters as JSON \u escapes.
	Wire []historyWire `json:"wire"`
	// Conversations are conversations at one --max.
	Conversations []historyConversation `json:"conversations"`
	// Scans are conversations replayed at many --max values, where the engine starts to drop the history.
	Scans []historyScan `json:"scans"`
}

// historyWire is one conversation whose request bodies spell the runes of Escaped as \u escapes and every other rune
// as itself, so that the engine's reading of the escapes shows in its counts.
type historyWire struct {
	ID      string     `json:"id"`
	Covers  string     `json:"covers"`
	Escaped string     `json:"escaped"`
	Max     int        `json:"max"`
	Steps   []turnStep `json:"steps"`
}

// historyConversation is one conversation on a fresh `needle --serve` session over a corpus tool set, or, with
// Library, through libneedle3's needle_init and needle_complete, whose token limit is per turn.
type historyConversation struct {
	ID      string        `json:"id"`
	Group   string        `json:"group"`
	Covers  string        `json:"covers"`
	Toolset string        `json:"toolset"`
	System  string        `json:"system,omitempty"`
	Library bool          `json:"library,omitempty"`
	Max     int           `json:"max"`
	Refuse  bool          `json:"fail_input_overflow"`
	Steps   []historyStep `json:"steps"`
}

// historyStep is one turn of a conversation: its input, its token limit through the library (0: the
// conversation's), the SHA-256 of the text, the answer and, through the library, needle_complete's result.
type historyStep struct {
	Input     turnInput       `json:"input"`
	Max       int             `json:"max,omitempty"`
	SHA256    string          `json:"sha256,omitempty"`
	Answer    json.RawMessage `json:"answer,omitempty"`
	Generated *int32          `json:"generated,omitempty"`
}

// historyScan is one conversation replayed on fresh sessions at the --max values Runs lists, found by bisection
// between Lo and Hi: the answers change only where --max makes the engine drop part of the history.
type historyScan struct {
	ID      string       `json:"id"`
	Group   string       `json:"group"`
	Covers  string       `json:"covers"`
	Toolset string       `json:"toolset"`
	Refuse  bool         `json:"fail_input_overflow"`
	Inputs  []turnInput  `json:"inputs"`
	SHA256  []string     `json:"sha256"`
	Lo      int          `json:"lo"`
	Hi      int          `json:"hi"`
	Runs    []historyRun `json:"runs"`
}

// historyRun is one replay of a scan at --max Max: the answer to each input.
type historyRun struct {
	Max     int               `json:"max"`
	Answers []json.RawMessage `json:"answers"`
}

// text is t as plain input.
func text(t string) turnInput { return turnInput{Tail: t} }

// The wire probes: a tool result after a call, or a user turn, whose strings hold <, & or > or another character,
// sent with Go's escaping (json.Marshal writes <, & and > as \u escapes), without it, and with other characters
// escaped. The probe is about a thousand tokens at --max 7000, so the engine refuses it with its length.
func historyWires() []historyWire {
	tool := func(esc string) turnInput {
		return turnInput{Head: "[", Unit: `{"v": "a` + esc + `b"}, `, Repeat: 150, Tail: `{"v": "a"}]`}
	}
	user := func(esc string) turnInput {
		return turnInput{Unit: "note a" + esc + "b and ", Repeat: 200, Tail: "that is all"}
	}

	var ws []historyWire

	add := func(id, covers, escaped string, steps ...turnInput) {
		w := historyWire{ID: id, Covers: covers, Escaped: escaped, Max: 7000}
		for _, s := range steps {
			w.Steps = append(w.Steps, turnStep{Input: s})
		}

		ws = append(ws, w)
	}

	for _, c := range []struct{ id, esc string }{{"lt", "<"}, {"amp", "&"}, {"gt", ">"}, {"lt_amp_gt", "<&>"}} {
		add("tool_"+c.id+"_go", "a tool result with "+c.esc+", as json.Marshal sends it", "<&>", turnCall, tool(c.esc))
		add("tool_"+c.id+"_raw", "a tool result with "+c.esc+", sent unescaped", "", turnCall, tool(c.esc))
		add("user_"+c.id+"_go", "a user turn with "+c.esc+", as json.Marshal sends it", "<&>", user(c.esc))
		add("user_"+c.id+"_raw", "a user turn with "+c.esc+", sent unescaped", "", user(c.esc))
	}

	for _, c := range []struct{ id, esc, escaped string }{
		{"A", "A", "A"},
		{"u_umlaut", "ü", "ü"},
		{"e_acute", "é", "é"},
		{"slash", "/", "/"},
		{"quote_single", "'", "'"},
		{"euro", "€", "€"},
	} {
		add("tool_"+c.id+"_escaped", "a tool result with "+c.esc+" sent as a \\u escape", c.escaped, turnCall, tool(c.esc))
		add("user_"+c.id+"_escaped", "a user turn with "+c.esc+" sent as a \\u escape", c.escaped, user(c.esc))
	}

	add("user_A_plain", "a user turn with A, unescaped (the control)", "", user("A"))
	add("user_u_umlaut_plain", "a user turn with ü, unescaped (the control)", "", user("ü"))

	return ws
}

// cmdHistory records the history probes: how the engine reads the request's escapes, conversations whose later
// answers depend on earlier turns, and scans over --max where the engine drops the history (docs/FINDINGS.md F-063).
func cmdHistory(ctx context.Context, cfg config, log io.Writer) error {
	o, err := openOracle(cfg, false)
	if err != nil {
		return err
	}
	defer o.close()

	_, budget, err := cactHeads(o.needle3Path())
	if err != nil {
		return err
	}

	p, err := spm.Load(filepath.Join(o.cfg.needle3Dir, "tokenizer.model"))
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	fx := historyFixture{
		header: o.header("needle --model needle3.cact --tools <toolset> --system <system file> --threads 4 " +
			"--max <max> [--fail-input-overflow] --serve; POST /complete per turn, POST /reset"),
		System: o.corpus.System, Context: budget,
	}

	for _, w := range historyWires() {
		rec, err := o.historyWire(ctx, w)
		if err != nil {
			return fmt.Errorf("%s: %w", w.ID, err)
		}

		fx.Wire = append(fx.Wire, rec)
		_, _ = fmt.Fprintf(log, "%-28s %s\n", w.ID, rec.Steps[len(rec.Steps)-1].Answer)
	}

	cs := historyConversations()

	cut, err := cutConversations(p, o.corpus.System, o.corpus.tools(assistantSet), budget)
	if err != nil {
		return err
	}

	for _, c := range append(cs, cut...) {
		rec, err := o.historyConversation(ctx, c)
		if err != nil {
			return fmt.Errorf("%s: %w", c.ID, err)
		}

		fx.Conversations = append(fx.Conversations, rec)

		for i, st := range rec.Steps {
			_, _ = fmt.Fprintf(log, "%-28s %d %.700s\n", c.ID, i, st.Answer)
		}
	}

	for _, sc := range historyScans() {
		rec, err := o.historyScan(ctx, p, budget, sc, log)
		if err != nil {
			return fmt.Errorf("%s: %w", sc.ID, err)
		}

		fx.Scans = append(fx.Scans, rec)
	}

	return writeJSON(cfg.outDir, "history.json", fx)
}

// historyConversation runs one conversation on its own server, or through the library.
func (o *oracle) historyConversation(ctx context.Context, c historyConversation) (historyConversation, error) {
	system := o.corpus.System
	if c.System != "" {
		system = c.System
	}

	if c.Library {
		return o.historyLibrary(c, system)
	}

	args, err := o.baseArgs(c.Toolset, system)
	if err != nil {
		return c, err
	}

	args = append(args, "--max", strconv.Itoa(c.Max))
	if c.Refuse {
		args = append(args, "--fail-input-overflow")
	}

	s, err := startServe(ctx, o.needle(), args)
	if err != nil {
		return c, err
	}
	defer s.close()

	for i := range c.Steps {
		in := c.Steps[i].Input
		if in.Reset {
			err := s.reset(ctx)
			if err != nil {
				return c, err
			}

			continue
		}

		t := in.text()
		sum := sha256.Sum256([]byte(t))

		a, err := s.completeRaw(ctx, t)
		if err != nil {
			return c, fmt.Errorf("turn %d: %w", i+1, err)
		}

		c.Steps[i].SHA256, c.Steps[i].Answer = hex.EncodeToString(sum[:]), compact(a)
	}

	return c, nil
}

// historyLibrary runs one conversation through libneedle3, each turn at its own token limit.
func (o *oracle) historyLibrary(c historyConversation, system string) (historyConversation, error) {
	lib, err := o.historyLib()
	if err != nil {
		return c, err
	}

	_, err = lib.session(system, o.corpus.tools(c.Toolset))
	if err != nil {
		return c, err
	}

	lib.reset()

	for i := range c.Steps {
		in := c.Steps[i].Input
		if in.Reset {
			lib.reset()

			continue
		}

		maxNew := c.Max
		if c.Steps[i].Max > 0 {
			maxNew = c.Steps[i].Max
		}

		t := in.text()
		sum := sha256.Sum256([]byte(t))

		r, err := lib.completeText(t, maxNew)
		if err != nil {
			return c, fmt.Errorf("turn %d: %w", i+1, err)
		}

		a, err := normaliseResponse(r.out)
		if err != nil {
			return c, err
		}

		rc := r.rc
		c.Steps[i].SHA256, c.Steps[i].Answer, c.Steps[i].Generated = hex.EncodeToString(sum[:]), compact(a), &rc
	}

	return c, nil
}

// historyLibOnce opens libneedle3 once for the library conversations.
var (
	historyLibOnce sync.Once
	historyLibVal  *library
	errHistoryLib  error
)

func (o *oracle) historyLib() (*library, error) {
	historyLibOnce.Do(func() { historyLibVal, errHistoryLib = openLibrary(o.cfg.engineDir, o.needle3Path()) })

	return historyLibVal, errHistoryLib
}

// historyWire runs one wire probe on its own server over the assistant tools.
func (o *oracle) historyWire(ctx context.Context, w historyWire) (historyWire, error) {
	args, err := o.baseArgs(assistantSet, o.corpus.System)
	if err != nil {
		return w, err
	}

	s, err := startServe(ctx, o.needle(), append(args, "--max", strconv.Itoa(w.Max), "--fail-input-overflow"))
	if err != nil {
		return w, err
	}
	defer s.close()

	for i := range w.Steps {
		t := w.Steps[i].Input.text()
		sum := sha256.Sum256([]byte(t))

		a, err := s.completeEscaped(ctx, t, w.Escaped)
		if err != nil {
			return w, err
		}

		w.Steps[i].SHA256, w.Steps[i].Answer = hex.EncodeToString(sum[:]), compact(a)
	}

	return w, nil
}

// completeEscaped answers one turn whose request body spells the runes of escaped as \u escapes and every other
// rune as itself (only '"', '\\' and control characters escaped as JSON requires).
func (s *serveSession) completeEscaped(ctx context.Context, input, escaped string) (node, error) {
	var b bytes.Buffer

	b.WriteString(`{"input":"`)

	for _, r := range input {
		switch {
		case strings.ContainsRune(escaped, r):
			fmt.Fprintf(&b, `\`+"u%04x", r)
		case r == '"' || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r < 0x20:
			fmt.Fprintf(&b, `\`+"u%04x", r)
		default:
			b.WriteRune(r)
		}
	}

	b.WriteString(`"}`)

	if !utf8.Valid(b.Bytes()) || !json.Valid(b.Bytes()) {
		return node{}, fmt.Errorf("request body %.40q is not JSON", b.String())
	}

	js, err := postBody(ctx, s.base+"/complete", b.Bytes())
	if err != nil {
		return node{}, err
	}

	return normaliseResponse(js)
}

// completeRaw answers one turn with the input sent unescaped but for what JSON requires, unlike complete, whose
// json.Marshal sends <, & and > as \u escapes the engine reads as their hex digits (docs/FINDINGS.md F-063).
func (s *serveSession) completeRaw(ctx context.Context, input string) (node, error) {
	return s.completeEscaped(ctx, input, "")
}

// steps are inputs as the steps of a conversation.
func steps(inputs ...turnInput) []historyStep {
	out := make([]historyStep, len(inputs))
	for i, in := range inputs {
		out[i].Input = in
	}

	return out
}

// historyConversations are the recorded conversations, by group (docs/FINDINGS.md F-063):
//
//   - follow: two to five turns at --max 512 whose later answer needs an earlier turn, after a call (with and
//     without its tool result), two calls, a respond answer, an empty call and a suppressed call;
//   - truncated: through the library, a first answer cut at a small token limit, then a turn at 512;
//   - refused: a turn --fail-input-overflow refuses (--max 7000, 878 tokens a turn) between or before others;
//   - cut: an oversized turn without the flag where a turn has 150 or 200 tokens, the request placed at measured
//     token offsets from the end, so that the cut lies inside the KV window;
//   - long: conversations past the context of 8192 tokens.
func historyConversations() []historyConversation {
	var cs []historyConversation

	conv := func(id, group, covers, toolset string, maxNew int, refuse bool, inputs ...turnInput) {
		cs = append(cs, historyConversation{
			ID: id, Group: group, Covers: covers, Toolset: toolset, Max: maxNew, Refuse: refuse, Steps: steps(inputs...),
		})
	}

	followConversations(conv)

	cs = append(cs, truncatedConversations()...)

	refusedConversations(conv)

	cs = append(cs, longConversations()...)

	return cs
}

// addConversation records one conversation: id, group, what it covers, the tool set, --max, the flag and the inputs.
type addConversation func(id, group, covers, toolset string, maxNew int, refuse bool, inputs ...turnInput)

// Turns the conversations share.
var (
	timerBread  = text("set a timer for 25 minutes for the bread")
	timerPasta  = text("set a timer for 10 minutes for the pasta")
	sameLength  = text("now set another one of the same length for the pasta")
	started     = text(`{"status": "started"}`)
	okResult    = text(`{"status": "ok"}`)
	messageAnna = text("text Anna that I'm running late")
	twoCalls    = text("text Anna that I'm running late and set a timer for 20 minutes")
	pizza       = text("order a large pepperoni pizza")
	wakeMe      = text("wake me up at 6:45 tomorrow")
)

// followConversations are conversations whose later answer depends on an earlier turn.
func followConversations(add addConversation) {
	const g = "follow"

	f := func(id, covers, toolset string, inputs ...turnInput) {
		add(id, g, covers, toolset, defaultMax, false, inputs...)
	}

	f("timer_same", "a call, its result, a request for the same length", assistantSet, timerBread, started, sameLength)
	f("timer_same_direct", "a call, then a request for the same length without a result", assistantSet,
		timerBread, sameLength)
	f("timer_instead", "a call, its result, a change of the length", assistantSet,
		timerPasta, text(`{"status": "ok", "timer_id": "t1"}`), text("make it 15 minutes instead"))
	f("timer_instead_direct", "a call, then a change of the length", assistantSet, timerPasta,
		text("make it 15 minutes instead"))
	f("timer_cancel", "a call, its result, cancel that", assistantSet, timerPasta, okResult, text("cancel that"))
	f("message_same", "a message, its result, the same message to another", assistantSet,
		messageAnna, text(`{"status": "sent"}`), text("send the same message to Tom"))
	f("weather_tomorrow", "a weather call, its result, and tomorrow?", assistantSet,
		text("what's the weather in Lisbon today?"), text(`{"forecast": "sunny", "high_c": 24}`), text("and tomorrow?"))
	f("weather_paris", "a weather call, then another city", assistantSet,
		text("what's the weather in Berlin today?"), text("and in Paris?"))
	f("event_move", "an event, its result, move it", assistantSet,
		text("add the dentist tomorrow at 3pm"), text(`{"status": "created", "event_id": "e1"}`), text("move it to 4pm"))
	f("alarm_change", "an alarm, its result, another time", assistantSet,
		text("set an alarm for 6:45"), okResult, text("make it 7:15 instead"))
	f("two_calls_same", "two calls, their results, the same timer again", assistantSet,
		twoCalls, text(`[{"status": "sent"}, {"status": "started"}]`), text("set the same timer again for the tea"))
	f("two_calls_message", "two calls, their results, the same message to another", assistantSet,
		twoCalls, text(`[{"status": "sent"}, {"status": "started"}]`), text("send that message to Tom as well"))
	f("lights_same", "lights off, the result, the same elsewhere", homeSet,
		text("turn off the kitchen lights"), okResult, text("do the same for the bedroom"))
	f("lights_same_direct", "lights on, then the same elsewhere", homeSet,
		text("turn on the living room lights"), text("do the same for the kitchen"))
	f("thermostat_warmer", "a temperature, its result, warmer", homeSet,
		text("set the heating to 21 degrees"), okResult, text("a bit warmer, make it 23"))
	f("lock_unlock", "lock, the result, unlock it again", homeSet,
		text("lock the front door"), text(`{"status": "locked"}`), text("unlock it again"))
	f("music_same", "music, the result, the same in another room", homeSet,
		text("play some jazz in the living room"), text(`{"status": "playing"}`), text("play the same in the kitchen"))
	f("reminder_later", "a reminder, its result, another an hour later", catalogueSet,
		text("remind me to call the bank tomorrow at 9:00"), okResult,
		text("and remind me to call the dentist an hour later"))
	f("list_same", "a list item, its result, another to the same list", catalogueSet,
		text("add milk to the shopping list"), okResult, text("add eggs to the same list"))

	followMore(f)
}

// followMore are the follow conversations after a respond answer, an empty and a suppressed call, and the longer
// ones.
func followMore(f func(id, covers, toolset string, inputs ...turnInput)) {
	f("respond_then_timer", "a respond answer, then a request", assistantSet,
		text("what can you do?"), text("ok, then set a timer for 5 minutes"))
	f("hello_then_timer", "a greeting, then a request", assistantSet,
		text("hello"), text("set a timer for 5 minutes for the tea"))
	f("result_thanks_same", "a call, its result, thanks, and the same length again", assistantSet,
		timerPasta, text(`{"status": "ok", "ends_at": "09:40"}`), text("thanks!"),
		text("set one more for the sauce, same length"))
	f("refusal_then_lights", "an empty call, then a request", homeSet, pizza, text("then turn on the kitchen lights"))
	f("refusal_then_timer", "an empty call, then a request that refers to it", assistantSet,
		pizza, text("fine, set a timer for 20 minutes for it instead"))
	f("suppressed_then_change", "a suppressed call, then a change", assistantSet, wakeMe, text("make it 7:00 instead"))
	f("suppressed_then_confirm", "a suppressed call, then a confirmation", assistantSet, wakeMe,
		text("yes, set that alarm"))
	f("five_turns", "two calls with results, then the first call's length", assistantSet,
		timerBread, started, text("text Anna that the bread is in the oven"), text(`{"status": "sent"}`),
		text("set a timer for the pasta with the same length as the bread"))
	f("weather_message", "a weather result, then a message about it", assistantSet,
		text("what's the weather in Lisbon today?"), text(`{"forecast": "rain", "high_c": 12}`),
		text("text Tom what the weather is"))
	f("timer_error_retry", "a failed call, then try again", assistantSet,
		timerPasta, text(`{"error": "timer service unavailable"}`), text("try again"))
	f("message_reply_event", "a message whose result carries a reply, then an event for it", assistantSet,
		messageAnna, text(`{"status": "sent", "reply": "ok, see you at 8pm"}`), text("add an event for that time with Anna"))
	f("home_two_unlock", "two calls, their results, unlock it", homeSet,
		text("turn off the kitchen lights and lock the front door"), text(`[{"status": "ok"}, {"status": "locked"}]`),
		text("now unlock it"))
	f("german_same", "a German timer, its result, another of the same length", assistantSet,
		text("Stell einen Timer auf 15 Minuten für die Eier"), okResult,
		text("und noch einen gleich langen für den Reis"))
	f("chat_then_weather", "a respond answer to chit-chat, then a request", assistantSet,
		text("how are you today?"), text("what's the weather in Paris today?"))
	f("joke_then_timer", "a respond answer to a joke request, then a request", assistantSet,
		text("tell me a joke"), text("set a timer for 3 minutes"))
	f("result_thanks_weather", "a call, its result, thanks, then another request", assistantSet,
		timerPasta, okResult, text("thanks!"), text("what's the weather in Rome today?"))
	f("weather_thanks_timer", "a weather result, thanks, then a request", assistantSet,
		text("what's the weather in Lisbon today?"), text(`{"forecast": "sunny", "high_c": 24}`),
		text("great, thanks"), text("and set a timer for 5 minutes"))
	f("lights_result_thanks", "a home call, its result, thanks, then the same elsewhere", homeSet,
		text("turn off the kitchen lights"), okResult, text("thanks"), text("do the same for the bedroom"))
	f("reset_same", "a call, a reset, a request for the same length", assistantSet,
		timerBread, started, turnReset, sameLength)
}

// truncatedConversations run through the library: the first answer cut at a token limit, then a turn at 512.
func truncatedConversations() []historyConversation {
	var cs []historyConversation

	for _, c := range []struct {
		id, toolset string
		limits      []int
		inputs      []turnInput
	}{
		{"truncated_timer", assistantSet, []int{4, 10, 15, 20, 25}, []turnInput{timerBread, sameLength}},
		{"truncated_two_calls", assistantSet, []int{8, 25, 45, 60}, []turnInput{twoCalls, text("thanks")}},
		{"truncated_lights", homeSet, []int{12, 20}, []turnInput{text("turn off the kitchen lights"), text("do the same for the bedroom")}},
		{"truncated_weather", assistantSet, []int{10, 30}, []turnInput{text("what's the weather in Lisbon today?"), text("and tomorrow?")}},
		{"truncated_message", assistantSet, []int{15, 35}, []turnInput{messageAnna, text("send the same message to Tom")}},
	} {
		for _, k := range c.limits {
			st := steps(c.inputs...)
			st[0].Max = k

			cs = append(cs, historyConversation{
				ID: fmt.Sprintf("%s_%d", c.id, k), Group: "truncated",
				Covers:  fmt.Sprintf("the first answer cut at %d tokens, then a turn at 512", k),
				Toolset: c.toolset, Library: true, Max: defaultMax, Steps: st,
			})
		}
	}

	// The library against the server on whole answers.
	for _, c := range []struct {
		id      string
		toolset string
		inputs  []turnInput
	}{
		{"timer_same", assistantSet, []turnInput{timerBread, started, sameLength}},
		{"respond_then_timer", assistantSet, []turnInput{text("what can you do?"), text("ok, then set a timer for 5 minutes")}},
	} {
		cs = append(cs, historyConversation{
			ID: "library_" + c.id, Group: "truncated", Covers: "the follow conversation through the library",
			Toolset: c.toolset, Library: true, Max: defaultMax, Steps: steps(c.inputs...),
		})
	}

	return cs
}

// refusedConversations put a turn --fail-input-overflow refuses before or between others.
func refusedConversations(add addConversation) {
	const g = "refused"

	long := turnInput{Unit: fillSentence, Repeat: 100}
	longJSON := probeCompact

	r := func(id, covers string, inputs ...turnInput) { add(id, g, covers, assistantSet, 7000, true, inputs...) }

	r("between_same", "a call, a refused turn, a request for the same length", timerBread, long, sameLength)
	r("between_result", "a call, a refused tool result, a result", timerPasta, longJSON, okResult)
	r("between_weather", "a weather call, a refused turn, and tomorrow?",
		text("what's the weather in Lisbon today?"), long, text("and tomorrow?"))
	r("first_then_timer", "a refused first turn, then a request", long, timerPasta)
	r("first_then_hello", "a refused first turn, then a greeting", long, text("hello"))
	r("hello_then_timer", "a greeting, a refused turn, a request", text("hello"), long,
		text("set a timer for 5 minutes"))
	r("same_no_refusal", "the control of between_same without the refused turn", timerBread, sameLength)
	r("first_twice_then_timer", "two refused first turns, then a request", long, long, timerPasta)
}

// longConversations pass the context of 8192 tokens in their last turn only (past it the engine decodes a token
// in about five seconds): 13 turns of 30 filler sentences (about 510 tokens each) that end in a timer request and
// their answers come to about 7830 tokens, and a last turn of the same length passes 8192. A fact in the system
// text or the first turn, and the last request needs it; the same over the server, with and without the flag.
func longConversations() []historyConversation {
	items := []string{"bread", "pasta", "eggs", "rice", "tea", "laundry", "oven", "soup", "beans", "cake"}
	fill := func(k int) []turnInput {
		in := make([]turnInput, 0, k)
		for i := range k {
			in = append(in, text(windowFiller(30)+fmt.Sprintf(" Set a timer for %d minutes for the %s.", 11+i%9, items[i%len(items)])))
		}

		return in
	}

	const factSystem = "date: 2026-10-11 Sun 09:30\nThe user's default timer length is 7 minutes."

	cross := text(windowFiller(30) + " Set a timer for 19 minutes for the cake.")
	askDefault := text(windowFiller(30) + " Set a timer of my default length for the soup.")
	tellDefault := text("My default timer length is 7 minutes. Set a timer of my default length for the tea.")

	var cs []historyConversation

	add := func(id, covers, system string, library, refuse bool, inputs ...turnInput) {
		cs = append(cs, historyConversation{
			ID: id, Group: "long", Covers: covers, Toolset: assistantSet, System: system, Library: library, Max: 100,
			Refuse: refuse, Steps: steps(inputs...),
		})
	}

	add("long_cross", "a last turn past the context", "", true, false, append(fill(13), cross)...)
	add("long_cross_refuse", "long_cross over the server with --fail-input-overflow, which still answers", "", false, true,
		append(fill(13), cross)...)
	add("long_system_fact", "the default length in the system text, asked past the context", factSystem, true, false,
		append(fill(13), askDefault)...)
	add("long_system_fact_near", "the default length in the system text, asked just before the context is passed",
		factSystem, true, false, append(fill(12), askDefault)...)
	add("long_turn_fact", "the default length in the first turn, asked past the context", "", true, false,
		append(append([]turnInput{tellDefault}, fill(13)...), askDefault)...)
	add("short_system_fact", "the control: the default length in the system text, asked at once", factSystem, true,
		false, text("Set a timer of my default length for the soup."))
	add("short_turn_fact", "the control: the default length in the first turn, asked in the next", "", true, false,
		tellDefault, text("Set a timer of my default length for the soup."))

	return cs
}

// cutConversations are oversized turns without --fail-input-overflow where a turn has room for 120 or 200 tokens
// (--max = context − static prefix − room), so that what the engine keeps of them lies inside its KV window: 40 filler
// sentences, the timer request, then filler words, chosen so that the request starts the given number of tokens
// before or after the cut the port made when they were recorded (the room less the turn's markers, F-051), as a first
// and as a later turn.
func cutConversations(p *spm.Processor, system, tools string, budget int) ([]historyConversation, error) {
	n, err := toolcall.Normalise([]byte(tools))
	if err != nil {
		return nil, fmt.Errorf("tools: %w", err)
	}

	c := cutText{
		p: p, prefix: windowFiller(40), request: " Set a timer for 25 minutes for the eggs.",
		words: strings.Fields(windowFiller(24)),
	}
	c.base = len(p.Encode(c.prefix))
	static := len(toolcall.StaticPrefix(p, system, n.JSON))

	var cs []historyConversation

	for _, room := range []int{120, 200} {
		for _, form := range []string{cutFirst, cutLater} {
			markers := 6 // "\n" before and "<|im_end|>\n<|im_start|>assistant\n" after a first turn's input
			if form == cutLater {
				markers = 9
			}

			seen := map[int]bool{}

			for _, off := range []int{-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12} {
				w := c.wordsFor(room - markers - off)
				if w < 0 || seen[w] {
					continue
				}

				seen[w] = true

				var in []turnInput
				if form == cutLater {
					in = append(in, text("what's the weather in Lisbon today?"))
				}

				cs = append(cs, historyConversation{
					ID: fmt.Sprintf("cut_%s_%d_%+d", form, room, off), Group: "cut",
					Covers: fmt.Sprintf("a %s turn cut to a room of %d tokens, the request %d tokens inside the port's cut",
						form, room, off),
					Toolset: assistantSet, Max: budget - static - room, Steps: steps(append(in, text(c.text(w)))...),
				})
			}
		}
	}

	return cs, nil
}

// The forms of the cut probes: a first turn and a later one.
const (
	cutFirst = "first"
	cutLater = "later"
)

// cutText builds a cut probe's input: the filler prefix, the request and w filler words after it.
type cutText struct {
	p               *spm.Processor
	prefix, request string
	words           []string
	base            int // the prefix's tokens
}

func (c cutText) text(w int) string {
	if w == 0 {
		return c.prefix + c.request
	}

	return c.prefix + c.request + " " + strings.Join(c.words[:w], " ")
}

// wordsFor is the number of filler words after which the request and they take tokens tokens, -1 for none.
func (c cutText) wordsFor(tokens int) int {
	for w := 0; w <= len(c.words); w++ {
		if len(c.p.Encode(c.text(w)))-c.base == tokens {
			return w
		}
	}

	return -1
}

// The scans: conversations whose last answer shows whether the engine dropped an earlier turn to make room for the
// answer's --max (docs/FINDINGS.md F-063).
func historyScans() []historyScan {
	return []historyScan{
		{
			ID: "timer_result_same", Group: "drop", Covers: "a call, its tool result, and a request that needs the first turn's length",
			Toolset: assistantSet, Refuse: true,
			Inputs: []turnInput{
				text("set a timer for 25 minutes for the bread"), text(`{"status": "started"}`),
				text("now set another one of the same length for the pasta"),
			},
		},
	}
}

// historyScan replays sc at --max values between the point where every turn of it fits beside the others and the
// point where its longest turn would no longer fit, bisecting wherever two replays differ.
func (o *oracle) historyScan(ctx context.Context, p *spm.Processor, budget int, sc historyScan, log io.Writer) (historyScan, error) {
	n, err := toolcall.Normalise([]byte(o.corpus.tools(sc.Toolset)))
	if err != nil {
		return sc, fmt.Errorf("tools: %w", err)
	}

	static := len(toolcall.StaticPrefix(p, o.corpus.System, n.JSON))
	longest, total := 0, 0

	sc.SHA256 = nil

	for _, in := range sc.Inputs {
		t := in.text()
		sum := sha256.Sum256([]byte(t))
		sc.SHA256 = append(sc.SHA256, hex.EncodeToString(sum[:]))

		k := len(p.Encode(t)) + 16 // the markers and room for an answer's history
		longest = max(longest, k)
		total += k + 80
	}

	sc.Hi = budget - static - longest
	sc.Lo = max(1, budget-static-total-64)

	runs := map[int][]json.RawMessage{}

	run := func(maxNew int) ([]json.RawMessage, error) {
		if a, ok := runs[maxNew]; ok {
			return a, nil
		}

		a, err := o.historyReplay(ctx, sc.Toolset, maxNew, sc.Refuse, sc.Inputs)
		if err != nil {
			return nil, err
		}

		runs[maxNew] = a

		return a, nil
	}

	err = bisectRuns(sc.Lo, sc.Hi, run)
	if err != nil {
		return sc, err
	}

	keys := make([]int, 0, len(runs))
	for k := range runs {
		keys = append(keys, k)
	}

	slices.Sort(keys)

	for _, k := range keys {
		sc.Runs = append(sc.Runs, historyRun{Max: k, Answers: runs[k]})
		_, _ = fmt.Fprintf(log, "%-24s max %5d %s\n", sc.ID, k, runs[k][len(runs[k])-1])
	}

	return sc, nil
}

// bisectRuns replays a scan at lo and hi and, where the answers differ, at the --max values between them, halving
// the interval until two neighbours differ.
func bisectRuns(lo, hi int, run func(int) ([]json.RawMessage, error)) error {
	a, err := run(lo)
	if err != nil {
		return err
	}

	b, err := run(hi)
	if err != nil {
		return err
	}

	same := slices.EqualFunc(a, b, func(x, y json.RawMessage) bool { return bytes.Equal(x, y) })
	if same || hi-lo < 2 {
		return nil
	}

	mid := (lo + hi) / 2

	err = bisectRuns(lo, mid, run)
	if err != nil {
		return err
	}

	return bisectRuns(mid, hi, run)
}

// historyReplay runs inputs on a fresh server at --max maxNew and returns the answers.
func (o *oracle) historyReplay(ctx context.Context, toolset string, maxNew int, refuse bool, inputs []turnInput) ([]json.RawMessage, error) {
	args, err := o.baseArgs(toolset, o.corpus.System)
	if err != nil {
		return nil, err
	}

	args = append(args, "--max", strconv.Itoa(maxNew))
	if refuse {
		args = append(args, "--fail-input-overflow")
	}

	s, err := startServe(ctx, o.needle(), args)
	if err != nil {
		return nil, err
	}
	defer s.close()

	out := make([]json.RawMessage, 0, len(inputs))

	for _, in := range inputs {
		if in.Reset {
			err := s.reset(ctx)
			if err != nil {
				return nil, err
			}

			out = append(out, json.RawMessage(`null`))

			continue
		}

		a, err := s.completeRaw(ctx, in.text())
		if err != nil {
			return nil, err
		}

		out = append(out, compact(a))
	}

	return out, nil
}

// postBody posts a JSON body as is.
func postBody(ctx context.Context, url string, body []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req) // #nosec G704 -- a loopback URL built from a free port.
	if err != nil {
		return nil, fmt.Errorf("post %s: %w", url, err)
	}

	defer func() { _ = resp.Body.Close() }()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", url, err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("post %s: %s: %s", url, resp.Status, b)
	}

	return b, nil
}
