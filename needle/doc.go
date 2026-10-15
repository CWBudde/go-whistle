// Package needle is a pure-Go port of the Needle 3 text model of Cactus Compute: tool calling from text, the
// envelope the engine (libneedle3) answers with, and the Python client's needle.Needle on top of it (Complete, Run
// with Go tool functions, Extract, Reset). No cgo and no Python at run time. It is rebuilt from the published
// weights and checked against the closed-source engine as a test oracle; docs/FINDINGS.md records every decision.
//
// # Loading
//
// Load opens the directory that `just fetch-needle3` fills: needle3.cact (2/4-bit CQ weights) and tokenizer.model.
// By default the model runs the engine's numerics: the packed weights on the exact kernel, int8 activations and KV
// cache, and the KV window of 256 positions with the static prefix (system facts and tools) kept in view. WithoutInt8
// keeps float32 activations, WithLUT multiplies on the faster lookup-table kernel, and WithFloatWeights runs the
// float32 checkpoint needle3.safetensors instead.
//
//	m, err := needle.Load("models/needle3")
//	if err != nil {
//		return err
//	}
//	defer m.Close()
//
// Several agents may use one Model at the same time. Close it when done, and not while a call runs.
//
// # Agents and the envelope
//
// NewAgent starts a conversation over a tool set, a JSON array of tool schemas ({"name", "description",
// "parameters"} or OpenAI's {"type": "function", "function": {…}}), with optional system facts. Complete answers
// a turn and returns the engine's envelope:
//
//	a, err := m.NewAgent(tools, needle.AgentOptions{System: "home: Berlin"})
//	r, err := a.Complete("set a timer for 10 minutes", needle.CompleteOptions{})
//	js, _ := json.Marshal(r)
//	// {"type":"call","success":true,"error":null,"error_code":null,"reason":null,
//	//  "function_calls":[{"name":"set_timer","arguments":{"minutes":10}}],"suppressed_calls":[],
//	//  "reasoning":"…","confidence":…,"validation":{"ungrounded":[],"negation":false}}
//
// Response marshals to the engine's JSON byte for byte. Calls carry the caller's names; the engine renders snake_case
// names to the model and maps them back. Calls below a confidence of 0.1, or that a grounding gate withholds, come
// as SuppressedCalls; a truncated answer fails with ErrorCode "truncated".
//
// Every Complete adds its turn to the conversation, as the engine's needle_complete does: after an answer that made
// a call, a JSON input is that call's result, anything else a new user turn. Reset starts over; Stateless resets
// before every call. A turn longer than the room the context leaves (8192 positions less the static prefix and
// MaxNewTokens) is cut to fit and the conversation before it dropped, as the engine does; with FailInputOverflow it
// is refused with an *OverflowError instead. A conversation whose history outgrows the context answers the next
// turn alone, a port choice (docs/FINDINGS.md F-064). An empty tool list answers every request with the engine's
// empty call.
//
// AutoDate puts today's date fact in front of the system facts, as the Python client does by default; the engine
// itself adds no date, and neither does an agent without it.
//
// # Run and Extract
//
// Run is the Python client's agent loop: it answers a query, runs the calls with Go functions, feeds their results
// back as one tool-result turn and repeats until an answer makes no call or MaxSteps rounds have run. By default it
// does not run a call with a field the engine reports as ungrounded (or a date whose year the query does not
// license) and feeds back {"error": "ungrounded …"} instead; Lenient turns that off. Unknown tools and tool errors
// come back as {"error": …} too, and Response.Results holds every result.
//
//	r, err := a.Run("set a timer for 10 minutes", map[string]needle.ToolFunc{
//		"set_timer": func(args json.RawMessage) (any, error) { return map[string]any{"ok": true}, nil },
//	}, needle.RunOptions{})
//
// Extract fills one schema from a text: the arguments of the first call (or of the first suppressed call), checked
// unless Lenient against the text as upstream's _validate_extraction does; an *ExtractionValidationError names the
// fields it cannot ground. ExtractInto decodes them into a Go value.
//
// # What is not ported
//
// The Python client's telemetry, its warning after four turns without Reset, Pydantic models as schemas and
// fine-tuned weights. Text embeddings (needle_embed) and audio input (Whistle's transcript into a tool call) are
// later items of PLAN.md.
package needle
