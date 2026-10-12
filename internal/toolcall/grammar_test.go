package toolcall

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/spm"
)

// grammarCase is one tool's parameters and the arguments its grammar
// accepts and rejects, each a whole call block of one call.
type grammarCase struct {
	params   string
	accept   []string
	reject   []string
	prefixes []string // arguments prefixes that must stay alive
	dead     []string // arguments prefixes that must not
}

// oneCall is the call block of one call of t with the arguments args.
func oneCall(args string) string {
	return `[{"name":"t","arguments":` + args + `}]`
}

func checkGrammar(t *testing.T, name string, c grammarCase) {
	t.Helper()

	n, err := Normalise([]byte(`[{"name":"t","parameters":` + c.params + `}]`))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}

	g := n.Grammar()

	for _, a := range c.accept {
		if !g.Accepts([]byte(oneCall(a))) {
			t.Errorf("%s: rejects %s", name, a)
		}
	}

	for _, a := range c.reject {
		if g.Accepts([]byte(oneCall(a))) {
			t.Errorf("%s: accepts %s", name, a)
		}
	}

	head := strings.TrimSuffix(oneCall(""), "}]")

	for _, p := range c.prefixes {
		if !g.Alive([]byte(head + p)) {
			t.Errorf("%s: prefix %s is dead", name, p)
		}
	}

	for _, p := range c.dead {
		if g.Alive([]byte(head + p)) {
			t.Errorf("%s: prefix %s is alive", name, p)
		}
	}
}

// TestGrammarTypes: each JSON type, compact JSON only, the keys of the
// schema only, each once (docs/FINDINGS.md F-055).
func TestGrammarTypes(t *testing.T) {
	checkGrammar(t, "types", grammarCase{
		params: `{"type":"object","properties":{"s":{"type":"string"},"i":{"type":"integer"},` +
			`"n":{"type":"number"},"b":{"type":"boolean"},"z":{"type":"null"},` +
			`"a":{"type":"array","items":{"type":"integer"}},"o":{"type":"object","properties":{"x":{"type":"string"}}},` +
			`"u":{"type":["string","null"]},"free":{}}}`,
		accept: []string{
			`{}`,
			`{"s":"x","i":-12,"n":1.5e3,"b":true,"z":null,"a":[1,2],"o":{"x":"y"}}`,
			`{"o":{},"a":[],"i":0,"n":-0.25,"b":false}`,
			`{"s":"caf\u00e9 \"q\" \\ \n","u":null}`,
			`{"s":"café ☕ 𝄞","u":"x"}`,
			`{"free":{"any":[1,"x",null,{"k":true}]}}`,
		},
		reject: []string{
			`{"i":1.5}`, `{"i":1e2}`, `{"i":01}`, `{"i":-}`, `{"i":+1}`,
			`{"s":1}`, `{"s":"a` + "\x01" + `"}`, `{"s":"` + "\xff" + `"}`, `{"s":"` + "\xc3" + `"}`,
			`{"s":"\x"}`, `{"s":"\u12"}`,
			`{"n":"1"}`, `{"n":1.}`, `{"n":.5}`,
			`{"b":1}`, `{"b":True}`, `{"z":false}`,
			`{"a":["x"]}`, `{"a":[1,]}`,
			`{"o":{"y":"z"}}`, `{"u":1}`,
			`{ "s":"x"}`, `{"s": "x"}`, `{"s":"x" }`,
			`{"q":1}`, `{"s":"a","s":"b"}`,
		},
		prefixes: []string{`{"s":"ab`, `{"i":-1`, `{"n":2.`, `{"o":{"x":"`},
		dead:     []string{`{"sx`, `{"i":1.`, `{"q`},
	})
}

// TestGrammarEnum: an enum is one of its values as the tools spell them.
func TestGrammarEnum(t *testing.T) {
	checkGrammar(t, "enum", grammarCase{
		params: `{"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]},` +
			`"level":{"enum":[1,2.5,null,"x y"]}}}`,
		accept: []string{`{"unit":"celsius"}`, `{"unit":"fahrenheit"}`, `{"level":2.5}`, `{"level":null}`, `{"level":"x y"}`},
		reject: []string{
			`{"unit":"kelvin"}`, `{"unit":"Celsius"}`, `{"unit":"celsius "}`, `{"unit":"cel"}`,
			`{"level":3}`, `{"level":2.50}`, `{"level":"1"}`,
		},
		prefixes: []string{`{"unit":"c`, `{"unit":"f`},
		dead:     []string{`{"unit":"k`, `{"level":2.51`},
	})
}

// TestGrammarConst: a const is its one value.
func TestGrammarConst(t *testing.T) {
	checkGrammar(t, "const", grammarCase{
		params:   `{"type":"object","properties":{"mode":{"const":"on"},"pair":{"const":{"a":1,"b":[true]}}}}`,
		accept:   []string{`{"mode":"on"}`, `{"pair":{"a":1,"b":[true]}}`},
		reject:   []string{`{"mode":"off"}`, `{"mode":"onn"}`, `{"pair":{"b":[true],"a":1}}`, `{"pair":{"a":2,"b":[true]}}`},
		prefixes: []string{`{"mode":"o`},
		dead:     []string{`{"mode":"of`},
	})
}

// TestGrammarBounds: numbers stay within minimum, maximum and their
// exclusive forms ("a bounded number cannot grow past its maximum"),
// strings within their lengths in characters, arrays within their item
// counts; a numeric literal stops at twenty characters, an array at
// sixty-four items.
func TestGrammarBounds(t *testing.T) {
	ints := make([]string, 65)
	for i := range ints {
		ints[i] = "1"
	}

	checkGrammar(t, "bounds", grammarCase{
		params: `{"type":"object","properties":{"brightness":{"type":"integer","minimum":0,"maximum":100},` +
			`"t":{"type":"number","exclusiveMinimum":0,"maximum":1.5},"neg":{"type":"integer","minimum":-20,"maximum":-10},` +
			`"s":{"type":"string","minLength":2,"maxLength":3},"l":{"type":"array","minItems":1,"maxItems":2},` +
			`"big":{"type":"integer"},"many":{"type":"array"}}}`,
		accept: []string{
			`{"brightness":0}`, `{"brightness":100}`, `{"brightness":55}`,
			`{"t":0.5}`, `{"t":1.5}`, `{"t":0.001}`, `{"neg":-15}`, `{"neg":-20}`,
			`{"s":"ab"}`, `{"s":"abc"}`, `{"s":"äöü"}`, `{"s":"\u00e4b"}`,
			`{"l":[1]}`, `{"l":[1,"x"]}`,
			`{"big":12345678901234567890}`,
			`{"many":[` + strings.Join(ints[:64], ",") + `]}`,
		},
		reject: []string{
			`{"brightness":-1}`, `{"brightness":101}`, `{"brightness":1000}`, `{"brightness":100.0}`,
			`{"t":0}`, `{"t":-0.1}`, `{"t":1.51}`, `{"t":2}`, `{"t":1e-3}`, `{"neg":-9}`, `{"neg":-21}`, `{"neg":5}`,
			`{"s":"a"}`, `{"s":"abcd"}`, `{"s":""}`,
			`{"l":[]}`, `{"l":[1,2,3]}`,
			`{"big":123456789012345678901}`,
			`{"many":[` + strings.Join(ints, ",") + `]}`,
		},
		prefixes: []string{`{"brightness":1`, `{"brightness":10`, `{"t":1.`, `{"t":0.0`, `{"neg":-1`, `{"neg":-2`},
		dead: []string{
			`{"brightness":101`, `{"brightness":-1`, `{"t":1.6`, `{"t":3`, `{"neg":-3`, `{"neg":-21`, `{"s":"abcd`,
			// No state without a way out: no escape or UTF-8 sequence past
			// maxLength, no unfinished literal at the cap.
			`{"s":"abc\`, `{"s":"abc` + "\xc3", `{"big":1234567890123456789.`,
		},
	})
}

// TestGrammarDate: format: date is YYYY-MM-DD, months 01–12, days 01–31.
func TestGrammarDate(t *testing.T) {
	checkGrammar(t, "date", grammarCase{
		params: `{"type":"object","properties":{"due_date":{"type":"string","format":"date"}}}`,
		accept: []string{`{"due_date":"2026-10-12"}`, `{"due_date":"2024-02-29"}`, `{"due_date":"2026-12-31"}`},
		reject: []string{
			`{"due_date":"2026-13-01"}`, `{"due_date":"2026-00-10"}`, `{"due_date":"2026-10-32"}`,
			`{"due_date":"2026-10-00"}`, `{"due_date":"2026-10-1"}`, `{"due_date":"26-10-12"}`,
			`{"due_date":"2026/10/12"}`, `{"due_date":"2026-10-12T09:00"}`, `{"due_date":"October 30"}`,
			`{"due_date":"\u0032026-10-12"}`,
		},
		prefixes: []string{`{"due_date":"202`, `{"due_date":"2026-1`},
		dead:     []string{`{"due_date":"O`, `{"due_date":"2026-2`, `{"due_date":"2026-10-12 `},
	})
}

// TestGrammarRequired: an object closes only after its required keys, in
// any order; keys outside the schema stay out (docs/FINDINGS.md F-055).
func TestGrammarRequired(t *testing.T) {
	checkGrammar(t, "required", grammarCase{
		params: `{"type":"object","properties":{"room":{"type":"string"},"on":{"type":"boolean"},` +
			`"brightness":{"type":"integer"}},"required":["room","on"]}`,
		accept: []string{`{"room":"x","on":true}`, `{"on":false,"room":"x","brightness":5}`},
		reject: []string{`{"room":"x"}`, `{}`, `{"on":true,"brightness":5}`, `{"room":"x","off":false}`},
		dead:   []string{`{"room":"x"}`, `{"room":"x","of`, `{"room":"x","on":true,"action`},
	})
}

// TestGrammarAdditionalProperties: an object with properties takes those
// keys only, whatever its additionalProperties (the engine's grammar.json
// probes additional_true, additional_integer); one without properties takes
// any keys, valued under an additionalProperties schema, none under false
// (docs/FINDINGS.md F-055).
func TestGrammarAdditionalProperties(t *testing.T) {
	checkGrammar(t, "additionalProperties", grammarCase{
		params: `{"type":"object","properties":{"tags":{"type":"object","additionalProperties":{"type":"integer"}},` +
			`"meta":{"type":"object"},"open":{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true},` +
			`"typed":{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"integer"}},` +
			`"shut":{"type":"object","additionalProperties":false},"odd":{"type":"object","properties":{"a":{"type":"string"}},` +
			`"required":["a","missing"]},"never":{"type":"object","properties":{"a":false,"b":{"type":"string"}}}}}`,
		accept: []string{
			`{"tags":{"x":1,"y z":2}}`, `{"meta":{"k":[1,{"j":null}]}}`, `{"open":{"a":"s"}}`, `{"shut":{}}`,
			`{"odd":{"a":"s"}}`, `{"never":{"b":"s"}}`,
		},
		reject: []string{
			`{"tags":{"x":"1"}}`, `{"tags":{"x":1,"x":2}}`, `{"open":{"a":1}}`, `{"open":{"a":"s","b":1}}`,
			`{"typed":{"a":"s","b":1}}`, `{"shut":{"k":1}}`, `{"odd":{}}`,
		},
		dead: []string{`{"never":{"a`, `{"open":{"a":"s",`},
	})
}

// TestGrammarAnyOf: one branch of anyOf (and oneOf) holds, merged with the
// schema around it.
func TestGrammarAnyOf(t *testing.T) {
	checkGrammar(t, "anyOf", grammarCase{
		params: `{"type":"object","properties":{"when":{"anyOf":[{"type":"string","format":"date"},` +
			`{"type":"integer","minimum":1}]},"pick":{"oneOf":[{"const":"a"},{"const":"b"}]}},` +
			`"anyOf":[{"properties":{"x":{"type":"string"}},"required":["x"]},{"properties":{"y":{"type":"integer"}},"required":["y"]}]}`,
		accept: []string{
			`{"x":"1","when":"2026-10-12"}`, `{"y":1,"when":5}`, `{"pick":"b","x":"q"}`,
		},
		reject: []string{
			`{"x":"1","when":"tomorrow"}`, `{"x":"1","when":0}`, `{"pick":"c","x":"q"}`, `{}`, `{"when":5}`,
			`{"y":"1"}`,
			// A branch's object is closed to the other branch's keys (port-side).
			`{"x":"a","y":2}`,
		},
	})
}

// TestGrammarAllOf: every allOf branch holds, merged (port-side).
func TestGrammarAllOf(t *testing.T) {
	checkGrammar(t, "allOf", grammarCase{
		params: `{"type":"object","allOf":[{"properties":{"a":{"type":"string"}},"required":["a"]},` +
			`{"properties":{"b":{"type":"integer"}},"required":["b"]}]}`,
		accept: []string{`{"a":"x","b":1}`, `{"b":1,"a":"x"}`},
		reject: []string{`{"a":"x"}`, `{"b":1}`, `{"a":"x","b":1,"c":2}`},
	})
}

// TestGrammarRef: $ref resolves against the parameters, recursion
// included, with its sibling keys merged.
func TestGrammarRef(t *testing.T) {
	checkGrammar(t, "$ref", grammarCase{
		params: `{"type":"object","$defs":{"point":{"type":"object","properties":{"x":{"type":"number"},` +
			`"y":{"type":"number"}},"required":["x","y"]},"node":{"type":"object","properties":{"v":{"type":"integer"},` +
			`"next":{"$ref":"#/$defs/node"}},"required":["v"]}},"properties":{"p":{"$ref":"#/$defs/point"},` +
			`"ps":{"type":"array","items":{"$ref":"#/$defs/point"}},"list":{"$ref":"#/$defs/node"},` +
			`"q":{"$ref":"#/$defs/point","description":"a point"},"r":{"$ref":"#/$defs/point","required":["x","y","z"],` +
			`"properties":{"z":{"type":"boolean"}}},"lost":{"$ref":"#/$defs/missing"}}}`,
		accept: []string{
			`{"p":{"x":1,"y":2.5}}`, `{"ps":[{"x":1,"y":2},{"y":0,"x":0}]}`, `{"q":{"x":1,"y":2}}`,
			`{"list":{"v":1,"next":{"v":2,"next":{"v":3}}}}`, `{"r":{"x":1,"y":2,"z":true}}`, `{"lost":["anything"]}`,
		},
		reject: []string{
			`{"p":{"x":1}}`, `{"p":{"x":"1","y":2}}`, `{"ps":[{"x":1}]}`, `{"list":{"v":1,"next":{"next":{"v":3}}}}`,
			`{"r":{"x":1,"y":2}}`,
		},
	})
}

// TestGrammarCalls: the call block around the arguments: the empty list,
// several calls, only the tools' names, "name" before "arguments".
func TestGrammarCalls(t *testing.T) {
	tools := `[{"name":"setTimer","parameters":{"type":"object","properties":{"minutes":{"type":"integer"}},` +
		`"required":["minutes"]},"triggers":["\\btimer\\b"]},{"name":"lock_door","parameters":{"type":"object",` +
		`"properties":{"door":{"type":"string"}},"required":["door"]}},{"name":"ping"}]`

	n, err := Normalise([]byte(tools))
	if err != nil {
		t.Fatal(err)
	}

	g := n.Grammar()

	for _, b := range []string{
		`[]`,
		`[{"name":"set_timer","arguments":{"minutes":5}}]`,
		`[{"name":"set_timer","arguments":{"minutes":5}},{"name":"lock_door","arguments":{"door":"front"}}]`,
		`[{"name":"ping","arguments":{}}]`,
	} {
		if !g.Accepts([]byte(b)) {
			t.Errorf("rejects %s", b)
		}
	}

	for _, b := range []string{
		`[{"name":"setTimer","arguments":{"minutes":5}}]`,
		`[{"name":"set_timer","arguments":{"door":"x"}}]`,
		`[{"arguments":{"minutes":5},"name":"set_timer"}]`,
		`[{"name":"nope","arguments":{}}]`,
		`[{"name":"ping","arguments":{"x":1}}]`,
		`[{"name":"set_timer","arguments":{"minutes":5}},]`,
		`[{"name":"set_timer","arguments":{"minutes":5}}`,
		`[] `,
		` []`,
	} {
		if g.Accepts([]byte(b)) {
			t.Errorf("accepts %s", b)
		}
	}

	// A trigger routes the request: only the triggered tool, and a call.
	routed := n.GrammarFor("start a timer")
	if routed.Accepts([]byte(`[]`)) || routed.Accepts([]byte(`[{"name":"lock_door","arguments":{"door":"x"}}]`)) ||
		!routed.Accepts([]byte(`[{"name":"set_timer","arguments":{"minutes":5}}]`)) {
		t.Error("the routed grammar is not the triggered tool's")
	}

	// A negated request is not routed (negation_triggered keeps lock_door).
	if !n.GrammarFor("don't start a timer").Accepts([]byte(`[]`)) || !n.GrammarFor("lock it").Accepts([]byte(`[]`)) {
		t.Error("a negated or untriggered request is routed")
	}
}

// TestGrammarBooleanBranches: a false branch of anyOf admits nothing, a
// false branch of allOf makes the schema admit nothing (PR #65 review).
func TestGrammarBooleanBranches(t *testing.T) {
	checkGrammar(t, "boolean branches", grammarCase{
		params: `{"type":"object","$defs":{"no":false},"properties":{"s":{"anyOf":[false,{"type":"string"}]},` +
			`"r":{"anyOf":[{"$ref":"#/$defs/no"},{"type":"integer"}]},"none":{"allOf":[false]},` +
			`"yes":{"allOf":[true,{"type":"boolean"}]}}}`,
		accept: []string{`{"s":"x"}`, `{"r":1}`, `{"yes":true}`, `{}`},
		reject: []string{`{"s":1}`, `{"s":null}`, `{"r":"x"}`, `{"none":1}`, `{"none":"x"}`, `{"yes":1}`},
		dead:   []string{`{"none`},
	})

	// "parameters": false is no arguments at all, not none declared: the
	// tool is not callable, and a trigger for it routes nothing.
	n, err := Normalise([]byte(`[{"name":"t","parameters":false,"triggers":["go"]},{"name":"u"}]`))
	if err != nil {
		t.Fatal(err)
	}

	if g := n.Grammar(); g.Accepts([]byte(`[{"name":"t","arguments":{}}]`)) ||
		!g.Accepts([]byte(`[{"name":"u","arguments":{}}]`)) {
		t.Error("a tool with parameters false is callable, or one without parameters is not")
	}

	if g := n.GrammarFor("go"); g.required || !g.Accepts([]byte(`[]`)) {
		t.Error("a trigger for an uncallable tool requires a call")
	}

	// "parameters": true takes any arguments; a required property whose
	// schema admits nothing makes the tool uncallable (PR #65 review).
	n, err = Normalise([]byte(`[{"name":"any","parameters":true},` +
		`{"name":"req","parameters":{"type":"object","properties":{"f":false},"required":["f"]}}]`))
	if err != nil {
		t.Fatal(err)
	}

	g := n.Grammar()
	for _, block := range []string{`[{"name":"any","arguments":{}}]`, `[{"name":"any","arguments":{"x":1,"y":["z"]}}]`} {
		if !g.Accepts([]byte(block)) {
			t.Errorf("parameters true rejects %s", block)
		}
	}

	if g.Accepts([]byte(`[{"name":"req","arguments":{}}]`)) {
		t.Error("a tool with an impossible required property is callable")
	}
}

// TestGrammarSurrogatePairs: an escaped surrogate pair is one character
// under the length bounds, as its UTF-8 spelling (PR #65 review).
func TestGrammarSurrogatePairs(t *testing.T) {
	checkGrammar(t, "surrogate pairs", grammarCase{
		params: `{"type":"object","properties":{"c":{"type":"string","maxLength":1},"two":{"type":"string","minLength":2}}}`,
		accept: []string{`{"c":"𝄞"}`, `{"c":"𝄞"}`, `{"c":"é"}`, `{"two":"𝄞x"}`},
		reject: []string{`{"c":"𝄞x"}`, `{"c":"ab"}`, `{"two":"𝄞"}`},
		// Past a high surrogate at the limit, only its low surrogate fits.
		prefixes: []string{`{"c":"\uD834\uD`},
		dead:     []string{`{"c":"\uD834\u0`, `{"c":"\uD834\uDB`, `{"c":"\uD834x`, `{"c":"\uD834\n`},
	})
}

// stuckState is a state of g, reached from the start, where no byte
// continues and the block has not ended, or "" when g has none; it
// explores up to limit distinct states.
func stuckState(g *Grammar, limit int) string {
	type visit struct {
		states []*stack
		text   string
	}

	seen := map[string]bool{}
	queue := []visit{{states: g.start()}}

	for len(queue) > 0 && len(seen) < limit {
		v := queue[0]
		queue = queue[1:]

		key := stateKey(v.states)
		if seen[key] {
			continue
		}

		seen[key] = true
		next := 0

		for b := range 256 {
			if s := advanceAll(v.states, byte(b)); len(s) > 0 {
				next++

				queue = append(queue, visit{states: s, text: v.text + string(rune(b))})
			}
		}

		if next == 0 && !slices.ContainsFunc(v.states, (*stack).accepting) {
			return v.text
		}
	}

	return ""
}

// TestGrammarNoDeadEnds: every state the grammar lets the decoder reach
// can still end, so the token mask never runs empty, also for tools
// without parameters and for schemas no value matches (PR #65 review).
func TestGrammarNoDeadEnds(t *testing.T) {
	for name, tools := range map[string]string{
		"no parameters":    `[{"name":"t"}]`,
		"empty properties": `[{"name":"t","parameters":{"type":"object","properties":{}}}]`,
		"false parameters": `[{"name":"t","parameters":false},{"name":"u","parameters":{"type":"object"}}]`,
		"true parameters":  `[{"name":"t","parameters":true}]`,
		"min over max length": `[{"name":"t","parameters":{"type":"object","properties":{` +
			`"s":{"type":"string","minLength":2,"maxLength":1},"d":{"type":"string","format":"date","maxLength":5}}}}]`,
		"min over max items": `[{"name":"t","parameters":{"type":"object","properties":{` +
			`"a":{"type":"array","minItems":3,"maxItems":2},"b":{"type":"array","minItems":65},` +
			`"c":{"type":"array","minItems":1,"items":false}}}}]`,
		"empty number range": `[{"name":"t","parameters":{"type":"object","properties":{` +
			`"i":{"type":"integer","minimum":1.2,"maximum":1.8},"n":{"type":"number","minimum":2,"maximum":1},` +
			`"x":{"type":"number","exclusiveMinimum":1,"exclusiveMaximum":1}}}}]`,
		"empty enum":     `[{"name":"t","parameters":{"type":"object","properties":{"e":{"enum":[]}}}}]`,
		"false allOf":    `[{"name":"t","parameters":{"type":"object","properties":{"f":{"allOf":[false]}}}}]`,
		"false required": `[{"name":"t","parameters":{"type":"object","properties":{"f":false},"required":["f"]}}]`,
		"length limit": `[{"name":"t","parameters":{"type":"object","properties":{` +
			`"c":{"type":"string","maxLength":1}}}}]`,
	} {
		n, err := Normalise([]byte(tools))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		for _, g := range []*Grammar{n.Grammar(), n.grammar(n.tools, true)} {
			if s := stuckState(g, 20000); s != "" {
				t.Errorf("%s (required %v): stuck after %q", name, g.required, s)
			}
		}
	}
}

// testTokenizer is a small sentencepiece model: <unk>, <s>, </s>, the
// user-defined <tool_call>, the 256 byte-fallback pieces and normal pieces.
func testTokenizer(t *testing.T, normal ...string) *spm.Processor {
	t.Helper()

	pieces := make([]spm.Piece, 0, 4+256+len(normal))
	pieces = append(pieces,
		spm.Piece{Piece: "<unk>", Type: spm.TypeUnknown}, spm.Piece{Piece: "<s>", Type: spm.TypeControl},
		spm.Piece{Piece: "</s>", Type: spm.TypeControl}, spm.Piece{Piece: "<tool_call>", Type: spm.TypeUserDefined})

	for b := range 256 {
		pieces = append(pieces, spm.Piece{Piece: fmt.Sprintf("<0x%02X>", b), Type: spm.TypeByte})
	}

	for i, s := range normal {
		pieces = append(pieces, spm.Piece{Piece: s, Score: -float32(i), Type: spm.TypeNormal})
	}

	p, err := spm.New(&spm.ModelProto{
		Pieces: pieces,
		Trainer: spm.TrainerSpec{
			ModelType: spm.ModelBPE, ByteFallback: true, UnkPiece: "<unk>", BOSPiece: "<s>", EOSPiece: "</s>",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	return p
}

// TestTokenMask: a token is allowed iff its bytes keep the automaton
// alive; byte-fallback pieces stand for their byte (a UTF-8 sequence
// spans several of them), normal pieces for their text with "▁" a space,
// and other pieces never stand in the call block (docs/FINDINGS.md F-055).
func TestTokenMask(t *testing.T) {
	p := testTokenizer(t, `[{"`, `name`, `":"`, `set`, `_`, `timer`, `","`, `arguments`, `":{"`, `minutes`,
		`":`, `10`, `}}]`, `"}}]`, `label`, `▁door`, `▁`, `"`, `,"`, `ab`, `a"`, `[]`, `x`, `":1`, `0}`)
	m := newTokenMask(p, p.Len())

	tools := `[{"name":"set_timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer",` +
		`"maximum":100},"label":{"type":"string"}},"required":["minutes"]}}]`

	n, err := Normalise([]byte(tools))
	if err != nil {
		t.Fatal(err)
	}

	g := n.Grammar()

	for _, prefix := range []string{
		``, `[`, `[{"name":"`, `[{"name":"set_`, `[{"name":"set_timer","arguments":{"`,
		`[{"name":"set_timer","arguments":{"minutes":1`, `[{"name":"set_timer","arguments":{"minutes":10,"label":"`,
		`[{"name":"set_timer","arguments":{"minutes":10,"label":"` + "\xc3", `[{"name":"set_timer","arguments":{"minutes":10}}`,
	} {
		states := g.feed(g.start(), []byte(prefix))
		if len(states) == 0 {
			t.Fatalf("prefix %q is dead", prefix)
		}

		got := slices.Sorted(slices.Values(m.allowed(g, states)))

		var want []int

		for id := range p.Len() {
			bs := m.bytes[id]
			if bs != nil && g.Alive(append([]byte(prefix), bs...)) {
				want = append(want, id)
			}
		}

		if !slices.Equal(got, want) {
			t.Errorf("after %q: allowed %v, want %v", prefix, pieces(p, got), pieces(p, want))
		}

		// Cached, the same.
		if again := slices.Sorted(slices.Values(m.allowed(g, states))); !slices.Equal(again, got) {
			t.Errorf("after %q: cached %v, want %v", prefix, again, got)
		}
	}

	byteID := func(b byte) int {
		id, _ := p.PieceID(fmt.Sprintf("<0x%02X>", b))

		return id
	}

	inString := g.feed(g.start(), []byte(`[{"name":"set_timer","arguments":{"minutes":10,"label":"`))
	allowed := m.allowed(g, inString)

	for b, want := range map[byte]bool{'a': true, ' ': true, 0xC3: true, 0xE2: true, 0x80: false, 0xFF: false, 0x01: false} {
		if slices.Contains(allowed, byteID(b)) != want {
			t.Errorf("in a string, <0x%02X> allowed %v, want %v", b, !want, want)
		}
	}

	afterLead := m.allowed(g, g.feed(inString, []byte{0xC3}))
	if !slices.Contains(afterLead, byteID(0xA4)) || slices.Contains(afterLead, byteID('a')) {
		t.Error("after a UTF-8 lead byte, the continuation bytes and only they are allowed")
	}

	for _, piece := range []string{"<unk>", "<s>", "</s>", "<tool_call>"} {
		id, _ := p.PieceID(piece)
		if m.bytes[id] != nil {
			t.Errorf("%s has bytes", piece)
		}
	}
}

// TestConstraint: the phases of an answer under a grammar: the reasoning
// free, the part after </think> free unless routed, where "\n" and
// <tool_call> are forced, then the call block's mask with </tool_call>
// once it may end.
func TestConstraint(t *testing.T) {
	p := testTokenizer(t, `</think>`, `</tool_call>`, `[]`, `[`, `]`, `▁ok`)

	id := func(piece string) int {
		i, ok := p.PieceID(piece)
		if !ok {
			t.Fatalf("no %s", piece)
		}

		return i
	}

	d := &Decoder{
		p: p, mask: newTokenMask(p, p.Len()), thinkEnd: id("</think>"), open: id("<tool_call>"),
		call: id("</tool_call>"),
	}

	n, err := Normalise([]byte(`[{"name":"t","triggers":["go"]}]`))
	if err != nil {
		t.Fatal(err)
	}

	for _, routed := range []bool{false, true} {
		input := "stop"
		if routed {
			input = "go"
		}

		c := &constraint{d: d, g: n.GrammarFor(input)}

		mustFree := func(when string) {
			ids, err := c.allowed()
			if err != nil || ids != nil {
				t.Errorf("routed %v, %s: allowed %v, %v; want any", routed, when, ids, err)
			}
		}

		mustOnly := func(when string, want ...int) {
			ids, err := c.allowed()
			if err != nil || !slices.Equal(slices.Sorted(slices.Values(ids)), want) {
				t.Errorf("routed %v, %s: allowed %v, %v; want %v", routed, when, ids, err, want)
			}
		}

		mustFree("in the reasoning")
		c.took(id("▁ok"))
		mustFree("in the reasoning")
		c.took(d.thinkEnd)

		if routed {
			mustOnly("after </think>", id("<0x0A>"))
			c.took(id("<0x0A>"))
			mustOnly("after the newline", d.open)
		} else {
			mustFree("after </think>")
			c.took(id("<0x0A>"))
		}

		c.took(d.open)

		ids, _ := c.allowed()
		if slices.Contains(ids, id("[]")) == routed || !slices.Contains(ids, id("[")) || slices.Contains(ids, d.call) {
			t.Errorf("routed %v, after <tool_call>: allowed %v", routed, pieces(p, ids))
		}

		if !routed {
			c.took(id("[]"))
			mustOnly("after the empty list", d.call)
		}
	}
}

func pieces(p *spm.Processor, ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = p.Piece(id)
	}

	return out
}

// TestTokenMaskVocabulary: over Needle 3's whole vocabulary, the mask is
// the tokens whose bytes keep the automaton alive, at every fifth prefix
// of a call block of the corpus's home tools.
func TestTokenMaskVocabulary(t *testing.T) {
	p := loadTokenizer(t)
	m := newTokenMask(p, p.Len())

	var corpus engineCorpus

	readJSON(t, engineCases, &corpus)

	n, err := Normalise(corpus.Toolsets["home"])
	if err != nil {
		t.Fatal(err)
	}

	g := n.Grammar()
	block := `[{"name":"set_lights","arguments":{"room":"salón","on":false,"brightness":40}},` +
		`{"name":"set_thermostat","arguments":{"temperature":21.5,"unit":"celsius"}}]`

	for i := 0; i < len(block); i += 5 {
		prefix := block[:i]

		got := slices.Sorted(slices.Values(m.allowed(g, g.feed(g.start(), []byte(prefix)))))

		var want []int

		for id, bs := range m.bytes {
			if bs != nil && g.Alive(append([]byte(prefix), bs...)) {
				want = append(want, id)
			}
		}

		if !slices.Equal(got, want) {
			t.Fatalf("after %q: %d tokens allowed, want %d", prefix, len(got), len(want))
		}
	}
}

const engineGrammar = "../../testdata/text/engine/grammar.json"

// The grammar.json probes that differ from the engine's, and why
// (docs/FINDINGS.md F-055).
var (
	// grammarCallsDiffer: the calls (dispatched or suppressed).
	grammarCallsDiffer = map[string]string{
		"bounds_max":       "the repair drops brightness 100, which the request does not state",
		"required_missing": "the repair capitalises the label",
		"ref":              "decoded after a different reasoning, the engine writes a phone",
		"number_bounds":    "the engine truncates",
		"any_of":           "the engine truncates",
		"max_items":        "the engine truncates",
		"max_length":       "the engine truncates",
	}
	// grammarReasoningNearTie: the reasoning, at a near-tie (as
	// reasoningNearTie).
	grammarReasoningNearTie = map[string]bool{"bounds_min_max": true, "ref": true}
)

// TestGrammarMatchesEngine decodes the engine's grammar probes, requests
// whose free decode would break the tool's schema, under the grammar with
// the dequantised needle3.cact: the calls (dispatched or suppressed) and
// the reasoning are the engine's but on the probes listed, which the test
// checks still differ (docs/FINDINGS.md F-055).
func TestGrammarMatchesEngine(t *testing.T) {
	if testing.Short() {
		t.Skip("decodes the grammar probes")
	}

	p := loadTokenizer(t)
	m := loadModel(t, needle3Cact)

	d, err := NewDecoder(m, p)
	if err != nil {
		t.Fatal(err)
	}

	var fx struct {
		System string `json:"system"`
		Probes []struct {
			ID     string         `json:"id"`
			Tools  string         `json:"tools"`
			Query  string         `json:"query"`
			Answer engineEnvelope `json:"answer"`
		} `json:"probes"`
	}

	readJSON(t, engineGrammar, &fx)

	for _, pr := range fx.Probes {
		turn := decodeTurns(t, d, m, p, fx.System, []byte(pr.Tools), []string{pr.Query}, 512)[0]

		js, err := json.Marshal(turn.env)
		if err != nil {
			t.Fatal(err)
		}

		var got engineEnvelope

		err = json.Unmarshal(js, &got)
		if err != nil {
			t.Fatal(err)
		}

		want := pr.Answer
		diff := diffCalls(append(got.Calls, got.Suppressed...), append(want.Calls, want.Suppressed...))
		think := reflect.DeepEqual(got.Reasoning, want.Reasoning)
		margin := minMargin(turn.a, 0, slices.Index(turn.a.Tokens, d.thinkEnd))

		t.Logf("%-18s confidence %.4f engine %.4f  reasoning equal %-5v margin %.3f  %q", pr.ID, got.Confidence,
			want.Confidence, think, margin, diff)

		if why, listed := grammarCallsDiffer[pr.ID]; listed == (diff == "") {
			t.Errorf("%s: calls %q (listed: %q)", pr.ID, diff, why)
		}

		// A truncated answer has no reasoning.
		truncated := want.ErrorCode != nil && *want.ErrorCode == "truncated"
		if near := grammarReasoningNearTie[pr.ID]; !truncated && (near == think || near && margin > 0.06) {
			t.Errorf("%s: reasoning equal %v at margin %.3f (listed as a near-tie: %v)", pr.ID, think, margin, near)
		}
	}
}
