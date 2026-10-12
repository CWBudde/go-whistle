package toolcall

import (
	"encoding/json"
	"reflect"
	"testing"
)

// engineNormalise is the fixture of `just oracle-text normalise`.
const engineNormalise = "../../testdata/text/engine/normalise.json"

// normaliseFile is testdata/text/engine/normalise.json (tools/oracle/text
// normalise.go).
type normaliseFile struct {
	System string `json:"system"`
	Probes []struct {
		ID    string `json:"id"`
		Tools string `json:"tools"`
		Init  int    `json:"init"`
		Error string `json:"error"`
	} `json:"probes"`
	Answers []struct {
		Probe  string `json:"probe"`
		Query  string `json:"query"`
		Answer struct {
			Calls []call `json:"function_calls"`
		} `json:"answer"`
	} `json:"answers"`
}

// TestNormaliseMatchesEngine: on every tool set that needs normalising
// (OpenAI-style entries, camelCase, triggers, extra keys, nested schemas,
// colliding names, odd names), the static prefix of the normalised tools has
// needle_init's count (docs/FINDINGS.md F-050).
func TestNormaliseMatchesEngine(t *testing.T) {
	p := loadTokenizer(t)

	var fx normaliseFile

	readJSON(t, engineNormalise, &fx)

	if len(fx.Probes) < 50 {
		t.Fatalf("%d probes", len(fx.Probes))
	}

	changed := 0

	for _, pr := range fx.Probes {
		if pr.Error != "" {
			t.Errorf("%s: needle_init failed: %s", pr.ID, pr.Error)

			continue
		}

		n, err := Normalise([]byte(pr.Tools))
		if err != nil {
			t.Errorf("%s: %v", pr.ID, err)

			continue
		}

		if got := len(StaticPrefix(p, fx.System, n.JSON)); got != pr.Init {
			t.Errorf("%s: static prefix %d tokens, needle_init %d\n  %s", pr.ID, got, pr.Init, n.JSON)
		}

		plain, err := ToolsJSON([]byte(pr.Tools))
		if err != nil {
			t.Fatal(err)
		}

		if len(StaticPrefix(p, fx.System, plain)) != pr.Init {
			changed++
		}
	}

	if changed < len(fx.Probes)/2 {
		t.Errorf("only %d of %d probes need normalising; the fixture does not test it", changed, len(fx.Probes))
	}

	t.Logf("%d probes, %d whose count normalising changes", len(fx.Probes), changed)
}

// TestNormaliseAliasesMatchEngine: the engine's calls over camelCase,
// OpenAI-style and colliding tools carry the caller's tool and argument
// names, and the alias table maps the model's names (the normalised
// tools') back to exactly those.
func TestNormaliseAliasesMatchEngine(t *testing.T) {
	var fx normaliseFile

	readJSON(t, engineNormalise, &fx)

	tools := map[string]string{}
	for _, pr := range fx.Probes {
		tools[pr.ID] = pr.Tools
	}

	if len(fx.Answers) == 0 {
		t.Fatal("no answers")
	}

	for _, a := range fx.Answers {
		n, err := Normalise([]byte(tools[a.Probe]))
		if err != nil {
			t.Fatal(err)
		}

		if len(a.Answer.Calls) != 1 {
			t.Errorf("%s %q: %d calls", a.Probe, a.Query, len(a.Answer.Calls))

			continue
		}

		c := a.Answer.Calls[0]

		model := ""

		for m, caller := range n.names {
			if caller == c.Name {
				model = m
			}
		}

		if model == "" {
			t.Errorf("%s %q: the engine calls %q, not one of the caller's names %v", a.Probe, a.Query, c.Name, n.names)

			continue
		}

		// The model writes the arguments under the normalised names.
		modelArgs := map[string]any{}
		for k, v := range c.Arguments {
			modelArgs[Snake(k)] = v
		}

		in, err := json.Marshal(modelArgs)
		if err != nil {
			t.Fatal(err)
		}

		out, err := n.CallerArgs(model, in)
		if err != nil {
			t.Fatal(err)
		}

		var got map[string]any

		err = json.Unmarshal(out, &got)
		if err != nil {
			t.Fatal(err)
		}

		if n.CallerName(model) != c.Name || !reflect.DeepEqual(got, c.Arguments) {
			t.Errorf("%s %q: alias table gives %s %v, the engine %s %v", a.Probe, a.Query,
				n.CallerName(model), got, c.Name, c.Arguments)
		}
	}
}

func TestSnake(t *testing.T) {
	for in, want := range map[string]string{
		"setTimer": "set_timer", "SetTimer": "set_timer", "HTTPServer": "http_server", "getURL2": "get_url2",
		"getHTTPResponseCode": "get_http_response_code", "XMLHttpRequest": "xml_http_request", "IOError": "io_error",
		"turnOnTV": "turn_on_tv", "get2FACode": "get2_fa_code", "v2Api": "v2_api", "a1B2c3": "a1_b2c3",
		"ABC": "abc", "already_snake": "already_snake", "Mixed_Case": "mixed_case", "set timer": "set_timer",
		"_private": "private", "__zoom_in__": "zoom_in", "kebab-case-name": "kebab_case_name",
		"send.message": "send_message", "set-Timer": "set_timer", "foo__bar": "foo_bar",
		"ÄpfelZählen": "ÄpfelZählen", "naïveBayes": "naïveBayes", "___": "___", "": "",
	} {
		if got := Snake(in); got != want {
			t.Errorf("Snake(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNormalise(t *testing.T) {
	for name, c := range map[string]struct{ in, want string }{
		"unchanged": {
			`[{"name": "set_timer", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}]`,
			`[{"name":"set_timer","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}]`,
		},
		"openai, triggers dropped": {
			`[{"type":"function","function":{"name":"t","description":"d","triggers":["x"]},"extra":1}]`,
			`[{"name":"t","description":"d"}]`,
		},
		"camel names and required": {
			`[{"name":"setIt","parameters":{"type":"object","properties":{"maxCount":{"type":"object",` +
				`"properties":{"innerKey":{}}}},"required":["maxCount","other"]}}]`,
			`[{"name":"set_it","parameters":{"type":"object","properties":{"max_count":{"type":"object",` +
				`"properties":{"inner_key":{}}}},"required":["max_count","other"]}}]`,
		},
		"colliding tools": {
			`[{"name":"setTimer"},{"name":"set_timer"},{"name":"setTimer"},{"name":"other"}]`,
			`[{"name":"set_timer_1"},{"name":"set_timer_2"},{"name":"set_timer_1"},{"name":"other"}]`,
		},
		"identical tools": {`[{"name":"set_timer"},{"name":"set_timer"}]`, `[{"name":"set_timer"},{"name":"set_timer"}]`},
		"colliding properties": {
			`[{"name":"t","parameters":{"properties":{"maxCount":1,"max_count":2,"MaxCount":3},"required":["max_count"]}}]`,
			`[{"name":"t","parameters":{"properties":{"max_count":1,"max_count_1":2,"max_count_2":3},"required":["max_count_1"]}}]`,
		},
		"defs and items list untouched": {
			`[{"name":"t","parameters":{"properties":{"a":{"items":[{"properties":{"innerKey":{}}}]}},` +
				`"$defs":{"MyThing":{"properties":{"innerKey":{}}}}}}]`,
			`[{"name":"t","parameters":{"properties":{"a":{"items":[{"properties":{"innerKey":{}}}]}},` +
				`"$defs":{"MyThing":{"properties":{"innerKey":{}}}}}}]`,
		},
	} {
		n, err := Normalise([]byte(c.in))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		if n.JSON != c.want {
			t.Errorf("%s:\n got  %s\n want %s", name, n.JSON, c.want)
		}
	}

	for _, bad := range []string{`{}`, `[`, "[\xff]", `[] x`} {
		_, err := Normalise([]byte(bad))
		if err == nil {
			t.Errorf("Normalise(%q) succeeded", bad)
		}
	}
}

func TestCallerArgs(t *testing.T) {
	n, err := Normalise([]byte(`[{"name":"makeList","parameters":{"type":"object","properties":{` +
		`"listName":{"type":"string"},"itemsToAdd":{"type":"array","items":{"type":"object","properties":{` +
		`"itemName":{"type":"string"}}}},"pick":{"anyOf":[{"type":"object","properties":{"byKey":{}}}]},` +
		`"extra":{"type":"object","additionalProperties":{"type":"object","properties":{"deepKey":{}}}}}}}]`))
	if err != nil {
		t.Fatal(err)
	}

	if got := n.CallerName("make_list"); got != "makeList" {
		t.Errorf("CallerName = %q", got)
	}

	if got := n.CallerName("unknown"); got != "unknown" {
		t.Errorf("CallerName(unknown) = %q", got)
	}

	out, err := n.CallerArgs("make_list", []byte(`{"list_name":"a","items_to_add":[{"item_name":"b"}],`+
		`"pick":{"by_key":1},"extra":{"anyName":{"deep_key":2}},"unknown_key":3}`))
	if err != nil {
		t.Fatal(err)
	}

	var got, want any

	_ = json.Unmarshal(out, &got)
	_ = json.Unmarshal([]byte(`{"listName":"a","itemsToAdd":[{"itemName":"b"}],"pick":{"byKey":1},`+
		`"extra":{"anyName":{"deepKey":2}},"unknown_key":3}`), &want)

	if !reflect.DeepEqual(got, want) {
		t.Errorf("CallerArgs = %s", out)
	}

	_, err = n.CallerArgs("make_list", []byte(`{`))
	if err == nil {
		t.Error("CallerArgs accepted bad JSON")
	}

	// PR #61 review: integers past 2^53 keep their digits, and the items of
	// an array schema inside a branch are renamed back.
	n, err = Normalise([]byte(`[{"name":"t","parameters":{"properties":{"orderId":{"type":"integer"},` +
		`"rows":{"anyOf":[{"type":"array","items":{"properties":{"itemName":{}}}}]}}}}]`))
	if err != nil {
		t.Fatal(err)
	}

	out, err = n.CallerArgs("t", []byte(`{"order_id":12345678901234567890,"rows":[{"item_name":1.50}]}`))
	if err != nil {
		t.Fatal(err)
	}

	if want := `{"orderId":12345678901234567890,"rows":[{"itemName":1.50}]}`; string(out) != want {
		t.Errorf("CallerArgs = %s, want %s", out, want)
	}

	for _, bad := range []string{`{"order_id":1}]`, `{"order_id":1} {}`, `{"order_id":1}x`} {
		_, err = n.CallerArgs("t", []byte(bad))
		if err == nil {
			t.Errorf("CallerArgs accepted %s", bad)
		}
	}

	// Nested branches.
	n, err = Normalise([]byte(`[{"name":"t","parameters":{"properties":{"pick":{"anyOf":[{"allOf":[` +
		`{"properties":{"innerKey":{"type":"array","items":{"oneOf":[{"anyOf":[{"type":"array",` +
		`"items":{"properties":{"deepKey":{}}}}]}]}}}}]}]}}}}]`))
	if err != nil {
		t.Fatal(err)
	}

	out, err = n.CallerArgs("t", []byte(`{"pick":{"inner_key":[[{"deep_key":1}]]}}`))
	if err != nil {
		t.Fatal(err)
	}

	if want := `{"pick":{"innerKey":[[{"deepKey":1}]]}}`; string(out) != want {
		t.Errorf("CallerArgs = %s, want %s", out, want)
	}
}

// TestNormaliseUniqueNames: a generated suffix never repeats a name the
// tools or properties already have (PR #61 review).
func TestNormaliseUniqueNames(t *testing.T) {
	n, err := Normalise([]byte(`[{"name":"setTimer"},{"name":"set_timer"},{"name":"set_timer_1"},` +
		`{"name":"t","parameters":{"properties":{"fooBar":1,"foo_bar":2,"foo_bar_1":3}}}]`))
	if err != nil {
		t.Fatal(err)
	}

	want := `[{"name":"set_timer_1_1"},{"name":"set_timer_2"},{"name":"set_timer_1"},` +
		`{"name":"t","parameters":{"properties":{"foo_bar":1,"foo_bar_1":2,"foo_bar_1_1":3}}}]`
	if n.JSON != want {
		t.Errorf("got  %s\nwant %s", n.JSON, want)
	}

	for model, caller := range map[string]string{"set_timer_1_1": "setTimer", "set_timer_2": "set_timer", "set_timer_1": "set_timer_1"} {
		if got := n.CallerName(model); got != caller {
			t.Errorf("CallerName(%s) = %s, want %s", model, got, caller)
		}
	}

	// Two collision groups whose suffixes meet: every tool keeps a name of
	// its own.
	n, err = Normalise([]byte(`[{"name":"x"},{"name":"X"},{"name":"x_1"},{"name":"x-1"},{"name":"X"}]`))
	if err != nil {
		t.Fatal(err)
	}

	if want := `[{"name":"x_1_1"},{"name":"x_2"},{"name":"x_1_1_1"},{"name":"x_1_2"},{"name":"x_2"}]`; n.JSON != want {
		t.Errorf("got  %s\nwant %s", n.JSON, want)
	}

	for model, caller := range map[string]string{"x_1_1": "x", "x_2": "X", "x_1_1_1": "x_1", "x_1_2": "x-1"} {
		if got := n.CallerName(model); got != caller {
			t.Errorf("CallerName(%s) = %s, want %s", model, got, caller)
		}
	}
}

// TestCallerArgsBranches: additionalProperties inside a branch, and items
// from more than one array branch, map back (PR #61 review).
func TestCallerArgsBranches(t *testing.T) {
	n, err := Normalise([]byte(`[{"name":"t","parameters":{"properties":{` +
		`"bag":{"anyOf":[{"type":"string"},{"additionalProperties":{"properties":{"innerKey":{}}}}]},` +
		`"rows":{"anyOf":[{"type":"array","items":{"properties":{"firstKey":{}}}},` +
		`{"type":"array","items":{"properties":{"secondKey":{}}}}]}}}}]`))
	if err != nil {
		t.Fatal(err)
	}

	out, err := n.CallerArgs("t", []byte(`{"bag":{"anyName":{"inner_key":1}},"rows":[{"second_key":2},{"first_key":3}]}`))
	if err != nil {
		t.Fatal(err)
	}

	if want := `{"bag":{"anyName":{"innerKey":1}},"rows":[{"secondKey":2},{"firstKey":3}]}`; string(out) != want {
		t.Errorf("CallerArgs = %s, want %s", out, want)
	}

	// Branches that suffix one name differently: the branch that names the
	// whole object decides (PR #62 review).
	n, err = Normalise([]byte(`[{"name":"t","parameters":{"properties":{"rows":{"anyOf":[` +
		`{"type":"array","items":{"properties":{"fooBar":{}}}},` +
		`{"type":"array","items":{"properties":{"foo_bar":{},"fooBar":{}}}}]}}}}]`))
	if err != nil {
		t.Fatal(err)
	}

	out, err = n.CallerArgs("t", []byte(`{"rows":[{"foo_bar":1,"foo_bar_1":2},{"foo_bar":3}]}`))
	if err != nil {
		t.Fatal(err)
	}

	if want := `{"rows":[{"fooBar":2,"foo_bar":1},{"fooBar":3}]}`; string(out) != want {
		t.Errorf("CallerArgs = %s, want %s", out, want)
	}
}
