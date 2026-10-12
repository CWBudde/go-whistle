package toolcall

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Normalised is a tool set as the engine renders it, and the way back to the
// caller's names (docs/FINDINGS.md F-050).
type Normalised struct {
	// JSON is the rendered tools, the tools argument of Render and
	// StaticPrefix.
	JSON string

	names map[string]string       // the model's tool name -> the caller's
	args  map[string]*schemaAlias // the model's tool name -> its parameters' renames
}

// Normalise is the engine's preparation of the tools raw (a JSON array)
// before needle_init renders them, measured by needle_init's counts and the
// engine's answers (docs/FINDINGS.md F-050):
//
//   - an entry with a "function" object is that object (OpenAI's
//     {"type":"function","function":{…}}, whatever its type and other keys);
//   - a tool's top-level "triggers" is dropped; every other key stays;
//   - tool and property names become snake_case (Snake). A snake name that
//     tools with different names share gets _1, _2, … on each of them in
//     declaration order; within one object the first property keeps it and
//     the next get _1, _2, …;
//   - properties are renamed in "parameters" and in every property schema,
//     in "items" when it is an object, in the anyOf, oneOf and allOf
//     branches and in an "additionalProperties" schema; "required" follows
//     the renames. "$defs" is left as is.
//
// Without any of these the result is ToolsJSON's.
func Normalise(raw []byte) (*Normalised, error) {
	if !utf8.Valid(raw) {
		return nil, errors.New("toolcall: tools: not valid UTF-8")
	}

	p := &pyParser{s: raw}

	v, err := p.value()
	if err != nil {
		return nil, fmt.Errorf("toolcall: tools: %w", err)
	}

	p.space()

	if p.i != len(p.s) {
		return nil, fmt.Errorf("toolcall: tools: data after the JSON value at byte %d", p.i)
	}

	list, ok := v.([]any)
	if !ok {
		return nil, errors.New("toolcall: tools: not a JSON array")
	}

	n := &Normalised{names: map[string]string{}, args: map[string]*schemaAlias{}}

	tools := make([]any, len(list))
	for i, t := range list {
		tools[i] = unwrap(t)
	}

	for i, model := range toolNames(tools) {
		tool, ok := tools[i].(*object)
		if !ok {
			continue
		}

		tool = tool.without("triggers")

		if caller, ok := tool.vals["name"].(string); ok {
			tool = tool.with("name", model)
			n.names[model] = caller
		}

		if params, ok := tool.vals["parameters"].(*object); ok {
			params, alias := schema(params)
			tool = tool.with("parameters", params)

			if model != "" {
				n.args[model] = alias
			}
		}

		tools[i] = tool
	}

	var b strings.Builder

	writeValue(&b, tools)
	n.JSON = b.String()

	return n, nil
}

// unwrap is an OpenAI-style entry's "function" object, or the entry.
func unwrap(t any) any {
	if o, ok := t.(*object); ok {
		if f, ok := o.vals["function"].(*object); ok {
			return f
		}
	}

	return t
}

// toolNames is each tool's name as the model sees it, "" for an entry
// without a string name. Tools whose different names share a snake name
// all get a suffix, _1, _2, … in declaration order; tools of one name
// share theirs.
func toolNames(tools []any) []string {
	out := make([]string, len(tools))
	callers := map[string][]string{} // snake name -> its distinct caller names, in order

	for i, t := range tools {
		o, ok := t.(*object)
		if !ok {
			continue
		}

		name, ok := o.vals["name"].(string)
		if !ok {
			continue
		}

		s := Snake(name)
		out[i] = s

		if !slices.Contains(callers[s], name) {
			callers[s] = append(callers[s], name)
		}
	}

	for i, t := range tools {
		if out[i] == "" {
			continue
		}

		names := callers[out[i]]
		if len(names) < 2 {
			continue
		}

		caller, _ := t.(*object).vals["name"].(string)
		for k, c := range names {
			if c == caller {
				out[i] += "_" + strconv.Itoa(k+1)
				// A suffixed name another tool already has gets one more
				// (PR #61 review); the engine's choice is not measured.
				for _, ok := callers[out[i]]; ok; _, ok = callers[out[i]] {
					out[i] += "_" + strconv.Itoa(k+1)
				}
			}
		}
	}

	return out
}

// schemaAlias maps a schema's renamed keys back: props from the model's
// property names to the caller's, and the nested schemas by the model's
// names.
type schemaAlias struct {
	props    map[string]string
	sub      map[string]*schemaAlias
	items    *schemaAlias
	extra    *schemaAlias // additionalProperties
	branches []*schemaAlias
}

// schema renames the properties of a JSON schema object and of the schemas
// nested in it (Normalise), and returns the renamed copy and the way back.
func schema(o *object) (*object, *schemaAlias) {
	a := &schemaAlias{props: map[string]string{}, sub: map[string]*schemaAlias{}}

	if props, ok := o.vals["properties"].(*object); ok {
		renamed, back := a.properties(props)
		o = o.with("properties", renamed)

		if req, ok := o.vals["required"].([]any); ok {
			o = o.with("required", renameAll(req, back))
		}
	}

	if items, ok := o.vals["items"].(*object); ok {
		var it *object

		it, a.items = schema(items)
		o = o.with("items", it)
	}

	if extra, ok := o.vals["additionalProperties"].(*object); ok {
		var ex *object

		ex, a.extra = schema(extra)
		o = o.with("additionalProperties", ex)
	}

	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if list, ok := o.vals[key].([]any); ok {
			o = o.with(key, a.branchList(list))
		}
	}

	return o, a
}

// properties renames a "properties" object's keys (the first of a snake
// name keeps it, the next get _1, _2, …, skipping names already taken) and
// the schemas under them, and returns the renamed object and the caller's
// names mapped to the model's.
func (a *schemaAlias) properties(props *object) (*object, map[string]string) {
	renamed := &object{vals: map[string]any{}}
	back := map[string]string{}

	for _, k := range props.keys {
		model := Snake(k)
		for n := 1; slices.Contains(renamed.keys, model); n++ {
			model = Snake(k) + "_" + strconv.Itoa(n)
		}

		v := props.vals[k]
		if sub, ok := v.(*object); ok {
			v, a.sub[model] = schema(sub)
		}

		renamed.keys = append(renamed.keys, model)
		renamed.vals[model] = v
		a.props[model] = k
		back[k] = model
	}

	return renamed, back
}

// branchList renames the schemas of an anyOf, oneOf or allOf list.
func (a *schemaAlias) branchList(list []any) []any {
	out := make([]any, len(list))

	for i, b := range list {
		out[i] = b

		if bo, ok := b.(*object); ok {
			var ba *schemaAlias

			out[i], ba = schema(bo)
			a.branches = append(a.branches, ba)
		}
	}

	return out
}

// renameAll is a "required" list with each name the properties renamed
// replaced; other entries stay.
func renameAll(req []any, back map[string]string) []any {
	out := make([]any, len(req))

	for i, r := range req {
		out[i] = r

		if s, ok := r.(string); ok {
			if m, ok := back[s]; ok {
				out[i] = m
			}
		}
	}

	return out
}

// with is a copy of o with key set to v, in key's place.
func (o *object) with(key string, v any) *object {
	c := &object{keys: o.keys, vals: make(map[string]any, len(o.vals))}
	maps.Copy(c.vals, o.vals)

	if _, ok := c.vals[key]; !ok {
		c.keys = append(append([]string{}, o.keys...), key)
	}

	c.vals[key] = v

	return c
}

// without is a copy of o without key.
func (o *object) without(key string) *object {
	if _, ok := o.vals[key]; !ok {
		return o
	}

	c := &object{vals: make(map[string]any, len(o.vals))}

	for _, k := range o.keys {
		if k != key {
			c.keys = append(c.keys, k)
			c.vals[k] = o.vals[k]
		}
	}

	return c
}

var (
	snakeWord  = regexp.MustCompile(`(.)([A-Z][a-z]+)`)
	snakeUpper = regexp.MustCompile(`([a-z0-9])([A-Z])`)
	snakeSep   = regexp.MustCompile(`[^a-zA-Z0-9]+`)
)

// Snake is the engine's snake_case of a tool or property name: a capital
// starting a lower-case word, and one after a lower-case letter or a digit,
// get an underscore before them; every run of other characters than ASCII
// letters and digits becomes one underscore; underscores at the ends go,
// and the name is lower-cased ("getHTTPResponseCode" → "get_http_response_code",
// "set timer" → "set_timer", "_private" → "private"). A name with non-ASCII
// characters, or one that would become empty, is left as it is
// (docs/FINDINGS.md F-050). Separators that tokenise as "_" does ("-", ".",
// "/", ":") are not identified by needle_init's counts; they follow the
// porting notes' "snake_case".
func Snake(name string) string {
	for i := range len(name) {
		if name[i] >= utf8.RuneSelf {
			return name
		}
	}

	s := snakeWord.ReplaceAllString(name, "${1}_${2}")
	s = snakeUpper.ReplaceAllString(s, "${1}_${2}")
	s = strings.ToLower(strings.Trim(snakeSep.ReplaceAllString(s, "_"), "_"))

	if s == "" {
		return name
	}

	return s
}

// CallerName is the caller's name of the tool the model calls name; a name
// the tools do not have comes back as it is.
func (n *Normalised) CallerName(name string) string {
	if c, ok := n.names[name]; ok {
		return c
	}

	return name
}

// CallerArgs is a call's arguments (a JSON object) under the caller's
// property names: each key the model's schema renamed gets the caller's
// name back, nested objects and arrays included. Keys the schema does not
// know stay, and numbers keep their literals (no float64 rounding of large
// integers; PR #61 review).
func (n *Normalised) CallerArgs(name string, args []byte) ([]byte, error) {
	var v any

	dec := json.NewDecoder(bytes.NewReader(args))
	dec.UseNumber()

	err := dec.Decode(&v)
	if err == nil {
		// Anything after the value, a stray ']' included, is an error.
		var extra any
		if !errors.Is(dec.Decode(&extra), io.EOF) {
			err = errors.New("data after the arguments")
		}
	}

	if err != nil {
		return nil, fmt.Errorf("toolcall: arguments: %w", err)
	}

	out, err := json.Marshal(callerValue(v, n.args[name]))
	if err != nil {
		return nil, fmt.Errorf("toolcall: arguments: %w", err)
	}

	return out, nil
}

func callerValue(v any, a *schemaAlias) any {
	if a == nil {
		return v
	}

	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))

		for k, x := range t {
			caller, sub := a.lookup(k)
			out[caller] = callerValue(x, sub)
		}

		return out
	case []any:
		items := a.itemsAlias()

		out := make([]any, len(t))
		for i, x := range t {
			out[i] = callerValue(x, items)
		}

		return out
	default:
		return v
	}
}

// lookup is the caller's name for the model's key k and its schema's
// alias: from the properties, then the branches, then additionalProperties.
func (a *schemaAlias) lookup(k string) (string, *schemaAlias) {
	if c, sub, ok := a.find(k); ok {
		return c, sub
	}

	return k, a.extra
}

// find looks k up in the properties and then, depth first, in the anyOf,
// oneOf and allOf branches, nested ones included (PR #61 review).
func (a *schemaAlias) find(k string) (string, *schemaAlias, bool) {
	if c, ok := a.props[k]; ok {
		return c, a.sub[k], true
	}

	for _, b := range a.branches {
		if c, sub, ok := b.find(k); ok {
			return c, sub, true
		}
	}

	return "", nil, false
}

// itemsAlias is the alias of an array's item schema: its own "items", or
// that of the first anyOf, oneOf or allOf branch, nested ones included,
// that has one (PR #61 review).
func (a *schemaAlias) itemsAlias() *schemaAlias {
	if a.items != nil {
		return a.items
	}

	for _, b := range a.branches {
		if it := b.itemsAlias(); it != nil {
			return it
		}
	}

	return nil
}
