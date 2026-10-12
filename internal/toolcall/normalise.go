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

	// Ignored are the triggers that do not compile, "tool: regex", which the
	// engine skips with "needle: ignoring invalid trigger for tool: regex"
	// (docs/FINDINGS.md F-054).
	Ignored []string

	names    map[string]string       // the model's tool name -> the caller's
	args     map[string]*schemaAlias // the model's tool name -> its parameters' renames
	params   map[string]*object      // the model's tool name -> its parameters as rendered
	tools    []string                // the model's tool names, in declaration order
	triggers []toolTriggers          // in declaration order
}

// toolTriggers are one tool's compiled triggers, under the model's name.
type toolTriggers struct {
	tool string
	res  []*regexp.Regexp
}

// Normalise is the engine's preparation of the tools raw (a JSON array)
// before needle_init renders them, measured by needle_init's counts and the
// engine's answers (docs/FINDINGS.md F-050):
//
//   - an entry with a "function" object is that object (OpenAI's
//     {"type":"function","function":{…}}, whatever its type and other keys);
//   - a tool's top-level "triggers" is dropped (Triggered reads it); every
//     other key stays;
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

	n := &Normalised{names: map[string]string{}, args: map[string]*schemaAlias{}, params: map[string]*object{}}

	tools := make([]any, len(list))
	for i, t := range list {
		tools[i] = unwrap(t)
	}

	for i, model := range toolNames(tools) {
		tool, ok := tools[i].(*object)
		if !ok {
			continue
		}

		if model != "" {
			n.addTriggers(model, tool.vals["triggers"])
			n.tools = append(n.tools, model)
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
				n.args[model], n.params[model] = alias, params
			}
		}

		tools[i] = tool
	}

	var b strings.Builder

	writeValue(&b, tools)
	n.JSON = b.String()

	return n, nil
}

// compositions are the keywords whose branch lists compose a schema: one
// branch of each anyOf and oneOf applies, every allOf branch.
var compositions = []string{"anyOf", "oneOf", allOf}

const allOf = "allOf"

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

	// A suffixed name that a tool already has as its snake name, or that
	// another group's suffix took, gets one more (PR #61 review); the
	// engine's choice is not measured.
	assigned := map[string]bool{}
	byCaller := map[string]string{}

	for i, t := range tools {
		if out[i] == "" {
			continue
		}

		caller, _ := t.(*object).vals["name"].(string)
		if m, ok := byCaller[caller]; ok {
			out[i] = m

			continue
		}

		if names := callers[out[i]]; len(names) > 1 {
			k := strconv.Itoa(slices.Index(names, caller) + 1)

			out[i] += "_" + k
			for _, natural := callers[out[i]]; natural || assigned[out[i]]; _, natural = callers[out[i]] {
				out[i] += "_" + k
			}
		}

		assigned[out[i]] = true
		byCaller[caller] = out[i]
	}

	return out
}

// schemaAlias maps a schema's renamed keys back: props from the model's
// property names to the caller's, and the nested schemas by the model's
// names. allOf holds the branches that all apply; alts the anyOf and the
// oneOf list, of each of which one branch applies.
type schemaAlias struct {
	props map[string]string
	sub   map[string]*schemaAlias
	items *schemaAlias
	extra *schemaAlias // additionalProperties
	allOf []*schemaAlias
	alts  [][]*schemaAlias
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

	for _, key := range compositions {
		if list, ok := o.vals[key].([]any); ok {
			renamed, group := branchList(list)
			o = o.with(key, renamed)

			if key == allOf {
				a.allOf = group
			} else {
				a.alts = append(a.alts, group)
			}
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

// branchList renames the schemas of an anyOf, oneOf or allOf list and
// returns their aliases.
func branchList(list []any) ([]any, []*schemaAlias) {
	out := make([]any, len(list))

	var group []*schemaAlias

	for i, b := range list {
		out[i] = b

		if bo, ok := b.(*object); ok {
			var ba *schemaAlias

			out[i], ba = schema(bo)
			group = append(group, ba)
		}
	}

	return out, group
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
// integers; PR #61 review). Under allOf every branch names keys; of an anyOf
// or oneOf list one branch does, chosen by applicable (a port-side rule,
// docs/FINDINGS.md F-050).
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

// Triggered are the model's names of the tools with a trigger that matches
// input, in declaration order.
func (n *Normalised) Triggered(input string) []string {
	var tools []string

	for _, tt := range n.triggers {
		if slices.ContainsFunc(tt.res, func(re *regexp.Regexp) bool { return re.MatchString(input) }) {
			tools = append(tools, tt.tool)
		}
	}

	return tools
}

// addTriggers compiles a tool's "triggers", a list of regexes matched
// case-insensitively (llms.txt; docs/FINDINGS.md F-054). Entries that are
// not strings or do not compile (in Go's RE2 syntax) go to Ignored.
func (n *Normalised) addTriggers(tool string, v any) {
	list, ok := v.([]any)
	if !ok {
		return
	}

	tt := toolTriggers{tool: tool}

	for _, e := range list {
		s, ok := e.(string)
		if !ok {
			n.Ignored = append(n.Ignored, fmt.Sprintf("%s: %v", tool, e))

			continue
		}

		re, err := regexp.Compile("(?i)" + s)
		if err != nil {
			n.Ignored = append(n.Ignored, tool+": "+s)

			continue
		}

		tt.res = append(tt.res, re)
	}

	if tt.res != nil {
		n.triggers = append(n.triggers, tt)
	}
}

func callerValue(v any, a *schemaAlias) any {
	if a == nil {
		return v
	}

	switch t := v.(type) {
	case map[string]any:
		nodes := a.applicable(t)
		out := make(map[string]any, len(t))

		for k, x := range t {
			caller, sub := lookup(nodes, k)
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

// applicable are the schema nodes that name obj's keys (docs/FINDINGS.md
// F-050, port-side): a itself, every allOf branch, and of each anyOf and
// oneOf list the one branch whose own applicable nodes name most of obj's
// keys, then take most of the rest with an additionalProperties schema,
// the first on a tie; nested branches the same way. Values are not
// checked against the branches' types.
func (a *schemaAlias) applicable(obj map[string]any) []*schemaAlias {
	nodes := make([]*schemaAlias, 1, 1+len(a.allOf)+len(a.alts))
	nodes[0] = a

	for _, b := range a.allOf {
		nodes = append(nodes, b.applicable(obj)...)
	}

	for _, group := range a.alts {
		var (
			best         []*schemaAlias
			named, taken = -1, -1
		)

		for _, b := range group {
			cand := b.applicable(obj)
			if n, t := cover(cand, obj); n > named || n == named && t > taken {
				best, named, taken = cand, n, t
			}
		}

		nodes = append(nodes, best...)
	}

	return nodes
}

// cover is how many of obj's keys the nodes' properties name, and how many
// of the others an additionalProperties schema of theirs takes.
func cover(nodes []*schemaAlias, obj map[string]any) (named, taken int) {
	extra := slices.ContainsFunc(nodes, func(s *schemaAlias) bool { return s.extra != nil })

	for k := range obj {
		if slices.ContainsFunc(nodes, func(s *schemaAlias) bool { _, ok := s.props[k]; return ok }) {
			named++
		} else if extra {
			taken++
		}
	}

	return named, taken
}

// lookup is the caller's name for the model's key k and its schema: the
// name from the first node whose properties name it, the schemas of every
// node that does (they all apply, as allOf branches; PR #63 review); then
// from the first additionalProperties schema, else k unchanged.
func lookup(nodes []*schemaAlias, k string) (string, *schemaAlias) {
	var (
		caller string
		subs   []*schemaAlias
		named  bool
	)

	for _, s := range nodes {
		c, ok := s.props[k]
		if !ok {
			continue
		}

		if !named {
			caller, named = c, true
		}

		if sub := s.sub[k]; sub != nil {
			subs = append(subs, sub)
		}
	}

	switch {
	case len(subs) > 1:
		return caller, &schemaAlias{props: map[string]string{}, sub: map[string]*schemaAlias{}, allOf: subs}
	case named && len(subs) == 1:
		return caller, subs[0]
	case named:
		return caller, nil
	}

	for _, s := range nodes {
		if s.extra != nil {
			return k, s.extra
		}
	}

	return k, nil
}

// itemsAlias is the alias of an array's elements: its "items" and those of
// its allOf branches all apply, and of each anyOf and oneOf list's branches
// with items one is chosen per element, as for objects; nil without any.
func (a *schemaAlias) itemsAlias() *schemaAlias {
	it := &schemaAlias{props: map[string]string{}, sub: map[string]*schemaAlias{}}

	if a.items != nil {
		it.allOf = append(it.allOf, a.items)
	}

	for _, b := range a.allOf {
		if bi := b.itemsAlias(); bi != nil {
			it.allOf = append(it.allOf, bi)
		}
	}

	for _, group := range a.alts {
		var alts []*schemaAlias

		for _, b := range group {
			if bi := b.itemsAlias(); bi != nil {
				alts = append(alts, bi)
			}
		}

		if alts != nil {
			it.alts = append(it.alts, alts)
		}
	}

	if it.allOf == nil && it.alts == nil {
		return nil
	}

	return it
}
