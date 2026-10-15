package toolcall

import "testing"

// TestDumps pins Dumps against Python's json.dumps(json.loads(raw)) with
// the default arguments, the outputs as Python 3.14 prints them: ", " and
// ": ", every rune past '~' and every control character escaped (an astral
// one as a surrogate pair, in lower case), floats in repr and integers in
// full.
func TestDumps(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{`[{"error":"unknown tool: x"}]`, `[{"error": "unknown tool: x"}]`},
		{
			`[{"a":1,"b":[1.5,true,null]},"héllo 😀 <&>` + "\x7f" + `\u0001\n"]`,
			`[{"a": 1, "b": [1.5, true, null]}, "h\u00e9llo \ud83d\ude00 <&>\u007f\u0001\n"]`,
		},
		{`[1e2, 1e16, -0.0, 12345678901234567890]`, `[100.0, 1e+16, -0.0, 12345678901234567890]`},
		{` {} `, `{}`},
		{`[]`, `[]`},
		{`[{"nested":{"k":"v"}}]`, `[{"nested": {"k": "v"}}]`},
		{`{"b":1,"a":2,"b":3}`, `{"b": 3, "a": 2}`},
	} {
		got, err := Dumps([]byte(c.in))
		if err != nil {
			t.Errorf("Dumps(%s): %v", c.in, err)

			continue
		}

		if got != c.want {
			t.Errorf("Dumps(%s) = %s, want %s", c.in, got, c.want)
		}
	}

	for _, bad := range []string{``, `[`, `[1] 2`, "[\"\xff\"]", `["\ud800"]`} {
		_, err := Dumps([]byte(bad))
		if err == nil {
			t.Errorf("Dumps(%q) accepted", bad)
		}
	}
}
