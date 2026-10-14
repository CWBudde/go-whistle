package text

import "testing"

// The speed of the packed model against the dequantised float model on
// needle3.cact (docs/FINDINGS.md F-059): Forward over the timer_en prompt
// and its completion, and one Extend step after it.
//
//	go test -run '^$' -bench 'Forward|ExtendStep' ./internal/text/
func BenchmarkForward(b *testing.B) {
	requireFile(b, forwardFixture)

	tokens := openFixture(b).ints(b, "timer_en.tokens")

	for _, mode := range benchModes {
		b.Run(mode.name, func(b *testing.B) {
			m := mode.load(b)

			for b.Loop() {
				_, err := m.Forward(tokens, false)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkExtendStep feeds one token to a fork of a state that holds the
// timer_en prompt and completion: one decoding step at that length.
func BenchmarkExtendStep(b *testing.B) {
	requireFile(b, forwardFixture)

	tokens := openFixture(b).ints(b, "timer_en.tokens")
	prefix, next := tokens[:len(tokens)-1], tokens[len(tokens)-1:]

	for _, mode := range benchModes {
		b.Run(mode.name, func(b *testing.B) {
			m := mode.load(b)
			s := m.NewState()

			_, err := m.Extend(s, prefix)
			if err != nil {
				b.Fatal(err)
			}

			for b.Loop() {
				_, err := m.Extend(s.Fork(), next)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

var benchModes = []struct {
	name string
	load func(testing.TB) *Model
}{
	{"dequantised", loadDequantised},
	{"packed", func(t testing.TB) *Model { m, _ := loadPacked(t); return m }},
	{"packed-lut", func(t testing.TB) *Model { m, _ := packedLUT.load(t, OpenOptions{Packed: true, LUT: true}); return m }},
	{"packed-int8", loadEngine},
}

// benchLong is the prompt length of the long benchmarks, about that of a
// request over the catalogue's tools (docs/FINDINGS.md F-060).
const benchLong = 600

// BenchmarkForwardLong is BenchmarkForward over the first benchLong tokens
// of long_tools, packed with float32 and with the engine's int8 numerics
// (docs/FINDINGS.md F-060).
//
//	go test -run '^$' -bench 'Long' ./internal/text/
func BenchmarkForwardLong(b *testing.B) {
	requireFile(b, forwardFixture)

	tokens := openFixture(b).ints(b, longCase+".tokens")[:benchLong]

	for _, mode := range benchModes[1:] {
		b.Run(mode.name, func(b *testing.B) {
			m := mode.load(b)

			for b.Loop() {
				_, err := m.Forward(tokens, false)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkExtendStepLong is BenchmarkExtendStep after the first benchLong
// tokens of long_tools.
func BenchmarkExtendStepLong(b *testing.B) {
	requireFile(b, forwardFixture)

	tokens := openFixture(b).ints(b, longCase+".tokens")[:benchLong+1]
	prefix, next := tokens[:benchLong], tokens[benchLong:]

	for _, mode := range benchModes[1:] {
		b.Run(mode.name, func(b *testing.B) {
			m := mode.load(b)
			s := m.NewState()

			_, err := m.Extend(s, prefix)
			if err != nil {
				b.Fatal(err)
			}

			for b.Loop() {
				_, err := m.Extend(s.Fork(), next)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
