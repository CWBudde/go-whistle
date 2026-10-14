package text

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/cwbudde/go-whistle/internal/spm"
)

const needle3Tokenizer = "../../models/needle3/tokenizer.model"

// toolsEndID is the id of </tools>, the last token of the static prefix
// that the engine keeps in view past its KV window (docs/FINDINGS.md
// F-048, F-060).
func toolsEndID(t testing.TB) int {
	t.Helper()
	requireFile(t, needle3Tokenizer)

	p, err := spm.Load(needle3Tokenizer)
	if err != nil {
		t.Fatal(err)
	}

	id, ok := p.PieceID("</tools>")
	if !ok {
		t.Fatal("no piece </tools>")
	}

	return id
}

// engineModel is the packed needle3.cact with the engine's numerics,
// loaded once per test binary: int8 activations and KV cache, the KV
// window of 256 and the static prefix as its sink (docs/FINDINGS.md
// F-060).
var engineModel struct {
	once sync.Once
	m    *Model
	err  error
}

func loadEngine(t testing.TB) *Model {
	t.Helper()

	_, src := loadPacked(t)
	sink := toolsEndID(t)

	engineModel.once.Do(func() {
		cfg := src.TextConfig()
		engineModel.m, engineModel.err = LoadWith(src, cfg, LoadOptions{
			Int8: true, KVWindow: cfg.KVWindow, SinkToken: sink,
		})
	})

	if engineModel.err != nil {
		t.Fatal(engineModel.err)
	}

	return engineModel.m
}

// TestLoadWithOptions: the zero options are Load's model bit for bit, the
// engine's options set the KV window on every layer (the band stays on the
// local ones), and invalid options are rejected.
func TestLoadWithOptions(t *testing.T) {
	requireFile(t, forwardFixture)

	packed, src := loadPacked(t)
	cfg := src.TextConfig()
	tokens := openFixture(t).ints(t, "timer_en.tokens")

	zero, err := LoadWith(src, cfg, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}

	want, err := packed.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	got, err := zero.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got.Logits, want.Logits) {
		t.Error("LoadWith with the zero options differs from Load")
	}

	if cfg.KVWindow != 256 {
		t.Fatalf("needle3.cact's kv_window %d, want 256", cfg.KVWindow)
	}

	m := loadEngine(t)

	for l, b := range m.blocks {
		band := cfg.SlidingWindow
		if slices.Contains(cfg.GlobalLayers, l) {
			band = 0
		}

		if b.KVWindow() != 256 || b.Window() != band {
			t.Errorf("layer %d: KV window %d, band %d; want 256, %d", l, b.KVWindow(), b.Window(), band)
		}
	}

	for _, opts := range []LoadOptions{{KVWindow: -1}, {SinkToken: -1}, {SinkToken: cfg.Vocab}, {KVWindow: 2}} {
		_, err := LoadWith(src, cfg, opts)
		if err == nil {
			t.Errorf("LoadWith(%+v) accepted", opts)
		}
	}
}

// TestInt8Forward: the engine's int8 numerics move the packed model's
// logits and confidence by a quantisation step, not more: on every
// fixture case the logits stay within cosine 0.998 of the float32
// activations' (the worst row is at 0.99963, on long_tools) and the top-1
// is the same wherever the float32 margin is at least 1 (the largest
// margin a flip had was 0.82), while the confidence logit moves, but by
// less than 0.1 (docs/FINDINGS.md F-060).
func TestInt8Forward(t *testing.T) {
	requireFile(t, forwardFixture)

	packed, _ := loadPacked(t)
	m := loadEngine(t)
	fx := openFixture(t)
	v := m.cfg.OutVocab

	for _, name := range forwardCases {
		t.Run(name, func(t *testing.T) {
			tokens := fx.ints(t, name+".tokens")

			want, err := packed.Forward(tokens, true)
			if err != nil {
				t.Fatal(err)
			}

			got, err := m.Forward(tokens, true)
			if err != nil {
				t.Fatal(err)
			}

			c, row := rowCos(got.Logits, want.Logits, v)
			flips := 0

			for p := range tokens {
				wr := want.Logits[p*v : (p+1)*v]
				if argmax(got.Logits[p*v:(p+1)*v]) != argmax(wr) {
					flips++

					if margin(wr) >= 1 {
						t.Errorf("position %d: top-1 changes at a float32 margin of %.3f", p, margin(wr))
					}
				}
			}

			cw, err := packed.Confidence(want, len(tokens))
			if err != nil {
				t.Fatal(err)
			}

			cg, err := m.Confidence(got, len(tokens))
			if err != nil {
				t.Fatal(err)
			}

			t.Logf("logits worst row cosine %.6f at %d, top-1 changed at %d of %d, confidence logit %.4f float32 %.4f",
				c, row, flips, len(tokens), cg, cw)

			if c < 0.998 {
				t.Errorf("logits cosine %.6f at %d", c, row)
			}

			if d := math.Abs(float64(cg - cw)); d == 0 || d > 0.1 {
				t.Errorf("confidence logit %.4f against float32 %.4f", cg, cw)
			}
		})
	}
}

// TestInt8ExtendMatchesForward pins Extend to Forward under the engine's
// numerics on every fixture case: the prompt in one call and the
// completion a token at a time (AttendInt8Spans on the int8 caches, the KV
// window and the sink), and uneven chunks, which on the long case cross
// the 256 window and the 1024 band. The int8 grid lets a value round the
// other way between a block and a row, so the logits agree within cosine
// 0.9999 and the top-1 wherever Forward's margin is at least 0.1.
func TestInt8ExtendMatchesForward(t *testing.T) {
	requireFile(t, forwardFixture)

	m := loadEngine(t)
	fx := openFixture(t)
	v := m.cfg.OutVocab

	for _, name := range forwardCases {
		t.Run(name, func(t *testing.T) {
			tokens := fx.ints(t, name+".tokens")
			prompt := fx.ints(t, name+".prompt_len")[0]
			n := len(tokens)

			want, err := m.Forward(tokens, false)
			if err != nil {
				t.Fatal(err)
			}

			uneven := []int{7, 50, n - 57}
			if name == longCase {
				uneven = []int{1000, 30, n - 1030}
			}

			for _, sp := range []struct {
				name  string
				sizes []int
			}{
				{"prompt, then per token", append([]int{prompt}, ones(n-prompt)...)},
				{fmt.Sprint(uneven), uneven},
			} {
				got := extendRows(t, m, m.NewState(), tokens, sp.sizes)

				c, row := rowCos(got, want.Logits, v)
				if c < 0.9999 {
					t.Errorf("%s: logits cosine %.9f at %d", sp.name, c, row)
				}

				for p := range n {
					wr := want.Logits[p*v : (p+1)*v]
					if argmax(got[p*v:(p+1)*v]) != argmax(wr) && margin(wr) >= 0.1 {
						t.Errorf("%s: position %d: top-1 changes at a margin of %.3f", sp.name, p, margin(wr))
					}
				}

				t.Logf("%s: worst row cosine %.9f at %d", sp.name, c, row)
			}
		})
	}
}

// TestKVWindowSink: the static prefix as the sink keeps the KV window from
// cutting anything on long_tools, whose 1198-token prompt is mostly tools:
// with the sink the float32 model gives Load's logits bit for bit, and
// without it the window changes them (docs/FINDINGS.md F-060). A state's
// sink covers every position until </tools> is fed, then ends after it,
// and its forks keep it.
func TestKVWindowSink(t *testing.T) {
	requireFile(t, forwardFixture)

	packed, src := loadPacked(t)
	cfg := src.TextConfig()
	sink := toolsEndID(t)
	tokens := openFixture(t).ints(t, longCase+".tokens")

	want, err := packed.Forward(tokens, false)
	if err != nil {
		t.Fatal(err)
	}

	end := slices.Index(tokens, sink)
	if end < 0 || len(tokens)-end > cfg.KVWindow {
		t.Fatalf("</tools> at %d of %d", end, len(tokens))
	}

	for _, tc := range []struct {
		sink  int
		equal bool
	}{{sink, true}, {0, false}} {
		m, err := LoadWith(src, cfg, LoadOptions{KVWindow: cfg.KVWindow, SinkToken: tc.sink})
		if err != nil {
			t.Fatal(err)
		}

		got, err := m.Forward(tokens, false)
		if err != nil {
			t.Fatal(err)
		}

		if slices.Equal(got.Logits, want.Logits) != tc.equal {
			c, row := rowCos(got.Logits, want.Logits, cfg.OutVocab)
			t.Errorf("sink token %d: logits equal to Load's %v (worst row cosine %.9f at %d)", tc.sink, !tc.equal,
				c, row)
		}

		if !tc.equal {
			continue
		}

		s := m.NewState()

		_, err = m.Extend(s, tokens[:end])
		if err != nil {
			t.Fatal(err)
		}

		if s.sink != end || s.sinkFound || s.kv[0].Sink() != end {
			t.Errorf("sink %d (found %v, cache %d) before </tools> at %d", s.sink, s.sinkFound, s.kv[0].Sink(), end)
		}

		_, err = m.Extend(s, tokens[end:end+3])
		if err != nil {
			t.Fatal(err)
		}

		if f := s.Fork(); s.sink != end+1 || f.sink != end+1 || f.kv[0].Sink() != end+1 {
			t.Errorf("sink %d, fork %d/%d after </tools> at %d", s.sink, f.sink, f.kv[0].Sink(), end)
		}
	}
}

// TestInt8CacheMemory: a state's int8 caches take 4800 bytes per position
// over the 20 layers (2 KV heads of 48 key and 64 value codes and a scale
// each), against 17,920 for float32 (docs/FINDINGS.md F-060). The caches
// keep every position; the KV window narrows what a position reads. What
// the heap retains, append's spare capacity included, is logged after a
// prompt and decoding steps one token at a time.
func TestInt8CacheMemory(t *testing.T) {
	requireFile(t, forwardFixture)

	packed, _ := loadPacked(t)
	tokens := openFixture(t).ints(t, longCase+".tokens")

	for _, tc := range []struct {
		name        string
		m           *Model
		perPosition int
	}{{"float32", packed, 17920}, {"int8", loadEngine(t), 4800}} {
		s := tc.m.NewState()

		_, err := tc.m.Extend(s, tokens)
		if err != nil {
			t.Fatal(err)
		}

		var n int
		for _, c := range s.kv {
			n += c.Bytes()
		}

		t.Logf("%s: %d bytes for %d positions, %.2f MiB per 1000", tc.name, n, len(tokens),
			float64(n)/float64(len(tokens))*1000/(1<<20))

		if n != tc.perPosition*len(tokens) {
			t.Errorf("%s: %d bytes for %d positions, want %d per position", tc.name, n, len(tokens), tc.perPosition)
		}

		for _, tok := range tokens[:64] {
			_, err = tc.m.Extend(s, []int{tok})
			if err != nil {
				t.Fatal(err)
			}
		}

		var need, held int
		for _, c := range s.kv {
			need, held = need+c.Bytes(), held+c.AllocatedBytes()
		}

		t.Logf("%s: after 64 steps %d bytes needed, %d allocated (%.2f×)", tc.name, need, held,
			float64(held)/float64(need))

		if need != tc.perPosition*s.Len() || held < need {
			t.Errorf("%s: %d bytes needed and %d allocated for %d positions, want %d per position and at least that",
				tc.name, need, held, s.Len(), tc.perPosition)
		}
	}
}
