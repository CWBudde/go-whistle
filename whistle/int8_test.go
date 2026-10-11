package whistle

import (
	"maps"
	"path/filepath"
	"testing"

	"github.com/cwbudde/go-whistle/internal/needle"
)

// TestWithInt8Option: WithInt8 switches on both halves of the engine's
// int8 numerics, and Load's default is float32 activations and cache
// (docs/FINDINGS.md F-041).
func TestWithInt8Option(t *testing.T) {
	var o loadOptions
	if o.int8 != (needle.Int8{}) {
		t.Fatalf("default int8 %+v, want off", o.int8)
	}

	WithInt8()(&o)

	if o.int8 != (needle.Int8{Act: true, KV: true}) {
		t.Errorf("WithInt8 sets %+v, want Act and KV", o.int8)
	}
}

// int8Misses are the corpus clips whose transcript differs from the
// engine's under WithInt8: engineMisses, plus the second Polish clip
// ("większości … wyzwań" against the engine's "większość … wyzptów"),
// whose depth-ladder transcripts differ from the engine's in float32
// already (docs/FINDINGS.md F-041).
var int8Misses = func() map[string]bool {
	m := maps.Clone(engineMisses)
	m["fleurs-pl_pl-10013271501230525766"] = true

	return m
}()

// int8TranscribeFloor is the share (%) of the corpus that
// TestTranscribeMatchesEngine requires under WHISTLE_TEST_INT8=1: 30 of 32
// measured (93.8 %), every miss in int8Misses. PLAN.md 6.6's 95 % was set
// on the float32 default (docs/FINDINGS.md F-041).
const int8TranscribeFloor = 93

// int8WordFloors are wordFloors under WHISTLE_TEST_INT8=1, just below the
// measured 96.7 % (start and end within 80 ms) and 91.3 % (probability
// within 0.01); wordFloors were measured on float32 (F-041).
var int8WordFloors = struct{ both, prob float64 }{both: 96.5, prob: 88}

// TestInt8MatchesEngine runs the corpus with WithInt8: every transcript but
// int8Misses equals the engine's (30 of 32 measured), and Embed's mean
// frame cosine against needle_embed stays at least 0.9995 per corpus and
// 0.99 per clip; measured 0.999719 and 0.994891 (float32 0.999827 and
// 0.999034). The worst clip is low because of one fragile frame, which
// the int8 rounding moves to cosine 0.63 (F-041), so its bound leaves
// room for another architecture's rounding. Embed also differs from the
// float32 run, so the option takes effect.
func TestInt8MatchesEngine(t *testing.T) {
	requireFile(t, cactPath)

	m, err := Load(cactPath, WithInt8())
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = m.Close() })

	float, err := Load(cactPath)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = float.Close() })

	clips := corpus(t)

	sum, worst := 0.0, 1.0
	equal, moved := 0, false

	for _, c := range clips {
		res, err := m.Transcribe(c.Samples, TranscribeOptions{})
		if err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}

		if res.Text == c.Text && res.Language == c.Language {
			equal++
		} else if !int8Misses[c.ID] {
			t.Errorf("%s: [%s] %q, the engine [%s] %q", c.ID, res.Language, res.Text, c.Language, c.Text)
		}

		emb, err := m.Embed(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		ref := readEmbed(t, filepath.Join(oracleDir, c.ID+".embed.safetensors"))

		var mean float64
		for i := range emb.Frames {
			mean += cosine(emb.Frame(i), ref[i*512:(i+1)*512])
		}

		mean /= float64(emb.Frames)
		sum += mean
		worst = min(worst, mean)

		f, err := float.Embed(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		for i, v := range f.Values {
			moved = moved || v != emb.Values[i]
		}
	}

	mean := sum / float64(len(clips))
	t.Logf("%d clips: %d equal to the engine; Embed mean frame cosine %.6f, worst clip %.6f", len(clips), equal, mean,
		worst)

	if mean < 0.9995 || worst < 0.99 {
		t.Errorf("Embed mean cosine %.6f (want ≥ 0.9995), worst clip %.6f (want ≥ 0.99)", mean, worst)
	}

	if !moved {
		t.Error("Embed with WithInt8 equals float32's")
	}
}
