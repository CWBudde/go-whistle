package whistle

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/safetensors"
	"github.com/cwbudde/go-whistle/internal/search"
)

// Pinned inputs as laid out by `whistle download` and `just oracle-run`.
const (
	cactPath        = "../models/whistle/whistle.cact"
	safetensorsPath = "../models/whistle/whistle.safetensors"
	clipDir         = "../testdata/audio"
	oracleDir       = "../testdata/oracle"
)

// requireFile skips the test when path is missing.
func requireFile(t testing.TB, path string) {
	t.Helper()

	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", path)
	}
}

// loadModel loads the .cact model or skips the test.
func loadModel(t testing.TB) *Model {
	t.Helper()

	requireFile(t, cactPath)

	m, err := Load(cactPath)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = m.Close() })

	return m
}

// clip is one corpus clip with the engine's transcript.
type clip struct {
	ID       string
	Path     string
	Samples  []float32
	Language string
	Text     string
}

// clipIDs lists the corpus in manifest order.
func clipIDs(t testing.TB) []string {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(clipDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Clips []struct {
			ID string `json:"id"`
		} `json:"clips"`
	}

	err = json.Unmarshal(raw, &manifest)
	if err != nil {
		t.Fatal(err)
	}

	ids := make([]string, len(manifest.Clips))
	for i, c := range manifest.Clips {
		ids[i] = c.ID
	}

	return ids
}

// shortClips are three short clips (LibriSpeech test-other and test-clean,
// FLEURS de), the -short subset.
var shortClips = []string{"ls-other-3080-5040-0018", "ls-clean-5639-40744-0011", "fleurs-de_de-10308608197319017935"}

// loadClips reads the named clips, all 32 when ids is empty.
func loadClips(t testing.TB, ids ...string) []clip {
	t.Helper()

	if len(ids) == 0 {
		ids = clipIDs(t)
	}

	out := make([]clip, 0, len(ids))

	for _, id := range ids {
		path := filepath.Join(clipDir, id+".wav")

		samples, err := frontend.ReadWAV(path)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		js, err := os.ReadFile(filepath.Join(oracleDir, id+".json"))
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		var tr struct {
			Language string `json:"language"`
			Text     string `json:"text"`
		}

		err = json.Unmarshal(js, &tr)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		out = append(out, clip{ID: id, Path: path, Samples: samples, Language: tr.Language, Text: tr.Text})
	}

	return out
}

// corpus is all 32 clips, or the short subset under -short.
func corpus(t testing.TB) []clip {
	t.Helper()

	if testing.Short() {
		return loadClips(t, shortClips...)
	}

	return loadClips(t)
}

// engineMisses are the corpus clips on which the beam search's transcript
// differs from the engine's: a search miss on a non-Polish text, where the
// engine's prefix falls out of our beam 0.135 nats behind the fifth live
// hypothesis (docs/FINDINGS.md F-026).
var engineMisses = map[string]bool{"fleurs-pl_pl-10047713455444904767": true}

// Accept (PLAN.md 6.6): Transcribe gives the engine's text and language on
// at least 95 % of the corpus, every miss explained in FINDINGS, and equals
// search.Beam under search.Default on the same encoder output.
func TestTranscribeMatchesEngine(t *testing.T) {
	m := loadModel(t)
	clips := corpus(t)
	equal := 0

	for _, c := range clips {
		res, err := m.Transcribe(c.Samples, TranscribeOptions{})
		if err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}

		want := beamDecode(t, m, c.Samples, "")
		if res.Text != want.Text || res.Language != want.Language {
			t.Errorf("%s: Transcribe [%s] %q, search.Beam [%s] %q", c.ID, res.Language, res.Text, want.Language, want.Text)
		}

		if res.TTFT <= 0 || res.DecodeTPS <= 0 {
			t.Errorf("%s: ttft %v, decode_tps %v; want both > 0", c.ID, res.TTFT, res.DecodeTPS)
		}

		same := res.Text == c.Text && res.Language == c.Language
		if same {
			equal++
		} else if !engineMisses[c.ID] {
			t.Errorf("%s: [%s] %q, the engine [%s] %q", c.ID, res.Language, res.Text, c.Language, c.Text)
		}

		t.Logf("%s: [%s] ttft %.1f ms, %.1f tok/s, equal to the engine %v", c.ID, res.Language,
			float64(res.TTFT.Microseconds())/1000, res.DecodeTPS, same)
	}

	if 100*equal < 95*len(clips) {
		t.Errorf("%d of %d clips equal to the engine, want at least 95 %%", equal, len(clips))
	}

	t.Logf("%d clips: text and language equal to the engine on %d (%.1f %%)", len(clips), equal,
		100*float64(equal)/float64(len(clips)))
}

// beamDecode runs search.Beam under search.Default on the model's encoder
// output, outside the strategy seam.
func beamDecode(t *testing.T, m *Model, pcm []float32, lang string) search.Result {
	t.Helper()

	emb, err := m.Embed(pcm)
	if err != nil {
		t.Fatal(err)
	}

	mem, err := m.dec.Memory(emb.Values, emb.Frames)
	if err != nil {
		t.Fatal(err)
	}

	opt := search.Default()
	opt.Language = lang

	r, err := search.Beam(m.dec, mem, opt)
	if err != nil {
		t.Fatal(err)
	}

	return r
}

// The greedy strategy stays reachable for tests: through Transcribe it
// equals decoder.Greedy (PLAN.md 6.5).
func TestGreedyStrategy(t *testing.T) {
	m := loadModel(t)
	m.strategy = greedy{}

	for _, c := range loadClips(t, shortClips...) {
		res, err := m.Transcribe(c.Samples, TranscribeOptions{})
		if err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}

		emb, err := m.Embed(c.Samples)
		if err != nil {
			t.Fatal(err)
		}

		want, err := m.dec.Greedy(emb.Values, emb.Frames, "")
		if err != nil {
			t.Fatal(err)
		}

		if res.Text != want.Text || res.Language != want.Language {
			t.Errorf("%s: Transcribe [%s] %q, decoder.Greedy [%s] %q", c.ID, res.Language, res.Text, want.Language, want.Text)
		}

		if res.TTFT <= 0 || res.DecodeTPS <= 0 {
			t.Errorf("%s: ttft %v, decode_tps %v; want both > 0", c.ID, res.TTFT, res.DecodeTPS)
		}
	}
}

// The beam strategy's clock: one tick per search step after the language
// token, so at least one per text token plus EOS, and a keyword hook in the
// options still sees every live hypothesis.
func TestBeamClock(t *testing.T) {
	m := loadModel(t)
	c := loadClips(t, shortClips[1])[0]

	emb, err := m.Embed(c.Samples)
	if err != nil {
		t.Fatal(err)
	}

	mem, err := m.dec.Memory(emb.Values, emb.Frames)
	if err != nil {
		t.Fatal(err)
	}

	opt := search.Default()
	calls := 0
	opt.Adjust = func([]int, []float64) { calls++ }

	var ms time.Duration

	clk := newClock(func() time.Time { ms += time.Millisecond; return time.Unix(0, 0).Add(ms) })

	out, err := beam{opt: opt}.decode(m.dec, mem, request{}, clk)
	if err != nil {
		t.Fatal(err)
	}

	steps := clk.after + 1
	if steps < len(out.tokens)+1 || calls < steps || calls > opt.Beams*steps {
		t.Errorf("%d text tokens: %d clock ticks, %d hook calls; want ≥ %d ticks and 1–%d calls per tick",
			len(out.tokens), steps, calls, len(out.tokens)+1, opt.Beams)
	}

	// The clock started before the search and ticks once per step.
	if clk.ttft() != time.Millisecond || clk.tps() != 1000 {
		t.Errorf("ttft %v, tps %v; want 1ms and 1000/s with one fake millisecond per reading", clk.ttft(), clk.tps())
	}
}

func TestTranscribeForcedLanguage(t *testing.T) {
	m := loadModel(t)
	c := loadClips(t, "fleurs-de_de-10308608197319017935")[0]

	res, err := m.Transcribe(c.Samples, TranscribeOptions{Language: "en"})
	if err != nil {
		t.Fatal(err)
	}

	want := beamDecode(t, m, c.Samples, "en")
	if res.Language != "en" || res.Text != want.Text {
		t.Errorf("forced en: [%s] %q, search.Beam %q", res.Language, res.Text, want.Text)
	}
}

func TestTranscribeWAV(t *testing.T) {
	m := loadModel(t)
	c := loadClips(t, shortClips[0])[0]

	res, err := m.TranscribeWAV(c.Path, TranscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if res.Text != c.Text || res.Language != c.Language {
		t.Errorf("TranscribeWAV = [%s] %q, engine [%s] %q", res.Language, res.Text, c.Language, c.Text)
	}

	js, err := res.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	shape := regexp.MustCompile(`^\{"text":"[^"]*","language":"en","ttft_ms":\d+\.\d,"decode_tps":\d+\.\d\}$`)
	if !shape.Match(js) {
		t.Errorf("JSON %s does not have the engine's shape", js)
	}
}

func TestTranscribeWAVNoAudio(t *testing.T) {
	m := loadModel(t)
	path := filepath.Join(t.TempDir(), "empty.wav")
	writeWAV(t, path, nil)

	_, err := m.TranscribeWAV(path, TranscribeOptions{})
	if !errors.Is(err, ErrNoAudio) {
		t.Errorf("empty WAV: %v, want ErrNoAudio", err)
	}
}

// writeWAV writes 16-bit mono 16 kHz PCM.
func writeWAV(t *testing.T, path string, pcm []int16) {
	t.Helper()

	n := uint32(2 * len(pcm))
	b := []byte("RIFF")
	b = binary.LittleEndian.AppendUint32(b, 36+n)
	b = append(b, "WAVEfmt "...)
	b = binary.LittleEndian.AppendUint32(b, 16)
	b = binary.LittleEndian.AppendUint16(b, 1)
	b = binary.LittleEndian.AppendUint16(b, 1)
	b = binary.LittleEndian.AppendUint32(b, SampleRate)
	b = binary.LittleEndian.AppendUint32(b, 2*SampleRate)
	b = binary.LittleEndian.AppendUint16(b, 2)
	b = binary.LittleEndian.AppendUint16(b, 16)
	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, n)

	for _, v := range pcm {
		b = binary.LittleEndian.AppendUint16(b, uint16(v))
	}

	err := os.WriteFile(path, b, 0o600)
	if err != nil {
		t.Fatal(err)
	}
}

func TestTranscribeLimit(t *testing.T) {
	m := loadModel(t)
	pcm := make([]float32, MaxSamples+1)

	_, err := m.Transcribe(pcm, TranscribeOptions{})
	if !errors.Is(err, ErrTooLong) || !strings.Contains(err.Error(), "30 s") {
		t.Errorf("Transcribe(%d samples): %v, want ErrTooLong naming 30 s", len(pcm), err)
	}

	_, err = m.Embed(pcm)
	if !errors.Is(err, ErrTooLong) || !strings.Contains(err.Error(), "30 s") {
		t.Errorf("Embed(%d samples): %v, want ErrTooLong naming 30 s", len(pcm), err)
	}

	// Exactly 30 s is accepted: 375 frames.
	emb, err := m.Embed(pcm[:MaxSamples])
	if err != nil || emb.Frames != 375 || emb.Dim != 512 {
		t.Errorf("Embed(%d samples) = %d frames of %d, %v; want 375 of 512", MaxSamples, emb.Frames, emb.Dim, err)
	}
}

func TestTranscribeTooShortForAFrame(t *testing.T) {
	m := loadModel(t)

	for _, n := range []int{0, 10, 159} {
		res, err := m.Transcribe(make([]float32, n), TranscribeOptions{})
		if err != nil || res.Text != "" || res.Language != "" || res.Words != nil {
			t.Errorf("%d samples: %+v, %v; want Result{}", n, res, err)
		}

		emb, err := m.Embed(make([]float32, n))
		if err != nil || emb.Frames != 0 || len(emb.Values) != 0 {
			t.Errorf("Embed(%d samples) = %d frames, %v; want none", n, emb.Frames, err)
		}
	}
}

func TestTranscribeOptionsRejected(t *testing.T) {
	m := loadModel(t)
	pcm := make([]float32, SampleRate)

	for _, tc := range []struct {
		name    string
		opts    TranscribeOptions
		unsupp  bool
		message string
	}{
		{"depth 1", TranscribeOptions{Depth: 1}, false, "outside 2..8"},
		{"depth 9", TranscribeOptions{Depth: 9}, false, "outside 2..8"},
		{"language", TranscribeOptions{Language: "xx"}, false, `unknown language "xx"`},
	} {
		_, err := m.Transcribe(pcm, tc.opts)
		if err == nil || errors.Is(err, ErrNotSupported) != tc.unsupp || !strings.Contains(err.Error(), tc.message) {
			t.Errorf("%s: %v; want an error naming %q, ErrNotSupported %v", tc.name, err, tc.message, tc.unsupp)
		}
	}

	for _, opts := range []TranscribeOptions{
		{Depth: 8}, {Depth: 2}, {Depth: 7}, {Keywords: []string{"Siobhan"}}, {WordTimestamps: true},
	} {
		_, err := m.Transcribe(pcm, opts)
		if err != nil {
			t.Errorf("%+v: %v", opts, err)
		}
	}
}

func TestLanguages(t *testing.T) {
	m := loadModel(t)

	got := strings.Join(m.Languages(), " ")
	if got != "en de fr es it nl pl" {
		t.Errorf("Languages = %s", got)
	}
}

// Accept (PLAN.md 8.2, Phase 8): Embed matches needle_embed at cosine
// ≥ 0.99 on every clip.
func TestEmbedMatchesEngine(t *testing.T) {
	m := loadModel(t)
	clips := corpus(t)

	var sum, worst float64

	worst = 1

	for _, c := range clips {
		emb, err := m.Embed(c.Samples)
		if err != nil {
			t.Fatalf("%s: %v", c.ID, err)
		}

		ref := readEmbed(t, filepath.Join(oracleDir, c.ID+".embed.safetensors"))
		if emb.Dim != 512 || len(emb.Values) != len(ref) || emb.Frames*emb.Dim != len(ref) {
			t.Fatalf("%s: %d frames of %d, engine %d values", c.ID, emb.Frames, emb.Dim, len(ref))
		}

		var mean float64
		for i := range emb.Frames {
			mean += cosine(emb.Frame(i), ref[i*512:(i+1)*512])
		}

		mean /= float64(emb.Frames)

		if mean < 0.99 {
			t.Errorf("%s: mean frame cosine %.6f < 0.99", c.ID, mean)
		}

		sum += mean
		worst = min(worst, mean)
	}

	mean := sum / float64(len(clips))
	t.Logf("%d clips: mean frame cosine %.6f, worst clip %.6f", len(clips), mean, worst)

	// F-020 measured mean 0.999828 on all 32; keep the port from drifting.
	if mean < 0.999 {
		t.Errorf("mean cosine %.6f < 0.999", mean)
	}
}

// readEmbed reads the [frames, 512] needle_embed rows of a fixture.
func readEmbed(t testing.TB, path string) []float32 {
	t.Helper()

	store, err := safetensors.OpenStore(path, safetensors.StoreOptions{})
	if err != nil {
		t.Fatal(err)
	}

	defer store.Close()

	ts, err := store.Tensor("embed")
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}

	if len(ts.Shape) != 2 || ts.Shape[1] != 512 {
		t.Fatalf("%s: embed shape %v", path, ts.Shape)
	}

	return ts.Data
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64

	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}

	return ab / math.Sqrt(aa*bb)
}

func TestLoadSafetensors(t *testing.T) {
	requireFile(t, safetensorsPath)
	requireFile(t, cactPath)

	// The tokenizer comes from the whistle.cact next to the checkpoint.
	m, err := Load(safetensorsPath)
	if err != nil {
		t.Fatal(err)
	}

	defer m.Close()

	c := loadClips(t, shortClips[1])[0]

	res, err := m.Transcribe(c.Samples, TranscribeOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if res.Language != c.Language || res.Text == "" {
		t.Errorf("safetensors: [%s] %q, engine [%s] %q", res.Language, res.Text, c.Language, c.Text)
	}

	t.Logf("safetensors: [%s] %q", res.Language, res.Text)
}

func TestLoadSafetensorsNeedsVocabulary(t *testing.T) {
	requireFile(t, safetensorsPath)
	requireFile(t, cactPath)

	abs, err := filepath.Abs(safetensorsPath)
	if err != nil {
		t.Fatal(err)
	}

	link := filepath.Join(t.TempDir(), "whistle.safetensors")

	err = os.Symlink(abs, link)
	if err != nil {
		t.Skipf("symlink: %v", err)
	}

	_, err = Load(link)
	if err == nil || !strings.Contains(err.Error(), "WithVocabulary") {
		t.Errorf("Load without a vocabulary: %v, want an error naming WithVocabulary", err)
	}

	m, err := Load(link, WithVocabulary(cactPath), WithoutMmap())
	if err != nil {
		t.Fatalf("Load with WithVocabulary: %v", err)
	}

	_ = m.Close()
}

func TestLoadMissing(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "none.cact"))
	if err == nil {
		t.Error("Load of a missing file: no error")
	}
}
