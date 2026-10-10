//go:build oracle

// The data of the alignment probe (probe_test.go, docs/FINDINGS.md F-033):
// every engine Transcript recorded with --audio-word-timestamps, teacher
// forced in its most likely spelling under our model with the
// cross-attention captured. Building it takes minutes, so it is cached in
// a gob file (WHISTLE_ALIGN_CACHE, default in the temp directory;
// WHISTLE_ALIGN_FRESH=1 rebuilds it).

package align

import (
	"cmp"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/decoder"
	"github.com/cwbudde/go-whistle/internal/encoder"
	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/search"
	"github.com/cwbudde/go-whistle/internal/tokenizer"
	"github.com/cwbudde/go-whistle/internal/weights"
)

// Pinned inputs as laid out by `whistle download`, `just oracle-run` and
// `just oracle-librispeech-audio`.
const (
	cactPath    = "../../models/whistle/whistle.cact"
	clipDir     = "../../testdata/audio"
	oracleDir   = "../../testdata/oracle"
	lsAudioDir  = "../../models/corpus/librispeech"
	cacheFormat = "align-probe v2"
)

// engineWord is one word of an engine Transcript.
type engineWord struct {
	Word        string  `json:"word"`
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Probability float64 `json:"probability"`
}

// Transcript is an engine JSON object with words.
type Transcript struct {
	Language string       `json:"language"`
	Text     string       `json:"text"`
	Words    []engineWord `json:"words"`
}

// item is one engine Transcript and what our model makes of it.
type item struct {
	Set      string // fixtures, librispeech, keywords, depth
	ID       string // the clip, case or "<clip>.d<N>"
	Clip     string // the audio
	Depth    int    // the ladder depth, 0 for the full decoder
	Keywords []string
	Transcript

	Samples, Frames int
	Prompt          int   // tokens before the language token
	Tokens          []int // the text tokens, the engine's text in its most likely spelling
	// Per text token, then EOS:
	LP     []float64   // log p over the whole vocabulary
	Top    [][]float64 // the row's 8 best log p, descending
	TextLS []float64   // log Σ p over the text pieces (NORMAL, USER_DEFINED and BYTE)
	Adj    []float64   // the keyword bias' adjusted log p (= LP without keywords)
	AdjLS  []float64   // log Σ exp of the adjusted row
	// Attn is the teacher-forced pass over prompt, language, text and EOS:
	// one Attention per layer that ran, a row per position.
	Attn []decoder.Attention
}

// first is the position of the first text token in the teacher-forced
// sequence: the prompt, then the language token.
func (it *item) first() int { return it.Prompt + 1 }

// probeSetup is the model the probe runs.
type probeSetup struct {
	dec *decoder.Decoder
	enc *encoder.Encoder
	tok *tokenizer.Tokenizer
}

func openSetup(t testing.TB) probeSetup {
	t.Helper()

	_, err := os.Stat(cactPath)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `whistle download` to fetch the pinned weights", cactPath)
	}

	src, err := weights.Open(cactPath, weights.Options{})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = src.Close() })

	dec, err := decoder.Load(src, nil, decoder.Best())
	if err != nil {
		t.Fatal(err)
	}

	enc, err := encoder.Load(src, encoder.Best())
	if err != nil {
		t.Fatal(err)
	}

	return probeSetup{dec: dec, enc: enc, tok: dec.Tokenizer()}
}

// loadItems returns the probe's items, from the cache when it holds them.
func loadItems(t *testing.T, ps probeSetup) []*item {
	t.Helper()

	path := os.Getenv("WHISTLE_ALIGN_CACHE")
	if path == "" {
		path = filepath.Join(os.TempDir(), "whistle-align-probe.gob")
	}

	if os.Getenv("WHISTLE_ALIGN_FRESH") == "" {
		items, err := readCache(path)
		if err == nil {
			t.Logf("%d items from %s", len(items), path)

			return items
		}

		t.Logf("cache %s: %v; building it", path, err)
	}

	start := time.Now()
	items := buildItems(t, ps, oracleItems(t))

	err := writeCache(path, items)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("%d items built in %s, cached in %s", len(items), time.Since(start).Round(time.Second), path)

	return items
}

func readCache(path string) ([]*item, error) {
	f, err := os.Open(path) // #nosec G304 -- the probe's own cache.
	if err != nil {
		return nil, err
	}

	defer func() { _ = f.Close() }()

	dec := gob.NewDecoder(f)

	var format string

	err = dec.Decode(&format)
	if err != nil {
		return nil, err
	}

	if format != cacheFormat {
		return nil, fmt.Errorf("format %q, want %q", format, cacheFormat)
	}

	var items []*item

	err = dec.Decode(&items)

	return items, err
}

func writeCache(path string, items []*item) error {
	f, err := os.Create(path) // #nosec G304 -- the probe's own cache.
	if err != nil {
		return err
	}

	enc := gob.NewEncoder(f)

	err = enc.Encode(cacheFormat)
	if err == nil {
		err = enc.Encode(items)
	}

	return errors.Join(err, f.Close())
}

// oracleItems lists the engine transcripts with words: the 32 fixtures,
// the LibriSpeech subset, the keyword cases and the depth runs (F-025).
func oracleItems(t *testing.T) []*item {
	t.Helper()

	var out []*item

	var audio struct {
		Clips []struct {
			ID string `json:"id"`
		} `json:"clips"`
	}

	readJSON(t, filepath.Join(clipDir, "manifest.json"), &audio)

	for _, c := range audio.Clips {
		it := &item{Set: "fixtures", ID: c.ID, Clip: filepath.Join(clipDir, c.ID+".wav")}
		readJSON(t, filepath.Join(oracleDir, c.ID+".json"), &it.Transcript)
		out = append(out, it)
	}

	var ls map[string]Transcript

	readJSON(t, filepath.Join(oracleDir, "librispeech", "transcripts.json"), &ls)

	for _, id := range sortedKeys(ls) {
		out = append(out, &item{Set: "librispeech", ID: id, Clip: filepath.Join(lsAudioDir, id+".wav"), Transcript: ls[id]})
	}

	var kw struct {
		Cases []struct {
			Name     string   `json:"name"`
			Clip     string   `json:"clip"`
			Keywords []string `json:"keywords"`
		} `json:"cases"`
	}

	var kwText map[string]Transcript

	readJSON(t, filepath.Join(oracleDir, "keywords", "manifest.json"), &kw)
	readJSON(t, filepath.Join(oracleDir, "keywords", "transcripts.json"), &kwText)

	for _, c := range kw.Cases {
		out = append(out, &item{
			Set: "keywords", ID: c.Name, Clip: filepath.Join(clipDir, c.Clip+".wav"), Keywords: c.Keywords,
			Transcript: kwText[c.Name],
		})
	}

	var depth map[string]Transcript

	readJSON(t, filepath.Join(oracleDir, "depth", "transcripts.json"), &depth)

	for _, id := range sortedKeys(depth) {
		clip, d, _ := strings.Cut(id, ".d")

		n, err := strconv.Atoi(d)
		if err != nil {
			t.Fatalf("depth key %q", id)
		}

		out = append(out, &item{
			Set: "depth", ID: id, Clip: filepath.Join(clipDir, clip+".wav"), Depth: n, Transcript: depth[id],
		})
	}

	return slices.DeleteFunc(out, func(it *item) bool { return len(it.Words) == 0 })
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	sort.Strings(keys)

	return keys
}

func readJSON(t testing.TB, path string, v any) {
	t.Helper()

	raw, err := os.ReadFile(path) // #nosec G304 -- pinned fixtures.
	if err != nil {
		t.Fatal(err)
	}

	err = json.Unmarshal(raw, v)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

// buildItems encodes, spells and teacher-forces every item on all cores.
func buildItems(t *testing.T, ps probeSetup, items []*item) []*item {
	t.Helper()

	_, err := os.Stat(lsAudioDir)
	if err != nil {
		t.Logf("%s missing (just oracle-librispeech-audio): the LibriSpeech items are skipped", lsAudioDir)

		items = slices.DeleteFunc(items, func(it *item) bool { return it.Set == "librispeech" })
	}

	ladder := map[int]*decoder.Decoder{0: ps.dec}

	for n := 2; n < 8; n++ {
		ladder[n], err = ps.dec.WithDepth(n)
		if err != nil {
			t.Fatal(err)
		}
	}

	jobs := make(chan int)
	errs := make([]error, len(items))

	var wg sync.WaitGroup

	for range max(1, runtime.NumCPU()/2) {
		wg.Go(func() {
			for i := range jobs {
				it := items[i]

				err := buildItem(ps, ladder[it.Depth], it)
				if err != nil {
					errs[i] = fmt.Errorf("%s/%s: %w", it.Set, it.ID, err)
				}
			}
		})
	}

	for i := range items {
		jobs <- i
	}

	close(jobs)
	wg.Wait()

	err = errors.Join(errs...)
	if err != nil {
		t.Fatal(err)
	}

	return items
}

// buildItem fills one item.
func buildItem(ps probeSetup, d *decoder.Decoder, it *item) error {
	pcm, err := frontend.ReadWAV(it.Clip)
	if err != nil {
		return err
	}

	enc, frames, err := ps.enc.Embed(pcm)
	if err != nil {
		return err
	}

	mem, err := d.Memory(enc, frames)
	if err != nil {
		return err
	}

	var adjust search.Adjust

	if len(it.Keywords) > 0 {
		kw, err := search.NewKeywords(ps.tok, it.Keywords, search.DefaultBias())
		if err != nil {
			return err
		}

		adjust = kw.Adjust
	}

	lang := -1

	for _, l := range d.Languages() {
		if l.Code == it.Language {
			lang = l.ID
		}
	}

	if lang < 0 {
		return fmt.Errorf("language %q", it.Language)
	}

	it.Samples, it.Frames, it.Prompt = len(pcm), frames, len(d.Prompt())

	it.Tokens, err = bestTokens(d, mem, lang, it.Text, adjust)
	if err != nil {
		return err
	}

	text, err := d.Decode(it.Tokens)
	if err != nil || text != it.Text {
		return fmt.Errorf("spelling %q of %q: %w", text, it.Text, err)
	}

	return force(d, mem, lang, it, adjust)
}

// force runs the teacher-forced pass with the capture on and fills the
// token scores and the attention.
func force(d *decoder.Decoder, mem *decoder.Memory, lang int, it *item, adjust search.Adjust) error {
	seq := append(append(d.Prompt(), lang), it.Tokens...)
	seq = append(seq, d.Tokenizer().EOSID())

	s := d.NewState(mem)
	s.SetCapture(func(a decoder.Attention) { it.Attn = append(it.Attn, a) })

	logits, err := d.Extend(s, seq)
	if err != nil {
		return err
	}

	v, tk := d.Vocab(), d.Tokenizer()
	text := make([]bool, v)

	for id := range text {
		ty := tk.Type(id)
		text[id] = ty == tokenizer.TypeNormal || ty == tokenizer.TypeUserDefined || ty == tokenizer.TypeByte
	}

	for k := range len(it.Tokens) + 1 {
		pos := it.first() + k - 1 // the row that predicts text token k (k = len: EOS)
		row := logSoftmax(logits[pos*v : (pos+1)*v])
		want := seq[pos+1]

		it.LP = append(it.LP, row[want])

		top := slices.Clone(row)
		slices.SortFunc(top, func(a, b float64) int { return cmp.Compare(b, a) })
		it.Top = append(it.Top, top[:8])

		var ls []float64

		for id, ok := range text {
			if ok {
				ls = append(ls, row[id])
			}
		}

		it.TextLS = append(it.TextLS, logSumExp(ls))

		if adjust != nil {
			adjust(slices.Clone(it.Tokens[:k]), row)
		}

		it.Adj = append(it.Adj, row[want])
		it.AdjLS = append(it.AdjLS, logSumExp(row))
	}

	return nil
}

// tokenBytes is the bytes every piece spells: NORMAL pieces their surface
// with "▁" as a space, BYTE pieces their byte, every other piece nil.
func tokenBytes(tk *tokenizer.Tokenizer) [][]byte {
	out := make([][]byte, tk.Len())

	for id := range out {
		switch tk.Type(id) {
		case tokenizer.TypeNormal:
			out[id] = []byte(strings.ReplaceAll(tk.Piece(id), "▁", " "))
		case tokenizer.TypeByte:
			v, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(tk.Piece(id), "<0x"), ">"), 16, 8)
			if err == nil {
				out[id] = []byte{byte(v)}
			}
		}
	}

	return out
}

// bestTokens is the most likely spelling of text under our model (and the
// keyword bias, when adjust is set), as internal/search's probe finds it
// (docs/FINDINGS.md F-026): a beam search of width 16 over the token
// sequences whose bytes spell text exactly, ranked by summed log p with
// EOS.
func bestTokens(d *decoder.Decoder, mem *decoder.Memory, lang int, text string, adjust search.Adjust) ([]int, error) {
	const width = 16

	type path struct {
		st   *decoder.State
		toks []int
		off  int
		sum  float64
	}

	target, spell := []byte(text), tokenBytes(d.Tokenizer())
	root := d.NewState(mem)

	logits, err := d.Extend(root, append(d.Prompt(), lang))
	if err != nil {
		return nil, err
	}

	v := d.Vocab()
	scores := func(p *path, logits []float32) []float64 {
		row := logSoftmax(logits)
		if adjust != nil {
			adjust(slices.Clone(p.toks), row)
		}

		return row
	}

	live := []*path{{st: root}}
	rows := map[*path][]float64{live[0]: scores(live[0], logits[len(logits)-v:])}

	var best *path

	bestSum := math.Inf(-1)

	for len(live) > 0 {
		var next []*path

		for _, p := range live {
			lp := rows[p]

			if p.off == len(target) {
				if s := p.sum + lp[d.Tokenizer().EOSID()]; s > bestSum {
					best, bestSum = p, s
				}

				continue
			}

			for id, b := range spell {
				if len(b) > 0 && len(b) <= len(target)-p.off && string(target[p.off:p.off+len(b)]) == string(b) {
					next = append(next, &path{
						st: p.st, toks: append(slices.Clone(p.toks), id), off: p.off + len(b), sum: p.sum + lp[id],
					})
				}
			}
		}

		slices.SortStableFunc(next, func(a, b *path) int { return cmp.Compare(b.sum, a.sum) })
		live = next[:min(width, len(next))]

		if len(live) == 0 || live[0].sum <= bestSum {
			break
		}

		for _, p := range live {
			p.st = p.st.Fork()

			l, err := d.Extend(p.st, p.toks[len(p.toks)-1:])
			if err != nil {
				return nil, err
			}

			rows[p] = scores(p, l)
		}
	}

	if best == nil {
		return nil, errors.New("no spelling")
	}

	return best.toks, nil
}

// logSoftmax is the log softmax of v in float64.
func logSoftmax(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}

	lse := logSumExp(out)
	for i := range out {
		out[i] -= lse
	}

	return out
}

// logSumExp is log Σ exp(v).
func logSumExp(v []float64) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		m = math.Max(m, x)
	}

	var s float64
	for _, x := range v {
		s += math.Exp(x - m)
	}

	return m + math.Log(s)
}
