package whistle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/go-whistle/internal/eval"
	"github.com/cwbudde/go-whistle/internal/frontend"
)

// The LibriSpeech test-clean subset of PLAN.md 6.1 (docs/FINDINGS.md F-025).
const (
	librispeechDir = "../testdata/oracle/librispeech"
	// librispeechEnv runs TestEvalLibriSpeech; `just eval-librispeech` sets it.
	librispeechEnv = "WHISTLE_EVAL_LIBRISPEECH"
	// werTolerancePP is PLAN.md 6.6's bound on our WER minus the engine's,
	// in percentage points.
	werTolerancePP = 0.3
)

// lsUtterance is one utterance of subset.json with the engine's transcript
// from manifest.json.
type lsUtterance struct {
	ID         string `json:"id"`
	Samples    int    `json:"samples"`
	SHA256     string `json:"sha256"`
	Transcript string `json:"transcript"` // the LibriSpeech reference

	engine       string // the engine's text
	engineLang   string
	engineErrors int // as manifest.json counts them
}

// loadLibriSpeech reads the subset and the engine's transcripts.
func loadLibriSpeech(t *testing.T) (utts []lsUtterance, audioDir string, engineWords, engineErrors int) {
	t.Helper()

	var subset struct {
		AudioDir   string        `json:"audio_dir"`
		Utterances []lsUtterance `json:"utterances"`
	}

	readJSON(t, filepath.Join(librispeechDir, "subset.json"), &subset)

	var manifest struct {
		WERRule    string `json:"wer_rule"`
		Words      int    `json:"words"`
		Errors     int    `json:"errors"`
		Utterances []struct {
			ID       string `json:"id"`
			Text     string `json:"text"`
			Language string `json:"language"`
			Errors   int    `json:"errors"`
		} `json:"utterances"`
	}

	readJSON(t, filepath.Join(librispeechDir, "manifest.json"), &manifest)

	if len(subset.Utterances) != len(manifest.Utterances) {
		t.Fatalf("%d utterances in subset.json, %d in manifest.json", len(subset.Utterances), len(manifest.Utterances))
	}

	for i, m := range manifest.Utterances {
		u := &subset.Utterances[i]
		if u.ID != m.ID {
			t.Fatalf("utterance %d: %s in subset.json, %s in manifest.json", i, u.ID, m.ID)
		}

		u.engine, u.engineLang, u.engineErrors = m.Text, m.Language, m.Errors
	}

	t.Logf("WER rule: %s", manifest.WERRule)

	return subset.Utterances, filepath.Join("..", subset.AudioDir), manifest.Words, manifest.Errors
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	err = json.Unmarshal(raw, v)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

// wordErrors is the manifest's wer_rule: the word edit distance of hyp
// against ref, both split by eval.Words (case folded, split at every rune
// other than a letter, digit or apostrophe), and ref's word count.
func wordErrors(ref, hyp string) (errs, words int) {
	r := eval.Words(ref)

	return eval.EditDistance(r, eval.Words(hyp)), len(r)
}

// scores teacher-forces the beam search's transcript of the clip at path
// and the engine's text, spelt by the tokenizer, and returns both ranking
// scores under search.Default: Σ log p of the text tokens and EOS over the
// text tokens (F-026). The engine may have spelt its text otherwise, which
// can only score higher, so its score is a lower bound.
func scores(t *testing.T, m *Model, path, lang, engineText string) (ours, engine float64) {
	t.Helper()

	pcm, err := frontend.ReadWAV(path)
	if err != nil {
		t.Fatal(err)
	}

	emb, err := m.Embed(pcm)
	if err != nil {
		t.Fatal(err)
	}

	mem, err := m.dec.Memory(emb.Values, emb.Frames)
	if err != nil {
		t.Fatal(err)
	}

	r := beamDecode(t, m, pcm, "")
	seq := slices.Concat(m.dec.Prompt(), []int{langID(t, m, r.Language)}, r.Tokens, []int{m.dec.Tokenizer().EOSID()})

	eng, err := m.dec.Targets(lang, engineText)
	if err != nil {
		t.Fatal(err)
	}

	score := func(seq []int) float64 {
		f, err := m.dec.TeacherForce(mem, seq)
		if err != nil {
			t.Fatal(err)
		}

		var sum float64
		for _, lp := range f.LogProbs[1:] {
			sum += lp
		}

		return sum / float64(len(f.LogProbs)-2)
	}

	return score(seq), score(eng)
}

func langID(t *testing.T, m *Model, code string) int {
	t.Helper()

	for _, l := range m.dec.Languages() {
		if l.Code == code {
			return l.ID
		}
	}

	t.Fatalf("unknown language %q", code)

	return 0
}

// TestEvalLibriSpeech is `just eval-librispeech` (PLAN.md 6.6): the public
// pipeline's WER on the 200-utterance test-clean subset against the
// engine's, under one normalisation, and how often the transcripts are
// equal. It runs only with WHISTLE_EVAL_LIBRISPEECH set, and needs the
// model and the audio (`just oracle-librispeech-audio`).
func TestEvalLibriSpeech(t *testing.T) {
	if os.Getenv(librispeechEnv) == "" {
		t.Skipf("set %s=1 or run `just eval-librispeech`", librispeechEnv)
	}

	utts, audioDir, engineWords, engineErrors := loadLibriSpeech(t)

	_, err := os.Stat(audioDir)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing; run `just oracle-librispeech-audio`", audioDir)
	}

	m := loadModel(t)

	var (
		words, ours, engine, exact, sameWords, sameLang int
		audio, wall                                     time.Duration
		better, worse                                   []string
		searchMiss, modelPrefers                        int
	)

	for _, u := range utts {
		path := filepath.Join(audioDir, u.ID+".wav")

		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%v; run `just oracle-librispeech-audio`", err)
		}

		if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != u.SHA256 {
			t.Fatalf("%s: SHA-256 differs from subset.json; run `just oracle-librispeech-audio`", path)
		}

		start := time.Now()

		res, err := m.TranscribeWAV(path, TranscribeOptions{})
		if err != nil {
			t.Fatalf("%s: %v", u.ID, err)
		}

		wall += time.Since(start)
		audio += time.Duration(u.Samples) * time.Second / SampleRate

		e, n := wordErrors(u.Transcript, res.Text)
		ee, _ := wordErrors(u.Transcript, u.engine)

		if ee != u.engineErrors {
			t.Errorf("%s: the engine's text has %d errors under the rule, manifest.json says %d", u.ID, ee, u.engineErrors)
		}

		words += n
		ours += e
		engine += ee

		if res.Text == u.engine {
			exact++
		}

		if strings.Join(eval.Words(res.Text), " ") == strings.Join(eval.Words(u.engine), " ") {
			sameWords++
		}

		if res.Language == u.engineLang {
			sameLang++
		}

		switch {
		case e < ee:
			better = append(better, u.ID)
		case e > ee:
			worse = append(worse, u.ID)
		}

		if res.Text != u.engine {
			ours, eng := scores(t, m, path, u.engineLang, u.engine)
			if eng >= ours {
				searchMiss++
			} else {
				modelPrefers++
			}

			t.Logf("%s: errors ours %d, engine %d; score ours %.4f, the engine's text %.4f (%+.4f)\n  engine: %s\n  ours:   %s",
				u.ID, e, ee, ours, eng, eng-ours, u.engine, res.Text)
		}
	}

	if words != engineWords || engine != engineErrors {
		t.Errorf("the rule gives the engine %d errors in %d words, manifest.json %d in %d", engine, words, engineErrors,
			engineWords)
	}

	n := len(utts)
	ourWER, engineWER := 100*float64(ours)/float64(words), 100*float64(engine)/float64(words)
	diff := ourWER - engineWER

	t.Logf("LibriSpeech test-clean subset: %d utterances, %d reference words, %.1f s of audio", n, words, audio.Seconds())
	t.Logf("WER Go pipeline   %d/%d = %.2f %%", ours, words, ourWER)
	t.Logf("WER engine        %d/%d = %.2f %%", engine, words, engineWER)
	t.Logf("difference        %+.2f percentage points (bound ±%.1f)", diff, werTolerancePP)
	t.Logf("equal to the engine: text %d/%d (%.1f %%), words after the rule %d/%d (%.1f %%), language %d/%d", exact, n,
		100*float64(exact)/float64(n), sameWords, n, 100*float64(sameWords)/float64(n), sameLang, n)
	t.Logf("per utterance: fewer errors than the engine %d, more %d", len(better), len(worse))
	t.Logf("of the %d texts that differ, the engine's (canonical spelling) scores higher under our model on %d "+
		"(a search miss), ours on %d", n-exact, searchMiss, modelPrefers)
	t.Logf("speed: %.1f s for %.1f s of audio (real-time factor %.4f), %.0f ms per utterance", wall.Seconds(),
		audio.Seconds(), wall.Seconds()/audio.Seconds(), float64(wall.Milliseconds())/float64(n))

	if math.Abs(diff) > werTolerancePP {
		t.Errorf("WER %.2f %% vs the engine's %.2f %%: %+.2f points, outside ±%.1f", ourWER, engineWER, diff, werTolerancePP)
	}
}
