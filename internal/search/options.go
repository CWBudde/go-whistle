package search

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ErrOptions reports an unset or out-of-range field in Options.
var ErrOptions = errors.New("search: invalid options")

// Norm is the length normalisation that ranks hypotheses of different
// lengths: the score is the summed log-probability over a penalty of the
// length L (Length).
type Norm int

const (
	// NormNone ranks by the summed log-probability (penalty 1).
	NormNone Norm = iota + 1
	// NormLen divides by L.
	NormLen
	// NormPow divides by L^α.
	NormPow
	// NormGNMT divides by ((5 + L) / 6)^α (Wu et al. 2016).
	NormGNMT
)

// Length is what L counts.
type Length int

const (
	// LenText counts the text tokens, without EOS (Whisper's ranker).
	LenText Length = iota + 1
	// LenTextEOS counts the text tokens and EOS once it is emitted.
	LenTextEOS
	// LenSeq counts the whole sequence: prompt, language token, text
	// tokens and EOS once it is emitted.
	LenSeq
)

// Stop is the rule that ends the search.
type Stop int

const (
	// StopAll keeps finished hypotheses in the beam, ranked against the
	// live ones by their normalised score, and ends when all Beams
	// hypotheses in the beam are finished.
	StopAll Stop = iota + 1
	// StopBest ends when the best finished hypothesis' normalised score
	// is at least the best live one's.
	StopBest
	// StopK ends as soon as Beams hypotheses have finished (Whisper with
	// patience 1).
	StopK
	// StopHF keeps the best Beams finished hypotheses and ends when there
	// are Beams of them and the worst is at least the best live one's
	// normalised score (Hugging Face, early_stopping=False).
	StopHF
)

// Admit is which EOS candidates of a step finish a hypothesis, for the
// rules that keep finished hypotheses out of the beam (all but StopAll).
type Admit int

const (
	// AdmitTopK admits an EOS candidate ranked among the step's best
	// Beams candidates (Hugging Face).
	AdmitTopK Admit = iota + 1
	// AdmitAboveLive admits every EOS candidate ranked above the last
	// live hypothesis kept (Whisper).
	AdmitAboveLive
)

// LangStep is how the language token is chosen when no language is
// forced.
type LangStep int

const (
	// LangArgmax takes the most likely language token and leaves its
	// log-probability out of the score (decoder.Greedy, F-024).
	LangArgmax LangStep = iota + 1
	// LangArgmaxScored takes the most likely language token and adds its
	// log-probability to the score.
	LangArgmaxScored
	// LangBeam starts one hypothesis per language token, up to Beams, with
	// its log-probability in the score.
	LangBeam
)

// Adjust changes the next-token log-probabilities of one live hypothesis
// before the candidates are ranked; tokens is its text so far (after the
// language token). The search ranks by the adjusted values, while
// Result.LogProbs keeps the model's. It is the hook for keyword biasing
// (PLAN.md 6.3); the search calls it from one goroutine at a time, and it
// must not keep logp or tokens.
type Adjust func(tokens []int, logp []float64)

// Options configures Beam. The zero value is invalid; start from Default.
type Options struct {
	Beams     int     // hypotheses kept per step
	MaxTokens int     // tokens decoded after the prompt, the language token included; 0 is the decoder's (320)
	Language  string  // a forced language code; empty detects it
	Norm      Norm    // length normalisation
	Alpha     float64 // the exponent of NormPow and NormGNMT
	Length    Length  // what the normalisation's L counts
	Stop      Stop    // when the search ends
	Admit     Admit   // which EOS candidates finish a hypothesis (not StopAll)
	Lang      LangStep
	Adjust    Adjust // nil for none
}

// Default is the variant the evidence on the 32 fixture clips picked
// (docs/FINDINGS.md F-026): 5 beams; Whisper's search, which ends once 5
// hypotheses have finished, admits every EOS candidate ranked above the
// last live hypothesis kept and ranks the finished ones by their summed
// log-probability over the number of text tokens; the language token by
// argmax. It reproduces the engine's transcript on 31 of 32 clips; the
// variants that tie with it (L with EOS, a scored or beamed language
// step) give the same transcripts on all 32.
func Default() Options {
	return Options{
		Beams: 5, Norm: NormLen, Alpha: 1, Length: LenText, Stop: StopK, Admit: AdmitAboveLive, Lang: LangArgmax,
	}
}

// Validate reports an unset or unknown field as ErrOptions.
func (o Options) Validate() error {
	switch {
	case o.Beams < 1:
		return fmt.Errorf("%w: %d beams", ErrOptions, o.Beams)
	case o.MaxTokens < 0:
		return fmt.Errorf("%w: max tokens %d", ErrOptions, o.MaxTokens)
	case o.Norm < NormNone || o.Norm > NormGNMT:
		return fmt.Errorf("%w: norm %d", ErrOptions, o.Norm)
	case o.Length < LenText || o.Length > LenSeq:
		return fmt.Errorf("%w: length %d", ErrOptions, o.Length)
	case o.Stop < StopAll || o.Stop > StopHF:
		return fmt.Errorf("%w: stop %d", ErrOptions, o.Stop)
	case o.Admit < AdmitTopK || o.Admit > AdmitAboveLive:
		return fmt.Errorf("%w: admit %d", ErrOptions, o.Admit)
	case o.Lang < LangArgmax || o.Lang > LangBeam:
		return fmt.Errorf("%w: language step %d", ErrOptions, o.Lang)
	case math.IsNaN(o.Alpha) || math.IsInf(o.Alpha, 0):
		return fmt.Errorf("%w: alpha %v", ErrOptions, o.Alpha)
	}

	return nil
}

var (
	normNames  = []string{"none", "len", "pow", "gnmt"}
	lenNames   = []string{"text", "text+eos", "seq"}
	stopNames  = []string{"all", "best", "k", "hf"}
	admitNames = []string{"topk", "above-live"}
	langNames  = []string{"argmax", "argmax-scored", "beam"}
)

func name(names []string, v int) string {
	if v < 1 || v > len(names) {
		return "?" + strconv.Itoa(v)
	}

	return names[v-1]
}

// String names the variant, e.g. "norm=pow α=0.6 len=text+eos stop=k
// admit=above-live lang=argmax beams=5"; the hook and the forced language
// are left out.
func (o Options) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "norm=%s", name(normNames, int(o.Norm)))

	if o.Norm == NormPow || o.Norm == NormGNMT {
		fmt.Fprintf(&b, " α=%g", o.Alpha)
	}

	fmt.Fprintf(&b, " len=%s stop=%s", name(lenNames, int(o.Length)), name(stopNames, int(o.Stop)))

	if o.Stop != StopAll {
		fmt.Fprintf(&b, " admit=%s", name(admitNames, int(o.Admit)))
	}

	fmt.Fprintf(&b, " lang=%s beams=%d", name(langNames, int(o.Lang)), o.Beams)

	if o.MaxTokens > 0 {
		fmt.Fprintf(&b, " max=%d", o.MaxTokens)
	}

	return b.String()
}

// penalty is the divisor of a summed log-probability at length l.
func (o Options) penalty(l int) float64 {
	switch o.Norm {
	case NormLen:
		return float64(max(l, 1))
	case NormPow:
		return math.Pow(float64(max(l, 1)), o.Alpha)
	case NormGNMT:
		return math.Pow((5+float64(l))/6, o.Alpha)
	default:
		return 1
	}
}
