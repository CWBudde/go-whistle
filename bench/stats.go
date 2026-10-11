// Package bench holds the results, the statistics and the report of the
// speed and accuracy benchmarks (PLAN.md 9.6 and the Phase 9 Accept,
// docs/FINDINGS.md F-040): the Go side measured by bench/cmd/whistle-bench,
// the engine side by tools/oracle's bench command, both rendered into
// bench/RESULTS.md. It has no model dependency, so the oracle tool can
// share its types.
package bench

import (
	"math"
	"slices"
)

// Summary is a sample set's size, median, 90th percentile and range.
type Summary struct {
	N      int     `json:"n"`
	Median float64 `json:"median"`
	P90    float64 `json:"p90"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

// Summarise is v's Summary. The median of an even count is the mean of the
// middle two; the 90th percentile is the nearest rank, the smallest value
// with at least 90 % of the samples at or below it.
func Summarise(v []float64) Summary {
	if len(v) == 0 {
		return Summary{}
	}

	s := slices.Sorted(slices.Values(v))
	n := len(s)

	med := s[n/2]
	if n%2 == 0 {
		med = (s[n/2-1] + s[n/2]) / 2
	}

	return Summary{N: n, Median: med, P90: s[nearestRank(n, 0.9)], Min: s[0], Max: s[n-1]}
}

// nearestRank is the index of the p-quantile of n sorted samples.
func nearestRank(n int, p float64) int {
	return max(int(math.Ceil(p*float64(n)))-1, 0)
}

// Samples collects one clip set's measurements, one entry per
// transcription, and the audio and wall time per pass for the real-time
// factor.
type Samples struct {
	TTFT, DecodeTPS, Wall []float64 // ms, tokens/s, ms
	passWall, passAudio   []float64 // s, per pass
}

// Add records one transcription of seconds of audio in pass.
func (s *Samples) Add(pass int, seconds, ttftMS, tps, wallMS float64) {
	s.TTFT = append(s.TTFT, ttftMS)
	s.DecodeTPS = append(s.DecodeTPS, tps)
	s.Wall = append(s.Wall, wallMS)

	for len(s.passWall) <= pass {
		s.passWall = append(s.passWall, 0)
		s.passAudio = append(s.passAudio, 0)
	}

	s.passWall[pass] += wallMS / 1000
	s.passAudio[pass] += seconds
}

// Set summarises the samples of a set of clips clips.
func (s *Samples) Set(clips int) Set {
	rtf := make([]float64, 0, len(s.passWall))

	var audio float64

	for i, w := range s.passWall {
		if s.passAudio[i] > 0 {
			rtf = append(rtf, w/s.passAudio[i])
			audio = s.passAudio[i]
		}
	}

	return Set{
		Clips: clips, Passes: len(rtf), AudioSeconds: audio,
		TTFT: Summarise(s.TTFT), DecodeTPS: Summarise(s.DecodeTPS), Wall: Summarise(s.Wall), RTF: Summarise(rtf),
	}
}

// Set is one clip set transcribed one way: every transcription pooled
// over the passes.
type Set struct {
	Clips        int     `json:"clips"`
	Passes       int     `json:"passes"`
	AudioSeconds float64 `json:"audio_seconds"` // per pass
	TTFT         Summary `json:"ttft_ms"`       // the engine's ttft_ms or Result.TTFT
	DecodeTPS    Summary `json:"decode_tps"`    // the engine's decode_tps or Result.DecodeTPS
	Wall         Summary `json:"wall_ms"`       // per transcription
	RTF          Summary `json:"rtf"`           // per pass: Σ wall / Σ audio
}
