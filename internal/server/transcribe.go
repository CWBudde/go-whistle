package server

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/cwbudde/go-whistle/internal/frontend"
	"github.com/cwbudde/go-whistle/internal/pcm"
	"github.com/cwbudde/go-whistle/whistle"
)

// formatWAV is the /transcribe body format without ?format=.
const formatWAV = "wav"

// transcribe is POST /transcribe.
func (s *Server) transcribe(w http.ResponseWriter, r *http.Request) {
	opts, format, err := transcribeQuery(r.URL.Query())
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())

		return
	}

	// A clip under 160 samples gives the silent result after the option
	// checks (whistle.Transcribe), so an empty call validates the options.
	_, err = s.m.Transcribe(nil, opts)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())

		return
	}

	// Admitted requests hold their audio in memory: at most the ones that
	// run and wait.
	if s.admitted.Add(1) > int64(s.cfg.Concurrency+s.cfg.Queue) {
		s.admitted.Add(-1)
		s.busy(w)

		return
	}

	defer s.admitted.Add(-1)

	x, status, err := s.readAudio(w, r, format)
	if err != nil {
		writeError(w, status, err.Error())

		return
	}

	if len(x) > whistle.MaxSamples {
		writeError(w, http.StatusUnprocessableEntity, tooLong(len(x)))

		return
	}

	err = s.gate.enter(r.Context(), true)
	if err != nil {
		s.busy(w)

		return
	}

	res, err := func() (whistle.Result, error) {
		defer s.gate.leave()

		return s.m.Transcribe(x, opts)
	}()
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, whistle.ErrTooLong) {
			status = http.StatusUnprocessableEntity
		}

		writeError(w, status, err.Error())

		return
	}

	line, err := res.MarshalJSON()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())

		return
	}

	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(append(line, '\n')) // #nosec G705 -- the engine's JSON, served as application/json
}

// busy answers 503: the passes run and the queue is full.
func (s *Server) busy(w http.ResponseWriter) {
	w.Header().Set("Retry-After", "1")
	writeError(w, http.StatusServiceUnavailable,
		fmt.Sprintf("%d passes run and %d requests wait; try again", s.cfg.Concurrency, s.cfg.Queue))
}

// tooLong is the 422 message for n samples: the engine's error, as
// whistle.Transcribe words it, and where longer audio goes.
func tooLong(n int) string {
	return fmt.Sprintf("%v: %d samples, at most %d (/transcribe takes one pass of at most 30 s; /stream takes longer audio)",
		whistle.ErrTooLong, n, whistle.MaxSamples)
}

// readAudio reads the body, at most MaxBody bytes within ReadTimeout, as
// format and returns its 16 kHz mono samples, or the status and error to
// answer with.
func (s *Server) readAudio(w http.ResponseWriter, r *http.Request, format string) ([]float32, int, error) {
	tooBig := fmt.Errorf("the body is over the limit of %d bytes", s.cfg.MaxBody)

	if r.ContentLength > s.cfg.MaxBody {
		return nil, http.StatusRequestEntityTooLarge, tooBig
	}

	_ = http.NewResponseController(w).SetReadDeadline(time.Now().Add(s.cfg.ReadTimeout))

	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.cfg.MaxBody))
	if err != nil {
		var mb *http.MaxBytesError
		if errors.As(err, &mb) {
			return nil, http.StatusRequestEntityTooLarge, tooBig
		}

		if errors.Is(err, os.ErrDeadlineExceeded) {
			return nil, http.StatusRequestTimeout, fmt.Errorf("the body took over %v", s.cfg.ReadTimeout)
		}

		return nil, http.StatusBadRequest, fmt.Errorf("read the body: %w", err)
	}

	var x []float32

	if format == formatWAV {
		var status int

		x, status, err = decodeWAV(data)
		if status != 0 {
			return nil, status, err
		}
	} else {
		width, _ := pcm.Width(format) // transcribeQuery checked it

		// The length is known from the body, so audio over 30 s gets its
		// 422 before Decode allocates its samples (PR #35 review); a body
		// that ends inside a sample gets Decode's 400 first.
		if n := len(data) / width; len(data)%width == 0 && n > whistle.MaxSamples {
			return nil, http.StatusUnprocessableEntity, errors.New(tooLong(n))
		}

		x, err = pcm.Decode(data, width)
	}

	if err != nil {
		return nil, http.StatusBadRequest, fmt.Errorf("%s body: %w", format, err)
	}

	if len(x) == 0 {
		return nil, http.StatusBadRequest, fmt.Errorf("the body %w", whistle.ErrNoAudio)
	}

	return x, 0, nil
}

// decodeWAV decodes a WAV body to 16 kHz mono a second at a time, which
// gives frontend.DecodeWAV's samples, and stops with 422 once the audio is
// over whistle.MaxSamples. A long file at a low rate (16 MiB of 1 kHz 8-bit
// is 268 million samples at 16 kHz) is thus refused before it is resampled
// whole (PR #35 review). A format error returns status 0 and the error.
func decodeWAV(data []byte) ([]float32, int, error) {
	st, err := frontend.OpenWAV(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, 0, err
	}

	frames := st.Frames() // before Read, which consumes them

	var x []float32

	for {
		chunk, err := st.Read(whistle.SampleRate)
		x = append(x, chunk...)

		if len(x) > whistle.MaxSamples {
			return nil, http.StatusUnprocessableEntity, errors.New(tooLong(wavSamples(frames, st.SampleRate)))
		}

		if errors.Is(err, io.EOF) {
			return x, 0, nil
		}

		if err != nil {
			return nil, 0, err
		}
	}
}

// wavSamples is the number of 16 kHz samples frames at rate decode to: the
// resampled length rounds, as in frontend.OpenWAV, so the 422 names the
// count the whole file gives, not the part decodeWAV read before it stopped
// (PR #35 review).
func wavSamples(frames int64, rate int) int {
	if rate == whistle.SampleRate {
		return int(frames)
	}

	return int(math.Round(float64(frames) * whistle.SampleRate / float64(rate)))
}

// transcribeQuery reads the options and the body format of /transcribe.
func transcribeQuery(q url.Values) (whistle.TranscribeOptions, string, error) {
	err := checkQuery(q, "format", "language", "keywords", "depth", "word_timestamps")
	if err != nil {
		return whistle.TranscribeOptions{}, "", err
	}

	format := formatWAV
	if f := q.Get("format"); f != "" && f != formatWAV {
		_, err = pcm.Width(f)
		if err != nil {
			return whistle.TranscribeOptions{}, "", fmt.Errorf("%w, or wav", err)
		}

		format = f
	}

	opts := whistle.TranscribeOptions{Language: q.Get("language"), Keywords: q["keywords"]}

	if d := q.Get("depth"); d != "" {
		opts.Depth, err = strconv.Atoi(d)
		if err != nil {
			return whistle.TranscribeOptions{}, "", fmt.Errorf("depth %q: want a number of decoder layers, 2..8", d)
		}
	}

	if v, ok := q["word_timestamps"]; ok && v[0] != "" {
		opts.WordTimestamps, err = strconv.ParseBool(v[0])
		if err != nil {
			return whistle.TranscribeOptions{}, "", fmt.Errorf("word_timestamps %q: want true or false", v[0])
		}
	} else {
		opts.WordTimestamps = ok
	}

	return opts, format, nil
}

// checkQuery rejects a parameter outside known, and a repeat of any but
// keywords.
func checkQuery(q url.Values, known ...string) error {
	for k, v := range q {
		if !slices.Contains(known, k) {
			return fmt.Errorf("unknown parameter %q; this endpoint takes %v", k, known)
		}

		if len(v) > 1 && k != "keywords" {
			return fmt.Errorf("parameter %q given %d times", k, len(v))
		}
	}

	return nil
}
