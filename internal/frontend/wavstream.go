package frontend

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/cwbudde/algo-dsp/dsp/resample"
)

// fmtPrefix is how much of a fmt chunk parseFmt reads.
const fmtPrefix = 40

// WAVStream decodes a WAV file a block at a time: the same 16 kHz mono
// samples as DecodeWAV, bit for bit, without holding the file or its
// samples in memory. The resampler keeps its filter history between blocks,
// so the output does not depend on how it is read.
type WAVStream struct {
	Format

	r         io.ReaderAt
	off, end  int64 // the unread whole frames of the data chunk
	frameSize int

	// Resampling (nil when the file is at SampleRate): the zeros resampleWith
	// pads with, the raw outputs it skips and the outputs it keeps.
	res        *resample.Resampler
	lead, tail int
	skip, left int

	buf     []byte
	pending []float32
}

// OpenWAV parses the header of the size-byte WAV file r as ParseWAV does,
// with the same errors, and returns a stream positioned at its first sample.
func OpenWAV(r io.ReaderAt, size int64) (*WAVStream, error) {
	if size < 8 {
		return nil, fmt.Errorf("%w: %d bytes, no RIFF header", ErrTruncated, size)
	}

	head := make([]byte, min(size, 12))

	_, err := r.ReadAt(head, 0)
	if err != nil {
		return nil, fmt.Errorf("frontend: %w", err)
	}

	if string(head[:4]) != "RIFF" {
		return nil, ErrNotRIFF
	}

	riffSize := int64(binary.LittleEndian.Uint32(head[4:8]))
	body := min(size-8, riffSize) // the body ParseWAV walks, from offset 8

	if body < 4 || string(head[8:12]) != "WAVE" {
		return nil, ErrNotWAVE
	}

	var format *Format

	for pos := int64(4); pos+8 <= body; {
		var ch [8]byte

		_, err = r.ReadAt(ch[:], 8+pos)
		if err != nil {
			return nil, fmt.Errorf("frontend: %w", err)
		}

		id := string(ch[:4])
		csize := int64(binary.LittleEndian.Uint32(ch[4:]))
		pos += 8
		avail := min(pos+csize, body) - pos

		switch id {
		case "fmt ":
			b := make([]byte, min(avail, fmtPrefix)) // parseFmt reads no further

			_, err = r.ReadAt(b, 8+pos)
			if err != nil {
				return nil, fmt.Errorf("frontend: %w", err)
			}

			f, err := parseFmt(b)
			if err != nil {
				return nil, err
			}

			format = &f
		case "data":
			if format == nil {
				return nil, ErrDataBeforeFmt
			}

			return newWAVStream(*format, csize, r, 8+pos, avail)
		}

		pos += csize + csize&1
		if pos > riffSize {
			return nil, fmt.Errorf("%w: chunk %q ends past the RIFF chunk", ErrTruncated, id)
		}
	}

	return nil, ErrMissingChunk
}

// newWAVStream is newWAV for a data chunk of size bytes declared, avail of
// them in the file from off, plus resampleWith's set-up.
func newWAVStream(f Format, size int64, r io.ReaderAt, off, avail int64) (*WAVStream, error) {
	frameSize := int64(f.Channels * f.SampleWidth)

	n := min(avail, size/frameSize*frameSize)
	if f.SampleWidth > 1 && n%int64(f.SampleWidth) != 0 {
		return nil, fmt.Errorf("%w: data ends inside a %d-byte sample", ErrTruncated, f.SampleWidth)
	}

	frames := n / frameSize
	s := &WAVStream{Format: f, r: r, off: off, end: off + frames*frameSize, frameSize: int(frameSize)}

	if f.SampleRate == SampleRate || frames == 0 {
		return s, nil
	}

	if f.SampleRate < minRate {
		return nil, fmt.Errorf("%w: %d Hz is below %d Hz", ErrSampleRate, f.SampleRate, minRate)
	}

	scaled := math.Round(float64(frames) * SampleRate / float64(f.SampleRate))
	if scaled > math.MaxInt32 {
		return nil, fmt.Errorf("frontend: %d samples at %d Hz resample to %.0f, more than %d", frames,
			f.SampleRate, scaled, math.MaxInt32)
	}

	res, err := newResampler(f.SampleRate, hqSpec)
	if err != nil {
		return nil, err
	}

	s.res, s.left = res, int(scaled)
	if s.left == 0 {
		return s, nil
	}

	up, down := res.Ratio()
	s.lead, s.skip = alignment(res.TapsPerPhase()*up, up, down)
	total := max(int64(s.lead)+frames, ceilDiv(int64(s.skip+s.left)*int64(down), int64(up))+1)
	s.tail = int(total - int64(s.lead) - frames)

	return s, nil
}

// Frames is the number of sample frames in the file, at its own rate.
func (s *WAVStream) Frames() int64 { return (s.end - s.off) / int64(s.frameSize) }

// Read returns the next at most n samples at SampleRate, fewer only at the
// end, and io.EOF once every sample has been returned.
func (s *WAVStream) Read(n int) ([]float32, error) {
	if n < 1 {
		return nil, fmt.Errorf("frontend: read of %d samples", n)
	}

	for len(s.pending) < n && s.more() {
		err := s.fill(n)
		if err != nil {
			return nil, err
		}
	}

	if len(s.pending) == 0 {
		return nil, io.EOF
	}

	k := min(n, len(s.pending))
	out := slices.Clone(s.pending[:k])
	s.pending = s.pending[k:]

	return out, nil
}

// more reports whether fill has anything left to decode.
func (s *WAVStream) more() bool {
	if s.res == nil {
		return s.off < s.end
	}

	return s.left > 0
}

// fill decodes about n more output samples into pending.
func (s *WAVStream) fill(n int) error {
	frames := int64(n)

	if s.res != nil {
		up, down := s.res.Ratio()
		frames = int64(n)*int64(down)/int64(up) + 1
	}

	frames = min(frames, (s.end-s.off)/int64(s.frameSize))

	mono, err := s.decode(int(frames))
	if err != nil {
		return err
	}

	if s.res == nil {
		s.pending = append(s.pending, mono...)

		return nil
	}

	// resampleWith's input: lead zeros, the samples, then tail zeros.
	in := make([]float64, s.lead, s.lead+len(mono)+s.tail)
	s.lead = 0

	for _, v := range mono {
		in = append(in, float64(v))
	}

	last := s.off == s.end
	if last {
		in = append(in, make([]float64, s.tail)...)
		s.tail = 0
	}

	raw := s.res.Process(in)

	drop := min(s.skip, len(raw))
	raw, s.skip = raw[drop:], s.skip-drop

	keep := min(s.left, len(raw))
	for _, v := range raw[:keep] {
		s.pending = append(s.pending, float32(v))
	}

	s.left -= keep

	if last && s.left > 0 {
		return fmt.Errorf("frontend: resampler stopped %d samples short", s.left)
	}

	return nil
}

// decode reads the next n frames and mixes them to mono as Mono does.
func (s *WAVStream) decode(n int) ([]float32, error) {
	size := n * s.frameSize
	if cap(s.buf) < size {
		s.buf = make([]byte, size)
	}

	b := s.buf[:size]

	_, err := s.r.ReadAt(b, s.off)
	if err != nil {
		return nil, fmt.Errorf("frontend: %w", err)
	}

	s.off += int64(size)

	return (&WAV{Format: s.Format, Data: b}).Mono(), nil
}
