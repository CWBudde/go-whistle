package whistle_test

import (
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/cwbudde/go-whistle/whistle"
)

// The paths `whistle download` and the repository's fixtures use. The
// examples that load the model compile under go test but do not run.
const (
	modelPath = "models/whistle/whistle.cact"
	clipPath  = "testdata/audio/ls-other-3080-5040-0018.wav"
)

// Load the model once and transcribe a WAV file.
func Example() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := whistle.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	res, err := m.TranscribeWAV(clipPath, whistle.TranscribeOptions{})
	if err != nil {
		log.Print(err)

		return
	}

	fmt.Printf("[%s] %s\n", res.Language, res.Text)
}

// Force the language, favour a keyword and time every word.
func ExampleModel_TranscribeWAV() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := whistle.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	res, err := m.TranscribeWAV("testdata/audio/ls-other-3080-5040-0011.wav", whistle.TranscribeOptions{
		Language:       "en",
		Keywords:       []string{"Cowley's"},
		WordTimestamps: true,
	})
	if err != nil {
		log.Print(err)

		return
	}

	fmt.Println(res.Text)

	for _, w := range res.Words {
		fmt.Printf("%5.2f %5.2f %.2f %s\n", w.Start.Seconds(), w.End.Seconds(), w.Probability, w.Text)
	}
}

// Silence gives the empty result without decoding, here in the engine's
// JSON with word timestamps on.
func ExampleModel_Transcribe() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := whistle.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	pcm := make([]float32, 2*whistle.SampleRate) // 16 kHz mono in [-1, 1]

	res, err := m.Transcribe(pcm, whistle.TranscribeOptions{WordTimestamps: true})
	if err != nil {
		log.Print(err)

		return
	}

	js, err := res.MarshalJSON()
	if err != nil {
		log.Print(err)

		return
	}

	fmt.Println(string(js))
}

// A clip over 30 s fails with ErrTooLong; a Stream takes it.
func ExampleModel_Transcribe_tooLong() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := whistle.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	_, err = m.Transcribe(make([]float32, whistle.MaxSamples+1), whistle.TranscribeOptions{})
	fmt.Println(errors.Is(err, whistle.ErrTooLong), err)
}

// Feed audio a second at a time and print the words as they are committed.
func ExampleModel_NewStream() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := whistle.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	s, err := m.NewStream(whistle.StreamOptions{})
	if err != nil {
		log.Print(err)

		return
	}

	audio := make([]float32, 5*whistle.SampleRate) // live audio, here 5 s of silence

	for len(audio) > 0 {
		n := min(whistle.SampleRate, len(audio))

		r, err := s.Process(audio[:n])
		if err != nil {
			log.Print(err)

			return
		}

		audio = audio[n:]

		fmt.Printf("%.2f s: committed %q, pending %q\n", r.Received, r.Text, r.Pending)
	}

	fmt.Printf("stop: committed %q\n", s.Stop().Text)
}

// The encoder output: one row of 512 values per 80 ms frame.
func ExampleModel_Embed() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := whistle.Load(modelPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	e, err := m.Embed(make([]float32, whistle.SampleRate))
	if err != nil {
		log.Print(err)

		return
	}

	fmt.Println(e.Frames, e.Dim, len(e.Frame(0)))
}

// The engine's JSON, with and without word timestamps.
func ExampleResult_MarshalJSON() {
	res := whistle.Result{
		Text:     `He said "hi".`,
		Language: "en",
		Words: []whistle.Word{
			{Text: "He", Start: 320 * time.Millisecond, End: 480 * time.Millisecond, Probability: 0.98},
			{Text: "said", Start: 480 * time.Millisecond, End: 720 * time.Millisecond, Probability: 0.951},
			{Text: `"hi".`, Start: 720 * time.Millisecond, End: 1040 * time.Millisecond, Probability: 0.6125},
		},
		TTFT:      12345 * time.Microsecond,
		DecodeTPS: 1401.26,
	}

	js, _ := res.MarshalJSON()
	fmt.Println(string(js))

	js, _ = whistle.Silence().MarshalJSON()
	fmt.Println(string(js))

	js, _ = whistle.Result{}.MarshalJSON()
	fmt.Println(string(js))
	// Output:
	// {"text":"He said \"hi\".","language":"en","words":[{"word":"He","start":0.32,"end":0.48,"probability":0.980},{"word":"said","start":0.48,"end":0.72,"probability":0.951},{"word":"\"hi\".","start":0.72,"end":1.04,"probability":0.613}],"ttft_ms":12.3,"decode_tps":1401.3}
	// {"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}
	// {"text":"","language":"","ttft_ms":0.0,"decode_tps":0.0}
}

// Read the engine's own output back.
func ExampleResult_UnmarshalJSON() {
	var res whistle.Result

	err := res.UnmarshalJSON([]byte(`{"text":"Directed for your master.","language":"en","words":[` +
		`{"word":"Directed","start":0.56,"end":1.12,"probability":0.893}],"ttft_ms":7.4,"decode_tps":1433.6}`))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res.Text, res.Language, res.TTFT)
	fmt.Println(res.Words[0].Text, res.Words[0].Start, res.Words[0].End, res.Words[0].Probability)
	// Output:
	// Directed for your master. en 7.4ms
	// Directed 560ms 1.12s 0.893
}

// The engine's stream JSON of one call.
func ExampleStreamResult_MarshalJSON() {
	r := whistle.StreamResult{
		Text:     "Directed for",
		Words:    []whistle.Word{{Text: "Directed", Start: 560 * time.Millisecond, End: 1120 * time.Millisecond, Probability: 0.889}, {Text: "for", Start: 1120 * time.Millisecond, End: 1520 * time.Millisecond, Probability: 0.652}},
		Pending:  "your master.",
		Language: "en",
		Received: 2,
		Pass:     11500 * time.Microsecond,
	}

	js, _ := r.MarshalJSON()
	fmt.Println(string(js))
	// Output:
	// {"text":"Directed for","words":[{"word":"Directed","start":0.56,"end":1.12,"probability":0.889},{"word":"for","start":1.12,"end":1.52,"probability":0.652}],"pending":"your master.","language":"en","received":2.00,"pass_ms":11.5}
}

// Rows of an embedding.
func ExampleEmbedding_Frame() {
	e := whistle.Embedding{Values: []float32{1, 2, 3, 4, 5, 6}, Frames: 2, Dim: 3}

	for i := range e.Frames {
		fmt.Println(whistle.FrameDuration*time.Duration(i), e.Frame(i))
	}
	// Output:
	// 0s [1 2 3]
	// 80ms [4 5 6]
}
