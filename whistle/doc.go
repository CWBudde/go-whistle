// Package whistle is a pure-Go port of Whistle, the Cactus Compute speech-to-text model.
//
// Load opens whistle.cact (or the float32 whistle.safetensors checkpoint) and returns a Model:
//
//	m, err := whistle.Load("models/whistle/whistle.cact")
//	if err != nil { … }
//	defer m.Close()
//
//	res, err := m.TranscribeWAV("clip.wav", whistle.TranscribeOptions{})
//	fmt.Println(res.Text, res.Language)
//
// Transcribe takes 16 kHz mono float32 samples, at most 30 s; Embed returns the encoder output per 80 ms frame, as
// the engine's needle_embed. Result marshals to the engine's JSON byte for byte. Transcribe gates silence and decodes
// with the engine's beam search, as the engine does, with keywords, a decoder depth and word timestamps on request.
//
// NewStream is live transcription without a length limit, the engine's needle_stream_transcribe_process and _stop:
//
//	s, err := m.NewStream(whistle.StreamOptions{})
//	for chunk := range chunks { // about a second each
//		r, err := s.Process(chunk)
//		fmt.Print(r.Text, " ") // the words two passes agreed on
//	}
//	fmt.Println(s.Stop().Text) // the rest
//
// A Model is safe for concurrent use; a Stream belongs to one goroutine.
package whistle
