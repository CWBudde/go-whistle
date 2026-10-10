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
// the engine's needle_embed. Result marshals to the engine's JSON byte for byte. Decoding is greedy for now; beam
// search, keywords, depth, word timestamps and streaming follow phase by phase (see PLAN.md).
package whistle
