# go-whistle

A pure-Go port of **Whistle**, Cactus Compute's 16.9 MB on-device speech-to-text model (55M parameters, 7 languages:
en, de, fr, es, it, nl, pl).

> **Status: planning.** There is no working code yet. See [PLAN.md](PLAN.md) for the roadmap.

## Why this is a re-implementation, not a translation

In [cactus-compute/needle](https://github.com/cactus-compute/needle), Whistle is a thin Python wrapper around a
closed-source C++ engine. So this port rebuilds the model from three sources:

- the published weights on [Hugging Face](https://huggingface.co/Cactus-Compute/whistle): `whistle.safetensors` (FP32)
  and `whistle.cact` (2/4-bit quantised)
- the open JAX reference for the shared Needle 3 blocks
- the prebuilt engine, which is used only as a test oracle for encoder embeddings and transcripts

## Goals

- No cgo, no Python and no ONNX at runtime
- Load both `whistle.safetensors` and the compact `whistle.cact`
- An API and JSON output compatible with upstream `needle.transcribe`, including language detection, keyword biasing,
  word timestamps, streaming and embeddings
- Transcripts that match the reference engine on a fixture set, and competitive CPU speed

## License

Apache-2.0, the same as the upstream code and weights. Whistle and Needle are by
[Cactus Compute](https://cactuscompute.com); this project is not affiliated with them.
