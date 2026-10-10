set shell := ["bash", "-uc"]

# Default recipe - show available commands
default:
    @just --list

# Format all code using treefmt
fmt:
    treefmt --allow-missing-formatter

# Check if code is formatted correctly
check-formatted:
    treefmt --allow-missing-formatter --fail-on-change

# Run linters
lint:
    golangci-lint run --timeout=2m ./...

# Run linters with auto-fix
lint-fix:
    golangci-lint run --fix --timeout=2m ./...

# Ensure go.mod is tidy
check-tidy:
    go mod tidy
    git diff --exit-code go.mod

# Run all tests
test:
    go test -v ./...

# Run tests with race detector
test-race:
    go test -race ./...

# Run tests with coverage report
test-coverage:
    go test -v -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Run all checks (formatting, linting, tests, tidiness)
ci: check-formatted test lint check-tidy

# Fetch the pinned Needle 3 engine (test oracle) into engine/<platform>/
fetch-engine *args:
    scripts/fetch-engine.sh {{args}}

# Snapshot upstream cactus-compute/needle at the pinned commit into original/needle
fetch-upstream:
    scripts/fetch-upstream.sh

# Fetch the oracle clip corpus (LibriSpeech + FLEURS excerpts) into testdata/audio
oracle-clips:
    go run -tags oracle ./tools/oracle clips

# Record Needle 3 engine transcripts and embeddings for the corpus into testdata/oracle
oracle-run:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle run

# Record Needle 3 engine embeddings of the synthetic encoder probes into testdata/oracle/probes
oracle-probes:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle probes

# Record Needle 3 engine transcripts at --audio-depth 2..7 for the corpus into testdata/oracle/depth
oracle-depth:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle depth

# Record Needle 3 engine transcripts with and without keyword files into testdata/oracle/keywords
oracle-keywords:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle keywords

# Record the Needle 3 engine's empty-JSON silence sweep into testdata/oracle/silence
oracle-silence:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle silence

# Record the Needle 3 engine's length limits (0 … 480,001 samples) into testdata/oracle/limits
oracle-limits:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle limits

# Fetch the seeded LibriSpeech test-clean subset (200 utterances) into models/corpus/librispeech
oracle-librispeech-audio:
    go run -tags oracle ./tools/oracle librispeech-audio

# Record Needle 3 engine transcripts of the LibriSpeech subset into testdata/oracle/librispeech (fetches the audio first)
oracle-librispeech:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 CGO_ENABLED=0 go run -tags oracle ./tools/oracle librispeech

# Regenerate the sentencepiece fixture for internal/tokenizer (testdata/tokenizer); --needle3 <tokenizer.model> cross-checks
tokenizer-fixture *args:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python scripts/tokenizer_fixture.py {{args}}

# Regenerate the shared Needle 3 block fixtures for internal/needle (testdata/needle); --f64-check compares against float64
needle-fixtures *args:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python scripts/needle_fixtures.py {{args}}

# Regenerate the soxr-HQ resampling fixture for internal/frontend (testdata/frontend)
resample-fixture *args:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python scripts/resample_fixture.py {{args}}

# Regenerate the numpy STFT fixture for internal/frontend (testdata/frontend)
mel-fixture *args:
    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python scripts/mel_fixture.py {{args}}

# Search the encoder hypotheses against the committed needle_embed rows (WHISTLE_PROBE_SRC/BEAM/TOP/CLIPS/KEY)
encoder-probe *args:
    go test -tags oracle -count=1 -v -timeout 60m -run 'TestProbe(Baselines|Search)$' ./internal/encoder/ {{args}}

# Search the decoder hypotheses by teacher forcing the committed engine transcripts, then decode every clip greedily (WHISTLE_PROBE_SRC/BEAM/TOP/KEY)
decoder-probe *args:
    go test -tags oracle -count=1 -v -timeout 60m -run 'TestProbe(Baselines|Search|Greedy)$' ./internal/decoder/ {{args}}

# Score the beam search variants against the committed engine transcripts and report the leader (WHISTLE_PROBE_TOP/KEY)
search-probe *args:
    go test -tags oracle -count=1 -v -timeout 60m -run 'TestProbeBeam$' ./internal/search/ {{args}}

# Clean build artifacts
clean:
    rm -rf bin coverage.out coverage.html
