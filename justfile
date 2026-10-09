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

# Clean build artifacts
clean:
    rm -rf bin coverage.out coverage.html
