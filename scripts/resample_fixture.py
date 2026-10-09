"""Generate the soxr reference fixture for internal/frontend resampling.

whistle.py `_read_wav` resamples every non-16 kHz WAV with
`soxr.resample(mono, rate, 16000, quality="HQ")`. This script runs exactly
that call on a set of test signals and stores the outputs, so the Go tests can
measure how far the algo-dsp resampler is from soxr (docs/FINDINGS.md F-010).

Inputs are not stored: the Go test regenerates them with the same formulas
and checks them against the SHA-256 recorded here. Every input is quantised
to 16-bit PCM and divided by 32768, as `_read_wav` does for a 16-bit mono WAV,
so libm differences in sin() never reach the comparison:

    tone   round(32767 * a * sin(2*pi*f*i/rate))
    chirp  round(32767 * a * sin(2*pi*(f0*t + (f1-f0)*t*t/(2*T)))), t = i/rate
    noise  round(32767 * a * (2*u - 1)), u from splitmix64 (53-bit mantissa)

Rounding is half to even (Python round, Go math.RoundToEven).

16-bit quantisation noise hides anything below about -95 dB, so the alias
cases feed float32 tones just above 8 kHz instead,
float32(a * sin(2*pi*f*i/rate)), and store only soxr's output level in dB
relative to the input (over the same interior the Go test uses).

Deterministic: fixed seed, no timestamps. Regenerate with

    just resample-fixture

which runs

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/resample_fixture.py [--out testdata/frontend/soxr_hq.safetensors]
"""

import argparse
import hashlib
import json
import math
import os

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

import numpy as np  # noqa: E402
import soxr  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

TARGET = 16000
SECONDS = 0.4
AMPLITUDE = 0.5
SEED = 20261010
RATES = (8000, 22050, 44100, 48000)
NEAR_NYQUIST = 0.75  # fraction of the lower Nyquist rate; internal/frontend nearNyquist

ALIAS_RATES = (22050, 44100, 48000)
ALIAS_FREQS = (8100, 8200, 8400, 9500)
EDGE = 400  # output samples dropped at each end; internal/frontend edge

MASK64 = (1 << 64) - 1


def splitmix64(state):
    """Return (next state, output) of splitmix64."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def pcm(values):
    """Quantise [-1, 1] floats to int16 the way the Go test does."""
    return np.array([round(32767 * v) for v in values], dtype=np.int16)


def tone(rate, n, freq):
    return pcm(AMPLITUDE * math.sin(2 * math.pi * freq * i / rate) for i in range(n))


def chirp(rate, n, f0, f1):
    duration = n / rate
    out = []
    for i in range(n):
        t = i / rate
        out.append(AMPLITUDE * math.sin(2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration))))
    return pcm(out)


def noise(n, seed):
    state, out = seed, []
    for _ in range(n):
        state, z = splitmix64(state)
        out.append(AMPLITUDE * (2 * ((z >> 11) * 2.0**-53) - 1))
    return pcm(out)


def band(freq, nyquist):
    """Where a tone sits relative to the lower of the two Nyquist rates: the
    top quarter below it is near-nyquist, anything above 8 kHz is aliased."""
    if freq > TARGET / 2:
        return "stopband"
    return "near-nyquist" if freq >= NEAR_NYQUIST * nyquist else "passband"


def cases(rate):
    """Yield (name, band, params, int16 input) for one input rate. band is
    passband, near-nyquist, stopband or broadband (split by frequency in the
    Go test)."""
    n = int(SECONDS * rate)
    nyquist = min(rate, TARGET) / 2
    for freq in (220, 1000, 3500, 7000, 9500):
        if freq < 0.95 * rate / 2:
            yield f"tone{freq}", band(freq, nyquist), {"kind": "tone", "freq": freq}, tone(rate, n, freq)
    f1 = 0.98 * nyquist
    yield "chirp", "broadband", {"kind": "chirp", "f0": 50.0, "f1": f1}, chirp(rate, n, 50.0, f1)
    seed = SEED + rate
    yield "noise", "broadband", {"kind": "noise", "seed": seed}, noise(n, seed)


def alias_level(rate, freq):
    """soxr's output level in dB for a float32 tone above 8 kHz."""
    n = int(SECONDS * rate)
    x = np.array([AMPLITUDE * math.sin(2 * math.pi * freq * i / rate) for i in range(n)], dtype=np.float32)
    y = soxr.resample(x, rate, TARGET, quality="HQ")[EDGE:-EDGE].astype(np.float64)
    return float(10 * np.log10(np.mean(y * y) / (AMPLITUDE * AMPLITUDE / 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="testdata/frontend/soxr_hq.safetensors")
    args = parser.parse_args()

    tensors, index = {}, []
    for rate in RATES:
        for name, band, params, samples in cases(rate):
            mono = samples.astype(np.float32) / np.float32(32768)
            out = soxr.resample(mono, rate, TARGET, quality="HQ")
            key = f"{rate}/{name}"
            tensors[key] = np.ascontiguousarray(out, dtype=np.float32)
            index.append({
                "key": key,
                "rate": rate,
                "band": band,
                "amplitude": AMPLITUDE,
                "samples": int(samples.size),
                "input_sha256": hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest(),
                **params,
            })
            print(f"{key:16s} {band:12s} in {samples.size:6d} out {out.size:5d}")

    alias = []
    for rate in ALIAS_RATES:
        for freq in ALIAS_FREQS:
            level = round(alias_level(rate, freq), 2)
            alias.append({"rate": rate, "freq": freq, "amplitude": AMPLITUDE, "samples": int(SECONDS * rate), "soxr_level_db": level})
            print(f"alias {rate}/{freq}: soxr {level:.2f} dB")

    # One metadata key: safetensors writes __metadata__ in hash order, which
    # would make the file differ from run to run.
    fixture = {
        "generator": "scripts/resample_fixture.py",
        "soxr": soxr.__version__,
        "quality": "HQ",
        "target_rate": TARGET,
        "cases": index,
        "alias": alias,
    }
    metadata = {"fixture": json.dumps(fixture, separators=(",", ":"))}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    print(f"wrote {args.out}: {len(tensors)} outputs, {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
