"""Generate the numpy STFT reference fixture for internal/frontend.

Whistle's front end frames 16 kHz audio with a 400-sample Hann window
zero-padded to 512 and a hop of 160 (docs/FINDINGS.md F-012). The window
shape and the framing are open hypotheses, so this script computes the power
spectrum |rfft|^2 for every window x padding variant the Go side implements,
with plain numpy and none of the Go framing tricks:

    window   hann-periodic   np.hanning(401)[:400]   0.5 - 0.5 cos(2 pi k / 400)
             hann-symmetric  np.hanning(400)         0.5 - 0.5 cos(2 pi k / 399)
    padding  reflect / zero  np.pad(x, 256, "reflect" / "constant"); frame i is
                             xp[i*160 : i*160 + 512] times the window centred in
                             512 (56 zeros each side, as torch.stft pads it);
                             n // 160 frames
             none            frame i is x[i*160 : i*160 + 400] times the
                             window, zero-padded to 512 by rfft(n=512);
                             1 + (n - 400) // 160 frames

Everything is float64; the stored power is rounded once to float32.

The inputs are stored too (float32): a 4321-sample mix of two tones and
seeded noise, which is not a multiple of the hop, and a 300-sample noise
burst, where reflect padding mirrors more than half the signal.

Deterministic: fixed seed, no timestamps, one metadata key. Regenerate with

    just mel-fixture

which runs

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/mel_fixture.py [--out testdata/frontend/stft.safetensors]
"""

import argparse
import json
import os

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

import numpy as np  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

RATE = 16000
NFFT = 512
WIN = 400
HOP = 160
SEED = 20261010

WINDOWS = {
    "hann-periodic": np.hanning(WIN + 1)[:WIN],
    "hann-symmetric": np.hanning(WIN),
}
PADDINGS = ("reflect", "zero", "none")


def signals():
    rng = np.random.default_rng(SEED)
    t = np.arange(4321) / RATE
    mix = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t) + 0.05 * rng.standard_normal(t.size)
    burst = 0.2 * rng.standard_normal(300)
    return {"mix": mix.astype(np.float32), "burst": burst.astype(np.float32)}


def power(x, win, padding):
    x = x.astype(np.float64)
    if padding == "none":
        frames = 1 + (x.size - WIN) // HOP if x.size >= WIN else 0
        rows = [x[i * HOP : i * HOP + WIN] * win for i in range(frames)]
    else:
        mode = "reflect" if padding == "reflect" else "constant"
        xp = np.pad(x, NFFT // 2, mode=mode)
        w = np.pad(win, (NFFT - WIN) // 2)
        frames = x.size // HOP
        rows = [xp[i * HOP : i * HOP + NFFT] * w for i in range(frames)]
    if frames == 0:
        return np.zeros((0, NFFT // 2 + 1))
    return np.abs(np.fft.rfft(np.array(rows), n=NFFT, axis=-1)) ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="testdata/frontend/stft.safetensors")
    args = parser.parse_args()

    tensors, cases = {}, []
    for name, x in signals().items():
        tensors[f"{name}/input"] = x
        for wname, win in WINDOWS.items():
            for padding in PADDINGS:
                spec = power(x, win, padding)
                if spec.shape[0] == 0:
                    print(f"{name}/{wname}/{padding}: no frames, skipped")
                    continue
                key = f"{name}/{wname}/{padding}"
                tensors[key] = np.ascontiguousarray(spec, dtype=np.float32)
                cases.append({"key": key, "signal": name, "window": wname, "padding": padding, "frames": spec.shape[0]})
                print(f"{key:32s} {spec.shape[0]:3d} frames, peak {spec.max():.4g}")

    # One metadata key: safetensors writes __metadata__ in hash order, which
    # would make the file differ from run to run.
    fixture = {
        "generator": "scripts/mel_fixture.py",
        "numpy": np.__version__,
        "rate": RATE,
        "nfft": NFFT,
        "win": WIN,
        "hop": HOP,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata={"fixture": json.dumps(fixture, separators=(",", ":"))})
    print(f"wrote {args.out}: {len(tensors)} tensors, {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
