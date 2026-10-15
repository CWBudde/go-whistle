"""Record the six needle/environments suites with the engine (PLAN.md 11.7).

Upstream ships six pre-configured environments (needle/environments at
af2654dffa: data_capture, kitchen_appliance, media_player, productivity,
smart_home, wearable), each with TOOLS, SYSTEM and 32 frozen TEST_CASES, and
a harness (_harness.run_tests) that scores the engine on them. This script
runs that harness through upstream's own Python client, which reaches the
engine (libneedle3) through ctypes, and records every answer and both of the
harness's verdicts, so that needle/environments_test.go replays the corpus
through the Go API and the Go port of the harness without Python.

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 python3 -I scripts/environments_fixture.py \
        [--out testdata/text/engine/environments.json]

Standard library only; the client in original/needle needs nothing else.
Telemetry is off (both variables are set here too, before needle is
imported, and its track function is replaced by a no-op). The script

  - puts original/needle on sys.path and points the client at this machine's
    engine/<platform>/libneedle3.* (the environment variable the client reads,
    needle.agent.fetch.lib_path_env(3)) and at models/needle3/needle3.cact
    (needle._base_weights_path patched), whose sha256 must be the pin of
    internal/hub/needle3.lock.json;
  - pins the clock: the agent's system is "date: 2026-10-11 Sun 09:30; " +
    SYSTEM, exactly what the client's _with_date_fact builds with that clock
    (it leaves a system that holds "date:" alone), the date of
    testdata/text/engine/cases.json;
  - per suite, puts that agent into _harness._agents, so that upstream's own
    run_tests uses it, wraps its complete to record every response, and runs
    run_tests at the gates 0.0 (the raw output) and 0.4 (the production
    contract), keeping the line it prints and the bool it returns. The
    second run answers every case again; the script fails if an answer
    differs from the first's.

The fixture holds per suite the module's SYSTEM, the system the client sent
(system_text), the tools as the client sent them, the cases with the
client's response (annotated by _annotate_ungrounded, as run_tests reads it)
without the engine's top-level timing keys (*_tps, *_ms, *_mb: the oracle's
isTimingKey), and both run_tests outcomes. No timestamps: two recordings
must diff clean (engine determinism). Format it with treefmt afterwards.
"""

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import re
import subprocess
import sys

os.environ["NEEDLE_TELEMETRY"] = "0"
os.environ["DO_NOT_TRACK"] = "1"
sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(ROOT, "original", "needle")
UPSTREAM_PIN = "af2654dffa"
DATE_FACT = "date: 2026-10-11 Sun 09:30"
WEIGHTS = os.path.join(ROOT, "models", "needle3", "needle3.cact")
WEIGHTS_LOCK = os.path.join(ROOT, "internal", "hub", "needle3.lock.json")
FETCH_ENGINE = os.path.join(ROOT, "scripts", "fetch-engine.sh")
SUITES = ("data_capture", "kitchen_appliance", "media_player", "productivity", "smart_home", "wearable")
GATES = (0.0, 0.4)
TIMING_SUFFIXES = ("_tps", "_ms", "_mb")
OUTCOME = re.compile(r"^(\d+)/(\d+) passed, (\d+) critical failures \(confidence gate ([0-9.]+)\)$")


def engine_dir():
    """engine/<platform> for this machine, as scripts/fetch-engine.sh lays it out."""
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        return os.path.join(ROOT, "engine", "macos-arm64"), "libneedle3.dylib"
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        return os.path.join(ROOT, "engine", "linux-x86_64"), "libneedle3.so"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return os.path.join(ROOT, "engine", "linux-arm64"), "libneedle3.so"
    sys.exit(f"no engine for {system}/{machine}; run scripts/fetch-engine.sh")


def check_engine(edir, pin):
    """Exits unless engine/<platform>/PIN names the engine scripts/fetch-engine.sh pins (its REVISION and
    ENGINE_VERSION) and this platform, so a fixture is never recorded with another engine release."""
    with open(FETCH_ENGINE, encoding="utf-8") as f:
        script = f.read()
    want = {}
    for key, var in (("revision", "REVISION"), ("engine_version", "ENGINE_VERSION")):
        m = re.search(rf'^{var}="([^"]+)"$', script, re.MULTILINE)
        if m is None:
            sys.exit(f"{FETCH_ENGINE} pins no {var}")
        want[key] = m.group(1)
    want["platform"] = os.path.basename(edir)
    problems = [f"{key}={pin.get(key)!r}, want {value!r}" for key, value in want.items() if pin.get(key) != value]
    if problems:
        sys.exit(f"{edir}/PIN is not the pinned engine (run scripts/fetch-engine.sh):\n  " + "\n  ".join(problems))


def read_pin(path):
    """The key=value lines of engine/<platform>/PIN."""
    pin = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep:
                pin[key] = value
    return pin


def check_upstream(root):
    """Exits unless root is a git checkout at UPSTREAM_PIN without local changes under needle/, the package this
    script imports and runs (as scripts/needle_fixtures.py checks its inputs)."""
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        changes = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "needle"], capture_output=True,
                                 text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        sys.exit(f"{root} is not a git checkout of cactus-compute/needle at {UPSTREAM_PIN}")
    problems = []
    if not rev.startswith(UPSTREAM_PIN):
        problems.append(f"{root} is at {rev}, not {UPSTREAM_PIN}")
    if changes:
        problems.append(f"{root} has local changes under needle/: " + ", ".join(changes[:5]))
    if problems:
        sys.exit("unpinned upstream:\n  " + "\n  ".join(problems))


def weights_sha256():
    """The sha256 of models/needle3/needle3.cact; exits unless it is the pinned one."""
    with open(WEIGHTS_LOCK, encoding="utf-8") as f:
        lock = json.load(f)
    want = next(e["sha256"] for e in lock["files"] if e["local"] == "needle3.cact")
    h = hashlib.sha256()
    with open(WEIGHTS, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != want:
        sys.exit(f"{WEIGHTS}: sha256 {got}, the pin is {want}; run `just fetch-needle3`")
    return got


def without_timings(response):
    """The response without the engine's top-level timing keys (the oracle's isTimingKey)."""
    return {k: v for k, v in response.items() if not k.endswith(TIMING_SUFFIXES)}


def run_gate(harness, module, gate):
    """Upstream's run_tests at one gate: the line it prints, parsed, and the bool it returns."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        passed = harness.run_tests(module, min_confidence=gate, verbose=False)
    line = out.getvalue().strip()
    m = OUTCOME.match(line)
    if not m:
        sys.exit(f"{module.__name__}: unexpected run_tests output {line!r}")
    return {
        "min_confidence": gate,
        "line": line,
        "passed": int(m.group(1)),
        "total": int(m.group(2)),
        "critical_failures": int(m.group(3)),
        "pass": bool(passed),
    }


def record_suite(needle, harness, module):
    system_text = DATE_FACT + "; " + module.SYSTEM
    agent = needle.Needle(tools=module.TOOLS, system=system_text)
    if agent._system_text != system_text:
        sys.exit(f"{module.__name__}: the client sent {agent._system_text!r}")
    harness._agents[module.__name__] = agent

    recorded = []
    complete = agent.complete

    def recording(text="", max_new_tokens=512):
        response = complete(text, max_new_tokens)
        recorded.append((text, json.loads(json.dumps(without_timings(response)))))
        return response

    agent.complete = recording

    outcomes = []
    answers = None
    queries = [c["query"] for c in module.TEST_CASES]
    for gate in GATES:
        recorded.clear()
        outcomes.append(run_gate(harness, module, gate))
        if [q for q, _ in recorded] != queries:
            sys.exit(f"{module.__name__}: run_tests asked {len(recorded)} queries, not the suite's")
        if answers is None:
            answers = [r for _, r in recorded]
        elif answers != [r for _, r in recorded]:
            sys.exit(f"{module.__name__}: the engine answered differently at gate {gate}")

    cases = []
    for c, response in zip(module.TEST_CASES, answers):
        cases.append({
            "query": c["query"],
            "calls": c["calls"],
            "category": c["category"],
            "critical": bool(c.get("critical", False)),
            "response": response,
        })

    return {
        "name": module.__name__.rsplit(".", 1)[-1],
        "system": module.SYSTEM,
        "system_text": agent._system_text,
        "tools": json.loads(agent._tools_json),
        "cases": cases,
        "run_tests": outcomes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "testdata", "text", "engine", "environments.json"))
    args = ap.parse_args()

    edir, lib = engine_dir()
    pin = read_pin(os.path.join(edir, "PIN"))
    check_engine(edir, pin)
    sha = weights_sha256()
    check_upstream(UPSTREAM)

    sys.path.insert(0, UPSTREAM)
    from needle.agent import fetch

    os.environ[fetch.lib_path_env(3)] = os.path.join(edir, lib)

    import needle
    from needle.environments import _harness

    needle._track = lambda *a, **k: None
    needle._base_weights_path = lambda generation: WEIGHTS

    suites = []
    for name in SUITES:
        module = importlib.import_module(f"needle.environments.{name}")
        suite = record_suite(needle, _harness, module)
        for o in suite["run_tests"]:
            print(f"{name:18s} {o['line']}  -> {o['pass']}", file=sys.stderr)
        suites.append(suite)

    fixture = {
        "engine_version": pin["engine_version"],
        "engine_revision": pin["revision"],
        "model_sha256": sha,
        "upstream": UPSTREAM_PIN,
        "date_fact": DATE_FACT,
        "suites": suites,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
