"""Generate the Needle 3 text model forward-pass fixtures for internal/text.

Imports upstream's SimpleAttentionNetwork (needle/model/architecture.py) and
render_example (needle/model/finetune.py), binds the network to the pinned
needle3.safetensors parameters (F16, widened to float32) and runs it in float32
(`highest` matmul precision, flash=False, no quantisation) on fixed tool-calling
prompts, encoded with sentencepiece on the pinned tokenizer.model as run.py
does ([BOS] + encode(prompt)). For each prompt it decodes greedily as run.py
`generate` does (temperature 0, stop at EOS) and then runs one forward pass over
prompt + completion, so every position's next-token logits are teacher-forced
on the reference's own greedy text. testdata/text/forward/README.md is the
contract (docs/FINDINGS.md F-045).

Written to testdata/text/forward/forward.safetensors (weights never stored):

    <case>.tokens        I64 (T)         [BOS] + prompt ids + greedy completion
    <case>.prompt_len    I64 (1)         positions of [BOS] + prompt
    <case>.argmax        I64 (T)         top-1 next token at every position
    <case>.margin        F32 (T)         top-1 minus top-2 logit at every position
    <case>.final         F16 (T, 768)    stack output after final_norm
    <case>.hidden_last   F32 (L, 768)    lane mean after each block, last position
    <case>.logits_last   F32 (2, 8192)   logits at the last 2 positions
    tiny.hidden          F32 (L, T, 768) lane mean after each block, all positions
                                         (the pre-norm stream exceeds the F16 range)
    tiny.embed           F32 (T, 768)    scaled input embedding
    tiny.logits          F32 (T, 8192)   logits at every position

Deterministic: no randomness, no timestamps, metadata in sorted key order.
Regenerate with

    just text-forward-fixtures

which runs

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/text_forward_fixtures.py
"""

import argparse
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
import types

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import traverse_util  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

UPSTREAM_PIN = "af2654dffa"
WEIGHTS_PIN = "93e88ede"
HUB_LOCK = "internal/hub/needle3.lock.json"
LAST = 2  # logits_last rows
MAX_NEW = 32  # greedy completion limit per prompt

SYSTEM = "date: 2026-10-11 Sun 09:30"

TIMER = {
    "name": "set_timer",
    "description": "Start a countdown timer",
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "description": "Duration in minutes"},
            "label": {"type": "string", "description": "What the timer is for"},
        },
        "required": ["minutes"],
    },
}
MESSAGE = {
    "name": "send_message",
    "description": "Send a text message to a contact",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Contact name"},
            "body": {"type": "string", "description": "Message text"},
        },
        "required": ["recipient", "body"],
    },
}
LIGHTS = {
    "name": "set_lights",
    "description": "Switch the lights in a room on or off and set the brightness",
    "parameters": {
        "type": "object",
        "properties": {
            "room": {"type": "string"},
            "on": {"type": "boolean"},
            "brightness": {"type": "integer", "description": "Percent, 0 to 100"},
        },
        "required": ["room", "on"],
    },
}

# name -> render_example input (answers are left out: the model writes them).
CASES = {
    "timer_en": {"system": SYSTEM, "tools": [TIMER], "query": "set a timer for 10 minutes for the pasta"},
    "message_de": {
        "system": SYSTEM,
        "tools": [TIMER, MESSAGE],
        "query": "Schreib Anna, dass ich heute zwanzig Minuten später komme",
    },
    "lights_two_calls": {
        "system": SYSTEM,
        "tools": [LIGHTS, TIMER],
        "query": "turn off the kitchen lights and dim the living room to 30 percent",
    },
}
TINY = "Hello, world"  # plain text, no chat format: [BOS] + encode(TINY)


def upstream_rev(root):
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned(local):
    with open(HUB_LOCK) as f:
        return next(e["sha256"] for e in json.load(f)["files"] if e["local"] == local)


def check_pins(upstream, safetensors, spm):
    problems = []
    rev = upstream_rev(upstream)
    if rev is None or not rev.startswith(UPSTREAM_PIN):
        problems.append(f"{upstream} is at {rev or 'no git revision'}, not {UPSTREAM_PIN}")
    for path, local in ((safetensors, "needle3.safetensors"), (spm, "tokenizer.model")):
        if file_digest(path) != pinned(local):
            problems.append(f"{path} is not the pinned {local} of {HUB_LOCK}")
    if problems:
        sys.exit("unpinned inputs:\n  " + "\n  ".join(problems))


def load_upstream(root):
    """Import needle.model.* without running needle/__init__.py, which loads
    the engine and telemetry."""
    pkg = types.ModuleType("needle")
    pkg.__path__ = [os.path.join(root, "needle")]
    sys.modules["needle"] = pkg
    arch = importlib.import_module("needle.model.architecture")
    finetune = importlib.import_module("needle.model.finetune")
    tokenizer = importlib.import_module("needle.model.tokenizer")
    return arch, finetune, tokenizer


def load_params(path):
    """The checkpoint as a nested Flax params dict, F16 widened to float32."""
    flat = {}
    with safe_open(path, "np") as f:
        meta = f.metadata()
        for key in f.keys():
            a = f.get_tensor(key)
            assert a.dtype == np.float16, (key, a.dtype)
            flat[tuple(key.split("/"))] = a.astype(np.float32)
    return traverse_util.unflatten_dict(flat), json.loads(meta["config"])


def write_sorted_metadata(path, tensors, metadata):
    """save_file, then rewrite the header with __metadata__ keys sorted, so
    two runs give the same bytes."""
    save_file(tensors, path, metadata=metadata)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        body = f.read()
    header["__metadata__"] = dict(sorted(header["__metadata__"].items()))
    raw = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + body)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", default="original/needle")
    ap.add_argument("--safetensors", default="models/needle3/needle3.safetensors")
    ap.add_argument("--tokenizer", default="models/needle3/tokenizer.model")
    ap.add_argument("--out", default="testdata/text/forward")
    args = ap.parse_args()

    check_pins(args.upstream, args.safetensors, args.tokenizer)
    jax.config.update("jax_default_matmul_precision", "highest")

    arch, finetune, tokmod = load_upstream(args.upstream)
    params, saved = load_params(args.safetensors)
    params = jax.tree.map(jnp.asarray, params)
    config = arch.TransformerConfig.from_saved(saved)
    config.dtype, config.flash = "float32", False
    model = arch.SimpleAttentionNetwork(config)
    tok = tokmod.SANTokenizer(args.tokenizer)
    bos, eos = tokmod.BOS_ID, tokmod.EOS_ID

    def stack_out(mdl, toks):
        """The stack output after final_norm, the per-block lane means and
        the scaled input embedding, as SimpleAttentionNetwork.__call__ and
        hidden_states compute them."""
        mask = arch.make_causal_mask(toks.shape[1])
        x = mdl._input_embeddings(toks)
        final, hidden = mdl.stack(x, mask=mask, rope=mdl._rope(toks.shape[1]),
                                  engram_kv=mdl._engram_kv(toks, mask, False), collect_hidden=True)
        return final, hidden, x

    @jax.jit
    def run(params, tokens):
        logits = model.apply({"params": params}, tokens)
        final, hidden, embed = model.apply({"params": params}, tokens, method=stack_out)
        return logits, final, hidden, embed

    @jax.jit
    def step_logits(params, tokens):
        return model.apply({"params": params}, tokens)[0]

    def forward(ids):
        logits, final, hidden, embed = run(params, jnp.asarray([ids], jnp.int32))
        return (np.asarray(logits[0], np.float32), np.asarray(final[0], np.float32),
                np.asarray(hidden[:, 0], np.float32), np.asarray(embed[0], np.float32))

    tensors, prompts = {}, {}
    for name, case in CASES.items():
        prompt, _ = finetune.render_example(case)
        ids = [bos] + tok.encode(prompt)
        prompt_len = len(ids)
        # run.py generate: a PAD buffer of prompt + MAX_NEW positions, greedy,
        # stop at EOS (not appended). The mask is causal, so the padding after
        # a position does not reach its logits.
        buf = np.full((1, prompt_len + MAX_NEW), tokmod.PAD_ID, np.int32)
        buf[0, :prompt_len] = ids
        for pos in range(prompt_len - 1, prompt_len + MAX_NEW - 1):
            nxt = int(np.argmax(np.asarray(step_logits(params, jnp.asarray(buf))[pos])))
            if nxt == eos:
                break
            ids.append(nxt)
            buf[0, pos + 1] = nxt
        logits, final, hidden, _ = forward(ids)
        top2 = np.sort(logits, axis=-1)[:, -2:]
        prompts[name] = {"prompt": prompt, "completion": tok.decode(ids[prompt_len:])}
        tensors[f"{name}.tokens"] = np.asarray(ids, np.int64)
        tensors[f"{name}.prompt_len"] = np.asarray([prompt_len], np.int64)
        tensors[f"{name}.argmax"] = np.argmax(logits, axis=-1).astype(np.int64)
        tensors[f"{name}.margin"] = (top2[:, 1] - top2[:, 0]).astype(np.float32)
        tensors[f"{name}.final"] = final.astype(np.float16)
        tensors[f"{name}.hidden_last"] = hidden[:, -1]
        tensors[f"{name}.logits_last"] = logits[-LAST:]
        print(f"{name}: {prompt_len} prompt + {len(ids) - prompt_len} completion tokens: "
              f"{prompts[name]['completion']!r}", file=sys.stderr)

    ids = [bos] + tok.encode(TINY)
    logits, _, hidden, embed = forward(ids)
    prompts["tiny"] = {"prompt": TINY, "completion": ""}
    tensors["tiny.tokens"] = np.asarray(ids, np.int64)
    tensors["tiny.hidden"] = hidden
    tensors["tiny.embed"] = embed
    tensors["tiny.logits"] = logits

    os.makedirs(args.out, exist_ok=True)
    metadata = {
        "upstream": UPSTREAM_PIN,
        "weights": WEIGHTS_PIN,
        "safetensors_sha256": pinned("needle3.safetensors"),
        "tokenizer_sha256": pinned("tokenizer.model"),
        "precision": "float32, jax_default_matmul_precision=highest, flash=False, quant=False",
        "prompts": json.dumps(prompts, ensure_ascii=False, sort_keys=True),
    }
    path = os.path.join(args.out, "forward.safetensors")
    # save_file writes a strided view's underlying buffer (hidden[:, -1]),
    # not its elements: make every tensor contiguous first.
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    write_sorted_metadata(path, tensors, metadata)
    print(f"wrote {path} ({os.path.getsize(path)} bytes, {len(tensors)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
