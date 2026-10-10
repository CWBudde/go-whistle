"""Generate the shared Needle 3 block fixtures for internal/needle.

Imports the upstream Flax modules from needle/model/architecture.py, binds
them to the real Whistle parameters from whistle.safetensors (Flax paths) and
runs each block on seeded random inputs in float32. Every block writes
testdata/needle/<block>.safetensors with its inputs, its outputs and the
intermediates a Go port needs for debugging; weights are never stored.
testdata/needle/README.md is the contract (docs/FINDINGS.md F-009).

The intermediates come from the module itself where Flax exposes them
(`capture_intermediates`: every submodule's output). The rest is recomputed
here from upstream's own helpers in upstream's expression order, and that
recomputation must reproduce the module's output before anything is written.

Deterministic: fixed seeds, numpy inputs, no timestamps, and the header's
__metadata__ is written in sorted key order (safetensors writes it in hash map
order otherwise). Regenerate with

    just needle-fixtures

which runs

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/needle_fixtures.py [--f64-check]

`--f64-check` writes nothing. It reruns every block in float64 and prints how
far the float32 fixtures are from that reference, i.e. the float32 noise a Go
port has to live with.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import struct
import subprocess
import sys
import types

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

import flax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import linen as nn  # noqa: E402
from flax import traverse_util  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

UPSTREAM_PIN = "af2654dffa"
WEIGHTS_PIN = "b358ddad"
SEED = 20261010

# Whistle decoder geometry (safetensors config, PLAN.md reference card).
D = 512
HEADS = 8  # q_proj is 512 -> 384 = 8 x qk_dim
KV_HEADS = 2
QK = 48
V = 64
LAYERS = 8
LANES = 4
ENGRAM_SUB_DIM = 128


HUB_LOCK = "internal/hub/manifest.lock.json"
CHECKPOINT = "checkpoints/whistle.safetensors"


def upstream_rev(root):
    """The checkout's HEAD commit, or None when root is not a git checkout."""
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def upstream_changes(root):
    """The checkout's local changes under needle/, the package load_upstream imports, as `git status` lists them
    (untracked files included); None when root is not a git checkout."""
    try:
        out = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "needle"], capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.splitlines()


def pinned_digest(lock=HUB_LOCK):
    """The SHA-256 the hub manifest pins for the safetensors checkpoint."""
    with open(lock) as f:
        files = json.load(f)["files"]
    return next(e["sha256"] for e in files if e["path"] == CHECKPOINT)


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(upstream, safetensors, allow_unpinned):
    """The upstream and weights pins to record: the pins themselves when the inputs are the pinned ones. A mismatch
    is an error, or with allow_unpinned is recorded as what was actually used."""
    rev, digest = upstream_rev(upstream), file_digest(safetensors)
    changes = upstream_changes(upstream) if rev else None
    problems = []
    if rev is None or not rev.startswith(UPSTREAM_PIN):
        problems.append(f"{upstream} is at {rev or 'no git revision'}, not {UPSTREAM_PIN}")
    if rev is not None and changes is None:
        problems.append(f"{upstream}: git status failed, so its worktree is unverified")
    if changes:
        problems.append(f"{upstream} has local changes under needle/: " + "; ".join(changes[:5]))
    if digest != pinned_digest():
        problems.append(f"{safetensors} has sha256 {digest}, not the {WEIGHTS_PIN} checkpoint of {HUB_LOCK}")
    if not problems:
        return UPSTREAM_PIN, WEIGHTS_PIN
    if not allow_unpinned:
        sys.exit("unpinned inputs (pass --allow-unpinned to record them as such):\n  " + "\n  ".join(problems))
    for p in problems:
        print("warning:", p, file=sys.stderr)
    if rev is None or changes is None:
        return "unverified", "sha256:" + digest
    return rev[:10] + ("-dirty" if changes else ""), "sha256:" + digest


def load_upstream(root):
    """Import needle.model.architecture without running needle/__init__.py,
    which loads the engine and telemetry."""
    pkg = types.ModuleType("needle")
    pkg.__path__ = [os.path.join(root, "needle")]
    sys.modules["needle"] = pkg
    return importlib.import_module("needle.model.architecture")


class F64Numpy(types.ModuleType):
    """jax.numpy with float32 mapped to float64, for --f64-check: upstream
    casts to jnp.float32 explicitly in ZCRMSNorm, _rms_unit, the RoPE tables
    and the mHC heads."""

    def __init__(self):
        super().__init__("jnp_f64")

    def __getattr__(self, name):
        return jnp.float64 if name == "float32" else getattr(jnp, name)


def cact_names(path):
    """Map 'source[layer]' (or 'source') to '<index> <name>' from the pinned
    golden of cact.WhistleNames() (docs/FINDINGS.md F-006)."""
    names = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            idx, name, _shape, _transform, source = line.split()
            if source != "-":
                names[source] = f"{int(idx)} {name}"
    return names


class Weights:
    """The safetensors checkpoint, read by Flax path, with a log of what each
    fixture used."""

    def __init__(self, path, names):
        self.f = safe_open(path, "np")
        self.names = names
        self.config = json.loads(self.f.metadata()["config"])
        self.used = []

    def get(self, path, layer=None):
        a = self.f.get_tensor(path)
        key = path if layer is None else f"{path}[{layer}]"
        if layer is not None:
            a = a[layer]
        if key not in self.used:
            self.used.append(key)
        assert a.dtype == np.float32, (key, a.dtype)
        return a

    def take_log(self):
        """'flax_path[layer] = .cact <index> <name>'. The mHC parameters are
        one .cact record for all layers, so their entry names the slice."""
        used, self.used = self.used, []
        out = []
        for k in used:
            if k in self.names:
                out.append(f"{k} = .cact {self.names[k]}")
            else:
                path, layer = k[:-1].split("[")
                out.append(f"{k} = .cact {self.names[path]}, layer {layer}")
        return out


class Ctx:
    def __init__(self, arch, weights, dt):
        self.arch, self.w, self.dt = arch, weights, dt

    def arr(self, a):
        """A numpy input or weight as a JAX array in the run's float type."""
        a = np.asarray(a)
        return jnp.asarray(a.astype(self.dt) if a.dtype.kind == "f" else a)

    def bind(self, module, args, kwargs, prefix, layer):
        """Fill the module's parameter tree from the checkpoint: the leaf at
        Flax path p comes from '<prefix>/<p>' (sliced at layer). Shapes must
        match exactly."""
        shapes = jax.eval_shape(module.init, jax.random.PRNGKey(0), *args, **kwargs)
        flat = {}
        for path, leaf in traverse_util.flatten_dict(shapes["params"]).items():
            name = "/".join((prefix, *path))
            a = self.w.get(name, layer)
            assert a.shape == leaf.shape, (name, a.shape, leaf.shape)
            flat[path] = self.arr(a)
        return {"params": traverse_util.unflatten_dict(flat)}


def run(module, params, *args, **kwargs):
    """Apply the module and return its output plus every submodule's output,
    keyed by its Flax path."""
    out, state = module.apply(params, *args, capture_intermediates=True,
                              mutable=["intermediates"], **kwargs)
    inter = {"/".join(k[:-1]): v[0]
             for k, v in traverse_util.flatten_dict(state["intermediates"]).items()}
    return out, inter


def same(name, got, want, tol=0.0):
    """The recomputed intermediate must reproduce upstream's value."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    bound = tol * max(1.0, float(np.max(np.abs(want))))
    assert err <= bound, f"{name}: recomputation differs by {err:g} (bound {bound:g})"
    return err


def attention(arch, dt, taps):
    return arch.MultiHeadAttention(HEADS, KV_HEADS, D, LAYERS, dt, flash=False, qk_head_dim=QK,
                                   v_head_dim=V, qkv_conv_taps=taps)


def block(arch, dt, taps):
    return arch.Block(HEADS, KV_HEADS, D, LAYERS, dt, False, qk_head_dim=QK, v_head_dim=V,
                      qkv_conv_taps=taps)


# --- fixtures -----------------------------------------------------------------


def fx_zcrmsnorm(c, rng):
    """ZCRMSNorm with layer 0's input norm scale, and _rms_unit. Rows span
    seven decades of scale plus an all-zero row; eps dominates the mean
    square in the smallest rows."""
    a = c.arch
    x = rng.standard_normal((2, 8, D)).astype(np.float32)
    row_scale = np.array([[1, 2, 0.5, 10, 100, 0.1, 1e-2, 1],
                          [1e-3, 1e-4, 1e-5, 0, 1, 1, 1, 1]], np.float32)
    x = x * row_scale[..., None]
    x[1, 5] += 3.0  # non-zero mean: RMS, not standard deviation
    xj = c.arr(x)
    m = a.ZCRMSNorm(dtype=c.dt)
    p = c.bind(m, (xj,), {}, "stack/layers/block/ZCRMSNorm_0", 0)
    y = m.apply(p, xj)
    rms = jnp.sqrt(jnp.mean(xj ** 2, axis=-1, keepdims=True) + 1e-6)
    same("y", (1 + p["params"]["scale"]) * xj / rms, y)
    return {"x": x, "rms": rms, "y": y, "rms_unit": a._rms_unit(xj)}, {"layer": "0", "eps": "1e-6"}


def attention_steps(c, m, p, x, mask, rope, taps):
    """Run MultiHeadAttention (flash=False) and recompute its steps."""
    a, P = c.arch, p["params"]
    B, T, _ = x.shape
    out, it = run(m, p, x, mask=mask, rope=rope)
    q, k, v = it["q_proj"], it["k_proj"], it["v_proj"]
    t = {"x": x, "q_proj": q, "k_proj": k, "v_proj": v}
    if taps:
        tap_ok = [None] + [(a._mask_diag(mask, j)[..., None].astype(q.dtype) if mask is not None else None)
                           for j in range(1, taps)]

        def tapped(z, j):
            s = a._shift_right(z, j)
            return s if tap_ok[j] is None else s * tap_ok[j]

        q = sum(P["q_taps"][j] * tapped(q, j) for j in range(taps))
        k = sum(P["k_taps"][j] * tapped(k, j) for j in range(taps))
        v = sum(P["v_taps"][j] * tapped(v, j) for j in range(taps))
        t.update(q_tapped=q, k_tapped=k, v_tapped=v)
    qh = q.reshape(B, T, HEADS, QK).transpose(0, 2, 1, 3)
    kh = k.reshape(B, T, KV_HEADS, QK).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, KV_HEADS, V).transpose(0, 2, 1, 3)
    norm = a.ZCRMSNorm(dtype=c.dt)
    same("q_norm", norm.apply({"params": P["q_norm"]}, qh), it["q_norm"])
    same("k_norm", norm.apply({"params": P["k_norm"]}, kh), it["k_norm"])
    qh, kh = it["q_norm"], it["k_norm"]
    t.update(q_normed=qh, k_normed=kh)
    if rope is not None:
        qh, kh = a.apply_rope(qh, *rope), a.apply_rope(kh, *rope)
        t.update(q_rope=qh, k_rope=kh)
    kr, vr = jnp.repeat(kh, HEADS // KV_HEADS, axis=1), jnp.repeat(vh, HEADS // KV_HEADS, axis=1)
    w = jnp.matmul(qh, kr.transpose(0, 1, 3, 2)) / jnp.sqrt(c.dt(QK))
    if mask is not None:
        w = jnp.where(mask, w, jnp.finfo(w.dtype).min)
    probs = nn.softmax(w, axis=-1)
    attn = jnp.matmul(probs, vr).transpose(0, 2, 1, 3).reshape(B, T, HEADS * V)
    gate = nn.sigmoid(it["gate_proj"])
    gated = attn * gate
    same("out", gated @ P["out_proj"]["kernel"], out)
    t.update(probs=probs, attn=attn, gate=gate, gated=gated, out=out)
    return t


def fx_attention_decoder(c, rng):
    """Decoder self-attention, layer 0: causal mask, 3-tap Q/K/V conv, q/k
    norm, RoPE with the config's theta, GQA 8:2 and the sigmoid gate."""
    a, cfg = c.arch, c.w.config
    B, T = 2, 8
    x = c.arr(rng.standard_normal((B, T, D)).astype(np.float32))
    mask = a.make_causal_mask(T)
    theta = float(cfg["rope"])
    rope = a.precompute_rope_freqs(QK, T, theta)
    m = attention(a, c.dt, cfg["taps"])
    p = c.bind(m, (x,), {"mask": mask, "rope": rope}, "stack/layers/block/self_attn", 0)
    t = attention_steps(c, m, p, x, mask, rope, cfg["taps"])
    t.update(rope_cos=rope[0], rope_sin=rope[1])
    flash = m.clone(flash=True)
    err = same("flash", flash.apply(p, x, mask=mask, rope=rope), t["out"], tol=1e-5)
    print(f"  attention_decoder: flash=True differs from flash=False by {err:.3g} max abs")
    return t, {"layer": "0", "mask": "causal", "rope_theta": repr(theta), "taps": str(cfg["taps"]),
               "flash": "False"}


def fx_attention_encoder(c, rng):
    """Encoder-style attention with encoder layer 0's weights: no mask, no
    taps. Whether the encoder uses RoPE is open (PLAN.md U3), so `out` has
    none and `out_rope` uses the decoder's tables."""
    a, cfg = c.arch, c.w.config
    B, T = 2, 8
    x = c.arr(rng.standard_normal((B, T, D)).astype(np.float32))
    m = attention(a, c.dt, 0)
    p = c.bind(m, (x,), {}, "encoder/layers/block/self_attn", 0)
    t = attention_steps(c, m, p, x, None, None, 0)
    t["out_rope"] = m.apply(p, x, rope=a.precompute_rope_freqs(QK, T, float(cfg["rope"])))
    return t, {"layer": "0", "mask": "none", "taps": "0", "flash": "False",
               "rope_theta": repr(float(cfg["rope"])) + " (out_rope only)"}


def fx_hadamard_mlp(c, rng):
    """HadamardMLP, decoder layer 0: rank-8 condition, three Monarch 16x32
    Kronecker factors, SiLU and the permutations from _hada_perms(512)."""
    a = c.arch
    x = c.arr(rng.standard_normal((2, 8, D)).astype(np.float32))
    m = a.HadamardMLP(D, c.dt)
    p = c.bind(m, (x,), {}, "stack/layers/block/hadamard_mlp", 0)
    out = m.apply(p, x)
    P = p["params"]
    assert a._hada_blocks(D) == (16, 32)
    p1, p2 = a._hada_perms(D, False)
    f = [(P[f"w{s}a"], P[f"w{s}b"]) for s in (1, 2, 3)]
    cond_logits = x @ P["cond_v"]
    cond = 1 + nn.softmax(cond_logits, axis=-1) @ P["cond_u"]
    z1 = a._kron_apply(P["d1"] * x, *f[0])[..., p1]
    a2 = P["d2"] * cond * z1 + P["b2"]
    z2 = a._kron_apply(nn.silu(a2), *f[1])[..., p2]
    z3 = a._kron_apply(P["d3"] * z2, *f[2])
    same("out", P["d4"] * z3, out)
    return ({"x": x, "cond_logits": cond_logits, "cond": cond, "z1": z1, "a2": a2, "z2": z2, "z3": z3,
             "out": out, "p1": np.asarray(p1, np.int64), "p2": np.asarray(p2, np.int64)},
            {"layer": "0", "cond_rank": str(a.HADA_COND_RANK), "blocks": "16x32", "split_perms": "False"})


def mhc_step(c, body, p, hc, layer, stream, mask, rope, tag):
    """One _ScanBody step, plus its advance() recomputed for the heads."""
    a = c.arch
    B, T, n, C = stream.shape
    new, _ = body.apply(p, stream, (None, c.arr(0.0), hc, jnp.int32(layer)), mask, None, rope, False,
                        None, jnp.int32(0))
    nx = a._rms_unit(stream.reshape(B, T, n * C))
    pre_logits = hc["a_pre"] * (nx @ hc["phi_pre"]) + hc["b_pre"] + hc["pre_off"]
    hpre = nn.sigmoid(pre_logits)
    u = jnp.einsum("btn,btnc->btc", hpre, stream)
    block_out = block(a, c.dt, c.w.config["taps"]).apply({"params": p["params"]["block"]}, u, mask=mask,
                                                         rope=rope)
    y = block_out - u
    post_logits = hc["a_post"] * (nx @ hc["phi_post"]) + hc["b_post"] + hc["post_off"]
    hpost = 2 * nn.sigmoid(post_logits)
    res_logits = hc["a_res"] * (nx @ hc["phi_res"]).reshape(B, T, n, n) + hc["b_res"]
    hres = a._sinkhorn(res_logits)
    out = jnp.einsum("btij,btjc->btic", hres, stream) + hpost[..., None] * y[:, :, None, :]
    same(f"{tag} stream_out", out, new)
    return {f"{tag}.stream_in": stream, f"{tag}.nx": nx, f"{tag}.h_pre_logits": pre_logits,
            f"{tag}.h_pre": hpre, f"{tag}.u": u, f"{tag}.block_out": block_out, f"{tag}.y": y,
            f"{tag}.h_post_logits": post_logits, f"{tag}.h_post": hpost, f"{tag}.h_res_logits": res_logits,
            f"{tag}.h_res": hres, f"{tag}.stream_out": new}


def fx_mhc(c, rng):
    """mHC around one decoder layer, as _ScanBody does it. Case layer0 starts
    from the Stack's lane expand of x and ends with the lane collapse (mean)
    and final_norm. Case layer1 feeds four different lanes into layer 1, so
    the Sinkhorn mix and the lane-1 offsets matter."""
    a, cfg = c.arch, c.w.config
    B, T = 1, 8
    x = c.arr(rng.standard_normal((B, T, D)).astype(np.float32))
    lanes = (rng.standard_normal((B, T, LANES, D)) * np.array([1.0, 0.5, 2.0, 0.25])[:, None]).astype(np.float32)
    mask = a.make_causal_mask(T)
    rope = a.precompute_rope_freqs(QK, T, float(cfg["rope"]))
    body = a._ScanBody(HEADS, KV_HEADS, D, LAYERS, c.dt, False, qk_head_dim=QK, v_head_dim=V,
                       qkv_conv_taps=cfg["taps"])
    lane = np.eye(LANES, dtype=np.float32)[np.arange(LAYERS) % LANES]
    t = {"x": x}
    # Stack: x[:, :, None, :] broadcast to n lanes.
    for layer, stream, tag in ((0, jnp.broadcast_to(x[:, :, None, :], (B, T, LANES, D)), "layer0"),
                               (1, c.arr(lanes), "layer1")):
        hc = {k: c.arr(c.w.get(f"stack/mhc_{k}", layer)) for k in
              ("phi_pre", "phi_post", "phi_res", "b_pre", "b_post", "b_res", "a_pre", "a_post", "a_res")}
        hc["pre_off"] = c.arr(8 * lane[layer] - 4)
        hc["post_off"] = c.arr(-4 * (1 - lane[layer]))
        args = (stream, (None, c.arr(0.0), hc, jnp.int32(layer)), mask, None, rope, False, None, jnp.int32(0))
        p = c.bind(body, args, {}, "stack/layers", layer)
        t.update(mhc_step(c, body, p, hc, layer, stream, mask, rope, tag))
    del t["layer0.stream_in"]  # the lane expand of x
    collapsed = jnp.mean(t["layer0.stream_out"], axis=2)
    norm = a.ZCRMSNorm(dtype=c.dt)
    t["layer0.collapsed"] = collapsed
    t["layer0.final"] = norm.apply(c.bind(norm, (collapsed,), {}, "stack/final_norm", None), collapsed)
    return t, {"layer": "0,1", "lanes": str(LANES), "sinkhorn_iters": "20", "mask": "causal",
               "rope_theta": repr(float(cfg["rope"])), "engram": "none (site_flags are 0 at layers 0 and 1)"}


def fx_engram(c, rng):
    """Engram site 0 (decoder layer 3): hash indices from token ids, table
    lookup, key/value projections, the dilated 4-tap conv on v, and the gate
    Block applies at layer 3."""
    a, cfg = c.arch, c.w.config
    B, T = 2, 12
    tokens = rng.integers(0, cfg["text"] + 7, (B, T)).astype(np.int64)
    tokens[0, 6:9] = tokens[0, 1:4]  # repeated trigram: order-2 indices of 7, 8 = 2, 3; order-3 of 8 = 3
    tokens[1, 0] = 0  # pad id
    orders, slots, heads = tuple(cfg["orders"]), cfg["slots"], cfg["hashes"]
    geometry = a.engram_geometry(a.TransformerConfig(d_model=D, engram_orders=orders, engram_heads=heads))
    assert geometry == (orders, heads, ENGRAM_SUB_DIM), geometry
    dilation = max(orders)
    mask = a.make_causal_mask(T)
    indices = a.engram_indices(jnp.asarray(tokens, jnp.int32), orders, heads, slots, 0)
    ngram_ok = jnp.stack([a._mask_diag(mask, o - 1) for o in orders for _ in range(heads)], axis=-1).astype(c.dt)
    tap_ok = jnp.stack([a._mask_diag(mask, j * dilation) for j in range(a.ENGRAM_CONV_TAPS)]).astype(c.dt)
    m = a.Engram(D, len(orders) * heads, slots, ENGRAM_SUB_DIM, LAYERS, dilation, c.dt)
    p = c.bind(m, (indices, ngram_ok, tap_ok), {}, "engrams_0", None)
    (k, v), it = run(m, p, indices, ngram_ok, tap_ok)
    P = p["params"]
    fetched = P["embedding"][jnp.arange(len(orders) * heads), indices] * ngram_ok[..., None]
    e = fetched.reshape(B, T, len(orders) * heads * ENGRAM_SUB_DIM)
    same("key_proj", e @ P["key_proj"]["kernel"], k)
    v_raw = it["value_proj"]
    same("v", sum(P["taps"][j] * a._shift_right(v_raw, j * dilation) * tap_ok[j][..., None]
                  for j in range(a.ENGRAM_CONV_TAPS)), v)

    # The gate, as Block runs it at layer 3 with the Stack's site flags [1, 0]:
    # site 1 (engrams_1, layer 7) is multiplied by 0 there, so only site 0 is kept.
    x = c.arr(rng.standard_normal((B, T, D)).astype(np.float32))
    ek, ev = k[None], v[None]
    flags = c.arr(np.ones(1, np.float32))
    alpha = nn.sigmoid(jnp.einsum("btd,sbtd->sbt", a._rms_unit(x), a._rms_unit(ek)) / math.sqrt(D))
    x_out = x + jnp.einsum("s,sbt,sbtd->btd", flags, alpha, ev).astype(x.dtype)
    blk = block(a, c.dt, cfg["taps"])
    rope = a.precompute_rope_freqs(QK, T, float(cfg["rope"]))
    log = list(c.w.used)
    bp = c.bind(blk, (x,), {"mask": mask, "rope": rope}, "stack/layers/block", 3)
    c.w.used = log  # the check's block weights are not part of the fixture
    with_engram = blk.apply(bp, x, mask=mask, rope=rope, engram_kv=(ek, ev), site_flags=flags)
    same("block(x_out)", blk.apply(bp, x_out, mask=mask, rope=rope), with_engram)
    return ({"tokens": tokens, "indices": np.asarray(indices, np.int64), "ngram_ok": ngram_ok[0],
             "tap_ok": tap_ok[:, 0], "e": e, "k": k, "v_raw": v_raw, "v": v, "x": x, "alpha": alpha[0],
             "x_out": x_out},
            {"layer": "3", "site": "0", "orders": json.dumps(list(orders)), "hashes": str(heads),
             "slots": str(slots), "dilation": str(dilation), "mask": "causal"})


def fx_block(c, rng):
    """One decoder Block at layer 0, without cross-attention (not in
    upstream) and without engram (layer 0 is no site)."""
    a, cfg = c.arch, c.w.config
    B, T = 2, 8
    x = c.arr(rng.standard_normal((B, T, D)).astype(np.float32))
    mask = a.make_causal_mask(T)
    rope = a.precompute_rope_freqs(QK, T, float(cfg["rope"]))
    m = block(a, c.dt, cfg["taps"])
    p = c.bind(m, (x,), {"mask": mask, "rope": rope}, "stack/layers/block", 0)
    out, it = run(m, p, x, mask=mask, rope=rope)
    h = x + nn.sigmoid(p["params"]["attn_gate"]).astype(c.dt) * it["post_attn_norm"]
    same("out", h + it["hadamard_mlp"], out)
    return ({"x": x, "norm_in": it["ZCRMSNorm_0"], "attn": it["self_attn"], "post_attn": it["post_attn_norm"],
             "h": h, "pre_hada": it["pre_hada_norm"], "mlp": it["hadamard_mlp"], "out": out},
            {"layer": "0", "mask": "causal", "rope_theta": repr(float(cfg["rope"])), "taps": str(cfg["taps"]),
             "flash": "False", "cross_attention": "none", "engram": "none"})


# name, builder; fixture i draws its inputs from np.random.default_rng(SEED + i).
FIXTURES = [
    ("zcrmsnorm", fx_zcrmsnorm),
    ("attention_decoder", fx_attention_decoder),
    ("attention_encoder", fx_attention_encoder),
    ("hadamard_mlp", fx_hadamard_mlp),
    ("mhc", fx_mhc),
    ("engram", fx_engram),
    ("block", fx_block),
]


# --- output -------------------------------------------------------------------


def to_numpy(tensors):
    out = {}
    for name, v in tensors.items():
        v = np.asarray(v)
        if v.dtype.kind in "iu":
            out[name] = np.ascontiguousarray(v.astype(np.int64))
        else:
            assert v.dtype == np.float32, (name, v.dtype)
            assert np.all(np.isfinite(v)), name
            out[name] = np.ascontiguousarray(v)
    return out


def canonical_header(path):
    """Rewrite the header with __metadata__ keys sorted. safetensors (0.8.0)
    serialises the metadata from a hash map, so its key order changes from
    run to run; the tensor entries are already sorted. The JSON keeps its
    length, so the data offsets stay valid."""
    with open(path, "r+b") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
        text = raw.decode("utf-8")
        stripped = text.rstrip(" ")
        header = json.loads(stripped)
        meta = header.pop("__metadata__")
        canon = json.dumps({"__metadata__": dict(sorted(meta.items())), **header},
                           separators=(",", ":"), ensure_ascii=False)
        assert json.loads(canon) == json.loads(stripped) and len(canon) == len(stripped), path
        f.seek(8)
        f.write(canon.encode("utf-8"))


def write(path, tensors, meta):
    save_file(tensors, path, metadata=meta)
    canonical_header(path)
    with safe_open(path, "np") as f:
        assert f.metadata() == meta
        assert sorted(f.keys()) == sorted(tensors)
        for name, v in tensors.items():
            assert np.array_equal(f.get_tensor(name), v), name


def f64_check(ctx, out_dir):
    print("float64 rerun vs the float32 fixtures: max abs diff, max |f64|, cosine")
    for i, (name_, fn) in enumerate(FIXTURES):
        tensors, _ = fn(ctx, np.random.default_rng(SEED + i))
        ctx.w.take_log()
        with safe_open(os.path.join(out_dir, f"{name_}.safetensors"), "np") as f:
            for name in sorted(tensors):
                ref = np.asarray(tensors[name])
                got = f.get_tensor(name)
                if ref.dtype.kind in "iu":
                    assert np.array_equal(ref, got), (name_, name)
                    continue
                ref, got = ref.astype(np.float64), got.astype(np.float64)
                err = float(np.max(np.abs(got - ref)))
                den = np.linalg.norm(got) * np.linalg.norm(ref)
                cos = float(np.dot(got.ravel(), ref.ravel()) / den) if den else 1.0
                print(f"  {name_:18s} {name:22s} {err:9.3g} {float(np.max(np.abs(ref))):9.3g} {cos:.12f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safetensors", default="models/whistle/whistle.safetensors")
    ap.add_argument("--upstream", default="original/needle", help="cactus-compute/needle checkout at " + UPSTREAM_PIN)
    ap.add_argument("--names", default="internal/cact/testdata/whistle_names.txt",
                    help="golden of cact.WhistleNames(), for the .cact names in the metadata")
    ap.add_argument("--out", default="testdata/needle")
    ap.add_argument("--f64-check", action="store_true", help="compare against a float64 rerun; writes nothing")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="accept an upstream checkout or checkpoint other than the pinned ones and record what was used")
    args = ap.parse_args()

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", args.f64_check)
    jax.config.update("jax_default_matmul_precision", "highest")
    arch = load_upstream(args.upstream)
    dt = jnp.float32
    if args.f64_check:
        arch.jnp = F64Numpy()
        dt = jnp.float64
    weights = Weights(args.safetensors, cact_names(args.names))
    cfg = weights.config
    geometry = (cfg["dim"], cfg["kv_heads"], cfg["qk_dim"], cfg["v_dim"], cfg["dec_layers"], cfg["lanes"])
    assert geometry == (D, KV_HEADS, QK, V, LAYERS, LANES), geometry
    ctx = Ctx(arch, weights, dt)
    if args.f64_check:
        f64_check(ctx, args.out)
        return

    upstream_pin, weights_pin = provenance(args.upstream, args.safetensors, args.allow_unpinned)
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for i, (name, fn) in enumerate(FIXTURES):
        tensors, extra = fn(ctx, np.random.default_rng(SEED + i))
        tensors = to_numpy(tensors)
        meta = {
            "block": name,
            "seed": str(SEED + i),
            "upstream": upstream_pin,
            "weights_pin": weights_pin,
            "weights": json.dumps(weights.take_log()),
            "shapes": json.dumps({k: list(v.shape) for k, v in sorted(tensors.items())}),
            "dtype": "float32",
            "matmul_precision": "highest",
            "versions": f"jax {jax.__version__}, flax {flax.__version__}, numpy {np.__version__}",
            **extra,
        }
        path = os.path.join(args.out, f"{name}.safetensors")
        write(path, tensors, meta)
        size = os.path.getsize(path)
        total += size
        print(f"wrote {path}: {len(tensors)} tensors, {size} bytes")
    print(f"total {total} bytes")


if __name__ == "__main__":
    main()
