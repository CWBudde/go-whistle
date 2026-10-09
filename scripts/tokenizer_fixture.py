"""Generate the sentencepiece fixture for internal/tokenizer.

Reads the RAW tokenizer blob from whistle.cact, parses it with upstream
`parse_tokenizer_blob`, and rebuilds a sentencepiece ModelProto from its
pieces, scores and types. sentencepiece then decodes a fixed set of id
sequences, and upstream `RefTokenizer` encodes the round-trip samples. The Go
tests compare against both (docs/FINDINGS.md F-005).

Deterministic: the sequences come from a seeded PRNG and the output carries no
timestamps. Regenerate with

    just tokenizer-fixture

which runs

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/tokenizer_fixture.py [--needle3 tokenizer.model]

`--needle3` takes Cactus-Compute/needle3 tokenizer/tokenizer.model and checks
the rebuilt model against it (pieces, scores, types, decode and encode over
the shared ids). The check only prints; the fixture does not depend on it.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import struct
import sys
import types

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

import sentencepiece as spm  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as pb  # noqa: E402

UPSTREAM_PIN = "af2654dffa"
SEED = 20261009

# Round-trip samples: Latin, accented European, Cyrillic, CJK, Arabic, emoji
# and other byte-fallback text, markers and whitespace. None contains "▁",
# which encode cannot tell apart from an escaped space.
SAMPLES = [
    ("english", "The quick brown fox jumps over the lazy dog."),
    ("german", "Grüße aus Köln! Die Straße ist schön, oder?"),
    ("french", "Où est la bibliothèque ? Ça coûte 12,50 €."),
    ("spanish", "¿Dónde está el niño? ¡Mañana será otro día!"),
    ("polish", "Zażółć gęślą jaźń, powiedział Krzysztof."),
    ("dutch", "Het is een mooie dag in Utrecht, zei ze."),
    ("italian", "Perché è così difficile? Più tardi, forse."),
    ("russian", "Привет, мир! Как дела?"),
    ("ukrainian", "Їжак ґречно з'їв яблуко."),
    ("chinese", "你好，世界。今天天气很好。"),
    ("japanese", "こんにちは、世界。カタカナと漢字。"),
    ("korean", "안녕하세요 세계"),
    ("arabic", "مرحبا بالعالم، كيف حالك؟"),
    ("hebrew", "שלום עולם"),
    ("hindi", "नमस्ते दुनिया"),
    ("greek", "Γειά σου Κόσμε"),
    ("emoji", "🙂 ok 👍🏽 🇩🇪 👨‍👩‍👧"),
    ("symbols", "x² + y² = z²; π ≈ 3.14159; 100% ✓"),
    ("whitespace", "  leading, double  space, trailing "),
    ("control chars", "tab\there\nnew line\r\nend"),
    ("markers", "<|im_start|>user\nHallo<|im_end|><speech></speech>"),
    ("language tokens", "<|de|>Guten Tag<|en|>hello"),
    ("specials spelled out", "<s> </s> <pad> <unk> <0x41>"),
    ("digits", "Call 0800 123 4567 at 10:30, ref #A-42/b."),
    ("mixed", "Siobhan said «Привет» and 你好 to Zoë 🙂"),
]


def load_upstream(root):
    """Import needle.model.export without running needle/__init__.py, which
    loads the engine and telemetry."""
    pkg = types.ModuleType("needle")
    pkg.__path__ = [os.path.join(root, "needle")]
    sys.modules["needle"] = pkg
    return importlib.import_module("needle.model.export")


def raw_blob(export, path):
    """Return the file and its first RAW blob, walking the .cact directory
    with upstream's header and record formats."""
    with open(path, "rb") as f:
        data = f.read()
    hdr = struct.unpack_from(export._HDR_FMT, data, 0)
    num_tensors, cb_n = hdr[1], hdr[2]
    off = struct.calcsize(export._HDR_FMT) + 4 * cb_n
    for _ in range(num_tensors):
        rec = struct.unpack_from(export._REC_FMT, data, off)
        off += export.REC_SIZE
        if rec[0] == export.RAW:
            return data, data[rec[7]:rec[7] + rec[8]]
    raise SystemExit(f"{path}: no RAW tensor")


def build_model(export, meta):
    """Rebuild a sentencepiece BPE ModelProto from the parsed blob. The
    normalizer and trainer settings copy needle3's tokenizer.model; only
    unk_surface differs (F-005)."""
    kind = {
        export.TK_NORMAL: pb.ModelProto.SentencePiece.NORMAL,
        export.TK_UNKNOWN: pb.ModelProto.SentencePiece.UNKNOWN,
        export.TK_CONTROL: pb.ModelProto.SentencePiece.CONTROL,
        export.TK_USER_DEFINED: pb.ModelProto.SentencePiece.USER_DEFINED,
        export.TK_BYTE: pb.ModelProto.SentencePiece.BYTE,
    }
    m = pb.ModelProto()
    for piece, score, t in zip(meta["pieces"], meta["scores"], meta["types"]):
        m.pieces.add(piece=piece, score=score, type=kind[t])
    ts = m.trainer_spec
    ts.model_type = pb.TrainerSpec.BPE
    ts.vocab_size = len(meta["pieces"])
    ts.byte_fallback = meta["byte_fallback"]
    ts.split_by_whitespace = True
    ts.max_sentencepiece_length = 16
    ts.unk_id, ts.bos_id, ts.eos_id, ts.pad_id = meta["unk_id"], meta["bos_id"], meta["eos_id"], meta["pad_id"]
    # The engine and RefTokenizer drop <unk>; sentencepiece prints " ⁇ " by default.
    ts.unk_surface = ""
    ns = m.normalizer_spec
    ns.name = "identity"
    ns.add_dummy_prefix = meta["add_dummy_prefix"]
    ns.remove_extra_whitespaces = False
    ns.escape_whitespaces = True
    return m


def check_model(export, sp, meta):
    """The loaded model must hold exactly the blob's pieces, scores and types.
    The Python API has no IsUserDefined, so the types are read back from the
    model proto sentencepiece serialises, and cross-checked with its
    IsControl/IsUnknown/IsByte."""
    kind = {pb.ModelProto.SentencePiece.NORMAL: export.TK_NORMAL,
            pb.ModelProto.SentencePiece.UNKNOWN: export.TK_UNKNOWN,
            pb.ModelProto.SentencePiece.CONTROL: export.TK_CONTROL,
            pb.ModelProto.SentencePiece.USER_DEFINED: export.TK_USER_DEFINED,
            pb.ModelProto.SentencePiece.BYTE: export.TK_BYTE}
    m = pb.ModelProto()
    m.ParseFromString(sp.serialized_model_proto())
    assert sp.GetPieceSize() == len(m.pieces) == len(meta["pieces"])
    for i, (piece, score, t) in enumerate(zip(meta["pieces"], meta["scores"], meta["types"])):
        assert sp.IdToPiece(i) == piece, i
        assert sp.GetScore(i) == score, i
        assert kind[m.pieces[i].type] == t, i
        assert sp.IsControl(i) == (t == export.TK_CONTROL), i
        assert sp.IsUnknown(i) == (t == export.TK_UNKNOWN), i
        assert sp.IsByte(i) == (t == export.TK_BYTE), i


def decode_cases(export, sp, ref, meta):
    n = len(meta["pieces"])
    unk, pad = meta["unk_id"], meta["pad_id"]
    byte_ids = [i for i, t in enumerate(meta["types"]) if t == export.TK_BYTE]

    def b(*bs):
        return [ref.byte_id[x] for x in bs]

    def p(*ps):
        return [ref.p2id[x] for x in ps]

    cases = [
        ("empty", []),
        ("specials", [meta["bos_id"], pad, meta["eos_id"], unk]),
        ("unk between pieces", p("▁t") + [unk] + p("▁t")),
        ("markers", [i for i, t in enumerate(meta["types"]) if t == export.TK_USER_DEFINED]),
        ("leading meta space", p("▁t", "er")),
        ("valid byte run", b(0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x99, 0x82)),
        ("truncated sequence", b(0xE2, 0x82)),
        ("truncated 4-byte sequence", b(0xF0, 0x9F, 0x98) + p("▁t")),
        ("lone continuation", b(0x80, 0xBF)),
        ("invalid lead bytes", b(0xC0, 0xC1, 0xF5, 0xFF)),
        ("surrogate", b(0xED, 0xA0, 0x80)),
        ("overlong", b(0xC0, 0x80, 0xE0, 0x80, 0x80)),
        ("above U+10FFFF", b(0xF4, 0x90, 0x80, 0x80)),
        ("valid U+FFFD bytes", b(0xEF, 0xBF, 0xBD)),
        ("lead byte before piece", b(0xC3) + p("er")),
        ("run split by control", b(0xC3) + [pad] + b(0xA9)),
        ("run split by unk", b(0xC3) + [unk] + b(0xA9)),
        ("run split by marker", b(0xC3) + p("<speech>") + b(0xA9)),
        ("meta space from bytes", b(0xE2, 0x96, 0x81)),
        ("space and control bytes", b(0x20, 0x09, 0x0A, 0x00, 0x7F)),
    ]
    for name, text in SAMPLES:
        cases.append((f"sentencepiece encode: {name}", sp.encode(text)))
    rng = random.Random(SEED)
    for k in range(48):
        cases.append((f"random {k}", [rng.randrange(n) for _ in range(rng.randint(1, 48))]))
    for k in range(16):
        # Mostly byte pieces, so random byte runs hit valid and invalid UTF-8.
        ids = [rng.choice(byte_ids) if rng.random() < 0.8 else rng.randrange(n) for _ in range(rng.randint(1, 32))]
        cases.append((f"random bytes {k}", ids))
    return [{"name": name, "ids": ids, "sentencepiece": sp.decode(ids), "ref": ref.decode(ids)}
            for name, ids in cases]


def vocab_digest(sp, n):
    """sha256 over the decode of every single id: u32 LE length, then UTF-8."""
    h = hashlib.sha256()
    for i in range(n):
        s = sp.decode([i]).encode("utf-8")
        h.update(struct.pack("<I", len(s)) + s)
    return h.hexdigest()


def needle3_check(export, sp, meta, path, cases):
    m3 = pb.ModelProto()
    with open(path, "rb") as f:
        m3.ParseFromString(f.read())
    shared = len(m3.pieces)
    kind = {1: export.TK_NORMAL, 2: export.TK_UNKNOWN, 3: export.TK_CONTROL, 4: export.TK_USER_DEFINED,
            6: export.TK_BYTE}
    piece_diff = sum(1 for i, q in enumerate(m3.pieces)
                     if (q.piece, q.score, kind[q.type]) != (meta["pieces"][i], meta["scores"][i], meta["types"][i]))
    ns, ts = m3.normalizer_spec, m3.trainer_spec
    print(f"needle3: {shared} pieces, {piece_diff} differ from the blob over ids 0..{shared - 1}")
    print(f"needle3 normalizer: name={ns.name!r} add_dummy_prefix={ns.add_dummy_prefix} "
          f"remove_extra_whitespaces={ns.remove_extra_whitespaces} escape_whitespaces={ns.escape_whitespaces}")
    print(f"needle3 trainer: model_type={ts.model_type} byte_fallback={ts.byte_fallback} "
          f"split_by_whitespace={ts.split_by_whitespace} max_sentencepiece_length={ts.max_sentencepiece_length} "
          f"unk_surface={ts.unk_surface!r}")
    ts.unk_surface = ""
    sp3 = spm.SentencePieceProcessor(model_proto=m3.SerializeToString())
    dec = [c for c in cases if all(i < shared for i in c["ids"])]
    dec_diff = sum(1 for c in dec if sp3.decode(c["ids"]) != c["sentencepiece"])
    print(f"needle3 decode: {len(dec) - dec_diff}/{len(dec)} shared-id sequences equal")
    enc = [t for name, t in SAMPLES if name != "language tokens"]
    enc_diff = sum(1 for t in enc if sp3.encode(t) != sp.encode(t))
    print(f"needle3 encode: {len(enc) - enc_diff}/{len(enc)} samples without language tokens equal")
    single_diff = sum(1 for i in range(shared) if sp3.decode([i]) != sp.decode([i]))
    print(f"needle3 single-id decode: {shared - single_diff}/{shared} equal")


def dump(fixture):
    """JSON with one decode or encode case per line, so a regeneration diffs
    by case."""
    def one(v):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

    lines = ["{"]
    for key, val in fixture.items():
        if isinstance(val, list):
            lines.append(f"{one(key)}:[")
            lines += [one(c) + ("," if i < len(val) - 1 else "") for i, c in enumerate(val)]
            lines.append("],")
        else:
            lines.append(f"{one(key)}:{one(val)},")
    lines[-1] = lines[-1].rstrip(",")
    return "\n".join(lines) + "\n}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cact", default="models/whistle/whistle.cact")
    ap.add_argument("--upstream", default="original/needle", help="cactus-compute/needle checkout at " + UPSTREAM_PIN)
    ap.add_argument("--out", default="testdata/tokenizer/whistle.json")
    ap.add_argument("--needle3", help="needle3 tokenizer.model to cross-check against")
    args = ap.parse_args()

    export = load_upstream(args.upstream)
    data, blob = raw_blob(export, args.cact)
    meta = export.parse_tokenizer_blob(blob)
    ref = export.RefTokenizer(meta)
    sp = spm.SentencePieceProcessor(model_proto=build_model(export, meta).SerializeToString())
    check_model(export, sp, meta)

    decode = decode_cases(export, sp, ref, meta)
    encode = []
    for name, text in SAMPLES:
        ids = ref.encode(text)
        encode.append({"name": name, "text": text, "ref": ids, "sentencepiece": sp.encode(text),
                       "decoded": sp.decode(ids)})

    n = len(meta["pieces"])
    fixture = {
        "comment": "Generated by scripts/tokenizer_fixture.py; see docs/FINDINGS.md F-005. Do not edit.",
        "source": {
            "cact_sha256": hashlib.sha256(data).hexdigest(),
            "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "pieces": n,
            "upstream": UPSTREAM_PIN,
            "sentencepiece": spm.__version__,
            "seed": SEED,
        },
        "vocab_decode_sha256": vocab_digest(sp, n),
        "decode": decode,
        "encode": encode,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(dump(fixture))

    ids_total = sum(len(c["ids"]) for c in decode)
    ref_diff = [c["name"] for c in decode if c["ref"] != c["sentencepiece"]]
    sp_enc_diff = [e["name"] for e in encode if e["ref"] != e["sentencepiece"]]
    trip_diff = [e["name"] for e in encode if e["decoded"] != e["text"]]
    print(f"wrote {args.out}: {len(decode)} decode sequences ({ids_total} ids), {len(encode)} encode samples")
    print(f"RefTokenizer decode differs from sentencepiece on {len(ref_diff)}: {ref_diff}")
    print(f"RefTokenizer encode differs from sentencepiece encode on {len(sp_enc_diff)}: {sp_enc_diff}")
    print(f"round trip (sentencepiece decode of RefTokenizer encode) fails on {len(trip_diff)}: {trip_diff}")
    if args.needle3:
        needle3_check(export, sp, meta, args.needle3, decode)


if __name__ == "__main__":
    main()
