"""Generate the sentencepiece fixture for internal/spm (Needle 3 text tokenizer).

Loads Cactus-Compute/needle3 tokenizer/tokenizer.model with the Python
`sentencepiece` library, exactly as upstream needle/model/tokenizer.py
SANTokenizer does, and records for a corpus of texts the ids of
`Encode(text, out_type=int)` and the `Decode` of those ids, plus the decode of
fixed and seeded random id sequences and a digest over the decode of every
single id. The corpus is mostly tool-calling prompts and targets rendered with
upstream needle/model/finetune.py `render_example` (copied below so the script
does not import the upstream package), in several languages, plus edge cases:
whitespace runs, markers, spelled-out special pieces, digits, URLs, code and
characters that need byte fallback. A few inputs are raw bytes with invalid
UTF-8 (recorded as hex), which sentencepiece maps to U+FFFD.

When the needle3.cact archive is present (--cact), upstream export.py
`RefTokenizer` also encodes every text from the archive's RAW tokenizer blob;
its ids are recorded as `ref` where they differ from sentencepiece's. The Go
tests use that to compare internal/tokenizer (the RefTokenizer port) with
internal/spm.

Deterministic: the random sequences come from a seeded PRNG and the output
carries no timestamps. The output is JSON Lines (a header object, then one
case per line), so a regeneration diffs by case. Regenerate with

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/text_tokenizer_fixture.py [models/needle3/tokenizer.model]
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
HF_REVISION = "93e88ede867e4311c39151285e74df513e4b835b"
SEED = 20261011

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_START, THINK_END = "<think>", "</think>"
TOOLS_START, TOOLS_END = "<tools>", "</tools>"
TOOL_CALL_START, TOOL_CALL_END = "<tool_call>", "</tool_call>"

SYSTEM_DATE = "date: 2026-10-11 Sun 09:30"


def render_example(example):
    """Upstream needle/model/finetune.py render_example @ af2654dffa."""
    tools = example.get("tools", [])
    tools_json = tools if isinstance(tools, str) else json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    answers = example.get("answers", example.get("function_calls", []))
    answers_json = answers if isinstance(answers, str) else json.dumps(answers, separators=(",", ":"), ensure_ascii=False)
    reasoning = (example.get("reasoning") or "").strip()
    system = (example.get("system") or "").strip()
    prefix = IM_START + "system\n" + system + IM_END + "\n" if system else ""
    prompt = (prefix + IM_START + "user\n" + TOOLS_START + tools_json + TOOLS_END + "\n"
              + example["query"] + IM_END + "\n" + IM_START + "assistant\n")
    think = THINK_START + "\n" + reasoning + "\n" + THINK_END + "\n" if reasoning else ""
    target = think + TOOL_CALL_START + answers_json + TOOL_CALL_END + IM_END
    return prompt, target


def tool(name, description, props, required=()):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}


def s(description, **extra):
    return {"type": "string", "description": description, **extra}


WEATHER = tool("get_weather", "Get the current weather for a location",
               {"location": s("City name, e.g. Berlin"),
                "unit": s("Temperature unit", enum=["celsius", "fahrenheit"])}, ["location"])
ALARM = tool("set_alarm", "Set an alarm", {"time": s("Time as HH:MM"), "label": s("Optional label")}, ["time"])
TIMER = tool("set_timer", "Start a countdown timer",
             {"minutes": {"type": "integer", "minimum": 1, "maximum": 1440}}, ["minutes"])
MESSAGE = tool("send_message", "Send a text message to a contact",
               {"recipient": s("Contact name"), "text": s("Message body")}, ["recipient", "text"])
EVENT = tool("create_event", "Create a calendar event",
             {"title": s("Event title"), "start": s("ISO 8601 start"), "end": s("ISO 8601 end"),
              "location": s("Where")}, ["title", "start"])
SEARCH = tool("web_search", "Search the web",
              {"query": s("Search terms"), "max_results": {"type": "integer", "default": 5}}, ["query"])
MUSIC = tool("play_music", "Play music",
             {"artist": s("Artist"), "song": s("Song title"), "shuffle": {"type": "boolean", "default": False}})
CURRENCY = tool("convert_currency", "Convert an amount between currencies",
                {"amount": {"type": "number"}, "from": s("ISO 4217 code"), "to": s("ISO 4217 code")},
                ["amount", "from", "to"])
NAVIGATE = tool("navigate", "Start navigation",
                {"destination": s("Address or place"),
                 "mode": s("Travel mode", enum=["driving", "walking", "cycling", "transit"])}, ["destination"])
LIGHTS_DE = tool("licht_schalten", "Schaltet das Licht in einem Raum ein, aus oder dimmt es",
                 {"raum": s("Raum, z. B. Küche oder Schlafzimmer"),
                  "helligkeit": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Helligkeit in %"},
                  "farbe": s("Farbe als Hex, z. B. #FFAA00")}, ["raum"])
THERMOSTAT = tool("set_thermostat", "Set the heating target temperature",
                  {"room": s("Room"), "celsius": {"type": "number", "minimum": 5, "maximum": 30}}, ["celsius"])
NOTE_FR = tool("creer_note", "Crée une note avec un titre et un contenu",
               {"titre": s("Titre de la note"), "contenu": s("Contenu, peut contenir des retours à la ligne")},
               ["titre"])

EXAMPLES = [
    ("en weather", {"system": SYSTEM_DATE, "tools": [WEATHER, ALARM],
                    "query": "What's the weather like in Berlin tomorrow?",
                    "reasoning": "The user asks for the weather in Berlin -> get_weather",
                    "answers": [{"name": "get_weather", "arguments": {"location": "Berlin"}}]}),
    ("de alarm", {"system": SYSTEM_DATE + "; Antworte auf Deutsch.", "tools": [ALARM, TIMER],
                  "query": "Weck mich morgen um 6:45 Uhr, Beschriftung „Joggen“.",
                  "reasoning": "Wecker für 06:45 mit Beschriftung 'Joggen' -> set_alarm",
                  "answers": [{"name": "set_alarm", "arguments": {"time": "06:45", "label": "Joggen"}}]}),
    ("fr message", {"system": SYSTEM_DATE, "tools": [MESSAGE, NOTE_FR],
                    "query": "Envoie un message à Chloé : « J'arrive dans 10 minutes, désolé ! »",
                    "reasoning": "Message à Chloé -> send_message",
                    "answers": [{"name": "send_message",
                                 "arguments": {"recipient": "Chloé", "text": "J'arrive dans 10 minutes, désolé !"}}]}),
    ("es alarm", {"system": SYSTEM_DATE, "tools": [ALARM],
                  "query": "¿Puedes poner una alarma a las 7:30 para mañana? ¡Gracias, señor!",
                  "answers": [{"name": "set_alarm", "arguments": {"time": "07:30"}}]}),
    ("it weather", {"system": SYSTEM_DATE, "tools": [WEATHER],
                    "query": "Che tempo fa a Napoli? Dimmi anche la temperatura in gradi Fahrenheit, perché è più facile.",
                    "reasoning": "Meteo a Napoli in fahrenheit",
                    "answers": [{"name": "get_weather", "arguments": {"location": "Napoli", "unit": "fahrenheit"}}]}),
    ("nl timer", {"system": SYSTEM_DATE, "tools": [TIMER, ALARM],
                  "query": "Zet een timer van 25 minuten voor de pizza, alsjeblieft.",
                  "answers": [{"name": "set_timer", "arguments": {"minutes": 25}}]}),
    ("pl message", {"system": SYSTEM_DATE, "tools": [MESSAGE],
                    "query": "Wyślij wiadomość do Łukasza: „Spóźnię się o źdźbło, przepraszam”. Zażółć gęślą jaźń.",
                    "reasoning": "Wiadomość do Łukasza -> send_message",
                    "answers": [{"name": "send_message",
                                 "arguments": {"recipient": "Łukasz", "text": "Spóźnię się o źdźbło, przepraszam"}}]}),
    ("ja weather", {"system": SYSTEM_DATE, "tools": [WEATHER],
                    "query": "明日の東京の天気を教えてください。",
                    "reasoning": "東京の天気 -> get_weather",
                    "answers": [{"name": "get_weather", "arguments": {"location": "東京"}}]}),
    ("zh lights", {"system": SYSTEM_DATE, "tools": [LIGHTS_DE],
                   "query": "帮我把客厅的灯调到50%亮度。",
                   "answers": [{"name": "licht_schalten", "arguments": {"raum": "客厅", "helligkeit": 50}}]}),
    ("emoji music", {"system": SYSTEM_DATE, "tools": [MUSIC],
                     "query": "Play some music 🎵🎶 by Daft Punk, shuffle on 🔀 👍🏽",
                     "reasoning": "Daft Punk, shuffle=true",
                     "answers": [{"name": "play_music", "arguments": {"artist": "Daft Punk", "shuffle": True}}]}),
    ("currency two calls", {"system": SYSTEM_DATE, "tools": [CURRENCY],
                            "query": "Convert 1,234.56 EUR to JPY and to USD.",
                            "reasoning": "Two conversions: EUR->JPY and EUR->USD",
                            "answers": [{"name": "convert_currency", "arguments": {"amount": 1234.56, "from": "EUR", "to": "JPY"}},
                                        {"name": "convert_currency", "arguments": {"amount": 1234.56, "from": "EUR", "to": "USD"}}]}),
    ("long numbers", {"system": SYSTEM_DATE, "tools": [TIMER, MESSAGE],
                      "query": "Text +49 (0) 30 1234-5678: order #000123456789 ships 2026-10-11T09:30:00Z; timer 90 min.",
                      "answers": [{"name": "send_message", "arguments": {"recipient": "+49 (0) 30 1234-5678",
                                                                         "text": "order #000123456789 ships 2026-10-11T09:30:00Z"}},
                                  {"name": "set_timer", "arguments": {"minutes": 90}}]}),
    ("url search", {"system": SYSTEM_DATE, "tools": [SEARCH],
                    "query": "Search https://example.com/path/to/page?q=needle&lang=de#top and give me 3 results",
                    "answers": [{"name": "web_search",
                                 "arguments": {"query": "https://example.com/path/to/page?q=needle&lang=de#top",
                                               "max_results": 3}}]}),
    ("multiline event", {"system": SYSTEM_DATE + "; timezone: Europe/Berlin", "tools": [EVENT],
                         "query": "Create \"Team sync\" on Monday from 10:00 to 10:30 in Room 4.01\nAdd: bring the slides.",
                         "reasoning": "Monday after 2026-10-11 is 2026-10-12\n10:00-10:30",
                         "answers": [{"name": "create_event",
                                      "arguments": {"title": "Team sync", "start": "2026-10-12T10:00:00",
                                                    "end": "2026-10-12T10:30:00", "location": "Room 4.01"}}]}),
    ("no tools no answer", {"tools": [], "query": "Hello there!", "answers": []}),
    ("upstream template", {"tools": [], "query": "q", "reasoning": "'q' -> f",
                           "answers": [{"name": "f", "arguments": {}}]}),
    ("code-ish tools", {"system": SYSTEM_DATE,
                        "tools": [{"name": "run_query", "description": "Run SQL: SELECT * FROM t WHERE a <> b;",
                                   "parameters": {"type": "object", "properties": {
                                       "sql": {"type": "string"}, "limit": {"type": ["integer", "null"]},
                                       "params": {"type": "array", "items": {"type": "number"}},
                                       "dry_run": {"type": "boolean", "default": True}},
                                       "required": ["sql"]}}],
                        "query": "count rows where x >= 10 && y != 'z'",
                        "answers": [{"name": "run_query", "arguments": {
                            "sql": "SELECT COUNT(*) FROM t WHERE x >= ? AND y <> ?", "params": [10, 0.5e-3],
                            "limit": None, "dry_run": False}}]}),
    ("fr navigate", {"system": SYSTEM_DATE, "tools": [NAVIGATE],
                     "query": "Où se trouve le café le plus proche ? Navigue à pied, s'il te plaît.",
                     "answers": [{"name": "navigate", "arguments": {"destination": "café le plus proche", "mode": "walking"}}]}),
    ("pt music", {"system": SYSTEM_DATE, "tools": [MUSIC],
                  "query": "Toque a música 'Garota de Ipanema' de Tom Jobim, por favor.",
                  "answers": [{"name": "play_music", "arguments": {"artist": "Tom Jobim", "song": "Garota de Ipanema"}}]}),
    ("whitespace query", {"system": SYSTEM_DATE, "tools": [LIGHTS_DE],
                          "query": "  set   lights\tin the  kitchen to 80 %  ",
                          "answers": [{"name": "licht_schalten", "arguments": {"raum": "Küche", "helligkeit": 80}}]}),
    ("json system", {"system": '{"user":"Anna","locale":"de-DE","date":"2026-10-11"}', "tools": [THERMOSTAT],
                     "query": "Mach es im Wohnzimmer wärmer, 21,5 °C bitte.",
                     "answers": [{"name": "set_thermostat", "arguments": {"room": "Wohnzimmer", "celsius": 21.5}}]}),
    ("de lights", {"system": SYSTEM_DATE, "tools": [LIGHTS_DE, THERMOSTAT],
                   "query": "Schalte das Licht im Schlafzimmer aus. Grüße an Straße, Maß, Äpfel, Öl und Übermut!",
                   "reasoning": "Licht aus = Helligkeit 0",
                   "answers": [{"name": "licht_schalten", "arguments": {"raum": "Schlafzimmer", "helligkeit": 0}}]}),
    ("string tools", {"tools": '[{"name":"ping"}]', "query": "ping", "answers": '[{"name":"ping","arguments":{}}]'}),
]

TEXTS = [
    ("empty", ""),
    ("single space", " "),
    ("spaces only", "   "),
    ("leading spaces", "   leading"),
    ("trailing spaces", "trailing   "),
    ("space runs", "a  b   c    d"),
    ("tab", "\t"),
    ("newline", "\n"),
    ("crlf", "line one\r\nline two\r\n"),
    ("tsv", "name\tvalue\tunit\nwidth\t12\tcm\n"),
    ("blank lines", "line1\n\n\nline4"),
    ("mixed whitespace", " \n \t   　 "),
    ("nbsp", "100 km h"),
    ("all markers", "".join(["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tools>", "</tools>",
                             "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>", "<context>",
                             "</context>", "<image>", "</image>", "<speech>", "</speech>", "<tts>", "</tts>",
                             "<imagen>", "</imagen>", "<extract>", "</extract>", "<schema>", "</schema>"])),
    ("lone im_start", "<|im_start|>"),
    ("lone im_end", "<|im_end|>"),
    ("lone think", "<think>"),
    ("lone tool_call end", "</tool_call>"),
    ("doubled markers", "<think><think></think></think>"),
    ("markers with spaces", " <think> x </think> "),
    ("near markers", "<think <|im_start| <tool_call<tool_call>> < think > <THINK> <Think> </ think>"),
    ("marker in word", "pre<tools>post"),
    ("specials spelled out", "<s> </s> <pad> <unk> <0x41> <0xff> <0xFF>"),
    ("specials glued", "<s></s><pad><unk><0x00>"),
    ("tool result block", "<tool_result>{\"temp_c\":12.5,\"sky\":\"cloudy\"}</tool_result>"),
    ("context block", "<context>Anna's flight LH 400 leaves FRA at 13:05.</context>"),
    ("digits", "0 1 22 333 4444 55555 666666 7777777"),
    ("long number", "1234567890123456789012345678901234567890"),
    ("decimals", "3.14159265358979 -42 +7 1e-9 6.022e23 0.5"),
    ("times and dates", "12:30 09:05:59 2026-10-11 11.10.2026 10/11/26"),
    ("phone", "+1-800-555-0199 (030) 123 45 67"),
    ("url", "https://huggingface.co/Cactus-Compute/needle3/resolve/93e88ede/tokenizer/tokenizer.model?download=true"),
    ("email", "anna.mueller+test@example.de"),
    ("python", "def f(x):\n    return {\"a\": [1, 2.5, None, True]}\n"),
    ("c code", "if (a && b) { x->y = z[0]; } // done"),
    ("json escapes", "{\"k\":\"\\u00fc\\n\\t\\\"\",\"n\":null}"),
    ("meta space literal", "▁"),
    ("meta space words", "▁▁hello▁world ▁"),
    ("english", "The quick brown fox jumps over the lazy dog."),
    ("german", "Grüße aus Köln! Die Straße ist schön, oder? Maß, Äpfel, Öl, Übermut, ẞ."),
    ("french", "Où est la bibliothèque ? Ça coûte 12,50 € — à bientôt, Noël !"),
    ("spanish", "¿Dónde está el niño? ¡Mañana será otro día!"),
    ("italian", "Perché è così difficile? Più tardi, forse."),
    ("dutch", "Het is een mooie dag in Utrecht, zei ze tegen Eva en Ruud."),
    ("polish", "Zażółć gęślą jaźń, powiedział Krzysztof."),
    ("turkish", "İstanbul'da ılık bir gün; Iğdır."),
    ("portuguese", "Não sei se você já viu a promoção."),
    ("japanese", "こんにちは、世界。カタカナと漢字。"),
    ("chinese", "你好，世界。今天天气很好。"),
    ("korean", "안녕하세요 세계"),
    ("russian", "Привет, мир! Как дела?"),
    ("greek", "Γειά σου Κόσμε"),
    ("arabic", "مرحبا بالعالم"),
    ("hindi", "नमस्ते दुनिया"),
    ("emoji", "🙂 ok 👍🏽 🇩🇪 👨‍👩‍👧 🦜🪿"),
    ("math symbols", "x² + y² = z²; π ≈ 3.14159; ∑ ∞ ≠ ∅; 100% ✓"),
    ("fraktur", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢"),
    ("combining", "é vs é; Å vs Å; ﬁ ligature"),
    ("control chars", "nul\u0000del\u007fesc\u001bbell\u0007"),
    ("bom and zw", "﻿bom zero​width zw‍j"),
    ("plane 16 and fffd", "\U0010ffff � \U0001f600"),
    ("repeats", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa hahahahaha ...... !!!??? ------"),
    ("mixed scripts", "Siobhan said «Привет» and 你好 to Zoë 🙂 at 9:30."),
]

# Raw byte inputs with invalid UTF-8; sentencepiece replaces each invalid
# byte by U+FFFD before encoding.
BYTE_TEXTS = [
    ("invalid ff", b"a\xffb"),
    ("truncated euro", b"\xe2\x82"),
    ("truncated at end", b"ok \xc3"),
    ("surrogate", b"x\xed\xa0\x80y"),
    ("overlong", b"\xc0\xaf\xe0\x80\xaf"),
    ("above 10ffff", b"\xf4\x90\x80\x80"),
    ("lone continuation", b"\x80\xbf hi"),
    ("valid mixed invalid", "Grüße ".encode() + b"\xfe" + " <think>".encode()),
]


def load_ref(upstream, cact):
    """RefTokenizer over the archive's RAW tokenizer blob, importing
    needle.model.export without running needle/__init__.py, which loads the
    engine and telemetry. Returns (tokenizer, blob sha256) or (None, None)."""
    if not cact or not os.path.exists(cact):
        return None, None
    pkg = types.ModuleType("needle")
    pkg.__path__ = [os.path.join(upstream, "needle")]
    sys.modules["needle"] = pkg
    export = importlib.import_module("needle.model.export")
    blob = export.read_tokenizer_blob(cact)
    return export.RefTokenizer(export.parse_tokenizer_blob(blob)), hashlib.sha256(blob).hexdigest()


def decode_sequences(sp, m):
    n = sp.GetPieceSize()
    types_ = [p.type for p in m.pieces]
    byte_ids = [i for i, t in enumerate(types_) if t == pb.ModelProto.SentencePiece.BYTE]
    users = [i for i, t in enumerate(types_) if t == pb.ModelProto.SentencePiece.USER_DEFINED]

    def b(*bs):
        return [sp.PieceToId(f"<0x{x:02X}>") for x in bs]

    def p(*ps):
        ids = [sp.PieceToId(x) for x in ps]
        assert all(sp.IdToPiece(i) == x for i, x in zip(ids, ps)), ps
        return ids

    unk, pad, bos, eos = sp.unk_id(), sp.pad_id(), sp.bos_id(), sp.eos_id()
    cases = [
        ("empty", []),
        ("specials", [bos, pad, eos, unk]),
        ("unk alone", [unk]),
        ("unk run", [unk, unk, unk]),
        ("unk between pieces", p("▁t") + [unk] + p("▁t")),
        ("bos text eos", [bos] + p("▁t", "en") + [eos]),
        ("all user-defined", users),
        ("leading meta space", p("▁t", "er")),
        ("valid byte run", b(0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x99, 0x82)),
        ("truncated sequence", b(0xE2, 0x82)),
        ("truncated 4-byte sequence", b(0xF0, 0x9F, 0x98) + p("▁t")),
        ("lone continuation", b(0x80, 0xBF)),
        ("invalid lead bytes", b(0xC0, 0xC1, 0xF5, 0xFF)),
        ("surrogate bytes", b(0xED, 0xA0, 0x80)),
        ("overlong bytes", b(0xC0, 0x80, 0xE0, 0x80, 0x80)),
        ("above U+10FFFF", b(0xF4, 0x90, 0x80, 0x80)),
        ("valid U+FFFD bytes", b(0xEF, 0xBF, 0xBD)),
        ("lead byte before piece", b(0xC3) + p("er")),
        ("run split by control", b(0xC3) + [pad] + b(0xA9)),
        ("run split by unk", b(0xC3) + [unk] + b(0xA9)),
        ("run split by marker", b(0xC3) + p("<think>") + b(0xA9)),
        ("meta space from bytes", b(0xE2, 0x96, 0x81)),
        ("space and control bytes", b(0x20, 0x09, 0x0A, 0x00, 0x7F)),
        ("first and last id", [0, n - 1]),
    ]
    rng = random.Random(SEED)
    for k in range(40):
        cases.append((f"random {k}", [rng.randrange(n) for _ in range(rng.randint(1, 48))]))
    for k in range(16):
        # Mostly byte pieces, so random byte runs hit valid and invalid UTF-8.
        ids = [rng.choice(byte_ids) if rng.random() < 0.8 else rng.randrange(n) for _ in range(rng.randint(1, 32))]
        cases.append((f"random bytes {k}", ids))
    return [{"kind": "decode", "name": name, "ids": ids, "text": sp.decode(ids)} for name, ids in cases]


def vocab_digest(sp):
    """sha256 over the decode of every single id: u32 LE length, then UTF-8."""
    h = hashlib.sha256()
    for i in range(sp.GetPieceSize()):
        d = sp.decode([i]).encode("utf-8")
        h.update(struct.pack("<I", len(d)) + d)
    return h.hexdigest()


def encode_cases(sp, ref):
    texts = []
    for name, example in EXAMPLES:
        prompt, target = render_example(example)
        texts.append((f"{name} prompt", prompt))
        texts.append((f"{name} target", target))
    texts += TEXTS

    out = []
    for name, text in texts:
        ids = sp.encode(text, out_type=int)
        case = {"kind": "encode", "name": name, "text": text, "ids": ids, "decoded": sp.decode(ids)}
        if ref is not None:
            r = ref.encode(text)
            if r != ids:
                case["ref"] = r
        out.append(case)
    for name, raw in BYTE_TEXTS:
        ids = sp.encode(raw, out_type=int)
        out.append({"kind": "encode", "name": name, "text_hex": raw.hex(), "ids": ids, "decoded": sp.decode(ids)})
    return out


def spec_dict(msg, fields):
    return {f: getattr(msg, f) for f in fields}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="models/needle3/tokenizer.model")
    ap.add_argument("--cact", default="models/needle3/needle3.cact",
                    help="needle3.cact whose RAW tokenizer blob RefTokenizer encodes with (skipped when missing)")
    ap.add_argument("--upstream", default="original/needle", help="cactus-compute/needle checkout at " + UPSTREAM_PIN)
    ap.add_argument("--out", default="testdata/text/tokenizer/needle3.jsonl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        data = f.read()
    m = pb.ModelProto()
    m.ParseFromString(data)
    sp = spm.SentencePieceProcessor(model_proto=data)
    ref, blob_sha = load_ref(args.upstream, args.cact)

    cases = encode_cases(sp, ref) + decode_sequences(sp, m)
    header = {
        "comment": "Generated by scripts/text_tokenizer_fixture.py. Do not edit.",
        "source": {
            "model_sha256": hashlib.sha256(data).hexdigest(),
            "model_bytes": len(data),
            "hf_repo": "Cactus-Compute/needle3",
            "hf_revision": HF_REVISION,
            "cact_blob_sha256": blob_sha,
            "upstream": UPSTREAM_PIN,
            "sentencepiece": spm.__version__,
            "seed": SEED,
        },
        "pieces": sp.GetPieceSize(),
        "special": {"unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id(), "pad": sp.pad_id()},
        "normalizer": spec_dict(m.normalizer_spec, ["name", "add_dummy_prefix", "remove_extra_whitespaces",
                                                    "escape_whitespaces"]),
        "trainer": spec_dict(m.trainer_spec, ["model_type", "vocab_size", "byte_fallback", "split_digits",
                                              "split_by_whitespace", "treat_whitespace_as_suffix",
                                              "max_sentencepiece_length", "unk_surface"]),
        "vocab_decode_sha256": vocab_digest(sp),
    }

    def one(v):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(one(header) + "\n")
        for c in cases:
            f.write(one(c) + "\n")

    enc = [c for c in cases if c["kind"] == "encode"]
    dec = [c for c in cases if c["kind"] == "decode"]
    trip = [c["name"] for c in enc if "text" in c and c["decoded"] != c["text"]]
    refd = [c["name"] for c in enc if "ref" in c]
    print(f"wrote {args.out}: {len(enc)} encode texts ({sum(len(c['ids']) for c in enc)} ids), "
          f"{len(dec)} decode sequences ({sum(len(c['ids']) for c in dec)} ids)")
    print(f"round trip differs on {len(trip)}: {trip}")
    if ref is None:
        print("RefTokenizer: skipped, no archive at", args.cact)
    else:
        print(f"RefTokenizer encode differs from sentencepiece on {len(refd)}: {refd}")


if __name__ == "__main__":
    main()
