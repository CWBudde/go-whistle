"""Generate the Needle 3 prompt-rendering fixture for internal/toolcall.

Renders tool-calling requests with upstream's finetune.render_example
(needle/model/finetune.py @ af2654dffa) and encodes them as run.py does
([BOS] + sentencepiece.encode(prompt)) with the pinned tokenizer.model. Each
case gives its tools as JSON text, which render_example sees as
json.loads(text), so the fixture pins Python's json.dumps(...,
separators=(",", ":"), ensure_ascii=False) byte for byte, key order and number
spelling included. The date cases run upstream's _with_date_fact
(needle/__init__.py) with the clock fixed; the function is lifted out of the
module, since importing needle/__init__ loads the engine and telemetry.

Written to testdata/text/prompt/render.json:

    {"upstream": ..., "tokenizer_sha256": ..., "now": "2026-10-12T09:30:00",
     "cases": [{"name", "system", "tools", "query", "prompt", "ids"}, ...],
     "dates": [{"system", "want"}, ...]}

The cases are the engine corpus's (testdata/text/engine/cases.json), the
forward fixture's prompts, and edge cases for the JSON and the system block.
Deterministic: two runs give the same bytes. Regenerate with

    just text-prompt-fixtures

which runs

    NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 uv run --project scripts python \
        scripts/text_prompt_fixtures.py
"""

import argparse
import ast
import datetime
import json
import os
import re
import sys

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import text_forward_fixtures as forward  # noqa: E402

NOW = datetime.datetime(2026, 10, 12, 9, 30)  # a Monday

EDGE_TOOLS = [
    {
        "name": "escapes",
        "description": 'Quote " backslash \\ newline \n tab \t bell \x07 less < greater > amp & '
        "line separator   umlaut ä emoji \U0001f600",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "zeta": 1,
        "alpha": {"z": True, "a": False, "m": None},
        "name": "numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "big": {"type": "integer", "minimum": -12345678901234567890, "maximum": 0},
                "half": {"type": "number", "default": 21.5},
                "one": {"type": "number", "default": 1.0},
                "tiny": {"type": "number", "default": 1e-07},
                "huge": {"type": "number", "default": 1e16},
                "below": {"type": "number", "default": 9999999999999998.0},
                "small": {"type": "number", "default": 0.0001},
                "neg": {"type": "number", "default": -0.0},
            },
        },
    },
]

# Edge cases: name -> (system, tools JSON text, query). The tools text is
# what a caller passes; render_example sees json.loads of it.
EDGE_CASES = {
    "no_system": ("", '[{"name":"x","parameters":{"type":"object"}}]', "do x"),
    "blank_system": ("  \n\t ", '[{"name":"x","parameters":{"type":"object"}}]', "do x"),
    "padded_system": ("\n date: 2026-10-12 Mon 09:30 \n", "[]", "hello"),
    "no_tools": ("date: 2026-10-12 Mon 09:30", "[]", "what can you do"),
    "edge_tools": ("date: 2026-10-12 Mon 09:30", json.dumps(EDGE_TOOLS), "line one\nline two"),
    "float_spellings": (
        "",
        '[{"name":"f","parameters":{"a":1e2,"b":1.50,"c":2E-5,"d":-0,"e":100,"f":0.1e1,"g":123456789012345678}}]',
        "q",
    ),
    "duplicates_overflow": ("", '[{"a":1,"b":{"x":1,"y":2,"x":3},"a":[1e400,-1e400],"c":-0.0}]', "q"),
    "nonfinite_literals": ("", '[{"a":Infinity,"b":-Infinity,"c":NaN,"d":[ -Infinity ]}]', "q"),
    "escaped_input": ("", '[{"name":"caf\\u00e9","description":"\\u2028\\/\\u0041"}]', "q"),
    "query_markers": ("", "[]", "say <tools> and </tools>"),
}

DATE_SYSTEMS = [
    "",
    "   ",
    "locale: de-DE",
    "date: 2026-01-01 Thu 08:00",
    "update: weekly; locale: en-US",
    "candidate:yes",
    "today is 2026-02-03",
    '{"locale": "fr-FR"}',
    "  {not json",
    "mydate:x",
]


def with_date_fact(upstream):
    """upstream's _with_date_fact with datetime.now fixed to NOW."""
    path = os.path.join(upstream, "needle", "__init__.py")
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    keep = {"_ISO_STAMP", "_DATE_KEY", "_with_date_fact"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in keep)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in keep for t in n.targets))
    ]
    if len(nodes) != len(keep):
        sys.exit(f"{path}: found {len(nodes)} of {sorted(keep)}")

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW

    clock = type("datetime_module", (), {"datetime": FixedDatetime})
    ns = {"re": re, "datetime": clock}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns["_with_date_fact"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", default="original/needle")
    ap.add_argument("--safetensors", default="models/needle3/needle3.safetensors")
    ap.add_argument("--tokenizer", default="models/needle3/tokenizer.model")
    ap.add_argument("--engine-cases", default="testdata/text/engine/cases.json")
    ap.add_argument("--out", default="testdata/text/prompt")
    args = ap.parse_args()

    forward.check_pins(args.upstream, args.safetensors, args.tokenizer)
    _, finetune, tokmod = forward.load_upstream(args.upstream)
    tok = tokmod.SANTokenizer(args.tokenizer)

    with open(args.engine_cases) as f:
        corpus = json.load(f)

    requests = []
    for case in corpus["cases"]:
        tools = json.dumps(corpus["toolsets"][case["toolset"]], separators=(",", ":"), ensure_ascii=False)
        requests.append(("engine/" + case["id"], case.get("system", corpus["system"]), tools, case["query"]))
    for name, case in forward.CASES.items():
        tools = json.dumps(case["tools"], separators=(",", ":"), ensure_ascii=False)
        requests.append(("forward/" + name, case["system"], tools, case["query"]))
    for name, (system, tools, query) in EDGE_CASES.items():
        requests.append(("edge/" + name, system, tools, query))

    cases = []
    for name, system, tools, query in requests:
        prompt, _ = finetune.render_example({"system": system, "tools": json.loads(tools), "query": query})
        ids = [tokmod.BOS_ID] + tok.encode(prompt)
        cases.append({"name": name, "system": system, "tools": tools, "query": query, "prompt": prompt, "ids": ids})

    date_fact = with_date_fact(args.upstream)
    dates = [{"system": s, "want": date_fact(s)} for s in DATE_SYSTEMS]

    out = {
        "upstream": forward.UPSTREAM_PIN,
        "tokenizer_sha256": forward.pinned("tokenizer.model"),
        "now": NOW.isoformat(),
        "cases": cases,
        "dates": dates,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "render.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {path}: {len(cases)} cases, {len(dates)} date cases", file=sys.stderr)


if __name__ == "__main__":
    main()
