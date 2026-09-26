"""Check a (cheap) judge model on closed questions where the right score is known from the official key.

    python3 scripts/check_closed_judge.py gpt-4.1-nano gpt-5-nano

For every closed question it grades 3 synthetic answers (all correct / one wrong / all wrong), never touches
cache/judge, and reports disagreements with the score the marking scheme prescribes. Costs a few cents.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matura import config
from matura.eval import judge, scheme
from matura.router import route
from matura.schema import load_exam

FLIP = {"P": "F", "F": "P", "A": "B", "B": "C", "C": "D", "D": "A"}


def key_of(rules: str) -> list[tuple[str, str]]:
    body = rules.split("Rozwiązanie", 1)[1]
    pairs = re.findall(r"^(\d+)\s*[–-]\s*([A-DPF])\s*$", body, re.M)
    return pairs or [("", re.search(r"^\s*([A-D])\s*$", body, re.M).group(1))]


def expected(n_correct: int, n: int) -> int:
    return {3: {3: 2, 2: 1}, 2: {2: 2, 1: 1}, 1: {1: 1}}[n].get(n_correct, 0)


def fmt(pairs):
    return " ".join(f"{k}: {v}" for k, v in pairs) if pairs[0][0] else pairs[0][1]


models = sys.argv[1:] or [judge.CLOSED_MODEL]
_, items = load_exam(config.EXAM_PATH)
rules = scheme.load()
from openai import OpenAI
client = OpenAI(api_key=judge.api_key())
for model in models:
    bad = total = tok = 0
    for it in items:
        if route(it).kind != "closed":
            continue
        key = key_of(rules[it.id]["rules"])
        variants = {"all correct": key, "one wrong": [key[0]] + [(k, v) for k, v in key[1:]]}
        variants["one wrong"] = [(key[0][0], FLIP[key[0][1]])] + key[1:]
        variants["all wrong"] = [(k, FLIP[v]) for k, v in key]
        for name, ans in variants.items():
            n_ok = sum(a == b for a, b in zip(ans, key))
            want = expected(n_ok, len(key))
            g = judge.grade(it, fmt(ans), model, client, save=False)
            total += 1
            tok += g["tokens"]["in"] + g["tokens"]["out"] + g["tokens"]["reasoning"]
            if g["points"] != want:
                bad += 1
                print(f"  [{model}] Q{it.id} {name}: answer {fmt(ans)!r} -> judge {g['points']}, expected {want}. {g['reason']}")
    print(f"{model}: {total - bad}/{total} correct, {tok} tokens total")
