"""Exact, free grading of closed questions (P/F statements, A-D choices) against the official key.

The key is parsed from the marking scheme ("Rozwiązanie": "1 – F / 2 – P / 3 – P" or a single letter).
Partial credit follows the scheme: 3 statements -> 3 right = 2 pts, 2 right = 1 pt; 2 slots -> 2 right = 2, 1 right = 1.
"""
import re

from . import scheme

_POINTS = {3: {3: 2, 2: 1}, 2: {2: 2, 1: 1}, 1: {1: 1}}   # slots -> {n correct: points}


def key_of(item_id: str) -> dict[str, str]:
    body = scheme.load()[item_id]["rules"].split("Rozwiązanie", 1)[1]
    pairs = re.findall(r"^(\d+)\s*[–-]\s*([A-DPF])\s*$", body, re.M)
    if pairs:
        return dict(pairs)
    m = re.search(r"^\s*([A-D])\s*$", body, re.M)
    if not m:
        raise ValueError(f"cannot read the answer key of task {item_id}")
    return {"": m.group(1)}


def parse_answer(text: str, key: dict[str, str]) -> dict[str, str]:
    """'1: P 2: F 3: P' -> {'1': 'P', ...};  'C' -> {'': 'C'}."""
    if "" in key:
        m = re.search(r"\b([A-D])\b", text.upper())
        return {"": m.group(1)} if m else {}
    return dict(re.findall(r"(\d+)\s*[:.\-–)]\s*([A-DPF])\b", text.upper()))


def grade(item_id: str, answer: str) -> tuple[int, int, str]:
    """Returns (points, max_points, reason)."""
    key = key_of(item_id)
    given = parse_answer(answer, key)
    ok = sum(1 for k, v in key.items() if given.get(k) == v)
    mx = _POINTS[len(key)][len(key)]
    pts = _POINTS[len(key)].get(ok, 0)
    if not given:
        return 0, mx, "answer not in the expected format"
    wrong = [k or "answer" for k, v in key.items() if given.get(k) != v]
    return pts, mx, f"{ok}/{len(key)} correct" + (f" (wrong: {', '.join(wrong)})" if wrong else "")
