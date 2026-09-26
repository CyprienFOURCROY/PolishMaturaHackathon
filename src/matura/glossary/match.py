"""Find glossary terms in Polish text, tolerant of diacritics and inflection."""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_EXTRA = {"ł": "l", "Ł": "l"}


def normalize(text: str) -> str:
    """Lowercase + strip diacritics, one output char per input char (so spans map back)."""
    out = []
    for ch in text:
        if ch in _EXTRA:
            out.append(_EXTRA[ch])
            continue
        base = unicodedata.normalize("NFKD", ch)[0]
        out.append(base.lower())
    return "".join(out)


@lru_cache(maxsize=None)
def load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pattern(form: str) -> re.Pattern:
    f = normalize(form)
    # long forms match by stem (covers Polish endings); short ones must match exactly
    if len(f) >= 6 and " " not in f:
        return re.compile(r"\b" + re.escape(f[: max(5, len(f) - 2)]) + r"\w*")
    return re.compile(r"\b" + re.escape(f) + r"\b")


def find(text: str, entries: list[dict]) -> list[tuple[int, int, dict]]:
    """Non-overlapping (start, end, entry) spans, longest match first."""
    norm = normalize(text)
    hits = []
    for e in entries:
        for form in e["pl"]:
            for m in _pattern(form).finditer(norm):
                hits.append((m.start(), m.end(), e))
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    taken, spans = [], []
    for s, e_, entry in hits:
        if all(e_ <= a or s >= b for a, b in taken):
            taken.append((s, e_))
            spans.append((s, e_, entry))
    return sorted(spans, key=lambda h: h[0])
