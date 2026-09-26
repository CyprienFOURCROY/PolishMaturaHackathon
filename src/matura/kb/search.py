"""Keyword (BM25) search over fact cards in data/kb/*.jsonl. No embedding model, so no size cost."""
import json
import math
import re
from collections import Counter
from functools import lru_cache

from ..config import KB_DIR, KB_TOP_K

STOP = set("the a an of in on at to and or is was were by for with as from that this it its be are".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


@lru_cache(maxsize=1)
def _index():
    cards = [json.loads(l) for p in sorted(KB_DIR.glob("*.jsonl")) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    # keywords are repeated to weight them above body text
    docs = [tokens(c["title"] + " " + " ".join(c["keywords"]) * 3 + " " + c["text"]) for c in cards]
    df = Counter(t for d in docs for t in set(d))
    avg = sum(map(len, docs)) / max(len(docs), 1)
    return cards, docs, df, avg


def search(query: str, k: int = KB_TOP_K, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    cards, docs, df, avg = _index()
    q, n, scored = tokens(query), len(docs), []
    for card, doc in zip(cards, docs):
        tf, s = Counter(doc), 0.0
        for t in set(q):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg))
        if s > 0:
            scored.append((s, card))
    return [c for _, c in sorted(scored, key=lambda x: -x[0])[:k]]
