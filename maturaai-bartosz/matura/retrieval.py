"""BM25 po fragmentach Wikipedii PL (bez modelu neuronowego → 0 MB w limicie rozmiaru, D9).

Budowa: `uv run python -m matura.retrieval --buduj` → data/wiki/bm25/ (indeks) + offsety.npy.
Tokenizacja: małe litery, słowa, stop-lista, „stem" = pierwsze 6 znaków (tani trik na fleksję PL).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "data" / "wiki" / "fragmenty.jsonl"
IDX = ROOT / "data" / "wiki" / "bm25"
RE_SLOWO = re.compile(r"[0-9a-ząćęłńóśźż]+")
STOP = set("i w z na do się że to jest nie o od po jak za a co by ku przez oraz lub czy ten ta te tego tej tym są był była było były jego jej ich go mu im który która które którego której których przy dla już także też tylko jako może można został została zostały ze we".split())


def tokeny(t: str) -> list[str]:
    return [w[:6] for w in RE_SLOWO.findall(t.lower()) if w not in STOP and len(w) > 1]


class Wikipedia:
    def __init__(self, katalog: Path = IDX):
        import bm25s
        self.bm = bm25s.BM25.load(str(katalog))
        self.off = np.load(katalog / "offsety.npy")
        self.f = open(FRAG, "rb")

    def fragment(self, i: int) -> dict:
        self.f.seek(int(self.off[i]))
        return json.loads(self.f.readline())

    def szukaj(self, zapytanie: str, k: int = 4) -> list[dict]:
        tk = tokeny(zapytanie)
        if not tk:
            return []
        wyn, oc = self.bm.retrieve([tk], k=k, show_progress=False)
        return [dict(self.fragment(int(i)), score=float(s)) for i, s in zip(wyn[0], oc[0])]


def buduj(limit: int | None = None) -> None:
    import bm25s
    t0, off, korpus, pos = time.time(), [], [], 0
    with open(FRAG, "rb") as f:
        for n, linia in enumerate(f):
            if limit and n >= limit:
                break
            off.append(pos); pos += len(linia)
            d = json.loads(linia)
            korpus.append(tokeny(d["tytul"] + " " + d["tekst"]))
            if n % 500000 == 0:
                print(f"tokenizacja {n} ({time.time()-t0:.0f}s)", flush=True)
    bm = bm25s.BM25()
    bm.index(korpus, show_progress=False)
    IDX.mkdir(parents=True, exist_ok=True)
    bm.save(str(IDX))
    np.save(IDX / "offsety.npy", np.array(off, dtype=np.int64))
    print(f"indeks: {len(off)} fragmentów, {time.time()-t0:.0f}s → {IDX}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buduj", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--szukaj")
    a = ap.parse_args()
    if a.buduj:
        buduj(a.limit)
    if a.szukaj:
        for r in Wikipedia().szukaj(a.szukaj, 5):
            print(f"[{r['score']:.1f}] {r['tytul']}: {r['tekst'][:200]}")
    sys.exit(0)
