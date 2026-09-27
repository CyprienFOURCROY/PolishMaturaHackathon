"""Wyszukiwanie materiału do eseju (B9, sesja 27.09): kawałki haseł i kart, zapytania per aspekt, BM25 + reranker.

Co to jest: nowe tryby materiału dla ucznia eseju (`esej_sft.material_v2`), obok dotychczasowego `esej_sft.material`
(3 hasła po zapytaniu, każde przycięte do 1500 znaków), który zostaje domyślny (uczniowie kb/kb3 i e10 muszą dać się
odtworzyć). Lejek z 92 faktów wzorcowych dev: 68% jest w jakimś haśle, 33% w 3 hasłach po tezie, 26% widzi model po
przycięciu; stąd trzy naprawy:
- sito 3 (przycięcie): materiał składany z całych kawałków pod budżetem znaków, nigdy nie tnie w środku zdania;
- sito 2 (wyszukiwanie): osobne zapytania dla tezy i dla każdego aspektu albo elementu tematu, rankingi łączone RRF;
- kawałki: hasła pocięte po zdaniach na 300-500 znaków (z tytułem hasła) + karty kanoniczne (data/karty/kanon.jsonl);
- hybryda: BM25 na kawałkach → top-50 na zapytanie → reranker BAAI/bge-reranker-v2-m3 (transformers, GPU, fp16).
Po co: więcej właściwych faktów w materiale ucznia (materiał to dźwignia: S2 0,67 → 4,25/15, B4 runda kb).
Co zrobić: `esej_sft.material_v2(temat, budzet=4500, wariant="bm25_rerank")`; pomiar pokrycia:
review/wyszukiwanie-2026-09-27/pomiar.py (B10).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import esej
from .retrieval import tokeny

ROOT = Path(__file__).resolve().parent.parent
KARTY = ROOT / "data" / "karty" / "kanon.jsonl"
RERANKER = "BAAI/bge-reranker-v2-m3"
WARIANTY = ("hasla_pelne", "bm25_kawalki", "bm25_rerank", "rerank_wszystko", "hasla_plus_rerank")
MIN_ZN, MAX_ZN = 300, 500
RRF_K = 60
RE_ZDANIE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„\"(])")


# ---------------------------------------------------------------- kawałki

def potnij(tytul: str, tekst: str, min_zn: int = MIN_ZN, max_zn: int = MAX_ZN) -> list[str]:
    """Tekst hasła → kawałki „[tytuł] zdania…” o długości treści około min_zn-max_zn znaków, cięte tylko na granicy
    zdań (zdanie dłuższe niż max_zn zostaje całe); ostatni kawałek krótszy niż min_zn dołączany do poprzedniego."""
    zdania = [z.strip() for z in RE_ZDANIE.split(" ".join(tekst.split())) if z.strip()]
    kaw: list[str] = []
    biez = ""
    for z in zdania:
        if biez and len(biez) + 1 + len(z) > max_zn:
            kaw.append(biez)
            biez = z
        else:
            biez = f"{biez} {z}".strip()
    if biez:
        if kaw and len(biez) < min_zn:
            kaw[-1] = f"{kaw[-1]} {biez}"
        else:
            kaw.append(biez)
    return [f"[{tytul}] {k}" for k in kaw]


def _tresc_hasla(h: dict) -> str:
    """Tekst hasła bez tytułu na początku (tytuł dokleja `potnij` do każdego kawałka)."""
    from .wiedza import _tekst_hasla
    t = _tekst_hasla(h)
    pref = f"[{h['tytul']}] "
    return t[len(pref):] if t.startswith(pref) else t


def kawalki_hasel(hasla: list[dict]) -> list[dict]:
    return [{"tekst": k, "zrodlo": "haslo", "id": h["id"], "tytul": h["tytul"]}
            for h in hasla for k in potnij(h["tytul"], _tresc_hasla(h))]


def kawalki_kart(karty: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(karty):
        kto = f"{c['postac']}: " if c.get("postac") else ""
        data = f" ({c['data']})" if c.get("data") else ""
        out.append({"tekst": f"[{c.get('tytul') or c.get('dzial')}] {kto}{c['fakt']}{data}", "zrodlo": "karta",
                    "id": f"karta-{i}", "tytul": c.get("tytul") or ""})
    return out


# ---------------------------------------------------------------- zapytania i łączenie

def zapytania(temat: str) -> list[str]:
    """Teza osobno + teza z każdym aspektem (albo z rodzajem elementów przy temacie z wyborem trzech przykładów)."""
    teza, aspekty, wybor = esej.rozbierz(temat)
    if wybor:
        return [teza, f"{teza} {wybor}"]
    return [teza] + [f"{teza} aspekt {a}" for a in aspekty]


def rrf(rankingi: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: suma 1/(k + pozycja) po rankingach; remis → wcześniejsze pierwsze wystąpienie."""
    wynik: dict[int, float] = {}
    for r in rankingi:
        for poz, i in enumerate(r):
            wynik[i] = wynik.get(i, 0.0) + 1.0 / (k + poz + 1)
    return sorted(wynik, key=lambda i: -wynik[i])


def spakuj(teksty: list[str], budzet: int, sep: str = "\n") -> str:
    """Całe teksty w kolejności rankingu, dopóki mieszczą się w budżecie znaków (za duży tekst pomijany, szukamy
    dalej); nic nie jest przycinane. Kolejne teksty poprzedzone „- ” jak w `esej_sft.material`."""
    wybrane, dl = [], 0
    for t in teksty:
        wpis = f"- {t}"
        dodatek = len(wpis) + (len(sep) if wybrane else 0)
        if dl + dodatek <= budzet:
            wybrane.append(wpis)
            dl += dodatek
    return sep.join(wybrane)


# ---------------------------------------------------------------- indeks i reranker

class Reranker:
    """Cross-encoder BAAI/bge-reranker-v2-m3 (fp16 na GPU, tryb offline z ~/.cache/huggingface)."""

    def __init__(self, nazwa: str = RERANKER, urzadzenie: str = "cuda", partia: int = 64):
        import os
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.torch, self.urz, self.partia = torch, urzadzenie, partia
        self.tok = AutoTokenizer.from_pretrained(nazwa)
        self.m = AutoModelForSequenceClassification.from_pretrained(nazwa, torch_dtype=torch.float16).to(urzadzenie).eval()

    def oceny(self, zapytanie: str, teksty: list[str]) -> list[float]:
        wyn: list[float] = []
        with self.torch.no_grad():
            for i in range(0, len(teksty), self.partia):
                cz = teksty[i:i + self.partia]
                b = self.tok([zapytanie] * len(cz), cz, padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(self.urz)
                wyn += self.m(**b).logits.view(-1).float().tolist()
        return wyn


class Indeks:
    """Kawałki haseł (+ karty) z indeksem BM25 (tokeny jak w całym projekcie: `retrieval.tokeny`)."""

    def __init__(self, hasla: list[dict] | None = None, karty: list[dict] | None = None, z_kartami: bool = True,
                 reranker=None):
        import bm25s
        if hasla is None:
            from .wiedza import BazaHasel
            hasla = BazaHasel().h
        if karty is None and z_kartami:
            karty = [json.loads(l) for l in open(KARTY, encoding="utf-8") if l.strip()]
        self.hasla = hasla
        self.kaw = kawalki_hasel(hasla) + (kawalki_kart(karty) if z_kartami and karty else [])
        self.bm = bm25s.BM25()
        self.bm.index([tokeny(k["tekst"]) for k in self.kaw], show_progress=False)
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def bm25(self, zapytanie: str, k: int = 50) -> list[int]:
        tk = tokeny(zapytanie)
        if not tk:
            return []
        idx, _ = self.bm.retrieve([tk], k=min(k, len(self.kaw)), show_progress=False)
        return [int(i) for i in idx[0]]

    def rerank(self, zapytanie: str, kandydaci: list[int]) -> list[int]:
        oc = self.reranker.oceny(zapytanie, [self.kaw[i]["tekst"] for i in kandydaci])
        return [i for _, i in sorted(zip(oc, kandydaci), key=lambda x: -x[0])]

    def ranking(self, temat: str, wariant: str, k: int = 50) -> list[int]:
        """Kolejność kawałków dla tematu: bm25_kawalki (RRF z BM25 per zapytanie), bm25_rerank (top-k BM25 z każdego
        zapytania, reranker per zapytanie, RRF), rerank_wszystko (reranker na wszystkich kawałkach, RRF)."""
        qs = zapytania(temat)
        if wariant == "bm25_kawalki":
            return rrf([self.bm25(q, 2 * k) for q in qs])
        if wariant == "bm25_rerank":
            pula = list(dict.fromkeys(i for q in qs for i in self.bm25(q, k)))
            return rrf([self.rerank(q, pula) for q in qs])
        if wariant == "rerank_wszystko":
            wszystkie = list(range(len(self.kaw)))
            return rrf([self.rerank(q, wszystkie) for q in qs])
        raise ValueError(f"wariant rankingu: bm25_kawalki | bm25_rerank | rerank_wszystko, jest {wariant!r}")

    def hasla_pelne(self, temat: str, budzet: int) -> str:
        """Wariant (1): hasła po zapytaniu z tezy w kolejności BM25 haseł, każde całe (jego kawałki po kolei), aż do
        budżetu; kawałek, który się nie mieści, kończy hasło (bez cięcia w środku zdania)."""
        from .esej_sft import zapytanie_teza
        from .wiedza import BazaHasel
        baza = BazaHasel() if not hasattr(self, "_baza") else self._baza
        self._baza = baza
        teksty = []
        for h in baza.szukaj(zapytanie_teza(temat), 10):
            teksty += potnij(h["tytul"], _tresc_hasla(h))
        return spakuj(teksty, budzet)

    def hasla_plus_rerank(self, temat: str, budzet: int, n_hasel: int = 3, znaki_hasla: int = 1500) -> str:
        """Najlepszy wariant pomiaru B10 (review/wyszukiwanie-2026-09-27): n haseł po zapytaniu z tezy, każde przycięte
        do znaki_hasla znaków (jak dziś), a resztę budżetu wypełniają kawałki z rankingu bm25_rerank spoza tych haseł."""
        from .esej_sft import zapytanie_teza
        from .wiedza import BazaHasel, _tekst_hasla
        if not hasattr(self, "_baza"):
            self._baza = BazaHasel()
        hasla = self._baza.szukaj(zapytanie_teza(temat), n_hasel)
        ids = {h["id"] for h in hasla}
        reszta = [self.kaw[i]["tekst"] for i in self.ranking(temat, "bm25_rerank") if self.kaw[i]["id"] not in ids]
        return spakuj([_tekst_hasla(h)[:znaki_hasla] for h in hasla] + reszta, budzet)

    def material(self, temat: str, budzet: int = 4500, wariant: str = "bm25_rerank", k: int = 50) -> str:
        if wariant not in WARIANTY:
            raise ValueError(f"wariant materiału: {WARIANTY}, jest {wariant!r}")
        if wariant == "hasla_pelne":
            return self.hasla_pelne(temat, budzet)
        if wariant == "hasla_plus_rerank":
            return self.hasla_plus_rerank(temat, budzet)
        return spakuj([self.kaw[i]["tekst"] for i in self.ranking(temat, wariant, k)], budzet)
