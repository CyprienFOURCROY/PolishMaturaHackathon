"""Baza wiedzy poza limitem rozmiaru (D22): akapity argumentacyjne do eseju (e6) i hasła do zadań krótkich.

Co to jest: pliki JSONL od zespołu (instrukcja: dokument „MaturaAI: knowledge base cards (team guide)”) w
data/wiedza/: `akapity_esej*.jsonl` (teza, podmiot, okres, tryb, pozycja = aspekt albo przykład, kierunek
potwierdza/przeczy, tekst 90-140 słów) i `hasla*.jsonl`. Wyszukiwanie: BM25 w pamięci, ta sama tokenizacja co
Wikipedia (`retrieval.tokeny`), bez modelu neuronowego (0 MB).
Po co: fakty z bazy są poprawne, a małe modele zmyślają; esej e6 wkleja akapit bazy dosłownie.
Co zrobić: pliki zespołu wrzucić do data/wiedza/ (wiele plików naraz jest scalanych); `BazaAkapitow().kandydaci(...)`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .retrieval import tokeny

ROOT = Path(__file__).resolve().parent.parent
KAT = ROOT / "data" / "wiedza"


def _stemy(t: str) -> set[str]:
    return set(tokeny(t))


def _wczytaj(katalog: Path, wzor: str) -> list[dict]:
    wiersze, widziane = [], set()
    for p in sorted(Path(katalog).glob(wzor)):
        for l in p.open(encoding="utf-8"):
            if l.strip():
                w = json.loads(l)
                if w.get("id") not in widziane:
                    widziane.add(w.get("id")); wiersze.append(w)
    return wiersze


class BazaAkapitow:
    """Akapity eseju z plików `akapity_esej*.jsonl`; pusta baza, gdy plików brak (e6 przechodzi wtedy na e5)."""

    def __init__(self, katalog: Path = KAT):
        self.a = _wczytaj(katalog, "akapity_esej*.jsonl")
        self.bm = None
        if self.a:
            import bm25s
            self.bm = bm25s.BM25()
            self.bm.index([tokeny(f"{w['teza']} {w['podmiot']} {w['pozycja']} {w['tekst']}") for w in self.a],
                          show_progress=False)

    def __len__(self) -> int:
        return len(self.a)

    def kandydaci(self, teza: str, pozycja: str, *, kierunek: str | None = None, n: int = 5, tryb: str | None = None,
                  pomin: set[str] | frozenset = frozenset()) -> list[dict]:
        """Najlepsze akapity dla tezy i pozycji: najpierw zgodna pozycja (wspólny rdzeń słowa) i kierunek, potem reszta."""
        if not self.a:
            return []
        tk = tokeny(f"{teza} {pozycja}")
        if not tk:
            return []
        wyn, oc = self.bm.retrieve([tk], k=min(len(self.a), 60), show_progress=False)
        sp = _stemy(pozycja)
        lista = []
        for i, s in zip(wyn[0], oc[0]):
            w = self.a[int(i)]
            if w["id"] in pomin or (tryb and w.get("tryb") != tryb):
                continue
            zgodna = bool(sp & _stemy(w["pozycja"]))
            lista.append((not zgodna, kierunek is not None and w["kierunek"] != kierunek, -float(s), w))
        lista.sort(key=lambda x: x[:3])
        return [dict(w, score=-s) for _, _, s, w in lista[:n]]

    def przyklady(self, teza: str, wybor: str, *, kierunek: str | None = None, n: int = 3) -> list[str]:
        """Temat „trzech wybranych …”: nazwy przykładów z bazy (tryb elementy), najlepiej pasujące do tezy."""
        pozycje: list[str] = []
        for c in self.kandydaci(teza, wybor, kierunek=kierunek, n=40, tryb="elementy"):
            if c["pozycja"] not in pozycje:
                pozycje.append(c["pozycja"])
        return pozycje[:n]


def _tekst_hasla(h: dict) -> str:
    """Hasło jako tekst dla promptu: tytuł, fakty, dokumenty i znane ilustracje (identyfikacja źródeł)."""
    czesci = [f"[{h['tytul']}] {h['tekst']}"]
    dok = [f"{d['nazwa']} ({d['data']}): {d['tresc']}" for d in h.get("dokumenty", [])]
    ilu = [f"{i['nazwa']}, {i['autor']}, {i['data']}: {i['co_przedstawia']}" for i in h.get("ikonografia", [])]
    if dok:
        czesci.append("Dokumenty: " + "; ".join(dok))
    if ilu:
        czesci.append("Znane ilustracje: " + "; ".join(ilu))
    return " ".join(czesci)


class BazaHasel:
    """Hasła do zadań krótkich z plików `hasla*.jsonl` (Task 1 dokumentu dla zespołu); BM25 w pamięci."""

    def __init__(self, katalog: Path = KAT):
        self.h = _wczytaj(katalog, "hasla*.jsonl")
        self.bm = None
        if self.h:
            import bm25s
            self.bm = bm25s.BM25()
            self.bm.index([tokeny(_tekst_hasla(h) + " " + " ".join(h.get("postacie", []))
                                  + " " + " ".join(p["termin"] for p in h.get("pojecia", []))) for h in self.h],
                          show_progress=False)

    def __len__(self) -> int:
        return len(self.h)

    def szukaj(self, zapytanie: str, n: int = 2) -> list[dict]:
        tk = tokeny(zapytanie)
        if not self.h or not tk:
            return []
        wyn, oc = self.bm.retrieve([tk], k=min(n, len(self.h)), show_progress=False)
        return [dict(self.h[int(i)], score=float(s)) for i, s in zip(wyn[0], oc[0])]


def zapytanie(z: dict, blok_ilustracji: str = "") -> str:
    """Zapytanie do bazy dla zadania krótkiego: polecenie, podpisy źródeł („Źródło N. …”, pierwsze linie), początek
    tekstu źródeł i opis ilustracji; do 1500 znaków."""
    zr = z.get("zrodla_tekst", "")
    linie = [l for l in zr.splitlines() if l.strip()]
    podpisy = [l for l in linie if l.lstrip().startswith(("Źródło", "Fragment", "Mapa", "Ilustracja", "Plakat",
                                                           "Karykatura", "Fotografia", "Na podstawie"))]
    return " ".join([z["polecenie"], " ".join(podpisy), zr[:300], blok_ilustracji[:500]])[:1500]
