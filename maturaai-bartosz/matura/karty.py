"""Bank kart faktów i słownik pojęć do wypracowań (wiedza offline, poza limitem 8 GB jak baza RAG).

Budowa (nauczyciel = model frontier, TYLKO w fazie budowy): uv run python -m matura.karty --buduj [--sedzia claude:fable]
1. Zakres = działy podstawy programowej wyciągnięte z Informatora i zasad oceniania CKE (np. „IV. Społeczeństwo, życie
   polityczne i kultura starożytnego Rzymu") → data/karty/dzialy.json.
2. Dla każdego działu: 10 fragmentów Wikipedii (BM25) → nauczyciel wypisuje karty {fakt, data, postac, termin, aspekt,
   tytul} i pojęcia {termin, definicja}.
3. Walidacja (lekcja B12): karta zostaje, gdy jej data (jeśli jest) i postać/termin występują dosłownie we wskazanym
   fragmencie; reszta odrzucona i policzona w logu.
Wyjście: data/karty/karty.jsonl, data/karty/slownik.jsonl. Wyszukiwanie: BM25 w pamięci (`Karty().szukaj`).

Karty kanoniczne (`--kanon`, 2026-09-26): próbka kart z Wikipedii okazała się w ~70% przypadkowymi ciekawostkami
(BM25 po samej nazwie działu zwraca losowe artykuły). Wersja kanoniczna: nauczyciel dostaje nazwę działu i jego
wymagania z podstawy programowej (Informator CKE) i wypisuje kluczowe fakty maturalne z własnej wiedzy.
Wyjście: data/karty/kanon.jsonl, data/karty/slownik_kanon.jsonl; `Karty()` czyta kanon, gdy istnieje.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdf

from .retrieval import tokeny

ROOT = Path(__file__).resolve().parent.parent
KAT = ROOT / "data" / "karty"
_lock = threading.Lock()
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["karty", "pojecia"], "properties": {
    "karty": {"type": "array", "items": {"type": "object", "additionalProperties": False,
              "required": ["fakt", "data", "postac", "termin", "aspekt", "tytul"],
              "properties": {k: {"type": "string"} for k in ("fakt", "data", "postac", "termin", "aspekt", "tytul")}}},
    "pojecia": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["termin", "definicja"],
                "properties": {"termin": {"type": "string"}, "definicja": {"type": "string"}}}}}}
INSTR = """Przygotowujesz materiał do wypracowań maturalnych z historii (poziom rozszerzony), dział: {dzial}.
Z podanych fragmentów Wikipedii wypisz 25–40 kart faktów i 10–20 pojęć historycznych.
Karta: fakt = jedno zdanie po polsku, konkretne; data = rok lub zakres lat DOKŁADNIE jak we fragmencie (albo pusty);
postac = osoba z fragmentu (albo pusty); termin = pojęcie historyczne (albo pusty); aspekt = jeden z: polityczny,
militarny, ustrojowy, społeczny, gospodarczy, kulturowy, religijny, międzynarodowy; tytul = tytuł fragmentu źródłowego.
Tylko fakty wprost zawarte we fragmentach. Pojęcie: definicja w jednym zdaniu. Nie uruchamiaj poleceń. Tylko JSON.

FRAGMENTY:
"""


def dzialy() -> list[str]:
    tekst = ""
    for p in sorted((ROOT / "data" / "zasady" / "zrodla").glob("*.pdf")):
        tekst += "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(str(p)).pages)
    wyn = {}
    for m in re.finditer(r"\b([IVXL]{1,7})\.\s+([A-ZŁŚŻŹĆ][^.\n]{6,140}?(?:\n[^.\n]{0,100})?)\.\s*Zdający", tekst):
        nazwa = " ".join(m.group(2).split())
        if "Chronologia" in nazwa or "Analiza" in nazwa or "Tworzenie narracji" in nazwa:
            continue  # wymagania ogólne, nie działy treści
        wyn.setdefault(nazwa, m.group(1))
    return sorted(wyn)


def wymagania() -> dict[str, str]:
    """Dział podstawy programowej → treść jego wymagań („Zdający: …") z Informatora i zasad CKE (pierwsze wystąpienie)."""
    tekst = ""
    for p in sorted((ROOT / "data" / "zasady" / "zrodla").glob("*.pdf")):
        tekst += "\n".join(pg.extract_text() or "" for pg in pypdf.PdfReader(str(p)).pages)
    naglowki = list(re.finditer(r"\b([IVXL]{1,7})\.\s+([A-ZŁŚŻŹĆ][^.\n]{6,140}?(?:\n[^.\n]{0,100})?)\.\s*Zdający", tekst))
    wyn = {}
    for i, m in enumerate(naglowki):
        nazwa = " ".join(m.group(2).split())
        if nazwa in wyn or "Chronologia" in nazwa or "Analiza" in nazwa or "Tworzenie narracji" in nazwa:
            continue
        koniec = naglowki[i + 1].start() if i + 1 < len(naglowki) else m.end() + 1500
        wyn[nazwa] = " ".join(tekst[m.end():min(koniec, m.end() + 1500)].split())
    return wyn


INSTR_KANON = """Przygotowujesz bank faktów do wypracowań i zadań maturalnych z historii (poziom rozszerzony, formuła 2023).
Dział podstawy programowej: {dzial}
Wymagania szczegółowe (zdający …): {wym}

Wypisz 45–60 kart faktów i 15–25 pojęć, które najczęściej są potrzebne na maturze z tego działu. Karta: fakt = jedno
zdanie po polsku z konkretem (wydarzenie, przyczyna, skutek, postać, dokument); data = rok albo zakres lat (pusty, gdy
nie dotyczy); postac = kluczowa osoba (albo pusty); termin = pojęcie historyczne (albo pusty); aspekt = jeden z:
polityczny, militarny, ustrojowy, społeczny, gospodarczy, kulturowy, religijny, międzynarodowy; tytul = krótka nazwa
tematu (np. „Unia lubelska”). Pokryj wszystkie wymagania działu i różne aspekty. Tylko fakty pewne i zgodne z
podręcznikową wiedzą historyczną; daty wyłącznie, jeśli jesteś ich pewien. Pojęcie: definicja w jednym zdaniu.
Nie uruchamiaj poleceń. Zwróć wyłącznie JSON zgodny ze schematem."""


def buduj_kanon(nauczyciel: str, rown: int) -> None:
    from .sedzia import llm_json
    KAT.mkdir(parents=True, exist_ok=True)
    wym = wymagania()
    wy, sl = KAT / "kanon.jsonl", KAT / "slownik_kanon.jsonl"
    gotowe = {json.loads(l)["dzial"] for l in open(wy, encoding="utf-8") if l.strip()} if wy.exists() else set()
    print(f"działów: {len(wym)}, gotowych: {len(gotowe)}", flush=True)

    def jeden(d):
        if d in gotowe:
            return
        for proba in range(3):
            try:
                w = llm_json(INSTR_KANON.format(dzial=d, wym=wym[d] or "(brak opisu)"), SCHEMA, nauczyciel)
                break
            except Exception as e:
                print(f"{d[:50]}: błąd (próba {proba + 1}) {str(e)[:150]}", flush=True)
        else:
            return
        karty = [dict(k, dzial=d) for k in w["karty"] if k["fakt"].strip()]
        with _lock:
            with open(wy, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in karty)
            with open(sl, "a", encoding="utf-8") as f:
                f.writelines(json.dumps(dict(p, dzial=d), ensure_ascii=False) + "\n" for p in w["pojecia"])
        print(f"{d[:60]}: kart {len(karty)}, pojęć {len(w['pojecia'])}", flush=True)

    with ThreadPoolExecutor(rown) as ex:
        list(ex.map(jeden, list(wym)))


def waliduj(karta: dict, teksty: dict[str, str]) -> bool:
    src = teksty.get(karta["tytul"], "")
    if not src:
        return False
    if karta["data"] and not all(r in src for r in re.findall(r"\d{3,4}", karta["data"])):
        return False
    kotwica = karta["postac"] or karta["termin"]
    return bool(kotwica) and kotwica.split()[-1][:6].lower() in src.lower()


def buduj(sedzia: str, rown: int) -> None:
    from .retrieval import Wikipedia
    from .sedzia import llm_json
    KAT.mkdir(parents=True, exist_ok=True)
    lista = dzialy()
    (KAT / "dzialy.json").write_text(json.dumps(lista, ensure_ascii=False, indent=1), encoding="utf-8")
    gotowe = {json.loads(l)["dzial"] for l in open(KAT / "karty.jsonl", encoding="utf-8")} if (KAT / "karty.jsonl").exists() else set()
    wiki = Wikipedia()
    print(f"działów: {len(lista)}, gotowych: {len(gotowe)}", flush=True)

    def jeden(d):
        if d in gotowe:
            return
        frs = wiki.szukaj(d, 10)
        teksty: dict[str, str] = {}
        for f in frs:
            teksty[f["tytul"]] = teksty.get(f["tytul"], "") + " " + f["tekst"]
        try:
            w = llm_json(INSTR.format(dzial=d) + "\n\n".join(f"[{f['tytul']}] {f['tekst']}" for f in frs), SCHEMA, sedzia)
        except Exception as e:
            print(f"{d[:50]}: błąd {str(e)[:150]}", flush=True); return
        ok = [dict(k, dzial=d) for k in w["karty"] if waliduj(k, teksty)]
        with _lock:
            with open(KAT / "karty.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(k, ensure_ascii=False) + "\n" for k in ok)
            with open(KAT / "slownik.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(dict(p, dzial=d), ensure_ascii=False) + "\n" for p in w["pojecia"])
        print(f"{d[:60]}: kart {len(ok)}/{len(w['karty'])} po walidacji, pojęć {len(w['pojecia'])}", flush=True)

    with ThreadPoolExecutor(rown) as ex:
        list(ex.map(jeden, lista))


class Karty:
    def __init__(self, plik: str | None = None):
        """plik: nazwa w data/karty/; domyślnie kanon.jsonl, gdy istnieje, inaczej karty.jsonl (z Wikipedii)."""
        import bm25s
        self.k = []
        plik = plik or ("kanon.jsonl" if (KAT / "kanon.jsonl").exists() else "karty.jsonl")
        for l in open(KAT / plik, encoding="utf-8"):
            try:
                self.k.append(json.loads(l))
            except json.JSONDecodeError:  # linia w trakcie dopisywania przez równoległą budowę kart
                continue
        self.epoka_dzialu = self._epoki_dzialow()
        self.bm = bm25s.BM25()
        self.bm.index([tokeny(f"{c['dzial']} {c['fakt']} {c['postac']} {c['termin']} {c['aspekt']}") for c in self.k],
                      show_progress=False)

    def _epoki_dzialow(self) -> dict[str, str]:
        """Dział → epoka z mediany lat jego kart (nazwa działu bywa myląca, np. „Europa w okresie krucjat”)."""
        from .router import epoka_roku
        lata: dict[str, list[int]] = {}
        for c in self.k:
            m = re.search(r"\d{3,4}", c.get("data") or "")
            if m and "p.n.e" not in (c.get("data") or ""):
                lata.setdefault(c["dzial"], []).append(int(m.group()))
            elif "p.n.e" in (c.get("data") or ""):
                lata.setdefault(c["dzial"], []).append(0)
        return {d: epoka_roku(sorted(v)[len(v) // 2]) for d, v in lata.items() if v}

    def szukaj(self, zapytanie: str, n: int = 8, epoka: str | None = None) -> list[dict]:
        """BM25 po kartach; z `epoka` tylko karty z działów tej epoki (gdy jest ich co najmniej n, inaczej bez filtra)."""
        tk = tokeny(zapytanie)
        if not tk or not self.k:
            return []
        k = min(n * 6 if epoka else n, len(self.k))
        idx, _ = self.bm.retrieve([tk], k=k, show_progress=False)
        wyn = [self.k[int(i)] for i in idx[0]]
        if epoka:
            zgodne = [c for c in wyn if self.epoka_dzialu.get(c["dzial"]) == epoka]
            if len(zgodne) >= n:
                return zgodne[:n]
        return wyn[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buduj", action="store_true"); ap.add_argument("--dzialy", action="store_true")
    ap.add_argument("--kanon", action="store_true", help="karty kanoniczne z wymagań podstawy programowej")
    ap.add_argument("--sedzia", default="claude:fable"); ap.add_argument("--rownolegle", type=int, default=3)
    a = ap.parse_args()
    if a.dzialy:
        for d in dzialy():
            print(d)
    if a.buduj:
        buduj(a.sedzia, a.rownolegle)
    if a.kanon:
        buduj_kanon(a.sedzia, a.rownolegle)
    sys.exit(0)
