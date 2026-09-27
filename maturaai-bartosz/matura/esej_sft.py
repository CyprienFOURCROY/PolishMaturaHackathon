"""Esej po polsku z modelu po SFT na esejach nauczyciela (eksperyment 27.09).

Co to jest: jeden format wiadomości dla danych treningowych (`przyklad`) i dla generacji na egzaminie (`wiadomosci`),
żeby uczeń widział na egzaminie dokładnie ten prompt, na którym był uczony.
Po co: uczniowie S1 (Bielik-1.5B-v3, pełny SFT) i S2 (Qwen3.5-4B + adapter LoRA) piszą esej po polsku bez tłumacza,
więc kryterium B nie traci na Marianie.
Co zrobić: dane `przyklad(temat, esej, stanowisko)` → JSONL `{"prompt": [...], "completion": [...]}` dla
`matura.trening trenuj --dane ...`; na egzaminie `wiadomosci(temat, "zgadzam")` i `slowa()` do ponowienia < 300 słów.
"""
from __future__ import annotations

import re

SYSTEM = ("Jesteś uczniem, który zdaje maturę rozszerzoną z historii. Piszesz wypracowanie po polsku. "
          "Używasz prostych słów i krótkich zdań. We wstępie zajmujesz stanowisko wobec tezy. "
          "Potem piszesz po jednym akapicie na każdy aspekt z polecenia. W każdym akapicie podajesz konkretne fakty "
          "(daty, nazwy, postacie) i piszesz, co z nich wynika dla tezy. Na końcu piszesz krótkie zakończenie. "
          "Nie używasz nagłówków ani list.")
DLUGOSC = "Twoja wypowiedź powinna liczyć minimum 300 wyrazów."
STANOWISKA = {"zgadzam": "Stanowisko: zgadzam się z tezą.", "nie zgadzam": "Stanowisko: nie zgadzam się z tezą."}
RE_SLOWO = re.compile(r"\w+(?:[-']\w+)*")
# runda z materiałem (sesja 27.09, po B1: model po SFT bez wiedzy zmyśla fakty): hasła bazy wiedzy w prompcie
NAGLOWEK_MATERIALU = "Materiał pomocniczy z bazy wiedzy (może być nietrafny; bierz z niego tylko fakty pasujące do tematu):"
RE_FORMULKA = re.compile(r"Zajmij stanowisko wobec powyższej tezy i je uzasadnij,?|uwzględniając w swojej argumentacji|"
                         + re.escape(DLUGOSC), re.I)
ZNAKI_HASLA = 1500
_BAZA = None


def zapytanie(temat: str) -> str:
    """Zapytanie do bazy haseł z tematu eseju (teza + aspekty, bez formułek polecenia): ta sama funkcja w danych
    treningowych i na egzaminie."""
    return " ".join(RE_FORMULKA.sub(" ", temat).split())


def zapytanie_teza(temat: str) -> str:
    """Zapytanie z samej tezy (tekst przed „Zajmij stanowisko…”): na 202 tematach treningowych materiał dzieli ≥ 3 rdzenie
    z tezą w 183 wobec 176 przy pełnym tekście (dopiski „trzy wybrane decyzje…”, aspekty przestawiały ranking BM25)."""
    return zapytanie(re.split(r"Zajmij stanowisko", temat)[0])


ZAPYTANIA = {"pelny": zapytanie, "teza": zapytanie_teza}


def material(temat: str, n: int = 3, baza=None, tryb: str = "pelny") -> str:
    """n haseł bazy wiedzy (BM25, `matura.wiedza.BazaHasel`) dla tematu, każde przycięte do ZNAKI_HASLA znaków.
    tryb: „pelny” (runda kb) albo „teza” (runda kb2) = funkcja zapytania z ZAPYTANIA."""
    global _BAZA
    from .wiedza import _tekst_hasla
    if baza is None:
        if _BAZA is None:
            from .wiedza import BazaHasel
            _BAZA = BazaHasel()
        baza = _BAZA
    return "\n".join(f"- {_tekst_hasla(h)[:ZNAKI_HASLA]}" for h in baza.szukaj(ZAPYTANIA[tryb](temat), n))


_INDEKS = None


def material_v2(temat: str, budzet: int = 4500, wariant: str = "bm25_rerank", indeks=None) -> str:
    """Materiał z nowego wyszukiwania (B9, `matura/wyszukiwanie_esej.py`): kawałki haseł i kart bez przycinania
    w środku zdania, zapytania per aspekt, opcjonalnie reranker. `material` (3 hasła × 1500 znaków) zostaje domyślny."""
    global _INDEKS
    if indeks is None:
        if _INDEKS is None:
            from .wyszukiwanie_esej import Indeks
            _INDEKS = Indeks()
        indeks = _INDEKS
    return indeks.material(temat, budzet=budzet, wariant=wariant)


def polecenie(temat: str, stanowisko: str | None = None, material: str | None = None) -> str:
    """Treść dla ucznia jak `devset.tresc_dla_modelu` zadania eseju (wymóg długości, pusta linia, temat) + opcjonalne
    stanowisko. Temat, który już zaczyna się od wymogu długości (zadanie z arkusza), nie dostaje go drugi raz."""
    t = temat.strip()
    if not t.startswith(DLUGOSC):
        t = f"{DLUGOSC}\n\n{t}"
    if stanowisko is not None:
        if stanowisko not in STANOWISKA:
            raise ValueError(f"stanowisko: {sorted(STANOWISKA)} albo None, jest {stanowisko!r}")
        t = f"{t}\n\n{STANOWISKA[stanowisko]}"
    if material:
        t = f"{NAGLOWEK_MATERIALU}\n{material}\n\n{t}"
    return t


def wiadomosci(temat: str, stanowisko: str | None = None, material: str | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": polecenie(temat, stanowisko, material)}]


def przyklad(temat: str, esej: str, stanowisko: str | None = None, material: str | None = None) -> dict:
    """Przykład SFT w formacie `data/sft/*.jsonl` (strata tylko na odpowiedzi, `completion_only_loss`)."""
    return {"prompt": wiadomosci(temat, stanowisko, material),
            "completion": [{"role": "assistant", "content": esej.strip()}]}


def slowa(tekst: str) -> int:
    return len(RE_SLOWO.findall(tekst))
