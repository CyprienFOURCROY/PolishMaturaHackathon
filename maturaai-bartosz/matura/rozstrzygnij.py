"""Zadania „rozstrzygnij”: głosowanie nad rozstrzygnięciem i krótkie uzasadnienie (plan docs/plans/2026-09-27-plan-ostatnie-10h.md).

Co to jest: 3 próbki odpowiedzi (t = 0; 0,7; 0,7) na zadanie otwarte z poleceniem „Rozstrzygnij…”; z każdej odczytane
rozstrzygnięcie (tak / nie, wskazana litera opcji albo pierwsze słowa linii „Rozstrzygnięcie”), wygrywa większość
(remis: pierwsza próbka), zostaje cała odpowiedź próbki z wygrywającym rozstrzygnięciem. Prompt prosi o krótkie
uzasadnienie (1-2 zdania) bez faktów, których model nie jest pewien.
Po co: na dev zadania „rozstrzygnij” tracą 20 z 45 pkt zadań otwartych, najczęściej przez złe rozstrzygnięcie i dopisane
błędne fakty (uzasadnienia sędziego z `review/oceny_cache.jsonl`); głosowanie w zadaniach zamkniętych dało +4 pkt na
walidacji.
Co zrobić: konfiguracja `goly_vlm_kb_zr` w matura/harness.py (zamknięte jak `goly_vlm_kb_z` + to); pomiar
`scripts/zamkniete_pomiar.py --filtr rozstrzygnij`. MATURA_TEMPERATURA wyłącza sens głosowania.
"""
from __future__ import annotations

import re
import time
from collections import Counter

SYS_R = ("Jesteś uczniem zdającym maturę z historii. Odpowiedz na zadanie po polsku. Najpierw rozstrzygnij, potem uzasadnij "
         "krótko: jednym albo dwoma zdaniami, z odwołaniem do źródła i do wiedzy, której jesteś pewien. Nie dopisuj faktów, "
         "dat ani nazwisk, których nie jesteś pewien.")
INSTR_R = ("Zacznij od linii „Rozstrzygnięcie:” z samym rozstrzygnięciem (tak albo nie, albo wskazana opcja), a potem "
           "napisz linię „Uzasadnienie:” z krótkim uzasadnieniem.")
TEMPERATURY = (0.0, 0.7, 0.7)


def czy_rozstrzygnij(z: dict) -> bool:
    return z.get("typ") != "zamkniete" and not z.get("esej") and bool(re.search(r"\brozstrzygnij", z.get("polecenie", ""), re.I))


def decyzja(tekst: str) -> str | None:
    """Klucz rozstrzygnięcia: „tak” / „nie”, litera opcji albo pierwsze 6 słów linii rozstrzygnięcia (małe litery)."""
    t = (tekst or "").replace("**", "").replace("*", "")
    m = re.search(r"rozstrzygni[ęe]cie\s*:?\s*(.*?)(?:\n\s*\n|\n\s*uzasadnienie|$)", t, re.I | re.S)
    linia = (m.group(1) if m else next((l for l in t.splitlines() if l.strip()), "")).strip()
    if not linia:
        return None
    pierwsze = linia.split("\n")[0]
    w = re.match(r"\s*(tak|nie)\b", pierwsze, re.I)
    if w:
        return w.group(1).lower()
    lit = re.search(r"\b(?:wersj\w*|fragment\w*|źródł\w*|ilustracj\w*|relief\w*|opcj\w*|odpowied\w*)?\s*\b([A-F])\b(?!\.\s*[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż])", pierwsze)
    if lit:
        return lit.group(1)
    slowa = re.findall(r"[\wąćęłńóśźż]+", pierwsze.lower())
    return " ".join(slowa[:6]) or None


def odpowiedz(url: str, user: str, *, bez_myslenia: bool = True, czat_fn=None, max_tokens: int = 400,
              temperatury=TEMPERATURY) -> tuple[str, float, dict]:
    if czat_fn is None:
        from .llm import czat as czat_fn
    t0 = time.time()
    probki = []
    for temp in temperatury:
        t, _ = czat_fn(url, SYS_R, user + "\n\n" + INSTR_R, None, max_tokens=max_tokens, temperature=temp,
                       bez_myslenia=bez_myslenia)
        probki.append(t)
    decyzje = [decyzja(p) for p in probki]
    znane = [d for d in decyzje if d]
    if not znane:
        return probki[0], time.time() - t0, {"decyzje": decyzje}
    c = Counter(znane)
    naj = max(c.values())
    wygrana = next(d for d in decyzje if d and c[d] == naj)
    wybrana = next(p for p, d in zip(probki, decyzje) if d == wygrana)
    return wybrana, time.time() - t0, {"decyzje": decyzje}
