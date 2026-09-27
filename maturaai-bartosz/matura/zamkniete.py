"""Zadania zamknięte: ścisły format odpowiedzi i głosowanie (krok Z1 planu docs/plans/2026-09-27-plan-ostatnie-10h.md).

Co to jest: forma odpowiedzi z `answer_format` (paczka egzaminu) albo ze struktury klucza (dev; tylko etykiety i typ,
nigdy wartości): jedna litera, pary etykieta → wartość (P/F, litera, cyfra) albo kolejność liter. Prompt opisuje format
słowami (bez przykładowej odpowiedzi, bo małe modele kopiują przykłady), model krótko uzasadnia i kończy linią
„ODPOWIEDŹ: …”; 3 próbki (t = 0; 0,7; 0,7), na każdej pozycji wygrywa większość (remis: pierwsza próbka). Wynik w
postaci czytanej przez `klucz.ocen` i `egzamin.normalizuj`.
Po co: na próbnym 2023 zestaw 1,95 GB miał 1/11 pkt z zadań zamkniętych (dev 8/14); ścisły format i głosowanie mają w
harnessie inne zespoły (romanszlazakai, SlayerLab).
Co zrobić: konfiguracja `goly_vlm_kb_z` w matura/harness.py; pomiar `scripts/zamkniete_pomiar.py` (automat klucza).
Uwaga: zmienna MATURA_TEMPERATURA nadpisuje temperaturę w `llm.czat` i wyłącza sens głosowania.
"""
from __future__ import annotations

import re
import time
from collections import Counter

SYS_Z = ("Jesteś uczniem zdającym maturę z historii. Rozwiązujesz zadanie zamknięte. Najpierw krótko, najwyżej trzema "
         "zdaniami, uzasadnij wybór na podstawie źródeł i swojej wiedzy. W ostatniej linii podaj samą odpowiedź.")
TEMPERATURY = (0.0, 0.7, 0.7)
_PF = {"p": "P", "prawda": "P", "t": "P", "true": "P", "tak": "P", "f": "F", "fałsz": "F", "falsz": "F", "false": "F",
       "nie": "F"}


def forma(z: dict) -> dict | None:
    """Forma odpowiedzi zadania zamkniętego albo None (nieobsługiwana: zwykła ścieżka harnessu)."""
    fmt = (z.get("answer_format") or "").strip()
    if fmt:
        if re.fullmatch(r"[A-F](?:\s*,\s*[A-F])+", fmt):
            return {"rodzaj": "kolejnosc", "litery": sorted(set(re.findall(r"[A-F]", fmt)))}
        pary = [l.split(":", 1) for l in fmt.splitlines() if ":" in l]
        if not pary:
            return {"rodzaj": "jedna"} if re.fullmatch(r"[A-F]", fmt) else None
        probka = pary[0][1].strip()
        typ = "pf" if probka in ("P", "F") else "cyfra" if probka.isdigit() else "litera" if re.fullmatch(r"[A-F]", probka) else None
        return {"rodzaj": "pary", "typ": typ, "etykiety": [e.strip() for e, _ in pary]} if typ else None
    if z.get("typ") != "zamkniete":
        return None
    from .klucz import parsuj_klucz
    k = parsuj_klucz(z)
    if not k:
        return None
    if k["rodzaj"] == "jedna":
        return {"rodzaj": "jedna"}
    if k["rodzaj"] == "kolejnosc":
        return {"rodzaj": "kolejnosc", "litery": sorted(k["ciag"])}
    if k["rodzaj"] == "pary" and k["typ"] in ("pf", "litera", "cyfra"):
        return {"rodzaj": "pary", "typ": k["typ"], "etykiety": list(k["pary"])}
    return None


def instrukcja(f: dict) -> str:
    if f["rodzaj"] == "jedna":
        return "W ostatniej linii napisz „ODPOWIEDŹ:” i jedną wielką literę wybranej odpowiedzi."
    if f["rodzaj"] == "kolejnosc":
        return ("W ostatniej linii napisz „ODPOWIEDŹ:” i litery " + ", ".join(f["litery"]) + " ułożone we właściwej "
                "kolejności, oddzielone przecinkami.")
    et = ", ".join(f["etykiety"])
    if f["typ"] == "pf":
        return (f"W ostatniej linii napisz „ODPOWIEDŹ:” i dla zdań {et} kolejno literę P (prawda) albo F (fałsz), "
                "oddzielone przecinkami.")
    co = "literę" if f["typ"] == "litera" else "numer"
    return (f"W ostatniej linii napisz „ODPOWIEDŹ:” i dla każdej pozycji {et} wybraną {co} w postaci pozycja=wybór, "
            "oddzielone przecinkami.")


def _linia(tekst: str) -> str:
    linie = [l for l in (tekst or "").replace("**", "").splitlines() if l.strip()]
    for l in reversed(linie):
        if re.search(r"odpowied[źz]|answer", l, re.I):
            return re.split(r"odpowied[źz]\w*\s*:?|answer\s*:?", l, maxsplit=1, flags=re.I)[-1]
    return linie[-1] if linie else ""


def odczytaj(tekst: str, f: dict) -> list[str] | None:
    """Wartości z ostatniej linii „ODPOWIEDŹ: …” w kolejności etykiet albo None."""
    s = _linia(tekst)
    if f["rodzaj"] == "jedna":
        m = re.findall(r"\b([A-F])\b", s)
        return [m[0]] if m else None
    if f["rodzaj"] == "kolejnosc":
        m = [x for x in re.findall(r"\b([A-F])\b", s) if x in f["litery"]]
        return m if sorted(m) == f["litery"] else None
    n = len(f["etykiety"])
    if f["typ"] == "pf":
        m = [_PF[w.lower()] for w in re.findall(r"\b(prawda|fałsz|falsz|true|false|tak|nie|P|F|T)\b", s, re.I)]
        return m if len(m) == n else None
    klasa = r"[A-F]" if f["typ"] == "litera" else r"\d+"
    wart = {}
    for e in f["etykiety"]:
        m = re.search(rf"(?<![\w]){re.escape(e)}\s*[=:\-–)]\s*({klasa})\b", s)
        if m:
            wart[e] = m.group(1)
    if len(wart) == n:
        return [wart[e] for e in f["etykiety"]]
    ciag = re.findall(rf"(?<![\w])({klasa})(?![\w])", re.sub(rf"(?<![\w])(?:{'|'.join(map(re.escape, f['etykiety']))})\s*[=:\-–)]", " ", s))
    return ciag if len(ciag) == n else None


def sformatuj(f: dict, wartosci: list[str]) -> str:
    if f["rodzaj"] == "jedna":
        return wartosci[0]
    if f["rodzaj"] == "kolejnosc":
        return ", ".join(wartosci)
    return "\n".join(f"{e}: {v}" for e, v in zip(f["etykiety"], wartosci))


def odpowiedz(f: dict, url: str, user: str, *, bez_myslenia: bool = True, czat_fn=None,
              temperatury=TEMPERATURY) -> tuple[str, float, dict]:
    """Próbki → głosowanie na każdej pozycji → (odpowiedź w formacie, sekundy, meta). Bez żadnego odczytu: tekst
    pierwszej próbki (egzamin.normalizuj i sędzia dostają wtedy surowy tekst)."""
    if czat_fn is None:
        from .llm import czat as czat_fn
    t0 = time.time()
    probki, odczyty = [], []
    for temp in temperatury:
        t, _ = czat_fn(url, SYS_Z, user + "\n\n" + instrukcja(f), None, max_tokens=300, temperature=temp,
                       bez_myslenia=bez_myslenia)
        probki.append(t)
        o = odczytaj(t, f)
        if o is not None:
            odczyty.append(o)
    sek = time.time() - t0
    if not odczyty:
        return probki[0], sek, {"glosy": 0, "probki": probki}
    wynik = []
    for i in range(len(odczyty[0])):
        c = Counter(o[i] for o in odczyty)
        naj = max(c.values())
        wynik.append(next(o[i] for o in odczyty if c[o[i]] == naj))
    return sformatuj(f, wynik), sek, {"glosy": len(odczyty), "probki": probki}
