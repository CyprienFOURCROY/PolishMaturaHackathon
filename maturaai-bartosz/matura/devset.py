"""Zbiory zadań w jednym formacie.

Nazwy zbiorów (lista `sesje` w konfiguracji):
- "2023-maj" itd.  → arkusze CKE formuły 2023 (data/cke/json). 2023–2025 = split "dev", 2026 = "test".
- "synt"            → sztuczne arkusze z Wikipedii (data/synt/arkusze.json, matura/syntetyk.py) = split "synt".
- "eseje-cke"       → 24 tematy wypracowań CKE 2023–2026 (każde zadanie esejowe rozcięte na 3 tematy); split dev/test.
- "eseje-synt"      → sztuczne tematy w stylu CKE (data/synt/eseje.json) = split "synt".
- "eseje-kalibracja"→ 8 wypracowań z Informatora CKE z oficjalną oceną (data/zasady/kalibracja_esejow.json).
Zbiory sesji CKE („2023-maj" itd.) NIE zawierają wypracowania — eseje oceniamy osobno (obszar „esej", ~50% nocy).
- Pula treningowa/few-shot: CKE 2020–2022 (`pula_treningowa()`), NIGDY dev/test/synt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "data" / "cke" / "json"
SYNT = ROOT / "data" / "synt" / "arkusze.json"
ZAMKNIETE = re.compile(r"Zaznacz|Oceń prawdziwość|Wybierz|Przyporządkuj|Podkreśl|Uporządkuj|Uzupełnij tabelę", re.I)
SESJE_CKE = ["2023-maj", "2023-czerwiec", "2024-maj", "2024-czerwiec", "2025-maj", "2025-czerwiec", "2026-maj", "2026-czerwiec"]
ESEJE_SYNT = ROOT / "data" / "synt" / "eseje.json"
KALIBRACJA = ROOT / "data" / "zasady" / "kalibracja_esejow.json"
SESJE_TRENINGOWE = ["2020-czerwiec", "2021-maj", "2021-czerwiec", "2022-maj", "2022-czerwiec"]


def typ_zadania(z: dict) -> str:
    if z.get("esej"):
        return "esej"
    if ZAMKNIETE.search(z["polecenie"]) and len(z["rozwiazanie"]) <= 60:
        # klucz z samych oznaczeń (litery A-F, P/F, numery); „Uzupełnij tabelę” nazwami (2025-maj-4) to zadanie otwarte
        tok = re.sub(r"[\d.\s–\-:,;()]+", " ", z["rozwiazanie"]).split()
        if all(re.fullmatch(r"[A-FP]+", t) for t in tok):
            return "zamkniete"
    return "otwarte"


def _esej(zid: str, temat: str, split: str, **extra) -> dict:
    return {"id": zid, "temat": temat, "polecenie": "Twoja wypowiedź powinna liczyć minimum 300 wyrazów.\n\n" + temat,
            "zrodla_tekst": "", "pkt_max": 15, "esej": True, "typ": "esej", "split": split, "obrazy": [],
            "zasady_oceniania": "", "rozwiazanie": "", **extra}


def _eseje_cke() -> list[dict]:
    from .esej import tematy
    out = []
    for s in SESJE_CKE:
        for z in json.loads((JSON / f"historia-{s}.json").read_text(encoding="utf-8")):
            if z["esej"]:
                for k, t in enumerate(tematy(z["polecenie"]), 1):
                    out.append(_esej(f"{z['id']}-t{k}", t, "test" if z["rok"] >= 2026 else "dev", rok=z["rok"]))
    return out


def _cke(sesja: str, z_esejem: bool = False) -> list[dict]:
    out = []
    for z in json.loads((JSON / f"historia-{sesja}.json").read_text(encoding="utf-8")):
        if z["esej"] and not z_esejem:
            continue
        z["typ"] = typ_zadania(z)
        z["split"] = "test" if z["rok"] >= 2026 else ("dev" if z["rok"] >= 2023 else "train")
        z["obrazy"] = [str(ROOT / o) for o in z["obrazy"]]
        out.append(z)
    return out


def wczytaj(sesje: list[str]) -> list[dict]:
    out = []
    for s in sesje:
        if s == "synt":
            if not SYNT.exists():
                continue  # arkusze sztuczne jeszcze nie wygenerowane
            for z in json.loads(SYNT.read_text(encoding="utf-8")):
                z.setdefault("obrazy", []); z.setdefault("esej", False); z.setdefault("zrodla_tekst", "")
                z["typ"] = typ_zadania(z); z["split"] = "synt"
                out.append(z)
        elif s == "eseje-cke":
            out += _eseje_cke()
        elif s == "eseje-synt":
            if ESEJE_SYNT.exists():
                out += [_esej(e["id"], e["temat"], "synt") for e in json.loads(ESEJE_SYNT.read_text(encoding="utf-8"))]
        elif s == "eseje-kalibracja":
            if KALIBRACJA.exists():
                out += [_esej(k["id"], k["temat"], "kalibracja", wzorzec_cke=k["wypracowanie"], pkt_cke=k["pkt"])
                        for k in json.loads(KALIBRACJA.read_text(encoding="utf-8"))]
        else:
            out += _cke(s)
    return out


def pula_treningowa() -> list[dict]:
    """CKE 2020–2022 z rozwiązaniem (few-shot i dane SFT). Bez esejów (stara formuła, inna rubryka)."""
    out = []
    for s in SESJE_TRENINGOWE:
        if (JSON / f"historia-{s}.json").exists():
            out += [z for z in _cke(s, z_esejem=True) if z["rozwiazanie"] and not z["esej"] and (z["pkt_max"] or 0) <= 4]
    return out


RE_PUNKTACJA = re.compile(r"(?:\b\d+(?:\.\d+)*\.\s*)?(?<!\()\b0–1(?:–[23])?\b(?!\))")   # „20.2. 0–1–2” (kratka punktów z arkusza)
RE_SZABLON_PF = re.compile(r"(?:^|\s+)P\s*/?\s*F\s*$")                          # „… zdanie. P F” (kratki do zaznaczenia)
RE_ETYKIETY = re.compile(r"^\s*[A-D](?:\s+[A-D])+\s*$")                     # „A B C D” (podpisy ilustracji)
RE_SAM_NUMER = re.compile(r"^\s*\d+(?:\.\d+)*\.\s*$")                         # „19.1.” (numer kratki bez treści)


def _oczysc(t: str) -> str:
    """Usuwa artefakty arkusza, które małe modele przepisują zamiast odpowiadać (lekcja z nocy 26.09):
    kratki punktacji „0–1–2”, szablon „P F” na końcu zdań, same etykiety „A B C D”. Nie rusza treści zadania."""
    linie = []
    for l in t.splitlines():
        l = RE_SZABLON_PF.sub("", RE_PUNKTACJA.sub("", l)).rstrip()
        if l.strip() and (RE_ETYKIETY.match(l) or RE_SAM_NUMER.match(l)):
            continue
        if l.strip() or (linie and linie[-1].strip()):
            linie.append(l)
    return "\n".join(linie).strip()


def tresc_dla_modelu(z: dict) -> str:
    """Tekst zadania tak, jak widzi go zdający: źródła + polecenie (bez artefaktów arkusza, `_oczysc`)."""
    czesci = [_oczysc(z.get("zrodla_tekst", "")), _oczysc(z["polecenie"])]
    return "\n\n".join(c for c in czesci if c)
