"""Zbiór SFT uczniów eseju z esejów nauczyciela (eksperyment 27.09).

Co to jest: `data/sft/train-esej-nauczyciel.jsonl` w formacie `matura.trening` (prompt/completion), zbudowany
wyłącznie przez `matura.esej_sft.przyklad`, więc uczeń na egzaminie dostaje ten sam prompt co w treningu.
Po co: jeden plik danych dla obu uczniów (S1 Bielik-1.5B pełny SFT, S2 Qwen3.5-4B LoRA), z kontrolą jakości.
Kontrola: bez tematów z WYKLUCZONE (bliskie ocenianym), esej 300-900 słów, bez markdown i list, bez nagłówka „Temat”; drukuje średnią długość zdania (proste słownictwo).
Co zrobić: `uv run python scripts/dane_eseju_sft.py [--wejscie data/esej/sft/eseje_nauczyciel.jsonl] [--wyjscie ...]`.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from matura import esej_sft  # noqa: E402

TEMATY = ROOT / "data" / "esej" / "sft" / "tematy_trening.jsonl"
TEMATY_KB = ROOT / "data" / "esej" / "sft" / "tematy_trening_kb.jsonl"   # runda z materiałem (pole material)
TEMATY_KB2 = ROOT / "data" / "esej" / "sft" / "tematy_trening_kb2.jsonl"   # runda kb2 (zapytanie z samej tezy)
RE_MARKDOWN = re.compile(r"(^\s*#|\*\*|^\s*[-*•]\s|^\s*\d+[.)]\s)", re.M)
RE_ZDANIE = re.compile(r"[^.!?]+[.!?]")
RE_NAGLOWEK = re.compile(r"\s*temat\s*(?:\d|:|nr)", re.I)   # „Temat 2.”, „Temat:” (nie zdanie „Temat dotyczy…”)
MIN_SLOW, MAX_SLOW = 300, 900
# tematy bliskie ocenianym (ręczna kontrola bliskości tematów): Galicja ≈ 2023-czerwiec-t2, Piłsudski ≈ 2024-maj-t3,
# wojny XVII w. ≈ 2024-maj-t2, rewolucja francuska ≈ 2023-maj-t2, kryzysy a totalitaryzmy ≈ 2024-czerwiec-t3
WYKLUCZONE = {"t097", "t214", "t177", "t207", "t074", "t037", "t190"}


def powod_odrzucenia(esej: str) -> str | None:
    n = esej_sft.slowa(esej)
    if n < MIN_SLOW:
        return "za krótki"
    if n > MAX_SLOW:
        return "za długi"
    if RE_MARKDOWN.search(esej):
        return "markdown/lista"
    if RE_NAGLOWEK.match(esej):
        return "nagłówek tematu"
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wejscie", nargs="+", default=[str(ROOT / "data" / "esej" / "sft" / "eseje_nauczyciel.jsonl")])
    ap.add_argument("--wyjscie", default=str(ROOT / "data" / "sft" / "train-esej-nauczyciel.jsonl"))
    a = ap.parse_args(argv)
    tematy = {t["id"]: t for p in (TEMATY, TEMATY_KB, TEMATY_KB2) if p.exists()
              for t in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    rekordy = {}
    for plik in a.wejscie:
        for l in Path(plik).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                rekordy[r["id"]] = r   # ostatni rekord id wygrywa (ponowienia generacji)
    przyklady, odrzucone, dl_zdan, slowa = [], Counter(), [], []
    for rid, r in rekordy.items():
        if rid.split("-")[0] in WYKLUCZONE:
            odrzucone["temat bliski ocenianym"] += 1
            continue
        t = tematy.get(rid, {})
        temat, stan, esej = r.get("temat") or t.get("temat"), r.get("stanowisko") or t.get("stanowisko"), r.get("esej", "")
        if not temat or not esej:
            odrzucone["brak tematu lub eseju"] += 1
            continue
        if (p := powod_odrzucenia(esej)):
            odrzucone[p] += 1
            continue
        przyklady.append(esej_sft.przyklad(temat, esej, stan, r.get("material") or t.get("material")))
        slowa.append(esej_sft.slowa(esej))
        dl_zdan += [esej_sft.slowa(z) for z in RE_ZDANIE.findall(esej)]
    wy = Path(a.wyjscie)
    wy.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in przyklady), encoding="utf-8")
    print(f"eseje: {len(rekordy)}, przykłady: {len(przyklady)}, odrzucone: {dict(odrzucone)} → {wy}")
    if slowa:
        print(f"słowa: średnio {statistics.mean(slowa):.0f} (min {min(slowa)}, max {max(slowa)}); "
              f"zdanie: średnio {statistics.mean(dl_zdan):.1f} słowa, mediana {statistics.median(dl_zdan):.0f}")


if __name__ == "__main__":
    main()
