"""Opisy ilustracji z lokalnych modeli wizyjnych (kandydaci na opisywacz w limicie rozmiaru) do pomiaru na pełnej maturze.

Co to jest: uruchamia llama-server z modelem wizyjnym (GGUF + mmproj), opisuje ilustracje zadań z podanych sesji
tym samym schematem co opis wzorcowy (scripts/opisy_wzorcowe.py: typ, napisy, co przedstawione, rozpoznanie),
po polsku albo po angielsku, i zapisuje w cache matura/obrazy pod wariantem `--nazwa` (np. vlm_q2b_en).
Po co: porównać realne opisywacze z sufitem (opis Claude) na pełnej maturze (scripts/pomiar_wzorzec.py --rejestruj).
Co zrobić: `uv run python scripts/opisy_kandydaci.py --nazwa vlm_q2b_en --model data/modele/qwen3.5-2b-q4/Qwen3.5-2B-Q4_K_M.gguf
--mmproj data/modele/qwen3.5-2b-q4/mmproj-F16.gguf --jezyk en`.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matura import devset, obrazy  # noqa: E402
from matura.llm import obraz_url  # noqa: E402
from matura.noc import Serwer  # noqa: E402

PROMPTY = obrazy.PROMPTY_VLM   # prompt w matura/obrazy.py (wspólny z egzaminem)


opisz = obrazy.opisz_vlm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nazwa", required=True, help="wariant w cache, prefiks vlm_")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--jezyk", choices=sorted(PROMPTY), default="pl")
    ap.add_argument("--sesje", default="2024-maj,2025-maj")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--rownolegle", type=int, default=4)
    a = ap.parse_args()
    if not a.nazwa.startswith("vlm_"):
        raise SystemExit("nazwa wariantu musi zaczynać się od vlm_")
    ilu = {}
    for s in a.sesje.split(","):
        for z in devset._cke(s):
            for il in obrazy.ilustracje(z):
                ilu.setdefault(il["rel"], il)
    todo = [il for rel, il in ilu.items() if obrazy.klucz(rel, a.nazwa) not in obrazy.wczytaj_cache()]
    print(f"ilustracji: {len(ilu)}, do opisania: {len(todo)}", flush=True)
    if not todo:
        return 0
    logi = ROOT / "review" / "b0-obrazy-2026-09-26" / "logi"; logi.mkdir(parents=True, exist_ok=True)
    s = Serwer(os.environ.get("LLAMA_SERVER", str(ROOT / "data/bin/llama-cuda/llama-server")), ROOT / a.model, ROOT / a.mmproj,
               a.port, a.rownolegle, 8192, logi / f"serwer_{a.nazwa}.log", ["--image-min-tokens", "1024"])
    czasy = []
    try:
        with ThreadPoolExecutor(a.rownolegle) as ex:
            futs = {ex.submit(opisz, s.url, il, a.jezyk): il for il in todo}
            for i, f in enumerate(as_completed(futs), 1):
                il = futs[f]
                tekst, sek = f.result()
                czasy.append(sek)
                obrazy._zapisz(il["rel"], a.nazwa, tekst, sek, model=Path(a.model).name, jezyk=a.jezyk)
                print(f"[{i}/{len(todo)}] {il['rel']} ({sek:.1f} s): {tekst[:100]}…", flush=True)
    finally:
        s.stop()
    print(f"KONIEC; średnio {sum(czasy) / len(czasy):.2f} s na obraz (przy {a.rownolegle} równolegle)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
