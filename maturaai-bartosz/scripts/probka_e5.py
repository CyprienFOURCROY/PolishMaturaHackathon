"""Próbka jakości eseju e5 (bez sędziego): model × tematy × oba stanowiska → plik odpowiedzi w formacie nocy.

Co to jest: jednorazowy pomiar jakości i czasu wypracowania e5 (matura/esej.py, `napisz_e5`) na tematach CKE.
Po co: przed pomiarem sędzią zobaczyć, czy akapity trzymają stanowisko, ile trwa esej i ile dat weryfikator poprawia.
Co zrobić (maszyna z GPU; własny llama-server na porcie 8096, albo --url do działającego serwera):
  uv run python scripts/probka_e5.py --model bielik-1.5b-q8 --tematy 2025-maj-25-t1,2025-maj-25-t2,2025-maj-25-t3
Wyjście: review/etapB-2026-09-26/probka_e5.jsonl (pola jak odpowiedzi nocy: id, model, konfig, odpowiedz, sekundy
+ meta e5) i probka_e5.json (podsumowanie: słowa, sekundy, próby na esej).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matura import devset, esej  # noqa: E402
from matura.noc import Serwer  # noqa: E402

MODELE = {"bielik-1.5b-q8": "data/modele/bielik-1.5b-q8/Bielik-1.5B-v3.0-Instruct.Q8_0.gguf",
          "qwen3.5-2b-q4": "data/modele/qwen3.5-2b-q4/Qwen3.5-2B-Q4_K_M.gguf",
          "bielik-4.5b-q8": "data/modele/bielik-4.5b-q8/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"}
KONFIG = {"nie_zgadzam": "e5", "zgadzam": "e5_zgadzam"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bielik-1.5b-q8", choices=sorted(MODELE))
    ap.add_argument("--tematy", default="2025-maj-25-t1,2025-maj-25-t2,2025-maj-25-t3")
    ap.add_argument("--stanowiska", default="nie_zgadzam,zgadzam")
    ap.add_argument("--url", help="działający llama-server (inaczej własny na --port)")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--wyniki", default="review/etapB-2026-09-26")
    ap.add_argument("--bez-wiki", action="store_true")
    a = ap.parse_args()
    wyn = ROOT / a.wyniki
    wyn.mkdir(parents=True, exist_ok=True)
    ids = a.tematy.split(",")
    zad = {z["id"]: z for z in devset.wczytaj(["eseje-cke"]) if z["id"] in ids}
    from matura.karty import Karty
    karty = Karty()
    wiki = None
    if not a.bez_wiki:
        from matura.retrieval import Wikipedia
        t0 = time.time()
        wiki = Wikipedia()
        print(f"Wikipedia wczytana ({time.time() - t0:.0f} s)", flush=True)
    serwer = None
    url = a.url
    if not url:
        (wyn / "logi").mkdir(exist_ok=True)
        serwer = Serwer(str(ROOT / "data/bin/llama-cuda/llama-server"), ROOT / MODELE[a.model], None, a.port, 1, 8192,
                        wyn / "logi" / f"serwer_{a.model}.log")
        url = serwer.url
    plik = wyn / "probka_e5.jsonl"
    podsum = []
    try:
        for st in a.stanowiska.split(","):
            for zid in ids:
                t0 = time.time()
                tekst, sek, meta = esej.napisz_e5(zad[zid], url=url, wiki=wiki, karty=karty, stanowisko=st)
                sciana = time.time() - t0
                rek = {"id": zid, "model": a.model, "konfig": KONFIG[st], "odpowiedz": tekst, "sekundy": round(sek, 2),
                       "sekundy_calosc": round(sciana, 2), **meta}
                with open(plik, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rek, ensure_ascii=False) + "\n")
                podsum.append({k: rek[k] for k in ("id", "konfig", "slow", "sekundy", "sekundy_calosc", "proby",
                                                   "regeneracje_dlugosci", "daty_podmienione", "daty_usuniete",
                                                   "powtorzenia_usuniete", "sprzeczne_usuniete", "sprzeczne_pozostale")})
                print(f"{zid} {st}: {meta['slow']} słów, model {sek:.1f} s, całość {sciana:.1f} s, próby {meta['proby']}",
                      flush=True)
    finally:
        if serwer:
            serwer.stop()
    (wyn / "probka_e5.json").write_text(json.dumps({"model": a.model, "czas": time.strftime("%Y-%m-%d %H:%M"),
                                                     "eseje": podsum}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
