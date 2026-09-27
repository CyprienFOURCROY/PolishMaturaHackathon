"""Pomiar zadań zamkniętych (krok Z1 planu docs/plans/2026-09-27-plan-ostatnie-10h.md): tylko zadania zamknięte, automat.

Co to jest: generuje odpowiedzi na zadania zamknięte wskazanych arkuszy w konfiguracjach harnessu (np. `goly_vlm_kb` i
`goly_vlm_kb_z`) na Qwen3.5-4B UD-IQ3_XXS i zapisuje je jak noc.py; wynik liczy potem `matura.pelna_matura` (kolumna
„zamknięte”; zadania z odczytaną odpowiedzią ocenia automat klucza, sędzia tylko nieodczytane).
Po co: zdecydować na dev, czy ścisły format i głosowanie (`matura/zamkniete.py`) podnoszą punkty zamkniętych, potem raz
sprawdzić na arkuszach walidacyjnych (D25), bez kosztu sędziego tam, gdzie automat czyta odpowiedź.
Co zrobić: uv run python scripts/zamkniete_pomiar.py --sesje 2024-maj,2025-maj --konfig goly_vlm_kb_z
[--opisz] (--opisz: najpierw opisy brakujących ilustracji opisywaczem Qwen3.5-2B, port 8092). Porty 8092-8093
(8095-8099 zajmują inne przebiegi). `--typ rozstrzygnij`: zamiast zamkniętych zadania otwarte „Rozstrzygnij…”
(konfiguracja `goly_vlm_kb_zr`, wyniki w review/rozstrzygnij-2026-09-27/, ocena sędzią). Bez MATURA_TEMPERATURA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matura import devset, harness, obrazy, rozstrzygnij  # noqa: E402
from matura.noc import Serwer  # noqa: E402

BIN = str(ROOT / "data/bin/llama-cuda/llama-server")
GGUF = ROOT / "data/modele/qwen3.5-4b-iq3xxs/Qwen3.5-4B-UD-IQ3_XXS.gguf"
OPIS_GGUF = ROOT / "data/modele/qwen3.5-2b-q4/Qwen3.5-2B-Q4_K_M.gguf"
OPIS_MMPROJ = ROOT / "data/modele/qwen3.5-2b-q4/mmproj-F16.gguf"
WYNIKI = {"zamkniete": ROOT / "review/zamkniete-2026-09-27", "rozstrzygnij": ROOT / "review/rozstrzygnij-2026-09-27"}
FILTR = {"zamkniete": lambda z: z.get("typ") == "zamkniete", "rozstrzygnij": rozstrzygnij.czy_rozstrzygnij}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesje", default="2024-maj,2025-maj")
    ap.add_argument("--konfig", default="goly_vlm_kb_z", help="konfiguracje po przecinku")
    ap.add_argument("--model", default="qwen3.5-4b-iq3xxs")
    ap.add_argument("--opisz", action="store_true")
    ap.add_argument("--rownolegle", type=int, default=4)
    ap.add_argument("--typ", choices=sorted(WYNIKI), default="zamkniete")
    a = ap.parse_args(argv)
    if os.environ.get("MATURA_TEMPERATURA"):
        print("UWAGA: MATURA_TEMPERATURA ustawiona: głosowanie bez sensu (wszystkie próbki z tą samą temperaturą)")
    WYN = WYNIKI[a.typ]
    zz = [z for s in a.sesje.split(",") for z in devset._cke(s) if FILTR[a.typ](z)]
    (WYN / "logi").mkdir(parents=True, exist_ok=True)
    print(f"zadań ({a.typ}): {len(zz)} ({a.sesje})", flush=True)
    if a.opisz:
        ilu = [il for z in zz for il in obrazy.ilustracje(z)]
        s = Serwer(BIN, OPIS_GGUF, OPIS_MMPROJ, 8092, 4, 8192, WYN / "logi/serwer_opisywacz.log", obrazy.SERWER_VLM_ARGS)
        try:
            n = obrazy.opisz_brakujace(ilu, s.url, "vlm_q2b_en", "en", rownolegle=4, model=OPIS_GGUF.name)
        finally:
            s.stop()
        print(f"opisy: {len(ilu)} ilustracji, {n} nowych", flush=True)
    s = Serwer(BIN, GGUF, None, 8093, a.rownolegle, 8192, WYN / "logi/serwer_baza.log")
    lock = threading.Lock()
    try:
        for konfig in a.konfig.split(","):
            p = WYN / "odpowiedzi" / f"{a.model}__{konfig}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            gotowe = {json.loads(l)["id"] for l in p.open(encoding="utf-8")} if p.exists() else set()
            todo = [z for z in zz if z["id"] not in gotowe]
            t0 = time.time()

            def jedno(z):
                try:
                    t, sek, meta = harness.odpowiedz(konfig, z, url=s.url, wiki=None, obrazy=False, bez_myslenia=True)
                except Exception as e:  # noqa: BLE001
                    t, sek, meta = f"(BŁĄD: {e})", 0.0, {}
                with lock, p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": z["id"], "model": a.model, "konfig": konfig, "odpowiedz": t,
                                        "sekundy": round(sek, 2), "glosy": meta.get("glosy")}, ensure_ascii=False) + "\n")

            with ThreadPoolExecutor(a.rownolegle) as ex:
                list(ex.map(jedno, todo))
            print(f"{konfig}: {len(todo)} odpowiedzi w {time.time() - t0:.0f} s → {p.relative_to(ROOT)}", flush=True)
    finally:
        s.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
