"""Ciągły sędzia nocny: co 2 min skanuje katalog wyników i ocenia wszystko, czego nie ma w cache.

Sędzia główny = `sedzia_model` (claude:fable, D17); sędzia kontrolny = `sedzia_kontrola` (Astra, tym ocenia
organizator) na deterministycznej próbce ~10% odpowiedzi (T2) → raport liczy zgodność obu sędziów.
Cache kluczowany (sędzia, id, odpowiedź): oceny różnych sędziów nie mieszają się.
Działa niezależnie od GPU (generacja i trening idą równolegle). Kończy się, gdy istnieje plik STOP
w katalogu wyników i pełny przebieg nie znalazł nic do oceny. Po każdym przebiegu odświeża RAPORT.md.
Uruchomienie: uv run python -m matura.ocen_wszystko --config noc/wspolne.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import devset, raport, sedzia
from .noc import ROOT, log, wczytaj_config, wczytaj_odp


def pozycje_do_oceny(p: Path, zadania: dict, cache: dict, sedzia_model: str) -> list[dict]:
    """Odpowiedzi z pliku `p`, których dany sędzia jeszcze nie ocenił → pozycje dla `sedzia.ocen_partie`."""
    poz = []
    for d in wczytaj_odp(p).values():
        z = zadania.get(d["id"])
        if z is None:
            continue
        k = sedzia.klucz(z["id"], d["odpowiedz"], sedzia_model)
        if k in cache:
            continue
        poz.append({"klucz": k, "id": z["id"], "pkt_max": z["pkt_max"], "polecenie": devset.tresc_dla_modelu(z),
                    "zasady": z["zasady_oceniania"], "rozwiazanie": z["rozwiazanie"], "odpowiedz": d["odpowiedz"],
                    "esej": z["esej"], "model": p.stem.split("__")[0], "split": z.get("split")})
    return poz


KOLEJNOSC_SPLIT = {"kalibracja": 0, "synt": 1, "test": 2, "dev": 3}


def priorytet(x: dict, rozmiary: dict) -> tuple:
    """Kolejność oceny przy ograniczonym budżecie sędziego: kalibracja, potem wypracowania (obszar ~50% nocy),
    potem modele od najmniejszego (kategoria „Mały"); w wierszu najpierw synt (najbliżej egzaminu), potem test, dev."""
    mb = rozmiary.get(x["model"], {}).get("razem", 0)
    return (x["model"] != "_kalibracja", not x["esej"], mb, x["model"], KOLEJNOSC_SPLIT.get(x["split"], 9))


def kontrolna(pozycje: list[dict], udzial: float = 0.1) -> list[dict]:
    """Deterministyczna próbka odpowiedzi (hasz id+odpowiedź, niezależny od sędziego) dla sędziego kontrolnego."""
    return [p for p in pozycje if int(sedzia.klucz(p["id"], p["odpowiedz"])[:8], 16) % 1000 < udzial * 1000]


def _partie(lista: list[dict], partia: int, rozmiary: dict | None = None) -> list[list[dict]]:
    """Partie w kolejności `priorytet`: wypracowania pojedynczo, krótkie po `partia` (w obrębie jednego modelu)."""
    uniq = sorted({x["klucz"]: x for x in lista}.values(), key=lambda x: priorytet(x, rozmiary or {}))
    out, biezaca = [], []
    for x in uniq:
        if x["esej"]:
            if biezaca:
                out.append(biezaca); biezaca = []
            out.append([x]); continue
        if biezaca and (len(biezaca) >= partia or biezaca[0]["model"] != x["model"]):
            out.append(biezaca); biezaca = []
        biezaca.append(x)
    if biezaca:
        out.append(biezaca)
    return out


def przebieg(wyn: Path, zadania: dict, model: str, rown: int, partia: int, kontrola: str | None = None,
             udzial_kontroli: float = 0.1, ocen_fn=sedzia.ocen_partie, cache: dict | None = None) -> int:
    """Jeden przebieg: ocena sędzią głównym wszystkiego bez oceny + sędzią kontrolnym próbki. Zwraca liczbę pozycji."""
    cache = sedzia.wczytaj_cache() if cache is None else cache
    pliki = sorted((wyn / "odpowiedzi").glob("*.jsonl"))
    rozm = json.loads((wyn / "rozmiary.json").read_text()) if (wyn / "rozmiary.json").exists() else {}
    glowne = [x for p in pliki for x in pozycje_do_oceny(p, zadania, cache, model)]
    partie = [(model, pt) for pt in _partie(glowne, partia, rozm)]
    n = len({x["klucz"] for x in glowne})
    if kontrola:
        kontr = kontrolna([x for p in pliki for x in pozycje_do_oceny(p, zadania, cache, kontrola)], udzial_kontroli)
        partie += [(kontrola, pt) for pt in _partie(kontr, partia, rozm)]
        n += len({x["klucz"] for x in kontr})
    if not partie:
        return 0
    log(f"[sędzia] {n} pozycji do oceny w {len(partie)} partiach ({model}" + (f" + kontrola {kontrola}" if kontrola else "") + ")")
    bledy = 0
    with ThreadPoolExecutor(rown) as ex:
        futs = {ex.submit(ocen_fn, pt, sm): pt for sm, pt in partie}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                bledy += 1
                log(f"[sędzia] partia nieudana: {str(e)[:300]}")
    if bledy:
        log(f"[sędzia] nieudanych partii: {bledy} (spróbuję w następnym przebiegu)")
        time.sleep(60 * min(bledy, 5))  # prawdopodobny limit subskrypcji → zwolnij
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="noc/wspolne.toml")
    ap.add_argument("--raz", action="store_true", help="jeden przebieg i koniec")
    a = ap.parse_args()
    og = wczytaj_config(a.config)["ogolne"]
    wyn = ROOT / og["katalog_wynikow"]
    while True:
        wszystkie = devset.wczytaj(og["sesje_wszystkie"])
        zd = {z["id"]: z for z in wszystkie}
        n = przebieg(wyn, zd, og["sedzia_model"], og.get("sedzia_rownolegle", 3), og.get("sedzia_partia", 12),
                     kontrola=og.get("sedzia_kontrola"), udzial_kontroli=og.get("udzial_kontroli", 0.1))
        if (wyn / "odpowiedzi").exists():
            raport.zapisz(wyn, wszystkie, {"ogolne": og})
        if a.raz or (n == 0 and (wyn / "STOP").exists()):
            log("[sędzia] koniec"); return 0
        if n == 0:
            time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
