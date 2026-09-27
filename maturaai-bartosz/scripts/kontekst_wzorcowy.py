"""Kontekst wzorcowy (sufit wiedzy, krok B1 planu): Claude wypisuje fakty potrzebne do rozwiązania każdego zadania krótkiego dev.

Co to jest: dla zadań krótkich z podanych sesji Claude (Opus, --effort low) dostaje treść zadania i wypisuje 3-6 faktów z
wiedzy historycznej (identyfikacja źródła lub ilustracji, daty, postaci, pojęcia) BEZ rozwiązania (bez litery, P/F,
numeru, rozstrzygnięcia). Cache: data/wiedza/kontekst_wzorcowy.jsonl ({id, fakty}), wznawialny.
Po co: konfiguracja `goly_vlm_wiedza` (harness) pokazuje, ile punktów dałaby idealna baza wiedzy dla zadań krótkich,
zanim zespół ją zbuduje. TYLKO pomiar na dev w fazie budowy; na egzaminie zamknięte API są zakazane.
Co zrobić: uv run python scripts/kontekst_wzorcowy.py --sesje 2024-maj,2025-maj [--partia 8] [--rownolegle 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from matura import devset, pelna_matura  # noqa: E402
from buduj_akapity_eseju import claude  # noqa: E402

CACHE = ROOT / "data" / "wiedza" / "kontekst_wzorcowy.jsonl"
PROMPT = """Uczeń rozwiązuje zadania maturalne z historii (poziom rozszerzony). Dla KAŻDEGO zadania wypisz 3-6 faktów
z wiedzy historycznej, które są potrzebne, żeby je rozwiązać: identyfikacja źródła lub ilustracji (co to jest, kto,
kiedy), daty, postaci, pojęcia, związki przyczynowe. NIE podawaj rozwiązania: nie wskazuj litery, P/F, numeru ani
rozstrzygnięcia i nie pisz, która odpowiedź jest poprawna. Każdy fakt jednym zdaniem po polsku. Tylko JSON.

ZADANIA:
{zadania}"""
_s = {"type": "string"}
SCHEMAT = {"type": "object", "additionalProperties": False, "required": ["konteksty"], "properties": {"konteksty": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "fakty"],
                               "properties": {"id": _s, "fakty": {"type": "array", "items": _s}}}}}}
_lock = threading.Lock()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesje", default="2024-maj,2025-maj")
    ap.add_argument("--partia", type=int, default=8)
    ap.add_argument("--rownolegle", type=int, default=4)
    a = ap.parse_args(argv)
    gotowe = {json.loads(l)["id"] for l in CACHE.open(encoding="utf-8")} if CACHE.exists() else set()
    zad = [z for s in a.sesje.split(",") for z in pelna_matura.zadania_sesji(s)["krotkie"] if z["id"] not in gotowe]
    partie = [zad[i:i + a.partia] for i in range(0, len(zad), a.partia)]
    print(f"zadań do opisania: {len(zad)} w {len(partie)} partiach", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    def jedna(p):
        tekst = "\n\n".join(f"### id: {z['id']}\n{devset.tresc_dla_modelu(z)[:3000]}" for z in p)
        try:
            wyn = claude(PROMPT.format(zadania=tekst), SCHEMAT, "low")["konteksty"]
        except Exception as e:  # noqa: BLE001
            print("partia NIEUDANA:", str(e)[:200], flush=True); return
        ids = {z["id"] for z in p}
        with _lock, CACHE.open("a", encoding="utf-8") as f:
            for k in wyn:
                if k["id"] in ids:
                    f.write(json.dumps({"id": k["id"], "fakty": k["fakty"]}, ensure_ascii=False) + "\n")
        print(f"partia: {len(wyn)} kontekstów", flush=True)

    with ThreadPoolExecutor(a.rownolegle) as ex:
        list(ex.map(jedna, partie))
    return 0


if __name__ == "__main__":
    sys.exit(main())
