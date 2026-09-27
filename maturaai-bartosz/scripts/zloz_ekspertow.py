"""Złożenie odpowiedzi ekspertów wg typu zadania (zamknięte z jednego pliku, otwarte z drugiego) do pomiaru pełnej matury.

Co to jest: łączy dwa pliki odpowiedzi (format review/*/odpowiedzi/*.jsonl) w jeden: zadania zamknięte z `--zamkniete`,
pozostałe zadania krótkie z `--otwarte` (typ wg devset.typ_zadania). Pole `zrodlo` mówi, skąd jest odpowiedź.
Po co: złożenie wirtualne zestawu ekspertów (np. zamknięte z bazy + adapter LoRA, otwarte z bazy bez adaptera, D21)
oceniane potem przez matura/pelna_matura.py jak każdy inny zestaw.
Co zrobić: `uv run python scripts/zloz_ekspertow.py --zamkniete A.jsonl --otwarte B.jsonl --wyjscie C.jsonl
--sesje 2024-maj,2025-maj`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matura import devset  # noqa: E402


def wczytaj(p: Path) -> dict[str, dict]:
    return {d["id"]: d for d in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def zloz(zamkniete: dict, otwarte: dict, typy: dict[str, str]) -> list[dict]:
    """Rekordy złożone: dla każdego zadania z `typy` odpowiedź z pliku właściwego dla jego typu (brak → pomijane)."""
    out = []
    for zid, t in typy.items():
        zr, plik = ("zamkniete", zamkniete) if t == "zamkniete" else ("otwarte", otwarte)
        if zid in plik:
            out.append({**plik[zid], "zrodlo": zr})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zamkniete", required=True)
    ap.add_argument("--otwarte", required=True)
    ap.add_argument("--wyjscie", required=True)
    ap.add_argument("--sesje", default="2024-maj,2025-maj")
    a = ap.parse_args()
    typy = {z["id"]: z["typ"] for s in a.sesje.split(",") for z in devset._cke(s)}
    rek = zloz(wczytaj(Path(a.zamkniete)), wczytaj(Path(a.otwarte)), typy)
    Path(a.wyjscie).parent.mkdir(parents=True, exist_ok=True)
    Path(a.wyjscie).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rek), encoding="utf-8")
    n_z = sum(r["zrodlo"] == "zamkniete" for r in rek)
    print(f"{a.wyjscie}: {len(rek)} odpowiedzi ({n_z} zamkniętych z {a.zamkniete}, {len(rek) - n_z} otwartych z {a.otwarte})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
