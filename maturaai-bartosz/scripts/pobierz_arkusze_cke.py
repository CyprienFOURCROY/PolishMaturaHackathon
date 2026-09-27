"""Pobiera arkusze CKE z historii (poziom rozszerzony) i zasady oceniania z arkusze.pl.

Cel: repo nie zawiera plików CKE (regulamin hackathonu); ten skrypt odtwarza katalog
data/cke/ z sieci. Uruchomienie: `uv run python scripts/pobierz_arkusze_cke.py`.
Wyjście: data/cke/historia-{rok}-{sesja}-{arkusz|odpowiedzi}.pdf + data/cke/MANIFEST.json
(URL, rozmiar, SHA256 każdego pliku). Powtarzalne: istniejące pliki o właściwym rozmiarze pomija.
Źródło: https://arkusze.pl/historia-matura-poziom-rozszerzony/ (arkusze CKE, domena publiczna
dokumentów urzędowych; serwis udostępnia kopie).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

BAZA = "https://arkusze.pl/maturalne/historia-{rok}-{sesja}-matura-rozszerzona{suf}.pdf"
INFORMATOR = "https://arkusze.pl/informatory/informator-maturalny-historia-2025.pdf"
LATA = range(2015, 2027)
SESJE = ("maj", "czerwiec")
KATALOG = Path(__file__).resolve().parent.parent / "data" / "cke"


def pobierz(url: str, cel: Path) -> dict | None:
    """Pobiera url do cel, zwraca wpis manifestu albo None przy 404."""
    if cel.exists() and cel.stat().st_size > 10_000:
        dane = cel.read_bytes()
    else:
        odp = requests.get(url, timeout=60)
        if odp.status_code == 404:
            return None
        odp.raise_for_status()
        dane = odp.content
        cel.write_bytes(dane)
    return {"plik": cel.name, "url": url, "bajty": len(dane), "sha256": hashlib.sha256(dane).hexdigest()}


def main() -> int:
    KATALOG.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for rok in LATA:
        for sesja in SESJE:
            for suf, rodzaj in (("", "arkusz"), ("-odpowiedzi", "odpowiedzi")):
                url = BAZA.format(rok=rok, sesja=sesja, suf=suf)
                cel = KATALOG / f"historia-{rok}-{sesja}-{rodzaj}.pdf"
                wpis = pobierz(url, cel)
                print(("OK  " if wpis else "404 ") + cel.name, flush=True)
                if wpis:
                    wpis.update(rok=rok, sesja=sesja, rodzaj=rodzaj)
                    manifest.append(wpis)
    wpis = pobierz(INFORMATOR, KATALOG / "informator-historia-2025.pdf")
    if wpis:
        wpis.update(rodzaj="informator")
        manifest.append(wpis)
    (KATALOG / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Pobrano {len(manifest)} plików → {KATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
