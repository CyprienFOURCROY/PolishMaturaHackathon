"""Angielska wersja bazy haseł do potoku angielskiego (krok C planu docs/plans/2026-09-26-plan-po-1a-1b.md).

Co to jest: jednorazowe tłumaczenie tekstu każdego hasła (`wiedza._tekst_hasla`: tytuł, tekst, dokumenty,
ikonografia) na angielski, offline: domyślnie lokalnie modelem Qwen3-4B-Instruct-2507 UD-IQ3_XXS (`--tlumacz baza`,
llama-server na porcie 8095; oszczędza limit Claude), albo Claude (`--tlumacz claude --effort low`, około 61 wywołań).
Baza wiedzy jest poza limitem rozmiaru (D22).
Wyjście: data/wiedza/tlumaczenia/hasla_en.jsonl ({klucz, id, pl, en}). Podkatalog, bo `wiedza.BazaHasel` wczytuje
`data/wiedza/hasla*.jsonl` i nie może dostać angielskich wpisów.
Po co: w potoku angielskim wyszukiwanie zostaje po polsku (polskie zapytanie i polskie hasła, jak w `goly_vlm_kb`),
a do promptu trafia angielski tekst znalezionych haseł (`matura/jezyk.py --kb`).
Co zrobić: uv run python scripts/hasla_en.py [--tlumacz baza|claude] [--rownolegle 4]; wznawialne (cache po treści hasła).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from matura import jezyk, wiedza  # noqa: E402

PLIK = ROOT / "data" / "wiedza" / "tlumaczenia" / "hasla_en.jsonl"
SYSTEM = ("You translate entries of a Polish history knowledge base into English for a small model that answers "
          "exam tasks in English. Translate every entry faithfully and completely: keep all facts, names, dates and "
          "numbers, and do not add or remove information. Use the standard English names of people, places, "
          "treaties, institutions and events (for example: Union of Krewo, Teutonic Order, Four-Year Sejm). "
          "Return the translations with the same ids.")


def klucz(h: dict) -> str:
    return jezyk.sha("haslo", h["id"], wiedza._tekst_hasla(h))


def tlumaczenia(plik: Path = PLIK) -> dict[str, str]:
    """id hasła → tekst angielski (tylko wpisy zgodne z bieżącą treścią hasła)."""
    c = jezyk.Cache(plik)
    return {h["id"]: c.get(klucz(h))["en"] for h in wiedza.BazaHasel().h if c.get(klucz(h))}


def _lokalnie(po_id: dict, c, model: str, port: int, rownolegle: int) -> None:
    """Tłumaczenie bazą (matura/tlumacz.Baza) na własnym llama-server; zapis co 16 haseł."""
    from matura.noc import Serwer
    from matura.tlumacz import Baza
    gguf, bez_myslenia, sloty = jezyk.MODELE[model]
    log = ROOT / "data" / "wiedza" / "tlumaczenia" / f"serwer_{model}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    s = Serwer(str(jezyk.LLAMA), ROOT / "data" / "modele" / gguf, None, port, sloty, 8192, log)
    try:
        tl = Baza(s.url, bez_myslenia, rownolegle=min(rownolegle, sloty))
        ids = list(po_id)
        for i in range(0, len(ids), 16):
            cz = ids[i:i + 16]
            wyn = tl.tlumacz([wiedza._tekst_hasla(po_id[j]) for j in cz], "pl-en")
            c.dodaj([{"klucz": klucz(po_id[j]), "id": j, "pl": wiedza._tekst_hasla(po_id[j]), "en": e,
                      "tlumacz": f"baza:{model}"} for j, e in zip(cz, wyn) if e.strip()])
            jezyk.log(f"[hasła EN] {min(i + 16, len(ids))}/{len(ids)}")
    finally:
        s.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tlumacz", choices=("baza", "claude"), default="baza")
    ap.add_argument("--model", default="qwen3-4b-2507-iq3xxs", help="--tlumacz baza: model z jezyk.MODELE")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--rownolegle", type=int, default=4)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--budzet", type=int, default=12000, help="znaków polskiego tekstu na jedno wywołanie")
    a = ap.parse_args(argv)
    hasla = wiedza.BazaHasel().h
    c = jezyk.Cache(PLIK)
    po_id = {h["id"]: h for h in hasla if klucz(h) not in c}
    jezyk.log(f"[hasła EN] do zrobienia {len(po_id)} z {len(hasla)} ({a.tlumacz})")
    if a.tlumacz == "baza":
        _lokalnie(po_id, c, a.model, a.port, max(a.rownolegle, 8))
        gotowe = len(tlumaczenia())
        jezyk.log(f"[hasła EN] gotowe {gotowe}/{len(hasla)}")
        return 0 if gotowe == len(hasla) else 1
    from buduj_akapity_eseju import claude
    partie = jezyk.partie_wg_znakow([(i, len(wiedza._tekst_hasla(h))) for i, h in po_id.items()], a.budzet)

    def jedna(ids: list[str]) -> int:
        ids = [i for i in ids if klucz(po_id[i]) not in c]
        if not ids:
            return 0
        lad = [{"id": i, "tekst": wiedza._tekst_hasla(po_id[i])} for i in ids]
        w = claude(SYSTEM + "\n\nENTRIES TO TRANSLATE (JSON):\n" + json.dumps(lad, ensure_ascii=False, indent=1),
                   jezyk.SCHEMA_TL, a.effort)
        got = jezyk.rozpakuj(w, ids)
        c.dodaj([{"klucz": klucz(po_id[i]), "id": i, "pl": wiedza._tekst_hasla(po_id[i]), "en": t,
                  "tlumacz": f"claude:opus --effort {a.effort}"} for i, t in got.items() if t])
        return len(got)

    nieudane = jezyk.wykonaj_partie(partie, jedna, a.rownolegle, "hasła EN")
    gotowe = len(tlumaczenia())
    jezyk.log(f"[hasła EN] gotowe {gotowe}/{len(hasla)}, nieudanych partii {nieudane}")
    return 0 if gotowe == len(hasla) else 1


if __name__ == "__main__":
    sys.exit(main())
