"""Automatyczna ocena zadań z kluczem dla wszystkich plików odpowiedzi (bez sędziego LLM).

Co to jest: tabela model × konfiguracja na zadaniach zamkniętych i krótkich „Podaj…" (matura/klucz.py),
z podziałem synt/test/dev, oraz zgodność z ocenami sędziów LLM z cache (walidacja automatu).
Po co: szybka pętla do poprawiania harnessu na małych modelach (sekundy zamiast godzin sędziego).
Co zrobić: `uv run python -m matura.auto_ocena` → review/<katalog_wynikow>/auto_klucz.json i auto_klucz.md.
Wynik „pewne %" liczy tylko rozstrzygnięte; widełki min/max zakładają 0 / pełne punkty za nierozstrzygnięte.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from . import devset, klucz, sedzia
from .noc import ROOT, wczytaj_config, wczytaj_odp

SPLITY = ("synt", "test", "dev")
SEDZIA_SESJI = "claude:opus-sesja"   # oceny Claude z sesji dla zadań nierozstrzygniętych przez automat


def wczytaj_sesji(wyn) -> dict:
    """Oceny sędziego z sesji (oceny_opus.jsonl) → {klucz: wpis}."""
    p = wyn / "oceny_opus.jsonl"
    if not p.exists():
        return {}
    return {w["klucz"]: w for w in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def ocen_plik(p, zd: dict, sesji: dict | None = None) -> dict:
    """Oceny automatu dla jednego pliku odpowiedzi → {id: ocena} (tylko zadania obsługiwane).
    Nierozstrzygnięte uzupełnia ocena sędziego z sesji, jeśli jest (pole zrodlo: automat / sędzia)."""
    out = {}
    for d in wczytaj_odp(p).values():
        z = zd.get(d["id"])
        if z is None or not klucz.obslugiwane(z):
            continue
        o = {**klucz.ocen(z, d["odpowiedz"]), "split": z.get("split"), "odpowiedz": d["odpowiedz"], "zrodlo": "automat"}
        g = (sesji or {}).get(sedzia.klucz(z["id"], d["odpowiedz"], SEDZIA_SESJI))
        if not o["pewne"] and g is not None:
            o.update(pkt=max(0, min(int(g["pkt"]), z["pkt_max"])), pewne=True, powod=g["uzasadnienie"], zrodlo=SEDZIA_SESJI)
        out[z["id"]] = o
    return out


def podsumuj(oceny: dict) -> dict:
    s = {}
    for sp in SPLITY + ("razem",):
        wyb = [o for o in oceny.values() if sp == "razem" or o["split"] == sp]
        pm = sum(o["pkt_max"] for o in wyb)
        pew = [o for o in wyb if o["pewne"]]
        pkt = sum(o["pkt"] for o in pew)
        pm_pew = sum(o["pkt_max"] for o in pew)
        s[sp] = {"zadan": len(wyb), "pewne": len(pew), "pkt": pkt, "pkt_max": pm,
                 "proc_pewne": round(100 * pkt / pm_pew, 1) if pm_pew else None,
                 "proc_min": round(100 * pkt / pm, 1) if pm else None,
                 "proc_max": round(100 * (pkt + pm - pm_pew) / pm, 1) if pm else None}
    s["bledy_serwera"] = sum(1 for o in oceny.values() if o["powod"] == "błąd serwera")
    s["puste"] = sum(1 for o in oceny.values() if o["powod"] == "pusta odpowiedź")
    return s


def zgodnosc_z_llm(wszystkie: dict, zd: dict, cache: dict) -> dict:
    """Porównanie z ocenami sędziów LLM (cache) na tych samych odpowiedziach: {sędzia: {n, zgodne, rozbieżne[]}}."""
    wyn = defaultdict(lambda: {"n": 0, "zgodne": 0, "rozbiezne": []})
    po_kluczu = defaultdict(list)
    for v in cache.values():
        po_kluczu[(v["id"], v.get("sedzia"))].append(v)
    sedziowie = {v.get("sedzia") for v in cache.values()}
    for plik, oceny in wszystkie.items():
        for zid, o in oceny.items():
            if not o["pewne"] or o["zrodlo"] != "automat":
                continue
            for sd in sedziowie:
                g = cache.get(sedzia.klucz(zid, o["odpowiedz"], sd))
                if g is None:
                    continue
                w = wyn[sd]
                w["n"] += 1
                if int(g["pkt"]) == o["pkt"]:
                    w["zgodne"] += 1
                else:
                    w["rozbiezne"].append({"plik": plik, "id": zid, "auto": o["pkt"], "llm": g["pkt"],
                                           "klucz": zd[zid]["rozwiazanie"], "odpowiedz": o["odpowiedz"][:400],
                                           "powod": o["powod"], "uzasadnienie_llm": g.get("uzasadnienie", "")[:300]})
    return dict(wyn)


def _md(tabela: list[dict], zg: dict) -> str:
    L = ["# Auto-ocena zadań z kluczem (bez sędziego LLM)", "",
         "**Co to jest:** wynik każdego pliku odpowiedzi na zadaniach zamkniętych i krótkich „Podaj…” (matura/klucz.py).",
         "**Po co:** szybka informacja zwrotna dla harnessu. **Co zrobić:** porównać wiersze; „pewne %” to wynik na "
         "rozstrzygniętych, widełki min–max obejmują nierozstrzygnięte (idą do sędziego).", "",
         "| model | konfig | MB | synt pewne % | test pewne % | dev pewne % | razem pewne % | min–max % | rozstrz. | błędy serw. | puste |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "—" if x is None else f"{x:.0f}"  # noqa: E731
    for w in tabela:
        s = w["wynik"]
        L.append(f"| {w['model']} | {w['konfig']} | {w['mb']} | {f(s['synt']['proc_pewne'])} | {f(s['test']['proc_pewne'])} | "
                 f"{f(s['dev']['proc_pewne'])} | **{f(s['razem']['proc_pewne'])}** | {f(s['razem']['proc_min'])}–"
                 f"{f(s['razem']['proc_max'])} | {s['razem']['pewne']}/{s['razem']['zadan']} | {s['bledy_serwera']} | {s['puste']} |")
    L += ["", "## Zgodność automatu z sędziami LLM (te same odpowiedzi)", ""]
    for sd, w in zg.items():
        L.append(f"- {sd}: {w['zgodne']}/{w['n']} zgodnych ({100 * w['zgodne'] / max(w['n'], 1):.1f}%), rozbieżnych {len(w['rozbiezne'])}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="noc/wspolne.toml")
    a = ap.parse_args()
    og = wczytaj_config(a.config)["ogolne"]
    wyn = ROOT / og["katalog_wynikow"]
    zd = {z["id"]: z for z in devset.wczytaj(og["sesje_wszystkie"])}
    rozm = json.loads((wyn / "rozmiary.json").read_text()) if (wyn / "rozmiary.json").exists() else {}
    wszystkie, tabela = {}, []
    sesji = wczytaj_sesji(wyn)
    for p in sorted((wyn / "odpowiedzi").glob("*.jsonl")):
        oceny = ocen_plik(p, zd, sesji)
        if not oceny:
            continue
        model, konfig = p.stem.split("__", 1)
        wszystkie[p.stem] = oceny
        tabela.append({"model": model, "konfig": konfig, "mb": round(rozm.get(model, {}).get("razem", 0) / 1e6), "wynik": podsumuj(oceny)})
    tabela.sort(key=lambda w: (-(w["wynik"]["razem"]["proc_pewne"] or -1), w["mb"]))
    zg = zgodnosc_z_llm(wszystkie, zd, sedzia.wczytaj_cache())
    (wyn / "auto_klucz.json").write_text(json.dumps({"tabela": tabela, "zgodnosc": zg,
                                                      "oceny": wszystkie}, ensure_ascii=False, indent=1), encoding="utf-8")
    (wyn / "auto_klucz.md").write_text(_md(tabela, zg), encoding="utf-8")
    print(_md(tabela, zg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
