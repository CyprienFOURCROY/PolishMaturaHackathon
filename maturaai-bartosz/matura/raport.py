"""Raport poranny: model × konfiguracja → wynik na CKE (dev/test) i arkuszach sztucznych, luka do 35%, rozmiar.

Zbiory: dev = CKE 2023–2025, test = CKE 2026 (hold-out), synt = sztuczne arkusze z Wikipedii (najbliższe
egzaminowi hackathonu). Rekomendacja = najmniejszy model (MB), który ma ≥35% na synt ORAZ na test.
Sekcje T3: wypracowania (średnie A i B, błędy merytoryczne, słowa), kalibracja sędziego na wypracowaniach
z Informatora CKE (próg: średnio |sędzia - CKE| ≤ 2 pkt), zgodność sędziego głównego z kontrolnym (Astra, tym
ocenia organizator) i czas generacji per obszar (esej / reszta, cel nocy ~50/50).
raport.json = {"wiersze", "kalibracja_cke", "zgodnosc", "czas", "stan_na"}.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from . import devset, sedzia

PROG = 0.35
KLASY = ("zamkniete", "otwarte", "esej", "dev", "test", "synt")
PLIK_CKE = "_kalibracja__kalibracja_cke.jsonl"  # odpowiedź = wypracowanie z Informatora CKE (harness: kalibracja_cke)
PROG_CKE = 2.0        # kalibracja: średnio |sędzia - CKE| ≤ 2 pkt (docs/PETLA-NOCNA.md, D; E0)
PROG_ROZJAZDU = 0.10  # zgodność sędziów: rozjazd powyżej 10% = sygnał ostrzegawczy


def _rekordy(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pliki(wyn: Path) -> list[tuple[str, str, list[dict]]]:
    """(model, konfig, rekordy) dla każdego pliku <wyn>/odpowiedzi/<model>__<konfig>.jsonl (format matura/noc.py)."""
    out = []
    for p in sorted((wyn / "odpowiedzi").glob("*.jsonl")):
        model, konfig = p.stem.split("__")
        out.append((model, konfig, _rekordy(p)))
    return out


def _srednia(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _slowa(d: dict) -> int:
    """Liczba słów odpowiedzi: pole "slow" z meta (matura/esej.py), a gdy go brak, podział po białych znakach."""
    return d["slow"] if d.get("slow") is not None else len(d["odpowiedz"].split())


def _esej_stat(eseje: list[tuple[dict, dict]]) -> dict:
    """Ocenione wypracowania wiersza [(ocena, odpowiedź)] → średnie A i B, suma błędów merytorycznych,
    średnia liczba słów; None, gdy wiersz nie ma ocenionych wypracowań."""
    bl = [o["bledy"] for o, _ in eseje if o.get("bledy") is not None]
    return {"esej_n": len(eseje),
            "esej_A": _srednia(o.get("pkt_A") for o, _ in eseje),
            "esej_B": _srednia(o.get("pkt_B") for o, _ in eseje),
            "esej_bledy": sum(bl) if bl else None,
            "esej_slow": _srednia(_slowa(d) for _, d in eseje)}


def _wiersze(wyn: Path, zd: dict, cache: dict, rozm: dict, sedzia_model: str) -> list[dict]:
    out = []
    for model, konfig, odp in _pliki(wyn):
        odp = [d for d in odp if d["id"] in zd]
        pkt, mx = defaultdict(float), defaultdict(float)
        ocen = bledy = traf = traf_n = 0
        eseje = []  # (ocena, odpowiedź) ocenionych wypracowań
        for d in odp:
            z = zd[d["id"]]
            bledy += d["odpowiedz"].startswith("(BŁĄD")
            if z["split"] == "synt" and "kontekst_tytuly" in d and z.get("zrodlo_tytul"):
                traf_n += 1; traf += z["zrodlo_tytul"] in d["kontekst_tytuly"]
            o = cache.get(sedzia.klucz(z["id"], d["odpowiedz"], sedzia_model))
            if o is None:
                continue
            ocen += 1
            for k in ("razem", z["typ"], z["split"]):
                pkt[k] += o["pkt"]; mx[k] += z["pkt_max"]
            if z.get("esej"):
                eseje.append((o, d))
        sek = [d["sekundy"] for d in odp if d.get("sekundy")]
        r = {"model": model, "konfig": konfig, "odp": len(odp), "ocen": ocen, "bledy": bledy,
             "mb": rozm.get(model, {}).get("razem", 0) / 1e6, "sek": statistics.median(sek) if sek else None,
             "recall_synt": traf / traf_n if traf_n else None}
        for k in ("razem",) + KLASY:
            r[k] = pkt[k] / mx[k] if mx[k] else None
        r.update(_esej_stat(eseje))
        out.append(r)
    return out


def _oceny_cke() -> dict[str, dict]:
    """id wypracowania z Informatora → ocena CKE {pkt_A, pkt_B, pkt} (data/zasady/kalibracja_esejow.json)."""
    if not devset.KALIBRACJA.exists():
        return {}
    return {k["id"]: k for k in json.loads(devset.KALIBRACJA.read_text(encoding="utf-8"))}


def kalibracja_cke(wyn: Path, zd: dict, cache: dict, sedzia_model: str) -> dict:
    """Ocena sędziego vs ocena CKE na wypracowaniach z Informatora (odpowiedzi `_kalibracja/kalibracja_cke`,
    zadania ze zbioru eseje-kalibracja z polem pkt_cke).

    Zwraca {"n": liczba ocenionych, "srednia_roznica": średnie |sędzia - CKE| (None bez ocen), "prog": PROG_CKE,
    "spelniony": średnia ≤ próg (None bez ocen), "wiersze": [{"id", "cke_A", "cke_B", "cke", "sedzia_A", "sedzia_B",
    "sedzia", "roznica"}]}; roznica = sędzia - CKE (dodatnia: sędzia zawyża), None bez oceny."""
    cke = _oceny_cke()
    wiersze = []
    for d in {d["id"]: d for d in _rekordy(wyn / "odpowiedzi" / PLIK_CKE)}.values():
        z = zd.get(d["id"])
        if z is None or z.get("pkt_cke") is None:
            continue
        o = cache.get(sedzia.klucz(z["id"], d["odpowiedz"], sedzia_model)) or {}
        c = cke.get(z["id"], {})
        wiersze.append({"id": z["id"], "cke_A": c.get("pkt_A"), "cke_B": c.get("pkt_B"), "cke": z["pkt_cke"],
                        "sedzia_A": o.get("pkt_A"), "sedzia_B": o.get("pkt_B"), "sedzia": o.get("pkt"),
                        "roznica": o["pkt"] - z["pkt_cke"] if o else None})
    sr = _srednia(abs(w["roznica"]) for w in wiersze if w["roznica"] is not None)
    return {"n": sum(w["roznica"] is not None for w in wiersze), "srednia_roznica": sr, "prog": PROG_CKE,
            "spelniony": None if sr is None else sr <= PROG_CKE, "wiersze": sorted(wiersze, key=lambda w: w["id"])}


def zgodnosc(wyn: Path, zd: dict, cache: dict, sedzia_model: str, sedzia_kontrola: str | None) -> dict:
    """Zgodność sędziego głównego i kontrolnego na parach (zadanie, odpowiedź) ocenionych przez obu, ze wszystkich
    plików odpowiedzi (para występująca w kilku plikach liczy się raz), osobno "esej" i "krotkie":
    n, srednia_roznica (średnie |pkt główny - pkt kontrolny|), identyczne (odsetek równych ocen), srednia_glowny,
    srednia_kontrola (średnie pkt sędziów), rozjazd (suma |różnic| / suma pkt max; powyżej PROG_ROZJAZDU = sygnał
    ostrzegawczy). Bez sędziego kontrolnego n = 0."""
    pary: dict[str, dict] = {"esej": {}, "krotkie": {}}
    if sedzia_kontrola:
        for _, _, odp in _pliki(wyn):
            for d in odp:
                z = zd.get(d["id"])
                if z is None:
                    continue
                g = cache.get(sedzia.klucz(z["id"], d["odpowiedz"], sedzia_model))
                k = cache.get(sedzia.klucz(z["id"], d["odpowiedz"], sedzia_kontrola))
                if g is not None and k is not None:
                    pary["esej" if z.get("esej") else "krotkie"][(z["id"], d["odpowiedz"])] = (g["pkt"], k["pkt"], z["pkt_max"])
    out = {"sedzia_model": sedzia_model, "sedzia_kontrola": sedzia_kontrola, "prog_rozjazdu": PROG_ROZJAZDU}
    for obszar, pp in pary.items():
        v = list(pp.values())
        mx = sum(m for _, _, m in v)
        out[obszar] = {"n": len(v),
                       "srednia_roznica": _srednia(abs(a - b) for a, b, _ in v),
                       "identyczne": sum(a == b for a, b, _ in v) / len(v) if v else None,
                       "srednia_glowny": _srednia(a for a, _, _ in v),
                       "srednia_kontrola": _srednia(b for _, b, _ in v),
                       "rozjazd": sum(abs(a - b) for a, b, _ in v) / mx if mx else None}
    return out


def czas_obszarow(wyn: Path, zd: dict) -> dict:
    """Czas generacji w minutach: suma pola "sekundy" z plików odpowiedzi (bez modelu _kalibracja) dla wypracowań
    ("esej_min") i pozostałych zadań ("reszta_min"). Pętla nocna pilnuje podziału ~50/50 (docs/PETLA-NOCNA.md, B)."""
    sek = {"esej": 0.0, "reszta": 0.0}
    for model, _, odp in _pliki(wyn):
        if model == "_kalibracja":
            continue
        for d in odp:
            z = zd.get(d["id"])
            if z is not None:
                sek["esej" if z.get("esej") else "reszta"] += d.get("sekundy") or 0.0
    return {"esej_min": sek["esej"] / 60, "reszta_min": sek["reszta"] / 60}


def _porzadek(r: dict) -> tuple:
    return r["model"] != "_kalibracja", r["mb"], r["model"], r["konfig"]


def _fmt(v, s: str = "{:.0%}") -> str:
    """Liczba w sekcjach T3; brak wartości = „b.d.”."""
    return "b.d." if v is None else s.format(v)


def _sekcja_kalibracji(w: list[dict], kal: dict) -> list[str]:
    lin = ["", "## Kalibracja sędziego", ""]
    wz = {r["konfig"]: r for r in w if r["model"] == "_kalibracja" and r["konfig"] in ("wzorzec", "pusty")}
    if wz:
        lin += ["Odpowiedzi kalibracyjne (wiersze `_kalibracja` w tabeli niżej): " + "; ".join(
            f"`{k}` {_fmt(wz[k]['razem'])}, oczekiwane {ocz} (ocen. {wz[k]['ocen']}/{wz[k]['odp']})"
            for k, ocz in (("wzorzec", "≈100%"), ("pusty", "≈0%")) if k in wz) + ".", ""]
    lin.append("Wypracowania z Informatora CKE (`_kalibracja` / `kalibracja_cke` na zbiorze `eseje-kalibracja`):")
    if kal["wiersze"]:
        lin += ["", "| wypracowanie | CKE A | CKE B | CKE razem | sędzia A | sędzia B | sędzia razem | różnica (sędzia - CKE) |",
                "|---|---|---|---|---|---|---|---|"]
        lin += [f"| {x['id']} | {_fmt(x['cke_A'], '{}')} | {_fmt(x['cke_B'], '{}')} | {_fmt(x['cke'], '{}')} | "
                f"{_fmt(x['sedzia_A'], '{}')} | {_fmt(x['sedzia_B'], '{}')} | {_fmt(x['sedzia'], '{}')} | "
                f"{_fmt(x['roznica'], '{:+g}')} |" for x in kal["wiersze"]]
    else:
        lin.append("brak odpowiedzi w tym katalogu wyników.")
    lin.append("")
    ile = f"{kal['n']} z {len(kal['wiersze'])} wypracowań" + (", wynik cząstkowy" if kal["n"] < len(kal["wiersze"]) else "")
    if kal["spelniony"] is None:
        lin.append("**Próg kalibracji CKE nie został sprawdzony:** brak ocenionych wypracowań z Informatora "
                   "(uruchom `_kalibracja` / `kalibracja_cke` na `eseje-kalibracja`), więc oceny esejów są niezweryfikowane.")
    elif kal["spelniony"]:
        lin.append(f"**Próg kalibracji CKE spełniony:** średnio |sędzia - CKE| = {kal['srednia_roznica']:.2f} pkt "
                   f"≤ {kal['prog']:g} pkt ({ile}).")
    else:
        lin.append(f"**Próg kalibracji CKE NIE jest spełniony:** średnio |sędzia - CKE| = {kal['srednia_roznica']:.2f} pkt "
                   f"> {kal['prog']:g} pkt ({ile}); popraw prompt sędziego przed dalszymi esejami (E0).")
    return lin


def _sekcja_wypracowan(w: list[dict]) -> list[str]:
    """Tabela „Wypracowania” tylko z wierszy, które mają ocenione eseje (bez takich wierszy sekcji nie ma)."""
    ww = sorted((r for r in w if r["esej_n"]), key=_porzadek)
    if not ww:
        return []
    lin = ["", "## Wypracowania", "",
           "Średnie z ocenionych wypracowań wiersza (CKE: A narracja 0-12, B spójność 0-3, B = 0 poniżej 300 słów); "
           "błędy merytoryczne sumowane.", "",
           "| model | konfig | MB | ocen. esejów | esej | A śr. | B śr. | błędy mer. | słowa śr. |",
           "|---|---|---|---|---|---|---|---|---|"]
    lin += [f"| {r['model']} | {r['konfig']} | {r['mb']:.0f} | {r['esej_n']} | {_fmt(r['esej'])} | "
            f"{_fmt(r['esej_A'], '{:.1f}')} | {_fmt(r['esej_B'], '{:.1f}')} | {_fmt(r['esej_bledy'], '{}')} | "
            f"{_fmt(r['esej_slow'], '{:.0f}')} |" for r in ww]
    return lin


def _sekcja_zgodnosci(zg: dict) -> list[str]:
    lin = ["", "## Zgodność Claude vs Astra", ""]
    g, k = zg["sedzia_model"], zg["sedzia_kontrola"]
    if not k:
        return lin + ["Sędzia kontrolny nie jest ustawiony (`sedzia_kontrola` w [ogolne]), więc brak porównania."]
    obszary = {"esej": "wypracowania", "krotkie": "krótkie"}
    lin += [f"Sędzia główny `{g}`, kontrolny `{k}` (tym ocenia organizator), na parach (zadanie, odpowiedź) ocenionych "
            f"przez obu (próbka ~10%). Rozjazd = suma |różnic| / suma pkt max; powyżej {zg['prog_rozjazdu']:.0%} to sygnał "
            "ostrzegawczy.", "",
            f"| obszar | pary | śr. różnica bezwzgl. (pkt) | identyczne | śr. pkt {g} | śr. pkt {k} | rozjazd |",
            "|---|---|---|---|---|---|---|"]
    for o, nazwa in obszary.items():
        x = zg[o]
        lin.append(f"| {nazwa} | {x['n']} | {_fmt(x['srednia_roznica'], '{:.2f}')} | {_fmt(x['identyczne'])} | "
                   f"{_fmt(x['srednia_glowny'], '{:.2f}')} | {_fmt(x['srednia_kontrola'], '{:.2f}')} | "
                   f"{_fmt(x['rozjazd'], '{:.1%}')} |")
    ostrz = [f"{nazwa} {zg[o]['rozjazd']:.1%}" for o, nazwa in obszary.items()
             if zg[o]["rozjazd"] is not None and zg[o]["rozjazd"] > zg["prog_rozjazdu"]]
    lin.append("")
    if ostrz:
        lin.append(f"**Sygnał ostrzegawczy:** rozjazd sędziów powyżej {zg['prog_rozjazdu']:.0%} ({', '.join(ostrz)}); "
                   f"wyniki według `{g}` mogą odbiegać od oceny organizatora.")
    elif any(zg[o]["n"] for o in obszary):
        lin.append(f"Rozjazd sędziów nie przekracza {zg['prog_rozjazdu']:.0%}: brak sygnału ostrzegawczego.")
    else:
        lin.append("Brak par ocenionych przez obu sędziów (kontrolny ocenia próbkę ~10% odpowiedzi).")
    return lin


def _sekcja_czasu(cz: dict) -> list[str]:
    e, r = cz["esej_min"], cz["reszta_min"]
    lin = ["", "## Czas", ""]
    if not e + r:
        return lin + ["Brak zmierzonego czasu generacji (pole `sekundy` w plikach odpowiedzi)."]
    lin.append(f"Czas generacji (suma pola `sekundy` z plików odpowiedzi, bez `_kalibracja`; przy generacji równoległej "
               f"większy niż czas zegarowy): esej {e:.1f} min ({e / (e + r):.0%}), reszta {r:.1f} min ({r / (e + r):.0%}).")
    if e != r:
        lin.append(f"Cel nocy ~50/50 (docs/PETLA-NOCNA.md, B): mniej czasu ma obszar {'ESEJ' if e < r else 'RESZTA'}, "
                   "więc następny eksperyment z tego obszaru.")
    return lin


def zapisz(wyn: Path, zadania: list[dict], cfg: dict, cache: dict | None = None) -> Path:
    """RAPORT.md i raport.json w katalogu wyników. cache: oceny sędziów (None = sedzia.wczytaj_cache())."""
    zd = {z["id"]: z for z in zadania}
    rozm = json.loads((wyn / "rozmiary.json").read_text()) if (wyn / "rozmiary.json").exists() else {}
    og = cfg["ogolne"]
    cache = sedzia.wczytaj_cache() if cache is None else cache
    w = _wiersze(wyn, zd, cache, rozm, og["sedzia_model"])
    kal = kalibracja_cke(wyn, zd, cache, og["sedzia_model"])
    zg = zgodnosc(wyn, zd, cache, og["sedzia_model"], og.get("sedzia_kontrola"))
    cz = czas_obszarow(wyn, zd)
    stan = time.strftime("%Y-%m-%d %H:%M")
    f = lambda v, s="{:.0%}": "—" if v is None else s.format(v)
    lin = [f"# RAPORT nocny — stan na {stan}", "",
           "**Co to jest:** wyniki modeli × konfiguracji harnesu na arkuszach CKE z historii (dev 2023–25, test 2026) "
           f"i na sztucznych arkuszach z Wikipedii (synt), ocenione przez {og['sedzia_model']} wg zasad CKE.",
           "**Po co:** zobaczyć lukę do 35% (kategoria „Mały, ale wariat”), wkład harnesu i treningu, wpływ kwantyzacji.",
           "**Co zrobić:** (1) sprawdzić `_kalibracja` (wzorzec ≈100%, pusty ≈0%) — jeśli nie, ocenom nie ufamy; "
           "(2) przeczytać Rekomendację; (3) porównać goly vs h0/h1 per model (to przyrost).", ""]
    kand = [r for r in w if r["model"] != "_kalibracja" and (r["synt"] or 0) >= PROG and (r["test"] or 0) >= PROG]
    lin.append("## Rekomendacja")
    if kand:
        b = min(kand, key=lambda r: (r["mb"], -(r["synt"] or 0)))
        lin.append(f"Najmniejszy zestaw ≥35% na synt i test: **{b['model']} / {b['konfig']}** — {b['mb']:.0f} MB, "
                   f"synt {f(b['synt'])}, test {f(b['test'])}, dev {f(b['dev'])}.")
    else:
        naj = sorted([r for r in w if r["model"] != "_kalibracja" and r["synt"] is not None], key=lambda r: -(r["synt"] or 0))[:3]
        lin.append("Żaden zestaw nie ma jeszcze ≥35% na synt i test. Najbliżej: " +
                   "; ".join(f"{r['model']}/{r['konfig']} synt {f(r['synt'])} test {f(r['test'])}" for r in naj) + ".")
    lin += _sekcja_kalibracji(w, kal)
    lin += ["", "## Wszystkie wiersze (sortowane po rozmiarze)", "",
            "| model | konfig | MB | ocen./odp. | synt | test | dev | zamkn. | otwarte | esej | luka synt do 35% | recall RAG synt | s/zad | błędy |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(w, key=_porzadek):
        luka = (PROG - r["synt"]) * 100 if r["synt"] is not None else None
        lin.append(f"| {r['model']} | {r['konfig']} | {r['mb']:.0f} | {r['ocen']}/{r['odp']} | {f(r['synt'])} | {f(r['test'])} | "
                   f"{f(r['dev'])} | {f(r['zamkniete'])} | {f(r['otwarte'])} | {f(r['esej'])} | {f(luka, '{:+.1f} p.p.')} | "
                   f"{f(r['recall_synt'])} | {f(r['sek'], '{:.1f}')} | {r['bledy']} |")
    lin += ["", "Uwagi: sędzia nie widzi obrazów; zadania ikonograficzne ocenia wg zasad i przykładowego rozwiązania. "
            "„recall RAG synt” = odsetek sztucznych zadań, dla których BM25 znalazł artykuł źródłowy w top-k. "
            "MB = GGUF + mmproj (jeśli wizja włączona)."]
    lin += _sekcja_wypracowan(w) + _sekcja_zgodnosci(zg) + _sekcja_czasu(cz)
    out = wyn / "RAPORT.md"; out.write_text("\n".join(lin) + "\n", encoding="utf-8")
    (wyn / "raport.json").write_text(json.dumps({"wiersze": w, "kalibracja_cke": kal, "zgodnosc": zg, "czas": cz,
                                                 "stan_na": stan}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
