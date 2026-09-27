"""Wynik CAŁEJ matury CKE (cały arkusz, 60 pkt, z wypracowaniem) dla zestawów odpowiedzi modeli.

Co to jest: jedyna metryka projektu. Zestaw = plik odpowiedzi na zadania krótkie + plik wypracowań (może być ten sam).
Zadania krótkie: automat klucza (matura/klucz.py), gdy wynik pewny; resztę ocenia sędzia (matura/sedzia.py, cache
review/oceny_cache.jsonl). Wypracowanie: każdy odpowiedziany temat (<id zadania esejowego>-t1..t3) ocenia sędzia;
punkty eseju arkusza = średnia z ocenionych tematów (do czasu, gdy potok będzie wybierał 1 z 3 tematów).
Po co: porównanie zestawów (model × harness) na pełnym arkuszu, bez ręcznego liczenia.
Co zrobić:
  uv run python -m matura.pelna_matura --wyniki review/noc-2026-09-26 --sesje 2024-maj,2025-maj \\
      --zestaw "bielik-1.5b-q8 | h0=review/noc-2026-09-26/odpowiedzi/bielik-1.5b-q8__h0.jsonl,review/.../x__e1.jsonl"
  → <wyniki>/pelna_matura.json i <wyniki>/PELNA_MATURA.md. `--bez-oceny` liczy tylko z cache (bez sędziego).
Sędzia ocenia na ślepo: pozycje nie niosą etykiety zestawu ani nazwy modelu, zestawy są wymieszane w partiach.
Ponowne uruchomienie dokańcza z cache (partie, które padły, są pomijane i widać je jako „nieocenione").
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import devset, klucz, sedzia
from .noc import ROOT, log, wczytaj_odp

TYPY = ("zamkniete", "otwarte", "esej")


# ---------------------------------------------------------------- wejście

def parsuj_zestaw(opis: str) -> dict:
    """„ETYKIETA=PLIK_KROTKIE,PLIK_ESEJE" → {etykieta, krotkie, eseje}. Rozdział po PIERWSZYM „=" (etykieta może
    mieć spacje i „|"). Jeden plik = ten sam dla krótkich i esejów (np. __goly.jsonl zawiera oba rodzaje)."""
    etykieta, rowna, pliki = opis.partition("=")
    sciezki = [p.strip() for p in pliki.split(",") if p.strip()]
    if not rowna or not etykieta.strip() or not 1 <= len(sciezki) <= 2:
        raise ValueError(f"zły opis zestawu (oczekiwano „ETYKIETA=PLIK_KROTKIE,PLIK_ESEJE”): {opis!r}")
    return {"etykieta": etykieta.strip(), "krotkie": sciezki[0], "eseje": sciezki[-1]}


def wczytaj_zestawy(p: Path) -> list[dict]:
    """Plik zbiorczy: jedna linia = jeden zestaw (jak w --zestaw); puste linie i „#…" pomijane."""
    return [parsuj_zestaw(l.strip()) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def sprawdz_sesje(sesje: list[str], pozwol_test: bool) -> None:
    """Straż testu: sesje 2026 (split test) wolno policzyć tylko raz, na końcu, jawnie (--pozwol-test)."""
    testowe = [s for s in sesje if s[:4].isdigit() and int(s[:4]) >= 2026]
    if testowe and not pozwol_test:
        raise ValueError(f"sesje testowe {testowe}: test 2026 wolno użyć tylko raz, na końcu (dodaj --pozwol-test)")


def pominiete(zid: str, pomin) -> bool:
    """id == p albo id zaczyna się od p + "." (2023-maj-7 pomija 2023-maj-7 i 2023-maj-7.1, ale nie 2023-maj-17)."""
    return any(zid == p or zid.startswith(p + ".") for p in pomin)


def _sciezka(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def zadania_sesji(sesja: str, pomin=()) -> dict:
    """Arkusz sesji → {krotkie: [zadania bez eseju], esej: zadanie esejowe | None, tematy: [...], pominiete: [id]}."""
    wszystkie = devset._cke(sesja, z_esejem=True)
    pom = [z["id"] for z in wszystkie if pominiete(z["id"], pomin)]
    zostaja = [z for z in wszystkie if z["id"] not in pom]
    eseje = [z for z in zostaja if z["esej"]]
    esej = eseje[0] if eseje else None
    tematy = []
    if esej is not None:
        tematy = sorted((t for t in devset.wczytaj(["eseje-cke"]) if t["id"].startswith(esej["id"] + "-t")),
                        key=lambda t: t["id"])
    return {"krotkie": [z for z in zostaja if not z["esej"]], "esej": esej, "tematy": tematy, "pominiete": pom}


def _pozycja(z: dict, odp: str, k: str) -> dict:
    """Pozycja dla `sedzia.ocen_partie`: tylko zadanie i odpowiedź (bez etykiety zestawu i nazwy modelu → ślepo)."""
    return {"klucz": k, "id": z["id"], "pkt_max": z["pkt_max"], "polecenie": devset.tresc_dla_modelu(z),
            "zasady": z["zasady_oceniania"], "rozwiazanie": z["rozwiazanie"], "odpowiedz": odp, "esej": bool(z["esej"])}


def _plan(z: dict, d: dict | None, cache: dict, sedzia_model: str) -> dict:
    """Stan jednego zadania / tematu: brak | blad | puste | automat | sedzia (pkt z cache albo None = nieocenione)."""
    if d is None:
        return {"stan": "brak", "pkt": 0}
    odp = d.get("odpowiedz") or ""
    if odp.lstrip().startswith("(BŁĄD"):
        return {"stan": "blad", "pkt": 0}
    if not odp.strip():
        return {"stan": "puste", "pkt": 0}
    if not z["esej"] and klucz.obslugiwane(z):
        o = klucz.ocen(z, odp)
        if o["pewne"]:
            return {"stan": "automat", "pkt": o["pkt"]}
    k = sedzia.klucz(z["id"], odp, sedzia_model)
    w = cache.get(k)
    return {"stan": "sedzia", "klucz": k, "pozycja": _pozycja(z, odp, k),
            "pkt": None if w is None else max(0, min(int(w["pkt"]), z["pkt_max"]))}


# ---------------------------------------------------------------- sędzia

def partie(pozycje: list[dict], partia: int, partia_esej: int) -> list[list[dict]]:
    """Unikalne pozycje (po kluczu), wymieszane (random.Random(0)), podzielone na partie jednorodne:
    krótkie po `partia`, wypracowania po `partia_esej`. Zestawy mieszają się w partiach (ocena na ślepo)."""
    uniq = sorted({p["klucz"]: p for p in pozycje}.values(), key=lambda p: p["klucz"])
    random.Random(0).shuffle(uniq)
    out = []
    for esej, n in ((False, partia), (True, partia_esej)):
        lista = [p for p in uniq if p["esej"] == esej]
        out += [lista[i:i + max(1, n)] for i in range(0, len(lista), max(1, n))]
    return out


def ocen_brakujace(pozycje: list[dict], sedzia_model: str, cache: dict, rownolegle: int = 6, partia: int = 10,
                   partia_esej: int = 3, timeout: int = 900, ocen_fn=None) -> int:
    """Ocena sędzią wszystkich pozycji bez oceny (równolegle). Wyniki trafiają do cache (plik przez sedzia,
    słownik `cache` tu). Partia, która padła, jest logowana i pomijana. Zwraca liczbę nieudanych partii."""
    ocen_fn = ocen_fn or sedzia.ocen_partie
    pt = partie(pozycje, partia, partia_esej)
    if not pt:
        return 0
    n = sum(len(p) for p in pt)
    ne = sum(len(p) for p in pt if p[0]["esej"])
    log(f"[pełna matura] sędzia {sedzia_model}: {n} pozycji ({n - ne} krótkich, {ne} wypracowań) "
        f"w {len(pt)} partiach, równolegle {rownolegle}")
    gotowe = nieudane = ocenione = 0
    with ThreadPoolExecutor(max(1, rownolegle)) as ex:
        futs = {ex.submit(ocen_fn, p, sedzia_model, timeout): p for p in pt}
        for f in as_completed(futs):
            p = futs[f]
            gotowe += 1
            try:
                wpisy = f.result() or []
            except Exception as e:  # noqa: BLE001 (partia pada: limit, timeout, zły JSON → dokończy następny przebieg)
                nieudane += 1
                log(f"[pełna matura] partia {gotowe}/{len(pt)} NIEUDANA ({len(p)} poz.): {str(e)[:300]}")
                continue
            for w in wpisy:
                cache[w["klucz"]] = w
            ocenione += len(wpisy)
            log(f"[pełna matura] partia {gotowe}/{len(pt)} gotowa ({len(wpisy)}/{len(p)} "
                f"{'wypracowań' if p[0]['esej'] else 'krótkich'}); ocenionych {ocenione}/{n}")
    if nieudane:
        log(f"[pełna matura] nieudanych partii: {nieudane}; uruchom ponownie, żeby dokończyć z cache")
    return nieudane


# ---------------------------------------------------------------- liczenie

def _l(x: float) -> float | int:
    """Liczba do raportu: całkowita bez „.0", inaczej 1 miejsce po przecinku."""
    x = round(x, 1)
    return int(x) if x == int(x) else x


def _rozmiar_mb(etykieta: str, krotkie: str, rozmiary: dict) -> float | None:
    """Model = część etykiety przed spacją; gdy nie ma jej w rozmiary.json, nazwa z pliku krótkich (<model>__<konfig>)."""
    for m in (etykieta.split(" ")[0], Path(krotkie).stem.split("__")[0]):
        if m in rozmiary and rozmiary[m].get("razem"):
            return round(rozmiary[m]["razem"] / 1e6, 1)
    return None


def _sesja(arkusz: dict, krotkie: dict, eseje: dict, cache: dict, sedzia_model: str) -> tuple[dict, list[dict]]:
    """Plany wszystkich zadań jednej sesji dla jednego zestawu → (plany, pozycje bez oceny sędziego)."""
    plany = {"krotkie": [(z, _plan(z, krotkie.get(z["id"]), cache, sedzia_model)) for z in arkusz["krotkie"]],
             "tematy": [(t, _plan(t, eseje.get(t["id"]), cache, sedzia_model)) for t in arkusz["tematy"]]}
    brak = [p["pozycja"] for _, p in plany["krotkie"] + plany["tematy"] if p["stan"] == "sedzia" and p["pkt"] is None]
    return plany, brak


def _odswiez(plany: dict, cache: dict) -> None:
    """Po ocenie sędziego: dociągnij punkty z cache do planów, które ich nie miały."""
    for z, p in plany["krotkie"] + plany["tematy"]:
        if p["stan"] == "sedzia" and p["pkt"] is None and p["klucz"] in cache:
            p["pkt"] = max(0, min(int(cache[p["klucz"]]["pkt"]), z["pkt_max"]))


def _ma_ilustracje(z: dict) -> bool:
    """Ilustracja przypisana do zadania (matura/obrazy.py: układ PDF arkusza), a nie cała strona z `z["obrazy"]`."""
    try:
        from .obrazy import obrazy_zadania
        return bool(obrazy_zadania(z))
    except Exception:  # noqa: BLE001 (brak PDF / zależności: zapas, zawyżone przypisanie per strona)
        return bool(z["obrazy"])


def podsumuj_sesje(arkusz: dict, plany: dict) -> dict:
    """Plany jednej sesji → wynik: punkty wg typu zadania, esej = średnia z ocenionych tematów, liczniki."""
    pkt = {t: 0.0 for t in TYPY}
    mx = {t: 0 for t in TYPY}
    il = [0.0, 0]
    licz = {"braki": 0, "bledy": 0, "puste": 0, "nieocenione": 0}
    zrodla = {"automat": 0, "sedzia": 0}
    for z, p in plany["krotkie"]:
        t = z["typ"] if z["typ"] in TYPY[:2] else "otwarte"
        mx[t] += z["pkt_max"]
        pk = p["pkt"] or 0
        pkt[t] += pk
        if _ma_ilustracje(z):
            il[0] += pk; il[1] += z["pkt_max"]
        if p["stan"] == "brak":
            licz["braki"] += 1
        elif p["stan"] == "blad":
            licz["bledy"] += 1
        elif p["stan"] == "puste":
            licz["puste"] += 1
        elif p["stan"] == "automat":
            zrodla["automat"] += 1
        elif p["pkt"] is None:
            licz["nieocenione"] += 1
        else:
            zrodla["sedzia"] += 1
    tematy, ocenione = {}, []
    for t, p in plany["tematy"]:
        if p["stan"] in ("brak", "blad"):
            tematy[t["id"]] = {"pkt": None, "stan": p["stan"]}
            licz["bledy"] += p["stan"] == "blad"
        elif p["stan"] == "puste":
            tematy[t["id"]] = {"pkt": 0, "stan": "puste"}; ocenione.append(0)
        elif p["pkt"] is None:
            tematy[t["id"]] = {"pkt": None, "stan": "nieoceniony"}; licz["nieocenione"] += 1
        else:
            tematy[t["id"]] = {"pkt": p["pkt"], "stan": "oceniony"}; ocenione.append(p["pkt"])
            zrodla["sedzia"] += 1
    brak_eseju = False
    if arkusz["esej"] is not None:
        mx["esej"] = arkusz["esej"]["pkt_max"]
        pkt["esej"] = round(sum(ocenione) / len(ocenione), 1) if ocenione else 0.0
        brak_eseju = all(v["stan"] in ("brak", "blad") for v in tematy.values())
    razem = [sum(pkt.values()), sum(mx.values())]
    return {**{t: [_l(pkt[t]), mx[t]] for t in TYPY}, "z_ilustracja": [_l(il[0]), il[1]],
            "razem": [_l(razem[0]), razem[1]], "proc": _l(100 * razem[0] / razem[1]) if razem[1] else None,
            **licz, "brak_eseju": brak_eseju, "eseje_tematy": tematy, "zrodla": zrodla, "pominiete": arkusz["pominiete"]}


def policz(zestawy: list[dict], sesje: list[str], wyniki: Path, sedzia_model: str = "claude:opus", pomin=(),
           bez_oceny: bool = False, rownolegle: int = 6, partia: int = 10, partia_esej: int = 3,
           pozwol_test: bool = False, timeout: int = 900, ocen_fn=None) -> dict:
    """Pełna matura dla zestawów × sesji. Brakujące oceny sędziego zbierane ze WSZYSTKICH zestawów naraz."""
    sprawdz_sesje(sesje, pozwol_test)
    pomin = [p for p in pomin if p]
    arkusze = {s: zadania_sesji(s, pomin) for s in sesje}
    for s, a in arkusze.items():
        log(f"[pełna matura] {s}: {len(a['krotkie'])} zadań krótkich, esej {a['esej']['id'] if a['esej'] else 'brak'} "
            f"({len(a['tematy'])} tematy), max {sum(z['pkt_max'] for z in a['krotkie']) + (a['esej'] or {}).get('pkt_max', 0)} pkt"
            + (f", pominięte {a['pominiete']}" if a["pominiete"] else ""))
    cache = sedzia.wczytaj_cache()
    pliki: dict[str, dict] = {}
    odp = lambda s: pliki.setdefault(s, wczytaj_odp(_sciezka(s)))  # noqa: E731
    plany, do_oceny = {}, []
    for i, zs in enumerate(zestawy):
        for s, a in arkusze.items():
            plany[i, s], brak = _sesja(a, odp(zs["krotkie"]), odp(zs["eseje"]), cache, sedzia_model)
            do_oceny += brak
    n_brak = len({p["klucz"] for p in do_oceny})
    log(f"[pełna matura] {len(zestawy)} zestawów × {len(sesje)} sesji; bez oceny sędziego: {n_brak} pozycji")
    if do_oceny and not bez_oceny:
        ocen_brakujace(do_oceny, sedzia_model, cache, rownolegle, partia, partia_esej, timeout, ocen_fn)
        for pl in plany.values():
            _odswiez(pl, cache)
    elif do_oceny:
        log(f"[pełna matura] --bez-oceny: {n_brak} pozycji liczonych jako 0 (nieocenione)")
    rozmiary_p = wyniki / "rozmiary.json"
    rozmiary = json.loads(rozmiary_p.read_text(encoding="utf-8")) if rozmiary_p.exists() else {}
    out = []
    for i, zs in enumerate(zestawy):
        wyn = {s: podsumuj_sesje(arkusze[s], plany[i, s]) for s in sesje}
        procenty = [100 * w["razem"][0] / w["razem"][1] for w in wyn.values() if w["razem"][1]]
        out.append({"etykieta": zs["etykieta"], "pliki": {"krotkie": zs["krotkie"], "eseje": zs["eseje"]},
                    "rozmiar_mb": _rozmiar_mb(zs["etykieta"], zs["krotkie"], rozmiary), "sesje": wyn,
                    "srednio_proc": _l(sum(procenty) / len(procenty)) if procenty else None})
    return {"co_to_jest": "Wynik całego arkusza CKE (z wypracowaniem) dla zestawów odpowiedzi (matura/pelna_matura.py).",
            "po_co": "Jedyna metryka projektu: porównanie zestawów model × harness na pełnej maturze.",
            "co_zrobic": "Porównać srednio_proc; nieocenione > 0 → uruchomić ponownie bez --bez-oceny (dokańcza z cache).",
            "meta": {"sedzia": sedzia_model, "data": time.strftime("%Y-%m-%d %H:%M"), "sesje": sesje, "pomin": pomin,
                     "bez_oceny": bez_oceny,
                     "uwagi": "Zadania krótkie: automat klucza, gdy pewny, inaczej sędzia. Esej arkusza = średnia z "
                              "ocenionych tematów (do czasu wyboru 1 z 3 w potoku). Typ zadania (zamknięte/otwarte) "
                              "wg devset.typ_zadania. z_ilustracja wg przypisania z układu PDF (matura/obrazy.py)."},
            "zestawy": out}


# ---------------------------------------------------------------- raport

def _pm(x: list) -> str:
    return f"{x[0]}/{x[1]}"


def _suma(zs: dict, pole: str) -> list:
    return [_l(sum(w[pole][0] for w in zs["sesje"].values())), sum(w[pole][1] for w in zs["sesje"].values())]


def md(wynik: dict) -> str:
    m = wynik["meta"]
    sesje = m["sesje"]
    L = ["# Pełna matura CKE (cały arkusz z wypracowaniem)", "",
         "**Co to jest:** wynik całego arkusza CKE z historii (zadania krótkie + wypracowanie) dla każdego zestawu "
         "odpowiedzi (model × harness); matura/pelna_matura.py.",
         "**Po co:** jedyna metryka projektu; porównanie zestawów i wybór modelu na finał.",
         "**Co zrobić:** porównać „średnio %”; gdy „nieocenione” > 0, uruchomić ponownie bez `--bez-oceny` "
         "(dokańcza z cache); JSON ze szczegółami obok.", "",
         f"Sędzia: `{m['sedzia']}`" + (" (**--bez-oceny**: tylko cache, nieocenione = 0)" if m["bez_oceny"] else "")
         + f" · data: {m['data']} · sesje: {', '.join(sesje)}"
         + (f" · pominięte: {', '.join(m['pomin'])}" if m["pomin"] else ""), "",
         "| zestaw | MB | " + " | ".join(f"{s} pkt/max" for s in sesje)
         + " | średnio % | zamknięte | otwarte | esej | z ilustracją* | braki / błędy / nieocenione |",
         "|---" * (8 + len(sesje)) + "|"]
    klucz_sort = lambda z: (z["rozmiar_mb"] is None, z["rozmiar_mb"] or 0, z["etykieta"])  # noqa: E731
    for zs in sorted(wynik["zestawy"], key=klucz_sort):
        w = zs["sesje"].values()
        flaga = " (brak eseju)" if any(x["brak_eseju"] for x in w) else ""
        L.append(f"| {zs['etykieta'].replace('|', '/')} | {zs['rozmiar_mb'] if zs['rozmiar_mb'] is not None else '?'} | "
                 + " | ".join(_pm(zs["sesje"][s]["razem"]) for s in sesje)
                 + f" | **{zs['srednio_proc']}** | {_pm(_suma(zs, 'zamkniete'))} | {_pm(_suma(zs, 'otwarte'))} | "
                 f"{_pm(_suma(zs, 'esej'))}{flaga} | {_pm(_suma(zs, 'z_ilustracja'))} | "
                 f"{sum(x['braki'] for x in w)} / {sum(x['bledy'] for x in w)} / {sum(x['nieocenione'] for x in w)} |")
    L += ["", "\\* „z ilustracją”: obrazy przypisane do zadań z układu PDF arkusza (matura/obrazy.py).",
          "Kolumny zamknięte/otwarte/esej/z ilustracją sumują wszystkie sesje. Esej arkusza = średnia z ocenionych "
          "tematów (t1..t3), do czasu wyboru 1 z 3 w potoku.", "", "## Szczegóły", ""]
    for zs in sorted(wynik["zestawy"], key=klucz_sort):
        L.append(f"- **{zs['etykieta']}** (krótkie: `{zs['pliki']['krotkie']}`, eseje: `{zs['pliki']['eseje']}`)")
        for s, x in zs["sesje"].items():
            tem = ", ".join(f"{k.rsplit('-', 1)[1]}={v['pkt'] if v['pkt'] is not None else v['stan']}"
                            for k, v in x["eseje_tematy"].items())
            L.append(f"  - {s}: {_pm(x['razem'])} ({x['proc']}%); zamknięte {_pm(x['zamkniete'])}, otwarte "
                     f"{_pm(x['otwarte'])}, esej {_pm(x['esej'])} [{tem}]; automat {x['zrodla']['automat']}, "
                     f"sędzia {x['zrodla']['sedzia']}; braki {x['braki']}, błędy {x['bledy']}, puste {x['puste']}, "
                     f"nieocenione {x['nieocenione']}")
    return "\n".join(L) + "\n"


def zapisz(wynik: dict, wyniki: Path, nazwa: str = "pelna_matura") -> tuple[Path, Path]:
    wyniki.mkdir(parents=True, exist_ok=True)
    pj, pm = wyniki / f"{nazwa}.json", wyniki / f"{nazwa.upper()}.md"
    pj.write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding="utf-8")
    pm.write_text(md(wynik), encoding="utf-8")
    return pj, pm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wynik całej matury CKE (60 pkt z wypracowaniem) dla zestawów odpowiedzi.")
    ap.add_argument("--wyniki", required=True, help="katalog wyników (rozmiary.json; tu trafia raport)")
    ap.add_argument("--sesje", required=True, help="np. 2024-maj,2025-maj")
    ap.add_argument("--zestaw", action="append", default=[], help="„ETYKIETA=PLIK_KROTKIE,PLIK_ESEJE” (wielokrotnie)")
    ap.add_argument("--zestawy", help="plik: jedna linia = jeden zestaw w formacie --zestaw")
    ap.add_argument("--sedzia", default="claude:opus")
    ap.add_argument("--rownolegle", type=int, default=6)
    ap.add_argument("--partia", type=int, default=10)
    ap.add_argument("--partia-esej", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900, help="sekundy na jedną partię sędziego")
    ap.add_argument("--pomin", default="", help="id zadań do pominięcia (z podzadaniami), np. 2023-maj-7,2023-maj-8")
    ap.add_argument("--bez-oceny", action="store_true", help="nie wołaj sędziego, licz z cache")
    ap.add_argument("--nazwa", default="pelna_matura")
    ap.add_argument("--pozwol-test", action="store_true", help="zgoda na sesje 2026 (test, tylko raz na końcu)")
    a = ap.parse_args(argv)
    try:
        zestawy = [parsuj_zestaw(z) for z in a.zestaw] + (wczytaj_zestawy(_sciezka(a.zestawy)) if a.zestawy else [])
        sesje = [s.strip() for s in a.sesje.split(",") if s.strip()]
        sprawdz_sesje(sesje, a.pozwol_test)
        nieznane = [s for s in sesje if not (devset.JSON / f"historia-{s}.json").exists()]
        if nieznane:
            raise ValueError(f"nieznane sesje (brak arkusza w {devset.JSON}): {nieznane}")
    except ValueError as e:
        print(f"BŁĄD: {e}", file=sys.stderr)
        return 2
    if not zestawy:
        print("BŁĄD: brak zestawów (--zestaw albo --zestawy)", file=sys.stderr)
        return 2
    for zs in zestawy:
        for k in ("krotkie", "eseje"):
            if not _sciezka(zs[k]).exists():
                log(f"[pełna matura] UWAGA: brak pliku {zs[k]} (zestaw {zs['etykieta']}): wszystko liczone jako braki")
    wyniki = _sciezka(a.wyniki)
    wynik = policz(zestawy, sesje, wyniki, a.sedzia, pomin=[p.strip() for p in a.pomin.split(",")],
                   bez_oceny=a.bez_oceny, rownolegle=a.rownolegle, partia=a.partia, partia_esej=a.partia_esej,
                   pozwol_test=a.pozwol_test, timeout=a.timeout)
    pj, pm = zapisz(wynik, wyniki, a.nazwa)
    print(md(wynik))
    log(f"[pełna matura] zapisano {pj.relative_to(ROOT) if pj.is_relative_to(ROOT) else pj} i {pm.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
