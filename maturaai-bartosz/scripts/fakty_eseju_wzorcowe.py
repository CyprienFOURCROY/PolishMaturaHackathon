"""Fakty wzorcowe do wypracowania (sufit eseju, krok 1b planu): Claude wypisuje fakty do tematu, model pisze esej.

Co to jest: dla każdego tematu wypracowania z podanych sesji (<id zadania esejowego>-t1..t3) Claude (Opus, --effort low)
dostaje sam temat i wypisuje 10-16 konkretnych faktów (daty, postaci, wydarzenia, pojęcia, przyczyny i skutki),
pogrupowanych pod człony tematu (trzy aspekty albo trzy wybrane przykłady) i dotyczących obu członów tezy (jej podmiotu
i tego, z czym teza go porównuje), BEZ gotowego eseju: bez tezy, bez oceny, bez zdań gotowych do wklejenia jako akapit.
Cache: data/wiedza/fakty_eseju_wzorcowe.jsonl ({id, temat, grupy: [{czlon, fakty: [{fakt, data}]}], uwagi}), poza
gitem, wznawialny. Druga część pliku rejestruje w czasie uruchomienia konfigurację `e5_fakty` (= e5_zgadzam, ale
materiałem akapitu są fakty wzorcowe jego członu zamiast kart, Wikipedii i bazy) oraz drugie generacje `<konfig>_g2`
dowolnej konfiguracji (ten sam potok, osobny plik odpowiedzi), bez zmiany harnessu, i uruchamia matura.noc.
Po co: pokazuje, ile punktów za wypracowanie daje modelowi IDEALNY materiał faktograficzny (decyzja o kroku 4 planu:
baza akapitów na 50 działów, dopracowany e7), zanim zespół zbuduje lokalną bazę.
UWAGA: TYLKO pomiar sufitu na dev w fazie budowy. Nigdy egzamin ani trening (na egzaminie zamknięte API są zakazane;
tematy dev nie mogą trafić do danych treningowych).
Co zrobić:
  1) uv run python scripts/fakty_eseju_wzorcowe.py --sesje 2024-maj,2025-maj [--rownolegle 3] [--effort low]
  2) uv run python scripts/fakty_eseju_wzorcowe.py --noc noc/esej_sufit_a.toml --bez-oceny   (reszta argumentów → noc)
  3) ocena: matura.pelna_matura --zestawy review/esej-sufit-2026-09-26/pm_zestawy.txt (README w katalogu wyników)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from matura import esej  # noqa: E402

CACHE = ROOT / "data" / "wiedza" / "fakty_eseju_wzorcowe.jsonl"
ZRODLO = "claude-opus --effort low, scripts/fakty_eseju_wzorcowe.py (tylko pomiar sufitu, nigdy egzamin ani trening)"
KONFIG = "e5_fakty"            # e5_zgadzam z faktami wzorcowymi jako materiałem
STANOWISKO = "zgadzam"         # jak domyślne e5_zgadzam (matura/egzamin.py)
MIN_FAKTOW, MAX_FAKTOW, MAX_SLOW_FAKTU = 10, 16, 30
# słowa rozstrzygające tezę albo typowe dla gotowego akapitu wypracowania (fakt ma być notatką, nie argumentem)
RE_ZAKAZANE = re.compile(r"\b(?:tez[aąeęy]|tezie|potwierdza\w*|przecz[ąy]|dowodz[iąy]|dowodem|świadcz(?:y|ą|ył\w*)|"
                         r"po (?:pierwsze|drugie|trzecie)|zgadzam|uważam|podsumowując)\b", re.I)

PROMPT = """Przygotowujesz materiał faktograficzny do wypracowania maturalnego z historii (poziom rozszerzony, CKE)
dla ucznia, który sam napisze wypracowanie. Temat wypracowania:
{temat}

Wypisz od 10 do 16 konkretnych, pewnych faktów (daty, postaci, wydarzenia, pojęcia, przyczyny i skutki) potrzebnych
do napisania tego wypracowania, pogrupowanych w dokładnie trzy grupy: {grupy}. W każdej grupie 3-6 faktów.
Fakty mają dotyczyć obu członów tezy: jej podmiotu oraz tego, z czym teza go porównuje albo co może jej przeczyć
(inne postaci, państwa, konflikty lub zjawiska, ograniczenia, skutki odwrotne), tak żeby z tych samych faktów dało się
uzasadnić zarówno stanowisko zgodne z tezą, jak i przeciwne.
Zasady:
- Nie rozstrzygaj tezy: nie oceniaj i nie pisz, czy fakt ją potwierdza albo jej przeczy; nie używaj słów „teza”,
  „potwierdza”, „przeczy”, „dowodzi”, „świadczy”.
- Każdy fakt to jedno krótkie zdanie informacyjne (najwyżej 25 słów) w stylu notatki z podręcznika, a nie zdanie
  wypracowania: bez wstępu, bez „Po pierwsze”, bez wniosków i bez odniesień do tematu.
- Zawsze pełna nazwa podmiotu (bez „on”, „ten władca”).
- Pole „data”: rok albo zakres lat (np. „1410”, „1386-1434”), pusty napis, gdy fakt nie ma daty. Datę podaj tylko,
  jeśli jesteś jej pewien. Tylko fakty pewne, po polsku.
Zwróć wyłącznie JSON."""
_s = {"type": "string"}
_lock = threading.Lock()


def czlony(temat: str) -> tuple[str, list[str], str | None]:
    """Temat → (teza, aspekty, wybor) jak w potoku eseju (esej.rozbierz)."""
    return esej.rozbierz(temat)


def opis_grup(temat: str) -> str:
    _, aspekty, wybor = czlony(temat)
    if wybor:
        return f"trzy wybrane przykłady ({wybor}); pole „czlon” = sama nazwa przykładu (np. nazwa państwa albo postaci)"
    return ("aspekty " + ", ".join(f"„{a}”" for a in aspekty)
            + "; pole „czlon” = dokładnie nazwa aspektu z tej listy")


def schemat(temat: str) -> dict:
    """JSON schema odpowiedzi; przy aspektach pole `czlon` ograniczone do nazw aspektów tematu."""
    _, aspekty, wybor = czlony(temat)
    czlon = _s if wybor else {"type": "string", "enum": aspekty}
    fakt = {"type": "object", "additionalProperties": False, "required": ["fakt", "data"],
            "properties": {"fakt": _s, "data": _s}}
    grupa = {"type": "object", "additionalProperties": False, "required": ["czlon", "fakty"],
             "properties": {"czlon": czlon, "fakty": {"type": "array", "items": fakt}}}
    return {"type": "object", "additionalProperties": False, "required": ["grupy"],
            "properties": {"grupy": {"type": "array", "items": grupa}}}


def sprawdz(grupy: list[dict], temat: str) -> list[str]:
    """Problemy odpowiedzi Claude (pusta lista = poprawna): liczba grup i faktów, nazwy członów, długość faktu,
    słowa rozstrzygające tezę lub typowe dla gotowego akapitu."""
    _, aspekty, wybor = czlony(temat)
    uw = []
    n = sum(len(g["fakty"]) for g in grupy)
    if len(grupy) != 3:
        uw.append(f"grup {len(grupy)} zamiast 3")
    if not MIN_FAKTOW <= n <= MAX_FAKTOW:
        uw.append(f"faktów {n} poza {MIN_FAKTOW}-{MAX_FAKTOW}")
    if not wybor and sorted(g["czlon"] for g in grupy) != sorted(aspekty):
        uw.append(f"człony {[g['czlon'] for g in grupy]} zamiast {aspekty}")
    for g in grupy:
        if len(g["fakty"]) < 3:
            uw.append(f"grupa {g['czlon']!r}: {len(g['fakty'])} fakty")
        for f in g["fakty"]:
            if len(f["fakt"].split()) > MAX_SLOW_FAKTU:
                uw.append(f"za długi fakt ({len(f['fakt'].split())} słów): {f['fakt'][:60]}")
            m = RE_ZAKAZANE.search(f["fakt"])
            if m:
                uw.append(f"słowo {m.group(0)!r}: {f['fakt'][:60]}")
    return uw


def tematy_sesji(sesje: list[str]) -> list[dict]:
    from matura import pelna_matura
    return [t for s in sesje for t in pelna_matura.zadania_sesji(s)["tematy"]]


def wczytaj(p: Path = CACHE) -> dict[str, list[dict]]:
    """Cache → {id tematu: grupy}; przy powtórzonym id wygrywa ostatni wpis."""
    if not p.exists():
        return {}
    return {d["id"]: d["grupy"] for d in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def fakty_tematu(t: dict, effort: str | None = "low", proby: int = 2, claude_fn=None) -> dict:
    """Jeden temat → wpis cache. Do `proby` wywołań; wygrywa odpowiedź z najmniejszą liczbą problemów."""
    if claude_fn is None:
        from buduj_akapity_eseju import claude as claude_fn
    temat = t.get("temat") or t["polecenie"]
    najl = None
    for _ in range(max(1, proby)):
        grupy = claude_fn(PROMPT.format(temat=temat, grupy=opis_grup(temat)), schemat(temat), effort)["grupy"]
        uw = sprawdz(grupy, temat)
        if najl is None or len(uw) < len(najl[1]):
            najl = (grupy, uw)
        if not uw:
            break
    return {"id": t["id"], "temat": temat, "grupy": najl[0], "uwagi": najl[1], "zrodlo": ZRODLO}


# ---------------------------------------------------------------- materiał akapitu i rejestracja konfiguracji

def _rdzenie(t: str) -> set[str]:
    return {w.lower()[:5] for w in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", t) if len(w) >= 4 or w.isupper()}


def grupa_dla(grupy: list[dict], pozycja: str) -> dict | None:
    """Grupa faktów dla aspektu albo przykładu akapitu: ta sama nazwa albo wspólny rdzeń słowa (Czechosłowacja /
    Czechosłowen, Związek Radziecki / Związku Radzieckiego); brak → None."""
    for g in grupy:
        if g["czlon"].strip().lower() == pozycja.strip().lower():
            return g
    rp = _rdzenie(pozycja)
    traf = [g for g in grupy if rp & _rdzenie(g["czlon"])]
    return traf[0] if len(traf) == 1 else None


def _zdanie_faktu(f: dict) -> str:
    t = f["fakt"].strip().rstrip(".")
    lata = re.findall(r"\d{3,4}", f.get("data") or "")
    return t + (f" ({f['data']})" if lata and not all(r in t for r in lata) else "") + "."


def material_fakty(grupy: list[dict], wiki, karty, teza: str, pozycja: str, element: bool = False, **_) -> dict:
    """Zamiennik `esej.material_e5` (ta sama sygnatura, wiki i karty pomijane): fakty grupy członu akapitu ciągłym
    tekstem; krok wyboru przykładów i przykład spoza grup dostają fakty wszystkich grup z nazwą członu.
    `fakty` (dla weryfikatora dat i uzupełnienia przykładów) mają pola fakt, data, tytul/postac = nazwa członu."""
    g = grupa_dla(grupy, pozycja)
    wybrane = [g] if g is not None else grupy
    fakty = [{"fakt": f["fakt"], "data": f.get("data") or "", "tytul": x["czlon"], "postac": x["czlon"] if element else ""}
             for x in wybrane for f in x["fakty"]]
    if g is not None:
        tekst = "FAKTY PEWNE (wykorzystaj je, ale pisz o tezie tematu): " + " ".join(_zdanie_faktu(f) for f in g["fakty"])
    else:
        tekst = "\n\n".join(f"FAKTY PEWNE ({x['czlon']}): " + " ".join(_zdanie_faktu(f) for f in x["fakty"])
                            for x in grupy)
    return {"tekst": tekst, "fakty": fakty, "fragmenty": [], "z_bazy": []}


RE_GENERACJA = re.compile(r"^(?P<baza>.+)_g(?P<n>[2-9])$")


def konfig_bazowy(konfig: str) -> str:
    """e5_zgadzam_g2 → e5_zgadzam (druga generacja: ten sam potok, osobny plik odpowiedzi)."""
    m = RE_GENERACJA.match(konfig)
    return m.group("baza") if m else konfig


def napisz_z_faktami(z: dict, *, url: str, bez_myslenia: bool = True, fakty: dict | None = None, czat_fn=None,
                     **_) -> tuple[str, float, dict]:
    fakty = wczytaj() if fakty is None else fakty
    if z["id"] not in fakty:
        raise KeyError(f"brak faktów wzorcowych dla {z['id']} (uruchom scripts/fakty_eseju_wzorcowe.py)")
    kw = {"czat_fn": czat_fn} if czat_fn else {}
    t, s, meta = esej.napisz_e5(z, url=url, bez_myslenia=bez_myslenia, stanowisko=STANOWISKO,
                                material_fn=partial(material_fakty, fakty[z["id"]]), **kw)
    return t, s, {**meta, "fakty_wzorcowe": sum(len(g["fakty"]) for g in fakty[z["id"]])}


def rejestruj(konfiguracje: list[str]) -> None:
    """Rejestracja w czasie uruchomienia (jak scripts/pomiar_wzorzec.py): `e5_fakty` i `<konfig>_g<N>` trafiają do
    zbiorów noc (Wikipedia, karty) i do `harness.odpowiedz` przez nakładkę; harness egzaminacyjny bez zmian."""
    from matura import harness, noc
    fakty = wczytaj()
    for k in konfiguracje:
        b = konfig_bazowy(k)
        if b in noc.KONFIGI_Z_WIKI:
            noc.KONFIGI_Z_WIKI.add(k)
        if b in noc.KONFIGI_Z_KARTAMI:
            noc.KONFIGI_Z_KARTAMI.add(k)
    oryginal = harness.odpowiedz

    def odpowiedz(konfig: str, z: dict, **kw):
        b = konfig_bazowy(konfig)
        if b == KONFIG:
            if not z["esej"]:
                raise ValueError(f"{konfig} tylko dla wypracowań, a {z['id']} to zadanie krótkie")
            return napisz_z_faktami(z, url=kw["url"], bez_myslenia=kw.get("bez_myslenia", True), fakty=fakty)
        return oryginal(b, z, **kw)

    harness.odpowiedz = odpowiedz


def uruchom_noc(argv: list[str]) -> int:
    """--noc CONFIG [argumenty matura.noc] → rejestracja konfiguracji z pliku i matura.noc.main()."""
    from matura import noc
    i = argv.index("--noc")
    config, reszta = argv[i + 1], argv[:i] + argv[i + 2:]
    konf = [k for m in noc.wczytaj_config(config)["model"] for k in m["konfiguracje"]]
    if any(konfig_bazowy(k) == KONFIG for k in konf):
        brak = [t["id"] for t in tematy_sesji(["2024-maj", "2025-maj"]) if t["id"] not in wczytaj()]
        if brak:
            print(f"UWAGA: brak faktów wzorcowych dla {brak} (te eseje dostaną błąd)", flush=True)
    rejestruj(konf)
    sys.argv = ["matura.noc", "--config", config] + reszta
    return noc.main()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--noc" in argv:
        return uruchom_noc(argv)
    ap = argparse.ArgumentParser(description="Fakty wzorcowe do tematów wypracowań (tylko pomiar sufitu).")
    ap.add_argument("--sesje", default="2024-maj,2025-maj")
    ap.add_argument("--rownolegle", type=int, default=3)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--ponow", default="", help="id tematów do wygenerowania od nowa (po przecinku)")
    a = ap.parse_args(argv)
    ponow = {x.strip() for x in a.ponow.split(",") if x.strip()}
    gotowe = set(wczytaj()) - ponow
    todo = [t for t in tematy_sesji([s.strip() for s in a.sesje.split(",") if s.strip()]) if t["id"] not in gotowe]
    print(f"tematów do opisania: {len(todo)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    bledy = []

    def jeden(t):
        try:
            w = fakty_tematu(t, a.effort)
        except Exception as e:  # noqa: BLE001 (limit, timeout, zły JSON): temat do ponowienia
            bledy.append(t["id"]); print(f"{t['id']}: BŁĄD {str(e)[:300]}", flush=True); return
        with _lock, CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"{t['id']}: {sum(len(g['fakty']) for g in w['grupy'])} faktów w {len(w['grupy'])} grupach"
              + (f"; UWAGI: {w['uwagi']}" if w["uwagi"] else ""), flush=True)

    with ThreadPoolExecutor(max(1, a.rownolegle)) as ex:
        list(ex.map(jeden, todo))
    print(f"koniec; błędy: {bledy}", flush=True)
    return 1 if bledy else 0


if __name__ == "__main__":
    sys.exit(main())
