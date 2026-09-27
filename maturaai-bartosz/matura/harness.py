"""Konfiguracje odpowiadania na zadanie: (konfig, zadanie) → (odpowiedź, sekundy, meta).

Konfiguracje (ablacje; każda to jedna zmiana względem poprzedniej):
- goly      baseline: minimalny prompt, bez wiedzy zewnętrznej (tak liczy się „goły model").
- h0_norag  prompt wg typu zadania (zamknięte / otwarte / rozstrzygnij / esej), bez Wikipedii.
- h0        h0_norag + fragmenty Wikipedii (BM25, k=4; esej k=8).
- h1        h0 + 2 przykłady few-shot tego samego typu z CKE 2020–2022 (pula treningowa, bez wycieku).
- h0_en     h0 + rozumowanie po angielsku, odpowiedź po polsku (D12, wariant b).
- h0_ocr    h0 + blok „OPIS ILUSTRACJI” (napisy z EasyOCR) przed „ZADANIE:” dla zadań z ilustracją (matura/obrazy.py,
            poprawione przypisanie obrazów); model NIE dostaje obrazu. Zadanie bez ilustracji: prompt identyczny z h0.
- h0_opis   jak h0_ocr, ale blok = napisy OCR + opis sceny z Qwen3.5-0.8B (wariant "ocr_opis").
            Oba czytają tylko cache data/cke/obrazy_opisy.jsonl (brak wpisu → błąd; policz: python -m matura.obrazy).
- h2        h0 + reguły odpowiedzi w prompcie systemowym zależnie od typu zadania (`system_h2`): zamknięte (zawsze
            wybierz, tylko oznaczenia), rozstrzygnij/uzasadnij (element źródła + fakt z wiedzy), podaj/wymień/wyjaśnij
            (krótko), liczba elementów z polecenia, ilustracja niewidoczna dla modelu (podpis, tytuł, data, wiedza).
            Wikipedia k=4 jak w h0 (jedna zmiana względem h0). Wymaga indeksu Wikipedii (wiki=None → błąd).
- h2_ocr    h2 + blok „OPIS ILUSTRACJI” jak w h0_ocr (model nie dostaje obrazu); bez ilustracji prompt = h2.
- wzorzec / pusty  kalibracja sędziego (rozwiązanie CKE → ~100%, „Nie wiem." → ~0%).
- kalibracja_cke   wypracowanie z Informatora CKE → porównanie oceny sędziego z oceną CKE.
Wypracowania: e0 (swobodny), e1–e4, e3_en — matura/esej.py. Konfiguracje krótkie na eseju działają jak e0.
e5 / e5_zgadzam: `esej.napisz_e5` ze stanowiskiem „nie zgadzam się” / „zgadzam się”; bez przekazanych kart
harness wczytuje karty kanoniczne sam (noc.py nie ładuje ich dla e5).
Lekcja z testu dymnego 2026-09-25: dosłowne przykłady formatu w prompcie systemowym model 0.8B kopiuje
do każdej odpowiedzi → prompty rozdzielone per typ, bez przykładowych odpowiedzi w instrukcji.
"""
from __future__ import annotations

import random
import re
import threading

from . import esej as esej_mod
from .devset import pula_treningowa, tresc_dla_modelu, typ_zadania
from .llm import czat

KONFIGI_ESEJ = ("e0", "e1", "e2", "e3", "e4", "e3_en", "e5", "e5_zgadzam", "e6", "e6_zgadzam", "e7", "e7_nie")
KONFIGI_E5 = {"e5": "nie_zgadzam", "e5_zgadzam": "zgadzam"}          # konfiguracja → stanowisko esej.napisz_e5
KONFIGI_E6 = {"e6": "nie_zgadzam", "e6_zgadzam": "zgadzam"}          # akapity z bazy wiedzy (esej.napisz_e6)
KONFIGI_E7 = {"e7": "zgadzam", "e7_nie": "nie_zgadzam"}             # e5 z akapitami bazy jako materiałem
KONFIGI_OPIS_ILUSTRACJI = {"h0_ocr": "ocr", "h0_opis": "ocr_opis", "h2_ocr": "ocr",
                           "h2_vlm": "vlm_q2b_en"}  # h2_vlm: opis Qwen3.5-2B z wizją po angielsku (pomiar B0)   # → wariant obrazy.blok_opisu
KONFIGI_H2 = ("h2", "h2_ocr", "h2_vlm")
# goły prompt + opis ilustracji z osobnego, mniejszego modelu wizyjnego (Qwen3.5-2B z mmproj, cache obrazy.py): baza
# tekstowa bez własnego projektora wizji (projektor bazy liczy się do rozmiaru, osobny mniejszy model nie; 26.09)
KONFIGI_GOLY_OPIS = {"goly_vlm": "vlm_q2b_en", "goly_vlm_pl": "vlm_q2b_pl"}
# pomiar sufitu wiedzy (B1): + fakty wzorcowe od Claude z cache (scripts/kontekst_wzorcowy.py), tylko dev, nigdy egzamin
KONFIGI_WIEDZA = {"goly_vlm_wiedza": "vlm_q2b_en", "goly_wiedza": None}
# baza haseł (data/wiedza/hasla*.jsonl, poza limitem rozmiaru): goły prompt + opis ilustracji + n najlepszych haseł
KONFIGI_KB = {"goly_vlm_kb": ("vlm_q2b_en", 2), "goly_vlm_kb3": ("vlm_q2b_en", 3), "goly_kb": (None, 2)}
# jak goly_vlm_kb, ale zadania zamknięte przez matura/zamkniete.py (ścisły format „ODPOWIEDŹ: …” + głosowanie 3 próbek)
KONFIGI_KB_Z = {"goly_vlm_kb_z": ("vlm_q2b_en", 2)}
# jak goly_vlm_kb_z, ale zadania „rozstrzygnij” przez matura/rozstrzygnij.py (głosowanie 3 próbek nad rozstrzygnięciem)
KONFIGI_KB_ZR = {"goly_vlm_kb_zr": ("vlm_q2b_en", 2)}
KONFIGI_KB_Z.update(KONFIGI_KB_ZR)
_BAZA_HASEL = None


def _baza_hasel():
    global _BAZA_HASEL
    with _LOCK_KARTY:
        if _BAZA_HASEL is None:
            from .wiedza import BazaHasel
            _BAZA_HASEL = BazaHasel()
    return _BAZA_HASEL
_WIEDZA_WZORCOWA: dict | None = None


def _wiedza_wzorcowa(zid: str) -> list[str]:
    global _WIEDZA_WZORCOWA
    with _LOCK_KARTY:
        if _WIEDZA_WZORCOWA is None:
            from pathlib import Path
            p = Path(__file__).resolve().parent.parent / "data" / "wiedza" / "kontekst_wzorcowy.jsonl"
            _WIEDZA_WZORCOWA = {}
            if p.exists():
                import json
                for l in p.open(encoding="utf-8"):
                    d = json.loads(l); _WIEDZA_WZORCOWA[d["id"]] = d["fakty"]
    if zid not in _WIEDZA_WZORCOWA:
        raise KeyError(f"brak kontekstu wzorcowego dla {zid} (scripts/kontekst_wzorcowy.py)")
    return _WIEDZA_WZORCOWA[zid]
MAX_TOKENS_ZAMKNIETE = 120   # h2 zamknięte: same oznaczenia; dłuższy limit pozwalał Bielikowi-1.5B zapętlić się (etap A)

SYS_GOLY = "Jesteś uczniem zdającym maturę z historii. Odpowiedz na zadanie po polsku."

_WSPOLNE = """Jesteś zdającym maturę z historii (poziom rozszerzony). Piszesz po polsku, krótko i konkretnie.
Nie przepisuj polecenia. Bez wstępów i podsumowań. Podawaj daty tylko, jeśli jesteś pewien."""

SYS_ZAMKNIETE = _WSPOLNE + """
To zadanie zamknięte. Podaj wyłącznie wybrane oznaczenia (litery albo cyfry; przy ocenie prawdziwości przy każdym
zdaniu jedną literę: P, gdy prawdziwe, albo F, gdy fałszywe), w kolejności z polecenia. Nie uzasadniaj, jeśli
polecenie tego nie wymaga."""

SYS_OTWARTE = _WSPOLNE + """
To zadanie otwarte. Udziel dokładnie tylu odpowiedzi, o ile prosi polecenie.
Jeśli polecenie każe odwołać się do źródła, wskaż konkretnie, co w źródle na to wskazuje, i połącz to z wiedzą historyczną."""

SYS_ROZSTRZYGNIJ = SYS_OTWARTE + """
Polecenie każe rozstrzygnąć i uzasadnić: pierwsza linia zaczyna się od „Rozstrzygnięcie:", druga od „Uzasadnienie:"."""

SYS_ESEJ = """Piszesz wypracowanie maturalne z historii (poziom rozszerzony), po polsku.
Wybierz JEDEN temat z zadania, ten, o którym masz najwięcej faktów w kontekście. Na początku napisz „Temat nr …".
Budowa (obowiązkowa):
1. Wstęp: przywołaj tezę tematu i zajmij JASNE stanowisko (zgadzam się / nie zgadzam się / zgadzam się częściowo).
2. Trzy akapity rozwinięcia: każdy o jednym z trzech aspektów wymienionych w temacie; w każdym 2–3 konkretne fakty
   (daty, postaci, nazwy, pojęcia) i zdanie wiążące je ze stanowiskiem.
3. Zakończenie: to samo stanowisko co we wstępie i krótkie podsumowanie.
Nie zmieniaj stanowiska. Długość 400–500 słów. Nie podawaj faktów, których nie jesteś pewien."""

SYS_EN = "\nNajpierw pod nagłówkiem REASONING przemyśl zadanie krótko PO ANGIELSKU. Potem pod nagłówkiem ODPOWIEDŹ: podaj ostateczną odpowiedź PO POLSKU."

# Reguły h2 (dopisywane do promptu h0 wg typu zadania). Bez dosłownych przykładów odpowiedzi (D16: model 0.8B
# kopiował „1. P, 2. F” do każdej odpowiedzi); etykiety linii brane z samego polecenia.
R_ZAMKNIETE = ("Zawsze wybierz odpowiedź przy każdej pozycji, także gdy nie masz pewności: za błędną odpowiedź nie "
               "odejmuje się punktów, a brak odpowiedzi to zawsze 0 pkt. Podaj tylko to, czego wymaga polecenie "
               "(wybrane litery, P albo F przy każdym zdaniu, numery albo krótkie nazwy przy kolejnych pozycjach), "
               "w kolejności z polecenia i bez uzasadnienia, chyba że polecenie go wymaga.")
R_ROZSTRZYGNIJ = ("W linii „Rozstrzygnięcie:” podaj jednoznaczny wybór (tak albo nie, albo wskazaną w poleceniu "
                  "możliwość), bez wahania i bez „częściowo”.")
R_UZASADNIENIE = ("W uzasadnieniu podaj dwa elementy: informację ze źródła (krótki cytat w cudzysłowie albo parafrazę, "
                  "z numerem źródła, którego dotyczy) oraz jeden konkretny fakt z wiedzy historycznej (data, postać, "
                  "wydarzenie albo pojęcie), który potwierdza odpowiedź.")
R_KROTKO = ("Odpowiadaj od razu treścią: bez wstępu i bez powtarzania polecenia; nazwa albo jedno zwięzłe zdanie na "
            "każdy element; przy „Wyjaśnij” podaj przyczynę lub mechanizm w 1-2 zdaniach.")
R_LICZBA = "Polecenie określa liczbę („{slowo}”): podaj dokładnie tyle elementów, ile wymaga, nie więcej i nie mniej."
R_ETYKIETY = "Zapisz odpowiedź w osobnych liniach zaczynających się od: {etykiety}."
R_ILUSTRACJA = ("Zadanie dotyczy ilustracji, której nie widzisz. Odpowiadaj na podstawie podpisu, tytułu i daty "
                "źródła, opisu ilustracji (jeśli jest podany) oraz wiedzy historycznej. Zawsze udziel konkretnej "
                "odpowiedzi; nigdy nie zostawiaj jej pustej i nie pisz, że nie widzisz obrazu.")
RE_POLECENIE_KROTKIE = re.compile(r"\b(?:Podaj|Wymień|Wyjaśnij|Wskaż|Nazwij|Określ)\b")
RE_LICZBA = re.compile(r"\b(?:Podaj|Wymień|Wskaż|Sformułuj|Przedstaw|Wyjaśnij|Porównaj|Określ|Napisz|Uzasadnij|"
                       r"Scharakteryzuj|Nazwij|uzasadnij|podaj|wymień|wskaż)\b[^.]*?\b(jeden|jedną|jedno|jednego|"
                       r"jednej|dwa|dwie|dwóch|dwu|trzy|trzech|cztery|czterech)\b")
RE_ETYKIETA = re.compile(r"(?m)^\s*([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{3,}(?:\s[a-ząćęłńóśźż]{2,})?(?:\s\d+\.?)?):\s*$")
_NIE_ETYKIETY = {"Zadanie", "Źródło", "Fragment", "Uwaga", "Odpowiedź", "Rozstrzygnięcie"}

_PULA: list[dict] | None = None
_KARTY_E5 = None
_LOCK_KARTY = threading.Lock()


def system_dla(z: dict) -> str:
    if z["esej"]:
        return SYS_ESEJ
    if z["typ"] == "zamkniete":
        return SYS_ZAMKNIETE
    if re.search(r"Rozstrzygnij", z["polecenie"], re.I):
        return SYS_ROZSTRZYGNIJ
    return SYS_OTWARTE


def _ma_ilustracje(z: dict) -> bool:
    """Czy zadanie ma ilustrację (po poprawionym przypisaniu z matura/obrazy.py; błąd przypisania → lista devsetu)."""
    if z.get("esej") or not z.get("obrazy"):
        return False
    from . import obrazy
    try:
        return bool(obrazy.ilustracje(z))
    except Exception:  # noqa: BLE001 (brak PDF arkusza, nietypowe zadanie): ostrożnie zakładamy ilustrację
        return True


def system_h2(z: dict, widzi_obraz: bool = False) -> str:
    """Prompt systemowy h2 = prompt h0 (`system_dla`) + reguły odpowiedzi wg typu zadania i treści polecenia.
    widzi_obraz=False (model bez obrazu) i zadanie z ilustracją → reguła odpowiadania z podpisu i wiedzy."""
    sys_ = system_dla(z)
    if z.get("esej"):
        return sys_
    p = z["polecenie"]
    reguly = []
    if (z.get("typ") or typ_zadania(z)) == "zamkniete":
        reguly.append(R_ZAMKNIETE)
    else:
        rozstrz = bool(re.search(r"Rozstrzygnij", p, re.I))
        if rozstrz:
            reguly.append(R_ROZSTRZYGNIJ)
        if rozstrz or re.search(r"uzasadnij", p, re.I):
            reguly.append(R_UZASADNIENIE)
        if RE_POLECENIE_KROTKIE.search(p):
            reguly.append(R_KROTKO)
        m = RE_LICZBA.search(p)
        if m:
            reguly.append(R_LICZBA.format(slowo=m.group(1)))
        et = [e for e in RE_ETYKIETA.findall(p) if e.split()[0] not in _NIE_ETYKIETY][:6]  # kratki z arkusza
        if et and not rozstrz:
            reguly.append(R_ETYKIETY.format(etykiety=", ".join(f"„{e}:”" for e in et)))
    if not widzi_obraz and _ma_ilustracje(z):
        reguly.append(R_ILUSTRACJA)
    return sys_ + ("\n" + "\n".join(reguly) if reguly else "")


def _karty_e5():
    """Karty kanoniczne dla e5, gdy wywołujący ich nie przekazał (noc.py ładuje karty tylko dla e3/e4/e3_en)."""
    global _KARTY_E5
    with _LOCK_KARTY:
        if _KARTY_E5 is None:
            from .karty import Karty
            _KARTY_E5 = Karty()
    return _KARTY_E5


_BAZA_E6 = None


def _baza_e6():
    """Baza akapitów eseju (data/wiedza/akapity_esej*.jsonl), wczytana raz na proces."""
    global _BAZA_E6
    with _LOCK_KARTY:
        if _BAZA_E6 is None:
            from .wiedza import BazaAkapitow
            _BAZA_E6 = BazaAkapitow()
    return _BAZA_E6


def _kontekst(wiki, z: dict, k: int) -> tuple[str, list[str]]:
    if wiki is None:
        return "", []
    frs = wiki.szukaj(z["polecenie"] + " " + z.get("zrodla_tekst", "")[:600], k=k)
    if not frs:
        return "", []
    return ("Fragmenty encyklopedii (mogą pomóc, ale nie muszą być trafne):\n"
            + "\n".join(f"- [{f['tytul']}] {f['tekst']}" for f in frs) + "\n\n"), [f["tytul"] for f in frs]


def _fewshot(z: dict, n: int = 2) -> str:
    global _PULA
    if _PULA is None:
        _PULA = pula_treningowa()
    kand = [p for p in _PULA if p["typ"] == z["typ"] and not p["obrazy"]] or _PULA
    rng = random.Random(z["id"])
    ex = rng.sample(kand, min(n, len(kand)))
    return "".join(f"PRZYKŁAD ZADANIA:\n{tresc_dla_modelu(e)[:1500]}\nPRZYKŁADOWA ODPOWIEDŹ:\n{e['rozwiazanie'][:600]}\n\n" for e in ex)


def _po_odpowiedzi(t: str) -> str:
    m = re.search(r"ODPOWIED[ŹZ]\s*:?\s*(.*)", t, re.S | re.I)
    return m.group(1).strip() if m else t


def _obrazy_zadania(z: dict) -> list[str]:
    """Obrazy należące do zadania (matura/obrazy.py, układ PDF arkusza), a nie cała strona z `z["obrazy"]`."""
    try:
        from .obrazy import obrazy_zadania
        return obrazy_zadania(z)
    except Exception:  # noqa: BLE001 (brak PDF/zależności: dawne przypisanie per strona)
        return z["obrazy"]


def odpowiedz(konfig: str, z: dict, *, url: str, wiki, obrazy: bool, bez_myslenia: bool,
              karty=None) -> tuple[str, float, dict]:
    if konfig == "kalibracja_cke":
        return z.get("wzorzec_cke", "(brak)"), 0.0, {}
    if z["esej"] and konfig in KONFIGI_E7:
        return esej_mod.napisz_e5(z, url=url, wiki=wiki, karty=karty if karty is not None else _karty_e5(),
                                  bez_myslenia=bez_myslenia, stanowisko=KONFIGI_E7[konfig], baza=_baza_e6())
    if z["esej"] and konfig in KONFIGI_E6:
        return esej_mod.napisz_e6(z, url=url, wiki=wiki, karty=karty if karty is not None else _karty_e5(),
                                  baza=_baza_e6(), bez_myslenia=bez_myslenia, stanowisko=KONFIGI_E6[konfig])
    if z["esej"] and konfig in KONFIGI_E5:
        return esej_mod.napisz_e5(z, url=url, wiki=wiki, karty=karty if karty is not None else _karty_e5(),
                                  bez_myslenia=bez_myslenia, stanowisko=KONFIGI_E5[konfig])
    if z["esej"] and konfig in KONFIGI_ESEJ[1:]:
        return esej_mod.napisz(z, konfig, url=url, wiki=wiki, karty=karty, bez_myslenia=bez_myslenia)
    if z["esej"] and konfig == "e0":
        konfig = "h0"
    if not z["esej"] and konfig in KONFIGI_ESEJ:
        raise ValueError(f"konfiguracja esejowa {konfig} dla zadania krótkiego {z['id']}")
    if konfig == "wzorzec":
        return (z["rozwiazanie"] or "(brak wzorca)"), 0.0, {}
    if konfig == "pusty":
        return "Nie wiem.", 0.0, {}
    obr = (_obrazy_zadania(z) or None) if obrazy else None
    tresc = tresc_dla_modelu(z)
    mt = 1400 if z["esej"] else 400
    if konfig == "goly":
        t, s = czat(url, SYS_GOLY, tresc, obr, max_tokens=mt, bez_myslenia=bez_myslenia)
        return t, s, {}
    if konfig in KONFIGI_KB or konfig in KONFIGI_KB_Z:
        from .obrazy import blok_opisu
        from .wiedza import _tekst_hasla, zapytanie
        wariant, n = KONFIGI_KB.get(konfig) or KONFIGI_KB_Z[konfig]
        blok = blok_opisu(z, wariant) if wariant else ""
        hasla = _baza_hasel().szukaj(zapytanie(z, blok), n)
        wiedza = ("Wiedza pomocnicza (baza wiedzy, może nie dotyczyć zadania):\n"
                  + "\n".join(f"- {_tekst_hasla(h)}" for h in hasla) + "\n\n") if hasla else ""
        user = wiedza + (blok + "\n\n" if blok else "") + tresc
        if konfig in KONFIGI_KB_Z and z.get("typ") == "zamkniete":
            from . import zamkniete
            f = zamkniete.forma(z)
            if f is not None:
                t, s, meta = zamkniete.odpowiedz(f, url, user, bez_myslenia=bez_myslenia)
                return t, s, {"hasla": [h["id"] for h in hasla], "glosy": meta["glosy"]}
        if konfig in KONFIGI_KB_ZR:
            from . import rozstrzygnij
            if rozstrzygnij.czy_rozstrzygnij(z):
                t, s, meta = rozstrzygnij.odpowiedz(url, user, bez_myslenia=bez_myslenia, max_tokens=mt)
                return t, s, {"hasla": [h["id"] for h in hasla], "glosy": meta["decyzje"]}
        t, s = czat(url, SYS_GOLY, user, None, max_tokens=mt, bez_myslenia=bez_myslenia)
        return t, s, {"hasla": [h["id"] for h in hasla]}
    if konfig in KONFIGI_WIEDZA:
        from .obrazy import blok_opisu
        blok = blok_opisu(z, KONFIGI_WIEDZA[konfig]) if KONFIGI_WIEDZA[konfig] else ""
        wiedza = "Wiedza pomocnicza:\n" + "\n".join(f"- {f}" for f in _wiedza_wzorcowa(z["id"])) + "\n\n"
        t, s = czat(url, SYS_GOLY, wiedza + (blok + "\n\n" if blok else "") + tresc, None, max_tokens=mt,
                    bez_myslenia=bez_myslenia)
        return t, s, {}
    if konfig in KONFIGI_GOLY_OPIS:
        from .obrazy import blok_opisu
        blok = blok_opisu(z, KONFIGI_GOLY_OPIS[konfig])
        t, s = czat(url, SYS_GOLY, (blok + "\n\n" if blok else "") + tresc, None, max_tokens=mt, bez_myslenia=bez_myslenia)
        return t, s, {"ilustracje_opisane": blok.count("OPIS ILUSTRACJI (")}
    sys_ = system_dla(z)
    ctx, tyt = ("", []) if konfig == "h0_norag" else _kontekst(wiki, z, 8 if z["esej"] else 4)
    fs = _fewshot(z) if konfig == "h1" and not z["esej"] else ""
    meta = {"kontekst_tytuly": tyt}
    if konfig == "h0_en":
        t, s = czat(url, sys_ + SYS_EN, ctx + "ZADANIE:\n" + tresc, obr, max_tokens=mt + 400, bez_myslenia=bez_myslenia)
        return _po_odpowiedzi(t), s, meta
    if konfig in ("h0", "h0_norag", "h1"):
        t, s = czat(url, sys_, fs + ctx + "ZADANIE:\n" + tresc, obr, max_tokens=mt, bez_myslenia=bez_myslenia)
        return t, s, meta
    if konfig in KONFIGI_H2:
        if wiki is None:
            raise ValueError(f"{konfig} wymaga indeksu Wikipedii (RAG jak h0); w noc.py dopisz {konfig} do KONFIGI_Z_WIKI")
        blok = ""
        if konfig in KONFIGI_OPIS_ILUSTRACJI:  # tekst zamiast obrazu, jak h0_ocr
            from .obrazy import blok_opisu
            blok = blok_opisu(z, KONFIGI_OPIS_ILUSTRACJI[konfig])
            meta["ilustracje_opisane"] = blok.count("OPIS ILUSTRACJI (")
            obr = None
        if (z.get("typ") or typ_zadania(z)) == "zamkniete":
            mt = MAX_TOKENS_ZAMKNIETE
        t, s = czat(url, system_h2(z, widzi_obraz=obr is not None), ctx + (blok + "\n\n" if blok else "")
                    + "ZADANIE:\n" + tresc, obr, max_tokens=mt, bez_myslenia=bez_myslenia)
        return t, s, meta
    if konfig in KONFIGI_OPIS_ILUSTRACJI:  # tekst zamiast obrazu: model nie dostaje pliku obrazu
        from .obrazy import blok_opisu
        blok = blok_opisu(z, KONFIGI_OPIS_ILUSTRACJI[konfig])
        meta["ilustracje_opisane"] = blok.count("OPIS ILUSTRACJI (")
        t, s = czat(url, sys_, ctx + (blok + "\n\n" if blok else "") + "ZADANIE:\n" + tresc, None,
                    max_tokens=mt, bez_myslenia=bez_myslenia)
        return t, s, meta
    raise ValueError(f"nieznana konfiguracja {konfig}")


def prompt_treningowy(z: dict, ctx: str) -> list[dict]:
    """Ten sam format co h0 → dane SFT uczą dokładnie tego, co model zobaczy na egzaminie."""
    return [{"role": "system", "content": system_dla(z)}, {"role": "user", "content": ctx + "ZADANIE:\n" + tresc_dla_modelu(z)}]
