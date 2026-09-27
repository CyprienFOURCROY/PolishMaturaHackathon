"""Esej e8: wypracowanie strukturalne, akapity pisane po angielsku z materiału, tłumaczone Marianem, rama po polsku.

Co to jest: potok eseju dla modeli, które piszą lepiej po angielsku niż po polsku (sonda 1a, E1). Temat rozbierany po
polsku (`esej.rozbierz`); rama (wstęp, zakończenie, zdania wiążące akapitów) z `esej.szablon_e5`, od razu po polsku;
materiał wybierany po polsku jak w e5 (karty kanoniczne + hasło bazy), do promptu idzie jego wersja angielska (pamięć
tłumaczeń `data/karty/tlumaczenia/kanon_en.jsonl`, tłumaczył lokalnie Qwen3-4B-2507); akapit EN → Marian EN→PL →
kontrole e5 na polskim tekście (daty wobec polskich kart, powtórzenia, długość, kierunek stanowiska).
Po co: esej to 15 z 60 pkt; nasze modele tracą na ogólnikach, błędach przypisania i braku porównania (plan, sekcja
„Esej: diagnoza i plan”). Wyniki: review/esej-e8-2026-09-26/README.md.
Co zrobić: raz `uv run python -m matura.esej_en tlumacz-karty` (GPU, port 8095), potem
`uv run python -m matura.esej_en generuj --modele lfm2-2.6b-q4km --sesje 2024-maj,2025-maj --gen 1`.
Przykłady w promptach i testach tylko z arkuszy spoza oceny (2015-2022).
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import esej, jezyk
from .llm import czat

ROOT = Path(__file__).resolve().parent.parent
PLIK_TM = ROOT / "data" / "karty" / "tlumaczenia" / "kanon_en.jsonl"
KARTY = ROOT / "data" / "karty" / "kanon.jsonl"
MODEL_TL = "qwen3.5-9b-q5"   # tłumacz kart (lokalnie; Qwen3-4B-2507 mylił nazwy i daty w ~15% faktów, próba 23:40)
PORT_TL = 8095

# ---------------------------------------------------------------- pamięć tłumaczeń kart (PL → EN, linia w linię)

SYS_TL = ("You translate numbered lines of a Polish history knowledge base into English. Translate every line "
          "faithfully and completely: keep all facts, names, dates and numbers, and do not add or remove information. "
          "Use the standard English names of people, places, treaties, institutions and events (for example: Peace of "
          "Westphalia, Duchy of Warsaw, Four-Year Sejm). Answer only with the translated lines, numbered as in the "
          "input, one line per input line.")
RE_NUM = re.compile(r"^\s*(\d{1,3})[.)]\s+(.+?)\s*$", re.M)


def _klucz_tm(pl: str) -> str:
    return jezyk.sha("tm", pl)


MIESIACE = {"stycz": "january", "lut": "february", "marc": "march", "kwie": "april", "maj": "may", "czerw": "june",
            "lip": "july", "sierp": "august", "wrze": "september", "paźdz": "october", "listop": "november",
            "grud": "december"}
RE_MIESIAC = re.compile(r"\b\d{1,2}\s+(stycz|lut|marc|kwie|maj|czerw|lip|sierp|wrze|paźdz|listop|grud)\w*", re.I)


def _wiarygodne(pl: str, en: str) -> bool:
    """Tłumaczenie niepuste, inne niż oryginał (poza krótkimi nazwami), o rozsądnej długości (model czasem streszcza
    albo dopisuje objaśnienia), z tymi samymi liczbami i miesiącami („25 czerwca 1976” → „May 25, 1976”: próba 23:40)."""
    if not en.strip():
        return False
    if not set(re.findall(r"\d+", pl)) <= set(re.findall(r"\d+", en)):
        return False
    if any(MIESIACE[m.group(1).lower()] not in en.lower() for m in RE_MIESIAC.finditer(pl)):
        return False
    if len(pl) < 25:
        return len(en) <= 3 * len(pl) + 20
    return en.strip() != pl.strip() and 0.4 <= len(en) / len(pl) <= 2.5


def tlumacz_linie(linie: list[str], c: "jezyk.Cache", *, url: str, czat_fn=czat, partia: int = 15,
                  bez_myslenia: bool = True, rownolegle: int = 8) -> dict[str, str]:
    """Linie PL → {linia PL: EN} przez pamięć tłumaczeń `c` (klucz = treść PL). Brakujące tłumaczone partiami
    numerowanych linii; partia z brakującą albo niewiarygodną linią dzielona na pół, aż do pojedynczych linii
    (pojedyncza linia bez numeru w odpowiedzi też się liczy). Linia nieprzetłumaczalna zostaje bez wpisu."""
    todo = list(dict.fromkeys(l for l in linie if l.strip() and _klucz_tm(l) not in c))

    def jedna(cz: list[str]) -> None:
        user = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(cz))
        try:
            out, _ = czat_fn(url, SYS_TL, user, None, max_tokens=sum(len(t) for t in cz) // 2 + 200, temperature=0.0,
                             bez_myslenia=bez_myslenia)
        except Exception as e:  # noqa: BLE001 (partia ponawiana mniejszymi kawałkami)
            jezyk.log(f"[tm] błąd partii {len(cz)}: {e}")
            out = ""
        got = {int(n): t for n, t in RE_NUM.findall(out)}
        if len(cz) == 1 and not got and out.strip() and "\n" not in out.strip():
            got = {1: out.strip()}
        if all(_wiarygodne(t, got.get(i + 1, "")) for i, t in enumerate(cz)):
            c.dodaj([{"klucz": _klucz_tm(t), "pl": t, "en": got[i + 1], "tlumacz": "baza"} for i, t in enumerate(cz)])
        elif len(cz) > 1:
            h = len(cz) // 2
            jedna(cz[:h])
            jedna(cz[h:])

    with ThreadPoolExecutor(max(1, rownolegle)) as ex:
        list(ex.map(jedna, [todo[i:i + partia] for i in range(0, len(todo), partia)]))
    return {l: c.get(_klucz_tm(l))["en"] for l in linie if c.get(_klucz_tm(l))}


def wczytaj_karty() -> list[dict]:
    return [json.loads(l) for l in open(KARTY, encoding="utf-8") if l.strip()]


def nazwy_kart(karty: list[dict]) -> list[str]:
    """Nazwy z pól postać, termin, tytuł (bez powtórzeń): źródło słowniczka PL↔EN."""
    return list(dict.fromkeys(x.strip() for k in karty for x in (k.get("postac"), k.get("termin"), k.get("tytul"))
                              if x and x.strip()))


def tlumacz_karty(url: str, bez_myslenia: bool = True, rownolegle: int = 8, c: "jezyk.Cache | None" = None) -> dict:
    """Fakty i nazwy wszystkich kart kanonicznych → pamięć tłumaczeń (wznawialne). Zwraca liczniki."""
    c = c or jezyk.Cache(PLIK_TM)
    karty = wczytaj_karty()
    fakty, nazwy = [k["fakt"] for k in karty], nazwy_kart(karty)
    jezyk.log(f"[tm] faktów {len(fakty)}, nazw {len(nazwy)}; w pamięci {len(c.d)}")
    f = tlumacz_linie(fakty, c, url=url, partia=12, bez_myslenia=bez_myslenia, rownolegle=rownolegle)
    jezyk.log(f"[tm] fakty {len(f)}/{len(set(fakty))}")
    n = tlumacz_linie(nazwy, c, url=url, partia=10, bez_myslenia=bez_myslenia, rownolegle=rownolegle)
    jezyk.log(f"[tm] nazwy {len(n)}/{len(nazwy)}")
    return {"fakty": len(f), "fakty_wszystkie": len(set(fakty)), "nazwy": len(n), "nazwy_wszystkie": len(nazwy)}


def pamiec_tm(p: Path = PLIK_TM) -> dict[str, str]:
    """{tekst PL: tekst EN} z pamięci tłumaczeń."""
    return {w["pl"]: w["en"] for w in jezyk.Cache(p).d.values()}


def slowniczek(tm: dict[str, str], nazwy: list[str]) -> list[tuple[str, str]]:
    """Pary (nazwa PL, nazwa EN) z pamięci tłumaczeń, najdłuższe najpierw (podmiana „Księstwo Warszawskie” przed
    „Warszawa”); bez nazw krótszych niż 4 znaki i bez par identycznych."""
    pary = [(n, tm[n]) for n in nazwy if n in tm and len(n) >= 4 and tm[n].strip() and tm[n].strip() != n]
    return sorted(dict.fromkeys(pary), key=lambda p: -len(p[0]))


def slowniczek_pewny(tm: dict[str, str], karty: list[dict]) -> list[tuple[str, str]]:
    """Słowniczek tylko z par potwierdzonych w karcie: nazwa EN (postać albo termin) występuje w angielskim tłumaczeniu
    faktu tej samej karty. Nazwy tłumaczone osobno bywają skażone sąsiednimi liniami partii („Duchowieństwo a
    edukacja” → „Monarchy and education”, próba 23:30); tytuły kart pomijane (opisowe, nie nazwy). Tylko nazwy własne
    (wielka litera) albo wielowyrazowe: pospolite słowa („sobór” → „Council”) podmieniałyby każdą „radę”."""
    pewne = []
    for k in karty:
        fakt_en = tm.get(k["fakt"], "").lower()
        for pole in ("postac", "termin"):
            n = (k.get(pole) or "").strip()
            if n and (any(ch.isupper() for ch in n) or len(n.split()) >= 2) and tm.get(n) and tm[n].lower() in fakt_en:
                pewne.append(n)
    return slowniczek(tm, list(dict.fromkeys(pewne)))


# ---------------------------------------------------------------- tłumaczenie akapitu EN → PL z ochroną nazw

def _rdzenie4(t: str) -> set[str]:
    return {w.lower()[:4] for w in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", t) if len(w) >= 4}


def nazwa_obecna(nazwa_pl: str, tekst_pl: str) -> bool:
    """Czy polska nazwa jest w tekście, z tolerancją odmiany: każde słowo nazwy (≥ 4 litery) ma w tekście słowo
    o tych samych 4 pierwszych literach („pokój westfalski” w „pokój Westfalii” tak, w „Rada Trentu” nie)."""
    st = _rdzenie4(nazwa_pl)
    return bool(st) and st <= _rdzenie4(tekst_pl)


def nazwa_w_tezie(nazwa_pl: str, teza: str) -> bool:
    """Ostrzej niż `nazwa_obecna` (5 liter): przy 4 literach „Poliklet” pasował do „Polityka” (próba 23:50)."""
    st = {w.lower()[:5] for w in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", nazwa_pl) if len(w) >= 4}
    return bool(st) and st <= {w.lower()[:5] for w in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", teza) if len(w) >= 4}


def na_polski(tekst_en: str, marian, slownik: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Akapit EN → PL Marianem, zdanie po zdaniu. Gdy zdanie EN zawiera nazwę ze słowniczka, a przekład nie ma jej
    polskiej formy (Marian przekręca nazwy: „Council of Trent” → „Rada Trentu”), zdanie tłumaczone ponownie z nazwą
    polską wstawioną w tekst EN (Marian ją przepisuje, w mianowniku: próba 23:35). Tylko wtedy, bo wstawiona nazwa nie
    jest odmieniona („pokonał Dariusz III”). Zwraca (tekst PL, poprawione nazwy)."""
    zd = esej.zdania(tekst_en)
    if not zd:
        return "", []
    pl = marian.tlumacz(zd, "en-pl")
    poprawione = []
    for i, (e, p) in enumerate(zip(zd, pl)):
        el = e.lower()
        brak = [(npl, nen) for npl, nen in slownik if nen.lower() in el
                and re.search(rf"(?<!\w){re.escape(nen)}(?!\w)", e, re.I) and not nazwa_obecna(npl, p)]
        if brak:
            e2 = e
            for npl, nen in brak:
                e2 = re.sub(rf"(?<!\w){re.escape(nen)}(?!\w)", npl, e2, flags=re.I)
            pl[i] = marian.tlumacz([e2], "en-pl")[0]
            poprawione += [npl for npl, _ in brak]
    return " ".join(p.strip() for p in pl if p.strip()), poprawione


# ---------------------------------------------------------------- rozbiór tematu i materiał

ASPEKTY_EN = {"polityczny": "political", "społeczno-gospodarczy": "socio-economic", "kulturowy": "cultural",
              "militarny": "military", "wojskowy": "military", "ustrojowy": "constitutional (system of government)",
              "dyplomatyczny": "diplomatic", "polityczno-ustrojowy": "political and constitutional",
              "gospodarczy": "economic", "społeczny": "social", "religijny": "religious", "ideologiczny": "ideological",
              "międzynarodowy": "international", "społeczno-polityczny": "socio-political", "prawny": "legal",
              "narodowościowy": "national (ethnic)", "terytorialny": "territorial", "demograficzny": "demographic",
              "kulturalny": "cultural", "gospodarczo-społeczny": "economic and social"}
# teza wymagająca porównania z innymi przykładami (sędzia: „brak porównania z innymi …” przy stopniu najwyższym)
RE_POROWNANIE = re.compile(r"(?<!\w)naj\w{3,}|przede wszystkim|w największym stopniu|bardziej niż", re.I)


def wymaga_porownania(teza: str) -> bool:
    return bool(RE_POROWNANIE.search(teza))


def podmiot_tezy(teza: str) -> list[str]:
    """Nazwy tego, co teza porównuje z innymi (reguła językowa, bez listy tematów):
    1. początkowy ciąg słów z wielkiej litery bez nazw państw i regionów, gdy ma ≥ 2 słowa („Aleksander Wielki był
       najwybitniejszym …”, „Polityka Napoleona Bonaparte przede wszystkim …”);
    2. inaczej nazwy własne po wyzwalaczu porównania, gdy wyzwalacz otwiera zdanie („Najważniejszym … był …”) albo
       jest nim „przede wszystkim” („… zawdzięczano przede wszystkim Ottonowi von Bismarckowi”);
    3. inaczej [] (podmiot nie jest nazwą, np. „Reformacja w największym stopniu osłabiła Kościół”: model wskazuje
       podmiot sam, karty bez wykluczania)."""
    def wl(slowa: list[str]) -> list[str]:
        return [w for w in slowa if w[:1].isupper() and len(w) > 2 and not w.lower().startswith(esej._OGOLNE_WLASNE)]

    slowa = re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", teza)
    ciag = []
    for w in slowa:
        if not w[:1].isupper():
            break
        ciag.append(w)
    ciag = wl(ciag)
    if len(ciag) >= 2:
        return ciag
    m = RE_POROWNANIE.search(teza)
    if m and (not teza[:m.start()].strip() or m.group(0).lower() == "przede wszystkim"):
        return wl(re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", teza[m.end():]))
    return []


def _rdzenie5(t: str) -> set[str]:
    return {w.lower()[:5] for w in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", t) if len(w) > 2}


def linia_faktu(c: dict, tm: dict[str, str]) -> str:
    """Karta → linia materiału z jawnym podmiotem i datą: „- Napoleon Bonaparte, 1807: <fakt EN>”."""
    fakt = tm.get(c["fakt"], c["fakt"])
    kto = tm.get(c["postac"], c["postac"]) if c.get("postac") else ""
    glowa = ", ".join(x for x in (kto, (c.get("data") or "").strip()) if x)
    return f"- {glowa}: {fakt}" if glowa else f"- {fakt}"


DZIAL_HISTORIOGRAFII = "Historia jako nauka"
RE_HISTORIOGRAFIA = re.compile(r"historiograf|historyk|szko\w* historyczn|dziejopis", re.I)


def material_e8(karty, baza_hasel, teza: str, pozycja: str, *, tm: dict[str, str], hasla_en: dict[str, str],
                element: bool = False, porownanie: bool = False, pomin: list[dict] | None = None,
                pomin_hasel: set[str] | None = None, stanowisko: str = "zgadzam", znaki_hasla: int = 1500,
                dzialy: list[str] | None = None) -> dict:
    """Materiał jednego akapitu: {tekst_en (do promptu), tekst_pl (do weryfikacji dat), fakty (karty PL), hasla}.
    Karty jak w e5 (`esej.material_e5`, bez Wikipedii); akapit porównawczy: karty z działów `dzialy` (działy kart
    akapitów aspektów, czyli ta sama epoka; bez tego teza bez daty dawała karty z innych stuleci, próba 23:50), najpierw
    w kolejności BM25 zapytania tezy bez podmiotu, bez kart o samym podmiocie (wszystkie rdzenie podmiotu w karcie).
    Do tego 1 hasło bazy (nieużyte wcześniej)."""
    uzyte = [c["fakt"] for c in (pomin or [])]
    if porownanie:
        from .router import epoka_pewna
        pod = podmiot_tezy(teza)
        st = _rdzenie5(" ".join(pod))
        zap = " ".join(w for w in re.findall(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ-]+", teza) if w not in pod)
        kand = esej._bez_powtorzen_kart(karty.szukaj(zap, 40, epoka=epoka_pewna(teza))) if karty is not None else []
        if dzialy and getattr(karty, "k", None):
            zst = _rdzenie5(zap)
            w_dziale = [c for c in kand if c.get("dzial") in dzialy]
            reszta = sorted((c for c in karty.k if c.get("dzial") in dzialy and c not in w_dziale),
                            key=lambda c: -len(zst & _rdzenie5(c["fakt"])))
            kand = esej._bez_powtorzen_kart(w_dziale + reszta)
        fakty = [c for c in kand if not (st and st <= _rdzenie5(f"{c['fakt']} {c.get('postac', '')}"))
                 and not any(esej._podobne(c["fakt"], u) for u in uzyte)][:8]
    else:
        fakty = esej.material_e5(None, karty, teza, pozycja, element=element, n_wiki=0, pomin=pomin,
                                 stanowisko=stanowisko)["fakty"] if karty is not None else []
    if not RE_HISTORIOGRAFIA.search(teza):   # karty o historykach rodziły „jak twierdził Szujski” (próba 23:50)
        fakty = [c for c in fakty if c.get("dzial") != DZIAL_HISTORIOGRAFII]
    hasla = []
    if baza_hasel is not None:
        hasla = [h for h in baza_hasel.szukaj(f"{teza} {pozycja}", 3) if h["id"] not in (pomin_hasel or set())][:1]
    czesci = []
    if fakty:
        czesci.append("FACTS (each fact belongs only to the person, state and date named in it):\n"
                      + "\n".join(linia_faktu(c, tm) for c in fakty))
    tla = [hasla_en.get(h["id"], "")[:znaki_hasla] for h in hasla]
    if any(tla):
        czesci.append("BACKGROUND (encyclopedia entry, may be partly unrelated):\n" + "\n".join(t for t in tla if t))
    from .wiedza import _tekst_hasla
    tekst_pl = " ".join(f"{c['fakt']} ({c.get('data') or ''})" for c in fakty) + " " + \
        " ".join(_tekst_hasla(h) for h in hasla)
    return {"tekst_en": "\n\n".join(czesci), "tekst_pl": tekst_pl, "fakty": fakty, "hasla": [h["id"] for h in hasla]}


# ---------------------------------------------------------------- prompt akapitu

SYS_E8 = """You write one body paragraph of a history essay for the Polish matura exam (extended level).
Write in English: one continuous paragraph of 7-9 full sentences (about 150 words). No lists, no headings, no bullet points.
{kierunek}
Use at least four specific facts from the MATERIAL; for each fact say who, when, what happened and what it changed.
Attribute each fact only to the person or state named in that fact; never move a fact to another ruler, country or period.
Do not add dates or names that are not in the MATERIAL.
Do not write an introduction or a conclusion of the whole essay, and do not start with "Firstly".
The last sentence explains how these facts support the author's position on the thesis."""
KIERUNEK_E8 = {
    "zgadzam": ("The author AGREES with the thesis. Show facts about the {obszar} that CONFIRM the thesis: causes, events "
                "and effects that prove it right. Never write that the thesis is false or only partly true."),
    "nie_zgadzam": ("The author DISAGREES with the thesis. Show facts about the {obszar} that CONTRADICT the thesis: other "
                    "factors mattered more, there were limits and failures, the effects were different. Never write that "
                    "the thesis is true."),
}
KIERUNEK_POROWNANIE = {
    "zgadzam": ("The author AGREES with the thesis. COMPARE {podmiot} with at least two other cases of the same kind from "
                "the MATERIAL (other rulers, states, wars, events or attempts) and show that they achieved or mattered "
                "less, so the thesis is right. Never write that the thesis is false."),
    "nie_zgadzam": ("The author DISAGREES with the thesis. COMPARE {podmiot} with at least two other cases of the same kind "
                    "from the MATERIAL and show that they achieved or mattered as much or more, so the thesis is wrong. "
                    "Never write that the thesis is true."),
}
POPRAWKI_E8 = {
    "krotki": "Write a longer paragraph: 8-9 full sentences, about 160 words.",
    "lista": "Write one continuous paragraph, without bullet points or headings.",
    "sprzecznosc": {"zgadzam": "The author agrees with the thesis: do not write that it is false.",
                    "nie_zgadzam": "The author disagrees with the thesis: do not write that it is true."},
    "powtorzenia": "Do not repeat sentences or ideas from the other paragraphs; use different facts.",
}
MAX_TOKENOW_AKAPITU_EN = 420


def prompt_e8(teza: str, teza_en: str, nazwy: list[tuple[str, str]], pozycja_en: str, rodzaj: str, stanowisko: str,
              material: str, poprawki: list[str], podmiot_en: str = "") -> tuple[str, str]:
    """(system, user) akapitu e8. rodzaj: aspekt | przyklad | porownanie."""
    if rodzaj == "porownanie":
        kier = KIERUNEK_POROWNANIE[stanowisko].format(podmiot=podmiot_en or "the subject of the thesis")
        zadanie = "Write the comparison paragraph."
    else:
        obszar = f"example: {pozycja_en}" if rodzaj == "przyklad" else f"{pozycja_en} aspect"
        kier = KIERUNEK_E8[stanowisko].format(obszar=obszar)
        zadanie = f"Write the paragraph about the {obszar}."
    rel = "agrees" if stanowisko == "zgadzam" else "disagrees"
    user = ((f"MATERIAL:\n{material}\n\n" if material else "")
            + f"Thesis (Polish original): {teza}\nThesis (English): {teza_en}\n"
            + ("Names in the thesis: " + "; ".join(f"{p} = {e}" for p, e in nazwy) + "\n" if nazwy else "")
            + f"The author {rel} with the thesis.\n"
            + (f"Example: {pozycja_en}\n" if rodzaj == "przyklad" else f"Aspect: {pozycja_en}\n" if rodzaj == "aspekt" else "")
            + f"\n{zadanie}" + ("\nNOTE: " + " ".join(poprawki) if poprawki else ""))
    return SYS_E8.format(kierunek=kier), user


# ---------------------------------------------------------------- esej e8

MAX_SLOW_AKAPITU_E8 = 200   # akapity EN małych modeli mają ~250 słów; 150 (e5) ucinało zdanie wiążące z tezą


def wykoncz_e8(ocena: dict, max_slow: int = MAX_SLOW_AKAPITU_E8) -> list[str]:
    """Jak `esej._wykoncz`, ale przycięcie usuwa przedostatnie zdania, a ostatnie (związek faktów z tezą, wymagany
    w prompcie) zostaje."""
    zd = list(ocena["zdania"])
    if ocena["sprzeczne"] and len(zd) - len(ocena["sprzeczne"]) >= 3:
        zd = [z for z in zd if z not in ocena["sprzeczne"]]
    while len(zd) > 3 and sum(esej._slowa(z) for z in zd) > max_slow:
        del zd[-2]
    return zd
RAMA_POROWNANIA = {
    "zgadzam": ("Po czwarte, trafność tezy potwierdza porównanie z innymi przykładami.",
                "Porównanie to potwierdza więc tezę."),
    "nie_zgadzam": ("Po czwarte, tezie przeczy porównanie z innymi przykładami.", "Porównanie to nie potwierdza więc tezy."),
}


def _elementy_e8(teza: str, teza_en: str, wybor: str, *, karty, tm, marian, url, czat_fn, bez_myslenia) -> list[str]:
    """Temat „trzech wybranych …”: kandydaci z kart materiału (postać przy władcach i postaciach, inaczej tytuł karty),
    model wybiera trzy numery z listy po angielsku; nazwy wracają po polsku (pole karty). Braki: kolejni kandydaci."""
    m = esej.material_e5(None, karty, teza, wybor, element=True, n_wiki=0) if karty is not None else {"fakty": []}
    pole = "postac" if re.search(r"władc|postac|osób|polityk|przywódc|wodz", wybor) else "tytul"
    kand = list(dict.fromkeys((c.get(pole) or "").strip() for c in m["fakty"] if (c.get(pole) or "").strip()))[:10]
    wybrane: list[str] = []
    if len(kand) > 3:
        wybor_en = marian.tlumacz([wybor], "pl-en")[0]
        lista = "\n".join(f"{i + 1}. {tm.get(k, k)}" for i, k in enumerate(kand))
        t, _ = czat_fn(url, "You answer briefly.",
                       f"Essay thesis: {teza_en}\nThe essay must discuss three {wybor_en}. Choose the three items from "
                       f"the list that fit the thesis best.\n{lista}\nAnswer with three numbers separated by commas.",
                       None, max_tokens=30, temperature=0.0, bez_myslenia=bez_myslenia)
        for n in re.findall(r"\d+", t):
            if 1 <= int(n) <= len(kand) and kand[int(n) - 1] not in wybrane:
                wybrane.append(kand[int(n) - 1])
    for k in kand:
        if len(wybrane) >= 3:
            break
        if k not in wybrane:
            wybrane.append(k)
    while len(wybrane) < 3:
        wybrane.append(f"kolejny przykład ({len(wybrane) + 1})")
    return wybrane[:3]


def napisz_e8(z: dict, *, url: str, karty, baza_hasel, marian, tm: dict[str, str], hasla_en: dict[str, str],
              slownik: list[tuple[str, str]], czat_fn=czat, bez_myslenia: bool = True, stanowisko: str = "zgadzam",
              temperatura: float = 0.7, proby: int = 2, regeneracje_dlugosci: int = 2) -> tuple[str, float, dict]:
    """Wypracowanie e8 → (tekst PL, sekundy, meta). Kolejno: rozbiór po polsku, teza EN (Marian), akapity EN (aspekty
    albo trzy przykłady, plus akapit porównawczy przy `wymaga_porownania`), każdy: prompt → Marian EN→PL z ochroną
    nazw → `esej.ocen_akapit` (daty wobec polskich kart, powtórzenia, sprzeczność kierunku, długość) → do `proby` prób;
    rama z `esej.szablon_e5` po polsku; esej < 300 słów → regeneracja najkrótszego akapitu."""
    if stanowisko not in KIERUNEK_E8:
        raise ValueError(f"stanowisko e8: {tuple(KIERUNEK_E8)}, jest {stanowisko!r}")
    temat = z.get("temat") or (esej.tematy(z["polecenie"]) or [z["polecenie"]])[0]
    teza, pozycje, wybor = esej.rozbierz(temat)
    sek, t0 = 0.0, time.time()
    teza_en = marian.tlumacz([teza], "pl-en")[0]
    nazwy = [(p, e) for p, e in slownik if nazwa_w_tezie(p, teza)][:6]
    if wybor:
        pozycje = _elementy_e8(teza, teza_en, wybor, karty=karty, tm=tm, marian=marian, url=url, czat_fn=czat_fn,
                               bez_myslenia=bez_myslenia)
        pozycje_en = [tm.get(p, p) for p in pozycje]
    else:
        brak = [p for p in pozycje if p not in ASPEKTY_EN]
        tl = dict(zip(brak, marian.tlumacz(brak, "pl-en"))) if brak else {}
        pozycje_en = [ASPEKTY_EN.get(p) or tl[p] for p in pozycje]
    rodzaje = ["przyklad" if wybor else "aspekt"] * len(pozycje)
    porownanie = wymaga_porownania(teza)
    podmiot_en = ""
    if porownanie:
        pod = " ".join(podmiot_tezy(teza))
        podmiot_en = next((e for p, e in nazwy if nazwa_w_tezie(p, pod) and nazwa_w_tezie(pod, p)), "") or \
            (marian.tlumacz([pod], "pl-en")[0] if pod else "")
        pozycje_en, rodzaje = pozycje_en + ["comparison"], rodzaje + ["porownanie"]
    rdzen = esej._rdzen_tezy(teza)
    mats: list[dict] = []
    for i, r in enumerate(rodzaje):
        dz = [d for d, _ in Counter(c.get("dzial") for m in mats for c in m["fakty"]).most_common(2)
              if d != DZIAL_HISTORIOGRAFII]
        mats.append(material_e8(karty, baza_hasel, teza, pozycje[i] if r != "porownanie" else "porównanie", tm=tm,
                                hasla_en=hasla_en, element=r == "przyklad", porownanie=r == "porownanie",
                                pomin=[c for m in mats for c in m["fakty"]],
                                pomin_hasel={h for m in mats for h in m["hasla"]}, stanowisko=stanowisko, dzialy=dz))
    wszystkie_fakty = [c for m in mats for c in m["fakty"]]
    tekst_mat = " ".join(m["tekst_pl"] for m in mats) + " " + teza
    surowe_en: list[str] = [""] * len(rodzaje)
    nazwy_pop: list[str] = []

    def generuj(i: int, poprzednie: list[str], poprawki0: list[str]) -> tuple[dict, int]:
        nonlocal sek
        najl, n, poprawki = None, 0, list(poprawki0)
        for _ in range(max(1, proby)):
            sys_, user = prompt_e8(teza, teza_en, nazwy, pozycje_en[i], rodzaje[i], stanowisko, mats[i]["tekst_en"],
                                   poprawki, podmiot_en)
            t, s = czat_fn(url, sys_, user, None, max_tokens=MAX_TOKENOW_AKAPITU_EN, temperature=temperatura,
                           bez_myslenia=bez_myslenia)
            sek += s
            n += 1
            en, _ = esej._sklej_liste(t)
            pl, pop = na_polski(en, marian, slownik)
            o = esej.ocen_akapit(pl, rdzen=rdzen, stanowisko=stanowisko, fakty=wszystkie_fakty,
                                 tekst_materialu=tekst_mat, poprzednie=poprzednie)
            o["en"], o["nazwy"] = en, pop
            if najl is None or o["kara"] < najl["kara"]:
                najl = o
            if not o["wady"]:
                break
            poprawki = list(dict.fromkeys(poprawki0 + [POPRAWKI_E8[w][stanowisko] if w == "sprzecznosc"
                                                       else POPRAWKI_E8[w] for w in o["wady"]]))
        return najl, n

    oceny, n_prob, gotowe = [], [], []
    for i in range(len(rodzaje)):
        o, n = generuj(i, [x for zd in gotowe for x in zd], [])
        zd = wykoncz_e8(o)
        oceny.append(o); n_prob.append(n); gotowe.append(zd)
    szab = esej.szablon_e5(teza, pozycje, bool(wybor), stanowisko)
    ramy = list(szab["akapity"]) + ([RAMA_POROWNANIA[stanowisko]] if porownanie else [])

    def sklad() -> str:
        ak = [f"{g} {' '.join(zd)} {o}".replace("  ", " ") for (g, o), zd in zip(ramy, gotowe)]
        return "\n\n".join([szab["wstep"], *ak, szab["zakonczenie"]])

    n_reg = 0
    while esej._slowa(sklad()) < esej.MIN_SLOW_ESEJU and n_reg < regeneracje_dlugosci:
        i = min(range(len(gotowe)), key=lambda k: sum(esej._slowa(x) for x in gotowe[k]))
        o, n = generuj(i, [x for k, zd in enumerate(gotowe) if k != i for x in zd], [POPRAWKI_E8["krotki"]])
        n_reg += 1
        n_prob[i] += n
        zd = wykoncz_e8(o)
        if sum(esej._slowa(x) for x in zd) > sum(esej._slowa(x) for x in gotowe[i]) and "sprzecznosc" not in o["wady"]:
            oceny[i], gotowe[i] = o, zd
    tekst = sklad()
    meta = {"stanowisko": stanowisko, "teza_en": teza_en, "pozycje": pozycje, "pozycje_en": pozycje_en,
            "porownanie": porownanie, "podmiot_en": podmiot_en, "proby": n_prob, "regeneracje_dlugosci": n_reg,
            "slow": esej._slowa(tekst), "slowa_akapitow": [sum(esej._slowa(x) for x in zd) for zd in gotowe],
            "wady_akapitow": [o["wady"] for o in oceny], "daty_podmienione": [d for o in oceny for d in o["podmienione"]],
            "daty_usuniete": [d for o in oceny for d in o["usuniete"]],
            "nazwy_poprawione": [x for o in oceny for x in o["nazwy"]],
            "karty": [len(m["fakty"]) for m in mats], "hasla": [m["hasla"] for m in mats],
            "akapity_en": [o["en"] for o in oceny], "sekundy_calosc": round(time.time() - t0, 1)}
    return tekst, sek, meta


# ---------------------------------------------------------------- generacja na arkuszach (pomiar)

class _Zablokowany:
    """Obiekt współdzielony przez wątki (Marian, BM25 kart i haseł): metody wołane pod jedną blokadą."""

    def __init__(self, obj, metody: tuple[str, ...]):
        self._o, self._l, self._m = obj, threading.Lock(), metody

    def __getattr__(self, n):
        a = getattr(self._o, n)
        if n not in self._m:
            return a

        def zablokowana(*args, **kw):
            with self._l:
                return a(*args, **kw)
        return zablokowana


PORTY_E8 = (8097, 8098, 8099)
WYN_E8 = ROOT / "review" / "esej-e8-2026-09-26" / "e8"


def generuj(modele: list[str], sesje: tuple[str, ...], gen: int = 1, wyniki: Path = WYN_E8,
            stanowisko: str = "zgadzam", limit: int | None = None, rownolegle: int = 4) -> list[Path]:
    """Eseje e8 na wszystkie tematy arkuszy `sesje` dla każdego modelu (llama-server na portach 8097-8099, Marian
    w procesie). Zapis przyrostowy `<wyniki>/odpowiedzi/<model>__e8[_g<gen>].jsonl`; wznowienie pomija gotowe id."""
    from .harness import _baza_hasel
    from .karty import Karty
    from .tlumacz import Marian
    testowe = [s for s in sesje if s[:4].isdigit() and int(s[:4]) >= 2026]
    if testowe:
        raise ValueError(f"sesje testowe {testowe} są niedozwolone w pomiarze e8")
    stare = (jezyk.WYN, jezyk.TYLKO_ESEJE)
    jezyk.WYN, jezyk.TYLKO_ESEJE = wyniki, True
    try:
        zz = jezyk.zadania(list(sesje))
    finally:
        jezyk.TYLKO_ESEJE = stare[1]
    zz = zz[:limit] if limit else zz
    karty = _Zablokowany(Karty(), ("szukaj",))
    baza = _Zablokowany(_baza_hasel(), ("szukaj",))
    marian = _Zablokowany(Marian(), ("tlumacz",))
    tm, hen = pamiec_tm(), jezyk.hasla_en()
    slownik = slowniczek_pewny(tm, wczytaj_karty())
    jezyk.log(f"[e8] tematów {len(zz)}, pamięć tłumaczeń {len(tm)}, słowniczek {len(slownik)}, hasła EN {len(hen)}")
    konfig = "e8" if gen == 1 else f"e8_g{gen}"
    (wyniki / "odpowiedzi").mkdir(parents=True, exist_ok=True)
    pliki = []

    def praca(m, s):
        p = wyniki / "odpowiedzi" / f"{m}__{konfig}.jsonl"
        pliki.append(p)
        gotowe = {i for i, d in jezyk.wczytaj_odp(p).items() if not jezyk.blad(d.get("odpowiedz"))}
        todo = [z for z in zz if z["id"] not in gotowe]
        jezyk.log(f"[e8 {m}/{konfig}] do zrobienia {len(todo)} z {len(zz)}")
        lock = threading.Lock()

        def jedno(z):
            try:
                t, sek, meta = napisz_e8(z, url=s.url, karty=karty, baza_hasel=baza, marian=marian, tm=tm,
                                         hasla_en=hen, slownik=slownik, bez_myslenia=s.bez_myslenia,
                                         stanowisko=stanowisko)
            except Exception as e:  # noqa: BLE001 (zapis błędu; wznowienie ponowi)
                t, sek, meta = f"(BŁĄD: {e})", 0.0, {}
            with lock, open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": z["id"], "model": m, "konfig": konfig, "odpowiedz": t, "sekundy": round(sek, 2),
                                    "meta": meta}, ensure_ascii=False) + "\n")
            jezyk.log(f"[e8 {m}] {z['id']}: {meta.get('slow')} słów, {sek:.0f} s")

        with ThreadPoolExecutor(max(1, min(rownolegle, s.rown))) as ex:
            list(ex.map(jedno, todo))

    wyniki.mkdir(parents=True, exist_ok=True)
    try:
        jezyk.WYN = wyniki
        jezyk.na_modelach(modele, praca, porty=PORTY_E8)
    finally:
        jezyk.WYN = stare[0]
    return pliki


# ---------------------------------------------------------------- wiersz poleceń

def _serwer_tl(port: int, model: str = MODEL_TL):
    from .noc import Serwer
    gguf, bez_myslenia, sloty = jezyk.MODELE[model]
    log = PLIK_TM.parent / f"serwer_{model}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return Serwer(str(jezyk.LLAMA), ROOT / "data" / "modele" / gguf, None, port, sloty, 8192, log), bez_myslenia, sloty


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Esej e8 (akapity EN, Marian EN→PL, rama PL).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tlumacz-karty", help="pamięć tłumaczeń kart kanonicznych PL→EN (lokalnie, GPU)")
    t.add_argument("--port", type=int, default=PORT_TL)
    t.add_argument("--model", default=MODEL_TL, help="tłumacz z jezyk.MODELE (budowa bazy, poza limitem rozmiaru)")
    t.add_argument("--limit", type=int, help="tylko N pierwszych kart (próba) do osobnego pliku w data/karty/tlumaczenia")
    g = sub.add_parser("generuj", help="eseje e8 na tematy arkuszy (pomiar; ocena: matura/pelna_matura.py)")
    g.add_argument("--modele", required=True, help="nazwy z jezyk.MODELE po przecinku (do 3 naraz, porty 8097-8099)")
    g.add_argument("--sesje", required=True, help="np. 2024-maj,2025-maj (2026 niedozwolone)")
    g.add_argument("--gen", type=int, default=1, help="numer generacji (plik __e8 albo __e8_g<N>)")
    g.add_argument("--wyniki", default=str(WYN_E8.relative_to(ROOT)))
    g.add_argument("--stanowisko", choices=tuple(KIERUNEK_E8), default="zgadzam")
    g.add_argument("--limit", type=int)
    g.add_argument("--rownolegle", type=int, default=4, help="eseje naraz na model")
    pr = sub.add_parser("proba", help="jeden model, tematy podane w wierszu (spoza oceny): tekst EN i PL na ekran")
    pr.add_argument("--model", required=True)
    pr.add_argument("--port", type=int, default=8096)
    pr.add_argument("--temat", action="append", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "generuj":
        sesje = tuple(x.strip() for x in a.sesje.split(",") if x.strip())
        pliki = generuj([m.strip() for m in a.modele.split(",")], sesje, a.gen, ROOT / a.wyniki, a.stanowisko,
                        a.limit, a.rownolegle)
        for p in pliki:
            print(p.relative_to(ROOT))
        return 0
    if a.cmd == "proba":
        from .harness import _baza_hasel
        from .karty import Karty
        from .noc import Serwer
        from .tlumacz import Marian
        gguf, bez_myslenia, sloty = jezyk.MODELE[a.model]
        log = ROOT / "review" / "esej-e8-2026-09-26" / "logi" / f"proba_{a.model}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        s = Serwer(str(jezyk.LLAMA), ROOT / "data" / "modele" / gguf, None, a.port, sloty, 8192, log)
        try:
            tm = pamiec_tm()
            for temat in a.temat:
                tekst, sek, meta = napisz_e8({"temat": temat}, url=s.url, karty=Karty(), baza_hasel=_baza_hasel(),
                                             marian=Marian(), tm=tm, hasla_en=jezyk.hasla_en(),
                                             slownik=slowniczek_pewny(tm, wczytaj_karty()), bez_myslenia=bez_myslenia)
                print("=" * 100, f"\n{temat}\n{sek:.0f} s | meta: " + json.dumps(
                    {k: v for k, v in meta.items() if k != "akapity_en"}, ensure_ascii=False))
                for en in meta["akapity_en"]:
                    print("--- EN:", en)
                print("--- PL:\n" + tekst)
        finally:
            s.stop()
        return 0
    if a.cmd == "tlumacz-karty":
        s, bez_myslenia, sloty = _serwer_tl(a.port, a.model)
        try:
            if a.limit:
                karty = wczytaj_karty()[:a.limit]
                c = jezyk.Cache(PLIK_TM.with_name("proba_en.jsonl"))
                f = tlumacz_linie([k["fakt"] for k in karty], c, url=s.url, partia=12, bez_myslenia=bez_myslenia,
                                  rownolegle=sloty)
                n = tlumacz_linie(nazwy_kart(karty), c, url=s.url, partia=10, bez_myslenia=bez_myslenia,
                                  rownolegle=sloty)
                for pl, en in list(f.items())[:8] + list(n.items())[:12]:
                    print(f"{pl}\n  → {en}")
                print(f"fakty {len(f)}/{len(karty)}, nazwy {len(n)}/{len(nazwy_kart(karty))}")
                return 0
            wyn = tlumacz_karty(s.url, bez_myslenia, sloty)
            return 0 if wyn["fakty"] == wyn["fakty_wszystkie"] else 1
        finally:
            s.stop()
    return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
