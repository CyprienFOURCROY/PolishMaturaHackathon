"""Obrazy źródeł → tekst dla modelu tekstowego (etap B0): poprawione przypisanie, OCR napisów, opis sceny.

Co to jest: narzędzie, które zamienia ilustracje zadania CKE (fotografie, mapy, karykatury, plakaty…) na tekst,
bo modele tekstowe (Bielik, Qwen bez mmproj) widzą dziś tylko podpis źródła. Małe modele lepiej radzą sobie
z opisem pisemnym niż z samym obrazem (benchmark jurorki). Zasada rozmiaru zestawu: liczy się największy model,
więc narzędzie jest małe: EasyOCR ['pl'] (~95 MB, data/modele/easyocr) + opcjonalnie Qwen3.5-0.8B z mmproj.

Po co:
1. `ilustracje(z)` / `obrazy_zadania(z)`: poprawka błędu devset.py (pole `obrazy` = obrazy ze STRONY arkusza, nie
   zadania; np. 2025-maj-7.1 dostawało plan miasta z zadania 6). Obraz należy do zadania, którego nagłówek
   „Zadanie N.” poprzedza go w kolejności czytania PDF (pozycje z pypdf); nagłówek „Źródło K.” nad obrazem daje
   numer źródła. Obraz po nagłówku podzadania „Zadanie N.M.” należy tylko do tego podzadania. Jednolite obrazy
   (np. czarne ramki tablicy genealogicznej z kanałem alfa) są pomijane. Bez PDF (albo gdy nie znaleziono
   wszystkich nagłówków) działa heurystyka słów: obraz zostaje przy zadaniu ze źródłem wizualnym (fotografia,
   mapa, plakat…), a zadanie z samymi źródłami tekstowymi nie dostaje obrazów, które ma zadanie wizualne ze strony.
2. Cache wyników w data/cke/obrazy_opisy.jsonl (klucz: ścieżka + sha1 pliku + wariant), wznawialny:
   wariant "ocr" (surowe detekcje EasyOCR z pewnością i ramką), wariant "opis" (opis sceny z Qwen3.5-0.8B).
3. `blok_opisu(z, "ocr" | "ocr_opis")`: blok „OPIS ILUSTRACJI (źródło N…): napisy: …; opis: …” do promptu
   (konfiguracje h0_ocr i h0_opis w matura/harness.py). Czyta tylko cache; brak wpisu → BrakOpisu.

Co zrobić (maszyna z GPU; opis startuje własny llama-server na porcie 8096):
    uv run python -m matura.obrazy --sesje 2024-maj,2025-maj --warianty ocr,opis
    uv run python -m matura.obrazy --wszystkie --warianty ocr,opis      # wszystkie sesje dev/test
    uv run python -m matura.obrazy --sesje 2025-maj --tabela             # przypisanie przed/po, bez liczenia
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
KAT_CKE = ROOT / "data" / "cke"
JSON_CKE = KAT_CKE / "json"
CACHE = KAT_CKE / "obrazy_opisy.jsonl"
MODELE_OCR = ROOT / "data" / "modele" / "easyocr"
QWEN_VL = ROOT / "data" / "modele" / "qwen3.5-0.8b-q4"
GGUF_OPIS, MMPROJ_OPIS = QWEN_VL / "Qwen3.5-0.8B-Q4_K_M.gguf", QWEN_VL / "mmproj-F16.gguf"
LLAMA_SERVER = ROOT / "data" / "bin" / "llama-cuda" / "llama-server"
PORT_OPIS = 8096
PROG_OCR = 0.4          # minimalna pewność linii OCR
MAG_OCR = 2.0           # powiększenie EasyOCR dla obrazów o boku < 1600 px
SILNIK_OCR = "easyocr-1.7.2-pl-mag2"
MAX_NAPISOW, MAX_ZNAKOW_NAPISOW, MAX_SLOW_OPISU = 60, 700, 130

SLOWA_WIZUALNE = re.compile(
    r"fotografi|zdjęci|\bmap[aęyie]\b|\bplan\b|ilustracj|rysun|rycin|drzeworyt|miedzioryt|litografi|grafik|"
    r"karykatur|plakat|afisz|ulotk|\bobraz|portret|fresk|mozaik|witraż|miniatur|kadr|monet|medal|banknot|"
    r"pieczęć|pieczęci|herb|znacz(?:ek|ki|ka)|rzeźb|relief|płaskorzeźb|pomnik|schemat|wykres|diagram|"
    r"genealogiczn|okładk|stron[ay] tytułow|budowl|naczy[ńn]", re.I)
RE_NA_PODSTAWIE = re.compile(r"^\s*(?:Na?\s+podstawie|https?://|www\.)", re.I)
RE_NAGL_ZAD = re.compile(r"^\s*Zadanie\s*(\d+)(?:\s*\.\s*(\d+))?")
RE_NAGL_ZR = re.compile(r"^\s*Źródło\s*(\d+)\.?\s*(.*)")
RE_ID = re.compile(r"^(\d{4}-(?:maj|czerwiec))-")

SYS_OPIS = "Opisujesz ilustrację ze źródła historycznego osobie, która jej nie widzi. Piszesz po polsku, rzeczowo, bez wstępów."
USER_OPIS = """Opisz tę ilustrację w 60–120 słowach, w kolejności:
1. Typ źródła (fotografia, mapa, plan, karykatura, plakat, rysunek, obraz, rzeźba, moneta, medal, banknot, znaczek, pieczęć, wykres, schemat, okładka).
2. Kto lub co jest przedstawione.
3. Widoczne napisy.
4. Symbole i atrybuty postaci.
5. Tylko dla mapy lub planu: obszary, granice, strzałki, legenda.
Pomiń punkt, jeśli nie dotyczy ilustracji. Nie zgaduj nazwisk, dat ani nazw, których nie widać."""

NAGLOWEK_BLOKU = "Ilustracje zadania zapisane tekstem (automatyczny odczyt napisów i opis obrazu, mogą zawierać błędy):"

_LOCK_PRZYP, _LOCK_CACHE, _LOCK_OCR = threading.Lock(), threading.Lock(), threading.Lock()


class BrakOpisu(RuntimeError):
    """Brak wyniku OCR/opisu w cache: policz go CLI `python -m matura.obrazy`."""


# ---------------------------------------------------------------- przypisanie obrazów do zadań

def sesja_zadania(z: dict) -> str | None:
    if z.get("rok") and z.get("sesja"):
        return f"{z['rok']}-{z['sesja']}"
    m = RE_ID.match(str(z.get("id", "")))
    return m.group(1) if m else None


def _wzgl(p: str | Path) -> str:
    """Ścieżka względem katalogu projektu (klucz cache niezależny od maszyny); spoza projektu: bez zmian."""
    p = Path(p)
    try:
        return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)
    except ValueError:
        return str(p)


@lru_cache(maxsize=1)
def _ozdobniki() -> frozenset[str]:
    """sha1 obrazów identycznych w ≥3 sesjach (logo egzaminu „EM23” na ostatniej stronie arkusza itp.)."""
    ile: dict[str, set[str]] = {}
    for p in (KAT_CKE / "obrazy").glob("*/*.png"):
        ile.setdefault(_sha1(p), set()).add(p.parent.name)
    return frozenset(h for h, s in ile.items() if len(s) >= 3)


@lru_cache(maxsize=None)
def _pusty(rel: str) -> bool:
    """Obraz jednolity (np. czarna ramka zapisana z kanału alfa): nic nie wnosi, pomijamy."""
    from PIL import Image, ImageStat
    p = ROOT / rel
    if not p.exists():
        return True
    return ImageStat.Stat(Image.open(p).convert("L")).stddev[0] < 3


def zrodla(z: dict) -> list[dict]:
    """Nagłówki źródeł zadania: [{nr, podpis, wizualne}]. nr=None, gdy zadanie ma jedno nienumerowane źródło."""
    tekst = (z.get("zrodla_tekst") or "") + "\n" + (z.get("polecenie") or "")
    linie = [l.strip() for l in tekst.splitlines() if l.strip()]
    out, widziane = [], set()
    for i, l in enumerate(linie):
        m = RE_NAGL_ZR.match(l)
        if m and (m.group(1), m.group(2)) not in widziane:  # CKE bywa niespójne: dwa razy „Źródło 1.”
            widziane.add((m.group(1), m.group(2)))
            podpis = f"Źródło {m.group(1)}. {m.group(2)}".strip()
            dalej = linie[i + 1] if i + 1 < len(linie) else ""
            out.append({"nr": int(m.group(1)), "podpis": podpis,
                        "wizualne": bool(SLOWA_WIZUALNE.search(m.group(2)) or RE_NA_PODSTAWIE.match(dalej))})
    if out:
        return out
    # jedno źródło bez numeru: pierwsza linia po nagłówku „Zadanie N.” (np. „Plan miasta”)
    for i, l in enumerate(linie[:-1]):
        if RE_NAGL_ZAD.match(l):
            tyt, dalej = linie[i + 1], (linie[i + 2] if i + 2 < len(linie) else "")
            if RE_NAGL_ZAD.match(tyt):
                continue
            wiz = bool(SLOWA_WIZUALNE.search(tyt) or RE_NA_PODSTAWIE.match(dalej))
            return [{"nr": None, "podpis": tyt, "wizualne": wiz}]
    return []


def _wczytaj_sesje(sesja: str) -> list[dict]:
    p = JSON_CKE / f"historia-{sesja}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _uklad_pdf(sesja: str) -> list[tuple] | None:
    """Zdarzenia w kolejności czytania arkusza: ("H", glowny, pod) nagłówek zadania, ("Z", nr) nagłówek źródła,
    ("I", rel, strona, y, x) obraz. None, gdy brak PDF albo nie da się umiejscowić któregoś zapisanego obrazu."""
    pdf = KAT_CKE / f"historia-{sesja}-arkusz.pdf"
    if not pdf.exists():
        return None
    import logging
    import pypdf
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    zdarzenia = []
    for nr_str, pg in enumerate(pypdf.PdfReader(str(pdf)).pages):
        if nr_str < 2:  # strona tytułowa i instrukcja
            continue
        kawalki, rysunki = [], {}

        def vt(text, cm, tm, fd, fs):
            if text.strip() and abs(tm[1]) < 0.01 and abs(cm[1]) < 0.01:
                x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
                if x >= 0:
                    kawalki.append((tm[4] * cm[1] + tm[5] * cm[3] + cm[5], x, text))

        def vo(op, args, cm, tm):
            if op == b"Do" and args:
                rysunki.setdefault(str(args[0]).lstrip("/"), []).append((cm[5] + cm[3] / 2, cm[4] + cm[0] / 2))

        pg.extract_text(visitor_text=vt, visitor_operand_before=vo)
        linie: dict[float, list] = {}
        for y, x, t in kawalki:
            k = next((k for k in linie if abs(k - y) < 1.5), y)
            linie.setdefault(k, []).append((x, t))
        for y, cz in linie.items():
            cz = sorted(c for c in cz if c[0] >= 55)  # bez kratek na marginesie („26.” przy x≈36)
            if not cz or cz[0][0] > 120:  # nagłówki tylko przy lewym marginesie tekstu (x≈71)
                continue
            t = "".join(c for _, c in cz)
            if m := RE_NAGL_ZAD.match(t):
                zdarzenia.append((nr_str, -y, 0, ("H", m.group(1), m.group(2))))
            elif m := RE_NAGL_ZR.match(t):
                zdarzenia.append((nr_str, -y, 0, ("Z", int(m.group(1)))))
        for k, im in enumerate(pg.images):
            rel = f"data/cke/obrazy/{sesja}/str{nr_str + 1:02d}_{k}.png"
            if not (ROOT / rel).exists():
                continue  # za mały albo ozdobnik (scripts/parsuj_arkusze.py)
            poz = rysunki.get(im.name.rsplit(".", 1)[0])
            if not poz:
                return None
            y, x = max(poz)
            zdarzenia.append((nr_str, -y, x, ("I", rel, nr_str, y, x)))
    zdarzenia.sort(key=lambda e: e[:3])
    return [e[3] for e in zdarzenia]


def _przypisz_z_pdf(sesja: str, zadania: list[dict]) -> dict[str, list[dict]] | None:
    uklad = _uklad_pdf(sesja)
    if uklad is None:
        return None
    glowne_pdf = {e[1] for e in uklad if e[0] == "H"}
    if any(z["nr_glowny"] not in glowne_pdf for z in zadania if not z["esej"]):
        return None  # nie znaleziono nagłówka któregoś zadania → pozycje niewiarygodne
    po_nr: dict[tuple, list[tuple]] = {}
    glowny = pod = zr = None
    for e in uklad:
        if e[0] == "H":
            glowny, pod, zr = e[1], e[2], None
        elif e[0] == "Z":
            zr = e[1]
        elif glowny is not None:
            po_nr.setdefault((glowny, pod), []).append((e[1], zr, e[2], e[3], e[4]))
    out = {}
    for z in zadania:
        nr = z["nr_zadania"]
        klucze = [(z["nr_glowny"], None)] + ([(z["nr_glowny"], nr.split(".", 1)[1])] if "." in nr else [])
        out[z["id"]] = [{"rel": rel, "zrodlo": zr, "strona": s, "y": y, "x": x}
                        for k in klucze for rel, zr, s, y, x in po_nr.get(k, [])]
    return out


def _przypisz_slowami(zadania: list[dict]) -> dict[str, list[dict]]:
    """Heurystyka bez PDF: obraz strony zostaje przy zadaniach ze źródłem wizualnym; gdy na stronie żadne zadanie
    nie ma źródła wizualnego, zostaje przy wszystkich (nie wiadomo czyj)."""
    wiz = {z["nr_glowny"]: any(zr["wizualne"] for zr in zrodla(z)) for z in zadania}
    kand: dict[str, set[str]] = {}
    for z in zadania:
        for o in z["obrazy"]:
            kand.setdefault(_wzgl(o), set()).add(z["nr_glowny"])
    out = {}
    for z in zadania:
        wiz_zr = [zr["nr"] for zr in zrodla(z) if zr["wizualne"]]
        lista = []
        for o in z["obrazy"]:
            k = kand[_wzgl(o)]
            if wiz[z["nr_glowny"]] or not any(wiz[g] for g in k):
                lista.append({"rel": _wzgl(o), "zrodlo": wiz_zr[0] if len(wiz_zr) == 1 else None, "strona": 0, "y": 0, "x": 0})
        out[z["id"]] = lista
    return out


@lru_cache(maxsize=None)
def _przypisanie_sesji(sesja: str) -> tuple[str, dict[str, list[dict]]]:
    """(metoda, {id zadania → [ilustracja]}) dla sesji CKE; metoda = "pdf" albo "slowa"."""
    zadania = _wczytaj_sesje(sesja)
    wynik = _przypisz_z_pdf(sesja, zadania)
    metoda = "pdf"
    if wynik is None:
        wynik, metoda = _przypisz_slowami(zadania), "slowa"
    return metoda, wynik


def _grupuj_w_wiersze(lista: list[dict]) -> list[dict]:
    """Kolejność czytania obrazów jednego źródła: wiersz (strona, y) → od lewej (x)."""
    out, wiersz = [], []
    for il in sorted(lista, key=lambda i: (i["strona"], -i["y"])):
        if wiersz and (il["strona"] != wiersz[0]["strona"] or abs(il["y"] - wiersz[0]["y"]) > 25):
            out += sorted(wiersz, key=lambda i: i["x"]); wiersz = []
        wiersz.append(il)
    return out + sorted(wiersz, key=lambda i: i["x"])


def ilustracje(z: dict) -> list[dict]:
    """Ilustracje zadania po poprawce przypisania: [{sciezka (absolutna), rel, zrodlo, podpis, i, n}]."""
    if z.get("esej"):
        return []
    if z.get("ilustracje_egz") is not None:  # paczka organizatorów: obrazy już wycięte i przypisane do pozycji
        return z["ilustracje_egz"]
    sesja = sesja_zadania(z)
    if sesja and (JSON_CKE / f"historia-{sesja}.json").exists():
        with _LOCK_PRZYP:
            _, przyp = _przypisanie_sesji(sesja)
        lista = przyp.get(z["id"])
        if lista is None:
            lista = _przypisz_slowami([z])[z["id"]]
    else:
        lista = _przypisz_slowami([z])[z["id"]]
    lista = [il for il in lista if not _pusty(il["rel"]) and _sha1(ROOT / il["rel"]) not in _ozdobniki()]
    zr: dict = {}
    for s in zrodla(z):  # przy powtórzonym numerze wygrywa źródło wizualne
        if s["nr"] not in zr or (s["wizualne"] and not zr[s["nr"]]["wizualne"]):
            zr[s["nr"]] = s
    zr = {k: v["podpis"] for k, v in zr.items()}
    grupy: dict = {}
    for il in lista:
        grupy.setdefault(il["zrodlo"], []).append(il)
    out = []
    for nr, g in grupy.items():
        g = _grupuj_w_wiersze(g)
        podpis = zr.get(nr) or (zr.get(None) if nr is None else f"Źródło {nr}.")
        if nr is None and podpis is None and len(zr) == 1:
            podpis = next(iter(zr.values()))
        for i, il in enumerate(g, 1):
            out.append({"sciezka": str(ROOT / il["rel"]), "rel": il["rel"], "zrodlo": nr,
                        "podpis": (podpis or "").strip()[:90], "i": i, "n": len(g)})
    return out


def obrazy_zadania(z: dict) -> list[str]:
    """Poprawione pole `obrazy`: absolutne ścieżki obrazów, które naprawdę należą do zadania."""
    return [il["sciezka"] for il in ilustracje(z)]


# ---------------------------------------------------------------- cache

_CACHE: dict[str, dict] | None = None


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def klucz(rel: str, wariant: str) -> str:
    return f"{rel}|{_sha1(ROOT / rel)}|{wariant}"


def wczytaj_cache(odswiez: bool = False) -> dict[str, dict]:
    global _CACHE
    if _CACHE is None or odswiez:
        _CACHE = {}
        if CACHE.exists():
            for l in CACHE.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    d = json.loads(l)
                    _CACHE[d["klucz"]] = d
    return _CACHE


def _zapisz(rel: str, wariant: str, wynik, sekundy: float, **extra) -> dict:
    d = {"klucz": klucz(rel, wariant), "sciezka": rel, "wariant": wariant, "wynik": wynik,
         "sekundy": round(sekundy, 3), "czas": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    with _LOCK_CACHE:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        wczytaj_cache()[d["klucz"]] = d
    return d


def z_cache(rel: str, wariant: str) -> dict:
    k = klucz(rel, wariant)
    d = wczytaj_cache().get(k) or wczytaj_cache(odswiez=True).get(k)
    if d is None:
        raise BrakOpisu(f"brak wariantu {wariant!r} dla {rel} w {CACHE.name}: "
                        f"uruchom uv run python -m matura.obrazy --sesje <sesja> --warianty ocr,opis")
    return d


# ---------------------------------------------------------------- OCR (EasyOCR ['pl'], GPU)

_OCR = None


def _czytnik():
    global _OCR
    if _OCR is None:
        import easyocr
        MODELE_OCR.mkdir(parents=True, exist_ok=True)
        _OCR = easyocr.Reader(["pl"], gpu=True, model_storage_directory=str(MODELE_OCR),
                              user_network_directory=str(MODELE_OCR / "user"), verbose=False)
    return _OCR


def ocr_surowy(sciezka: str | Path) -> list[dict]:
    """Detekcje EasyOCR: [{t: tekst, p: pewność, b: [x0, y0, x1, y1]}] (bez progu, w kolejności silnika)."""
    from PIL import Image
    with Image.open(sciezka) as im:
        mag = MAG_OCR if max(im.size) < 1600 else 1.0  # małe znaczki i monety: lepiej po powiększeniu
    wynik = _czytnik().readtext(str(sciezka), mag_ratio=mag)
    out = []
    for box, t, p in wynik:
        xs, ys = [float(q[0]) for q in box], [float(q[1]) for q in box]
        out.append({"t": str(t), "p": round(float(p), 3), "b": [min(xs), min(ys), max(xs), max(ys)]})
    return out


def kolejnosc_czytania(det: list[dict]) -> list[dict]:
    """Wiersze z góry na dół (środek ramki w połowie wysokości poprzedniej), w wierszu od lewej."""
    linie: list[dict] = []
    for d in sorted(det, key=lambda d: (d["b"][1] + d["b"][3]) / 2):
        yc, h = (d["b"][1] + d["b"][3]) / 2, d["b"][3] - d["b"][1]
        if linie and abs(yc - linie[-1]["yc"]) <= 0.5 * max(h, linie[-1]["h"]):
            linie[-1]["el"].append(d)
        else:
            linie.append({"yc": yc, "h": h, "el": [d]})
    return [d for l in linie for d in sorted(l["el"], key=lambda d: d["b"][0])]


def napisy(det: list[dict], prog: float = PROG_OCR) -> list[str]:
    """Linie tekstu z pewnością ≥ progu, w kolejności czytania."""
    return [d["t"].strip() for d in kolejnosc_czytania([d for d in det if d["p"] >= prog]) if d["t"].strip()]


def policz_ocr(rel: str, *, nadpisz: bool = False) -> dict:
    if not nadpisz:
        try:
            return z_cache(rel, "ocr")
        except BrakOpisu:
            pass
    t0 = time.perf_counter()
    with _LOCK_OCR:  # EasyOCR na GPU nie jest bezpieczny wątkowo
        det = ocr_surowy(ROOT / rel)
    return _zapisz(rel, "ocr", det, time.perf_counter() - t0, silnik=SILNIK_OCR)


# ---------------------------------------------------------------- opis sceny (Qwen3.5-0.8B + mmproj)

ETYKIETY_OPISU = [(r"Typ źródła", "Typ"), (r"Kto lub co jest (?:przedstawione|na ilustracji)", "Przedstawia"),
                  (r"Widoczne napisy", "Napisy"), (r"Symbole i atrybuty(?: postaci)?", "Symbole"),
                  (r"(?:Tylko )?[Dd]la mapy lub planu", "Mapa")]


def czysc_opis(t: str) -> str:
    """Jedna linia: bez markdown, numeracji i echa promptu (lista typów), krótkie etykiety („Typ: mapa; Przedstawia: …”),
    bez pustych punktów „nie dotyczy” i komentarzy modelu, bez powtórzeń, ≤ MAX_SLOW_OPISU słów."""
    t = re.sub(r"\*\*|__|`|^#+\s*", "", t, flags=re.M)
    t = re.sub(r"\s*\((?:[^()]*,){4,}[^()]*\)", "", t)  # przepisana z promptu lista typów źródeł
    linie = []
    for l in t.splitlines():
        l = re.sub(r"^\s*(?:[-•*]\s*|\d+\.\s*)", "", l).strip()
        if not l or re.match(r"^(?:Oto|Opis)\b[^.]*:$", l) or re.match(r"^Pomi(?:jam|ną|nął|nie|n)", l):
            continue
        if re.search(r"mapy lub planu|dla mapy", l, re.I) and re.search(r"nie ma|nie dotyczy|brak|nie jest", l, re.I):
            continue
        for wz, kr in ETYKIETY_OPISU:
            l = re.sub(rf"^{wz}\s*(?:[–—:-]\s*)+", kr + ": ", l).strip()
        if l.endswith(":") and re.fullmatch(r"\w+:", l):  # sama etykieta, treść w kolejnych liniach
            linie.append(l); continue
        if l not in linie:
            linie.append(l if l[-1] in ".!?;:…" else l + ";")
    zdania, out = re.split(r"(?<=[.!?;])\s+", " ".join(linie)), []
    for s in zdania:
        if s and s not in out:
            out.append(s)
    slowa = " ".join(out).split()
    tekst = " ".join(slowa[:MAX_SLOW_OPISU]) + (" …" if len(slowa) > MAX_SLOW_OPISU else "")
    return tekst.rstrip(";").strip()


def opisz_obraz(url: str, sciezka: str | Path, *, max_tokens: int = 300, timeout: int = 300) -> tuple[str, str, float]:
    """(opis oczyszczony, surowa odpowiedź, sekundy) ze stałym promptem; temperatura 0 i kara za powtórzenia."""
    from .llm import obraz_url
    body = {"messages": [{"role": "system", "content": SYS_OPIS},
                         {"role": "user", "content": [{"type": "text", "text": USER_OPIS},
                                                      {"type": "image_url", "image_url": {"url": obraz_url(str(sciezka))}}]}],
            "max_tokens": max_tokens, "temperature": 0.0, "repeat_penalty": 1.15,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    surowy = (r.json()["choices"][0]["message"].get("content") or "").strip()
    return czysc_opis(surowy), surowy, time.perf_counter() - t0


def policz_opis(rel: str, url: str, *, nadpisz: bool = False) -> dict:
    if not nadpisz:
        try:
            return z_cache(rel, "opis")
        except BrakOpisu:
            pass
    opis, surowy, s = opisz_obraz(url, ROOT / rel)
    return _zapisz(rel, "opis", opis, s, surowy=surowy, model=GGUF_OPIS.name)


# ---------------------------------------------------------------- blok do promptu

def _etykieta(il: dict) -> str:
    podpis = il["podpis"]
    if il["zrodlo"] is not None:
        e = podpis if podpis.lower().startswith("źródło") else f"źródło {il['zrodlo']}. {podpis}".strip()
    else:
        e = f"źródło: {podpis}" if podpis else "źródło"
    return e + (f", ilustracja {il['i']} z {il['n']}" if il["n"] > 1 else "")


def blok_opisu(z: dict, wariant: str = "ocr") -> str:
    """Blok tekstu o ilustracjach zadania (pusty, gdy zadanie nie ma ilustracji). wariant: "ocr" | "ocr_opis" |
    "wzorzec*" (opis Claude z wizją, tylko do pomiaru sufitu na dev) | "vlm_*" (lokalni kandydaci na opisywacz).
    Tylko z cache: brak wpisu → BrakOpisu."""
    if wariant not in ("ocr", "ocr_opis") and not wariant.startswith(("wzorzec", "vlm_")):
        raise ValueError(f"nieznany wariant bloku {wariant}")
    ilu = ilustracje(z)
    if not ilu:
        return ""
    linie = [NAGLOWEK_BLOKU]
    for il in ilu:
        if wariant.startswith(("wzorzec", "vlm_")):  # sam opis z cache: wzorzec* = pomiar sufitu (Claude, tylko dev),
            # vlm_* = kandydaci na lokalny opisywacz (scripts/opisy_kandydaci.py)
            linie.append(f"OPIS ILUSTRACJI ({_etykieta(il)}): {z_cache(il['rel'], wariant)['wynik']}")
            continue
        nap = napisy(z_cache(il["rel"], "ocr")["wynik"])
        tekst_nap = " | ".join(nap[:MAX_NAPISOW]) + (" | …" if len(nap) > MAX_NAPISOW else "")
        if len(tekst_nap) > MAX_ZNAKOW_NAPISOW:
            tekst_nap = tekst_nap[:MAX_ZNAKOW_NAPISOW].rsplit(" | ", 1)[0] + " | …"
        l = f"OPIS ILUSTRACJI ({_etykieta(il)}): napisy: {tekst_nap or '(brak)'}"
        if wariant == "ocr_opis":
            d = z_cache(il["rel"], "opis")
            opis = czysc_opis(d["surowy"]) if d.get("surowy") else d["wynik"]  # bieżące czyszczenie, bez ponownego liczenia
            l += f"; opis: {opis or '(brak)'}"
        linie.append(l)
    return "\n".join(linie)


# ---------------------------------------------------------------- opisywacz na żywo (egzamin)
# Prompt i wywołanie przeniesione z scripts/opisy_kandydaci.py (bez zmian treści), żeby egzamin (matura/egzamin.py)
# opisywał obrazy nowej paczki tym samym opisywaczem co pomiary (wariant vlm_q2b_en: Qwen3.5-2B Q4_K_M + mmproj F16).
SERWER_VLM_ARGS = ["--image-min-tokens", "1024"]   # jak przy tworzeniu opisów vlm_* do pomiarów
PROMPTY_VLM = {
    "pl": ("Opisujesz ilustracje ze źródeł historycznych dla ucznia, który ich nie widzi.",
           "Podpis źródła w arkuszu: „{podpis}”.\nOpisz ilustrację rzeczowo po polsku (60-150 słów, zwykły tekst): "
           "1) typ źródła; 2) wszystkie widoczne napisy dosłownie (daty, nazwy, hasła, legenda mapy); 3) co i kto jest "
           "przedstawiony: postacie, symbole, atrybuty; na mapie obszary, granice, strzałki, miasta; 4) jeśli rozpoznajesz "
           "znany obiekt, postać albo wydarzenie, nazwij je. Nie zgaduj, gdy nie jesteś pewien."),
    "en": ("You describe illustrations from historical sources for a student who cannot see them.",
           "Caption of the source in the exam sheet (Polish): \"{podpis}\".\nDescribe the illustration factually in English "
           "(60-150 words, plain text): 1) type of source; 2) all visible text verbatim, in the original language (dates, "
           "names, slogans, map legend); 3) what and who is depicted: figures, symbols, attributes; for maps: areas, borders, "
           "arrows, cities; 4) if you recognise a well-known object, person or event, name it. Do not guess when unsure."),
}


def opisz_vlm(url: str, il: dict, jezyk: str, timeout: int = 300) -> tuple[str, float]:
    """Opis jednej ilustracji przez llama-server z mmproj (t=0, 350 tokenów) → (tekst, sekundy)."""
    from .llm import obraz_url
    sys_, user = PROMPTY_VLM[jezyk]
    body = {"messages": [{"role": "system", "content": sys_},
                         {"role": "user", "content": [{"type": "text", "text": user.format(podpis=il["podpis"] or "(brak)")},
                                                      {"type": "image_url", "image_url": {"url": obraz_url(il["sciezka"])}}]}],
            "max_tokens": 350, "temperature": 0.0, "repeat_penalty": 1.15, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    t = (r.json()["choices"][0]["message"].get("content") or "").strip()
    t = re.sub(r"\*\*|__|`|^#+\s*", "", t, flags=re.M)
    return " ".join(t.split()), time.perf_counter() - t0


def opisz_brakujace(ilustracje_lista: list[dict], url: str, wariant: str, jezyk: str = "en", rownolegle: int = 4,
                    opisz_fn=None, **meta) -> int:
    """Opisuje ilustracje bez wpisu `wariant` w cache (unikalne po `rel`) i zapisuje je (`_zapisz`); zwraca liczbę
    nowych opisów. Potem `blok_opisu(z, wariant)` działa także dla paczki egzaminu."""
    from concurrent.futures import ThreadPoolExecutor
    opisz_fn = opisz_fn or opisz_vlm
    todo = {}
    for il in ilustracje_lista:
        if klucz(il["rel"], wariant) not in wczytaj_cache():
            todo.setdefault(il["rel"], il)

    def jeden(il):
        tekst, sek = opisz_fn(url, il, jezyk)
        _zapisz(il["rel"], wariant, tekst, sek, jezyk=jezyk, **meta)

    with ThreadPoolExecutor(max(1, rownolegle)) as ex:
        list(ex.map(jeden, todo.values()))
    return len(todo)


# ---------------------------------------------------------------- CLI

def _sesje_cli(a) -> list[str]:
    from .devset import SESJE_CKE
    if a.wszystkie:
        return [s for s in SESJE_CKE if (JSON_CKE / f"historia-{s}.json").exists()]
    return [s for s in (a.sesje or "").split(",") if s]


def tabela(sesja: str) -> tuple[list[str], dict]:
    """Wiersze tabeli zadanie → obrazy przed/po poprawce i podsumowanie (zadania, punkty z ilustracją)."""
    from .devset import wczytaj
    zad = [z for z in wczytaj([sesja]) if not z["esej"]]
    metoda, _ = _przypisanie_sesji(sesja)
    wiersze = [f"### {sesja} (metoda: {metoda})", "| zadanie | pkt | przed | po | źródła (W = wizualne) |", "|---|---|---|---|---|"]
    podsum = {"zadania": len(zad), "pkt": sum(z["pkt_max"] or 0 for z in zad), "przed_zad": 0, "przed_pkt": 0,
              "po_zad": 0, "po_pkt": 0, "metoda": metoda}
    for z in zad:
        przed = [Path(o).stem for o in z["obrazy"]]
        po = [f"{Path(il['rel']).stem}(zr{il['zrodlo'] or '-'})" for il in ilustracje(z)]
        podsum["przed_zad"] += bool(przed); podsum["przed_pkt"] += (z["pkt_max"] or 0) * bool(przed)
        podsum["po_zad"] += bool(po); podsum["po_pkt"] += (z["pkt_max"] or 0) * bool(po)
        zr = "; ".join(("W " if s["wizualne"] else "") + s["podpis"][:45] for s in zrodla(z))
        wiersze.append(f"| {z['id']} | {z['pkt_max']} | {', '.join(przed) or '-'} | {', '.join(po) or '-'} | {zr} |")
    return wiersze, podsum


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sesje", help="lista sesji po przecinku, np. 2024-maj,2025-maj")
    ap.add_argument("--wszystkie", action="store_true", help="wszystkie sesje dev/test CKE (devset.SESJE_CKE)")
    ap.add_argument("--warianty", default="", help="ocr,opis (puste = tylko tabela przypisania)")
    ap.add_argument("--tabela", action="store_true", help="wypisz tabelę przypisania przed/po")
    ap.add_argument("--url", help="gotowy llama-server z Qwen3.5-0.8B+mmproj (inaczej start na --port)")
    ap.add_argument("--port", type=int, default=PORT_OPIS)
    ap.add_argument("--nadpisz", action="store_true", help="policz od nowa mimo cache")
    a = ap.parse_args(argv)
    sesje = _sesje_cli(a)
    if not sesje:
        ap.error("podaj --sesje albo --wszystkie")
    from .devset import wczytaj
    rels: list[str] = []
    for s in sesje:
        if a.tabela:
            w, p = tabela(s)
            print("\n".join(w))
            print(f"{s}: z ilustracją przed {p['przed_zad']} zad./{p['przed_pkt']} pkt, po {p['po_zad']} zad./{p['po_pkt']} pkt"
                  f" (z {p['zadania']} zad./{p['pkt']} pkt; metoda {p['metoda']})\n", flush=True)
        for z in wczytaj([s]):
            rels += [il["rel"] for il in ilustracje(z) if il["rel"] not in rels]
    warianty = [w for w in a.warianty.split(",") if w]
    print(f"sesje: {sesje}; obrazów zadań (po poprawce): {len(rels)}; warianty: {warianty or '-'}", flush=True)
    czasy: dict[str, list[float]] = {}
    wczytaj_cache(odswiez=True)

    def do_policzenia(w: str) -> list[str]:
        return rels if a.nadpisz else [r for r in rels if klucz(r, w) not in wczytaj_cache()]

    if "ocr" in warianty and (todo := do_policzenia("ocr")):
        t_start = time.perf_counter(); _czytnik(); print(f"[ocr] EasyOCR wczytany w {time.perf_counter() - t_start:.1f} s", flush=True)
        for i, rel in enumerate(todo, 1):
            d = policz_ocr(rel, nadpisz=True)
            czasy.setdefault("ocr", []).append(d["sekundy"])
            print(f"[ocr] {i}/{len(todo)} {rel} {d['sekundy']:.2f} s, linii ≥{PROG_OCR}: {len(napisy(d['wynik']))}", flush=True)
    if "opis" in warianty:
        todo = do_policzenia("opis")
        serwer = None
        if todo:
            url = a.url
            if not url:
                from .noc import Serwer
                t_start = time.perf_counter()
                serwer = Serwer(str(LLAMA_SERVER), GGUF_OPIS, MMPROJ_OPIS, a.port, 1, 8192,
                                CACHE.with_name("obrazy_opisy_serwer.log"), ["--image-min-tokens", "1024"])
                url = serwer.url
                print(f"[opis] llama-server na porcie {a.port} gotowy w {time.perf_counter() - t_start:.1f} s", flush=True)
            try:
                for i, rel in enumerate(todo, 1):
                    d = policz_opis(rel, url, nadpisz=True)
                    czasy.setdefault("opis", []).append(d["sekundy"])
                    print(f"[opis] {i}/{len(todo)} {rel} {d['sekundy']:.2f} s, {len(d['wynik'].split())} słów", flush=True)
            finally:
                if serwer:
                    serwer.stop()
    for w, c in czasy.items():
        print(f"[{w}] policzono {len(c)}: średnio {statistics.mean(c):.2f} s/obraz, mediana {statistics.median(c):.2f} s, "
              f"max {max(c):.2f} s, razem {sum(c):.1f} s", flush=True)
    for w in warianty:
        brak = [r for r in rels if klucz(r, w) not in wczytaj_cache(odswiez=True)]
        print(f"[{w}] w cache {len(rels) - len(brak)}/{len(rels)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
