"""Eksperyment językowy: czy małe modele słabo zdają maturę, bo słabo znają polski, czy dlatego, że nie mają wiedzy?

Co to jest: dwa eksperymenty sparowane (te same zadania, te same modele, ten sam goły prompt, ten sam sędzia;
zmienia się tylko język).
1. Pełny arkusz CKE (2024-maj, 2025-maj, z wypracowaniem) po polsku i po angielsku:
   a) tłumaczenie zadań PL→EN (claude:opus, cache data/jezyk/zadania_en.jsonl),
   b) odpowiedzi gołym promptem w obu warunkach (goly_pl, goly_en), tym samym kodem,
   c) tłumaczenie wsteczne odpowiedzi EN→PL, dosłowne (goly_en_pl),
   d) kontrola tłumaczenia: odpowiedzi PL tłumaczone PL→EN→PL tym samym tłumaczem (goly_pl_rt),
   e) plik zestawów pm_zestawy.txt dla matura/pelna_matura.py (ocenę pełnej matury robi matura/pelna_matura.py).
2. Sonda wiedzy bez formatu matury: 100 krótkich pytań faktograficznych z kart kanonicznych (data/karty/kanon.jsonl,
   nie z arkuszy CKE), wersje PL i EN, odpowiedź do 40 tokenów, ocena claude:opus; kontrola rozumienia: 30 pytań
   z dołączonym jednym zdaniem zawierającym odpowiedź.
Po co: rozstrzygnąć, czy słaby wynik małych modeli to bariera języka (rozumienie polecenia, pisanie po polsku),
czy brak wiedzy.
Co zrobić (kroki osobno, każdy wznawialny; gotowe pozycje są pomijane):
  uv run python -m matura.jezyk --tlumacz     # zadania PL→EN
  uv run python -m matura.jezyk --generuj     # odpowiedzi goly_pl i goly_en (GPU, porty 8097–8099)
  uv run python -m matura.jezyk --wstecz      # EN→PL, PL→EN→PL, pm_zestawy.txt, rozmiary.json
  uv run python -m matura.jezyk --sonda       # pytania sondy, odpowiedzi modeli, oceny (--etap pytania|odpowiedzi|oceny)
  uv run python -m matura.jezyk --raport      # SONDA.md, raport.json; arkusz PL/EN, gdy są oceny sędziego w cache
Drabinka rozmiarów (krok 1a planu docs/plans/2026-09-26-strategia-od-ogolu-do-szczegolu.md): te same kroki z
  --wyniki review/jezyk-drabinka-2026-09-26 --modele qwen3.5-4b-iq4xs,... --opisy --bez-kontroli
  (--opisy: także warunki z blokiem opisów ilustracji vlm_q2b_en, jak harness goly_vlm; --bez-kontroli: bez PL→EN→PL).
Szerokie sito modeli (krok 2 planu): --generuj i --wstecz z --wyniki review/sito-2026-09-26 --sesje 2025-maj
  (--sesje: tylko wskazane arkusze; ocena pelna_matura z tym samym --sesje).
Wyniki: review/jezyk-2026-09-26/ (albo --wyniki). Dane pośrednie (tłumaczenia, pytania, oceny sondy): data/jezyk/ (poza gitem).
Ten moduł NIE ocenia pełnej matury (nie woła sedzia.ocen_partie); w raporcie tylko czyta cache ocen sędziego.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DANE = ROOT / "data" / "jezyk"
WYN = ROOT / "review" / "jezyk-2026-09-26"
ODP = WYN / "odpowiedzi"
SONDA_ODP = WYN / "sonda"
LLAMA = ROOT / "data" / "bin" / "llama-cuda" / "llama-server"   # domyślna; _SerwerModelu bierze noc.llama_server()
PORTY = (8097, 8098, 8099)
TLUMACZ = "claude:opus"
TLUMACZE = ("claude", "baza", "marian")   # --tl-zadan / --tl-odp: Claude (sufit) albo lokalnie (matura/tlumacz.py)
TL_ZADAN = "claude"
TL_ODP = "claude"
SESJE = ("2024-maj", "2025-maj")
ZIARNO = 0
_LOG = threading.Lock()

# nazwa → (plik GGUF względem data/modele, bez_myslenia jak w noc/f2_drabinka.toml, równoległe sloty serwera)
MODELE: dict[str, tuple[str, bool, int]] = {
    "qwen3.5-9b-q5": ("qwen3.5-9b-q5/Qwen3.5-9B-Q5_K_M.gguf", True, 4),
    "qwen3.5-2b-q4-bezwizji": ("qwen3.5-2b-q4-bezwizji/Qwen3.5-2B-Q4_K_M.gguf", True, 8),
    "bielik-1.5b-q8": ("bielik-1.5b-q8/Bielik-1.5B-v3.0-Instruct.Q8_0.gguf", False, 8),
    "gemma3-1b-q4": ("gemma3-1b-q4/gemma-3-1b-it-Q4_K_M.gguf", False, 8),
    "qwen3.5-0.8b-q4-bezwizji": ("qwen3.5-0.8b-q4-bezwizji/Qwen3.5-0.8B-Q4_K_M.gguf", True, 8),
    "qwen3-0.6b-q4": ("qwen3-0.6b-q4/Qwen3-0.6B-Q4_K_M.gguf", True, 8),
    "smollm2-360m-q4": ("smollm2-360m-q4/SmolLM2-360M-Instruct-Q4_K_M.gguf", False, 8),  # bez trybu myślenia
    # drabinka rozmiarów kroku 1a (pliki unsloth, tylko tekst)
    "qwen3.5-4b-iq4xs": ("qwen3.5-4b-iq4xs/Qwen3.5-4B-IQ4_XS.gguf", True, 8),
    "qwen3.5-4b-iq3xxs": ("qwen3.5-4b-iq3xxs/Qwen3.5-4B-UD-IQ3_XXS.gguf", True, 8),
    "qwen3.5-4b-iq2m": ("qwen3.5-4b-iq2m/Qwen3.5-4B-UD-IQ2_M.gguf", True, 8),
    "qwen3-4b-2507-iq3xxs": ("qwen3-4b-2507-iq3xxs/Qwen3-4B-Instruct-2507-UD-IQ3_XXS.gguf", False, 8),  # bez myślenia
    # szerokie sito kroku 2 (pliki < 1,9 GB, review/sito-2026-09-26/pobierz.py); bez przełącznika myślenia w szablonie,
    # poza SmolLM3: szablon unsloth ma enable_thinking (domyślnie true; false → pusty blok <think></think>);
    # 4 sloty przy pełnej uwadze z 8 głowami KV we wszystkich warstwach (bufor KV 8 slotów × 8192 to 7-10 GB),
    # żeby 3 serwery naraz zmieściły się w 32 GB obok innych przebiegów
    "qwen3-4b-2507-iq2m": ("qwen3-4b-2507-iq2m/Qwen3-4B-Instruct-2507-UD-IQ2_M.gguf", False, 4),
    "gemma3-4b-iq3xxs": ("gemma3-4b-iq3xxs/gemma-3-4b-it-UD-IQ3_XXS.gguf", False, 8),
    "phi4-mini-iq3xxs": ("phi4-mini-iq3xxs/microsoft_Phi-4-mini-instruct-IQ3_XXS.gguf", False, 4),
    "llama3.2-3b-iq3xxs": ("llama3.2-3b-iq3xxs/Llama-3.2-3B-Instruct-UD-IQ3_XXS.gguf", False, 4),
    "smollm3-3b-iq3xxs": ("smollm3-3b-iq3xxs/SmolLM3-3B-UD-IQ3_XXS.gguf", True, 8),
    "granite4-h-micro-iq3xxs": ("granite4-h-micro-iq3xxs/granite-4.0-h-micro-UD-IQ3_XXS.gguf", False, 8),
    "lfm2-2.6b-q4km": ("lfm2-2.6b-q4km/LFM2-2.6B-Q4_K_M.gguf", False, 8),
    "falcon-h1-3b-iq3xxs": ("falcon-h1-3b-iq3xxs/Falcon-H1-3B-Instruct-UD-IQ3_XXS.gguf", False, 8),
    "eurollm-1.7b-q4km": ("eurollm-1.7b-q4km/EuroLLM-1.7B-Instruct.Q4_K_M.gguf", False, 4),
    "ministral3-3b-iq3xxs": ("ministral3-3b-iq3xxs/Ministral-3-3B-Instruct-2512-UD-IQ3_XXS.gguf", False, 4),
    # esej po polsku bez tłumacza (eksperyment 27.09): model innego zespołu (SlayerLab, apache-2.0,
    # tylko pomiar) i nasz Bielik-1.5B-v3 po SFT na esejach nauczyciela (matura/esej_sft.py, eksport matura/trening.py)
    "slayer-bielik-1.5b-sft2": ("slayer-bielik-1.5b-sft2/sft2-IQ4_XS.gguf", False, 8),
    "bielik-1.5b-esej-v1": ("bielik-1.5b-esej-v1/bielik-1.5b-esej-v1-Q8_0.gguf", False, 8),   # scripts/trenuj_uczniow_eseju.sh v1
}

SYS_GOLY_EN = "You are a student taking the Polish history matura exam (extended level). Answer the task in English."
SYS_SONDA = {"pl": "Odpowiedz krótko.", "en": "Answer briefly."}
WARUNKI = {"goly_pl": "pl", "goly_en": "en"}          # konfiguracja generacji → język zadania
ZESTAWY = (("pl", "goly_pl"), ("en→pl", "goly_en_pl"), ("pl→en→pl", "goly_pl_rt"))
OPIS = "vlm_q2b_en"   # opisy ilustracji lokalnego opisywacza (Qwen3.5-2B z wizją, prompt EN), jak harness goly_vlm
WARUNKI_OPIS = {"goly_vlm_pl": "pl", "goly_vlm_en": "en"}
WSTECZ = {"goly_en": "goly_en_pl", "goly_vlm_en": "goly_vlm_en_pl", "goly_en_kb": "goly_en_kb_pl"}   # odpowiedzi EN → zestaw po tłumaczeniu na PL
ZESTAWY_OPIS = (("pl+opis", "goly_vlm_pl"), ("en→pl+opis", "goly_vlm_en_pl"))
# --kb (krok C planu po 1a/1b): goły prompt EN + 2 hasła bazy wyszukane po polsku (jak goly_kb), tekst hasła po angielsku
# z data/wiedza/tlumaczenia/hasla_en.jsonl (scripts/hasla_en.py); brak tłumaczenia → tekst polski
KB = False
TYLKO_ESEJE = False   # --tylko-eseje: tylko tematy wypracowań (walidacja reguły tematu bez części krótkiej)
ZESTAWY_KB = (("en→pl+kb", "goly_en_kb_pl"),)
NAGLOWEK_KB_EN = "Background knowledge (knowledge base, may be unrelated to the task):"
PLIK_HASLA_EN = ROOT / "data" / "wiedza" / "tlumaczenia" / "hasla_en.jsonl"
TEMPERATURA = {"krotkie": 0.0, "esej": 0.2}   # krótkie jak reguła 6 planu; esej jak w pierwszym przebiegu (jedno wywołanie)
NAGLOWEK_BLOKU_EN = "Illustrations of the task as text (automatic image description, may contain errors):"


def log(*a) -> None:
    with _LOG:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(*czesci: str) -> str:
    return hashlib.sha1("\x1f".join(czesci).encode()).hexdigest()


# ---------------------------------------------------------------- cache JSONL (wznawialny)

class Cache:
    """Plik JSONL: jeden wpis = jeden wynik z polem „klucz”. Dopisywanie przyrostowe (bezpieczne wątkowo);
    przy powtórzonym kluczu wygrywa ostatni wpis. Przerwany przebieg dokańcza się od miejsca przerwania."""

    def __init__(self, p: Path):
        self.p, self.lock, self.d = p, threading.Lock(), {}
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    try:
                        w = json.loads(l)
                    except json.JSONDecodeError:
                        continue  # urwana ostatnia linia po przerwaniu
                    self.d[w["klucz"]] = w

    def __contains__(self, k: str) -> bool:
        return k in self.d

    def get(self, k: str) -> dict | None:
        return self.d.get(k)

    def dodaj(self, wpisy: list[dict]) -> None:
        with self.lock:
            self.p.parent.mkdir(parents=True, exist_ok=True)
            with open(self.p, "a", encoding="utf-8") as f:
                for w in wpisy:
                    f.write(json.dumps(w, ensure_ascii=False) + "\n")
                    self.d[w["klucz"]] = w


def wczytaj_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    return out


def wczytaj_odp(p: Path) -> dict[str, dict]:
    """id → rekord (ostatni wygrywa: ponowiona po błędzie odpowiedź nadpisuje „(BŁĄD …)”)."""
    return {d["id"]: d for d in wczytaj_jsonl(p)}


def blad(odp: str | None) -> bool:
    return (odp or "").lstrip().startswith("(BŁĄD")


# ---------------------------------------------------------------- zadania eksperymentu 1

def zadania(sesje=None) -> list[dict]:
    """Zadania krótkie arkuszy (devset._cke z esejem, bez samego zadania esejowego) + tematy wypracowań
    (<id zadania esejowego>-t1..t3 z eseje-cke). Ten sam zbiór co matura/pelna_matura.zadania_sesji.
    Bez argumentu: arkusze z globalnego SESJE (opcja --sesje; domyślnie 2024-maj i 2025-maj).
    TYLKO_ESEJE (opcja --tylko-eseje): same tematy wypracowań."""
    from . import devset
    sesje = SESJE if sesje is None else sesje
    out, tematy = [], devset.wczytaj(["eseje-cke"])
    for s in sesje:
        wszystkie = devset._cke(s, z_esejem=True)
        out += [] if TYLKO_ESEJE else [z for z in wszystkie if not z["esej"]]
        for e in (z for z in wszystkie if z["esej"]):
            out += sorted((t for t in tematy if t["id"].startswith(e["id"] + "-t")), key=lambda t: t["id"])
    return out


def tekst_pl(z: dict) -> str:
    """Treść dla modelu, identyczna jak w harnessie `goly` (dla tematu eseju: oczyszczone pole polecenie)."""
    from .devset import tresc_dla_modelu
    return tresc_dla_modelu(z)


def blok_opisu_en(z: dict, wariant: str = OPIS) -> str:
    """Blok opisów ilustracji dla warunku EN: te same opisy co obrazy.blok_opisu (vlm_* są już po angielsku),
    nagłówek i etykiety po angielsku, bez polskiego podpisu (podpis jest w przetłumaczonej treści zadania).
    Pusty, gdy zadanie nie ma ilustracji; brak opisu w cache → obrazy.BrakOpisu."""
    from .obrazy import ilustracje, z_cache
    linie = []
    for il in ilustracje(z):
        e = f"source {il['zrodlo']}" if il["zrodlo"] is not None else "source"
        e += f", illustration {il['i']} of {il['n']}" if il["n"] > 1 else ""
        linie.append(f"IMAGE DESCRIPTION ({e}): {z_cache(il['rel'], wariant)['wynik']}")
    return "\n".join([NAGLOWEK_BLOKU_EN] + linie) if linie else ""


_HASLA_EN: dict[str, str] | None = None


def hasla_en() -> dict[str, str]:
    """id hasła → tekst angielski; wpis ważny, gdy jego pole „pl” równa się bieżącemu tekstowi hasła."""
    global _HASLA_EN
    if _HASLA_EN is None:
        from .harness import _baza_hasel
        from .wiedza import _tekst_hasla
        pl = {h["id"]: _tekst_hasla(h) for h in _baza_hasel().h}
        _HASLA_EN = {w["id"]: w["en"] for w in wczytaj_jsonl(PLIK_HASLA_EN) if pl.get(w["id"]) == w.get("pl")}
    return _HASLA_EN


def blok_kb_en(z: dict, n: int = 2) -> str:
    """n haseł bazy dla zadania (zapytanie po polsku: wiedza.zapytanie bez opisu ilustracji) jako blok EN przed
    treścią; pusty, gdy baza nic nie zwraca."""
    from .harness import _baza_hasel
    from .wiedza import _tekst_hasla, zapytanie
    hasla = _baza_hasel().szukaj(zapytanie(z, ""), n)
    en = hasla_en()
    linie = [f"- {en.get(h['id']) or _tekst_hasla(h)}" for h in hasla]
    return (NAGLOWEK_KB_EN + "\n" + "\n".join(linie) + "\n\n") if linie else ""


def _zrodla(opisy: bool) -> tuple[str, ...]:
    """Pliki odpowiedzi EN tłumaczone w --wstecz."""
    return ("goly_en",) + (("goly_vlm_en",) if opisy else ()) + (("goly_en_kb",) if KB else ())


def z_blokiem(blok: str, tresc: str) -> str:
    """Blok opisów przed treścią zadania, jak w harnessie goly_vlm."""
    return (blok + "\n\n" if blok else "") + tresc


def czy_pf(z: dict) -> bool:
    """Zadanie prawda/fałsz (klucz P/F albo „Oceń prawdziwość”): tu po tłumaczeniu wstecznym T → P."""
    if z.get("esej"):
        return False
    if re.search(r"Oceń prawdziwość", z.get("polecenie", ""), re.I):
        return True
    if z.get("typ") == "zamkniete":
        from .klucz import parsuj_klucz
        k = parsuj_klucz(z)
        return bool(k and k.get("typ") == "pf")
    return False


_LIT = "A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż"
_RE_CIAG_TF = re.compile(rf"(?<![{_LIT}])[TPF]{{2,}}(?![{_LIT}])")
_RE_T = re.compile(rf"(?<![{_LIT}0-9])T(?![{_LIT}])(?!\.\s*[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż])")  # nie inicjał „T. Nowak”


def tf_na_pf(tekst: str) -> str:
    """Oznaczenia prawda/fałsz z angielskiego na polskie: samodzielne „T” → „P”, ciągi „TFT” → „PFP”,
    „True”/„False” → „Prawda”/„Fałsz”. Zabezpieczenie po tłumaczeniu wstecznym (tylko zadania P/F, czy_pf);
    litery w wyrazach i inne oznaczenia (A–D, cyfry) bez zmian."""
    t = re.sub(r"\bTRUE\b|\bTrue\b", "Prawda", tekst)
    t = re.sub(r"\bFALSE\b|\bFalse\b", "Fałsz", t)
    t = _RE_CIAG_TF.sub(lambda m: m.group(0).replace("T", "P"), t)
    return _RE_T.sub("P", t)


# ---------------------------------------------------------------- tłumacz (claude:opus)

SCHEMA_TL = {"type": "object", "additionalProperties": False, "required": ["tlumaczenia"],
             "properties": {"tlumaczenia": {"type": "array", "items": {
                 "type": "object", "additionalProperties": False, "required": ["id", "tekst"],
                 "properties": {"id": {"type": "string"}, "tekst": {"type": "string"}}}}}}

SYS_ZADANIA_EN = """Jesteś zawodowym tłumaczem tekstów historycznych z polskiego na angielski. Tłumaczysz zadania
z polskiej matury z historii (arkusz CKE, poziom rozszerzony) dla zdającego, który czyta tylko po angielsku.
Tłumaczenie ma być wierne i kompletne:
- Przetłumacz CAŁY tekst: teksty źródłowe, opisy ilustracji i map, tabele, polecenia, podpisy i przypisy.
  Niczego nie pomijaj, nie streszczaj i nie dodawaj: żadnych wyjaśnień, podpowiedzi ani odpowiedzi.
- Cytaty ze źródeł przetłumacz na angielski (także staropolszczyznę, zachowując charakter tekstu). Opuszczenia […]
  i dopiski redakcyjne w nawiasach kwadratowych zachowaj. Fragmenty łacińskie zostaw po łacinie.
- Nazwy własne (osoby, miejsca, państwa, wydarzenia, dokumenty, instytucje) podaj w formie przyjętej
  w anglojęzycznej historiografii, a przy pierwszym użyciu dodaj w nawiasie polski oryginał, jeśli brzmi inaczej,
  np. „Sigismund III Vasa (Zygmunt III Waza)”, „Jogaila (Władysław II Jagiełło)”, „the Four-Year Sejm (Sejm
  Czteroletni)”, „Cracow (Kraków)”. Terminy utrwalone w angielskiej historiografii (szlachta, liberum veto,
  pacta conventa) zostaw; inne polskie terminy przetłumacz z polskim terminem w nawiasie przy pierwszym użyciu.
- Zachowaj układ i oznaczenia: numery zadań i podpunktów („Zadanie 3.2. (0–1)” → „Task 3.2. (0–1)”), litery
  odpowiedzi A, B, C, D, numerację 1., 2., punktory, tabele, puste miejsca do uzupełnienia, podział na akapity.
- Zadania prawda/fałsz: oznaczenia P/F zamień na T/F, np. „Zaznacz P, jeśli zdanie jest prawdziwe, albo F – jeśli
  jest fałszywe.” → „Mark T if the statement is true, or F if it is false.”
- Opisy bibliograficzne (autor, tytuł, miejsce, rok): autora i rok zostaw, tytuł przetłumacz, miejsce po angielsku.
- „Twoja wypowiedź powinna liczyć minimum 300 wyrazów.” → „Your answer should be at least 300 words long.”
Zwróć JSON zgodny ze schematem: dokładnie jedno tłumaczenie na każde id, pole tekst = pełne tłumaczenie."""

_DOSLOWNIE = """Tłumaczenie posłuży egzaminatorowi do oceny odpowiedzi, więc musi być DOSŁOWNE i WIERNE:
- Tłumacz całość, zdanie po zdaniu. NIE poprawiaj błędów merytorycznych, językowych ani logicznych; NIE dodawaj
  treści, wyjaśnień ani brakujących elementów; NIE skracaj, NIE streszczaj, NIE pomijaj powtórzeń. Odpowiedź
  urwana w połowie zostaje urwana. Bełkot tłumacz możliwie dosłownie, bez nadawania mu sensu.
- Zachowaj układ: akapity, numerację, wypunktowania, formatowanie Markdown (np. **pogrubienia**).
- Litery odpowiedzi (A, B, C, D), cyfry, numery i daty przepisz bez zmian.
- Treść zadania dostajesz TYLKO po to, żeby dobrać nazwy i terminy w formie użytej w zadaniu; nigdy nie
  uzupełniaj odpowiedzi treścią zadania ani własną wiedzą.
- Każdą odpowiedź tłumacz osobno: nie przenoś treści między odpowiedziami.
Zwróć JSON zgodny ze schematem: dokładnie jedno tłumaczenie na każde id odpowiedzi."""

SYS_ODP = {
    "en-pl": "Jesteś tłumaczem. Tłumaczysz odpowiedzi uczniów na zadania z historii z angielskiego na polski.\n"
             + _DOSLOWNIE + """
- Oznaczenia T/F (True/False) w zadaniach prawda/fałsz zamień na P/F (Prawda/Fałsz).
- Nazwy własne i terminy podaj w formie przyjętej w polskiej historiografii (np. Jogaila → Jagiełło,
  Sigismund III Vasa → Zygmunt III Waza), tak jak w polskiej treści zadania.
- Fragmenty napisane już po polsku przepisz bez zmian; fragmenty w innym języku przetłumacz na polski.""",
    "pl-en": "Jesteś tłumaczem. Tłumaczysz odpowiedzi uczniów na zadania z historii z polskiego na angielski.\n"
             + _DOSLOWNIE + """
- Oznaczenia P/F (Prawda/Fałsz) w zadaniach prawda/fałsz zamień na T/F (True/False).
- Nazwy własne i terminy podaj w formie przyjętej w anglojęzycznej historiografii, tak jak w angielskiej
  treści zadania (bez dopisywania polskich oryginałów w nawiasach, jeśli nie ma ich w odpowiedzi).
- Fragmenty napisane już po angielsku przepisz bez zmian; fragmenty w innym języku przetłumacz na angielski.""",
}


def partie_wg_znakow(grupy: list[tuple[str, int]], budzet: int) -> list[list[str]]:
    """Grupy (klucz, rozmiar w znakach) → partie kluczy, zachłannie w podanej kolejności; suma rozmiarów w partii
    ≤ budzet (grupa większa niż budżet idzie sama). Deterministyczne, bez dzielenia grup."""
    out, biez, suma = [], [], 0
    for k, n in grupy:
        if biez and suma + n > budzet:
            out.append(biez); biez, suma = [], 0
        biez.append(k); suma += n
    if biez:
        out.append(biez)
    return out


def _kontekst_zn(p: dict) -> int:
    return len(p["tresc_pl"][:3000]) + len(p["tresc_en"][:3000])


def partie_odpowiedzi(pozycje: list[dict], budzet: int) -> list[list[dict]]:
    """Pozycje (posortowane wg zadania) → partie: koszt pozycji = długość odpowiedzi + treść zadania (PL i EN),
    liczona raz na zadanie w partii. Nowa partia, gdy koszt przekroczyłby budżet (pozycja większa idzie sama)."""
    out, biez, suma, zad = [], [], 0, set()
    for p in pozycje:
        n = len(p["tekst"]) + (0 if p["id"] in zad else _kontekst_zn(p))
        if biez and suma + n > budzet:
            out.append(biez)
            biez, suma, zad = [], 0, set()
            n = len(p["tekst"]) + _kontekst_zn(p)
        biez.append(p); suma += n; zad.add(p["id"])
    if biez:
        out.append(biez)
    return out


def rozpakuj(wynik: dict, oczekiwane: list[str]) -> dict[str, str]:
    """Odpowiedź tłumacza → {id: tekst} tylko dla oczekiwanych id (nadmiarowe odrzucone, brakujące pominięte:
    wrócą w następnej próbie). Tekst z obciętymi białymi znakami na brzegach."""
    ocz = set(oczekiwane)
    out = {}
    for t in (wynik or {}).get("tlumaczenia", []) or []:
        i = str(t.get("id", "")).strip()
        if i in ocz and isinstance(t.get("tekst"), str) and i not in out:
            out[i] = t["tekst"].strip()
    return out


def _llm_json(prompt: str, schema: dict, system: str, timeout: int = 900) -> dict:
    from .sedzia import llm_json
    return llm_json(prompt, schema, TLUMACZ, system, timeout)


def wykonaj_partie(partie: list, fn, rownolegle: int, opis: str, proby: int = 3) -> int:
    """fn(partia) → liczba zrobionych pozycji (zapis do cache w fn). Partia, która padła, jest ponawiana
    (do `proby` przebiegów). Zwraca liczbę partii nieudanych po wszystkich próbach."""
    zostaly = list(partie)
    for proba in range(1, proby + 1):
        if not zostaly:
            return 0
        log(f"[{opis}] próba {proba}: {len(zostaly)} partii, równolegle {rownolegle}")
        nieudane = []
        with ThreadPoolExecutor(max(1, rownolegle)) as ex:
            futs = {ex.submit(fn, p): p for p in zostaly}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    ok = f.result()
                except Exception as e:  # noqa: BLE001 (limit, timeout, zły JSON: następna próba)
                    ok = -1
                    log(f"[{opis}] partia NIEUDANA: {str(e)[:300]}")
                if ok is None or ok < len(futs[f]):
                    nieudane.append(futs[f])
                if i % 10 == 0 or i == len(futs):
                    log(f"[{opis}] {i}/{len(futs)} partii")
        zostaly = nieudane
    return len(zostaly)


# ---------------------------------------------------------------- krok 1: tłumaczenie zadań

def plik_zadan_en(model: str | None = None) -> Path:
    """Cache tłumaczeń zadań wg TL_ZADAN: claude → zadania_en.jsonl, marian → zadania_en__marian.jsonl,
    baza → zadania_en__baza__<model>.jsonl (każdy model tłumaczy sam dla siebie)."""
    if TL_ZADAN == "claude":
        return DANE / "zadania_en.jsonl"
    if TL_ZADAN in MODELE:  # osobny model tłumacza (mniejszy od bazy), wspólny dla wszystkich baz
        return DANE / f"zadania_en__model__{TL_ZADAN}.jsonl"
    return DANE / (f"zadania_en__baza__{model}.jsonl" if TL_ZADAN == "baza" else f"zadania_en__{TL_ZADAN}.jsonl")


def tlumaczenia_zadan(model: str | None = None) -> dict[str, str]:
    """id zadania → tekst EN (z cache; wpis ważny tylko dla tej samej treści PL)."""
    c = Cache(plik_zadan_en(model))
    out = {}
    for z in zadania():
        w = c.get(sha("zadanie", z["id"], tekst_pl(z)))
        if w:
            out[z["id"]] = w["en"]
    return out


def tlumacz_zadania_lokalnie(zz: list[dict], tl, plik: Path, nazwa: str, porcja: int = 8) -> dict[str, str]:
    """Tłumaczenie zadań PL→EN lokalnym tłumaczem (matura/tlumacz.py) do cache `plik`, zapis co `porcja` zadań
    (przerwany przebieg dokańcza się); zwraca id → EN dla zadań z tłumaczeniem."""
    c = Cache(plik)
    todo = [z for z in zz if sha("zadanie", z["id"], tekst_pl(z)) not in c]
    log(f"[tłumacz zadań {nazwa}] do zrobienia {len(todo)} z {len(zz)}")
    t0 = time.time()
    for i in range(0, len(todo), porcja):
        cz = todo[i:i + porcja]
        wyn = tl.tlumacz([tekst_pl(z) for z in cz], "pl-en")
        c.dodaj([{"klucz": sha("zadanie", z["id"], tekst_pl(z)), "id": z["id"], "pl": tekst_pl(z), "en": e,
                  "tlumacz": nazwa} for z, e in zip(cz, wyn) if e.strip()])
    if todo:
        log(f"[tłumacz zadań {nazwa}] {len(todo)} zadań w {time.time() - t0:.0f} s")
    out = {}
    for z in zz:
        w = c.get(sha("zadanie", z["id"], tekst_pl(z)))
        if w:
            out[z["id"]] = w["en"]
    return out


def krok_tlumacz(rownolegle: int = 6, limit: int | None = None, budzet: int = 9000) -> dict[str, str]:
    zz = zadania()[:limit] if limit else zadania()
    if TL_ZADAN == "marian":
        from .tlumacz import Marian
        return tlumacz_zadania_lokalnie(zz, Marian(), plik_zadan_en(), "marian")
    if TL_ZADAN == "baza":
        log("[tłumacz zadań] --tl-zadan baza: tłumaczy każdy model na własnym serwerze w kroku --generuj")
        return {}
    if TL_ZADAN in MODELE:
        from .tlumacz import Baza

        def praca(m, s):
            tlumacz_zadania_lokalnie(zz, Baza(s.url, s.bez_myslenia, rownolegle=s.rown), plik_zadan_en(), f"model:{m}")

        na_modelach([TL_ZADAN], praca)
        return tlumaczenia_zadan()
    c = Cache(plik_zadan_en())
    todo = [z for z in zz if sha("zadanie", z["id"], tekst_pl(z)) not in c]
    log(f"[tłumacz zadań] do zrobienia {len(todo)} z {len(zz)}")
    po_id = {z["id"]: z for z in todo}
    partie = partie_wg_znakow([(z["id"], len(tekst_pl(z))) for z in todo], budzet)

    def jedna(wszystkie: list[str]) -> int:
        ids = [i for i in wszystkie if sha("zadanie", i, tekst_pl(po_id[i])) not in c]
        if not ids:
            return len(wszystkie)
        lad = [{"id": i, "tekst": tekst_pl(po_id[i])} for i in ids]
        w = _llm_json("ZADANIA DO PRZETŁUMACZENIA (JSON):\n" + json.dumps(lad, ensure_ascii=False, indent=1),
                      SCHEMA_TL, SYS_ZADANIA_EN)
        got = {i: t for i, t in rozpakuj(w, ids).items() if t}
        c.dodaj([{"klucz": sha("zadanie", i, tekst_pl(po_id[i])), "id": i, "pl": tekst_pl(po_id[i]), "en": t,
                  "tlumacz": TLUMACZ} for i, t in got.items()])
        return len(wszystkie) - len(ids) + len(got)

    nieudane = wykonaj_partie(partie, jedna, rownolegle, "tłumacz zadań")
    tl = tlumaczenia_zadan()
    for z in zz:
        if z["id"] in tl:
            r = len(tl[z["id"]]) / max(1, len(tekst_pl(z)))
            if not 0.6 <= r <= 1.6:
                log(f"[tłumacz zadań] UWAGA {z['id']}: stosunek długości EN/PL {r:.2f}")
    log(f"[tłumacz zadań] gotowe {sum(z['id'] in tl for z in zz)}/{len(zz)}, nieudanych partii {nieudane}")
    return tl


# ---------------------------------------------------------------- krok 2: generacja (GPU)

def plik_odp(model: str, konfig: str, wyn: Path | None = None) -> Path:
    return (wyn or WYN) / "odpowiedzi" / f"{model}__{konfig}.jsonl"


def generuj_warunek(model: str, konfig: str, pozycje: list[tuple[dict, str]], system: str, url: str,
                    bez_myslenia: bool, rownolegle: int, wyn: Path | None = None, czat_fn=None) -> Path:
    """pozycje: (zadanie, treść dla modelu). Rekordy {"id","model","konfig","odpowiedz","sekundy"} dopisywane
    przyrostowo; wznowienie pomija id z gotową odpowiedzią (odpowiedź „(BŁĄD …)” jest ponawiana)."""
    if czat_fn is None:
        from .llm import czat as czat_fn
    p = plik_odp(model, konfig, wyn)
    p.parent.mkdir(parents=True, exist_ok=True)
    gotowe = {i for i, d in wczytaj_odp(p).items() if not blad(d.get("odpowiedz"))}
    todo = [(z, t) for z, t in pozycje if z["id"] not in gotowe]
    log(f"[{model}/{konfig}] do zrobienia {len(todo)} z {len(pozycje)}")
    lock = threading.Lock()

    def jedno(zt):
        z, tresc = zt
        try:
            t, s = czat_fn(url, system, tresc, None, max_tokens=1400 if z["esej"] else 400, bez_myslenia=bez_myslenia,
                           temperature=TEMPERATURA["esej" if z["esej"] else "krotkie"])
        except Exception as e:  # noqa: BLE001 (zapis błędu; wznowienie ponowi)
            t, s = f"(BŁĄD: {e})", 0.0
        with lock, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": z["id"], "model": model, "konfig": konfig, "odpowiedz": t,
                                "sekundy": round(s, 2)}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max(1, rownolegle)) as ex:
        list(ex.map(jedno, todo))
    return p


def _gguf(model: str) -> Path:
    return ROOT / "data" / "modele" / MODELE[model][0]


class _SerwerModelu:
    """llama-server dla modelu tekstowego (bez mmproj) na porcie z puli 8097–8099 (wzorzec: matura/noc.Serwer)."""

    def __init__(self, model: str, port: int):
        from .noc import Serwer
        (WYN / "logi").mkdir(parents=True, exist_ok=True)
        _, self.bez_myslenia, self.rown = MODELE[model]
        from .noc import llama_server
        self.s = Serwer(llama_server(), _gguf(model), None, port, self.rown, 8192, WYN / "logi" / f"serwer_{model}.log")
        self.url = self.s.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.s.stop()


def na_modelach(modele: list[str], praca, porty=PORTY) -> None:
    """praca(model, serwer) dla każdego modelu; do len(porty) modeli naraz, każdy na własnym porcie."""
    wolne = list(porty)
    lock = threading.Lock()

    def jeden(m):
        with lock:
            port = wolne.pop()
        try:
            t0 = time.time()
            with _SerwerModelu(m, port) as s:
                log(f"[{m}] serwer gotowy na porcie {port}")
                praca(m, s)
            log(f"[{m}] gotowe w {time.time() - t0:.0f} s")
        except Exception as e:  # noqa: BLE001 (model pominięty, reszta idzie dalej)
            log(f"[{m}] POMINIĘTY: {e}")
        finally:
            with lock:
                wolne.append(port)

    with ThreadPoolExecutor(max(1, len(porty))) as ex:
        list(ex.map(jeden, modele))


def zapisz_rozmiary(modele: list[str]) -> None:
    """rozmiary.json w katalogu wyników (kolumna MB w matura/pelna_matura.py)."""
    p = WYN / "rozmiary.json"
    r = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    for m in modele:
        g = _gguf(m)
        if g.exists():
            r[m] = {"gguf": g.stat().st_size, "mmproj": 0, "razem": g.stat().st_size}
    WYN.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=1), encoding="utf-8")


def krok_generuj(modele: list[str], limit: int | None = None, opisy: bool = False) -> None:
    from .harness import SYS_GOLY
    from .obrazy import blok_opisu
    zz = zadania()[:limit] if limit else zadania()
    systemy = {"goly_pl": SYS_GOLY, "goly_en": SYS_GOLY_EN}
    if opisy:
        systemy |= {"goly_vlm_pl": SYS_GOLY, "goly_vlm_en": SYS_GOLY_EN}
        bloki = {z["id"]: (blok_opisu(z, OPIS), blok_opisu_en(z)) for z in zz}  # przed serwerami: BrakOpisu od razu
    if KB:
        systemy["goly_en_kb"] = SYS_GOLY_EN
        bloki_kb = {z["id"]: blok_kb_en(z) for z in zz}
        log(f"[generuj] --kb: {sum(1 for b in bloki_kb.values() if b)}/{len(zz)} zadań z hasłami, "
            f"tłumaczeń haseł EN: {len(hasla_en())}")

    def pozycje(tl: dict[str, str]) -> dict[str, list]:
        brak_en = [z["id"] for z in zz if z["id"] not in tl]
        if brak_en:
            log(f"[generuj] UWAGA: {len(brak_en)} zadań bez tłumaczenia (warunek EN je pominie): {brak_en[:5]}…")
        poz = {"goly_pl": [(z, tekst_pl(z)) for z in zz], "goly_en": [(z, tl[z["id"]]) for z in zz if z["id"] in tl]}
        if opisy:
            poz["goly_vlm_pl"] = [(z, z_blokiem(bloki[z["id"]][0], tekst_pl(z))) for z in zz]
            poz["goly_vlm_en"] = [(z, z_blokiem(bloki[z["id"]][1], tl[z["id"]])) for z in zz if z["id"] in tl]
        if KB:
            poz["goly_en_kb"] = [(z, bloki_kb[z["id"]] + tl[z["id"]]) for z in zz if z["id"] in tl]
        return poz

    tl = tlumaczenia_zadan() if TL_ZADAN != "baza" else None
    zapisz_rozmiary(modele)

    def praca(m, s):
        tl_m = tl
        if TL_ZADAN == "baza":  # tłumacz = ten sam model na tym samym serwerze
            from .tlumacz import Baza
            tl_m = tlumacz_zadania_lokalnie(zz, Baza(s.url, s.bez_myslenia, rownolegle=s.rown), plik_zadan_en(m),
                                            f"baza:{m}")
        poz = pozycje(tl_m)
        for k in poz:
            generuj_warunek(m, k, poz[k], systemy[k], s.url, s.bez_myslenia, s.rown)

    na_modelach(modele, praca)


# ---------------------------------------------------------------- kroki 3–4: tłumaczenie odpowiedzi

def plik_tl_odp(model: str | None = None) -> Path:
    """Cache tłumaczeń odpowiedzi wg TL_ODP: claude → odpowiedzi_tl.jsonl, marian → odpowiedzi_tl__marian.jsonl,
    baza → odpowiedzi_tl__baza__<model>.jsonl."""
    if TL_ODP == "claude":
        return DANE / "odpowiedzi_tl.jsonl"
    return DANE / (f"odpowiedzi_tl__baza__{model}.jsonl" if TL_ODP == "baza" else f"odpowiedzi_tl__{TL_ODP}.jsonl")


def tlumacz_odpowiedzi_lokalnie(pozycje: list[dict], tl, c: "Cache", nazwa: str, zamkniete: set[str],
                                porcja: int = 16) -> None:
    """pozycje: {id, tekst, tresc_pl}. EN→PL lokalnym tłumaczem do cache `c` (klucze jak w tlumacz_odpowiedzi).
    Zadania zamknięte (id w `zamkniete`): odpowiedź bez tłumaczenia (oznaczenia; T/F → P/F robi zbuduj_wstecz)."""
    uniq: dict[str, dict] = {}
    for p in pozycje:
        if not p["tekst"].strip() or blad(p["tekst"]):
            continue
        k = klucz_tl("en-pl", p["id"], p["tekst"])
        if k not in c:
            uniq.setdefault(k, {**p, "klucz": k})
    todo = list(uniq.values())
    bez = [p for p in todo if p["id"] in zamkniete]
    reszta = [p for p in todo if p["id"] not in zamkniete]
    log(f"[tłumacz en-pl {nazwa}] do zrobienia {len(reszta)} odpowiedzi (+{len(bez)} zamkniętych bez tłumaczenia)")
    c.dodaj([{"klucz": p["klucz"], "kierunek": "en-pl", "id": p["id"], "zrodlo": p["tekst"], "wynik": p["tekst"],
              "tlumacz": "bez tłumaczenia (zamknięte)"} for p in bez])
    t0 = time.time()
    for i in range(0, len(reszta), porcja):
        cz = reszta[i:i + porcja]
        wyn = tl.tlumacz([p["tekst"] for p in cz], "en-pl", konteksty=[p["tresc_pl"] for p in cz])
        c.dodaj([{"klucz": p["klucz"], "kierunek": "en-pl", "id": p["id"], "zrodlo": p["tekst"], "wynik": w,
                  "tlumacz": nazwa} for p, w in zip(cz, wyn) if w.strip()])
    if reszta:
        log(f"[tłumacz en-pl {nazwa}] {len(reszta)} odpowiedzi w {time.time() - t0:.0f} s")


def klucz_tl(kierunek: str, zid: str, tekst: str) -> str:
    return sha("odp", kierunek, zid, tekst)


def tlumacz_odpowiedzi(pozycje: list[dict], kierunek: str, rownolegle: int = 6, budzet: int = 18000,
                       llm_fn=None, cache: Cache | None = None) -> Cache:
    """pozycje: {id, tekst, tresc_pl, tresc_en}. Tłumaczenie odpowiedzi (kierunek "en-pl" albo "pl-en"), cache
    wspólny dla modeli (ta sama odpowiedź na to samo zadanie = jedno tłumaczenie). Partia = odpowiedzi na kilka
    zadań, treść zadania raz na partię; w partii tylko nieprzezroczyste id (bez nazw modeli i warunków)."""
    llm_fn = llm_fn or _llm_json
    c = cache or Cache(plik_tl_odp())
    uniq: dict[str, dict] = {}
    for p in pozycje:
        if not p["tekst"].strip() or blad(p["tekst"]):
            continue  # pusta odpowiedź i błąd serwera przechodzą bez tłumaczenia (tlumaczenie_z_cache)
        k = klucz_tl(kierunek, p["id"], p["tekst"])
        if k not in c:
            uniq.setdefault(k, {**p, "klucz": k})
    partie = partie_odpowiedzi(sorted(uniq.values(), key=lambda p: (p["id"], p["klucz"])), budzet)
    log(f"[tłumacz {kierunek}] do zrobienia {len(uniq)} odpowiedzi w {len(partie)} partiach")

    def jedna(wszystkie: list[dict]) -> int:
        ps = [p for p in wszystkie if p["klucz"] not in c]  # ponowiona partia: tylko brakujące
        if not ps:
            return len(wszystkie)
        lad, po_zid = [], {}
        for p in ps:
            if p["id"] not in po_zid:
                po_zid[p["id"]] = {"zadanie_pl": p["tresc_pl"][:3000], "zadanie_en": p["tresc_en"][:3000],
                                   "odpowiedzi": []}
                lad.append(po_zid[p["id"]])
            po_zid[p["id"]]["odpowiedzi"].append({"id": p["klucz"][:12], "tekst": p["tekst"]})
        ids = [p["klucz"][:12] for p in ps]
        w = llm_fn("ODPOWIEDZI DO PRZETŁUMACZENIA (JSON; treść zadania tylko do doboru nazw i terminów):\n"
                   + json.dumps(lad, ensure_ascii=False, indent=1), SCHEMA_TL, SYS_ODP[kierunek])
        got = rozpakuj(w, ids)
        wpisy = [{"klucz": p["klucz"], "kierunek": kierunek, "id": p["id"], "zrodlo": p["tekst"],
                  "wynik": got[p["klucz"][:12]], "tlumacz": TLUMACZ} for p in ps if got.get(p["klucz"][:12])]
        c.dodaj(wpisy)
        return len(wszystkie) - len(ps) + len(wpisy)

    nieudane = wykonaj_partie(partie, jedna, rownolegle, f"tłumacz {kierunek}")
    if nieudane:
        log(f"[tłumacz {kierunek}] nieudanych partii: {nieudane}; uruchom ponownie, żeby dokończyć z cache")
    return c


def tlumaczenie_z_cache(c: Cache, kierunek: str, zid: str, tekst: str) -> str | None:
    """Tłumaczenie odpowiedzi z cache; pusta odpowiedź i „(BŁĄD …)” bez zmian; brak w cache → None."""
    if not tekst.strip() or blad(tekst):
        return tekst
    w = c.get(klucz_tl(kierunek, zid, tekst))
    return w["wynik"] if w else None


def zbuduj_wstecz(model: str, zz: list[dict], tl_zad: dict[str, str], c: Cache, wyn: Path | None = None,
                  zrodla: tuple[str, ...] = ("goly_en",), kontrola: bool = True) -> dict:
    """Z cache tłumaczeń składa pliki <model>__<WSTECZ[zrodlo]>.jsonl (np. goly_en → goly_en_pl) i, z kontrolą,
    <model>__goly_pl_rt.jsonl (nadpisywane w całości). Zwraca liczniki kompletności (en_pl = pierwsze źródło,
    <zestaw> = każde źródło)."""
    pl = wczytaj_odp(plik_odp(model, "goly_pl", wyn)) if kontrola else {}
    po_id = {z["id"]: z for z in zz}
    stat = {"pl_rt": 0, "n": len(zz)}
    rek = {WSTECZ[k]: [] for k in zrodla}
    rek_rt = []
    for k in zrodla:
        en = wczytaj_odp(plik_odp(model, k, wyn))
        for zid, z in po_id.items():
            d = en.get(zid)
            if d is None:
                continue
            t = tlumaczenie_z_cache(c, "en-pl", zid, d["odpowiedz"])
            if t is not None:
                rek[WSTECZ[k]].append({"id": zid, "model": model, "konfig": WSTECZ[k],
                                       "odpowiedz": tf_na_pf(t) if czy_pf(z) else t, "odpowiedz_en": d["odpowiedz"],
                                       "sekundy": d.get("sekundy")})
    for zid, z in po_id.items():
        d = pl.get(zid)
        if d is not None:
            t1 = tlumaczenie_z_cache(c, "pl-en", zid, d["odpowiedz"])
            t2 = None if t1 is None else tlumaczenie_z_cache(c, "en-pl", zid, t1)
            if t2 is not None:
                rek_rt.append({"id": zid, "model": model, "konfig": "goly_pl_rt",
                               "odpowiedz": tf_na_pf(t2) if czy_pf(z) else t2, "odpowiedz_en": t1,
                               "odpowiedz_pl": d["odpowiedz"], "sekundy": d.get("sekundy")})
    if kontrola:
        rek["goly_pl_rt"] = rek_rt
    for konfig, r in rek.items():
        p = plik_odp(model, konfig, wyn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r), encoding="utf-8")
        stat[konfig] = len(r)
    stat["en_pl"], stat["pl_rt"] = len(rek[WSTECZ[zrodla[0]]]), len(rek_rt)
    return stat


def linie_zestawow(modele: list[str], wyn_rel: str = "review/jezyk-2026-09-26", zestawy=ZESTAWY) -> list[str]:
    """Linie pm_zestawy.txt: „<model> | pl=PLIK,PLIK” itd. (ten sam plik dla krótkich i esejów)."""
    out = []
    for m in modele:
        for etyk, konfig in zestawy:
            f = f"{wyn_rel}/odpowiedzi/{m}__{konfig}.jsonl"
            out.append(f"{m} | {etyk}={f},{f}")
    return out


def krok_wstecz_lokalnie(modele: list[str], limit: int | None = None, opisy: bool = False) -> dict:
    """--wstecz z lokalnym tłumaczem (TL_ODP baza albo marian), bez kontroli PL→EN→PL."""
    zz = zadania()[:limit] if limit else zadania()
    po_id = {z["id"]: z for z in zz}
    zamkniete = {z["id"] for z in zz if z.get("typ") == "zamkniete"}
    zrodla = _zrodla(opisy)

    def pozycje_modelu(m: str) -> list[dict]:
        return [{"id": zid, "tekst": d["odpowiedz"], "tresc_pl": tekst_pl(po_id[zid])}
                for k in zrodla for zid, d in wczytaj_odp(plik_odp(m, k)).items() if zid in po_id]

    stat: dict = {}
    if TL_ODP == "marian":
        from .tlumacz import Marian
        c = Cache(plik_tl_odp())
        tlumacz_odpowiedzi_lokalnie([p for m in modele for p in pozycje_modelu(m)], Marian(), c, "marian", zamkniete)
        stat = {m: zbuduj_wstecz(m, zz, {}, c, zrodla=zrodla, kontrola=False) for m in modele}
    else:
        from .tlumacz import Baza

        def praca(m, s):
            c = Cache(plik_tl_odp(m))
            tlumacz_odpowiedzi_lokalnie(pozycje_modelu(m), Baza(s.url, s.bez_myslenia, rownolegle=s.rown), c,
                                        f"baza:{m}", zamkniete)
            stat[m] = zbuduj_wstecz(m, zz, {}, c, zrodla=zrodla, kontrola=False)

        na_modelach(modele, praca)
    _zapisz_zestawy(modele, zz, stat, zrodla, False, opisy)
    return stat


def krok_wstecz(modele: list[str], rownolegle: int = 6, limit: int | None = None, opisy: bool = False,
                kontrola: bool = True) -> dict:
    if TL_ODP != "claude":
        return krok_wstecz_lokalnie(modele, limit, opisy)
    zz = zadania()[:limit] if limit else zadania()
    tl_zad = tlumaczenia_zadan()
    po_id = {z["id"]: z for z in zz}
    kontekst = lambda zid: {"tresc_pl": tekst_pl(po_id[zid]), "tresc_en": tl_zad.get(zid, "")}  # noqa: E731
    c = Cache(plik_tl_odp())
    zrodla = _zrodla(opisy)
    # EN→PL odpowiedzi EN i pierwsza noga kontroli (PL→EN) są niezależne: jedna pula, potem druga noga
    poz_en, poz_pl = [], []
    for m in modele:
        for k in zrodla:
            for zid, d in wczytaj_odp(plik_odp(m, k)).items():
                if zid in po_id:
                    poz_en.append({"id": zid, "tekst": d["odpowiedz"], **kontekst(zid)})
        if kontrola:
            for zid, d in wczytaj_odp(plik_odp(m, "goly_pl")).items():
                if zid in po_id:
                    poz_pl.append({"id": zid, "tekst": d["odpowiedz"], **kontekst(zid)})
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:  # dwa kierunki naraz, każdy z połową limitu równoległości
        pol = rownolegle // 2 if kontrola else rownolegle
        f1 = ex.submit(tlumacz_odpowiedzi, poz_en, "en-pl", max(1, pol), cache=c)
        f2 = ex.submit(tlumacz_odpowiedzi, poz_pl, "pl-en", max(1, rownolegle - pol), cache=c) if kontrola else None
        f1.result()
        if f2:
            f2.result()
    druga = []
    for p in poz_pl:
        t1 = tlumaczenie_z_cache(c, "pl-en", p["id"], p["tekst"])
        if t1 is not None:
            druga.append({**p, "tekst": t1})
    if druga:
        tlumacz_odpowiedzi(druga, "en-pl", rownolegle, cache=c)
    log(f"[wstecz] tłumaczenia w {time.time() - t0:.0f} s")
    stat = {m: zbuduj_wstecz(m, zz, tl_zad, c, zrodla=zrodla, kontrola=kontrola) for m in modele}
    _zapisz_zestawy(modele, zz, stat, zrodla, kontrola, opisy)
    return stat


def _zapisz_zestawy(modele: list[str], zz: list[dict], stat: dict, zrodla: tuple[str, ...], kontrola: bool,
                    opisy: bool) -> None:
    """Log kompletności i <WYN>/pm_zestawy.txt (+ rozmiary.json) po tłumaczeniu wstecznym."""
    for m, s in stat.items():
        log(f"[wstecz] {m}: " + ", ".join(f"{WSTECZ[k]} {s[WSTECZ[k]]}/{s['n']}" for k in zrodla)
            + (f", pl→en→pl {s['pl_rt']}/{s['n']}" if kontrola else ""))
    kompletne = all(s[WSTECZ[k]] == s["n"] for s in stat.values() for k in zrodla) and all(
        len(wczytaj_odp(plik_odp(m, "goly_pl"))) >= len(zz) for m in modele) and (
        not kontrola or all(s["pl_rt"] == s["n"] for s in stat.values())) and (
        not opisy or all(len(wczytaj_odp(plik_odp(m, "goly_vlm_pl"))) >= len(zz) for m in modele))
    zestawy = ((("pl", "goly_pl"), ("en→pl", "goly_en_pl")) + ((("pl→en→pl", "goly_pl_rt"),) if kontrola else ())
               + (ZESTAWY_OPIS if opisy else ()) + (ZESTAWY_KB if KB else ()))
    naglowek = ["# Zestawy eksperymentu językowego (matura/jezyk.py) dla matura/pelna_matura.py --zestawy",
                "# sesje: " + ",".join(SESJE) + "; etykieta: <model> | " + " / ".join(e for e, _ in zestawy)]
    if not kompletne:
        naglowek.append("# UWAGA: NIEKOMPLETNE (uruchom ponownie --wstecz, żeby dokończyć z cache)")
    (WYN / "pm_zestawy.txt").write_text(
        "\n".join(naglowek + linie_zestawow(modele, WYN.relative_to(ROOT).as_posix(), zestawy)) + "\n", encoding="utf-8")
    zapisz_rozmiary(modele)
    log(f"[wstecz] pm_zestawy.txt: {'kompletne' if kompletne else 'NIEKOMPLETNE'}")


# ---------------------------------------------------------------- eksperyment 2: sonda

EPOKI = (  # (epoka, wzorzec nazwy działu z kart kanonicznych); pierwsze trafienie wygrywa
    ("starozytnosc", r"starożytn|Pradzieje"),
    ("sredniowiecze", r"średniowiecz|Bizancjum|krucjat|wczesnopiastowsk|rozbicia dzielnicowego|XIV"),
    ("nowozytnosc", r"[Rr]enesans|Reformacj|Odkrycia geograficzne|XVI|oświecenia|Jagiellon|elekcje|Obojga Narodów|"
                    r"Upadek Rzeczypospolitej"),
    ("xix", r"napoleońsk|kongresie wiedeńskim|XIX|1815|Powstanie styczniowe"),
    ("xx_1945", r"I wojn|II Rzeczypospolitej|totalitaryzm|okupacją|odrodzenie państwa"),
    ("po_1945", r"1944|Stalinizm|1957|1981|Dekolonizacja|przełomie tysiącleci|Przemiany cywilizacyjne"),
)
RE_POLSKA = re.compile(r"Pols|pols|Rzeczypospolit|Jagiellon|elekcje|Powstanie styczniowe|okupacją|1981–1989")
# kwoty pytań: historia powszechna 50 (starożytność 10, pozostałe epoki po 8), historia Polski 50 (5 epok po 10)
KWOTY = {("powszechna", "starozytnosc"): 10, **{("powszechna", e): 8 for e, _ in EPOKI[1:]},
         **{("polska", e): 10 for e, _ in EPOKI[1:]}}
N_KONTROLA = 30


def epoka(dzial: str) -> str | None:
    for e, wz in EPOKI:
        if re.search(wz, dzial):
            return e
    return None


def zakres_dzialu(dzial: str) -> str:
    return "polska" if RE_POLSKA.search(dzial) else "powszechna"


def kandydaci_kart(karty: list[dict], zapas: float = 1.7, ziarno: int = ZIARNO) -> list[dict]:
    """Karty z konkretem (data, postać albo termin), bez działu „Historia jako nauka”, losowane z ziarnem
    w komórkach (zakres działu × epoka) po ceil(zapas × kwota)."""
    kom: dict[tuple, list] = {}
    for k in karty:
        e = epoka(k["dzial"])
        if e is None or not (k.get("data") or k.get("postac") or k.get("termin")):
            continue
        kom.setdefault((zakres_dzialu(k["dzial"]), e), []).append(k)
    out, rng = [], random.Random(ziarno)
    for kl in sorted(KWOTY):
        lista = sorted(kom.get(kl, []), key=lambda k: k["fakt"])
        out += rng.sample(lista, min(len(lista), math.ceil(zapas * KWOTY[kl])))
    return [{**k, "klucz": sha("karta", k["fakt"]), "epoka": epoka(k["dzial"])} for k in out]


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def wybierz_sonde(pytania: list[dict], kwoty: dict = KWOTY, n_kontrola: int = N_KONTROLA,
                  ziarno: int = ZIARNO) -> list[dict]:
    """Przyjęte pytania (ok=true) → sonda: kwoty w komórkach (zakres wg autora pytań × epoka), bez powtórzonych
    odpowiedzi; niedobór komórki uzupełniany z tego samego zakresu (inne epoki). id s001…; pole kontrola=True
    dla n_kontrola pytań (po połowie z historii Polski i powszechnej), losowanych z ziarnem."""
    rng = random.Random(ziarno)
    ok = sorted((p for p in pytania if p.get("ok")), key=lambda p: p["klucz"])
    rng.shuffle(ok)
    uzyte, wybrane = set(), []

    def bierz(p) -> bool:
        k = _norm(p["odpowiedz_pl"])
        if k in uzyte or p["klucz"] in {w["klucz"] for w in wybrane}:
            return False
        uzyte.add(k); wybrane.append(p)
        return True

    for kl in sorted(kwoty):
        n = 0
        for p in ok:
            if n >= kwoty[kl]:
                break
            if (p["zakres"], p["epoka"]) == kl and bierz(p):
                n += 1
    for zakres in ("polska", "powszechna"):
        cel = sum(v for (z, _), v in kwoty.items() if z == zakres)
        for p in ok:
            if sum(w["zakres"] == zakres for w in wybrane) >= cel:
                break
            if p["zakres"] == zakres:
                bierz(p)
    wybrane.sort(key=lambda p: (p["zakres"], [e for e, _ in EPOKI].index(p["epoka"]), p["klucz"]))
    out = [{**p, "id": f"s{i:03d}", "kontrola": False} for i, p in enumerate(wybrane, 1)]
    for zakres, n in (("polska", n_kontrola // 2), ("powszechna", n_kontrola - n_kontrola // 2)):
        kand = [p for p in out if p["zakres"] == zakres]
        for p in rng.sample(kand, min(n, len(kand))):
            p["kontrola"] = True
    return out


SCHEMA_PYTANIA = {"type": "object", "additionalProperties": False, "required": ["pytania"], "properties": {
    "pytania": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["id", "ok", "powod", "pytanie_pl", "pytanie_en", "odpowiedz_pl", "odpowiedz_en",
                             "warianty", "typ", "zakres", "zdanie_pl", "zdanie_en"],
                "properties": {"id": {"type": "string"}, "ok": {"type": "boolean"}, "powod": {"type": "string"},
                               "pytanie_pl": {"type": "string"}, "pytanie_en": {"type": "string"},
                               "odpowiedz_pl": {"type": "string"}, "odpowiedz_en": {"type": "string"},
                               "warianty": {"type": "array", "items": {"type": "string"}},
                               "typ": {"type": "string", "enum": ["rok", "osoba", "miejsce", "nazwa"]},
                               "zakres": {"type": "string", "enum": ["polska", "powszechna"]},
                               "zdanie_pl": {"type": "string"}, "zdanie_en": {"type": "string"}}}}}}

SYS_PYTANIA = """Układasz pytania do sondy wiedzy historycznej (zakres matury z historii w Polsce). Dla każdej karty
faktu napisz JEDNO krótkie pytanie faktograficzne, na które istnieje jedna jednoznaczna, krótka odpowiedź: rok,
osoba, miejsce albo nazwa (wydarzenia, dokumentu, bitwy, traktatu, instytucji, dynastii, państwa).
Zasady:
- Pytanie sprawdza wiedzę, nie rozumowanie; jest zrozumiałe bez karty; nie zawiera odpowiedzi ani podpowiedzi.
- Odpowiedź jednoznaczna: jeśli rok, to wydarzenie jednego roku; jeśli osoba, to jedna konkretna osoba.
- Fakt musi być zgodny z Twoją wiedzą historyczną. Gdy karta jest wątpliwa albo nie da się z niej ułożyć
  jednoznacznego pytania, ustaw ok=false i podaj powod (pozostałe pola mogą być puste).
- pytanie_en: wierny odpowiednik po angielsku, tej samej trudności, nazwy w formie przyjętej w anglojęzycznej
  historiografii, bez podpowiedzi i bez polskich oryginałów w nawiasach.
- odpowiedz_pl, odpowiedz_en: kanoniczna odpowiedź w danym języku. warianty: wszystkie akceptowalne formy
  (polskie, angielskie, oryginalne, skrócone), np. „Jagiełło”, „Władysław II Jagiełło”, „Jogaila”.
- typ: rok | osoba | miejsce | nazwa. zakres: polska (historia Polski i ziem polskich) albo powszechna.
- zdanie_pl, zdanie_en: jedno zdanie oznajmujące w danym języku, które zawiera odpowiedź na pytanie, sformułowane
  inaczej niż pytanie (np. z dodatkowym kontekstem), bez innych kandydatów na odpowiedź tego samego typu.
Zwróć JSON zgodny ze schematem: dokładnie jeden wpis na każde id karty."""

SCHEMA_OCENY = {"type": "object", "additionalProperties": False, "required": ["oceny"], "properties": {
    "oceny": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "poprawna"],
              "properties": {"id": {"type": "string"}, "poprawna": {"type": "boolean"}}}}}}

SYS_OCENA = """Oceniasz krótkie odpowiedzi na pytania faktograficzne z historii. Każda pozycja ma pytanie (po polsku
albo po angielsku), poprawną odpowiedź z akceptowanymi wariantami i odpowiedź zdającego.
- poprawna=true, gdy odpowiedź zdającego wskazuje poprawną odpowiedź: jeden z wariantów albo inną równoważną nazwę
  tej samej osoby, miejsca, wydarzenia lub dokumentu (także samo nazwisko, jeśli jednoznacznie wskazuje osobę).
- Bądź łagodny wobec pisowni, literówek, odmiany i języka odpowiedzi: odpowiedź po polsku, po angielsku albo
  w innym języku jest równie dobra. Oceniasz wyłącznie treść.
- Rok musi być dokładny. Zła liczba porządkowa władcy (np. Zygmunt II zamiast Zygmunt III) to błąd.
- Dodatkowe szczegóły nie szkodzą, chyba że przeczą odpowiedzi. Odpowiedź urwana jest poprawna, jeśli zawiera
  poprawną odpowiedź.
- poprawna=false: kilku różnych kandydatów bez wyboru, odmowa, „nie wiem”, samo powtórzenie pytania albo zdania
  z kontekstu bez wskazania odpowiedzi, bełkot, odpowiedź błędna.
Zwróć JSON zgodny ze schematem: dokładnie jedna ocena na każde id."""


def plik_sondy() -> Path:
    return DANE / "sonda.jsonl"


def sonda_pytania(rownolegle: int = 6, partia: int = 20) -> list[dict]:
    """Kandydaci z kart → pytania claude:opus (cache data/jezyk/sonda_kandydaci.jsonl) → wybór 100 → sonda.jsonl."""
    if plik_sondy().exists():
        s = wczytaj_jsonl(plik_sondy())
        log(f"[sonda] pytania już są: {len(s)} ({plik_sondy().relative_to(ROOT)})")
        return s
    karty = wczytaj_jsonl(ROOT / "data" / "karty" / "kanon.jsonl")
    kand = kandydaci_kart(karty)
    c = Cache(DANE / "sonda_kandydaci.jsonl")
    todo = [k for k in kand if k["klucz"] not in c]
    log(f"[sonda] kandydaci: {len(kand)} kart, do zrobienia {len(todo)}")
    po_kl = {k["klucz"][:12]: k for k in todo}

    def jedna(wszystkie: list[dict]) -> int:
        ks = [k for k in wszystkie if k["klucz"] not in c]
        if not ks:
            return len(wszystkie)
        lad = [{"id": k["klucz"][:12], "fakt": k["fakt"], "data": k["data"], "postac": k["postac"],
                "termin": k["termin"]} for k in ks]
        w = _llm_json("KARTY FAKTÓW (JSON):\n" + json.dumps(lad, ensure_ascii=False, indent=1), SCHEMA_PYTANIA,
                      SYS_PYTANIA)
        wpisy = []
        for p in w.get("pytania", []):
            k = po_kl.get(p.get("id"))
            if k is None or k["klucz"] in {x["klucz"] for x in wpisy}:
                continue
            wpisy.append({**p, "klucz": k["klucz"], "epoka": k["epoka"], "dzial": k["dzial"], "fakt": k["fakt"],
                          "zakres_dzialu": zakres_dzialu(k["dzial"]), "autor": TLUMACZ})
        c.dodaj(wpisy)
        return len(wszystkie) - len(ks) + len(wpisy)

    wykonaj_partie([todo[i:i + partia] for i in range(0, len(todo), partia)], jedna, rownolegle, "sonda: pytania")
    wszystkie = [c.get(k["klucz"]) for k in kand if k["klucz"] in c]
    s = wybierz_sonde(wszystkie)
    log(f"[sonda] przyjęte {sum(bool(p.get('ok')) for p in wszystkie)}/{len(wszystkie)}, wybrane {len(s)} "
        f"(polska {sum(p['zakres'] == 'polska' for p in s)}, kontrola {sum(p['kontrola'] for p in s)})")
    plik_sondy().write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in s), encoding="utf-8")
    return s


def prompty_sondy(p: dict) -> dict[str, tuple[str, str]]:
    """warunek → (system, user). Warunki pl, en; dla pytań kontrolnych także pl_ctx, en_ctx (zdanie z odpowiedzią
    w tym samym języku przed pytaniem)."""
    out = {"pl": (SYS_SONDA["pl"], p["pytanie_pl"]), "en": (SYS_SONDA["en"], p["pytanie_en"])}
    if p.get("kontrola"):
        out["pl_ctx"] = (SYS_SONDA["pl"], p["zdanie_pl"] + "\n\n" + p["pytanie_pl"])
        out["en_ctx"] = (SYS_SONDA["en"], p["zdanie_en"] + "\n\n" + p["pytanie_en"])
    return out


def plik_sondy_odp(model: str, wyn: Path | None = None) -> Path:
    return (wyn or WYN) / "sonda" / f"{model}.jsonl"


def sonda_odpowiedz_model(model: str, sonda: list[dict], url: str, bez_myslenia: bool, rownolegle: int,
                          wyn: Path | None = None, czat_fn=None) -> Path:
    """Odpowiedzi modelu na sondę: rekordy {id, model, warunek, odpowiedz, sekundy}; wznawialne po (id, warunek)."""
    if czat_fn is None:
        from .llm import czat as czat_fn
    p = plik_sondy_odp(model, wyn)
    p.parent.mkdir(parents=True, exist_ok=True)
    gotowe = {(d["id"], d["warunek"]) for d in wczytaj_jsonl(p) if not blad(d.get("odpowiedz"))}
    todo = [(q["id"], w, s, u) for q in sonda for w, (s, u) in prompty_sondy(q).items() if (q["id"], w) not in gotowe]
    log(f"[sonda/{model}] do zrobienia {len(todo)}")
    lock = threading.Lock()

    def jedno(x):
        zid, w, s, u = x
        try:
            t, sek = czat_fn(url, s, u, None, max_tokens=40, bez_myslenia=bez_myslenia)
        except Exception as e:  # noqa: BLE001
            t, sek = f"(BŁĄD: {e})", 0.0
        with lock, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": zid, "model": model, "warunek": w, "odpowiedz": t, "sekundy": round(sek, 2)},
                               ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max(1, rownolegle)) as ex:
        list(ex.map(jedno, todo))
    return p


def odpowiedzi_sondy(model: str, wyn: Path | None = None) -> dict[tuple[str, str], str]:
    return {(d["id"], d["warunek"]): d["odpowiedz"] for d in wczytaj_jsonl(plik_sondy_odp(model, wyn))}


def klucz_oceny(zid: str, jezyk: str, odp: str) -> str:
    return sha("sonda", zid, jezyk, odp)


def sonda_oceny(modele: list[str], sonda: list[dict], rownolegle: int = 6, partia: int = 40, llm_fn=None,
                cache: Cache | None = None, wyn: Path | None = None) -> Cache:
    """Ocena poprawności wszystkich odpowiedzi (claude:opus), na ślepo: pozycje wymieszane między modelami
    i językami, nieprzezroczyste id; ta sama instrukcja dla obu języków. Cache data/jezyk/sonda_oceny.jsonl."""
    llm_fn = llm_fn or _llm_json
    c = cache or Cache(DANE / "sonda_oceny.jsonl")
    po_id = {p["id"]: p for p in sonda}
    uniq: dict[str, dict] = {}
    for m in modele:
        for (zid, w), odp in odpowiedzi_sondy(m, wyn).items():
            if zid not in po_id or not odp.strip() or blad(odp):
                continue
            j = w[:2]
            k = klucz_oceny(zid, j, odp)
            if k not in c:
                p = po_id[zid]
                uniq[k] = {"klucz": k, "id": zid, "jezyk": j, "odpowiedz": odp,
                           "pytanie": p["pytanie_pl"] if j == "pl" else p["pytanie_en"],
                           "poprawna_odpowiedz": f"{p['odpowiedz_pl']} / {p['odpowiedz_en']}",
                           "warianty": p["warianty"]}
    poz = sorted(uniq.values(), key=lambda x: x["klucz"])
    random.Random(ZIARNO).shuffle(poz)
    log(f"[sonda: oceny] do oceny {len(poz)}")

    def jedna(wszystkie: list[dict]) -> int:
        ps = [p for p in wszystkie if p["klucz"] not in c]
        if not ps:
            return len(wszystkie)
        lad = [{"id": p["klucz"][:12], "pytanie": p["pytanie"], "poprawna_odpowiedz": p["poprawna_odpowiedz"],
                "akceptowane_warianty": p["warianty"], "odpowiedz_zdajacego": p["odpowiedz"][:600]} for p in ps]
        w = llm_fn("ODPOWIEDZI DO OCENY (JSON):\n" + json.dumps(lad, ensure_ascii=False, indent=1), SCHEMA_OCENY,
                   SYS_OCENA)
        oc = {str(o.get("id")): o.get("poprawna") for o in w.get("oceny", [])}
        wpisy = [{"klucz": p["klucz"], "id": p["id"], "jezyk": p["jezyk"], "odpowiedz": p["odpowiedz"],
                  "poprawna": bool(oc[p["klucz"][:12]]), "sedzia": TLUMACZ}
                 for p in ps if isinstance(oc.get(p["klucz"][:12]), bool)]
        c.dodaj(wpisy)
        return len(wszystkie) - len(ps) + len(wpisy)

    wykonaj_partie([poz[i:i + partia] for i in range(0, len(poz), partia)], jedna, rownolegle, "sonda: oceny")
    return c


def krok_sonda(modele: list[str], rownolegle: int = 6, etap: str | None = None) -> None:
    sonda = sonda_pytania(rownolegle) if etap in (None, "pytania") else wczytaj_jsonl(plik_sondy())
    if not sonda:
        raise SystemExit("brak data/jezyk/sonda.jsonl: uruchom --sonda --etap pytania")
    if etap in (None, "odpowiedzi"):
        na_modelach(modele, lambda m, s: sonda_odpowiedz_model(m, sonda, s.url, s.bez_myslenia, s.rown))
    if etap in (None, "oceny"):
        sonda_oceny(modele, sonda, rownolegle)


# ---------------------------------------------------------------- statystyka

_SLOWA_PL = frozenset("się jest nie oraz że był była było były który która które przez dla jako został została także "
                      "również roku w z na do od po jego ich".split())
_SLOWA_EN = frozenset("the and of is was were to in which that by with as for an his it this from are be".split())


def jezyk_tekstu(t: str) -> str:
    """Przybliżony język odpowiedzi (opis danych, nie ocena): "pl" | "en" | "inny" | "krotka" (za mało słów,
    np. „1. P 2. F”). Słowa funkcyjne PL/EN plus polskie znaki diakrytyczne."""
    litery = re.findall(r"[^\W\d_]", t)
    if len(litery) >= 4 and sum(ord(c) > 0x24F for c in litery) > 0.3 * len(litery):
        return "inny"  # pismo niełacińskie (np. chińskie)
    slowa = re.findall(r"[^\W\d_]+", t.lower())
    n_pl = sum(w in _SLOWA_PL for w in slowa) + len(re.findall(r"[ąćęłńśźż]", t.lower())) / 3
    n_en = sum(w in _SLOWA_EN for w in slowa)
    if len(slowa) < 4 or n_pl + n_en < 2:
        return "krotka"
    return "pl" if n_pl >= 2 * n_en else "en" if n_en >= 2 * n_pl else "inny"


def jezyki_odpowiedzi(teksty: list[str]) -> dict[str, int]:
    out = {"pl": 0, "en": 0, "inny": 0, "krotka": 0}
    for t in teksty:
        out[jezyk_tekstu(t)] += 1
    return out

def bootstrap_roznica(a, b, wagi=None, n: int = 10_000, ziarno: int = ZIARNO) -> dict:
    """Sparowany bootstrap różnicy b − a (te same pozycje w obu warunkach; losowanie pozycji ze zwracaniem).
    Bez wag: różnica średnich (np. trafności). Z wagami (pkt_max pozycji): (Σb − Σa) / Σwagi.
    → {roznica, lo, hi} (95%, percentyle 2,5 i 97,5), n pozycji."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("bootstrap: puste albo niesparowane wektory")
    w = np.ones_like(a) if wagi is None else np.asarray(wagi, float)
    rng = np.random.default_rng(ziarno)
    idx = rng.integers(0, a.size, size=(n, a.size))
    d = (b[idx] - a[idx]).sum(1) / w[idx].sum(1)
    return {"roznica": float((b - a).sum() / w.sum()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "n": int(a.size)}


def wyniki_sondy(modele: list[str], sonda: list[dict], c: Cache, wyn: Path | None = None) -> dict:
    """model → trafności PL/EN, różnica EN−PL z przedziałem, to samo dla kontroli rozumienia i wg zakresu."""
    out = {}
    for m in modele:
        odp = odpowiedzi_sondy(m, wyn)

        def wektor(ids, warunek):
            v = []
            for i in ids:
                o = odp.get((i, warunek))
                if o is None:
                    return None
                w = c.get(klucz_oceny(i, warunek[:2], o)) if o.strip() and not blad(o) else {"poprawna": False}
                if w is None:
                    return None
                v.append(1.0 if w["poprawna"] else 0.0)
            return v

        def para(ids, wa, wb):
            a, b = wektor(ids, wa), wektor(ids, wb)
            if a is None or b is None or not ids:
                return None
            return {"pl": float(np.mean(a)), "en": float(np.mean(b)), **bootstrap_roznica(a, b),
                    "tylko_en": int(sum(y > x for x, y in zip(a, b))), "tylko_pl": int(sum(x > y for x, y in zip(a, b)))}

        ids = [p["id"] for p in sonda]
        kon = [p["id"] for p in sonda if p.get("kontrola")]
        jez = {w: jezyki_odpowiedzi([o for (i, ww), o in odp.items() if ww == w]) for w in ("pl", "en")}
        out[m] = {"jezyk_odpowiedzi": jez, "sonda": para(ids, "pl", "en"), "kontrola_bez": para(kon, "pl", "en"),
                  "kontrola_ctx": para(kon, "pl_ctx", "en_ctx"),
                  "polska": para([p["id"] for p in sonda if p["zakres"] == "polska"], "pl", "en"),
                  "powszechna": para([p["id"] for p in sonda if p["zakres"] == "powszechna"], "pl", "en")}
    return out


def punkty_pozycji(z: dict, odp: str | None, cache: dict, sedzia_model: str) -> float | None:
    """Punkty jednej pozycji tak jak w matura/pelna_matura.py (automat klucza, gdy pewny, inaczej cache sędziego);
    temat eseju waży 1/3 (esej arkusza = średnia z 3 tematów). None = brak odpowiedzi albo brak oceny."""
    from . import klucz, sedzia
    if odp is None:
        return None
    if not odp.strip() or blad(odp):
        return 0.0
    if not z["esej"] and klucz.obslugiwane(z):
        o = klucz.ocen(z, odp)
        if o["pewne"]:
            return float(o["pkt"])
    w = cache.get(sedzia.klucz(z["id"], odp, sedzia_model))
    if w is None:
        return None
    pk = max(0, min(int(w["pkt"]), z["pkt_max"]))
    return pk / 3 if z["esej"] else float(pk)


def wyniki_arkusza(modele: list[str], sedzia_model: str = "claude:opus") -> dict:
    """Czyta (tylko czyta) cache ocen sędziego: punkty na pozycję w zestawach pl / en→pl / pl→en→pl,
    sumy i sparowane przedziały bootstrap różnic na pozycjach ocenionych we wszystkich trzech zestawach."""
    from .sedzia import wczytaj_cache
    cache = wczytaj_cache()
    zz = zadania()
    out = {}
    for m in modele:
        pliki = {k: wczytaj_odp(plik_odp(m, k)) for _, k in ZESTAWY}
        pkt = {k: {} for k in pliki}
        for z in zz:
            for k, d in pliki.items():
                v = punkty_pozycji(z, (d.get(z["id"]) or {}).get("odpowiedz"), cache, sedzia_model)
                if v is not None:
                    pkt[k][z["id"]] = v
        wspolne = [z for z in zz if all(z["id"] in pkt[k] for k in pkt)]
        wagi = [z["pkt_max"] / 3 if z["esej"] else z["pkt_max"] for z in wspolne]
        jez = {k: jezyki_odpowiedzi([d["odpowiedz"] for d in wczytaj_odp(plik_odp(m, k)).values()])
               for k in ("goly_pl", "goly_en")}
        r = {"jezyk_odpowiedzi": jez, "ocenione": {k: len(v) for k, v in pkt.items()}, "n_pozycji": len(zz),
             "n_wspolnych": len(wspolne),
             "max_wspolnych": round(sum(wagi), 1)}
        if wspolne:
            v = {k: [pkt[k][z["id"]] for z in wspolne] for k in pkt}
            r["proc"] = {k: round(100 * sum(x) / sum(wagi), 1) for k, x in v.items()}
            r["en_pl_minus_pl"] = bootstrap_roznica(v["goly_pl"], v["goly_en_pl"], wagi)
            r["pl_rt_minus_pl"] = bootstrap_roznica(v["goly_pl"], v["goly_pl_rt"], wagi)
            r["en_pl_minus_pl_rt"] = bootstrap_roznica(v["goly_pl_rt"], v["goly_en_pl"], wagi)
        out[m] = r
    return out


# ---------------------------------------------------------------- raport

def _pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def _przedzial(r: dict | None) -> str:
    return "brak" if not r else f"{_pp(r['roznica'])} [{_pp(r['lo'])}; {_pp(r['hi'])}]"


def md_sondy(ws: dict, sonda: list[dict], arkusz: dict | None) -> str:
    n, nk = len(sonda), sum(bool(p.get("kontrola")) for p in sonda)
    L = ["# Sonda wiedzy PL vs EN (eksperyment językowy)", "",
         "**Co to jest:** wynik sondy wiedzy bez formatu matury: te same pytania faktograficzne po polsku i po angielsku,",
         "te same modele, goły prompt („Odpowiedz krótko.” / „Answer briefly.”, do 40 tokenów), ten sam sędzia "
         f"(`{TLUMACZ}`). Pytania: {n} z kart kanonicznych (data/karty/kanon.jsonl, nie z arkuszy CKE), połowa "
         "historia Polski, połowa powszechna, wszystkie epoki; plik data/jezyk/sonda.jsonl. Kontrola rozumienia: "
         f"{nk} z tych pytań z dołączonym jednym zdaniem zawierającym odpowiedź (w języku pytania).",
         "**Po co:** rozdzielić brak wiedzy od bariery języka. Jeśli model wie, ale nie rozumie po polsku, trafność EN "
         "jest wyraźnie wyższa niż PL, a z odpowiedzią w kontekście różnica PL/EN powinna zostać (rozumienie).",
         "**Co zrobić:** czytać kolumnę EN−PL z 95% przedziałem (bootstrap sparowany po pytaniach, 10 000 losowań, "
         "ziarno 0); przedział obejmujący 0 = brak rozstrzygnięcia na tej próbie. Szczegóły: raport.json obok; "
         "odpowiedzi: sonda/<model>.jsonl; przeliczenie: `uv run python -m matura.jezyk --raport`.", "",
         "## Sonda (bez kontekstu)", "",
         f"| model | PL % | EN % | EN−PL p.p. [95% CI] | tylko EN / tylko PL | n |", "|---|---|---|---|---|---|"]
    for m, r in ws.items():
        s = r["sonda"]
        L.append(f"| {m} | " + (f"{100 * s['pl']:.0f} | {100 * s['en']:.0f} | {_przedzial(s)} | "
                                f"{s['tylko_en']} / {s['tylko_pl']} | {s['n']} |" if s else "brak | | | | |"))
    L += ["", "„tylko EN / tylko PL”: liczba pytań poprawnych tylko w jednym języku (pary niezgodne).", "",
          f"## Kontrola rozumienia ({nk} pytań; te same pytania bez i z kontekstem)", "",
          "| model | bez kontekstu PL % | EN % | EN−PL [95% CI] | z kontekstem PL % | EN % | EN−PL [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for m, r in ws.items():
        a, b = r["kontrola_bez"], r["kontrola_ctx"]
        fa = f"{100 * a['pl']:.0f} | {100 * a['en']:.0f} | {_przedzial(a)}" if a else "brak | | "
        fb = f"{100 * b['pl']:.0f} | {100 * b['en']:.0f} | {_przedzial(b)}" if b else "brak | | "
        L.append(f"| {m} | {fa} | {fb} |")
    L += ["", "## Według zakresu pytań (bez kontekstu)", "",
          "| model | historia Polski PL % | EN % | EN−PL [95% CI] | powszechna PL % | EN % | EN−PL [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for m, r in ws.items():
        cz = [f"{100 * x['pl']:.0f} | {100 * x['en']:.0f} | {_przedzial(x)}" if x else "brak | | "
              for x in (r["polska"], r["powszechna"])]
        L.append(f"| {m} | {cz[0]} | {cz[1]} |")
    L += ["", "## Pełny arkusz CKE 2024-maj + 2025-maj (eksperyment 1)", ""]
    if not arkusz or not any(r.get("n_wspolnych") for r in arkusz.values()):
        L += ["Brak ocen sędziego w cache dla zestawów z pm_zestawy.txt (ocena pełnej matury: "
              "`matura/pelna_matura.py --zestawy review/jezyk-2026-09-26/pm_zestawy.txt`); po ocenie uruchomić "
              "`--raport` ponownie."]
    else:
        L += ["Punkty jak w matura/pelna_matura.py (automat klucza albo cache sędziego; temat eseju waży 1/3), tylko "
              "pozycje ocenione we wszystkich trzech zestawach. Różnice w p.p. maksimum, bootstrap sparowany po "
              "pozycjach. „en→pl − pl→en→pl” porównuje dwa zestawy przechodzące przez tego samego tłumacza.", "",
              "| model | pozycje | pl % | en→pl % | pl→en→pl % | en→pl − pl [CI] | pl→en→pl − pl [CI] | "
              "en→pl − pl→en→pl [CI] |", "|---|---|---|---|---|---|---|---|"]
        for m, r in arkusz.items():
            if not r.get("n_wspolnych"):
                L.append(f"| {m} | 0/{r['n_pozycji']} | brak | | | | | |"); continue
            p = r["proc"]
            L.append(f"| {m} | {r['n_wspolnych']}/{r['n_pozycji']} | {p['goly_pl']} | {p['goly_en_pl']} | "
                     f"{p['goly_pl_rt']} | {_przedzial(r['en_pl_minus_pl'])} | {_przedzial(r['pl_rt_minus_pl'])} | "
                     f"{_przedzial(r['en_pl_minus_pl_rt'])} |")
    return "\n".join(L) + "\n"


def krok_raport(modele: list[str], sedzia_model: str = "claude:opus") -> dict:
    sonda = wczytaj_jsonl(plik_sondy())
    ws = wyniki_sondy(modele, sonda, Cache(DANE / "sonda_oceny.jsonl")) if sonda else {}
    ark = wyniki_arkusza(modele, sedzia_model)
    wynik = {"co_to_jest": "Eksperyment językowy (matura/jezyk.py): sonda wiedzy PL vs EN i pełny arkusz PL/EN.",
             "po_co": "Rozstrzygnąć, czy małe modele słabo zdają maturę przez język, czy przez brak wiedzy.",
             "co_zrobic": "Porównać różnice EN−PL z przedziałami; pełny arkusz po ocenie matura/pelna_matura.py.",
             "meta": {"data": time.strftime("%Y-%m-%d %H:%M"), "tlumacz_i_sedzia_sondy": TLUMACZ,
                      "sedzia_arkusza": sedzia_model, "bootstrap": {"n": 10_000, "ziarno": ZIARNO},
                      "sesje": list(SESJE), "n_pytan": len(sonda), "n_kontrola": sum(bool(p.get("kontrola")) for p in sonda)},
             "sonda": ws, "arkusz": ark}
    WYN.mkdir(parents=True, exist_ok=True)
    (WYN / "raport.json").write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding="utf-8")
    if sonda:
        (WYN / "SONDA.md").write_text(md_sondy(ws, sonda, ark), encoding="utf-8")
    log(f"[raport] zapisano {WYN.relative_to(ROOT)}/raport.json" + (" i SONDA.md" if sonda else ""))
    return wynik


def main(argv: list[str] | None = None) -> int:
    global WYN, SESJE, TL_ZADAN, TL_ODP, KB, TYLKO_ESEJE
    ap = argparse.ArgumentParser(description="Eksperyment językowy: pełny arkusz PL/EN i sonda wiedzy PL/EN.")
    ap.add_argument("--tlumacz", action="store_true", help="krok 1: tłumaczenie zadań PL→EN")
    ap.add_argument("--generuj", action="store_true", help="krok 2: odpowiedzi goly_pl i goly_en (GPU)")
    ap.add_argument("--wstecz", action="store_true", help="kroki 3–5: EN→PL, PL→EN→PL, pm_zestawy.txt")
    ap.add_argument("--sonda", action="store_true", help="eksperyment 2: pytania, odpowiedzi, oceny")
    ap.add_argument("--etap", choices=["pytania", "odpowiedzi", "oceny"], help="tylko jeden etap sondy")
    ap.add_argument("--raport", action="store_true", help="SONDA.md i raport.json")
    ap.add_argument("--modele", help="lista nazw po przecinku (domyślnie wszystkie z MODELE)")
    ap.add_argument("--wyniki", help="katalog wyników względem repo (domyślnie review/jezyk-2026-09-26)")
    ap.add_argument("--opisy", action="store_true",
                    help="--generuj/--wstecz: także warunki z blokiem opisów ilustracji (goly_vlm_pl, goly_vlm_en)")
    ap.add_argument("--bez-kontroli", action="store_true", help="--wstecz: bez kontroli PL→EN→PL")
    ap.add_argument("--rownolegle", type=int, default=6, help="równoległe wywołania claude:opus (maks. 6)")
    ap.add_argument("--limit", type=int, help="tylko N pierwszych zadań (test dymny)")
    ap.add_argument("--sesje", help="arkusze po przecinku, np. 2025-maj (domyślnie " + ",".join(SESJE) + ")")
    ap.add_argument("--sedzia", default="claude:opus", help="sędzia pełnej matury, którego cache czyta --raport")
    ap.add_argument("--tl-zadan", choices=TLUMACZE + tuple(MODELE), default="claude",
                    help="tłumacz zadań PL→EN (--tlumacz, --generuj): claude (sufit), baza (ten sam model), marian "
                         "albo nazwa modelu z MODELE (osobny tłumacz, najpierw --tlumacz)")
    ap.add_argument("--kb", action="store_true",
                    help="--generuj/--wstecz: także warunek goly_en_kb (2 hasła bazy po angielsku przed zadaniem EN)")
    ap.add_argument("--tl-odp", choices=TLUMACZE, default="claude",
                    help="tłumacz odpowiedzi EN→PL (--wstecz): claude, baza albo marian (lokalnie: bez kontroli)")
    ap.add_argument("--tylko-eseje", action="store_true",
                    help="tylko tematy wypracowań arkuszy (bez zadań krótkich), we wszystkich krokach")
    a = ap.parse_args(argv)
    modele = [m.strip() for m in a.modele.split(",")] if a.modele else list(MODELE)
    nieznane = [m for m in modele if m not in MODELE]
    if nieznane:
        print(f"BŁĄD: nieznane modele {nieznane}; dostępne: {list(MODELE)}", file=sys.stderr)
        return 2
    rown = max(1, min(6, a.rownolegle))
    if not any((a.tlumacz, a.generuj, a.wstecz, a.sonda, a.raport)):
        ap.print_help()
        return 2
    if a.sesje:
        from .devset import JSON
        sesje = tuple(s.strip() for s in a.sesje.split(",") if s.strip())
        nieznane = [s for s in sesje if not (JSON / f"historia-{s}.json").exists()]
        if not sesje or nieznane:
            print(f"BŁĄD: nieznane sesje {nieznane or a.sesje!r} (brak arkusza w {JSON})", file=sys.stderr)
            return 2
        testowe = [s for s in sesje if s[:4].isdigit() and int(s[:4]) >= 2026]
        if testowe:  # jak pelna_matura.sprawdz_sesje: test 2026 tylko raz, na końcu, nie w eksperymencie
            print(f"BŁĄD: sesje testowe {testowe} są niedozwolone w eksperymencie językowym", file=sys.stderr)
            return 2
        SESJE = sesje
    TL_ZADAN, TL_ODP, KB, TYLKO_ESEJE = a.tl_zadan, a.tl_odp, a.kb, a.tylko_eseje
    if a.wyniki:
        WYN = ROOT / a.wyniki
    DANE.mkdir(parents=True, exist_ok=True)
    WYN.mkdir(parents=True, exist_ok=True)
    if a.tlumacz:
        krok_tlumacz(rown, a.limit)
    if a.generuj:
        krok_generuj(modele, a.limit, a.opisy)
    if a.wstecz:
        krok_wstecz(modele, rown, a.limit, a.opisy, not a.bez_kontroli)
    if a.sonda:
        krok_sonda(modele, rown, a.etap)
    if a.raport:
        krok_raport(modele, a.sedzia)
    return 0


if __name__ == "__main__":
    sys.exit(main())
