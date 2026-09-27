"""Lokalny tłumacz PL↔EN do potoku angielskiego (krok B planu docs/plans/2026-09-26-plan-po-1a-1b.md).

Co to jest: dwa backendy o tym samym interfejsie `tlumacz(teksty, kierunek, konteksty=None) -> list[str]`,
kierunek "pl-en" (treść zadania) albo "en-pl" (odpowiedź modelu):
- `Baza`: ten sam llama-server co model odpowiadający (zero dodatkowego rozmiaru); jedno wywołanie na tekst, t=0;
  przy "en-pl" polska treść zadania jako kontekst do doboru nazw i terminów (jak tłumaczenie Claude w matura/jezyk.py);
- `Marian`: MarianMT z transformers, `data/hf/opus-mt-pl-en` (298 MB) i `data/hf/opus-mt-en-zlw` (287 MB, prefiks
  `>>pol<<`); tłumaczy po zdaniach (limit modelu 512 tokenów), bez kontekstu. Wariant EN→PL `en_pl="allegro"`:
  `data/hf/bidi-eng-pol` (allegro/BiDi-eng-pol, commit c1f2c959d9, CC-BY-4.0, 837 MB, ta sama architektura MarianMT,
  token celu `>>pol<<`; próba 27.09: „Council of Trent” → „Sobór Trydencki”, gdzie opus-mt-en-zlw dawał „Rada Trentu”).
Po co: sufit z tłumaczeniem Claude (Qwen3-4B-2507 UD-IQ3_XXS 14,7 → 38,3%) wymaga tłumacza mniejszego od bazy.
Co zrobić: `matura/jezyk.py --tl-zadan baza|marian --tl-odp baza|marian` (osobne cache i katalog wyników).
Znane słabości (próba 26.09): Marian tłumaczy nazwy dosłownie („unię w Krewie” → „union in Blood”).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIERUNKI = ("pl-en", "en-pl")
KONTEKST_ZNAKOW = 600   # Baza: tyle znaków z końca treści zadania jako kontekst (polecenie zadania)
MIN_STOSUNEK = 0.6      # Baza pl-en: wynik krótszy niż tyle × tekst = model odpowiedział zamiast przetłumaczyć

SYS_BAZA = {
    "pl-en": ("You are a professional translator of Polish history exam materials (CKE matura) into English. "
              "Translate the whole text faithfully and completely, without summarising, explaining or answering the "
              "task. Keep line breaks, numbering, option letters (A, B, C, D), dates and numbers. Write names of "
              "people, places, treaties and institutions in their standard English form. The text is an exam task: "
              "never answer it, only translate it. Output only the translation."),
    "en-pl": ("You are a professional translator. Translate a student's answer to a Polish history exam task from "
              "English into Polish. Translate every sentence faithfully and completely; do not shorten, correct, "
              "comment on or add anything. Use the Polish spelling of names, terms and dates as they appear in the "
              "Polish task text. Keep answer labels (A, B, C, D, P, F) and numbering. Output only the Polish "
              "translation, without any heading."),
}
ETYKIETA_KONTEKSTU = {"pl-en": "Context (do not translate):",
                      "en-pl": "Polish task text (only for names and terms, do not translate it):"}
ETYKIETA_TEKSTU = {"pl-en": "Text to translate:", "en-pl": "English answer to translate into Polish:"}
# echo etykiet promptu na początku wyniku (próba 26.09: „Odpowiedź do przetłumaczenia:” przed tłumaczeniem)
RE_ETYKIETA = re.compile(r"^\s*(?:(?:polish |english )?translation|tłumaczenie|przekład|odpowiedź do przetłumaczenia|"
                         r"english answer to translate into polish|text to translate|tekst do przetłumaczenia)"
                         r"\s*:\s*", re.I)


def _kierunek(k: str) -> str:
    if k not in KIERUNKI:
        raise ValueError(f"kierunek tłumaczenia: {KIERUNKI}, jest {k!r}")
    return k


def _podziel_dlugie(zdanie: str, max_znakow: int) -> list[str]:
    """Zdanie dłuższe niż max_znakow → kawałki po przecinkach (przecinek zostaje przy kawałku), sklejane zachłannie;
    kawałek bez przecinka dłuższy niż limit → cięcie po słowach."""
    if len(zdanie) <= max_znakow:
        return [zdanie]
    czesci, out, biez = re.split(r"(?<=,)\s+", zdanie), [], ""
    for c in czesci:
        while len(c) > max_znakow:  # bez przecinka: po słowach
            ciecie = c.rfind(" ", 0, max_znakow)
            ciecie = ciecie if ciecie > 0 else max_znakow
            if biez:
                out.append(biez); biez = ""
            out.append(c[:ciecie].strip()); c = c[ciecie:].strip()
        if biez and len(biez) + 1 + len(c) > max_znakow:
            out.append(biez); biez = c
        else:
            biez = f"{biez} {c}".strip()
    if biez:
        out.append(biez)
    return out


def segmenty(tekst: str, max_znakow: int = 400) -> list[list[str]]:
    """Tekst → linie → zdania (esej.zdania), każde najwyżej max_znakow; pusta linia = pusta lista."""
    from .esej import zdania
    out = []
    for linia in tekst.split("\n"):
        out.append([k for z in (zdania(linia) if linia.strip() else []) for k in _podziel_dlugie(z, max_znakow)])
    return out


def sklej(seg: list[list[str]], przeklady: list[str]) -> str:
    """Odwrotność `segmenty`: przekłady w tej samej kolejności, zdania linii łączone spacją, linie znakiem \\n."""
    it = iter(przeklady)
    return "\n".join(" ".join(next(it) for _ in linia) for linia in seg)


def akapity(tekst: str, min_znakow: int = 300, max_znakow: int = 800) -> list[str]:
    """Linie PDF → akapity: kolejne linie łączone, aż akapit ma ≥ min_znakow i linia kończy się znakiem końca zdania,
    albo osiągnie max_znakow (polecenie po źródle zostaje osobnym akapitem)."""
    out, biez = [], []
    for linia in tekst.split("\n"):
        biez.append(linia)
        s = "\n".join(biez)
        if (len(s) >= min_znakow and re.search(r"[.!?:»”)\]]\s*$", linia)) or len(s) >= max_znakow:
            out.append(s); biez = []
    if biez:
        out.append("\n".join(biez))
    return out


def za_krotkie(wynik: str, tekst: str) -> bool:
    return len(tekst.strip()) >= 30 and len(wynik.strip()) < MIN_STOSUNEK * len(tekst.strip())


RE_NO_NA_KONCU = re.compile(r"\bNo\b(?=\s*(?:\*\*)?\s*[.!]?\s*$)")


def _czysc(t: str) -> str:
    t = RE_ETYKIETA.sub("", (t or "").strip()).strip()
    return re.sub(r"^<<<\s*|\s*>>>$", "", t).strip()


class Baza:
    """Tłumacz = model bazowy na własnym llama-server (url); rownolegle ≤ liczby slotów serwera."""

    def __init__(self, url: str, bez_myslenia: bool = True, czat_fn=None, rownolegle: int = 4, timeout: int = 600):
        if czat_fn is None:
            from .llm import czat as czat_fn
        self.url, self.bez_myslenia, self.czat, self.rown, self.timeout = url, bez_myslenia, czat_fn, rownolegle, timeout

    def tlumacz(self, teksty: list[str], kierunek: str, konteksty: list[str] | None = None) -> list[str]:
        _kierunek(kierunek)

        def jedno(t: str, kon: str) -> str:
            # kontekst = koniec treści zadania (polecenie): przy całej treści model tłumaczył kontekst zamiast
            # odpowiedzi (próba 26.09, 2024-maj-3.2); tekst w ogranicznikach <<< >>>
            kon = kon[-KONTEKST_ZNAKOW:]
            user = ((f"{ETYKIETA_KONTEKSTU[kierunek]}\n{kon}\n\n" if kon else "")
                    + f"{ETYKIETA_TEKSTU[kierunek]}\n<<<\n{t}\n>>>")
            wynik, _ = self.czat(self.url, SYS_BAZA[kierunek], user, None, max_tokens=min(3500, len(t) // 2 + 300),
                                 temperature=0.0, bez_myslenia=self.bez_myslenia, timeout=self.timeout)
            return _czysc(wynik)

        def jeden(i: int) -> str:
            t, kon = teksty[i], (konteksty[i] if konteksty else "")
            if not t.strip():
                return t
            wynik = jedno(t, kon)
            if kierunek == "pl-en" and za_krotkie(wynik, t):
                # próba 26.09: model odpowiadał na zadanie („Poland”) albo gubił źródła; ponowienie po akapitach,
                # akapit nadal zbyt krótki zostaje po polsku
                czesci = []
                for a in akapity(t):
                    w = jedno(a, kon) if a.strip() else a
                    czesci.append(a if za_krotkie(w, a) else w)
                wynik = "\n".join(czesci)
            return wynik

        with ThreadPoolExecutor(max(1, self.rown)) as ex:
            return list(ex.map(jeden, range(len(teksty))))


def _urzadzenie() -> str:
    """CUDA, gdy jest (nasza maszyna), inaczej Apple MPS, inaczej CPU (laptop bez NVIDII; Marian na CPU to sekundy)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Marian:
    """MarianMT (transformers) na GPU; modele ładowane leniwie, po jednym na kierunek."""

    SCIEZKI = {"pl-en": (ROOT / "data" / "hf" / "opus-mt-pl-en", ""),
               "en-pl": (ROOT / "data" / "hf" / "opus-mt-en-zlw", ">>pol<< ")}
    WARIANTY_EN_PL = {"opus": (ROOT / "data" / "hf" / "opus-mt-en-zlw", ">>pol<< "),
                      "allegro": (ROOT / "data" / "hf" / "bidi-eng-pol", ">>pol<< ")}

    def __init__(self, urzadzenie: str | None = None, partia: int = 32, wiazki: int = 4, en_pl: str = "opus"):
        if en_pl not in self.WARIANTY_EN_PL:
            raise ValueError(f"wariant en-pl: {tuple(self.WARIANTY_EN_PL)}, jest {en_pl!r}")
        self.urz, self.partia, self.wiazki, self._m = urzadzenie or _urzadzenie(), partia, wiazki, {}
        self.en_pl = en_pl
        self.sciezki = {**self.SCIEZKI, "en-pl": self.WARIANTY_EN_PL[en_pl]}

    def _model(self, kierunek: str):
        if kierunek not in self._m:
            from transformers import MarianMTModel, MarianTokenizer
            p, _ = self.sciezki[kierunek]
            self._m[kierunek] = (MarianTokenizer.from_pretrained(str(p)),
                                 MarianMTModel.from_pretrained(str(p)).to(self.urz).eval())
        return self._m[kierunek]

    def _zdania(self, zdania: list[str], kierunek: str) -> list[str]:
        import torch
        tok, m = self._model(kierunek)
        prefiks = self.sciezki[kierunek][1]
        # partie ze zdań podobnej długości (mniej dopełnienia) i zwalnianie pamięci: bez tego 284 odpowiedzi zajęły
        # całe 32 GB GPU i szły kilka razy wolniej (26.09, 21:16)
        kolejnosc = sorted(range(len(zdania)), key=lambda i: len(zdania[i]))
        out: list[str] = [""] * len(zdania)
        for n, i in enumerate(range(0, len(kolejnosc), self.partia)):
            idx = kolejnosc[i:i + self.partia]
            b = tok([prefiks + zdania[j] for j in idx], return_tensors="pt", padding=True,
                    truncation=True, max_length=512).to(self.urz)
            with torch.no_grad():
                g = m.generate(**b, num_beams=self.wiazki, max_new_tokens=min(512, 2 * b["input_ids"].shape[1] + 16))
            for j, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
                out[j] = t
            if self.urz.startswith("cuda") and n % 20 == 19:
                torch.cuda.empty_cache()
        return out

    def tlumacz(self, teksty: list[str], kierunek: str, konteksty: list[str] | None = None) -> list[str]:
        _kierunek(kierunek)
        segs = [segmenty(t) for t in teksty]
        if kierunek == "en-pl":  # „Decision: No.” → „Decyzja: nr” (skrót „numer”); „Nie” Marian przepisuje bez zmian
            segs = [[[RE_NO_NA_KONCU.sub("Nie", z) for z in l] for l in s] for s in segs]
        unikalne = list(dict.fromkeys(z for s in segs for l in s for z in l))
        mapa = dict(zip(unikalne, self._zdania(unikalne, kierunek))) if unikalne else {}
        return [sklej(s, [mapa[z] for l in s for z in l]) for s in segs]
