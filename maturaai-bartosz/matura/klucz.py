"""Automatyczna ocena zadań z jednoznacznym kluczem (bez sędziego LLM).

Co to jest: parser klucza CKE (`rozwiazanie`) i odpowiedzi modelu dla zadań zamkniętych (litera, P/F,
przyporządkowanie, kolejność) oraz krótkich otwartych „Podaj nazwę / nazwisko / pojęcie".
Po co: szybka i darmowa informacja zwrotna dla harnessu; sędzia LLM potrzebny tylko do reszty.
Zasada: gdy odpowiedzi nie da się jednoznacznie odczytać, wynik jest NIEROZSTRZYGNIĘTY (pewne=False)
i zadanie idzie do sędziego. Nie zgadujemy. Punktacja CKE: pkt = max(0, pkt_max - liczba błędnych wskazań)
(1 pkt: wszystko albo nic; 2 pkt przy 3 wskazaniach: 3 → 2, 2 → 1).
"""
from __future__ import annotations

import re
import unicodedata

KRESKA = r"[–—\-−:=]"
SEP = r"(?:\s*[\.\)]\s*|\s*" + KRESKA + r"\s*|\s+)"
PF_SLOWA = {"p": "P", "f": "F", "prawda": "P", "fałsz": "F", "falsz": "F", "prawdziwe": "P", "fałszywe": "F",
            "prawdziwa": "P", "fałszywa": "F", "prawdziwy": "P", "fałszywy": "F", "tak": "P", "nie": "F"}
STOP = {"rok", "r", "w", "i", "się", "sie", "styl", "roku", "oraz"}


def _czysc(s: str) -> str:
    """Usuwa formatowanie Markdown i ujednolica myślniki."""
    s = re.sub(r"[*_`#>]", "", s)
    return s.replace(" ", " ").strip()


def _ascii(s: str) -> str:
    s = s.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _slowa(s: str) -> list[str]:
    return re.findall(r"[0-9]+|[a-z]+", _ascii(s))


# ---------------------------------------------------------------- klucz

def obslugiwane(z: dict) -> bool:
    """Zadania oceniane automatem: zamknięte + krótkie otwarte „Podaj…" (≤40 znaków klucza, 1 pkt)."""
    if z.get("esej"):
        return False
    if z.get("typ") == "zamkniete":
        return parsuj_klucz(z) is not None
    return (len(z["rozwiazanie"]) <= 40 and z["pkt_max"] == 1 and re.search(r"\bPodaj\b", z["polecenie"]) is not None
            and parsuj_klucz(z) is not None)


def _pary(tekst: str) -> list[tuple[str, str]] | None:
    """'1 – F\\n2 – P' / '1. P; 2. F' / 'A – franciszkanie' / 'Fragment A – Karol IX' → [(etykieta, wartość)]."""
    czesci = [c.strip() for c in re.split(r"[\n;]|,\s*(?=(?:\d+\.?|[A-Z])\s*(?:[–—\-.]|$))", tekst) if c.strip()]
    out = []
    for c in czesci:
        m = re.match(r"^(?:fragment\s+|zdanie\s+)?(\d+|[A-Z])\.?\s*(?:[–—\-:]\s*)?(.+?)\.?$", c, re.I)
        if not m or not re.match(r"^(?:fragment\s+|zdanie\s+)?(\d+|[A-Z])\.?\s*[–—\-:.\s]", c + " ", re.I):
            m2 = re.match(r"^([A-ZŁŚŻŹĆ][a-ząćęłńóśźż]+)\s*[–—\-:]\s*(.+?)\.?$", c)  # „Wystawca – Jan"
            if not m2:
                return None
            out.append((m2.group(1).lower(), m2.group(2).strip()))
            continue
        out.append((m.group(1).upper(), m.group(2).strip()))
    return out or None


def _rodzaj(wartosci: list[str]) -> str:
    if all(v in ("P", "F") for v in wartosci):
        return "pf"
    if all(re.fullmatch(r"[A-F]", v) for v in wartosci):
        return "litera"
    if all(re.fullmatch(r"\d+", v) for v in wartosci):
        return "cyfra"
    return "tekst"


def parsuj_klucz(z: dict) -> dict | None:
    r = _czysc(z["rozwiazanie"])
    uporzadkuj = re.search(r"Uporządkuj", z["polecenie"], re.I) is not None
    if re.fullmatch(r"[A-F]", r):
        return {"rodzaj": "jedna", "litera": r}
    if re.fullmatch(r"[PF]{2,}", r):
        return {"rodzaj": "pary", "typ": "pf", "pary": {str(i + 1): c for i, c in enumerate(r)}}
    if re.fullmatch(r"[A-F](?:\s*,\s*[A-F])+", r):
        return {"rodzaj": "kolejnosc", "ciag": re.findall(r"[A-F]", r)}
    pary = _pary(r)
    if pary:
        wart = [v for _, v in pary]
        typ = _rodzaj(wart)
        if typ == "cyfra" and uporzadkuj and all(re.fullmatch(r"[A-F]", e) for e, _ in pary):
            return {"rodzaj": "kolejnosc", "ciag": [e for e, _ in sorted(pary, key=lambda p: int(p[1]))]}
        if typ != "tekst" or len(pary) > 1 or re.fullmatch(r"(?:\d+|[A-Z])", pary[0][0]):
            return {"rodzaj": "pary", "typ": typ, "pary": dict(pary)}
    if z.get("typ") != "zamkniete":
        return {"rodzaj": "tekst", "pozycje": _pozycje_tekst(r)}
    return None


def _pozycje_tekst(r: str) -> list[list[list[str]]]:
    """'konsul\\ntrybun ludowy' → 2 pozycje wymagane; 'Zygmunt III [Waza], Zygmunt Waza' → 1 pozycja, 2 warianty.
    Wariant = lista wymaganych słów (bez treści w nawiasach kwadratowych i słów ogólnych)."""
    pozycje = []
    for poz in re.split(r"[\n;]", r):
        poz = re.sub(r"^\s*[A-ZŁŚŻ][a-ząćęłńóśźż]+\s*[–—\-:]\s*", "", poz)  # „Wystawca – Jan"
        warianty = []
        for w in re.split(r"\s*/\s*|,\s*", poz):
            slowa = [s for s in _slowa(re.sub(r"\[[^\]]*\]", " ", w)) if s not in STOP]
            if slowa:
                warianty.append(slowa)
        if warianty:
            pozycje.append(warianty)
    return pozycje


# ---------------------------------------------------------------- odpowiedź

def _pasuje_slowo(wzor: str, slowa: list[str]) -> bool:
    if wzor.isdigit():
        return wzor in slowa
    n = min(len(wzor), max(4, -(-len(wzor) * 3 // 4)))  # rdzeń: ~75% słowa, min. 4 znaki (odmiana)
    return any(s[:n] == wzor[:n] and len(s) >= n for s in slowa)


def _ocen_tekst(pozycje: list, odp: str) -> tuple[bool | None, str]:
    """True = pasuje, None = nierozstrzygnięte (synonimy, np. „Zakon Braci Mniejszych" = franciszkanie → sędzia)."""
    slowa = _slowa(odp[:300])
    if pozycje and all(any(all(_pasuje_slowo(s, slowa) for s in war) for war in poz) for poz in pozycje):
        return True, "dopasowanie"
    return None, "tekst niepewny"


def _jedna_litera(odp: str) -> str | None:
    s = _czysc(odp)
    m = re.fullmatch(r"\(?([A-F])\)?[\.\)]?", s)
    if m:
        return m.group(1)
    pierwsza = s.splitlines()[0].strip() if s else ""
    m = re.match(r"^(?:(?:odpowiedź|dokończ\w* zdanie|dokończono)(?:\s+(?:do\s+)?(?:zadani[ae]\s+)?\d+(?:\.\d+)?\.?)?\s*:?\s*)?"
                 r"\(?([A-F])(?:[\.\)]|\s*[–—\-:]|$)(?:\s|$)", pierwsza, re.I)
    if m and m.group(1).isupper():
        return m.group(1)
    wsk = {m.group(1) for m in re.finditer(
        r"(?:odpowied\w*|prawidłow\w*|poprawn\w*)(?:\s+(?:odpowiedź|jest|to))*\s*:?\s*\(?([A-F])\b(?![a-ząćęłńóśźż])", s, re.I)
        if m.group(1).isupper()}
    if len(wsk) == 1:  # kilka różnych liter („odpowiedź: A … odpowiedź: D") = niejednoznaczne → sędzia
        return wsk.pop()
    return None


def _wartosc(reszta: str, typ: str) -> str | None:
    reszta = reszta.strip()
    if typ == "pf":
        if re.match(r"^\(?P\s*(?:/|lub|albo|\s)\s*F\b", reszta):
            # przepisany szablon „P / F": decyzja bywa niżej („**P** – prawdziwa", „Odpowiedź: F", „Fałszywa")
            dalej = re.sub(r"^\(?P\s*(?:/|lub|albo|\s)\s*F\b\)?", "", reszta)
            dec = {PF_SLOWA[m.group(1).lower()] if len(m.group(1)) > 1 else m.group(1) for m in re.finditer(
                r"(?:^|\n|(?i:odpowiedź)\s*:?\s*|[–—\-]\s*)\(?(P|F|[Pp]rawdziw[aey]|[Ff]ałszyw[aey])\b", dalej)}
            return dec.pop() if len(dec) == 1 else None
        m = re.match(r"^\(?([A-Za-ząćęłńóśźż]+)", reszta)
        if m and (m.group(1) in ("P", "F") or m.group(1).lower() in PF_SLOWA and len(m.group(1)) > 1):
            return PF_SLOWA[m.group(1).lower()]
        return None
    if typ == "litera":
        m = re.match(r"^\(?([A-F])(?![a-ząćęłńóśźż])", reszta)
        return m.group(1) if m else None
    if typ == "cyfra":
        m = re.match(r"^(\d+)\b", reszta)
        return m.group(1) if m else None
    return reszta or None


def _linie_etykiet(odp: str, etykiety: list[str]) -> dict[str, str]:
    """Etykieta → reszta pierwszej linii, która się od niej zaczyna, plus kolejne linie do następnej etykiety."""
    out, biezaca = {}, None
    kawalki = [k for l in _czysc(odp).splitlines()
               for k in re.split(r"[;,]\s*(?=(?:\d+|[A-Z])\s*[\.\)–—\-:])", l)]
    for linia in kawalki:
        linia = linia.strip().lstrip("-•").strip()
        trafiona = None
        for e in etykiety:
            if e in out:
                continue
            wz = r"^(?:(?i:fragment|zdanie|nr)\s+)?" + re.escape(e) + r"(?:\s*[\.\)]\s*|\s*" + KRESKA + r"\s*|\s+)(.*)$"
            m = re.match(wz, linia, re.I if len(e) > 1 else 0)
            if m:
                out[e], trafiona = m.group(1), e
                break
        if trafiona:
            biezaca = trafiona
        elif biezaca and linia:
            out[biezaca] += "\n" + linia
    return out


def _pary_odp(klucz: dict, odp: str) -> tuple[dict | None, str]:
    etykiety = list(klucz["pary"])
    typ = klucz["typ"]
    if typ == "pf" and re.fullmatch(r"(?:\s*(?:\d+\s*[\.\)]?\s*(?:[–—\-:]\s*)?)?P\s*/?\s*F\s*)+", _czysc(odp)):
        return {e: "PF" for e in etykiety}, "przepisany szablon P F"  # sam szablon, bez żadnej decyzji → 0
    linie = _linie_etykiet(odp, etykiety)
    got = {e: _wartosc(linie[e], typ) for e in etykiety if e in linie}
    if all(got.get(e) for e in etykiety):
        return got, "linie z etykietami"
    if typ == "pf":
        m = re.search(r"(?<![A-Za-z])([PF]{%d})(?![A-Za-z])" % len(etykiety), _czysc(odp))
        if m:
            return dict(zip(etykiety, m.group(1))), "zwarty ciąg P/F"
        linie_s = [l.strip() for l in _czysc(odp).splitlines() if l.strip()]
        if len(linie_s) == len(etykiety) and all(l in ("P", "F") for l in linie_s):
            return dict(zip(etykiety, linie_s)), "kolejne linie P/F"
    return None, "nie odczytano wszystkich etykiet"


def _kolejnosc_odp(n_litery: list[str], odp: str) -> tuple[list[str] | None, str]:
    s = _czysc(odp)
    litery = sorted(n_litery)
    # a) „C – 1, B – 2"
    pary = dict(re.findall(r"(?<![A-Za-z])([A-F])\s*(?:[\.\)]\s*)?" + KRESKA + r"\s*(\d)\b", s))
    if set(pary) == set(litery) and sorted(pary.values()) == [str(i + 1) for i in range(len(litery))]:
        return [e for e, _ in sorted(pary.items(), key=lambda p: int(p[1]))], "litera–pozycja"
    linie = [l.strip() for l in s.splitlines() if l.strip()]
    # b) „1. C" / „1. D. Kapitulacja…"
    poz = {}
    for l in linie:
        m = re.match(r"^(\d)[\.\)]?\s*(?:" + KRESKA + r"\s*)?\(?([A-F])(?:[\.\)]|\s|$)", l)
        if m and m.group(1) not in poz:
            poz[m.group(1)] = m.group(2)
    if sorted(poz) == [str(i + 1) for i in range(len(litery))] and sorted(poz.values()) == litery:
        return [poz[str(i + 1)] for i in range(len(litery))], "pozycja–litera"
    # c) kolejne linie „C. Atak…"
    ciag = [m.group(1) for l in linie if (m := re.match(r"^\(?([A-F])[\.\)]\s", l + " "))]
    if sorted(ciag) == litery:
        return ciag, "kolejne linie z literą"
    # d) „C, B, A, D" / „CBAD" / „C – B – A – D"
    m = re.search(r"(?<![A-Za-z])([A-F](?:\s*(?:,|→|->|–|-|\s)\s*[A-F]){%d})(?![A-Za-z])" % (len(litery) - 1), s)
    if m:
        ciag = re.findall(r"[A-F]", m.group(1))
        if sorted(ciag) == litery:
            return ciag, "zwarty ciąg liter"
    m = re.search(r"(?<![A-Za-z])([A-F]{%d})(?![A-Za-z])" % len(litery), s)
    if m and sorted(m.group(1)) == litery:
        return list(m.group(1)), "zwarty ciąg liter"
    return None, "nie odczytano kolejności"


def ocen(z: dict, odp: str) -> dict:
    """→ {pkt, pkt_max, pewne, powod}. pewne=False: do sędziego LLM (pkt=None)."""
    pm = z["pkt_max"]
    k = parsuj_klucz(z)
    wynik = lambda pkt, powod: {"pkt": pkt, "pkt_max": pm, "pewne": pkt is not None, "powod": powod}  # noqa: E731
    if k is None:
        return wynik(None, "klucz nieobsługiwany")
    if not odp or not odp.strip():
        return wynik(0, "pusta odpowiedź")
    if odp.lstrip().startswith("(BŁĄD"):
        return wynik(0, "błąd serwera")
    if k["rodzaj"] == "jedna":
        lit = _jedna_litera(odp)
        return wynik(None, "nie odczytano litery") if lit is None else wynik(pm if lit == k["litera"] else 0, f"litera {lit}")
    if k["rodzaj"] == "kolejnosc":
        ciag, powod = _kolejnosc_odp(k["ciag"], odp)
        return wynik(None, powod) if ciag is None else wynik(pm if ciag == k["ciag"] else 0, powod)
    if k["rodzaj"] == "tekst":
        ok, powod = _ocen_tekst(k["pozycje"], odp)
        return wynik(None if ok is None else (pm if ok else 0), powod)
    got, powod = _pary_odp(k, odp)
    if got is None:
        return wynik(None, powod)
    bledne = 0
    for e, v in k["pary"].items():
        if k["typ"] == "tekst":
            ok, _ = _ocen_tekst(_pozycje_tekst(v), got[e])
            if ok is None:
                return wynik(None, f"tekst niepewny ({e})")
            bledne += not ok
        else:
            bledne += got[e] != v
    return wynik(max(0, pm - bledne), powod)
