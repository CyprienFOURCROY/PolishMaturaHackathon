"""Router bez modelu (0 MB): typ zadania i epoka. Używany przez harness do wyboru eksperta (hipotezy H1, H2)."""
from __future__ import annotations

import re

from .devset import typ_zadania

EPOKI = ("starozytnosc-sredniowiecze", "nowozytnosc", "xix", "xx-xxi")
RE_ROK = re.compile(r"\b(1[0-9]{3}|[1-9][0-9]{2})\b(?!\s*p\.\s*n\.\s*e)")
RE_WIEK = re.compile(r"\b(X{0,2}(?:IX|IV|V?I{0,3}))\s*w(?:iek|\.)", re.I)
SLOWA = {
    "starozytnosc-sredniowiecze": "p.n.e starożyt egipt mezopotam grec ateny sparta rzym republik cesarstwo bizancj karol wielki "
                                  "piast chrobry mieszko krzyż rycer feudal lenn zakon krzyżac kazimierz wielki jagiełło krewo grunwald",
    "nowozytnosc": "odkryc geograf reformacj luter kalwin kontrreformac renesans barok szlachec sejm walny unia lubelska "
                   "wolna elekcja wazowie potop sasi oświecen stanisław august rozbior konstytucja 3 maja kościuszk",
    "xix": "napoleon kongres wiedeń powstanie listopadowe styczniowe zabór praca organiczna pozytywizm romantyzm wiosna ludów "
           "rewolucja przemysłowa kolej belle époque bismarck zjednoczenie niemiec",
    "xx-xxi": "wojna światowa piłsudski ii rzeczpospolita dwudziestolecie hitler stalin nazizm komunizm holokaust okupac "
              "powstanie warszawskie prl solidarność zimna wojna 1956 1968 1970 1980 1989 unia europejska nato",
}


def epoka(tekst: str) -> str:
    """Epoka po latach (najpewniejsze), potem po wiekach, potem po słowach kluczowych."""
    lata = [int(r) for r in RE_ROK.findall(tekst)]
    if "p.n.e" in tekst or "p. n. e" in tekst:
        return EPOKI[0]
    if lata:
        r = sorted(lata)[len(lata) // 2]
        return EPOKI[0] if r < 1492 else EPOKI[1] if r < 1815 else EPOKI[2] if r < 1918 else EPOKI[3]
    t = tekst.lower()
    wyniki = {e: sum(t.count(w) for w in s.split()) for e, s in SLOWA.items()}
    naj = max(wyniki, key=wyniki.get)
    return naj if wyniki[naj] > 0 else EPOKI[3]


def epoka_pewna(tekst: str) -> str | None:
    """Jak `epoka`, ale None, gdy tekst nie ma żadnego sygnału (lat, wieków, słów kluczowych) — bez domyślnej XX-XXI."""
    t = tekst.lower()
    if RE_ROK.search(tekst) or "p.n.e" in tekst or "p. n. e" in tekst or RE_WIEK.search(tekst):
        return epoka(tekst)
    if any(t.count(w) for s in SLOWA.values() for w in s.split()):
        return epoka(tekst)
    return None


def epoka_roku(rok: int) -> str:
    return EPOKI[0] if rok < 1492 else EPOKI[1] if rok < 1815 else EPOKI[2] if rok < 1918 else EPOKI[3]


def klucz_eksperta(z: dict, routing: str) -> str:
    """routing: 'typ' → zamkniete/otwarte/rozstrzygnij/esej; 'epoka' → jedna z EPOKI; 'jeden' → 'wszystko'."""
    if routing == "typ":
        if z.get("esej"):
            return "esej"
        t = typ_zadania(z)
        if t == "otwarte" and re.search(r"Rozstrzygnij", z["polecenie"], re.I):
            return "rozstrzygnij"
        return t
    if routing == "epoka":
        return epoka(z.get("zrodla_tekst", "") + " " + z["polecenie"])
    return "wszystko"
