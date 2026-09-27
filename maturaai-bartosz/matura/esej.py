"""Wypracowanie (wypowiedź argumentacyjna, 15 pkt) — szablon trzymany przez kod, treść od modelu.

Rubryka CKE (data/zasady/ZASADY-OCENIANIA-HISTORIA.md): A 0–12 = trzy elementy/aspekty tematu × (4 bogata /
3 zadowalająca / 1 powierzchowna), odjęcia za błędy merytoryczne (chronologia, terminologia, przyczyny–skutki);
B 0–3 = spójność, warunek ≥300 słów; 0 za całość przy braku stanowiska lub argumentacji sprzecznej ze stanowiskiem.
Stąd projekt: stanowisko = zmienna wstawiana przez kod do wstępu i zakończenia; każdy aspekt = osobny akapit;
fakty z kontekstu (Wikipedia / karty faktów); weryfikator usuwa zdania z datami spoza kontekstu.

Warianty (konfiguracje harnessu): e0 swobodny, e1 szablon, e2 = e1 + BM25 per aspekt, e3 = e1 + karty faktów
i słownik pojęć, e4 = e3 + weryfikator dat, e3_en = e3 z planowaniem po angielsku.
e5 (`napisz_e5`, sekcja na końcu pliku): stanowisko KATEGORYCZNE (domyślnie „nie zgadzam się”), kierunek akapitu
wprost w prompcie, kontrola akapitu w kodzie (długość, listy, powtórzenia, sprzeczność kierunku) z regeneracją,
weryfikator dat podmieniający albo usuwający samą datę; `wybierz_temat` wybiera 1 z 3 tematów wg pokrycia materiałem
(regula="powszechna": najpierw tematy bez podmiotu z historii Polski).
"""
from __future__ import annotations

import html
import re
from collections import Counter

from .llm import czat
from .retrieval import tokeny

RE_TEMATY = re.compile(r"(?ms)^\s*([1-3])\.\s+(.*?)(?=^\s*[1-3]\.\s+[A-ZŁŚŻŹĆ]|\Z)")
RE_ROK = re.compile(r"\b(1[0-9]{3}|[1-9][0-9]{2})\b")


def _czysc_temat(t: str) -> str:
    t = re.split(r"WYPRACOWANIE\s+na\s+temat|HISTORIA\s+Poziom\s+rozszerzony", t)[0]
    t = re.sub(r"-\s+-?\s*", "-", " ".join(t.split()))
    return t.strip()


def tematy(polecenie: str) -> list[str]:
    """Zadanie esejowe CKE (3 tematy) → lista oczyszczonych tekstów tematów."""
    return [_czysc_temat(t) for _, t in RE_TEMATY.findall(polecenie)]


def _mianownik(przym: str) -> str:
    """politycznych/polityczną → polityczny (przymiotnik aspektu w mianowniku l.p. m.)."""
    for kon, nowa in (("ych", "y"), ("ich", "i"), ("ą", "y")):
        if przym.endswith(kon):
            return przym[: -len(kon)] + nowa
    return przym


def miejscownik(aspekt: str) -> str:
    """polityczny → politycznym (do „w aspekcie …")."""
    if aspekt.endswith("y") or aspekt.endswith("i"):
        return aspekt + "m"
    return aspekt


def rozbierz(temat: str) -> tuple[str, list[str], str | None]:
    """Temat → (teza, trzy aspekty, wybor). wybor != None, gdy temat każe wybrać trzy elementy
    (np. „trzech wybranych władców z tego okresu") — wtedy elementy wskazuje model w kroku planu."""
    temat = _czysc_temat(temat)
    teza = temat.split("Zajmij stanowisko")[0].strip()
    polecenie = temat[len(teza):]
    m = re.search(r"(?:trzech|trzy) (?:wybran\w+|różn\w+)\s+(.+?)\.?$", polecenie)
    if m:
        return teza, [], m.group(1).strip()
    m = re.search(r"aspekt\w*\s*:?\s*(.+?)\.?$", polecenie)
    if m:
        czesci = [c.strip(" ,.") for c in re.split(r",\s*|\s+i\s+|\s+oraz\s+", m.group(1)) if c.strip(" ,.")]
    else:
        m = re.search(r"([\w-]+(?:ych|ą)),\s*([\w-]+(?:ych|ą))\s+(?:i|oraz)\s+([\w-]+(?:ych|ą))\s+\w+", polecenie)
        czesci = [_mianownik(x) for x in m.groups()] if m else []
    if len(czesci) != 3:
        czesci = ["polityczny", "społeczno-gospodarczy", "kulturowy"]
    return teza, czesci, None


STANOWISKA = {
    "czesciowo": ("Uważam, że teza ta jest trafna tylko częściowo.", "teza jest słuszna jedynie częściowo"),
    "tak": ("Zgadzam się z tą tezą.", "teza jest słuszna"),
    "nie": ("Nie zgadzam się z tą tezą.", "teza nie jest słuszna"),
}

SYS_AKAPIT = """Piszesz jeden akapit wypracowania maturalnego z historii, po polsku, 5–6 zdań.
Akapit dotyczy WYŁĄCZNIE wskazanego aspektu tematu. Każde zdanie zawiera konkret: datę, postać, wydarzenie albo pojęcie
historyczne, i łączy go z tezą. Używaj tylko faktów z podanego materiału albo takich, których jesteś pewien.
Nie pisz wstępu ani zakończenia całej pracy. Nie zmieniaj stanowiska: {stanowisko}."""
SYS_AKAPIT_EN = "\nFirst, under PLAN, list in English the 3–4 facts you will use. Then under AKAPIT write the paragraph in Polish."


def _material(wiki, karty, teza: str, aspekt: str, tryb: str) -> str:
    kaw = []
    if karty is not None and tryb in ("e3", "e4", "e3_en"):
        from .router import epoka_pewna
        k = karty.szukaj(f"{teza} {aspekt}", 8, epoka=epoka_pewna(teza))
        if k:
            kaw.append("Fakty:\n" + "\n".join(f"- {c['fakt']}" + (f" ({c['data']})" if c.get("data") else "") for c in k))
            pojecia = sorted({c["termin"] for c in k if c.get("termin")})
            if pojecia:
                kaw.append("Pojęcia do użycia: " + ", ".join(pojecia))
    if wiki is not None and tryb in ("e2", "e3", "e4", "e3_en"):
        frs = wiki.szukaj(f"{teza} {aspekt}", 3)
        if frs:
            kaw.append("Fragmenty encyklopedii:\n" + "\n".join(f"- [{f['tytul']}] {f['tekst']}" for f in frs))
    return "\n\n".join(kaw)


def prompt_akapitu(teza: str, aspekt: str, material: str, kon_st: str, tryb: str = "e3") -> tuple[str, str]:
    """(system, user) dla jednego akapitu; ten sam format w harnessie i w danych eksperta akapitu (matura/akapity.py)."""
    sys_ = SYS_AKAPIT.format(stanowisko=kon_st) + (SYS_AKAPIT_EN if tryb == "e3_en" else "")
    user = f"Teza: {teza}\nAspekt: {aspekt}\n\n" + (f"MATERIAŁ:\n{material}\n\n" if material else "") + "Napisz akapit."
    return sys_, user


def _weryfikuj(akapit: str, material: str) -> str:
    """Usuwa zdania z rokiem, którego nie ma w materiale (odjęcia za błędy chronologiczne)."""
    lata = set(RE_ROK.findall(material))
    zdania = re.split(r"(?<=[.!?])\s+", akapit)
    zost = [z for z in zdania if all(r in lata for r in RE_ROK.findall(z))]
    return " ".join(zost) if len(zost) >= 3 else akapit


def _po_akapicie(t: str) -> str:
    m = re.search(r"AKAPIT\s*:?\s*(.*)", t, re.S)
    return (m.group(1) if m else t).strip()


def napisz(z: dict, tryb: str, *, url: str, wiki=None, karty=None, bez_myslenia: bool = True,
           stanowisko: str = "czesciowo", czat_fn=czat) -> tuple[str, float, dict]:
    """Zwraca (wypracowanie, sekundy, meta). z = zadanie esejowe (jeden temat w polu 'temat')."""
    temat = z.get("temat") or (tematy(z["polecenie"]) or [z["polecenie"]])[0]
    teza, aspekty, wybor = rozbierz(temat)
    wstep_st, kon_st = STANOWISKA[stanowisko]
    sek = 0.0
    if wybor:  # krok planu: model wskazuje trzy elementy (władców / rewolucje / postaci)
        mat = _material(wiki, None, teza, wybor, "e2")
        t, s = czat_fn(url, "Odpowiadasz krótko, po polsku.", (f"MATERIAŁ:\n{mat}\n\n" if mat else "")
                       + f"Temat wypracowania: {temat}\nWymień trzech {wybor} najlepiej pasujących do tego tematu. "
                         "Podaj tylko nazwy/imiona, oddzielone przecinkami.", None, max_tokens=80, bez_myslenia=bez_myslenia)
        sek += s
        aspekty = [x.strip(" .\n-•1234567890)") for x in re.split(r",|\n", t)]
        aspekty = [x for x in aspekty if x and len(x.split()) <= 6][:3]
        while len(aspekty) < 3:
            aspekty.append(f"kolejny przykład ({len(aspekty) + 1})")
    akapity = []
    for i, a in enumerate(aspekty):
        mat = _material(wiki, karty, teza, a, tryb)
        sys_, user = prompt_akapitu(teza, a, mat, kon_st, tryb)
        t, s = czat_fn(url, sys_, user, None, max_tokens=500, bez_myslenia=bez_myslenia)
        sek += s
        t = _po_akapicie(t) if tryb == "e3_en" else t.strip()
        if tryb == "e4":
            t = _weryfikuj(t, mat)
        otw = ["Po pierwsze", "Po drugie", "Po trzecie"][i]
        glowa = f"{otw}, {a}." if wybor else f"{otw}, w aspekcie {miejscownik(a)}:"
        akapity.append(f"{glowa} {t}\nTen {'przykład' if wybor else 'aspekt'} pokazuje, że {kon_st}.")
    for _ in range(3):
        if sum(len(x.split()) for x in akapity) >= 330:
            break
        i = min(range(3), key=lambda k: len(akapity[k].split()))
        t, s = czat_fn(url, SYS_AKAPIT.format(stanowisko=kon_st),
                       f"Teza: {teza}\nAspekt: {aspekty[i]}\n\nDotychczasowy akapit:\n{akapity[i]}\n\n"
                       "Dopisz 3 kolejne zdania z nowymi konkretami (bez powtórzeń). Podaj tylko nowe zdania.",
                       None, max_tokens=300, bez_myslenia=bez_myslenia)
        sek += s
        glowa, _, ogon = akapity[i].rpartition("\n")
        akapity[i] = f"{glowa} {t.strip()}\n{ogon}"
    lista = f"{aspekty[0]}, {aspekty[1]} oraz {aspekty[2]}"
    if wybor:
        zapowiedz = f"Uzasadnię to na trzech przykładach: {lista}."
        wniosek = f"Podsumowując, przykłady: {lista} prowadzą do wniosku, że {kon_st}."
    else:
        zapowiedz = f"Uzasadnię to, analizując kolejno aspekt {lista}."
        wniosek = f"Podsumowując, analiza aspektów: {lista} prowadzi do wniosku, że {kon_st}."
    wstep = f"„{teza}” — to teza, wobec której należy zająć stanowisko. {wstep_st} {zapowiedz}"
    zak = f"{wniosek} Przedstawione argumenty potwierdzają stanowisko zajęte we wstępie."
    tekst = "\n\n".join([wstep, *akapity, zak])
    return tekst, sek, {"slow": len(tekst.split()), "aspekty": aspekty, "stanowisko": stanowisko}


# ================================================================ e5: stanowisko kategoryczne, kontrola akapitu w kodzie
# Diagnoza (pełna matura 26.09): e1 „częściowo” + akapity chwalące tezę = sprzeczność stanowiska (CKE: 0 pkt albo
# mocne obniżenie); zmyślone daty zerują kryterium A; powtórzenia i wyliczanki dają B = 0-1. Stąd w e5:
# stanowisko kategoryczne w szablonie, kierunek każdego akapitu wprost w prompcie, a w kodzie kontrola akapitu
# (długość, listy, powtórzenia, sprzeczność kierunku) z regeneracją i weryfikator dat (podmiana albo usunięcie daty).

STANOWISKA_E5 = ("nie_zgadzam", "zgadzam")
PORZADKOWE = ("Po pierwsze", "Po drugie", "Po trzecie")
SLOWA_AKAPITU = (110, 150)      # cel długości akapitu modelu (bez nagłówka i zdania wiążącego z szablonu)
MIN_SLOW_AKAPITU = 90           # poniżej: regeneracja
MAX_SLOW_AKAPITU = 150          # powyżej: przycięcie całych zdań od końca
MIN_SLOW_ESEJU = 300            # warunek CKE (kryterium B)
MAX_TOKENOW_AKAPITU = 360

# Słowa odwracające sens zdania (liczy się parzystość: „nie jest niesłusznie” = twierdzenie).
NEGACJE = {"nie", "ani", "nigdy", "wcale", "bynajmniej", "żaden", "żadna", "żadne", "żadnego", "żadnej", "żadnych",
           "niesłusznie", "nieprawdą", "nieprawda"}
_STEMY_NEGACJI = {w[:6] for w in NEGACJE}
# Tokeny zbyt ogólne, żeby świadczyły o trafności karty dla tematu (wybierz_temat).
_OGOLNE = {"polski", "polska", "polsce", "polaków", "państw", "wieku", "wiek", "okresi", "okres", "latach", "czasie",
           "dzieja", "dziejó", "histor", "najważ", "wielki", "wielka"}
_SKROTY = {"tzw", "m.in", "np", "ok", "św", "gen", "ks", "im", "zob", "prof", "dr", "tj", "wg", "ang", "łac", "gr",
           "niem", "franc", "ros", "płk", "mjr", "kpt", "hr", "bp", "abp", "kard", "cz", "pt", "ul", "pw"}
# Nazwy własne zbyt ogólne, żeby wskazać kartę, o której mówi zdanie (państwa, regiony): prefiksy rdzeni.
_OGOLNE_WLASNE = ("pols", "litw", "czech", "węgr", "europ", "rosj", "niem", "fran", "angl", "prus", "austr", "szwe",
                  "turc", "środk", "wsch", "zach", "rzecz", "koron", "krakó", "krako", "warsz")
_ASPEKT_ALIASY = {"dyplom": {"między", "polity"}, "wojsko": {"milita"}, "narodo": {"polity", "społec"}}
# Pierwsze słowo tezy, które zostaje z wielkiej litery po „że …” (nazwy własne i imiona postaci z tematów CKE).
_PROPRIA = set("""Polska Polacy Rzeczpospolita Litwa Europa Francja Rosja Niemcy Anglia Austria Prusy Węgry Czechy Rzym
Bizancjum Kościół Władysław Kazimierz Józef Bolesław Zygmunt Stanisław Stefan Jan Jadwiga Karol Mieszko Henryk Ludwik
Aleksander Piotr Katarzyna Fryderyk Otto Juliusz Oktawian Napoleon Adolf Tadeusz Roman Ignacy Wojciech Ferdynand Maria
Elżbieta Iwan Lenin Stalin Hitler Piłsudski Dmowski Chrobry Krzywousty Łokietek Batory Sobieski""".split())

RE_ZDANIE = re.compile(r"[.!?…]+[”\"»)]*\s+(?=[„\"«(]?[A-ZĄĆĘŁŃÓŚŹŻ])")
RE_LISTA = re.compile(r"^\s*(?:[-•*·▪►\u2013\u2014]|\d{1,2}[.)]|[a-h][.)])\s+")
RE_LINIA_DATY = re.compile(r"^\s*(\d{3,4}(?:\s*[-\u2013]\s*\d{2,4})?)\s*(?:r\.)?\s*[-\u2013\u2014:]\s+(.+)$")
RE_ETYKIETA_ZDANIA = re.compile(r"^[A-ZĄĆĘŁŃÓŚŹŻ][^:.!?]{0,50}:\s+(?=[„\"]?[A-ZĄĆĘŁŃÓŚŹŻ0-9])")
RE_WYLICZENIE = re.compile(r"^(?:Po (?:pierwsze|drugie|trzecie|czwarte|piąte)|Po pierwsze i najważniejsze),\s*")
RE_NAGLOWEK_ZDANIE = re.compile(r"^(?:Akapit|Analiza \w+ aspektu|Aspekt \w+\.$|Wnioski)\b", re.I)
RE_ECHO = re.compile(r"^\s*(?:teza|aspekt|materiał|notatki|fakty|temat|stanowisko)\b[^:]{0,40}:", re.I)
_MIES = ("stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia")
_ROK = r"(?:1\d{3}|20[0-2]\d|[1-9]\d{2})"
RE_DATA = re.compile(
    r"(?P<nawias>\(\s*(?:ok\.\s*)?)?"
    r"(?P<przyim>\b(?:w\s+latach|w\s+roku|na\s+przełomie|w|we|od|do|po|przed|około|ok\.|roku|latach|między|"
    r"pomiędzy|z|ze)\s+)?"
    rf"(?P<dzien>\d{{1,2}}\s+(?:{_MIES})\s+)?"
    rf"(?<![\d,.])(?P<rok>{_ROK})(?![\d%])"
    rf"(?P<zakres>\s*(?:[-\u2013\u2014]|do|a)\s*(?P<rok2>{_ROK}|\d{{2}})(?![\d%]))?"
    r"(?P<suf>\s*r\.|\s+roku\b|\s+rok\b)?"
    r"(?P<nawias2>\s*\))?", re.I)


def zdania(t: str) -> list[str]:
    """Podział tekstu na zdania: koniec zdania = [.!?] + spacja + wielka litera, z wyjątkiem skrótów (tzw., m.in.,
    św., gen. …) i inicjałów (J. Piłsudski). „r.” przed wielką literą kończy zdanie (w 1410 r. Wojska …)."""
    t = " ".join(t.split())
    out, start = [], 0
    for m in RE_ZDANIE.finditer(t):
        przed = re.search(r"([\w.]+)$", t[:m.start()])
        slowo = przed.group(1) if przed else ""
        if slowo.lower() in _SKROTY or (len(slowo) == 1 and slowo.isupper()):
            continue
        out.append(t[start:m.end()].strip())
        start = m.end()
    if t[start:].strip():
        out.append(t[start:].strip())
    return out


def _slowa(t: str) -> int:
    return len(t.split())


def _stemy(t: str) -> set[str]:
    return set(tokeny(t))


def _podobne(a: str, b: str) -> bool:
    """Zdania „te same”: identyczne po normalizacji albo z wysokim podobieństwem rdzeni (Jaccard ≥ 0,7 przy ≥ 4
    rdzeniach; albo krótsze zawarte w dłuższym w ≥ 85% przy ≥ 5 rdzeniach)."""
    na, nb = re.sub(r"\W+", " ", a.lower()).strip(), re.sub(r"\W+", " ", b.lower()).strip()
    if na == nb:
        return True
    sa, sb = _stemy(a), _stemy(b)
    if min(len(sa), len(sb)) < 4:
        return False
    wsp = len(sa & sb)
    return wsp / len(sa | sb) >= 0.7 or (min(len(sa), len(sb)) >= 5 and wsp / min(len(sa), len(sb)) >= 0.85)


def _bez_powtorzen_kart(karty: list[dict]) -> list[dict]:
    """Karty kanoniczne mają duplikaty z bliźniaczych działów (ta sama nazwa działu zapisana na dwa sposoby)."""
    out: list[dict] = []
    for c in karty:
        sc = _stemy(c["fakt"])
        if not any(len(sc & _stemy(o["fakt"])) / max(1, len(sc | _stemy(o["fakt"]))) >= 0.5 for o in out):
            out.append(c)
    return out


def _okres_tezy(teza: str) -> tuple[int, int] | None:
    """Lata w tezie → okres: jeden rok → (rok-1, rok+1), kilka → (min, max); bez lat → None. „Sprzed 1789 roku”
    też daje okres 1788-1790 (karty z okolic punktu odniesienia tezy)."""
    lata = sorted(_lata(teza))
    if not lata:
        return None
    return (lata[0] - 1, lata[0] + 1) if len(lata) == 1 else (lata[0], lata[-1])


def _karty_okresu(karty: list[dict], okres: tuple[int, int], teza: str, pozycja: str, element: bool,
                  uzyte: list[str], stanowisko: str, n: int = 9) -> list[dict]:
    """Karty z datą w okresie tezy, uszeregowane: pasujący aspekt (albo nazwa przykładu) i wspólne rdzenie z tezą."""
    od, do = okres
    st_asp, tz, st_poz = _stemy_aspektu(pozycja), _stemy(teza) - _OGOLNE, _stemy(pozycja)
    kand = []
    for c in karty:
        lata = _lata_karty(c)
        if not lata or lata[-1] < od or lata[0] > do or any(_podobne(c["fakt"], u) for u in uzyte):
            continue
        st_c = _stemy(f"{c['fakt']} {c.get('postac', '')} {c.get('tytul', '')}")
        if element:
            wynik = 3 * len(st_poz & st_c) + len(tz & st_c)
        else:
            wynik = 2 * ((c.get("aspekt") or "")[:6] in st_asp) + len(tz & st_c)
        if stanowisko == "zgadzam" and not (tz & st_c) and not (element and st_poz & st_c):
            wynik -= 1
        kand.append((wynik, c))
    kand.sort(key=lambda x: -x[0])
    return _bez_powtorzen_kart([c for _, c in kand])[:n]


def _stemy_aspektu(aspekt: str) -> set[str]:
    """społeczno-gospodarczy → {społec, gospod}; dyplomatyczny → {dyplom, między, polity} (pole `aspekt` kart)."""
    st = {p[:6] for p in aspekt.lower().split("-") if p}
    for k, v in _ASPEKT_ALIASY.items():
        if k in st:
            st |= v
    return st


def material_e5(wiki, karty, teza: str, pozycja: str, element: bool = False, n_wiki: int = 2,
                pomin: list[dict] | None = None, stanowisko: str = "nie_zgadzam", baza=None,
                pomin_bazy: set[str] | None = None) -> dict:
    """Materiał jednego akapitu: {tekst (do promptu), fakty (karty), fragmenty (Wikipedia)}.
    Karty: 3 najlepsze z BM25 (teza + aspekt, filtr epoki jak w e3, bez duplikatów) + do 6 kart z tego samego działu
    z pasującym polem `aspekt` (np. dla Jagiełły w aspekcie militarnym: Malbork, Warna, wojna trzynastoletnia;
    w ustrojowym: przywileje jedlneńsko-krakowskie, unia horodelska), żeby akapit „nie zgadzam się” miał fakty
    z kontekstu, a nie tylko fakty chwalące podmiot tezy. `pomin` = karty użyte w poprzednich akapitach (BM25 po
    tezie zwraca te same karty dla każdego aspektu, a te same fakty w trzech akapitach to powtórzenia).
    Teza z latami („Lata 1871-1914 …”, „Rok 1956 …”, „Klęska Polski w 1939 roku …”): karty z datą w okresie tezy
    (rok ± 1 albo zakres), z pasującym aspektem na początku; BM25 po słowach tezy trafia tu w przypadkowe karty
    („przełom” → „Świat na przełomie tysiącleci”, „1914” → Legiony). Mniej niż 3 takie karty → jak niżej.
    Dział rozszerzenia musi mieć epokę tezy (gdy znana). Przy „zgadzam” karty działu muszą mieć słowo z tezy (inaczej
    fakty o porażkach innych władców psują akapit potwierdzający).
    element=True (temat z wyborem trzech przykładów): 8 kart z zapytania z nazwą przykładu, bez rozszerzenia o dział.
    Wikipedia: n_wiki fragmentów (mniej niż e3, bo przy małych modelach nietrafny fragment rodzi zmyślenia)."""
    fakty: list[dict] = []
    okres = _okres_tezy(teza)
    if karty is not None and okres and getattr(karty, "k", None):
        fakty = _karty_okresu(karty.k, okres, teza, pozycja, element, [c["fakt"] for c in (pomin or [])], stanowisko)
    if karty is not None and len(fakty) < 3:
        from .router import epoka_pewna
        zap = f"{pozycja} {pozycja} {teza}" if element else f"{teza} {pozycja}"
        uzyte = [c["fakt"] for c in (pomin or [])]
        ep = epoka_pewna(teza)
        # n=12: Karty.szukaj stosuje filtr epoki tylko, gdy znajdzie ≥ n zgodnych kart (przy n=24 często bez filtra)
        kand = [c for c in _bez_powtorzen_kart(karty.szukaj(zap, 12, epoka=ep))
                if not any(_podobne(c["fakt"], u) for u in uzyte)]
        glowne = kand[: 8 if element else 3]
        fakty = list(glowne)
        if not element and kand and getattr(karty, "k", None):
            epoki = getattr(karty, "epoka_dzialu", {})
            dzialy = [d for d, _ in Counter(c.get("dzial") for c in kand[:6]).most_common()
                      if not ep or epoki.get(d, ep) == ep]
            st, tz = _stemy_aspektu(pozycja), _stemy(teza) - _OGOLNE
            dod = [c for c in karty.k if dzialy and c.get("dzial") == dzialy[0] and (c.get("aspekt") or "")[:6] in st
                   and not any(_podobne(c["fakt"], u) for u in uzyte)
                   and (stanowisko != "zgadzam" or tz & _stemy(c["fakt"]))]
            dod.sort(key=lambda c: -len(tz & _stemy(c["fakt"])))
            fakty = _bez_powtorzen_kart(glowne + dod)[:9]
    frs = wiki.szukaj(f"{teza} {pozycja}", n_wiki) if wiki is not None and n_wiki else []
    czesci, z_bazy = [], []
    if baza is not None and len(baza):   # e7: akapity bazy wiedzy jako materiał (fakty pewne), trafne i bez powtórzeń
        tz, okres = _stemy(teza) - _OGOLNE - _STEMY_NEGACJI, _okres_tekstu(teza)
        kier = "potwierdza" if stanowisko == "zgadzam" else "przeczy"
        kand = [c for c in baza.kandydaci(teza, pozycja, kierunek=kier, n=12, tryb="elementy" if element else None,
                                          pomin=frozenset(pomin_bazy or ())) if _pasuje_do_tezy(c, tz, okres)]
        z_bazy = kand[:2]
        if z_bazy:
            czesci.append("AKAPITY Z BAZY WIEDZY (fakty pewne; wykorzystaj je, ale pisz o tezie tematu): "
                          + " ".join(c["tekst"] for c in z_bazy))
    if fakty:
        czesci.append("FAKTY: " + " ".join(c["fakt"].rstrip(". ") + (f" ({c['data']})" if c.get("data") else "") + "."
                                             for c in fakty))
    if frs:
        czesci.append("ENCYKLOPEDIA (może być nietrafna): " + " ".join(f"[{f['tytul']}] {f['tekst']}" for f in frs))
    return {"tekst": "\n\n".join(czesci), "fakty": fakty, "fragmenty": frs, "z_bazy": [c["id"] for c in z_bazy]}


# ---------------------------------------------------------------- weryfikator dat

def _lata(tekst: str) -> set[int]:
    return {int(r) for r in re.findall(rf"(?<!\d){_ROK}(?!\d)", tekst)}


def _lata_karty(c: dict) -> tuple[int, ...]:
    return tuple(sorted(_lata(c.get("data") or "")))


def _wlasne(t: str) -> set[str]:
    """Rdzenie nazw własnych (słowa z wielkiej litery poza początkiem zdania), bez państw i regionów."""
    wyn = set()
    for zd in zdania(t):
        for w in re.findall(r"[\wąćęłńóśźż]+", zd)[1:]:
            if w[:1].isupper() and len(w) > 2 and not w.lower().startswith(_OGOLNE_WLASNE):
                wyn.add(w.lower()[:6])
    return wyn


def _rok_konca(rok: int, rok2: str) -> int:
    return int(rok2) if len(rok2) > 2 else rok // 100 * 100 + int(rok2)


def weryfikuj_daty(akapit: str, fakty: list[dict], tekst_materialu: str) -> tuple[str, list[str], list[str]]:
    """Data sprzeczna z materiałem → podmiana na datę z karty albo usunięcie SAMEJ daty (z przyimkiem), nigdy zdania.

    Dla każdej daty (rok, zakres lat, „15 lipca 1410 r.”, „(1386)”) w zdaniu:
    1. lata daty występują w materiale (daty i treść kart, Wikipedia, teza) → data zostaje;
    2. inaczej, gdy zdanie jednoznacznie mówi o jednej karcie z jednym rokiem, odległym o ≤ 10 lat (pomyłka
       o kilka lat, np. 1409 zamiast 1410) → podmiana na rok karty. „Jednoznacznie”: ≥ 4 wspólne rdzenie słów,
       ≥ 2 wspólne nazwy własne z karty inne niż państwa i regiony (np. „Grunwald”, „Jagiełło”) i przewaga ≥ 2
       rdzeni nad najlepszą kartą z inną datą;
    3. w pozostałych przypadkach → usunięcie samej daty.
    Próbka 26.09 (Bielik-1.5B): podmiana przy słabszym dopasowaniu (3 rdzenie, jedna nazwa) psuła poprawne daty
    (1437 → 1490, 1956 → 1989), stąd warunki ostrzejsze; usunięcie niepewnej daty nie jest błędem merytorycznym.
    Liczby trzycyfrowe bez „r.”/„roku”/nawiasu nie są traktowane jako daty (np. „300 rycerzy”).
    Zwraca (akapit, podmienione „stara→nowa”, usunięte)."""
    znane = _lata(tekst_materialu) | {r for c in fakty for r in _lata(f"{c.get('data') or ''} {c['fakt']}")}
    datowane = [(c, _lata_karty(c), _stemy(f"{c['fakt']} {c.get('postac', '')} {c.get('termin', '')} {c.get('tytul', '')}"),
                 _wlasne(f"{c['fakt']} {c.get('postac', '')}")) for c in fakty if _lata_karty(c)]
    podm, usun, wyn = [], [], []
    for zd in zdania(akapit):
        st_zd = {s for s in _stemy(zd) if not s.isdigit()}
        rank = sorted(((len(st_zd & sc), lata, len(st_zd & wl)) for c, lata, sc, wl in datowane), key=lambda x: -x[0])
        karta = None
        if rank and rank[0][0] >= 4 and rank[0][2] >= 2 and len(rank[0][1]) == 1:
            inne = [r[0] for r in rank[1:] if r[1] != rank[0][1]]
            if not inne or rank[0][0] >= inne[0] + 2:
                karta = rank[0]

        def zamien(m: re.Match) -> str:
            rok = int(m.group("rok"))
            trzycyfr = len(m.group("rok")) == 3
            if trzycyfr and not (m.group("suf") or m.group("nawias") or (m.group("przyim") or "").strip().lower()
                                 in ("w roku", "roku")):
                return m.group(0)
            lata = [rok] + ([_rok_konca(rok, m.group("rok2"))] if m.group("zakres") else [])
            if all(r in znane for r in lata):
                return m.group(0)
            if karta is not None and not m.group("zakres") and abs(karta[1][0] - rok) <= 10:
                s = m.start("rok") - m.start()
                podm.append(f"{rok}→{karta[1][0]}")
                return m.group(0)[:s] + str(karta[1][0]) + m.group(0)[s + len(m.group("rok")):]
            usun.append(m.group(0).strip())
            if m.group("nawias") and not m.group("nawias2"):
                return "("  # nawias otwarty przed datą, a zamknięty później: zostaw nawias
            konczy_zdanie = m.group(0).rstrip().endswith(".") and not m.string[m.end():].strip()
            return "." if konczy_zdanie else " "

        nowe = RE_DATA.sub(zamien, zd)
        if nowe != zd:
            nowe = re.sub(r"\(\s*\)", "", nowe)
            nowe = re.sub(r"\s+([,.;:)])", r"\1", re.sub(r"\s{2,}", " ", nowe)).strip()
            nowe = re.sub(r"\(\s+", "(", re.sub(r",\s*,", ",", re.sub(r"\.\.+", ".", nowe)))
            nowe = re.sub(r"^[,;:\s]+", "", nowe)
            nowe = nowe[:1].upper() + nowe[1:]
            if not re.search(r"[.!?…”\")]$", nowe):
                nowe += "."
        wyn.append(nowe)
    return " ".join(w for w in wyn if w), podm, usun


# ---------------------------------------------------------------- kontrola akapitu

def _sklej_liste(surowy: str) -> tuple[str, int]:
    """Wyjście modelu → jeden akapit ciągłego tekstu: bez markdownu, nagłówków i echa promptu; punkty listy
    sklejone w zdania („1386 - koronacja …” → „Koronacja … (1386).”); ucięte ostatnie zdanie (limit tokenów)
    odrzucone. Zwraca (tekst, liczba linii listy)."""
    t = re.sub(r"\*\*|__|`+|^#+\s*", "", html.unescape(surowy.strip()), flags=re.M)
    t = re.sub(r"&\w{1,6};?|\u00a0", " ", t)  # resztki encji („&nb;”) i twarde spacje
    linie = [l for l in t.splitlines() if l.strip()]
    if linie and len(linie[-1].split()) >= 8 and not re.search(r"[.!?…”\")]\s*$", linie[-1]):
        ost = max(linie[-1].rfind(". "), linie[-1].rfind("! "), linie[-1].rfind("? "))  # ucięte limitem tokenów
        linie[-1] = linie[-1][:ost + 1] if ost > 0 else ""
    czesci, n_lista = [], 0
    for linia in linie:
        l = linia.strip()
        if not l:
            continue
        if RE_ECHO.match(l):
            reszta = l.split(":", 1)[1].strip()
            if re.match(r"(?i)(teza|temat|stanowisko)", l) or len(reszta.split()) < 4:
                continue  # echo tezy / nagłówek „Fakty:” bez treści
            l = reszta
        l = re.sub(r"^(?:akapit|odpowiedź|tekst)\s*:\s*", "", l, flags=re.I)
        m = RE_LINIA_DATY.match(l)
        if m:
            n_lista += 1
            l = f"{m.group(2).rstrip(' .;')} ({m.group(1)})."
        elif RE_LISTA.match(l):
            n_lista += 1
            l = RE_LISTA.sub("", l)
            if len(l.split()) <= 5 and not re.search(r"[.!?…”\")]$", l):
                continue  # punkt-nagłówek („1. Stabilizacja polityczna i monarchiczna”)
            m = re.match(r"^([^:.!?]{1,60}):\s+(?=\S)", l)
            if m and len(m.group(1).split()) <= 6:  # etykieta punktu „Grunwald (1410): …”
                l = l[m.end():]
        if l.endswith(":") and len(l.split()) <= 10:
            continue  # nagłówek („Czynniki militarne, które przeczą tezie:”)
        if not RE_LISTA.match(linia) and not m and len(l.split()) <= 7 and not re.search(r"[.!?…”\")]$", l):
            continue  # krótka linia bez kropki = nagłówek („Rozwój gospodarczy i przemysłowy”)
        l = l.rstrip(" ;,:")
        if not l:
            continue
        l = l[:1].upper() + l[1:]
        if not re.search(r"[.!?…”\")]$", l):
            l += "."
        czesci.append(l)
    t = re.sub(r"\b[Ff]akt\s*\d*\s*:\s*", "", " ".join(czesci))       # echo materiału („Fakt 2: …”)
    t = re.sub(r"([.!?])\s*\((?:ok\.\s*)?\d{3,4}(?:\s*[-\u2013]\s*\d{2,4})?\)\.?", r"\1", t)  # „… wy”. (1871)”
    t = re.sub(r"(?<=[.!?…])\s+\d{1,2}[.)](?=\s|$)", "", t)                 # osierocony numer punktu („… strony. 6.”)
    zd = []
    for z in zdania(t):
        z = RE_ETYKIETA_ZDANIA.sub("", z)            # „Stabilizacja i rozwój: Belle époque …”, „Fakt: W 1956 …”
        z = RE_WYLICZENIE.sub("", z)                 # „Po drugie, …” w środku akapitu (szablon numeruje akapity)
        z = z[:1].upper() + z[1:]
        if len(z.split()) <= 3 or RE_NAGLOWEK_ZDANIE.match(z):
            continue                                  # fragmenty i nagłówki („W latach 50.”, „Akapit o aspekcie …”)
        zd.append(z)
    return " ".join(zd), n_lista


def _parzystosc_negacji(t: str) -> int:
    return sum(1 for w in re.findall(r"\w+", t.lower()) if w in NEGACJE) % 2


def _rdzen_tezy(teza: str) -> dict:
    """Rdzeń tezy do wykrywania sprzeczności kierunku: rdzenie słów (bez negacji), przymiotniki stopnia najwyższego
    („najwybitniejszym”; przy „najbardziej X” także X), podmiot = początkowy ciąg co najmniej dwóch słów z wielkiej
    litery („Władysław Jagiełło”), parzystość negacji w tezie („niesłusznie” = 1)."""
    slowa = re.findall(r"[\wąćęłńóśźż]+", teza)
    naj = set()
    for i, w in enumerate(slowa):
        if w.lower().startswith("naj") and len(w) > 5:
            naj.add(w.lower()[:7])
            if w.lower() in ("najbardziej", "najmniej") and i + 1 < len(slowa):
                naj.add(slowa[i + 1].lower()[:6])
    podmiot = []
    for w in slowa:
        if w[:1].isupper() and not w.isupper():
            podmiot.append(w)
        else:
            break
    return {"stemy": _stemy(teza) - _STEMY_NEGACJI, "naj": naj,
            "podmiot": {w.lower()[:6] for w in podmiot} if len(podmiot) >= 2 else set(),
            "neg": _parzystosc_negacji(teza)}


def sprzeczne_z_kierunkiem(zdanie: str, rdzen: dict, stanowisko: str) -> bool:
    """Heurystyka sprzeczności kierunku. Zdanie „mówi o tezie”, gdy zawiera ≥ 60% rdzeni tezy albo jej przymiotnik
    stopnia najwyższego (wraz ze słowem podmiotu tezy, a przy tezie bez podmiotu z ≥ 40% rdzeni). Zdanie „twierdzi
    tezę”, gdy przy tym parzystość negacji (nie, ani, nigdy, wcale, niesłusznie …) jest taka sama jak w tezie; przy
    stopniu najwyższym liczy się tylko negacja w 5 słowach przed nim.
    nie_zgadzam: sprzeczne = twierdzi tezę („Jagiełło był najwybitniejszym władcą”); „Jagiełło nie był
    najwybitniejszym” i „To Kazimierz Jagiellończyk był najwybitniejszy” przechodzą.
    zgadzam: sprzeczne = mówi o tezie z odwrotną parzystością negacji („Jagiełło nie był najwybitniejszym”)."""
    st = _stemy(zdanie)
    if not rdzen["stemy"]:
        return False
    pokr = len(st & rdzen["stemy"]) / len(rdzen["stemy"])
    slowa = [w.lower() for w in re.findall(r"\w+", zdanie)]
    naj = [i for i, w in enumerate(slowa) if w.startswith("naj") and (w[:7] in rdzen["naj"] or w[:6] in rdzen["naj"])]
    ma_naj = bool(naj) and (bool(st & rdzen["podmiot"]) if rdzen["podmiot"] else pokr >= 0.4)
    if ma_naj and rdzen["neg"] == 0:
        # stopień najwyższy: liczy się negacja tuż przed nim (5 słów), nie w całym zdaniu; „choć niewątpliwie
        # najwybitniejszy …, nie był …” twierdzi tezę, „nie można go uznać za najwybitniejszego” nie.
        twierdzi = any(not any(w in NEGACJE for w in slowa[max(0, i - 5):i]) for i in naj)
        return twierdzi if stanowisko == "nie_zgadzam" else not twierdzi
    if pokr < 0.6:
        return False
    zgodna = _parzystosc_negacji(zdanie) == rdzen["neg"]
    return zgodna if stanowisko == "nie_zgadzam" else not zgodna


def ocen_akapit(surowy: str, *, rdzen: dict, stanowisko: str, fakty: list[dict], tekst_materialu: str,
                poprzednie: list[str]) -> dict:
    """Obróbka i ocena jednej próby akapitu: sklejenie list → weryfikacja dat → usunięcie zdań powtórzonych
    (w akapicie i względem `poprzednie`, czyli zdań innych akapitów) → wykrycie sprzeczności kierunku.
    Zwraca {tekst, zdania, slowa, wady, kara, …}; wady ⊂ {krotki, lista, sprzecznosc, powtorzenia}; mniejsza kara
    = lepsza próba. Pusty materiał (brak kart i Wikipedii) → daty nieweryfikowane."""
    tekst, n_lista = _sklej_liste(surowy)
    podm, usun = [], []
    if fakty or tekst_materialu.strip():  # bez materiału nie ma z czym porównać dat
        tekst, podm, usun = weryfikuj_daty(tekst, fakty, tekst_materialu)
    zost, powt = [], 0
    for z in zdania(tekst):
        if any(_podobne(z, x) for x in zost + poprzednie):
            powt += 1
        else:
            zost.append(z)
    sprz = [z for z in zost if sprzeczne_z_kierunkiem(z, rdzen, stanowisko)]
    slowa = sum(_slowa(z) for z in zost)
    wady = []
    if slowa < MIN_SLOW_AKAPITU:
        wady.append("krotki")
    if n_lista >= 2:
        wady.append("lista")
    if sprz:
        wady.append("sprzecznosc")
    if powt >= 2:
        wady.append("powtorzenia")
    lo, hi = SLOWA_AKAPITU
    kara = ((100 + MIN_SLOW_AKAPITU - slowa) if slowa < MIN_SLOW_AKAPITU else 0) + 20 * (n_lista >= 2) \
        + 40 * len(sprz) + 8 * powt + 3 * len(usun) + 0.5 * max(0, lo - slowa) + 0.2 * max(0, slowa - hi)
    return {"tekst": " ".join(zost), "zdania": zost, "slowa": slowa, "wady": wady, "kara": kara,
            "sprzeczne": sprz, "powtorzenia": powt, "lista": n_lista, "podmienione": podm, "usuniete": usun}


def _wykoncz(ocena: dict) -> tuple[list[str], int]:
    """Wybrana próba → zdania akapitu: bez zdań sprzecznych z kierunkiem (gdy zostają ≥ 3 zdania), przycięta do
    MAX_SLOW_AKAPITU całymi zdaniami od końca. Zwraca (zdania, liczba usuniętych zdań sprzecznych)."""
    zd = ocena["zdania"]
    usun = 0
    if ocena["sprzeczne"] and len(zd) - len(ocena["sprzeczne"]) >= 3:
        zd = [z for z in zd if z not in ocena["sprzeczne"]]
        usun = len(ocena["sprzeczne"])
    while len(zd) > 3 and sum(_slowa(z) for z in zd) > MAX_SLOW_AKAPITU:
        zd = zd[:-1]
    return zd, usun


# ---------------------------------------------------------------- prompty i szablon

SYS_E5 = """Piszesz jeden akapit wypracowania maturalnego z historii, po polsku: 6-8 pełnych zdań (około 130 słów),
jednym ciągłym tekstem. Nie używaj list, punktów, myślników na początku zdań ani nagłówków.
{kierunek}
Używaj tylko faktów z podanego materiału. Najwyżej 2-3 daty w akapicie; lepiej bez daty niż z datą niepewną.
Każde zdanie wnosi nową informację i łączy fakt z oceną tezy; nie powtarzaj zdań.
Nie pisz wstępu ani zakończenia całej pracy i nie zaczynaj od „Po pierwsze”."""

KIERUNEK_E5 = {
    "nie_zgadzam": ("Autor wypracowania NIE zgadza się z tezą. Pokaż fakty i czynniki z {obszar}, które PRZECZĄ tezie: "
                    "inne czynniki były ważniejsze, inni władcy lub zjawiska odegrali większą rolę, istniały "
                    "ograniczenia i porażki, skutki bywały odwrotne. Nigdy nie pisz, że teza jest prawdziwa, i nie "
                    "powtarzaj tezy jako prawdy."),
    "zgadzam": ("Autor wypracowania zgadza się z tezą. Pokaż fakty i czynniki z {obszar}, które POTWIERDZAJĄ tezę: "
                "przyczyny, przebieg i skutki, które dowodzą jej trafności. Nie pisz, że teza jest fałszywa lub "
                "słuszna tylko częściowo."),
}
KIERUNEK_ANTY = ("Autor wypracowania NIE zgadza się z tezą, bo uważa, że: {antyteza}. Pokaż fakty i czynniki z {obszar}, "
                 "które POTWIERDZAJĄ to stanowisko autora. Nigdy nie pisz, że teza jest prawdziwa.")
POPRAWKI_E5 = {
    "krotki": "Napisz dłuższy akapit: 7-8 pełnych zdań, około 140 słów.",
    "lista": "Pisz jednym ciągłym akapitem, bez punktów i bez nagłówków.",
    "sprzecznosc": {"nie_zgadzam": "Autor NIE zgadza się z tezą: nie pisz, że teza jest prawdziwa.",
                    "zgadzam": "Autor zgadza się z tezą: nie pisz, że teza jest fałszywa."},
    "powtorzenia": "Nie powtarzaj zdań ani myśli z poprzednich akapitów; wybierz inne fakty.",
}


def antyteza(teza: str) -> str | None:
    """Teza z negacją → jej zaprzeczenie bez podwójnej negacji: „… są niesłusznie określane …” → „… są słusznie
    określane …”, „… nie był …” → „… był …”. Teza bez negacji → None (wtedy „nie jest prawdą, że …”).
    Po co: przy „nie zgadzam się” z tezą zanegowaną mały model czytał „przeczą tezie” jako „przeczą nazwie
    epoki” i pisał argumenty ZA tezą (próbka 26.09, belle époque)."""
    t = teza.strip().rstrip(".")
    if re.search(r"\bniesłusznie\b", t, re.I):
        return re.sub(r"\b([Nn])iesłusznie\b", lambda m: ("S" if m.group(1) == "N" else "s") + "łusznie", t, count=1)
    if re.search(r"\bnie\s+(?!tylko\b)\w", t):
        return re.sub(r"\bnie\s+(?!tylko\b)", "", t, count=1)
    return None


def teza_w_zdaniu(teza: str) -> str:
    """Teza do wstawienia po „że …”: bez kropki, pierwsza litera mała, chyba że pierwsze słowo to nazwa własna
    (lista _PROPRIA: imiona postaci, państwa) albo liczebnik rzymski („II Rzeczpospolita”)."""
    t = teza.strip().rstrip(".")
    sl = t.split()
    if not sl:
        return t
    wlasna = re.fullmatch(r"[IVXLC]+", sl[0]) or sl[0].strip("„\"") in _PROPRIA or sl[0].isupper()
    return t if wlasna else t[:1].lower() + t[1:]


def _dopelniacz(przym: str) -> str:
    """militarny → militarnego, społeczno-gospodarczy → społeczno-gospodarczego (do „aspektu …”)."""
    if przym.endswith("y"):
        return przym[:-1] + "ego"
    if przym.endswith("i"):
        return przym + "ego"
    return przym


def _lista(el: list[str]) -> str:
    return f"{el[0]}, {el[1]} oraz {el[2]}" if len(el) == 3 else ", ".join(el)


def szablon_e5(teza: str, pozycje: list[str], wybor: bool, stanowisko: str) -> dict:
    """Wstęp, zakończenie i nagłówek / zdanie wiążące każdego akapitu. Stanowisko kategoryczne, to samo zdanie
    stanowiska („Nie zgadzam się z tą tezą.” / „Zgadzam się z tą tezą.”) we wstępie i w zakończeniu."""
    tz = teza_w_zdaniu(teza)
    nie = stanowisko == "nie_zgadzam"
    st = "Nie zgadzam się z tą tezą." if nie else "Zgadzam się z tą tezą."
    anty = antyteza(teza)
    ocena = (teza_w_zdaniu(anty) if anty else f"nie jest prawdą, że {tz}") if nie else tz
    dowod = "przeczą tej tezie" if nie else "potwierdzają tę tezę"
    if wybor:
        zapowiedz = (f"Swoje stanowisko uzasadnię na trzech przykładach: {_lista(pozycje)}, ponieważ każdy z nich "
                     f"dostarcza faktów, które {dowod}.")
        analiza = f"Omówione przykłady: {_lista(pozycje)} pokazały, że fakty historyczne {dowod}."
    else:
        zapowiedz = (f"Swoje stanowisko uzasadnię, omawiając kolejno aspekt {_lista(pozycje)}, ponieważ w każdym z nich "
                     f"można wskazać fakty, które {dowod}.")
        analiza = (f"Analiza aspektu {_lista([_dopelniacz(p) for p in pozycje])} pokazała, że fakty historyczne "
                   f"{dowod}.")
    cytat = teza.strip() if re.search(r"[.!?]$", teza.strip()) else teza.strip() + "."
    wstep = f"Teza tematu brzmi: „{cytat}” {st} Uważam, że {ocena}. {zapowiedz}"
    zak = (f"Podsumowując, {st[0].lower() + st[1:-1]}, ponieważ {ocena}. {analiza} Dlatego podtrzymuję stanowisko "
           f"przedstawione we wstępie: teza {'nie jest słuszna' if nie else 'jest słuszna'}.")
    akapity = []
    for i, p in enumerate(pozycje):
        if wybor:
            glowa = f"{PORZADKOWE[i]}, {'tezie przeczy' if nie else 'tezę potwierdza'} przykład: {p}."
            ogon = f"Ten przykład pokazuje więc, że teza {'nie jest słuszna' if nie else 'jest słuszna'}."
        else:
            glowa = (f"{PORZADKOWE[i]}, teza nie znajduje potwierdzenia w aspekcie {miejscownik(p)}." if nie
                     else f"{PORZADKOWE[i]}, tezę potwierdza aspekt {p}.")
            ogon = f"Aspekt {p} nie potwierdza więc tezy." if nie else f"Aspekt {p} potwierdza więc tezę."
        akapity.append((glowa, ogon))
    return {"wstep": wstep, "zakonczenie": zak, "akapity": akapity}


def prompt_e5(teza: str, pozycja: str, wybor: bool, stanowisko: str, material: str, poprawki: list[str]) -> tuple[str, str]:
    """(system, user) jednego akapitu e5. Materiał ciągłym tekstem (lista faktów w prompcie → lista w odpowiedzi).
    Teza z negacją przy „nie zgadzam się”: kierunek podany wprost jako antyteza („są słusznie określane …”)."""
    obszar = f"przykładu: {pozycja}" if wybor else f"aspektu {_dopelniacz(pozycja)}"
    anty = antyteza(teza) if stanowisko == "nie_zgadzam" else None
    if anty:
        kier = KIERUNEK_ANTY.format(obszar=obszar, antyteza=anty)
        cel = f"że {teza_w_zdaniu(anty)}"
    else:
        kier = KIERUNEK_E5[stanowisko].format(obszar=obszar)
        cel = "dlaczego teza jest nieprawdziwa" if stanowisko == "nie_zgadzam" else "dlaczego teza jest prawdziwa"
    sys_ = SYS_E5.format(kierunek=kier)
    rel = "NIE zgadza się" if stanowisko == "nie_zgadzam" else "zgadza się"
    user = ((f"MATERIAŁ:\n{material}\n\n" if material else "")
            + f"Teza, z którą autor {rel}: {teza}\n" + (f"Stanowisko autora: {anty}.\n" if anty else "")
            + (f"Przykład: {pozycja}\n" if wybor else f"Aspekt: {pozycja}\n")
            + f"\nNapisz akapit (ciągły tekst, 6-8 zdań) o {obszar}, który pokazuje, {cel}."
            + ("\nUWAGA: " + " ".join(poprawki) if poprawki else ""))
    return sys_, user


def elementy_z_odpowiedzi(t: str) -> list[str]:
    """Odpowiedź modelu w kroku wyboru → nazwy przykładów. Obsługuje jedną linię z przecinkami i listę numerowaną
    z opisami („1. **Węgry (1956)**: …”): bierze tekst przed „(”, „:”, „ - ”; nazwa 1-5 słów z wielkiej litery."""
    t = re.sub(r"\*\*|__|`+", "", t)
    linie = [l.strip() for l in t.splitlines() if l.strip()]
    lista = [RE_LISTA.sub("", l) for l in linie if RE_LISTA.match(l)]
    kand = lista if len(lista) >= 2 else [x for l in linie[:2] for x in re.split(r",|;|\s+oraz\s+|\s+i\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", l)]
    out: list[str] = []
    for x in kand:
        x = re.split(r"\s*[(:]|\s[-\u2013\u2014]\s", x)[0].strip(" .,;\"„”'")
        if x and x[:1].isupper() and 1 <= len(x.split()) <= 5 and not any(_podobne(x, o) or x.lower() == o.lower() for o in out):
            out.append(x)
    return out[:3]


def _wybierz_elementy(temat: str, teza: str, wybor: str, *, url, wiki, karty, czat_fn, bez_myslenia,
                      baza=None, material_fn=None) -> tuple[list[str], float]:
    """Krok planu dla tematów „trzech wybranych …”: model wskazuje trzy przykłady (materiał: karty + Wikipedia).
    Brakujące przykłady uzupełniane z kart materiału (postać przy władcach/postaciach, inaczej tytuł karty)."""
    m = (material_fn or material_e5)(wiki, karty, teza, wybor, element=True, n_wiki=2, baza=baza)
    t, s = czat_fn(url, "Odpowiadasz krótko, po polsku.", (f"MATERIAŁ:\n{m['tekst']}\n\n" if m["tekst"] else "")
                   + f"Temat wypracowania: {temat}\nWymień trzy przykłady ({wybor}) najlepiej pasujące do tego tematu. "
                     "Odpowiedz jedną linią: trzy nazwy oddzielone przecinkami, bez numerów, opisów i dat.", None,
                   max_tokens=120, bez_myslenia=bez_myslenia)
    el = elementy_z_odpowiedzi(t)
    pole = "postac" if re.search(r"władc|postac|osób|polityk|przywódc|wodz", wybor) else "tytul"
    for c in m["fakty"]:
        if len(el) >= 3:
            break
        x = (c.get(pole) or "").strip()
        if x and not any(_podobne(x, o) or x.lower() == o.lower() for o in el):
            el.append(x)
    while len(el) < 3:
        el.append(f"kolejny przykład ({len(el) + 1})")
    return el, s


def napisz_e5(z: dict, *, url: str, wiki=None, karty=None, bez_myslenia: bool = True, stanowisko: str = "nie_zgadzam",
              czat_fn=czat, temperatura: float = 0.7, proby: int = 2, regeneracje_dlugosci: int = 3,
              baza=None, material_fn=None) -> tuple[str, float, dict]:
    """Wypracowanie e5 → (tekst, sekundy, meta). Model pisze tylko akapity; wstęp, zakończenie, nagłówki i zdania
    wiążące akapitów trzyma kod (`szablon_e5`). Każdy akapit: do `proby` prób (temperatura `temperatura`), próba bez
    wad kończy, inaczej wygrywa próba z najmniejszą karą (`ocen_akapit`). Po wszystkich akapitach: gdy esej ma mniej
    niż 300 słów, najkrótszy akapit jest generowany od nowa (do `regeneracje_dlugosci` razy), bez dopisywania.
    `material_fn`: zamiennik `material_e5` o tej samej sygnaturze (np. pomiar sufitu z faktami wzorcowymi tematu,
    scripts/fakty_eseju_wzorcowe.py); None = `material_e5` (karty, Wikipedia, baza akapitów)."""
    if stanowisko not in STANOWISKA_E5:
        raise ValueError(f"stanowisko e5: {STANOWISKA_E5}, jest {stanowisko!r}")
    temat = z.get("temat") or (tematy(z["polecenie"]) or [z["polecenie"]])[0]
    teza, pozycje, wybor = rozbierz(temat)
    sek = 0.0
    if wybor:
        pozycje, s = _wybierz_elementy(temat, teza, wybor, url=url, wiki=wiki, karty=karty, czat_fn=czat_fn,
                                       bez_myslenia=bez_myslenia, baza=baza, material_fn=material_fn)
        sek += s
    rdzen = _rdzen_tezy(teza)
    mats: list[dict] = []
    for p in pozycje:  # kolejno: każdy akapit dostaje karty nieużyte w poprzednich
        mats.append((material_fn or material_e5)(wiki, karty, teza, p, element=bool(wybor),
                                                 pomin=[c for m in mats for c in m["fakty"]],
                                                 baza=baza, pomin_bazy={i for m in mats for i in m.get("z_bazy", [])},
                                                 stanowisko=stanowisko))

    # daty weryfikowane względem materiału CAŁEGO eseju (karty są rozdzielone między akapity, a data z karty
    # akapitu 1, np. Grunwald 1410, w akapicie 3 nadal jest poprawna)
    wszystkie_fakty = [c for m in mats for c in m["fakty"]]
    tekst_mat = (" ".join(m["tekst"] for m in mats) + " " + teza) if any(m["tekst"] for m in mats) else ""

    def generuj(i: int, poprzednie: list[str], poprawki0: list[str]) -> tuple[dict, int]:
        nonlocal sek
        najl, n, poprawki = None, 0, list(poprawki0)
        for _ in range(max(1, proby)):
            sys_, user = prompt_e5(teza, pozycje[i], bool(wybor), stanowisko, mats[i]["tekst"], poprawki)
            t, s = czat_fn(url, sys_, user, None, max_tokens=MAX_TOKENOW_AKAPITU, temperature=temperatura,
                           bez_myslenia=bez_myslenia)
            sek += s
            n += 1
            o = ocen_akapit(t, rdzen=rdzen, stanowisko=stanowisko, fakty=wszystkie_fakty, tekst_materialu=tekst_mat,
                            poprzednie=poprzednie)
            if najl is None or o["kara"] < najl["kara"]:
                najl = o
            if not o["wady"]:
                break
            poprawki = list(dict.fromkeys(poprawki0 + [POPRAWKI_E5[w][stanowisko] if w == "sprzecznosc"
                                                       else POPRAWKI_E5[w] for w in o["wady"]]))
        return najl, n

    oceny, n_prob, gotowe, n_sprz = [], [], [], []
    for i in range(len(pozycje)):
        o, n = generuj(i, [z for zd in gotowe for z in zd], [])
        zd, us = _wykoncz(o)
        oceny.append(o); n_prob.append(n); gotowe.append(zd); n_sprz.append(us)
    szab = szablon_e5(teza, pozycje, bool(wybor), stanowisko)

    def sklad() -> str:
        ak = [f"{g} {' '.join(zd)} {o}".replace("  ", " ") for (g, o), zd in zip(szab["akapity"], gotowe)]
        return "\n\n".join([szab["wstep"], *ak, szab["zakonczenie"]])

    n_reg = 0
    while _slowa(sklad()) < MIN_SLOW_ESEJU and n_reg < regeneracje_dlugosci:
        i = min(range(len(gotowe)), key=lambda k: sum(_slowa(x) for x in gotowe[k]))
        inne = [z for k, zd in enumerate(gotowe) if k != i for z in zd]
        o, n = generuj(i, inne, [POPRAWKI_E5["krotki"]])
        n_reg += 1
        n_prob[i] += n
        zd, us = _wykoncz(o)
        if sum(_slowa(x) for x in zd) > sum(_slowa(x) for x in gotowe[i]) and "sprzecznosc" not in o["wady"]:
            oceny[i], gotowe[i], n_sprz[i] = o, zd, us
    tekst = sklad()
    meta = {"stanowisko": stanowisko, "aspekty": pozycje, "wybor": bool(wybor), "proby": n_prob,
            "regeneracje_dlugosci": n_reg, "slow": _slowa(tekst), "slowa_akapitow": [sum(_slowa(x) for x in zd) for zd in gotowe],
            "wady_akapitow": [o["wady"] for o in oceny],
            "daty_podmienione": [d for o in oceny for d in o["podmienione"]],
            "daty_usuniete": [d for o in oceny for d in o["usuniete"]],
            "powtorzenia_usuniete": sum(o["powtorzenia"] for o in oceny),
            "sprzeczne_usuniete": sum(n_sprz), "sprzeczne_pozostale": sum(len(set(o["sprzeczne"]) & set(zd))
                                                                          for o, zd in zip(oceny, gotowe)),
            "karty": sum(len(m["fakty"]) for m in mats), "wiki": sum(len(m["fragmenty"]) for m in mats),
            "akapity_bazy": sum(len(m.get("z_bazy", [])) for m in mats)}
    return tekst, sek, meta


# ---------------------------------------------------------------- e6: akapity z bazy wiedzy

SYS_WYBOR_E6 = "Wybierasz akapit do wypracowania maturalnego z historii. Odpowiadasz tylko jednym numerem."


def _wybierz_akapit(teza: str, pozycja: str, stanowisko: str, kand: list[dict], *, url: str, czat_fn,
                    bez_myslenia: bool) -> tuple[dict | None, float]:
    """Model wskazuje numer kandydata (pytanie zamknięte, łatwe dla małego modelu); 0 albo śmieci → None."""
    st = "Zgadzam się z tą tezą." if stanowisko == "zgadzam" else "Nie zgadzam się z tą tezą."
    user = (f"Teza wypracowania: {teza}\nStanowisko: {st}\nAspekt albo przykład: {pozycja}\n"
            "Który akapit najlepiej uzasadnia to stanowisko w tym aspekcie? Odpowiedz jednym numerem "
            "(0, jeśli żaden nie pasuje).\n\n" + "\n\n".join(f"{i}. {c['tekst']}" for i, c in enumerate(kand, 1)))
    t, s = czat_fn(url, SYS_WYBOR_E6, user, None, max_tokens=8, temperature=0, bez_myslenia=bez_myslenia)
    m = re.search(r"\d+", t or "")
    k = int(m.group()) if m else 0
    return (kand[k - 1] if 1 <= k <= len(kand) else None), s


_RZYM = {"I": 1, "V": 5, "X": 10, "L": 50}


def _rzym(s: str) -> int:
    w = [_RZYM[c] for c in s]
    return sum(-v if i + 1 < len(w) and v < w[i + 1] else v for i, v in enumerate(w))


def _okres_tekstu(t: str) -> tuple[int, int] | None:
    """Okres tezy: lata (`_okres_tezy`), a bez lat wieki zapisane rzymskimi cyframi („XI–XII wieku”, „w XVII w.”)."""
    o = _okres_tezy(t)
    if o:
        return o
    m = re.search(r"\blat\w*\s+(\d)0\.?\s+([IVXL]{1,6})\s*w(?:iek|\.|\b)", t)       # „w latach 50. XX wieku”
    if m:
        p = 100 * (_rzym(m.group(2)) - 1) + 10 * int(m.group(1))
        return p, p + 9
    m = re.search(r"\b([IVXL]{1,6})\s*[–\-]\s*([IVXL]{1,6})\s*w(?:iek|\.|\b)", t)
    if m:
        return 100 * (_rzym(m.group(1)) - 1), 100 * _rzym(m.group(2)) - 1
    m = re.search(r"\b([IVXL]{1,6})\s*w(?:iek|\.|\b)", t)
    if m:
        c = _rzym(m.group(1))
        return 100 * (c - 1), 100 * c - 1
    return None


def _pasuje_do_tezy(c: dict, tz: set[str], okres: tuple[int, int] | None) -> bool:
    """Akapit bazy dotyczy tematu: przy znanym okresie tezy i akapitu okresy się pokrywają (±10 lat), inaczej akapit
    ma wspólny rdzeń z tezą (teza, podmiot albo pozycja akapitu). Chroni przed akapitem o innym podmiocie (temat o
    Karolu Wielkim, akapit o Justynianie) albo z innej epoki (XI-XII wiek, akapit o 1320 roku)."""
    if okres:
        lata = _lata(c.get("tekst", ""))   # lata z samego akapitu: teza bazy „1945-1991” obejmuje też Sajgon 1975
        if lata:
            return sum(okres[0] - 10 <= r <= okres[1] + 10 for r in lata) * 2 >= len(lata)
    oc = _okres_tezy(c.get("okres", "")) or _okres_tekstu(c.get("teza", ""))
    if okres and oc:
        return oc[0] <= okres[1] + 10 and okres[0] - 10 <= oc[1]
    # rdzenie 5-literowe: odmiana imion („Karola” / „Karol”); ogólne „panowanie”, „rozwój” nie wystarczą same
    t5, r5 = {w[:5] for w in tz}, lambda s: {w[:5] for w in _stemy(s)}
    return bool(t5 & r5(c.get("podmiot", ""))) or len(t5 & r5(c.get("teza", ""))) >= 2


def napisz_e6(z: dict, *, url: str, wiki=None, karty=None, baza=None, bez_myslenia: bool = True,
              stanowisko: str = "nie_zgadzam", czat_fn=czat, n_kand: int = 4) -> tuple[str, float, dict]:
    """Wypracowanie e6 → (tekst, sekundy, meta): akapity z bazy wiedzy (matura/wiedza.py) wklejone dosłownie, model
    tylko wybiera numer akapitu dla każdej pozycji; wstęp, zakończenie, nagłówki i zdania wiążące z `szablon_e5`.
    Styl nie jest oceniany, a przepisywanie wprowadza błędy faktów, więc tekst bazy zostaje bez zmian. Kandydaci
    tylko trafni (`_pasuje_do_tezy`); pozycja bez trafnego akapitu albo odrzucona przez model (0) dostaje akapit e5
    (ten sam szablon i stanowisko). Temat „trzech wybranych …” bez trzech przykładów w bazie → cały esej jak e5."""
    from .wiedza import BazaAkapitow
    baza = baza if baza is not None else BazaAkapitow()
    temat = z.get("temat") or (tematy(z["polecenie"]) or [z["polecenie"]])[0]
    teza, pozycje, wybor = rozbierz(temat)
    kier = "potwierdza" if stanowisko == "zgadzam" else "przeczy"
    tz, okres = _stemy(teza) - _OGOLNE - _STEMY_NEGACJI, _okres_tekstu(teza)
    trafne = lambda lista: [c for c in lista if _pasuje_do_tezy(c, tz, okres)]
    if wybor:
        pozycje = []
        for c in trafne(baza.kandydaci(teza, wybor, kierunek=kier, n=40, tryb="elementy") if len(baza) else []):
            if c["pozycja"] not in pozycje:
                pozycje.append(c["pozycja"])
        pozycje = pozycje[:3]

    def e5():
        t, s, m = napisz_e5(z, url=url, wiki=wiki, karty=karty, bez_myslenia=bez_myslenia, stanowisko=stanowisko,
                            czat_fn=czat_fn)
        return t, s, m

    if not len(baza) or len(pozycje) < 3:
        t, s, m = e5()
        return t, s, {**m, "z_bazy": 0, "zapas_e5": True}
    sek, uzyte, akapity, trafienia = 0.0, set(), [], 0
    for p in pozycje:
        kand = trafne(baza.kandydaci(teza, p, kierunek=kier, n=3 * n_kand, tryb="elementy" if wybor else None,
                                     pomin=uzyte))
        # aspekt akapitu zgodny z pozycją (sędzia karze treść w złym aspekcie: „Alkuin w aspekcie gospodarczym”)
        kand = [c for c in kand if _stemy(p) & _stemy(c["pozycja"])][:n_kand]
        c = None
        if kand:
            c, s = _wybierz_akapit(teza, p, stanowisko, kand, url=url, czat_fn=czat_fn, bez_myslenia=bez_myslenia)
            sek += s
        trafienia += c is not None
        if c is not None:
            uzyte.add(c["id"])
        akapity.append(c)
    braki = [i for i, c in enumerate(akapity) if c is None]
    if braki and (wybor or len(braki) == len(akapity)):   # przykłady z bazy bez akapitów albo nic z bazy → e5
        t, s, m = e5()
        return t, sek + s, {**m, "z_bazy": 0, "zapas_e5": True}
    szab = szablon_e5(teza, pozycje, bool(wybor), stanowisko)
    ak = [f"{g} {c['tekst'].strip()} {o}" if c is not None else None for (g, o), c in zip(szab["akapity"], akapity)]
    if braki:   # akapity e5 dla pozycji bez bazy: ten sam szablon (nagłówek i zdanie wiążące) i to samo stanowisko
        t5, s5, _ = e5()
        sek += s5
        czesci = t5.split("\n\n")
        for i in braki:
            ak[i] = czesci[1 + i] if len(czesci) == 5 else ""
    tekst = "\n\n".join([szab["wstep"], *[a for a in ak if a], szab["zakonczenie"]])
    meta = {"stanowisko": stanowisko, "aspekty": pozycje, "wybor": bool(wybor), "z_bazy": len(akapity) - len(braki),
            "wybrane_przez_model": trafienia, "akapity_bazy": [c["id"] if c else None for c in akapity],
            "kierunki": [c["kierunek"] if c else None for c in akapity], "akapity_e5": braki, "slow": _slowa(tekst)}
    return tekst, sek, meta


# ---------------------------------------------------------------- wybór tematu

def pokrycie_tematu(temat: str, karty, wiki) -> tuple[float, float, float]:
    """(pokrycie, najsłabszy aspekt, pokrycie bez limitu): na każdy aspekt liczba trafnych kart (≥ 2 wspólne rdzenie
    z tezą, bez słów ogólnych jak „polski”, „wieku”) + 0,5 × trafne fragmenty Wikipedii, z limitem 6 na aspekt
    (powyżej 6 trafnych faktów więcej materiału nie pomaga; bez limitu tylko do rozstrzygania remisów).
    Temat z wyborem trzech przykładów: jedno zapytanie (teza + rodzaj przykładu) liczone za trzy, × 0,8 (krok
    wyboru przykładów przez mały model to dodatkowe ryzyko)."""
    from .router import epoka_pewna
    teza, aspekty, wybor = rozbierz(temat)
    tz = _stemy(teza) - _OGOLNE - _STEMY_NEGACJI
    prog = 2 if len(tz) >= 4 else 1
    wyn = []
    for a in ([wybor] if wybor else aspekty):
        n = 0.0
        if karty is not None:
            ks = _bez_powtorzen_kart(karty.szukaj(f"{teza} {a}", 8, epoka=epoka_pewna(teza)))
            n += sum(1 for c in ks if len(tz & _stemy(f"{c['fakt']} {c.get('postac', '')} {c.get('tytul', '')}")) >= prog)
        if wiki is not None:
            frs = wiki.szukaj(f"{teza} {a}", 3)
            n += 0.5 * sum(1 for f in frs if len(tz & _stemy(f"{f['tytul']} {f['tekst']}")) >= prog)
        wyn.append(n)
    kap = [min(n, 6.0) for n in wyn]
    if wybor:
        return 3 * kap[0] * 0.8, kap[0] * 0.8, 3 * wyn[0] * 0.8
    return sum(kap), min(kap) if kap else 0.0, sum(wyn)


# Temat z podmiotem z historii Polski: wszystkie nasze modele piszą o nich najsłabiej, w obu językach (E1 na dev,
# review/esej-e8-2026-09-26; sonda 1a). Znaczniki: państwo i jego instytucje, dynastie, powstania, postacie.
# „Polacy” i „zabór” dodane 27.09 po walidacji (błąd słownika: brakowało tych form; walidacja nie jest dla
# poprawki niezależna, niezależna będzie prognoza 2024-czerwiec i 2025-czerwiec).
RE_TEMAT_POLSKI = re.compile(
    r"(?i:\bpols\w*|\bpolak\w*|\bpolacy\b|\bpolek\b|rzecz\w*pospolit|jagiell|piast|\blitw|\bsejm|szlacht|\bzabor|\bzabór|\brozbior|"
    r"galicj|legion\w* polsk|księstw\w* warszawsk|królestw\w* polsk|solidarnoś|"
    r"powstani\w* (?:listopadow|styczniow|kościuszkowsk|warszawsk|wielkopolsk|śląsk|krakowsk))|"
    r"\bPRL\b|\bII RP\b|Piłsudsk|Dmowsk|Paderewsk|Kościuszk|Sobiesk|Batorego|\bBatory|Mieszk|Chrobr|Krzywoust|"
    r"Łokiet|Kazimierz|Zygmunt|Władysław|Bolesław|Stanisław|Poniatowsk|Jadwig|Gomułk|Gier(?:ek|k)|Wałęs|Jaruzelsk|"
    r"Sikorsk|Mikołajczyk|Bierut|Traugutt|Czartorysk|Zamoysk|Kołłątaj|Wyszyńsk|Mazowieck|Wielopolsk|Witos|Daszyńsk")
REGULY_TEMATU = ("pokrycie", "powszechna")


def temat_polski(temat: str) -> bool:
    """Czy teza tematu ma podmiot z historii Polski (znaczniki `RE_TEMAT_POLSKI` w tezie, bez polecenia)."""
    return bool(RE_TEMAT_POLSKI.search(rozbierz(temat)[0]))


def wybierz_temat(tematy_arkusza: list, karty, wiki, regula: str = "pokrycie") -> int:
    """Indeks tematu (0-2). regula „pokrycie” (domyślna, potok PL): najlepsze pokrycie materiałem (`pokrycie_tematu`);
    remis: lepszy najsłabszy aspekt, potem pokrycie bez limitu, potem niższy numer. regula „powszechna” (E1, potok
    EN): najpierw tematy bez podmiotu z historii Polski (`temat_polski`), wśród nich jak „pokrycie”.
    tematy_arkusza: teksty tematów albo zadania z polem 'temat'."""
    if regula not in REGULY_TEMATU:
        raise ValueError(f"nieznana reguła tematu: {regula} (znane: {', '.join(REGULY_TEMATU)})")
    teksty = [t["temat"] if isinstance(t, dict) else t for t in tematy_arkusza]
    oceny = [pokrycie_tematu(t, karty, wiki) for t in teksty]
    pierwsze = [regula == "powszechna" and not temat_polski(t) for t in teksty]
    return max(range(len(oceny)), key=lambda i: (pierwsze[i], *oceny[i], -i))
