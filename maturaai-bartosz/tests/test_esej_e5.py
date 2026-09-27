"""Wypracowanie e5 (matura/esej.py, `napisz_e5`): stanowisko kategoryczne, kontrola akapitu, weryfikator dat, wybór tematu.

Co to jest: testy bez GPU i sieci; czat modelu podmieniany funkcją (czat_fn), karty i Wikipedia to małe atrapy.
Po co: pilnują zachowań, które decydują o punktach CKE (≥ 300 słów, brak list i powtórzeń, brak sprzeczności
stanowiska z akapitami, daty zgodne z materiałem). Co zrobić: uv run pytest -q tests/test_esej_e5.py
"""
import itertools
import re

import pytest

from matura import esej, harness

TEMAT = ("Władysław Jagiełło był najwybitniejszym władcą Polski z dynastii Jagiellonów. Zajmij stanowisko wobec "
         "powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji aspekty: militarny, ustrojowy i "
         "społeczno-gospodarczy.")
TEZA = "Władysław Jagiełło był najwybitniejszym władcą Polski z dynastii Jagiellonów."

# Sztuczne słowa o różnych 6-literowych rdzeniach: każde zdanie ma własny zestaw słów (brak fałszywych powtórzeń).
_SYL = ["ka", "to", "mi", "re", "su", "lo", "pe", "ni", "da", "wo", "gu", "ba"]
_SLOWA = ["".join(s) for s in itertools.product(_SYL, repeat=3)]


class Generator:
    """Atrapa modelu: kolejne wywołania dostają kolejne unikalne zdania; `plan` = liczba zdań w kolejnych
    odpowiedziach (po wyczerpaniu planu: `domyslnie`)."""

    def __init__(self, plan=(), domyslnie=8, wstaw=None):
        self.plan, self.domyslnie, self.wstaw = list(plan), domyslnie, wstaw or {}
        self.n_zdan, self.wolania = 0, []

    def zdanie(self) -> str:
        w = _SLOWA[self.n_zdan * 16:(self.n_zdan + 1) * 16]
        self.n_zdan += 1
        return (" ".join(w)).capitalize() + "."

    def __call__(self, url, system, user, obr=None, **kw):
        self.wolania.append({"system": system, "user": user, **kw})
        k = len(self.wolania) - 1
        if k in self.wstaw:
            return self.wstaw[k], 0.1
        n = self.plan[k] if k < len(self.plan) else self.domyslnie
        return " ".join(self.zdanie() for _ in range(n)), 0.1


def _z(temat=TEMAT):
    return {"id": "t-1", "temat": temat, "polecenie": temat, "esej": True}


# ---------------------------------------------------------------- długość i regeneracja

def test_krotkie_odpowiedzi_sa_regenerowane_do_ponad_300_slow():
    g = Generator(plan=[2, 2, 2, 2, 2, 2])  # każda pierwsza i druga próba: 2 zdania (32 słowa) < 90
    t, s, m = esej.napisz_e5(_z(), url=None, czat_fn=g)
    assert m["slow"] >= esej.MIN_SLOW_ESEJU and len(t.split()) == m["slow"]
    assert m["proby"] == [3, 2, 2]                # 2 próby na akapit (za krótkie) + regeneracja najkrótszego
    assert m["regeneracje_dlugosci"] == 1 and sum(m["proby"]) == len(g.wolania)
    assert all(c["temperature"] == pytest.approx(0.7) for c in g.wolania)
    assert "UWAGA: Napisz dłuższy akapit" in g.wolania[1]["user"]  # druga próba dostaje wskazówkę


def test_akapit_bez_wad_konczy_na_jednej_probie():
    g = Generator(domyslnie=8)  # 8 × 16 = 128 słów
    t, s, m = esej.napisz_e5(_z(), url=None, czat_fn=g)
    assert m["proby"] == [1, 1, 1] and len(g.wolania) == 3 and m["regeneracje_dlugosci"] == 0
    assert m["slow"] >= 300 and all(90 <= n <= esej.MAX_SLOW_AKAPITU for n in m["slowa_akapitow"])


def test_za_dlugi_akapit_przyciety_calymi_zdaniami():
    g = Generator(domyslnie=14)  # 224 słowa
    _, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g)
    assert all(n <= esej.MAX_SLOW_AKAPITU for n in m["slowa_akapitow"])


# ---------------------------------------------------------------- listy

def test_listy_sklejone_w_zdania():
    g = Generator(domyslnie=8)
    lista = ("**Fakty:**\n" + "\n".join(f"- {g.zdanie()[:-1]}" for _ in range(3))
             + "\n1. **Grunwald (1410):** " + g.zdanie() + "\n1386 - koronacja Jagiełły na króla Polski\n"
             + "\n".join(g.zdanie() for _ in range(4)))
    g.wstaw = {0: lista}
    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g, proby=1)  # jedna próba: sklejona lista zostaje w eseju
    assert m["wady_akapitow"][0] == ["lista"]
    for linia in t.splitlines():
        assert not re.match(r"\s*(?:[-•*]|\d+[.)])\s", linia), linia
    assert "**" not in t and "Fakty:" not in t and "Grunwald (1410):" not in t
    assert "Koronacja Jagiełły na króla Polski (1386)." in t  # bez materiału daty nie są weryfikowane
    assert len(t.split("\n\n")) == 5  # wstęp, trzy akapity, zakończenie


def test_lista_to_wada_i_druga_proba_bez_listy_wygrywa():
    g = Generator(domyslnie=8)
    g.wstaw = {0: "\n".join(f"- {g.zdanie()}" for _ in range(8))}
    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g)
    assert m["proby"][0] == 2 and m["wady_akapitow"][0] == []
    assert "UWAGA: Pisz jednym ciągłym akapitem" in g.wolania[1]["user"]


def test_sklejanie_usuwa_etykiety_encje_i_wewnetrzne_wyliczenia():
    t, _ = esej._sklej_liste("Stabilizacja i rozwój: Belle époque była okresem pokoju w Europie Zachodniej. "
                             "Po drugie, konflikty (np. I&nbsp;wojna światowa) zamknęły epokę. "
                             "Fakt: W 1905 roku wybuchła rewolucja w Rosji. W latach 50. Akapit o aspekcie kulturowym.")
    assert t == ("Belle époque była okresem pokoju w Europie Zachodniej. Konflikty (np. I wojna światowa) zamknęły "
                 "epokę. W 1905 roku wybuchła rewolucja w Rosji.")


def test_elementy_z_odpowiedzi_modelu():
    assert esej.elementy_z_odpowiedzi("Polska, Węgry, Czechosłowacja.") == ["Polska", "Węgry", "Czechosłowacja"]
    lista = ("Rok 1956 był przełomowym rokiem w systemie komunistycznym, który zmienił blok wschodni.\n\n"
             "1. **Węgry (1956)**: rewolucja\n2. Polska (strajki w 1956 roku)\n3. Czechosłowacja - odwilż")
    assert esej.elementy_z_odpowiedzi(lista) == ["Węgry", "Polska", "Czechosłowacja"]
    assert esej.elementy_z_odpowiedzi("1. Bolesław Chrobry (967–1025)\n2. Mieszko II\n3. Mieszko II") == ["Bolesław Chrobry", "Mieszko II"]


def test_ocena_akapitu_zglasza_liste():
    o = esej.ocen_akapit("- pierwsze zdanie o czymś\n- drugie zdanie o czymś innym\n- trzecie", rdzen=esej._rdzen_tezy(TEZA),
                         stanowisko="nie_zgadzam", fakty=[], tekst_materialu="", poprzednie=[])
    assert "lista" in o["wady"] and "\n" not in o["tekst"] and not o["tekst"].startswith("-")


# ---------------------------------------------------------------- powtórzenia

def test_powtorzone_zdania_usuniete_w_akapicie_i_miedzy_akapitami():
    g = Generator(domyslnie=8)
    powt = "Wojska polsko-litewskie pokonały zakon krzyżacki pod Grunwaldem, co osłabiło państwo zakonne."
    g.wstaw = {0: " ".join([powt, g.zdanie(), powt.upper()] + [g.zdanie() for _ in range(7)]),  # akapit 1: powtórka
               1: " ".join([powt.replace(",", ""), *[g.zdanie() for _ in range(8)]])}           # akapit 2: to samo zdanie
    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g)
    assert t.lower().count("pokonały zakon krzyżacki pod grunwaldem") == 1
    assert m["powtorzenia_usuniete"] == 2 and m["proby"] == [1, 1, 1]


def test_podobne_zdania_i_rozne_zdania():
    assert esej._podobne("Jagiełło pokonał Krzyżaków pod Grunwaldem w 1410 r.", "jagiełło pokonał krzyżaków pod grunwaldem w 1410 r")
    assert esej._podobne("Wojska Jagiełły pokonały Krzyżaków pod Grunwaldem i zdobyły obóz.",
                         "Wojska Jagiełły pokonały Krzyżaków pod Grunwaldem i zdobyły cały obóz.")
    assert not esej._podobne("Jagiełło pokonał Krzyżaków pod Grunwaldem.", "Kazimierz Jagiellończyk zawarł pokój toruński z zakonem.")


# ---------------------------------------------------------------- weryfikator dat

KARTY_DAT = [
    {"fakt": "Wojska polsko-litewskie dowodzone przez Władysława Jagiełłę pokonały pod Grunwaldem armię krzyżacką.",
     "data": "1410", "postac": "Władysław Jagiełło", "termin": "", "tytul": "Grunwald"},
    {"fakt": "Jagiełło przyjął chrzest, poślubił Jadwigę i został koronowany na króla Polski.", "data": "1386",
     "postac": "Jagiełło", "termin": "", "tytul": "Koronacja"},
]
MAT_DAT = " ".join(f"{k['fakt']} ({k['data']})" for k in KARTY_DAT)


def test_weryfikator_podmienia_date_sprzeczna_z_karta():
    a = "W 1409 r. wojska Jagiełły pokonały armię krzyżacką pod Grunwaldem. Było to wielkie zwycięstwo."
    t, podm, usun = esej.weryfikuj_daty(a, KARTY_DAT, MAT_DAT)
    assert t == "W 1410 r. wojska Jagiełły pokonały armię krzyżacką pod Grunwaldem. Było to wielkie zwycięstwo."
    assert podm == ["1409→1410"] and usun == []


def test_weryfikator_odlegly_rok_usuwa_a_rok_z_materialu_zostawia():
    a = "W 1310 r. wojska Jagiełły pokonały armię krzyżacką pod Grunwaldem."  # 100 lat od karty: inne wydarzenie?
    t, podm, usun = esej.weryfikuj_daty(a, KARTY_DAT, MAT_DAT)
    assert t == "Wojska Jagiełły pokonały armię krzyżacką pod Grunwaldem." and podm == [] and usun == ["W 1310 r."]
    a = "W 1386 r. wojska Jagiełły pokonały armię krzyżacką pod Grunwaldem."  # rok z materiału (inna karta)
    assert esej.weryfikuj_daty(a, KARTY_DAT, MAT_DAT) == (a, [], [])


def test_weryfikator_usuwa_sama_date_nie_zdanie():
    a = ("Po unii dynastycznej rozwijał się handel zbożem w 1501 r. W 1499 r., mimo wojen, miasta rosły. "
         "Szlachta zyskała przywileje (1505).")
    t, podm, usun = esej.weryfikuj_daty(a, KARTY_DAT, MAT_DAT)
    assert t == "Po unii dynastycznej rozwijał się handel zbożem. Mimo wojen, miasta rosły. Szlachta zyskała przywileje."
    assert len(esej.zdania(t)) == 3 and usun == ["w 1501 r.", "W 1499 r.", "(1505)"] and podm == []


def test_weryfikator_zostawia_daty_z_materialu_i_liczby():
    a = "Jagiełło został koronowany w 1386 roku. W bitwie walczyło 300 rycerzy z Czech."
    t, podm, usun = esej.weryfikuj_daty(a, KARTY_DAT, MAT_DAT)
    assert t == a and podm == usun == []


def test_weryfikator_nie_podmienia_przy_podobienstwie_samych_panstw():
    karty = [{"fakt": "Objęcie tronu węgierskiego przez Władysława Jagiellończyka sprawiło, że przedstawiciele dynastii "
                      "rządzili Polską, Litwą, Czechami i Węgrami.", "data": "1490", "postac": "", "termin": "", "tytul": ""}]
    a = "Zygmunt Luksemburski miał ambicje dynastyczne wobec Polską, Litwą i Czechami, ale zmarł w 1437 roku."
    t, podm, usun = esej.weryfikuj_daty(a, karty, "Władysław Jagiellończyk (1490).")
    assert podm == [] and usun == ["w 1437 roku"] and t.endswith("ale zmarł.")


def test_weryfikator_w_eseju_nie_skraca_akapitu_o_zdania():
    g = Generator(domyslnie=8)
    g.wstaw = {0: "W 1409 r. wojska Jagiełły pokonały armię krzyżacką pod Grunwaldem. W 1777 r. " + g.zdanie()[0].lower()
               + g.zdanie()[1:] + " " + " ".join(g.zdanie() for _ in range(6))}

    class Karty:
        k = []

        def szukaj(self, zap, n=8, epoka=None):
            return list(KARTY_DAT)

    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g, karty=Karty())
    assert "W 1410 r. wojska Jagiełły pokonały" in t and "1777" not in t
    assert m["daty_podmienione"] == ["1409→1410"] and m["daty_usuniete"] == ["W 1777 r."]
    assert len(esej.zdania(t.split("\n\n")[1])) == 2 + 8  # nagłówek + 8 zdań modelu + zdanie wiążące


def test_data_z_materialu_innego_akapitu_zostaje():
    class Karty:
        k = []

        def szukaj(self, zap, n=8, epoka=None):
            return list(KARTY_DAT)  # akapit 1 zabiera obie karty, akapity 2-3 dostają pusty materiał kart

    g = Generator(domyslnie=8)
    g.wstaw = {2: "Zwycięstwo pod Grunwaldem w 1410 r. wzmocniło pozycję króla wobec szlachty. "
                  + " ".join(g.zdanie() for _ in range(7))}
    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g, karty=Karty())
    assert "w 1410 r. wzmocniło" in t and m["daty_usuniete"] == []


# ---------------------------------------------------------------- sprzeczność kierunku

def test_sprzecznosc_kierunku_heurystyka():
    r = esej._rdzen_tezy(TEZA)
    assert esej.sprzeczne_z_kierunkiem("Władysław Jagiełło był najwybitniejszym władcą Polski.", r, "nie_zgadzam")
    assert esej.sprzeczne_z_kierunkiem("Jagiełło był najwybitniejszym królem.", r, "nie_zgadzam")
    assert not esej.sprzeczne_z_kierunkiem("Jagiełło nie był najwybitniejszym władcą.", r, "nie_zgadzam")
    assert not esej.sprzeczne_z_kierunkiem("To Kazimierz Jagiellończyk był najwybitniejszym władcą z tej dynastii.", r, "nie_zgadzam")
    assert not esej.sprzeczne_z_kierunkiem("Jagiełło pokonał Krzyżaków pod Grunwaldem.", r, "nie_zgadzam")
    assert esej.sprzeczne_z_kierunkiem("Jagiełło nie był najwybitniejszym władcą.", r, "zgadzam")
    assert not esej.sprzeczne_z_kierunkiem("Jagiełło był najwybitniejszym władcą.", r, "zgadzam")
    r2 = esej._rdzen_tezy("Lata 1871–1914 są niesłusznie określane jako belle époque (piękna epoka).")
    assert esej.sprzeczne_z_kierunkiem("Lata 1871–1914 są niesłusznie określane jako belle époque.", r2, "nie_zgadzam")
    assert not esej.sprzeczne_z_kierunkiem("Lata 1871–1914 słusznie określa się jako belle époque.", r2, "nie_zgadzam")


def test_sprzecznosc_w_zdaniu_przyzwalajacym_i_negacja_przed_stopniem_najwyzszym():
    r = esej._rdzen_tezy(TEZA)
    assert esej.sprzeczne_z_kierunkiem("Władysław Jagiełło, choć niewątpliwie najwybitniejszy władca Polski z dynastii "
                                       "Jagiellonów, nie był postacią przełomową.", r, "nie_zgadzam")
    assert not esej.sprzeczne_z_kierunkiem("Jagiełło był wybitny, ale nie można go uznać za najwybitniejszego władcę.",
                                           r, "nie_zgadzam")


def test_sprzecznosc_kierunku_wykryta_i_regenerowana():
    g = Generator(domyslnie=8)
    g.wstaw = {0: TEZA + " " + " ".join(g.zdanie() for _ in range(7))}  # pierwsza próba powtarza tezę jako prawdę
    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=g)
    assert m["proby"][0] == 2 and m["wady_akapitow"][0] == []
    assert "UWAGA: Autor NIE zgadza się z tezą" in g.wolania[1]["user"]
    rozwiniecie = "\n\n".join(t.split("\n\n")[1:4])
    assert "był najwybitniejszym władcą" not in rozwiniecie


def test_prompt_akapitu_ma_kierunek():
    s_nie, u_nie = esej.prompt_e5(TEZA, "militarny", False, "nie_zgadzam", "FAKTY: x.", [])
    s_tak, _ = esej.prompt_e5(TEZA, "militarny", False, "zgadzam", "FAKTY: x.", [])
    assert "PRZECZĄ tezie" in s_nie and "nie powtarzaj tezy jako prawdy" in s_nie and "aspektu militarnego" in s_nie
    assert "POTWIERDZAJĄ tezę" in s_tak and "PRZECZĄ" not in s_tak
    assert "najwyżej 2-3 daty" in s_nie.lower() and "tylko faktów z podanego materiału" in s_nie
    assert u_nie.startswith("MATERIAŁ:\nFAKTY: x.") and "NIE zgadza się" in u_nie


def test_material_akapitow_bez_tych_samych_kart():
    wspolne = [{"fakt": f"Jagiełło {x} jako król z dynastii.", "data": "", "dzial": "Polska", "aspekt": "polityczny"}
               for x in ("pokonał Krzyżaków", "zawarł unię", "odnowił uczelnię", "przyjął chrzest", "wydał przywilej")]
    dzial = [{"fakt": f"Fakt {x} o ustroju państwa.", "data": "", "dzial": "Polska", "aspekt": "ustrojowy"}
             for x in ("sejmiki ziemskie", "przywilej czerwiński", "starostowie grodowi")]

    class Karty:
        k = wspolne + dzial

        def szukaj(self, zap, n=8, epoka=None):
            return list(wspolne)[:n]

    m1 = esej.material_e5(None, Karty(), TEZA, "militarny")
    m2 = esej.material_e5(None, Karty(), TEZA, "ustrojowy", pomin=m1["fakty"])
    assert len(m1["fakty"]) == 3 and not {c["fakt"] for c in m1["fakty"]} & {c["fakt"] for c in m2["fakty"]}
    assert {c["fakt"] for c in dzial} <= {c["fakt"] for c in m2["fakty"]}  # karty działu z pasującym aspektem
    assert m2["tekst"].startswith("FAKTY: ") and "\n-" not in m2["tekst"]  # materiał ciągłym tekstem, bez listy


# ---------------------------------------------------------------- stanowisko we wstępie i zakończeniu

@pytest.mark.parametrize("stanowisko,haslo", [("nie_zgadzam", "nie zgadzam się z tą tezą"), ("zgadzam", "zgadzam się z tą tezą")])
def test_wstep_i_zakonczenie_to_samo_stanowisko(stanowisko, haslo):
    t, _, m = esej.napisz_e5(_z(), url=None, czat_fn=Generator(domyslnie=8), stanowisko=stanowisko)
    cz = t.split("\n\n")
    wstep, zak = cz[0].lower(), cz[-1].lower()
    assert haslo in wstep and haslo in zak and m["stanowisko"] == stanowisko
    assert "częściowo" not in t
    if stanowisko == "zgadzam":
        assert "nie zgadzam" not in wstep and "nie zgadzam" not in zak
    assert "aspekt militarny, ustrojowy oraz społeczno-gospodarczy" in cz[0]
    assert cz[1].startswith("Po pierwsze") and cz[2].startswith("Po drugie") and cz[3].startswith("Po trzecie")
    assert "—" not in t.replace(TEZA, "")  # szablon bez długiego myślnika


def test_zle_stanowisko_to_blad():
    with pytest.raises(ValueError):
        esej.napisz_e5(_z(), url=None, czat_fn=Generator(), stanowisko="czesciowo")


def test_temat_z_wyborem_trzech_przykladow():
    temat = ("W życiu politycznym państwa polskiego w okresie XI–XII wieku dominowały tendencje decentralizacyjne. "
             "Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji panowanie "
             "trzech wybranych władców z tego okresu.")
    g = Generator(domyslnie=8)
    g.wstaw = {0: "Bolesław Chrobry, Kazimierz Odnowiciel, Bolesław Krzywousty"}
    t, _, m = esej.napisz_e5(_z(temat), url=None, czat_fn=g)
    assert m["wybor"] and m["aspekty"] == ["Bolesław Chrobry", "Kazimierz Odnowiciel", "Bolesław Krzywousty"]
    assert "Po pierwsze, tezie przeczy przykład: Bolesław Chrobry." in t and m["slow"] >= 300

    class Karty:  # model podał jeden przykład → brakujące z pola `postac` kart materiału
        k = []

        def szukaj(self, zap, n=8, epoka=None):
            return [{"fakt": "Bolesław Śmiały koronował się na króla i wspierał reformę gregoriańską.", "data": "",
                     "postac": "Bolesław Śmiały"},
                    {"fakt": "Władysław Herman oddał faktyczną władzę palatynowi Sieciechowi.", "data": "",
                     "postac": "Władysław Herman"}]

    g = Generator(domyslnie=8)
    g.wstaw = {0: "Bolesław Chrobry"}
    _, _, m = esej.napisz_e5(_z(temat), url=None, czat_fn=g, karty=Karty())
    assert m["aspekty"] == ["Bolesław Chrobry", "Bolesław Śmiały", "Władysław Herman"]


def test_antyteza_dla_tezy_z_negacja_w_szablonie_i_prompcie():
    t2 = "Lata 1871–1914 są niesłusznie określane jako belle époque (piękna epoka)."
    assert esej.antyteza(t2) == "Lata 1871–1914 są słusznie określane jako belle époque (piękna epoka)"
    assert esej.antyteza("Klęska Polski w 1939 roku nie była nieunikniona.") == "Klęska Polski w 1939 roku była nieunikniona"
    assert esej.antyteza(TEZA) is None and esej.antyteza("Wojna była nie tylko klęską.") is None
    sz = esej.szablon_e5(t2, ["polityczny", "społeczno-gospodarczy", "kulturowy"], False, "nie_zgadzam")
    assert "Uważam, że lata 1871–1914 są słusznie określane jako belle époque" in sz["wstep"]
    assert "nie jest prawdą" not in sz["wstep"] and "słusznie określane" in sz["zakonczenie"]
    s_, u = esej.prompt_e5(t2, "kulturowy", False, "nie_zgadzam", "", [])
    assert "POTWIERDZAJĄ to stanowisko autora" in s_ and "Stanowisko autora: Lata 1871–1914 są słusznie" in u
    assert "ograniczenia i porażki" not in s_  # wskazówki „przeczących faktów” mylą przy tezie zanegowanej


def test_material_z_okresu_tezy():
    k = [{"fakt": "Maria Skłodowska-Curie odkryła polon i rad.", "data": "1898", "aspekt": "kulturowy", "dzial": "XIX"},
         {"fakt": "Utworzono Akademię Umiejętności w Krakowie.", "data": "1872", "aspekt": "kulturowy", "dzial": "XIX"},
         {"fakt": "Trójprzymierze zawarły Niemcy, Austro-Węgry i Włochy.", "data": "1882", "aspekt": "polityczny", "dzial": "XIX"},
         {"fakt": "Unia lubelska połączyła Koronę i Litwę.", "data": "1569", "aspekt": "polityczny", "dzial": "XVI"},
         {"fakt": "Wybuchła I wojna światowa po zamachu w Sarajewie.", "data": "1914", "aspekt": "militarny", "dzial": "XX"}]

    class Karty:
        def __init__(self):
            self.k = k

        def szukaj(self, zap, n=8, epoka=None):
            return [k[3]]  # BM25 trafia w przypadkową kartę

    assert esej._okres_tezy("Rok 1956 był przełomem.") == (1955, 1957)
    assert esej._okres_tezy("Lata 1871–1914 są niesłusznie określane.") == (1871, 1914) and esej._okres_tezy(TEZA) is None
    m = esej.material_e5(None, Karty(), "Lata 1871–1914 są niesłusznie określane jako belle époque.", "kulturowy")
    fakty = [c["fakt"] for c in m["fakty"]]
    assert "Unia lubelska połączyła Koronę i Litwę." not in fakty and len(fakty) == 4
    assert set(fakty[:2]) == {k[0]["fakt"], k[1]["fakt"]}  # pasujący aspekt najpierw


def test_sklejanie_usuwa_naglowki_bez_kropki_i_echo_faktow():
    surowy = ("Rozwój gospodarczy i przemysłowy\nWzrost produkcji przemysłowej przyczynił się do wzrostu dobrobytu.\n"
              "1. **Stabilizacja polityczna**\n"
              "Świętochowski wystąpił w artykule „My i wy”. (1871) Fakt 2: Napoleon utworzył licea państwowe. 6.")
    t, _ = esej._sklej_liste(surowy)
    assert t == ("Wzrost produkcji przemysłowej przyczynił się do wzrostu dobrobytu. Świętochowski wystąpił w artykule "
                 "„My i wy”. Napoleon utworzył licea państwowe.")


def test_teza_w_zdaniu():
    assert esej.teza_w_zdaniu("Rok 1956 był przełomem w systemie komunistycznym.") == "rok 1956 był przełomem w systemie komunistycznym"
    assert esej.teza_w_zdaniu(TEZA).startswith("Władysław Jagiełło")
    assert esej.teza_w_zdaniu("II Rzeczpospolita skutecznie poradziła sobie.").startswith("II Rzeczpospolita")
    assert esej.teza_w_zdaniu("Klęska Polski w 1939 roku była nieunikniona.").startswith("klęska Polski")


# ---------------------------------------------------------------- wybór tematu

class _KartyAtrapa:
    def __init__(self, karty_po_slowie: dict[str, list[dict]]):
        self.m, self.k = karty_po_slowie, [c for v in karty_po_slowie.values() for c in v]

    def szukaj(self, zap, n=8, epoka=None):
        for slowo, karty in self.m.items():
            if slowo in zap:
                return karty[:n]
        return [{"fakt": "Egipcjanie budowali piramidy w Gizie.", "data": "", "dzial": "Starożytność"}]


def test_wybierz_temat_wybiera_temat_z_lepszym_pokryciem():
    t1 = ("Rewolucja amerykańska i francuska miały podobne przyczyny. Zajmij stanowisko wobec powyższej tezy i je "
          "uzasadnij, uwzględniając w swojej argumentacji aspekt polityczny, społeczno-gospodarczy i kulturowy.")
    t3 = ("Zimna wojna osiągnęła apogeum w latach 50. XX wieku. Zajmij stanowisko wobec powyższej tezy i je "
          "uzasadnij, charakteryzując trzy wybrane wydarzenia z tego okresu.")
    karty = _KartyAtrapa({"Jagiełło": [
        {"fakt": f"Władysław Jagiełło {x} jako władca z dynastii Jagiellonów.", "data": "", "dzial": "Polska"}
        for x in ("pokonał Krzyżaków pod Grunwaldem", "zawarł unię w Krewie", "wydał przywilej jedlneński",
                  "odnowił Akademię Krakowską", "przyjął chrzest w Krakowie")]})
    assert esej.wybierz_temat([t1, TEMAT, t3], karty, None) == 1
    assert esej.wybierz_temat([{"temat": TEMAT}, {"temat": t1}], karty, None) == 0

    class Wiki:
        def szukaj(self, zap, k=3):
            return [{"tytul": "Rewolucja francuska", "tekst": "Rewolucja francuska i amerykańska: przyczyny, podobne hasła."}] * k \
                if "Rewolucja" in zap else []

    assert esej.wybierz_temat([TEMAT, t1, t3], None, Wiki()) == 1


@pytest.mark.parametrize("temat,polski", [
    (TEMAT, True),
    ("Niepodległość Polska zawdzięczała przede wszystkim przywództwu Józefa Piłsudskiego.", True),
    ("To właśnie wojny z Turcją, spośród wszystkich XVII-wiecznych konfliktów zbrojnych Rzeczypospolitej, w "
     "największym stopniu przyczyniły się do upadku jej znaczenia w tym stuleciu.", True),
    ("Sejm Czteroletni podjął skuteczną próbę naprawy państwa.", True),
    ("Powstanie styczniowe było zrywem całego społeczeństwa.", True),
    ("Najbardziej udaną próbą odnowienia tradycji imperium rzymskiego w średniowieczu było państwo Karola Wielkiego.",
     False),
    ("Lata 1871–1914 są niesłusznie określane jako belle époque (piękna epoka).", False),
    ("Rok 1956 był przełomem w systemie komunistycznym. Zajmij stanowisko wobec powyższej tezy i je uzasadnij, "
     "uwzględniając w swojej argumentacji wydarzenia z trzech wybranych państw bloku komunistycznego.", False),
    ("Układ Warszawski był narzędziem dominacji ZSRS.", False),
    ("Polacy licznie uczestniczyli w Wiośnie Ludów.", True),
    ("Zabór rosyjski był najbardziej represyjny wobec Polaków.", True),
    ("W zaborze pruskim rozwinęła się praca organiczna.", True),
])
def test_temat_polski(temat, polski):
    assert esej.temat_polski(temat) is polski


def test_regula_powszechna_woli_temat_bez_historii_polski_a_remis_rozstrzyga_pokrycie():
    t1 = ("Rewolucja amerykańska i francuska miały podobne przyczyny. Zajmij stanowisko wobec powyższej tezy i je "
          "uzasadnij, uwzględniając w swojej argumentacji aspekt polityczny, społeczno-gospodarczy i kulturowy.")
    t3 = ("Zimna wojna osiągnęła apogeum w latach 50. XX wieku. Zajmij stanowisko wobec powyższej tezy i je "
          "uzasadnij, charakteryzując trzy wybrane wydarzenia z tego okresu.")
    karty = _KartyAtrapa({
        "Jagiełło": [{"fakt": f"Władysław Jagiełło {x} jako władca z dynastii Jagiellonów.", "data": "", "dzial": "Polska"}
                     for x in ("pokonał Krzyżaków pod Grunwaldem", "zawarł unię w Krewie", "odnowił Akademię")],
        "Zimna": [{"fakt": f"Zimna wojna w latach 50. XX wieku: {x}, apogeum.", "data": "", "dzial": "Po 1945"}
                  for x in ("wojna koreańska", "blokada")]})
    assert esej.wybierz_temat([TEMAT, t1, t3], karty, None) == 0                        # domyślnie: pokrycie
    assert esej.wybierz_temat([TEMAT, t1, t3], karty, None, regula="powszechna") == 2   # bez Polski, lepsze pokrycie
    assert esej.wybierz_temat([TEMAT, TEMAT], karty, None, regula="powszechna") == 0    # same polskie: pokrycie
    with pytest.raises(ValueError):
        esej.wybierz_temat([TEMAT], karty, None, regula="losowa")


# ---------------------------------------------------------------- harness

def test_harness_kieruje_e5_ze_stanowiskiem(monkeypatch):
    wolania = []

    def napisz_e5(z, **kw):
        wolania.append(kw)
        return "esej", 1.0, {}

    monkeypatch.setattr(harness.esej_mod, "napisz_e5", napisz_e5)
    k = object()
    for konfig in ("e5", "e5_zgadzam"):
        assert harness.odpowiedz(konfig, _z(), url="u", wiki=None, obrazy=False, bez_myslenia=True, karty=k)[0] == "esej"
    assert [w["stanowisko"] for w in wolania] == ["nie_zgadzam", "zgadzam"] and all(w["karty"] is k for w in wolania)
    assert "e5" in harness.KONFIGI_ESEJ and "e5_zgadzam" in harness.KONFIGI_ESEJ
    with pytest.raises(ValueError):
        harness.odpowiedz("e5", {"id": "x", "esej": False, "polecenie": "p"}, url="u", wiki=None, obrazy=False,
                          bez_myslenia=True)
