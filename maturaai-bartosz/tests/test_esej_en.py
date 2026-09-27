"""Esej e8 po angielsku (matura/esej_en.py): pamięć tłumaczeń kart, słowniczek nazw, materiał, akapity, skład.

Co to jest: testy bez GPU i sieci; czat modelu i tłumacz Marian podmieniane atrapami.
Po co: pilnują tłumaczenia kart linia w linię (numeracja, dzielenie partii przy złym formacie), słowniczka nazw
PL↔EN i składu eseju (rama po polsku, akapity po Marianie). Przykłady tylko z arkuszy spoza oceny (2015-2022).
Co zrobić: uv run pytest -q tests/test_esej_en.py
"""
import re

from matura import esej_en, jezyk


class CzatNumerowany:
    """Atrapa tłumacza: odpowiada numerowanymi liniami „EN(<tekst>)”; `psuj` = liczba pierwszych wywołań z
    brakującą ostatnią linią (zły format → dzielenie partii)."""

    def __init__(self, psuj: int = 0):
        self.psuj, self.wywolania = psuj, []

    def __call__(self, url, system, user, obr=None, **kw):
        linie = re.findall(r"^(\d+)\. (.*)$", user, re.M)
        self.wywolania.append(len(linie))
        out = [f"{n}. EN({t})" for n, t in linie]
        if self.psuj > 0 and len(out) > 1:
            self.psuj -= 1
            out = out[:-1]
        return "\n".join(out), 0.01


def test_tlumacz_linie_numerowane_i_cache(tmp_path):
    c = jezyk.Cache(tmp_path / "tm.jsonl")
    czat = CzatNumerowany()
    linie = ["Reformacja osłabiła Kościół katolicki.", "Pokój westfalski", "Księstwo Warszawskie"]
    wyn = esej_en.tlumacz_linie(linie, c, url="x", czat_fn=czat, partia=10)
    assert wyn == {t: f"EN({t})" for t in linie}
    assert czat.wywolania == [3]
    assert esej_en.tlumacz_linie(linie, c, url="x", czat_fn=czat) == wyn     # z cache, bez wywołań
    assert czat.wywolania == [3]


def test_tlumacz_linie_zly_format_dzieli_partie(tmp_path):
    c = jezyk.Cache(tmp_path / "tm.jsonl")
    czat = CzatNumerowany(psuj=1)
    linie = [f"Fakt numer {i}." for i in range(4)]
    wyn = esej_en.tlumacz_linie(linie, c, url="x", czat_fn=czat, partia=4)
    assert wyn == {t: f"EN({t})" for t in linie}
    assert czat.wywolania[0] == 4 and len(czat.wywolania) >= 3               # 4 → 2 + 2


def test_slowniczek_z_nazw_kart_najdluzsze_najpierw():
    tm = {"Księstwo Warszawskie": "Duchy of Warsaw", "Warszawa": "Warsaw", "pokój westfalski": "Peace of Westphalia",
          "x": "x"}
    sl = esej_en.slowniczek(tm, ["Księstwo Warszawskie", "Warszawa", "pokój westfalski", "x"])
    assert sl[0] == ("Księstwo Warszawskie", "Duchy of Warsaw")
    assert ("x", "x") not in sl                                                # identyczne i krótkie pomijane


def test_slowniczek_pewny_tylko_nazwy_potwierdzone_w_fakcie():
    tm = {"Napoleon utworzył Księstwo Warszawskie.": "Napoleon created the Duchy of Warsaw.",
          "Księstwo Warszawskie": "Duchy of Warsaw", "sobór": "Council", "Sobór trydencki zwołano.": "The Council met.",
          "Hugo Kołłątaj": "Hugo Kollontay", "Kołłątaj pisał.": "Kołłątaj wrote."}
    karty = [{"fakt": "Napoleon utworzył Księstwo Warszawskie.", "termin": "Księstwo Warszawskie", "postac": ""},
             {"fakt": "Sobór trydencki zwołano.", "termin": "sobór", "postac": ""},
             {"fakt": "Kołłątaj pisał.", "termin": "", "postac": "Hugo Kołłątaj"}]
    assert esej_en.slowniczek_pewny(tm, karty) == [("Księstwo Warszawskie", "Duchy of Warsaw")]


def test_nazwa_obecna_z_odmiana():
    assert esej_en.nazwa_obecna("pokój westfalski", "W 1648 roku pokój Westfalii zakończył wojnę.")
    assert not esej_en.nazwa_obecna("sobór trydencki", "Rada Trentu wzmocniła Kościół katolicki.")


class MarianAtrapa:
    """EN→PL: słownik zdań; zdanie z polską nazwą wstawioną przez ochronę nazw → „PL[...]” z tą nazwą."""

    def __init__(self, mapa):
        self.mapa, self.wywolania = mapa, []

    def tlumacz(self, teksty, kierunek, konteksty=None):
        self.wywolania.append(list(teksty))
        return [self.mapa.get(t, f"PL[{t}]") for t in teksty]


def test_na_polski_poprawia_tylko_przekrecona_nazwe():
    m = MarianAtrapa({"The Council of Trent strengthened the Church.": "Rada Trentu wzmocniła Kościół.",
                      "The Duchy of Warsaw was created in 1807.": "Księstwo Warszawskie powstało w 1807 roku."})
    sl = [("Księstwo Warszawskie", "Duchy of Warsaw"), ("sobór trydencki", "Council of Trent")]
    pl, pop = esej_en.na_polski("The Council of Trent strengthened the Church. The Duchy of Warsaw was created in 1807.",
                                m, sl)
    assert pop == ["sobór trydencki"]
    assert "PL[The sobór trydencki strengthened the Church.]" in pl and "Księstwo Warszawskie powstało" in pl


def test_regula_porownania_i_podmiot():
    assert esej_en.wymaga_porownania("Reformacja w największym stopniu osłabiła Kościół katolicki w XVI wieku.")
    assert esej_en.wymaga_porownania("Polityka Napoleona Bonaparte przede wszystkim służyła interesom Francji.")
    assert esej_en.wymaga_porownania("Aleksander Wielki był najwybitniejszym wodzem starożytności.")
    assert esej_en.wymaga_porownania("Unia lubelska przyniosła Rzeczypospolitej bardziej korzyści niż straty.") is False
    assert esej_en.wymaga_porownania("Dla Wielkiej Brytanii handel był ważniejszy bardziej niż podboje.")
    assert not esej_en.wymaga_porownania("Reformacja zarówno osłabiła, jak i wzmocniła Kościół katolicki.")
    assert esej_en.podmiot_tezy("Polityka Napoleona Bonaparte przede wszystkim służyła Francji.") == [
        "Polityka", "Napoleona", "Bonaparte"]
    assert esej_en.podmiot_tezy("Aleksander Wielki był najwybitniejszym wodzem starożytności.") == ["Aleksander", "Wielki"]
    assert esej_en.podmiot_tezy("Zjednoczenie Niemiec zawdzięczano przede wszystkim Ottonowi von Bismarckowi.") == [
        "Ottonowi", "Bismarckowi"]
    assert esej_en.podmiot_tezy("Najważniejszym skutkiem wypraw krzyżowych był rozwój handlu.") == []
    assert esej_en.podmiot_tezy("Reformacja w największym stopniu osłabiła Kościół katolicki.") == []


def test_linia_faktu_z_podmiotem_i_data():
    tm = {"Napoleon utworzył Księstwo Warszawskie.": "Napoleon created the Duchy of Warsaw.",
          "Napoleon Bonaparte": "Napoleon Bonaparte"}
    c = {"fakt": "Napoleon utworzył Księstwo Warszawskie.", "postac": "Napoleon Bonaparte", "data": "1807"}
    assert esej_en.linia_faktu(c, tm) == "- Napoleon Bonaparte, 1807: Napoleon created the Duchy of Warsaw."
    assert esej_en.linia_faktu({"fakt": "Fakt bez tłumaczenia.", "postac": "", "data": ""}, tm) == "- Fakt bez tłumaczenia."


import itertools  # noqa: E402

_SYL = ["ka", "to", "mi", "re", "su", "lo", "pe", "ni", "da", "wo", "gu", "ba"]
_SLOWA = ["".join(s) for s in itertools.product(_SYL, repeat=3)]


class MarianPolski:
    """Atrapa Mariana: każde zdanie EN → inne „polskie” zdanie z 14 niepowtarzalnych słów (bez fałszywych powtórzeń)."""

    def __init__(self):
        self.i = 0

    def tlumacz(self, teksty, kierunek, konteksty=None):
        out = []
        for t in teksty:
            if kierunek == "pl-en":
                out.append(f"EN: {t}")
                continue
            sl = _SLOWA[self.i * 14:(self.i + 1) * 14]
            self.i += 1
            out.append(sl[0].capitalize() + " " + " ".join(sl[1:]) + ".")
        return out


class CzatAkapitow:
    def __init__(self):
        self.systemy, self.prompty = [], []

    def __call__(self, url, system, user, obr=None, **kw):
        self.systemy.append(system); self.prompty.append(user)
        return " ".join(f"Sentence number {i} about the facts." for i in range(9)), 0.01


class KartyAtrapa:
    k = [{"fakt": f"Fakt {i} o polityce Napoleona.", "data": "1807", "postac": "Napoleon Bonaparte", "termin": "",
          "aspekt": "polityczny", "tytul": "Napoleon", "dzial": "Epoka napoleońska"} for i in range(12)] + \
        [{"fakt": f"Fakt {i} o Metternichu.", "data": "1815", "postac": "Klemens von Metternich", "termin": "",
          "aspekt": "polityczny", "tytul": "Kongres", "dzial": "Epoka napoleońska"} for i in range(6)]
    epoka_dzialu = {}

    def szukaj(self, zap, n=8, epoka=None):
        return self.k[:n] if "Napoleon" in zap else self.k[12:12 + n]


TEMAT_NAPOLEON = ("Polityka Napoleona Bonaparte przede wszystkim służyła interesom Francji. Zajmij stanowisko wobec "
                  "powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji aspekty: polityczny, militarny i "
                  "gospodarczy.")


def test_napisz_e8_rama_porownanie_i_dlugosc():
    czat = CzatAkapitow()
    tekst, sek, meta = esej_en.napisz_e8({"temat": TEMAT_NAPOLEON}, url="x", karty=KartyAtrapa(), baza_hasel=None,
                                         marian=MarianPolski(), tm={}, hasla_en={}, slownik=[], czat_fn=czat)
    akapity = tekst.split("\n\n")
    assert akapity[0].startswith("Teza tematu brzmi") and "Zgadzam się z tą tezą." in akapity[0]
    assert len(akapity) == 6                                     # wstęp, 3 aspekty, porównanie, zakończenie
    assert akapity[1].startswith("Po pierwsze, tezę potwierdza aspekt polityczny.")
    assert akapity[4].startswith("Po czwarte, trafność tezy potwierdza porównanie")
    assert meta["porownanie"] and meta["pozycje_en"] == ["political", "military", "economic", "comparison"]
    assert meta["slow"] >= 300 and all(not w for w in meta["wady_akapitow"])
    assert all("Attribute each fact only to the person or state named" in s for s in czat.systemy)
    assert "COMPARE" in czat.systemy[3] and "COMPARE" not in czat.systemy[0]
    assert "- Napoleon Bonaparte, 1807: Fakt" in czat.prompty[0]      # materiał z podmiotem i datą
    assert "Fakt 0 o Metternichu" in czat.prompty[3]                  # porównanie: karty bez podmiotu tezy
    assert "o polityce Napoleona" not in czat.prompty[3]


def test_napisz_e8_bez_porownania():
    temat = ("Reformacja zarówno osłabiła, jak i wzmocniła Kościół katolicki. Zajmij stanowisko wobec powyższej tezy i "
             "je uzasadnij, uwzględniając w swojej argumentacji aspekty: polityczny, społeczny i kulturowy.")
    tekst, _, meta = esej_en.napisz_e8({"temat": temat}, url="x", karty=KartyAtrapa(), baza_hasel=None,
                                       marian=MarianPolski(), tm={}, hasla_en={}, slownik=[], czat_fn=CzatAkapitow())
    assert not meta["porownanie"] and len(tekst.split("\n\n")) == 5


def test_wykoncz_e8_zostawia_ostatnie_zdanie():
    zd = [" ".join(_SLOWA[i * 60:(i + 1) * 60]) + "." for i in range(4)] + ["To pokazuje, że teza jest słuszna."]
    wyn = esej_en.wykoncz_e8({"zdania": zd, "sprzeczne": []}, max_slow=150)
    assert wyn[-1] == "To pokazuje, że teza jest słuszna." and len(wyn) == 3


def test_tlumaczenie_z_inna_data_albo_miesiacem_odrzucone():
    pl = "Protesty wybuchły 25 czerwca 1976 roku w Radomiu i Płocku."
    assert not esej_en._wiarygodne(pl, "Protests broke out on May 25, 1976 in Radom and Płock.")
    assert not esej_en._wiarygodne(pl, "Protests broke out on June 25, 1975 in Radom and Płock.")
    assert esej_en._wiarygodne(pl, "Protests broke out on June 25, 1976 in Radom and Płock.")


def test_porownanie_tylko_z_dzialow_akapitow_aspektow():
    class Karty(KartyAtrapa):
        k = KartyAtrapa.k + [{"fakt": "Fakt z innej epoki o traktacie.", "data": "1917", "postac": "Raymond Poincaré",
                              "termin": "", "aspekt": "polityczny", "tytul": "Armia", "dzial": "I wojna światowa"}]

        def szukaj(self, zap, n=8, epoka=None):
            return self.k[:n] if "Napoleon" in zap else [self.k[-1]] + self.k[12:12 + n]

    m = esej_en.material_e8(Karty(), None, "Polityka Napoleona Bonaparte przede wszystkim służyła Francji.", "porównanie",
                            tm={}, hasla_en={}, porownanie=True, dzialy=["Epoka napoleońska"])
    assert m["fakty"] and all(c["dzial"] == "Epoka napoleońska" for c in m["fakty"])
    assert not any("Napoleona" in c["fakt"] for c in m["fakty"])


def test_nazwa_w_tezie_piec_liter_i_bez_historiografii():
    assert not esej_en.nazwa_w_tezie("Poliklet", "Polityka Napoleona Bonaparte służyła Francji.")
    assert esej_en.nazwa_w_tezie("Napoleon Bonaparte", "Polityka Napoleona Bonaparte służyła Francji.")

    class Karty(KartyAtrapa):
        k = [{"fakt": "Historyk ocenił politykę Napoleona.", "data": "", "postac": "Józef Szujski", "termin": "",
              "aspekt": "polityczny", "tytul": "x", "dzial": "Historia jako nauka"}] + KartyAtrapa.k

    m = esej_en.material_e8(Karty(), None, "Polityka Napoleona Bonaparte przede wszystkim służyła Francji.",
                            "polityczny", tm={}, hasla_en={})
    assert m["fakty"] and all(c["dzial"] != "Historia jako nauka" for c in m["fakty"])
