"""Sufit eseju (krok 1b): parametr `material_fn` w esej.napisz_e5 i scripts/fakty_eseju_wzorcowe.py.

Co to jest: testy bez GPU i sieci; czat modelu i Claude podmieniane atrapami.
Po co: pilnują, że wariant `e5_fakty` podaje modelowi fakty wzorcowe właściwego członu tematu (a nie karty ani
Wikipedię), że druga generacja `<konfig>_g2` idzie tym samym potokiem co konfiguracja bazowa i że kontrola faktów
wyłapuje zdania rozstrzygające tezę. Co zrobić: uv run pytest -q tests/test_fakty_eseju.py
"""
import importlib.util
from pathlib import Path

import pytest

from matura import esej, harness, noc

_p = Path(__file__).resolve().parent.parent / "scripts" / "fakty_eseju_wzorcowe.py"
_spec = importlib.util.spec_from_file_location("fakty_eseju_wzorcowe", _p)
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)

TEMAT = ("Władysław Jagiełło był najwybitniejszym władcą Polski z dynastii Jagiellonów. Zajmij stanowisko wobec "
         "powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji aspekty: militarny, ustrojowy i "
         "społeczno-gospodarczy.")
TEMAT_WYBOR = ("Rok 1956 był przełomem w systemie komunistycznym. Zajmij stanowisko wobec powyższej tezy i je "
               "uzasadnij, uwzględniając w swojej argumentacji wydarzenia z trzech wybranych państw bloku komunistycznego.")


def _g(czlon, *fakty):
    return {"czlon": czlon, "fakty": [{"fakt": f, "data": d} for f, d in fakty]}


GRUPY = [_g("militarny", ("Wojska polsko-litewskie pokonały Krzyżaków pod Grunwaldem", "1410"),
            ("Pokój toruński przywrócił Polsce ziemię dobrzyńską", "1411"),
            ("Władysław Warneńczyk zginął pod Warną", "1444")),
         _g("ustrojowy", ("Przywilej jedlneńsko-krakowski wprowadził nietykalność osobistą szlachty", "1430-1433"),
            ("Unia horodelska przyjęła bojarów litewskich do herbów polskich", "1413"),
            ("Kazimierz Jagiellończyk wydał przywileje cerekwicko-nieszawskie", "1454")),
         _g("społeczno-gospodarczy", ("Statut warcki pozwolił szlachcie wykupić sołectwa", "1423"),
            ("Władysław Jagiełło i Jadwiga odnowili Akademię Krakowską", "1400"),
            ("Pokój toruński przyłączył Pomorze Gdańskie do Polski", "1466"))]
GRUPY_WYBOR = [_g("ZSRR", ("Nikita Chruszczow wygłosił tajny referat na XX Zjeździe KPZR", "1956"),
                  ("Referat potępił kult jednostki Józefa Stalina", "1956"), ("Józef Stalin zmarł", "1953")),
               _g("Polska", ("Robotnicy Poznania wystąpili przeciw władzy", "1956"),
                  ("Władysław Gomułka został I sekretarzem KC PZPR", "1956"), ("Prymas Wyszyński wrócił do Warszawy", "1956")),
               _g("Węgry", ("Imre Nagy ogłosił wystąpienie Węgier z Układu Warszawskiego", "1956"),
                  ("Armia Radziecka stłumiła powstanie węgierskie", "1956"), ("János Kádár objął władzę", "1956"))]


class Czat:
    """Atrapa modelu: każda odpowiedź to 8 zdań po 17 słów (akapit bez wad); pierwsza może być podana."""

    def __init__(self, pierwsza=None):
        self.wolania, self.pierwsza, self.n = [], pierwsza, 0

    def __call__(self, url, system, user, obr=None, **kw):
        self.wolania.append(user)
        if self.pierwsza is not None and len(self.wolania) == 1:
            return self.pierwsza, 0.1
        zd = []
        for _ in range(8):
            self.n += 1
            zd.append(" ".join(f"slowo{self.n}x{i}" for i in range(16)).capitalize() + " koniec.")
        return " ".join(zd), 0.1


def _bez_material_e5(monkeypatch):
    def nie(*a, **k):
        raise AssertionError("material_e5 nie powinno być wołane przy material_fn")
    monkeypatch.setattr(esej, "material_e5", nie)


# ---------------------------------------------------------------- esej.napisz_e5(material_fn=...)

def test_material_fn_zastepuje_material_e5_dla_kazdego_aspektu(monkeypatch):
    _bez_material_e5(monkeypatch)
    c = Czat()
    t, _, m = esej.napisz_e5({"id": "x", "temat": TEMAT, "esej": True}, url=None, czat_fn=c, stanowisko="zgadzam",
                             material_fn=lambda *a, **k: fw.material_fakty(GRUPY, *a, **k))
    assert m["aspekty"] == ["militarny", "ustrojowy", "społeczno-gospodarczy"] and m["slow"] >= 300
    assert "Grunwaldem" in c.wolania[0] and "horodelska" not in c.wolania[0]      # akapit 1: tylko fakty militarne
    assert "horodelska" in c.wolania[1] and "Grunwaldem" not in c.wolania[1]
    assert "warcki" in c.wolania[2] and m["wiki"] == 0 and m["karty"] == 9


def test_material_fn_w_kroku_wyboru_przykladow(monkeypatch):
    _bez_material_e5(monkeypatch)
    c = Czat(pierwsza="Polska, Węgry, ZSRR")
    t, _, m = esej.napisz_e5({"id": "x", "temat": TEMAT_WYBOR, "esej": True}, url=None, czat_fn=c,
                             stanowisko="zgadzam", material_fn=lambda *a, **k: fw.material_fakty(GRUPY_WYBOR, *a, **k))
    assert all(f"FAKTY PEWNE ({g})" in c.wolania[0] for g in ("ZSRR", "Polska", "Węgry"))  # wybór: wszystkie grupy
    assert m["aspekty"] == ["Polska", "Węgry", "ZSRR"]
    assert "Gomułka" in c.wolania[1] and "Nagy" not in c.wolania[1] and "Nagy" in c.wolania[2]


# ---------------------------------------------------------------- materiał i dopasowanie członu

def test_grupa_dla_nazwy_i_rdzenia():
    grupy = [_g("Czechosłowacja", ("a", "")), _g("Związek Radziecki", ("b", "")), _g("Polska", ("c", ""))]
    assert fw.grupa_dla(grupy, "Polska")["czlon"] == "Polska"
    assert fw.grupa_dla(grupy, "Czechosłowen")["czlon"] == "Czechosłowacja"
    assert fw.grupa_dla(grupy, "Związku Radzieckiego")["czlon"] == "Związek Radziecki"
    assert fw.grupa_dla(grupy, "Rumunia") is None


def test_material_fakty_tekst_ciagly_z_datami_i_bez_dublowania_roku():
    m = fw.material_fakty(GRUPY, None, None, "teza", "militarny")
    assert m["tekst"].startswith("FAKTY PEWNE") and "\n-" not in m["tekst"]
    assert "pod Grunwaldem (1410)." in m["tekst"] and m["fragmenty"] == [] and len(m["fakty"]) == 3
    m2 = fw.material_fakty([_g("militarny", ("Bitwa pod Grunwaldem w 1410 roku", "1410"))], None, None, "t", "militarny")
    assert "(1410)" not in m2["tekst"]
    wszystkie = fw.material_fakty(GRUPY_WYBOR, None, None, "t", "państw bloku komunistycznego", element=True)
    assert len(wszystkie["fakty"]) == 9 and {f["postac"] for f in wszystkie["fakty"]} == {"ZSRR", "Polska", "Węgry"}


def test_sprawdz_wylapuje_rozstrzygniecie_tezy_i_zle_grupy():
    assert fw.sprawdz(GRUPY + [_g("militarny", ("x", ""))], TEMAT)  # 4 grupy
    dobre = [_g(g["czlon"], *[(f["fakt"], f["data"]) for f in g["fakty"]] + [("Fakt dodatkowy numer jeden", "")])
             for g in GRUPY]
    dobre[0]["fakty"].append({"fakt": "Kolejny fakt militarny", "data": ""})
    assert fw.sprawdz(dobre, TEMAT) == []                                          # 3 grupy, 13 faktów
    zle = [dict(g, fakty=list(g["fakty"])) for g in dobre]
    zle[0]["fakty"].append({"fakt": "Grunwald potwierdza tezę o wielkości Jagiełły", "data": "1410"})
    zle[1]["fakty"].append({"fakt": "Po pierwsze, unia z Litwą była sukcesem", "data": ""})
    uw = fw.sprawdz(zle, TEMAT)
    assert any("potwierdza" in u for u in uw) and any("Po pierwsze" in u for u in uw)
    zle_czlony = [dict(g, czlon="gospodarczy") if i == 2 else g for i, g in enumerate(dobre)]
    assert any("człony" in u for u in fw.sprawdz(zle_czlony, TEMAT))
    assert fw.sprawdz(GRUPY_WYBOR + [], TEMAT_WYBOR) == ["faktów 9 poza 10-16"]    # wybór: dowolne nazwy członów


def test_fakty_tematu_ponawia_przy_problemach_i_wybiera_lepsza():
    wolania = []

    def claude(prompt, schemat, effort):
        wolania.append((prompt, schemat, effort))
        return {"grupy": GRUPY[:2]} if len(wolania) == 1 else {"grupy": GRUPY + [_g("militarny", ("x", ""))]}

    w = fw.fakty_tematu({"id": "2025-maj-25-t1", "temat": TEMAT}, "low", claude_fn=claude)
    assert len(wolania) == 2 and w["id"] == "2025-maj-25-t1" and len(w["grupy"]) == 2  # obie złe: mniej problemów
    assert TEMAT in wolania[0][0] and "„militarny”" in wolania[0][0] and wolania[0][2] == "low"
    assert wolania[0][1]["properties"]["grupy"]["items"]["properties"]["czlon"]["enum"] == [
        "militarny", "ustrojowy", "społeczno-gospodarczy"]
    assert "nigdy egzamin ani trening" in w["zrodlo"]


# ---------------------------------------------------------------- rejestracja konfiguracji (jak pomiar_wzorzec.py)

def test_rejestracja_e5_fakty_i_drugiej_generacji(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "odpowiedz", harness.odpowiedz)          # przywrócenie po teście
    monkeypatch.setattr(noc, "KONFIGI_Z_WIKI", set(noc.KONFIGI_Z_WIKI))
    monkeypatch.setattr(noc, "KONFIGI_Z_KARTAMI", set(noc.KONFIGI_Z_KARTAMI))
    monkeypatch.setattr(fw, "wczytaj", lambda p=None: {"2025-maj-25-t1": GRUPY})
    wolania = []
    monkeypatch.setattr(harness.esej_mod, "napisz_e5", lambda z, **kw: (wolania.append(kw), ("esej", 1.0, {}))[1])
    fw.rejestruj(["e5_zgadzam", "e5_zgadzam_g2", "e5_fakty", "e5_fakty_g2", "e7_g2"])
    assert {"e5_zgadzam_g2", "e7_g2"} <= noc.KONFIGI_Z_WIKI and "e5_fakty" not in noc.KONFIGI_Z_WIKI
    z = {"id": "2025-maj-25-t1", "temat": TEMAT, "polecenie": TEMAT, "esej": True}
    for k in ("e5_zgadzam_g2", "e5_fakty", "e5_fakty_g2"):
        assert harness.odpowiedz(k, z, url="u", wiki=None, obrazy=False, bez_myslenia=True, karty=object())[0] == "esej"
    assert wolania[0]["stanowisko"] == "zgadzam" and "material_fn" not in wolania[0]     # _g2 = zwykłe e5_zgadzam
    assert all(w["stanowisko"] == "zgadzam" and w["material_fn"] is not None for w in wolania[1:])
    assert wolania[1]["material_fn"](None, None, "t", "ustrojowy")["fakty"][0]["data"] == "1430-1433"
    with pytest.raises(KeyError):
        harness.odpowiedz("e5_fakty", dict(z, id="inny"), url="u", wiki=None, obrazy=False, bez_myslenia=True)
    with pytest.raises(ValueError):
        harness.odpowiedz("e5_fakty", {"id": "k", "esej": False}, url="u", wiki=None, obrazy=False, bez_myslenia=True)
    assert fw.konfig_bazowy("e7_g2") == "e7" and fw.konfig_bazowy("e5_zgadzam") == "e5_zgadzam"
