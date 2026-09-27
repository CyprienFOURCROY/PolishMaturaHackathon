"""Obrazy → tekst (matura/obrazy.py): przypisanie obrazów do zadań, cache OCR/opisu, blok do promptu, h0_ocr.

Bez GPU i sieci: OCR i czat są podmieniane (monkeypatch). Testy na arkuszach CKE pomijane, gdy brak danych.
"""
import json
from pathlib import Path

import pytest
from PIL import Image

from matura import devset, harness, obrazy

potrzebne_cke = pytest.mark.skipif(not (devset.JSON / "historia-2025-maj.json").exists()
                                   or not (obrazy.KAT_CKE / "historia-2025-maj-arkusz.pdf").exists(),
                                   reason="brak danych CKE (json + pdf)")


@pytest.fixture
def cache_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(obrazy, "CACHE", tmp_path / "opisy.jsonl")
    monkeypatch.setattr(obrazy, "_CACHE", None)
    return tmp_path / "opisy.jsonl"


def _obraz(p: Path, kolor=(200, 30, 30)) -> str:
    im = Image.new("RGB", (40, 30), "white")
    for x in range(10, 30):
        im.putpixel((x, 15), kolor)
    im.save(p)
    return str(p)


def _zad(id_, zrodla, polecenie="Zadanie 9.1. (0–1)\nPodaj nazwę.", obrazy_=(), **kw):
    return {"id": id_, "nr_zadania": id_.split("-")[-1], "nr_glowny": id_.split("-")[-1].split(".")[0],
            "zrodla_tekst": zrodla, "polecenie": polecenie, "obrazy": list(obrazy_), "esej": False,
            "typ": "otwarte", "pkt_max": 1, **kw}


# ---------------------------------------------------------------- przypisanie na arkuszach CKE

@potrzebne_cke
def test_2025_maj_7_1_nie_dostaje_planu_miasta_z_zadania_6():
    z = {z["id"]: z for z in devset.wczytaj(["2025-maj"])}
    przed = [Path(o).name for o in z["2025-maj-7.1"]["obrazy"]]
    assert "str10_1.png" in przed  # błąd devset: obrazy ze strony
    po = [Path(o).name for o in obrazy.obrazy_zadania(z["2025-maj-7.1"])]
    assert "str10_1.png" not in po and "str11_0.png" in po  # zostaje mapa ze źródła 2.
    assert [Path(o).name for o in obrazy.obrazy_zadania(z["2025-maj-6"])] == ["str10_1.png"]
    assert {il["zrodlo"] for il in obrazy.ilustracje(z["2025-maj-7.1"])} == {2}


@potrzebne_cke
def test_fotografie_zachowuja_obrazy_a_zadanie_tekstowe_je_traci():
    z = {z["id"]: z for z in devset.wczytaj(["2025-maj"])}
    assert sorted(Path(o).name for o in obrazy.obrazy_zadania(z["2025-maj-1.1"])) == ["str04_1.png", "str04_2.png", "str04_3.png"]
    assert obrazy.obrazy_zadania(z["2025-maj-2"]) == []   # „Fragment tekstu religijnego”: fotografia z zadania 3 nie jego
    assert obrazy.obrazy_zadania(z["2025-maj-22"]) == []  # plakat należy do zadania 23
    il = obrazy.ilustracje(z["2025-maj-1.1"])
    assert all(i["podpis"] == "Źródło 1. Fotografie naczyń" for i in il) and [i["i"] for i in il] == [1, 2, 3]


@potrzebne_cke
def test_jednolite_ramki_pomijane():
    if not (obrazy.KAT_CKE / "historia-2024-maj-arkusz.pdf").exists():
        pytest.skip("brak PDF 2024-maj")
    z = {z["id"]: z for z in devset.wczytaj(["2024-maj"])}
    assert z["2024-maj-11.1"]["obrazy"] and obrazy.obrazy_zadania(z["2024-maj-11.1"]) == []  # czarne ramki tablicy


# ---------------------------------------------------------------- heurystyka słów (bez PDF)

def test_zrodla_rozpoznaje_wizualne_i_nienumerowane():
    z = _zad("2099-maj-7.1", "Zadanie 7.\nŹródło 1. Fragment dokumentu\nTreść…\nŹródło 2. Mapa\n\nNa podstawie: x")
    assert [(s["nr"], s["wizualne"]) for s in obrazy.zrodla(z)] == [(1, False), (2, True)]
    z = _zad("2099-maj-6", "", polecenie="Zadanie 6. (0–1)\nW. Dobrowolski – Armia pokoju\nhttps://encyklopediateatru.pl\nPodaj…")
    assert obrazy.zrodla(z) == [{"nr": None, "podpis": "W. Dobrowolski – Armia pokoju", "wizualne": True}]


def test_heurystyka_slow_nie_daje_obrazu_zadaniu_tekstowemu(tmp_path):
    img = _obraz(tmp_path / "str10_1.png")
    foto = _zad("2099-maj-3.1", "Zadanie 3.\nFotografie naczyń\nNa podstawie: https://x", obrazy_=[img])
    tekst = _zad("2099-maj-2", "", polecenie="Zadanie 2. (0–1)\nŹródło 1. Fragment tekstu religijnego\nCzekając…", obrazy_=[img])
    w = obrazy._przypisz_slowami([foto, tekst])
    assert [i["rel"] for i in w["2099-maj-3.1"]] == [img] and w["2099-maj-2"] == []
    samotny = _zad("2099-maj-4", "", polecenie="Zadanie 4. (0–1)\nTabela\nDane…", obrazy_=[img])
    assert [i["rel"] for i in obrazy._przypisz_slowami([samotny])["2099-maj-4"]] == [img]  # nikt inny go nie chce


# ---------------------------------------------------------------- OCR, cache, blok

def test_kolejnosc_czytania_wiersze_od_gory_i_od_lewej():
    det = [{"t": "prawy", "p": 0.9, "b": [200, 12, 260, 30]}, {"t": "dół", "p": 0.9, "b": [0, 100, 40, 120]},
           {"t": "lewy", "p": 0.9, "b": [10, 10, 60, 28]}, {"t": "śmieć", "p": 0.1, "b": [0, 50, 10, 60]}]
    assert obrazy.napisy(det) == ["lewy", "prawy", "dół"]


def test_cache_nie_liczy_drugi_raz(tmp_path, cache_tmp, monkeypatch):
    img = _obraz(tmp_path / "a.png")
    wywolania = {"ocr": 0, "opis": 0}

    def ocr(p):
        wywolania["ocr"] += 1
        return [{"t": "GDAŃSK", "p": 0.95, "b": [0, 0, 10, 5]}]

    def opis(url, p, **kw):
        wywolania["opis"] += 1
        return "Mapa.", "1. Mapa.", 0.1

    monkeypatch.setattr(obrazy, "ocr_surowy", ocr)
    monkeypatch.setattr(obrazy, "opisz_obraz", opis)
    for _ in range(2):
        assert obrazy.policz_ocr(img)["wynik"][0]["t"] == "GDAŃSK"
        assert obrazy.policz_opis(img, "http://nic")["wynik"] == "Mapa."
    assert wywolania == {"ocr": 1, "opis": 1}
    monkeypatch.setattr(obrazy, "_CACHE", None)  # nowy proces: czyta z pliku, nie liczy
    obrazy.policz_ocr(img)
    assert wywolania["ocr"] == 1 and len(cache_tmp.read_text(encoding="utf-8").splitlines()) == 2
    _obraz(tmp_path / "a.png", kolor=(0, 0, 250))  # inny plik (sha1) → nowy wpis
    obrazy.policz_ocr(img)
    assert wywolania["ocr"] == 2


def _fake_ilustracje(img):
    return [{"sciezka": img, "rel": img, "zrodlo": 2, "podpis": "Źródło 2. Mapa", "i": 1, "n": 2},
            {"sciezka": img, "rel": img, "zrodlo": None, "podpis": "Plan miasta", "i": 1, "n": 1}]


def test_blok_opisu_format(tmp_path, cache_tmp, monkeypatch):
    img = _obraz(tmp_path / "m.png")
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: _fake_ilustracje(img))
    obrazy._zapisz(img, "ocr", [{"t": "Królewiec", "p": 0.99, "b": [50, 0, 90, 10]}, {"t": "M. Bałtyckie", "p": 0.9, "b": [0, 0, 40, 10]},
                                {"t": "xq", "p": 0.2, "b": [0, 50, 5, 55]}], 0.1)
    obrazy._zapisz(img, "opis", "Typ źródła: mapa; granice państwa zakonnego.", 0.5)
    b = obrazy.blok_opisu({"id": "x"}, "ocr").splitlines()
    assert b[0] == obrazy.NAGLOWEK_BLOKU
    assert b[1] == "OPIS ILUSTRACJI (Źródło 2. Mapa, ilustracja 1 z 2): napisy: M. Bałtyckie | Królewiec"
    assert b[2] == "OPIS ILUSTRACJI (źródło: Plan miasta): napisy: M. Bałtyckie | Królewiec"
    b2 = obrazy.blok_opisu({"id": "x"}, "ocr_opis").splitlines()
    assert b2[1].endswith("napisy: M. Bałtyckie | Królewiec; opis: Typ źródła: mapa; granice państwa zakonnego.")


def test_blok_bez_cache_zglasza_blad(tmp_path, cache_tmp, monkeypatch):
    img = _obraz(tmp_path / "m.png")
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: _fake_ilustracje(img))
    with pytest.raises(obrazy.BrakOpisu):
        obrazy.blok_opisu({"id": "x"}, "ocr")


def test_czysc_opis_usuwa_numeracje_markdown_puste_punkty_i_powtorzenia():
    surowy = ("1. **Typ źródła** – moneta.\n2. Kto lub co jest przedstawione: kobieta z tarczą.\n"
              "3. Widoczne napisy: „CAMINIEC”.\n4. Symbole: tarcza.\n4. Symbole: tarcza.\n"
              "5. Tylko dla mapy lub planu: nie ma mapy ani planu.")
    t = obrazy.czysc_opis(surowy)
    assert "**" not in t and "1." not in t and "planu" not in t and t.count("tarcza.") == 1
    assert t == "Typ: moneta. Przedstawia: kobieta z tarczą. Napisy: „CAMINIEC”. Symbole: tarcza."
    echo = "1. Typ źródła (fotografia, mapa, plan, karykatura, plakat, rysunek) – medal\nPominąłbym punkt 5."
    assert obrazy.czysc_opis(echo) == "Typ: medal"
    assert len(obrazy.czysc_opis("słowo " * 500).split()) <= obrazy.MAX_SLOW_OPISU + 1


# ---------------------------------------------------------------- harness

def _przechwyc(monkeypatch):
    wolania = []

    def czat(url, system, user, obr=None, **kw):
        wolania.append((system, user, obr))
        return "odp", 0.1

    monkeypatch.setattr(harness, "czat", czat)
    return wolania


def test_h0_ocr_bez_ilustracji_to_dokladnie_h0(monkeypatch):
    w = _przechwyc(monkeypatch)
    z = _zad("synt-1", "Fragment kroniki\nTreść…", polecenie="Podaj nazwę dynastii.")
    for k in ("h0", "h0_ocr", "h0_opis"):
        harness.odpowiedz(k, z, url="u", wiki=None, obrazy=False, bez_myslenia=True)
    assert w[0] == w[1] == w[2]


def test_h0_ocr_wstawia_blok_przed_zadaniem_i_nie_wysyla_obrazu(tmp_path, cache_tmp, monkeypatch):
    img = _obraz(tmp_path / "m.png")
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: _fake_ilustracje(img)[:1])
    obrazy._zapisz(img, "ocr", [{"t": "Toruń", "p": 0.97, "b": [0, 0, 10, 5]}], 0.1)
    w = _przechwyc(monkeypatch)
    z = _zad("2099-maj-7.1", "Źródło 2. Mapa", polecenie="Rozstrzygnij, czy…", obrazy_=[img])
    t, s, meta = harness.odpowiedz("h0_ocr", z, url="u", wiki=None, obrazy=True, bez_myslenia=True)
    harness.odpowiedz("h0", z, url="u", wiki=None, obrazy=True, bez_myslenia=True)
    (sys_ocr, user_ocr, obr_ocr), (sys_h0, user_h0, obr_h0) = w
    assert sys_ocr == sys_h0 and obr_ocr is None and obr_h0 == [img]
    blok = obrazy.NAGLOWEK_BLOKU + "\nOPIS ILUSTRACJI (Źródło 2. Mapa, ilustracja 1 z 2): napisy: Toruń\n\n"
    assert user_ocr == blok + user_h0 and meta["ilustracje_opisane"] == 1
