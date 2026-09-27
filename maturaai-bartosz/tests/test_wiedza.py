"""Baza akapitów eseju (matura/wiedza.py) i esej e6 (matura/esej.py): kod składa, model tylko wybiera akapit.

Co to jest: testy bez GPU (czat to atrapa), na małej bazie w katalogu tymczasowym.
Po co: e6 ma przenieść akapit bazy do eseju dosłownie (styl nie jest oceniany, przepisywanie wprowadza błędy),
wybrać go numerem od modelu i bez bazy zachować się jak e5. Co zrobić: uv run pytest -q tests/test_wiedza.py
"""
import json

import pytest

from matura import esej, wiedza

TEZA = "Panowanie Kazimierza Wielkiego było okresem wszechstronnego rozwoju państwa polskiego"
TEMAT = (TEZA + ". Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji aspekty: "
         "polityczny, społeczno-gospodarczy i kulturowy.")


def _akapit(i, pozycja, kierunek, tekst):
    return {"id": f"E12-1{i:02d}", "dzial": "Polska w XIV i XV wieku", "epoka": "starozytnosc-sredniowiecze",
            "zrodlo": "test", "teza": TEZA, "podmiot": "Kazimierz Wielki", "okres": "1333-1370", "tryb": "aspekty",
            "pozycja": pozycja, "kierunek": kierunek, "tekst": tekst, "fakty": []}


@pytest.fixture
def baza(tmp_path):
    wiersze = [
        _akapit(1, "polityczny", "potwierdza", "Kazimierz Wielki zawarł pokój w Kaliszu z zakonem krzyżackim w 1343 roku. POL-TAK"),
        _akapit(2, "polityczny", "przeczy", "Kazimierz Wielki zrzekł się Pomorza Gdańskiego w pokoju kaliskim w 1343 roku. POL-NIE"),
        _akapit(3, "społeczno-gospodarczy", "potwierdza", "Kazimierz Wielki lokował wiele miast na prawie magdeburskim. GOS-TAK"),
        _akapit(4, "społeczno-gospodarczy", "przeczy", "Za Kazimierza Wielkiego chłopi nadal byli obciążeni daninami. GOS-NIE"),
        _akapit(5, "kulturowy", "potwierdza", "Kazimierz Wielki założył Akademię Krakowską w 1364 roku. KUL-TAK"),
        _akapit(6, "kulturowy", "przeczy", "Akademia Krakowska po śmierci Kazimierza Wielkiego w 1370 roku upadła. KUL-NIE"),
    ]
    (tmp_path / "akapity_esej_a.jsonl").write_text("\n".join(json.dumps(w, ensure_ascii=False) for w in wiersze) + "\n",
                                                  encoding="utf-8")
    return wiedza.BazaAkapitow(tmp_path)


def test_kandydaci_wolaja_pozycje_i_kierunek(baza):
    k = baza.kandydaci(TEZA, "kulturowy", kierunek="przeczy", n=3)
    assert [c["tekst"][-7:] for c in k[:2]] == ["KUL-NIE", "KUL-TAK"]   # zgodna pozycja, najpierw zgodny kierunek
    assert k[2]["pozycja"] != "kulturowy"                               # w bazie są tylko 2 akapity kulturowe


def test_e6_przenosi_wybrane_akapity_doslownie(baza):
    wolania = []

    def czat(url, system, user, obr=None, **kw):
        wolania.append(user)
        return "1", 0.01                                     # model wybiera pierwszego kandydata

    z = {"id": "2099-maj-26-t1", "temat": TEMAT, "polecenie": TEMAT, "esej": True}
    tekst, _, meta = esej.napisz_e6(z, url="x", baza=baza, czat_fn=czat, stanowisko="nie_zgadzam")
    assert all(s in tekst for s in ("POL-NIE", "GOS-NIE", "KUL-NIE"))
    assert "Nie zgadzam się z tą tezą." in tekst
    assert meta["z_bazy"] == 3 and len(wolania) == 3


def test_e6_bez_bazy_to_e5(tmp_path, monkeypatch):
    pusta = wiedza.BazaAkapitow(tmp_path)
    monkeypatch.setattr(esej, "napisz_e5", lambda z, **kw: ("E5", 0.0, {"e5": True}))
    z = {"id": "2099-maj-26-t1", "temat": TEMAT, "polecenie": TEMAT, "esej": True}
    assert esej.napisz_e6(z, url="x", baza=pusta, czat_fn=None, stanowisko="zgadzam")[0] == "E5"


def test_e6_odrzuca_akapit_o_innym_podmiocie_i_uzupelnia_e5(tmp_path, monkeypatch):
    """Karol Wielki / Justynian (etap Q, 26.09): akapit o innym podmiocie nie trafia do eseju; pozycja dostaje akapit e5."""
    wiersze = [_akapit(1, "polityczny", "przeczy", "Kazimierz Wielki zrzekł się Pomorza Gdańskiego w 1343 roku. POL-NIE"),
               _akapit(6, "kulturowy", "przeczy", "Akademia Krakowska po śmierci Kazimierza Wielkiego w 1370 roku upadła. KUL-NIE"),
               {**_akapit(4, "społeczno-gospodarczy", "przeczy", "Justynian I Wielki podniósł podatki w 532 roku. JUS"),
                "teza": "Panowanie Justyniana było sukcesem Bizancjum", "podmiot": "Justynian I", "okres": "527-565"}]
    (tmp_path / "akapity_esej_b.jsonl").write_text("\n".join(json.dumps(w, ensure_ascii=False) for w in wiersze) + "\n",
                                                  encoding="utf-8")
    monkeypatch.setattr(esej, "napisz_e5", lambda z, **kw: ("W\n\nE5-A\n\nE5-GOS\n\nE5-C\n\nZ", 0.0, {}))
    z = {"id": "2099-maj-26-t1", "temat": TEMAT, "polecenie": TEMAT, "esej": True}
    tekst, _, meta = esej.napisz_e6(z, url="x", baza=wiedza.BazaAkapitow(tmp_path), czat_fn=lambda *a, **k: ("1", 0.0),
                                    stanowisko="nie_zgadzam")
    assert "POL-NIE" in tekst and "KUL-NIE" in tekst and "E5-GOS" in tekst and "JUS" not in tekst
    assert meta["akapity_e5"] == [1] and meta["z_bazy"] == 2


def test_okres_tekstu_z_wiekow_rzymskich():
    assert esej._okres_tekstu("W okresie XI–XII wieku dominowały tendencje") == (1000, 1199)
    assert esej._okres_tekstu("wojny w XVII w. osłabiły państwo") == (1600, 1699)
    assert esej._okres_tekstu("Lata 1871–1914 są niesłusznie określane") == (1871, 1914)


def test_okres_z_lat_akapitu_nie_z_tezy_bazy():
    """Temat „Rok 1956 …” (etap Q): akapit o Sajgonie 1975 z tezą bazy „1945-1991” odpada, Poznań 1956 zostaje."""
    okres = esej._okres_tekstu("Rok 1956 był przełomem w systemie komunistycznym")
    tz = {"rok"}
    baza = {"teza": "Zimna wojna w latach 1945-1991", "okres": "1945-1991", "podmiot": "blok wschodni"}
    assert not esej._pasuje_do_tezy({**baza, "tekst": "W 1975 roku upadł Sajgon, a w 1975 roku Laos."}, tz, okres)
    assert esej._pasuje_do_tezy({**baza, "tekst": "W czerwcu 1956 roku robotnicy Poznania wyszli na ulice."}, tz, okres)
    assert esej._okres_tekstu("Zimna wojna osiągnęła apogeum w latach 50. XX wieku") == (1950, 1959)


def test_e7_material_z_bazy_trafia_do_promptu_akapitu(baza):
    """e7 = e5 z akapitami bazy jako materiałem: model pisze o tezie z arkusza, fakty bierze z bazy (etap Q, 26.09:
    wklejone akapity e6 dotyczyły tez bazy, nie tezy tematu)."""
    m = esej.material_e5(None, None, TEZA, "kulturowy", baza=baza, stanowisko="zgadzam")
    assert "KUL-TAK" in m["tekst"] and m["tekst"].startswith("AKAPITY Z BAZY")
    prompty = []

    def czat(url, system, user, obr=None, **kw):
        prompty.append(user)
        return ("Kazimierz Wielki założył Akademię Krakowską w 1364 roku, co rozwinęło kulturę państwa. " * 5), 0.01

    z = {"id": "2099-maj-26-t1", "temat": TEMAT, "polecenie": TEMAT, "esej": True}
    esej.napisz_e5(z, url="x", baza=baza, czat_fn=czat, stanowisko="zgadzam")
    assert any("KUL-TAK" in p for p in prompty) and any("POL-TAK" in p for p in prompty)


def test_baza_hasel_i_zapytanie(tmp_path):
    h = [{"id": "S3-01", "tytul": "Urzędy republiki rzymskiej", "tekst": "Konsulowie byli wybierani na rok, trybun ludowy miał prawo weta.",
          "daty": [], "postacie": [], "pojecia": [{"termin": "intercesja", "definicja": "weto trybuna"}],
          "dokumenty": [], "ikonografia": []},
         {"id": "S30-01", "tytul": "Powstanie styczniowe", "tekst": "Powstanie wybuchło w 1863 roku.", "daty": ["1863"],
          "postacie": ["Romuald Traugutt"], "pojecia": [], "dokumenty": [], "ikonografia": [
              {"nazwa": "Polonia", "autor": "Jan Matejko", "data": "1864", "co_przedstawia": "zakuwanie Polonii w kajdany"}]}]
    (tmp_path / "hasla_x.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in h) + "\n", encoding="utf-8")
    b = wiedza.BazaHasel(tmp_path)
    assert b.szukaj("Podaj nazwy urzędów: konsul i trybun ludowy", 1)[0]["id"] == "S3-01"
    assert "Jan Matejko" in wiedza._tekst_hasla(h[1])
    z = {"polecenie": "Podaj nazwę powstania.", "zrodla_tekst": "Źródło 1. Obraz Polonia\n[ilustracja]\nTreść."}
    q = wiedza.zapytanie(z, "OPIS ILUSTRACJI: kobieta w kajdanach")
    assert "Źródło 1. Obraz Polonia" in q and "kajdanach" in q
