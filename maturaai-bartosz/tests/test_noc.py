"""Testy T1 (matura/noc.py): karty i Wikipedia wg konfiguracji, filtry kalibracji, filtr --konfiguracje."""
import json

import pytest

from matura import devset, noc

pytestmark = pytest.mark.skipif(not (devset.JSON / "historia-2025-maj.json").exists(), reason="brak danych CKE")


def test_wikipedia_potrzebna_dla_konfiguracji_z_kontekstem():
    m = lambda *k: [{"nazwa": "m", "konfiguracje": list(k)}]
    assert noc.potrzebna_wiki(m("h0")) and noc.potrzebna_wiki(m("e0")) and noc.potrzebna_wiki(m("e2", "goly"))
    assert noc.potrzebna_wiki(m("h0_ocr")) and noc.potrzebna_wiki(m("h0_opis"))
    assert all(noc.potrzebna_wiki(m(k)) for k in ("h2", "h2_ocr", "h2_vlm", "e5", "e5_zgadzam"))
    assert noc.potrzebne_karty(m("e5")) and noc.potrzebne_karty(m("e5_zgadzam"))
    assert not noc.potrzebna_wiki(m("goly", "h0_norag", "e1"))


def test_karty_potrzebne_tylko_dla_e3_e4_e3_en():
    m = lambda *k: [{"nazwa": "m", "konfiguracje": list(k)}]
    assert noc.potrzebne_karty(m("e3")) and noc.potrzebne_karty(m("e4")) and noc.potrzebne_karty(m("e3_en"))
    assert not noc.potrzebne_karty(m("e2", "h0", "e1"))


def test_kalibracja_tylko_na_zadaniach_z_wzorcem():
    zad = devset.wczytaj(["2025-maj", "eseje-kalibracja", "eseje-cke"])
    wz = noc.zadania_kalibracji("wzorzec", zad)
    assert wz and all(z["rozwiazanie"] for z in wz) and not any(z["esej"] for z in wz)
    cke = noc.zadania_kalibracji("kalibracja_cke", zad)
    assert len(cke) == 8 and all(z.get("wzorzec_cke") for z in cke)
    assert noc.zadania_kalibracji("pusty", zad) == zad


def test_generuj_przekazuje_karty_do_harnessu(tmp_path, monkeypatch):
    zad = devset.wczytaj(["2025-maj"])[:2]
    znacznik = object()
    widziane = []
    monkeypatch.setattr(noc.harness, "odpowiedz", lambda konfig, z, **kw: (widziane.append(kw.get("karty")) or "x", 0.1, {}))
    p = noc.generuj(zad, "m", "e3", tmp_path, url="http://x", wiki=None, obrazy=False, bez_myslenia=True, rownolegle=1, karty=znacznik)
    assert widziane == [znacznik, znacznik] and len(noc.wczytaj_odp(p)) == 2


def test_filtr_modeli_i_konfiguracji():
    modele = [{"nazwa": "a", "konfiguracje": ["goly", "h0"]}, {"nazwa": "b", "konfiguracje": ["e1", "e3"]}]
    assert noc.filtruj_modele(modele, None, None) == modele
    assert noc.filtruj_modele(modele, "b", None) == [modele[1]]
    assert noc.filtruj_modele(modele, None, "h0,e3") == [{"nazwa": "a", "konfiguracje": ["h0"]}, {"nazwa": "b", "konfiguracje": ["e3"]}]
    assert noc.filtruj_modele(modele, None, "e1") == [{"nazwa": "b", "konfiguracje": ["e1"]}]


def test_karty_pomijaja_niedopisana_linie(tmp_path, monkeypatch):
    """Budowa kart dopisuje do pliku w trakcie nocy: ucięta ostatnia linia nie może wywracać wczytania."""
    from matura import karty
    p = tmp_path / "karty.jsonl"
    dobra = {"dzial": "d", "fakt": "Bitwa pod Grunwaldem w 1410 roku", "data": "1410", "postac": "Jagiełło", "termin": "", "aspekt": "militarny", "tytul": "t"}
    p.write_text(json.dumps(dobra, ensure_ascii=False) + "\n" + json.dumps(dobra)[:40], encoding="utf-8")
    monkeypatch.setattr(karty, "KAT", tmp_path)
    k = karty.Karty()
    assert len(k.k) == 1 and k.szukaj("Grunwald 1410")[0]["postac"] == "Jagiełło"


def test_filtr_prefiksow_id_bez_testu():
    zad = devset.wczytaj(["eseje-cke"])
    dev = noc.filtruj_prefiksy(zad, ["2023-maj-", "2024-maj-", "2025-maj-"])
    assert len(dev) == 9 and not any(z["id"].startswith("2026") for z in dev)
    assert noc.filtruj_prefiksy(zad, None) == zad
