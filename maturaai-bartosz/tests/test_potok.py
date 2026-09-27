"""Testy mechaniki potoku na REALNYCH danych (arkusze CKE z data/, poza gitem → pomijane, gdy brak)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "data" / "cke" / "json"
pytestmark = pytest.mark.skipif(not (JSON / "historia-2025-maj.json").exists(), reason="brak danych CKE (scripts/pobierz_arkusze_cke.py)")


def test_arkusz_2025_ma_60_pkt_i_esej_z_rubryka():
    w = json.loads((JSON / "historia-2025-maj.json").read_text(encoding="utf-8"))
    assert sum(z["pkt_max"] for z in w) == 60
    esej = [z for z in w if z["esej"]]
    assert len(esej) == 1 and "NARRACJA HISTORYCZNA" in esej[0]["zasady_oceniania"]
    assert all(z["rozwiazanie"] for z in w if not z["esej"])


def test_znak_wodny_nie_jest_zrodlem():
    w = json.loads((JSON / "historia-2025-maj.json").read_text(encoding="utf-8"))
    obrazy = {o for z in w for o in z["obrazy"]}
    assert not any("str02_" in o for o in obrazy)  # strona instrukcji ma tylko logo EM23
    assert any("str04_" in o for o in next(z for z in w if z["nr_zadania"] == "1.1")["obrazy"])


def test_podzial_zbiorow_bez_wycieku():
    from matura import devset
    z = devset.wczytaj(["2025-maj", "2026-maj"])
    assert {x["split"] for x in z if x["rok"] == 2025} == {"dev"}
    assert {x["split"] for x in z if x["rok"] == 2026} == {"test"}
    assert all(p["rok"] <= 2022 for p in devset.pula_treningowa())


def test_prompty_bez_doslownych_przykladow_formatu():
    from matura import devset, harness
    for s in (harness.SYS_ZAMKNIETE, harness.SYS_OTWARTE, harness.SYS_ROZSTRZYGNIJ):
        assert "1. P, 2. F" not in s  # lekcja z testu dymnego: 0.8B kopiował przykład
    z = {x["nr_zadania"]: x for x in devset.wczytaj(["2025-maj"])}
    assert harness.system_dla(z["1.1"]) == harness.SYS_ROZSTRZYGNIJ
    assert not any(x["esej"] for x in z.values())  # eseje są osobnym obszarem (eseje-cke)
    assert harness.system_dla(devset.wczytaj(["eseje-cke"])[0]) == harness.SYS_ESEJ


def test_czyszczenie_rozwiazania_cke():
    from matura.trening import _oczysc_cke
    assert _oczysc_cke("Rozstrzygnięcie: C\nPrzykładowe uzasadnienie: opis.") == "Rozstrzygnięcie: C\nUzasadnienie: opis."
    assert _oczysc_cke("• pierwsza\n• druga") == "pierwsza"


def test_konfiguracje_nocy_sie_wczytuja():
    from matura.noc import wczytaj_config
    for f in ("f2_drabinka", "f3_ablacje", "f4_kwanty", "f6_po_treningu", "e_esej"):
        c = wczytaj_config(f"noc/{f}.toml")
        assert c["ogolne"]["sedzia_model"].startswith("claude:") and c["model"]
        for m in c["model"]:
            assert set(m["konfiguracje"]) <= {"goly", "h0", "h0_norag", "h1", "h0_en", "e0", "e1", "e2", "e3", "e4", "e3_en"}
    e = wczytaj_config("noc/e_esej.toml")["ogolne"]
    assert e["sesje"] == ["eseje-cke", "eseje-synt", "eseje-kalibracja"] and e["kalibracja"] == ["kalibracja_cke"] and e["port"] != 8091


def test_tematy_esejow_cke_24_i_trzy_elementy():
    from matura import devset, esej
    e = devset.wczytaj(["eseje-cke"])
    assert len(e) == 24 and {x["split"] for x in e} == {"dev", "test"}
    for x in e:
        assert "WYPRACOWANIE" not in x["temat"]
        teza, asp, wybor = esej.rozbierz(x["temat"])
        assert teza and (wybor or len(asp) == 3)


def test_szablon_eseju_stanowisko_spojne_i_dlugosc():
    from matura import esej
    fake = lambda url, sys, user, obr, **k: ("W 1410 roku wojska Jagiełły pokonały Krzyżaków pod Grunwaldem, co wzmocniło państwo. " * 3, 0.0)
    z = {"temat": "Władysław Jagiełło był najwybitniejszym władcą Polski z dynastii Jagiellonów. Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji aspekty: militarny, ustrojowy i społeczno-gospodarczy."}
    t, _, m = esej.napisz(z, "e1", url=None, czat_fn=fake)
    akapity = t.split("\n\n")
    assert len(akapity) == 5 and "częściowo" in akapity[0] and "częściowo" in akapity[-1]
    assert "w aspekcie militarnym" in t and m["slow"] >= 300


def test_weryfikator_usuwa_zdanie_z_obca_data():
    from matura.esej import _weryfikuj
    a = "Bitwa pod Grunwaldem była w 1410 roku. Pokój zawarto w 1411. Unia w Krewie była w 1385. Coś się stało w 1999."
    assert "1999" not in _weryfikuj(a, "1410 1411 1385")


def test_router_epok():
    from matura.router import epoka
    assert epoka("Konstytucja 3 maja 1791") == "nowozytnosc" and epoka("Solidarność 1980") == "xx-xxi"


@pytest.mark.skipif(not (ROOT / "data" / "zasady" / "sedzia-esej.md").exists(), reason="brak pliku zasad")
def test_plik_zasad_z_oficjalnego_zrodla():
    t = (ROOT / "data" / "zasady" / "ZASADY-OCENIANIA-HISTORIA.md").read_text(encoding="utf-8")
    for fr in ("NARRACJA HISTORYCZNA", "SPÓJNOŚĆ WYPOWIEDZI", "300 słów", "błędy merytoryczne", "cke.gov.pl"):
        assert fr in t
    k = json.loads((ROOT / "data" / "zasady" / "kalibracja_esejow.json").read_text(encoding="utf-8"))
    assert len(k) == 8 and all(0 <= x["pkt"] <= 15 for x in k)


def test_brak_plikow_cke_w_git():
    import subprocess
    from fnmatch import fnmatch
    pliki = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    # wyjątek: baza wiedzy RAG od Claude wg działów podstawy (bez treści arkuszy); fakty „wzorcowe” z zadań dev nie
    baza = lambda f: (fnmatch(f, "data/wiedza/hasla*.jsonl") or fnmatch(f, "data/wiedza/akapity_esej*.jsonl")
                      or f in ("data/wiedza/tlumaczenia/hasla_en.jsonl", "data/karty/kanon.jsonl",
                               "data/karty/slownik_kanon.jsonl")) and "wzorc" not in f  # noqa: E731
    assert not [f for f in pliki if f.startswith(("data/", "docs/kontekst/rozmowa")) and not baza(f)
                or f.endswith(".pdf") and "zasady-organizatorow" not in f]
