"""Testy pełnej matury (matura/pelna_matura.py): 60 pkt z esejem, --pomin, esej = średnia tematów, straż testu 2026,
ślepota sędziego, cache. Bez sieci i bez LLM: sędzia podmieniony (monkeypatch), cache w tmp_path."""
import json

import pytest

from matura import devset, pelna_matura as pm, sedzia

pytestmark = pytest.mark.skipif(not (devset.JSON / "historia-2025-maj.json").exists(), reason="brak danych CKE")


@pytest.fixture
def cache_tmp(tmp_path, monkeypatch):
    """Cache sędziego w tmp_path (prawdziwy review/oceny_cache.jsonl nietknięty)."""
    monkeypatch.setattr(sedzia, "CACHE", tmp_path / "oceny_cache.jsonl")
    return tmp_path / "oceny_cache.jsonl"


def _sedzia(monkeypatch, pkt=lambda p: p["pkt_max"], zapisuj=True):
    """Podmiana sedzia.ocen_partie: zapamiętuje partie, zwraca pkt(p), dopisuje do cache (jak prawdziwy)."""
    wywolania = []

    def fake(pozycje, model, timeout=900):
        wywolania.append([dict(p) for p in pozycje])
        wpisy = [{"klucz": p["klucz"], "id": p["id"], "pkt": pkt(p), "pkt_max": p["pkt_max"], "uzasadnienie": "t",
                  "sedzia": model} for p in pozycje]
        if zapisuj:
            sedzia._zapisz(wpisy)
        return wpisy

    monkeypatch.setattr(sedzia, "ocen_partie", fake)
    return wywolania


def _plik(tmp_path, nazwa, rekordy):
    p = tmp_path / "odpowiedzi" / nazwa
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rekordy), encoding="utf-8")
    return str(p)


def _wzorcowe(tmp_path, sesja="2025-maj", nazwa="m__h0.jsonl", tematy=("t1", "t2", "t3")):
    """Plik z odpowiedziami = klucz CKE dla zadań krótkich + wypracowania dla wybranych tematów (ten sam plik)."""
    a = pm.zadania_sesji(sesja)
    rek = [{"id": z["id"], "odpowiedz": z["rozwiazanie"]} for z in a["krotkie"]]
    rek += [{"id": t["id"], "odpowiedz": f"wypracowanie {t['id']}"} for t in a["tematy"] if t["id"].rsplit("-", 1)[1] in tematy]
    return _plik(tmp_path, nazwa, rek)


def _zestaw(etykieta, krotkie, eseje=None):
    return {"etykieta": etykieta, "krotkie": krotkie, "eseje": eseje or krotkie}


def test_2025_maj_60_pkt_wzorce_i_braki(tmp_path, cache_tmp, monkeypatch):
    _sedzia(monkeypatch)  # sędzia daje pełne punkty
    pelny = _wzorcowe(tmp_path)
    pusty = _plik(tmp_path, "pusty__h0.jsonl", [])
    w = pm.policz([_zestaw("wzorzec", pelny), _zestaw("pusty", pusty)], ["2025-maj"], tmp_path)
    s = w["zestawy"][0]["sesje"]["2025-maj"]
    assert s["zamkniete"][1] + s["otwarte"][1] + s["esej"][1] == 60 == s["razem"][1]
    assert s["esej"] == [15, 15] and s["razem"] == [60, 60] and s["proc"] == 100
    assert s["zrodla"]["automat"] > 0 and s["braki"] == 0 and s["nieocenione"] == 0
    b = w["zestawy"][1]["sesje"]["2025-maj"]
    assert b["razem"] == [0, 60] and b["braki"] == len(pm.zadania_sesji("2025-maj")["krotkie"]) and b["brak_eseju"]


def test_automat_daje_punkty_za_klucz_bez_sedziego(tmp_path, cache_tmp, monkeypatch):
    wyw = _sedzia(monkeypatch, pkt=lambda p: 0)  # sędzia daje 0: zostają tylko punkty automatu
    w = pm.policz([_zestaw("wzorzec", _wzorcowe(tmp_path, tematy=()))], ["2025-maj"], tmp_path)
    s = w["zestawy"][0]["sesje"]["2025-maj"]
    auto = sum(z["pkt_max"] for z in pm.zadania_sesji("2025-maj")["krotkie"]
               if pm.klucz.obslugiwane(z) and pm.klucz.ocen(z, z["rozwiazanie"])["pewne"])
    assert auto > 0 and s["razem"][0] == auto
    assert not any(p["esej"] for pt in wyw for p in pt)  # brak wypracowań → sędzia nie ocenia esejów


def test_pomin_zadania_z_podzadaniami_ale_nie_17(tmp_path, cache_tmp, monkeypatch):
    assert pm.pominiete("2023-maj-7.1", ["2023-maj-7"]) and pm.pominiete("2023-maj-7", ["2023-maj-7"])
    assert not pm.pominiete("2023-maj-17", ["2023-maj-7"]) and not pm.pominiete("2023-maj-70", ["2023-maj-7"])
    a = pm.zadania_sesji("2023-maj", ["2023-maj-7", "2023-maj-8", "2023-maj-15"])
    assert "2023-maj-17" in {z["id"] for z in a["krotkie"]}
    assert not any(pm.pominiete(z["id"], ["2023-maj-7", "2023-maj-8", "2023-maj-15"]) for z in a["krotkie"])
    wyw = _sedzia(monkeypatch)
    w = pm.policz([_zestaw("x", _plik(tmp_path, "x__h0.jsonl", []))], ["2023-maj"], tmp_path,
                  pomin=["2023-maj-7", "2023-maj-8", "2023-maj-15"])
    s = w["zestawy"][0]["sesje"]["2023-maj"]
    assert s["razem"] == [0, 55] and "2023-maj-17" not in s["pominiete"] and not wyw


def test_esej_srednia_z_ocenionych_tematow_i_brak_eseju(tmp_path, cache_tmp, monkeypatch):
    oceny = {"2025-maj-25-t1": 4, "2025-maj-25-t2": 7}
    _sedzia(monkeypatch, pkt=lambda p: oceny.get(p["id"], p["pkt_max"]))
    krotkie = _plik(tmp_path, "m__h0.jsonl", [])
    eseje = _plik(tmp_path, "m__e1.jsonl", [{"id": "2025-maj-25-t1", "odpowiedz": "esej 1"},
                                            {"id": "2025-maj-25-t2", "odpowiedz": "esej 2"},
                                            {"id": "2025-maj-25-t3", "odpowiedz": "(BŁĄD: timeout)"}])
    w = pm.policz([_zestaw("m", krotkie, eseje), _zestaw("bez", krotkie)], ["2025-maj"], tmp_path)
    s = w["zestawy"][0]["sesje"]["2025-maj"]
    assert s["esej"] == [5.5, 15] and s["razem"] == [5.5, 60] and not s["brak_eseju"]
    assert s["eseje_tematy"]["2025-maj-25-t1"] == {"pkt": 4, "stan": "oceniony"}
    assert s["eseje_tematy"]["2025-maj-25-t3"]["stan"] == "blad" and s["bledy"] == 1
    b = w["zestawy"][1]["sesje"]["2025-maj"]
    assert b["esej"] == [0, 15] and b["brak_eseju"]


def test_straz_testu_2026(tmp_path, cache_tmp):
    with pytest.raises(ValueError, match="pozwol-test"):
        pm.sprawdz_sesje(["2025-maj", "2026-maj"], False)
    pm.sprawdz_sesje(["2025-maj", "2026-maj"], True)
    pm.sprawdz_sesje(["2023-maj", "2025-czerwiec"], False)
    with pytest.raises(ValueError):
        pm.policz([_zestaw("x", "brak.jsonl")], ["2026-maj"], tmp_path)
    assert pm.main(["--wyniki", str(tmp_path), "--sesje", "2026-maj", "--zestaw", "x=brak.jsonl", "--bez-oceny"]) == 2


def test_sedzia_na_slepo_bez_etykiety_i_modelu_zestawy_wymieszane(tmp_path, cache_tmp, monkeypatch):
    wyw = _sedzia(monkeypatch)
    a = pm.zadania_sesji("2025-maj")
    odp = lambda tag: [{"id": z["id"], "odpowiedz": f"tekst {tag}{i}"} for i, z in enumerate(a["krotkie"])]  # noqa: E731
    p1 = _plik(tmp_path, "tajnymodel__h0.jsonl", odp("XA"))
    p2 = _plik(tmp_path, "innymodel__h0.jsonl", odp("XB"))
    pm.policz([_zestaw("TAJNA-ETYKIETA | h0", p1), _zestaw("DRUGA-ETYKIETA", p2)], ["2025-maj"], tmp_path, partia=10)
    assert wyw
    tekst = json.dumps(wyw, ensure_ascii=False)
    for zakazane in ("TAJNA", "DRUGA", "ETYKIETA", "tajnymodel", "innymodel", "h0.jsonl"):
        assert zakazane not in tekst
    assert all(set(p) == {"klucz", "id", "pkt_max", "polecenie", "zasady", "rozwiazanie", "odpowiedz", "esej"}
               for pt in wyw for p in pt)
    assert all(len({p["esej"] for p in pt}) == 1 for pt in wyw)  # partie jednorodne
    assert any({"tekst XA" in p["odpowiedz"] for p in pt} == {True, False} for pt in wyw)  # zestawy wymieszane


def test_cache_druga_ocena_nie_wola_sedziego(tmp_path, cache_tmp, monkeypatch):
    wyw = _sedzia(monkeypatch, pkt=lambda p: 1 if p["pkt_max"] >= 1 else 0)
    zs = [_zestaw("m", _wzorcowe(tmp_path, tematy=("t1",)))]
    w1 = pm.policz(zs, ["2025-maj"], tmp_path)
    n = len(wyw)
    assert n > 0 and cache_tmp.exists()
    w2 = pm.policz(zs, ["2025-maj"], tmp_path)
    assert len(wyw) == n and w1["zestawy"] == w2["zestawy"]
    w3 = pm.policz(zs, ["2025-maj"], tmp_path, sedzia_model="astra:gpt-6-astra", bez_oceny=True)
    s3 = w3["zestawy"][0]["sesje"]["2025-maj"]
    assert len(wyw) == n and s3["nieocenione"] > 0  # inny sędzia: cache się nie miesza, --bez-oceny nie woła


def test_partia_ktora_padla_zostaje_nieoceniona(tmp_path, cache_tmp, monkeypatch):
    def pada(pozycje, model, timeout=900):
        raise RuntimeError("limit subskrypcji")
    monkeypatch.setattr(sedzia, "ocen_partie", pada)
    w = pm.policz([_zestaw("m", _wzorcowe(tmp_path))], ["2025-maj"], tmp_path)
    s = w["zestawy"][0]["sesje"]["2025-maj"]
    assert s["nieocenione"] > 0 and s["zrodla"]["automat"] > 0 and s["esej"] == [0, 15] and not s["brak_eseju"]


def test_parsuj_zestaw_i_raport(tmp_path, cache_tmp, monkeypatch):
    z = pm.parsuj_zestaw("bielik-1.5b-q8 | h0=a/x__h0.jsonl,a/y__e1.jsonl")
    assert z == {"etykieta": "bielik-1.5b-q8 | h0", "krotkie": "a/x__h0.jsonl", "eseje": "a/y__e1.jsonl"}
    assert pm.parsuj_zestaw("m=a/x__goly.jsonl")["eseje"] == "a/x__goly.jsonl"
    with pytest.raises(ValueError):
        pm.parsuj_zestaw("bez-plikow")
    (tmp_path / "zestawy.txt").write_text("# komentarz\n\nm=a.jsonl\nn x=b.jsonl,c.jsonl\n", encoding="utf-8")
    assert [z["etykieta"] for z in pm.wczytaj_zestawy(tmp_path / "zestawy.txt")] == ["m", "n x"]
    (tmp_path / "rozmiary.json").write_text(json.dumps({"mm": {"razem": 250_000_000}}), encoding="utf-8")
    _sedzia(monkeypatch)
    w = pm.policz([_zestaw("mm | h0", _wzorcowe(tmp_path)), _zestaw("inny", _wzorcowe(tmp_path, nazwa="mm__goly.jsonl"))],
                  ["2025-maj"], tmp_path)
    assert [z["rozmiar_mb"] for z in w["zestawy"]] == [250.0, 250.0]  # z etykiety, potem z nazwy pliku
    pj, pmd = pm.zapisz(w, tmp_path / "wyn", "pelna_matura")
    assert pmd.name == "PELNA_MATURA.md" and json.loads(pj.read_text(encoding="utf-8"))["meta"]["sesje"] == ["2025-maj"]
    linie = pmd.read_text(encoding="utf-8").splitlines()
    assert linie[2].startswith("**Co to jest:**") and linie[3].startswith("**Po co:**") and linie[4].startswith("**Co zrobić:**")
    assert "układu PDF" in pmd.read_text(encoding="utf-8")
