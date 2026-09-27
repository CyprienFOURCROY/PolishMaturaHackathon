"""Testy raportu (T3): wiersze esejów, kalibracja sędziego na wypracowaniach CKE, zgodność sędziów, czas per obszar.

Realne zadania z data/ (poza gitem, więc pomijane, gdy brak), katalogi wyników w tmp_path, cache ocen podawany
parametrem (nigdy review/oceny_cache.jsonl).
"""
import json
import re

import pytest

from matura import devset, raport, sedzia

pytestmark = pytest.mark.skipif(not (devset.JSON / "historia-2025-maj.json").exists() or not devset.KALIBRACJA.exists(),
                                reason="brak danych CKE albo kalibracji esejów")

G, K = "claude:fable", "astra:gpt-6-astra"  # sędzia główny i kontrolny


def _odp(wyn, model, konfig, rekordy):
    """Plik odpowiedzi w formacie matura/noc.py: <wyn>/odpowiedzi/<model>__<konfig>.jsonl."""
    p = wyn / "odpowiedzi" / f"{model}__{konfig}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps({"model": model, "konfig": konfig, "sekundy": 0.0, **r}, ensure_ascii=False) + "\n"
                         for r in rekordy), encoding="utf-8")


def _ocena(z, odp, sedzia_, pkt, pkt_A=None, pkt_B=0, bledy=0):
    """(klucz, wpis) cache w formacie sedzia.ocen_partie (zadanie krótkie: pkt_A = pkt, pkt_B = 0)."""
    k = sedzia.klucz(z["id"], odp, sedzia_)
    return k, {"klucz": k, "id": z["id"], "pkt": pkt, "pkt_A": pkt if pkt_A is None else pkt_A, "pkt_B": pkt_B,
               "bledy": bledy, "pkt_max": z["pkt_max"], "uzasadnienie": "test", "sedzia": sedzia_}


def test_wiersz_esejowy_liczy_A_B_bledy_i_slowa_tylko_z_ocenionych_esejow(tmp_path):
    wyn = tmp_path / "wyn"
    e1, e2, e3 = devset.wczytaj(["eseje-cke"])[:3]
    _odp(wyn, "m", "e1", [{"id": e1["id"], "odpowiedz": "esej pierwszy", "slow": 400},  # słowa z meta esej.napisz
                          {"id": e2["id"], "odpowiedz": "jeden dwa trzy cztery pięć"},  # bez meta: 5 słów
                          {"id": e3["id"], "odpowiedz": "bez oceny", "slow": 999}])     # bez oceny sędziego głównego
    cache = dict([_ocena(e1, "esej pierwszy", G, 11, pkt_A=9, pkt_B=2, bledy=1),
                  _ocena(e2, "jeden dwa trzy cztery pięć", G, 4, pkt_A=4, pkt_B=0, bledy=2),
                  _ocena(e3, "bez oceny", K, 15, pkt_A=12, pkt_B=3)])  # ocena innego sędziego się nie liczy
    r = raport._wiersze(wyn, {z["id"]: z for z in (e1, e2, e3)}, cache, {}, G)[0]
    assert (r["esej_n"], r["esej_A"], r["esej_B"], r["esej_bledy"], r["esej_slow"]) == (2, 6.5, 1.0, 3, 202.5)


def test_wiersz_bez_esejow_ma_puste_pola_esejowe(tmp_path):
    wyn = tmp_path / "wyn"
    k1, k2 = devset.wczytaj(["2025-maj"])[:2]
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a"}, {"id": k2["id"], "odpowiedz": "b"}])
    cache = dict([_ocena(k1, "a", G, 1), _ocena(k2, "b", G, 0)])
    r = raport._wiersze(wyn, {z["id"]: z for z in (k1, k2)}, cache, {}, G)[0]
    assert r["ocen"] == 2
    assert (r["esej_A"], r["esej_B"], r["esej_bledy"], r["esej_slow"]) == (None, None, None, None)


def _kalibracja(tmp_path, oceny):
    """Odpowiedzi _kalibracja/kalibracja_cke (wypracowania z Informatora) + cache sędziego głównego: id → (A, B)."""
    wyn = tmp_path / "wyn"
    kal = devset.wczytaj(["eseje-kalibracja"])
    zd = {z["id"]: z for z in kal}
    _odp(wyn, "_kalibracja", "kalibracja_cke", [{"id": z["id"], "odpowiedz": z["wzorzec_cke"]} for z in kal])
    cache = dict(_ocena(zd[i], zd[i]["wzorzec_cke"], G, a + b, pkt_A=a, pkt_B=b) for i, (a, b) in oceny.items())
    return wyn, kal, cache


def test_kalibracja_cke_srednia_roznica_i_wiersze_z_ocena_cke(tmp_path):
    # CKE (Informator): p1-r1 A9 B3 = 12, p1-r2 A2 B3 = 5, p2-r1 A12 B3 = 15, p2-r2 A7 B3 = 10, p3-r1 A12 B3 = 15
    wyn, kal, cache = _kalibracja(tmp_path, {"informator-p1-r1": (10, 3), "informator-p1-r2": (0, 2),
                                             "informator-p2-r1": (12, 3), "informator-p2-r2": (9, 3)})
    zd = {z["id"]: z for z in kal}
    p3 = zd["informator-p3-r1"]
    cache.update([_ocena(p3, p3["wzorzec_cke"], K, 6, pkt_A=5, pkt_B=1)])  # ocena kontrolnego nie wchodzi do kalibracji
    e1 = devset.wczytaj(["eseje-cke"])[0]  # temat CKE bez wypracowania z oceną CKE: harness daje „(brak)”
    zd[e1["id"]] = e1
    with open(wyn / "odpowiedzi" / "_kalibracja__kalibracja_cke.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": e1["id"], "model": "_kalibracja", "konfig": "kalibracja_cke", "odpowiedz": "(brak)",
                            "sekundy": 0.0}) + "\n")
    cache.update([_ocena(e1, "(brak)", G, 0, pkt_A=0)])
    k = raport.kalibracja_cke(wyn, zd, cache, G)
    assert k["n"] == 4 and k["srednia_roznica"] == 1.5  # (|+1| + |-3| + |0| + |+2|) / 4
    w = {x["id"]: x for x in k["wiersze"]}
    assert len(w) == 8
    assert w["informator-p1-r2"] == {"id": "informator-p1-r2", "cke_A": 2, "cke_B": 3, "cke": 5,
                                     "sedzia_A": 0, "sedzia_B": 2, "sedzia": 2, "roznica": -3}
    assert (w["informator-p3-r1"]["cke_A"], w["informator-p3-r1"]["cke"]) == (12, 15)
    assert w["informator-p3-r1"]["sedzia"] is None and w["informator-p3-r1"]["roznica"] is None


@pytest.mark.parametrize("oceny, werdykt", [
    ({"informator-p1-r1": (11, 3), "informator-p1-r2": (1, 2)}, "spełniony"),           # |+2|, |-2|: średnio 2.0 = próg
    ({"informator-p1-r1": (12, 3), "informator-p1-r2": (0, 1)}, "NIE jest spełniony"),  # |+3|, |-4|: średnio 3.5
    ({}, "nie został sprawdzony"),                                                      # brak ocen
])
def test_raport_md_rozstrzyga_prog_kalibracji_cke(tmp_path, oceny, werdykt):
    wyn, kal, cache = _kalibracja(tmp_path, oceny)
    md = raport.zapisz(wyn, kal, {"ogolne": {"sedzia_model": G}}, cache=cache).read_text(encoding="utf-8")
    assert "## Kalibracja sędziego" in md
    assert md.count("**Próg kalibracji CKE") == 1 and f"**Próg kalibracji CKE {werdykt}" in md


def test_sekcja_kalibracji_pokazuje_wzorzec_i_pusty(tmp_path):
    wyn = tmp_path / "wyn"
    k1, k2 = devset.wczytaj(["2025-maj"])[:2]
    _odp(wyn, "_kalibracja", "wzorzec", [{"id": z["id"], "odpowiedz": z["rozwiazanie"]} for z in (k1, k2)])
    _odp(wyn, "_kalibracja", "pusty", [{"id": z["id"], "odpowiedz": "Nie wiem."} for z in (k1, k2)])
    cache = dict([_ocena(k1, k1["rozwiazanie"], G, k1["pkt_max"]), _ocena(k2, k2["rozwiazanie"], G, 0),
                  _ocena(k1, "Nie wiem.", G, 0), _ocena(k2, "Nie wiem.", G, 0)])
    md = raport.zapisz(wyn, [k1, k2], {"ogolne": {"sedzia_model": G}}, cache=cache).read_text(encoding="utf-8")
    sekcja = md.split("## Kalibracja sędziego", 1)[1].split("\n## ", 1)[0]
    assert "`wzorzec` 50%" in sekcja and "`pusty` 0%" in sekcja  # k1 i k2 mają po 1 pkt max


def test_zgodnosc_sedziow_osobno_eseje_i_krotkie(tmp_path):
    wyn = tmp_path / "wyn"
    e1, e2 = devset.wczytaj(["eseje-cke"])[:2]  # pkt_max 15
    k1, k2, k3, k4 = devset.wczytaj(["2025-maj"])[:4]  # pkt_max 1
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a"}, {"id": k2["id"], "odpowiedz": "b"},
                          {"id": k3["id"], "odpowiedz": "c"}, {"id": k4["id"], "odpowiedz": "d"}])
    _odp(wyn, "n", "h0", [{"id": k1["id"], "odpowiedz": "a"}])  # ta sama para (zadanie, odpowiedź) liczy się raz
    _odp(wyn, "m", "e1", [{"id": e1["id"], "odpowiedz": "esej 1"}, {"id": e2["id"], "odpowiedz": "esej 2"}])
    cache = dict([_ocena(k1, "a", G, 1), _ocena(k1, "a", K, 1),
                  _ocena(k2, "b", G, 1), _ocena(k2, "b", K, 0),
                  _ocena(k3, "c", G, 0), _ocena(k3, "c", K, 0),
                  _ocena(k4, "d", G, 0),  # tylko sędzia główny: poza porównaniem
                  _ocena(e1, "esej 1", G, 12), _ocena(e1, "esej 1", K, 9),
                  _ocena(e2, "esej 2", G, 8), _ocena(e2, "esej 2", K, 8)])
    zg = raport.zgodnosc(wyn, {z["id"]: z for z in (e1, e2, k1, k2, k3, k4)}, cache, G, K)
    # krótkie: pary (1, 1), (1, 0), (0, 0); eseje: (12, 9), (8, 8)
    oczek = {"krotkie": {"n": 3, "srednia_roznica": 1 / 3, "identyczne": 2 / 3, "srednia_glowny": 2 / 3,
                         "srednia_kontrola": 1 / 3, "rozjazd": 1 / 3},
             "esej": {"n": 2, "srednia_roznica": 1.5, "identyczne": 0.5, "srednia_glowny": 10.0,
                      "srednia_kontrola": 8.5, "rozjazd": 0.1}}
    for obszar, o in oczek.items():
        assert {k: zg[obszar][k] for k in o} == pytest.approx(o)


def test_zgodnosc_bez_sedziego_kontrolnego_nie_ma_par(tmp_path):
    wyn = tmp_path / "wyn"
    k1 = devset.wczytaj(["2025-maj"])[0]
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a"}])
    zg = raport.zgodnosc(wyn, {k1["id"]: k1}, dict([_ocena(k1, "a", G, 1)]), G, None)
    assert zg["esej"]["n"] == 0 and zg["krotkie"]["n"] == 0


@pytest.mark.parametrize("pkt_kontrola, ostrzezenie", [(9, False), (8, True)])  # rozjazd 3/30 = 10% (próg) i 4/30
def test_raport_md_ostrzega_przy_rozjezdzie_sedziow_powyzej_10_procent(tmp_path, pkt_kontrola, ostrzezenie):
    wyn = tmp_path / "wyn"
    e1, e2 = devset.wczytaj(["eseje-cke"])[:2]
    _odp(wyn, "m", "e1", [{"id": e1["id"], "odpowiedz": "esej 1"}, {"id": e2["id"], "odpowiedz": "esej 2"}])
    cache = dict([_ocena(e1, "esej 1", G, 12), _ocena(e1, "esej 1", K, pkt_kontrola),
                  _ocena(e2, "esej 2", G, 8), _ocena(e2, "esej 2", K, 8)])
    md = raport.zapisz(wyn, [e1, e2], {"ogolne": {"sedzia_model": G, "sedzia_kontrola": K}},
                       cache=cache).read_text(encoding="utf-8")
    assert "## Zgodność Claude vs Astra" in md
    assert ("**Sygnał ostrzegawczy" in md) is ostrzezenie


def test_czas_obszarow_sumuje_sekundy_esej_i_reszta_bez_kalibracji(tmp_path):
    wyn = tmp_path / "wyn"
    e1, e2 = devset.wczytaj(["eseje-cke"])[:2]
    k1, k2 = devset.wczytaj(["2025-maj"])[:2]
    _odp(wyn, "m", "e1", [{"id": e1["id"], "odpowiedz": "x", "sekundy": 90.0},
                          {"id": e2["id"], "odpowiedz": "y", "sekundy": 150.0}])
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a", "sekundy": 20.0},
                          {"id": k2["id"], "odpowiedz": "b", "sekundy": 10.0}])
    _odp(wyn, "_kalibracja", "wzorzec", [{"id": k1["id"], "odpowiedz": "w", "sekundy": 600.0}])
    _odp(wyn, "_kalibracja", "kalibracja_cke", [{"id": e1["id"], "odpowiedz": "w", "sekundy": 600.0}])
    zd = {z["id"]: z for z in (e1, e2, k1, k2)}
    assert raport.czas_obszarow(wyn, zd) == {"esej_min": 4.0, "reszta_min": 0.5}  # 240 s i 30 s
    md = raport.zapisz(wyn, list(zd.values()), {"ogolne": {"sedzia_model": G}}, cache={}).read_text(encoding="utf-8")
    sekcja = md.split("## Czas", 1)[1]
    assert "4.0 min" in sekcja and "0.5 min" in sekcja
    assert "mniej czasu ma obszar RESZTA" in sekcja


def test_tabela_wypracowan_tylko_gdy_sa_ocenione_eseje(tmp_path):
    wyn = tmp_path / "wyn"
    e1 = devset.wczytaj(["eseje-cke"])[0]
    k1 = devset.wczytaj(["2025-maj"])[0]
    cfg = {"ogolne": {"sedzia_model": G}}
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a"}])
    cache = dict([_ocena(k1, "a", G, 1)])
    assert "## Wypracowania" not in raport.zapisz(wyn, [k1, e1], cfg, cache=cache).read_text(encoding="utf-8")
    _odp(wyn, "m", "e1", [{"id": e1["id"], "odpowiedz": "esej", "slow": 420}])
    cache.update([_ocena(e1, "esej", G, 11, pkt_A=9, pkt_B=2, bledy=1)])
    md = raport.zapisz(wyn, [k1, e1], cfg, cache=cache).read_text(encoding="utf-8")
    wiersz = next(l for l in md.split("## Wypracowania", 1)[1].splitlines() if l.startswith("| m | e1 |"))
    assert {"9.0", "2.0", "420"} <= {c.strip() for c in wiersz.split("|")}


def test_zapisz_raport_json_nowy_format_z_podanego_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sedzia, "wczytaj_cache", lambda: pytest.fail("zapisz(cache=...) nie może czytać cache z dysku"))
    wyn = tmp_path / "wyn"
    kal = devset.wczytaj(["eseje-kalibracja"])[:2]
    k1 = devset.wczytaj(["2025-maj"])[0]
    _odp(wyn, "_kalibracja", "kalibracja_cke", [{"id": z["id"], "odpowiedz": z["wzorzec_cke"]} for z in kal])
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a", "sekundy": 30.0}])
    cache = dict([_ocena(kal[0], kal[0]["wzorzec_cke"], G, 13, pkt_A=10, pkt_B=3),  # CKE p1-r1 = 12
                  _ocena(k1, "a", G, 1), _ocena(k1, "a", K, 0)])
    raport.zapisz(wyn, [*kal, k1], {"ogolne": {"sedzia_model": G, "sedzia_kontrola": K}}, cache=cache)
    j = json.loads((wyn / "raport.json").read_text(encoding="utf-8"))
    assert set(j) == {"wiersze", "kalibracja_cke", "zgodnosc", "czas", "stan_na"}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", j["stan_na"])
    assert {(r["model"], r["konfig"]) for r in j["wiersze"]} == {("_kalibracja", "kalibracja_cke"), ("m", "h0")}
    assert (j["kalibracja_cke"]["n"], j["kalibracja_cke"]["srednia_roznica"]) == (1, 1.0)
    assert j["zgodnosc"]["krotkie"]["n"] == 1 and j["zgodnosc"]["esej"]["n"] == 0
    assert j["czas"] == {"esej_min": 0.0, "reszta_min": 0.5}


def test_zapisz_bez_cache_czyta_cache_sedziego(tmp_path, monkeypatch):
    wyn = tmp_path / "wyn"
    k1 = devset.wczytaj(["2025-maj"])[0]
    _odp(wyn, "m", "h0", [{"id": k1["id"], "odpowiedz": "a"}])
    monkeypatch.setattr(sedzia, "wczytaj_cache", lambda: dict([_ocena(k1, "a", G, 1)]))
    raport.zapisz(wyn, [k1], {"ogolne": {"sedzia_model": G}})
    assert json.loads((wyn / "raport.json").read_text(encoding="utf-8"))["wiersze"][0]["ocen"] == 1
