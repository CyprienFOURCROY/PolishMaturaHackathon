"""Testy sędziego (T2): klucz cache z nazwą sędziego, wspólna lista pozycji do oceny, próbka kontrolna Astry."""
import json

import pytest

from matura import devset, sedzia

pytestmark = pytest.mark.skipif(not (devset.JSON / "historia-2025-maj.json").exists(), reason="brak danych CKE")


def test_klucz_cache_rozni_sedziow():
    assert sedzia.klucz("z1", "odp", "claude:fable") != sedzia.klucz("z1", "odp", "astra:gpt-6-astra")


def _wyn_z_odpowiedziami(tmp_path, zad):
    wyn = tmp_path / "wyn"
    (wyn / "odpowiedzi").mkdir(parents=True)
    with open(wyn / "odpowiedzi" / "m__h0.jsonl", "w", encoding="utf-8") as f:
        for z in zad:
            f.write(json.dumps({"id": z["id"], "odpowiedz": "odp " + z["id"]}) + "\n")
    return wyn


def test_pozycje_do_oceny_pomijaja_tylko_oceny_tego_samego_sedziego(tmp_path):
    from matura.ocen_wszystko import pozycje_do_oceny
    zad = devset.wczytaj(["2025-maj"])[:5]
    wyn = _wyn_z_odpowiedziami(tmp_path, zad)
    z0 = zad[0]
    cache = {sedzia.klucz(z0["id"], "odp " + z0["id"], "astra:gpt-6-astra"): {"pkt": 1}}
    poz = pozycje_do_oceny(wyn / "odpowiedzi" / "m__h0.jsonl", {z["id"]: z for z in zad}, cache, "claude:fable")
    assert len(poz) == 5  # ocena Astry nie zwalnia sędziego Claude
    assert poz[0]["klucz"] == sedzia.klucz(z0["id"], "odp " + z0["id"], "claude:fable")
    cache[poz[0]["klucz"]] = {"pkt": 1}
    assert len(pozycje_do_oceny(wyn / "odpowiedzi" / "m__h0.jsonl", {z["id"]: z for z in zad}, cache, "claude:fable")) == 4


def test_probka_kontrolna_deterministyczna_okolo_10_procent():
    from matura.ocen_wszystko import kontrolna
    poz = [{"id": f"z{i}", "odpowiedz": "odp"} for i in range(2000)]
    wyb = kontrolna(poz, 0.1)
    assert 150 <= len(wyb) <= 250
    assert [p["id"] for p in kontrolna(poz, 0.1)] == [p["id"] for p in wyb]


def test_przebieg_ocenia_sedzia_glownym_i_kontrolnym_na_probce(tmp_path):
    from matura import ocen_wszystko
    zad = devset.wczytaj(["2025-maj"])[:40]
    wyn = _wyn_z_odpowiedziami(tmp_path, zad)
    wywolania = []

    def fake(pt, model):
        wywolania.append((model, [p["klucz"] for p in pt]))
        return [{"klucz": p["klucz"], "pkt": 0} for p in pt]

    n = ocen_wszystko.przebieg(wyn, {z["id"]: z for z in zad}, "claude:fable", 1, 12,
                               kontrola="astra:gpt-6-astra", ocen_fn=fake, cache={})
    assert n == len(zad) + len({k for m, ks in wywolania for k in ks if m == "astra:gpt-6-astra"})
    glowne = {k for m, ks in wywolania for k in ks if m == "claude:fable"}
    kontr = {k for m, ks in wywolania for k in ks if m == "astra:gpt-6-astra"}
    assert glowne == {sedzia.klucz(z["id"], "odp " + z["id"], "claude:fable") for z in zad}
    oczek = {sedzia.klucz(p["id"], p["odpowiedz"], "astra:gpt-6-astra")
             for p in ocen_wszystko.kontrolna([{"id": z["id"], "odpowiedz": "odp " + z["id"]} for z in zad], 0.1)}
    assert kontr == oczek and 0 < len(kontr) < len(zad)


def test_raport_liczy_oceny_po_kluczu_sedziego(tmp_path):
    from matura import raport
    zad = devset.wczytaj(["2025-maj"])[:3]
    wyn = _wyn_z_odpowiedziami(tmp_path, zad)
    zd = {z["id"]: z for z in zad}
    cache = {sedzia.klucz(z["id"], "odp " + z["id"], "claude:fable"): {"pkt": 1, "pkt_max": z["pkt_max"]} for z in zad}
    assert raport._wiersze(wyn, zd, cache, {}, "claude:fable")[0]["ocen"] == 3
    assert raport._wiersze(wyn, zd, cache, {}, "astra:gpt-6-astra")[0]["ocen"] == 0


def test_przebieg_ocenia_w_kolejnosci_priorytetu(tmp_path):
    """Kolejność: kalibracja, potem wypracowania, potem modele od najmniejszego; w wierszu synt przed test przed dev."""
    from matura import ocen_wszystko
    zad = devset.wczytaj(["2025-maj", "2026-maj", "synt", "eseje-cke"])
    zd = {z["id"]: z for z in zad}
    wyn = tmp_path / "wyn"; (wyn / "odpowiedzi").mkdir(parents=True)
    krotkie = [z for z in zad if not z["esej"]]
    eseje = [z for z in zad if z["esej"]][:3]

    def plik(nazwa, lista):
        with open(wyn / "odpowiedzi" / nazwa, "w", encoding="utf-8") as f:
            for z in lista:
                f.write(json.dumps({"id": z["id"], "odpowiedz": f"odp {nazwa} {z['id']}"}) + "\n")
    plik("duzy__h0.jsonl", krotkie[:5]); plik("maly__h0.jsonl", krotkie); plik("maly__e1.jsonl", eseje)
    plik("_kalibracja__pusty.jsonl", krotkie[:5])
    (wyn / "rozmiary.json").write_text(json.dumps({"maly": {"razem": 100e6}, "duzy": {"razem": 1000e6}}))
    kolejnosc = []

    def fake(pt, model):
        kolejnosc.extend((p["model"], p["esej"], zd[p["id"]]["split"]) for p in pt)
        return []

    ocen_wszystko.przebieg(wyn, zd, "claude:fable", 1, 12, ocen_fn=fake, cache={})
    modele = [k[0] for k in kolejnosc]
    assert modele[0] == "_kalibracja" and modele.index("maly") < modele.index("duzy")
    maly = [k for k in kolejnosc if k[0] == "maly"]
    assert maly[0][1] is True and maly[-1][1] is False  # eseje przed krótkimi
    splity = [k[2] for k in maly if not k[1]]
    assert splity.index("test") > splity.index("synt") and splity.index("dev") > splity.index("test")
    assert splity == sorted(splity, key={"synt": 0, "test": 1, "dev": 2}.get)
