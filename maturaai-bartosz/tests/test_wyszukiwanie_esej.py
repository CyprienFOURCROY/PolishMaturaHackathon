"""Wyszukiwanie materiału do eseju (matura/wyszukiwanie_esej.py): kawałki, zapytania per aspekt, RRF, budżet.

Co to jest: testy bez GPU (reranker podmieniany atrapą); tematy spoza ocenianych arkuszy (2015-2022).
Po co: pilnują, że materiał nie jest cięty w środku zdania, mieści się w budżecie i pokrywa każdy aspekt tematu.
Co zrobić: uv run pytest -q tests/test_wyszukiwanie_esej.py
"""
import pytest

from matura import esej_sft
from matura import wyszukiwanie_esej as W

TEMAT = ("Polityka Napoleona Bonaparte wobec sprawy polskiej przede wszystkim służyła interesom Francji. Zajmij "
         "stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji aspekty: polityczny, "
         "militarny i gospodarczy.")


def test_potnij_na_granicy_zdan_z_tytulem():
    tekst = " ".join(f"Zdanie numer {i} opisuje wydarzenie z roku 18{i:02d} i jego skutki dla państwa." for i in range(20))
    kaw = W.potnij("Księstwo Warszawskie", tekst)
    assert len(kaw) > 1 and all(k.startswith("[Księstwo Warszawskie] ") for k in kaw)
    tresc = [k[len("[Księstwo Warszawskie] "):] for k in kaw]
    assert all(t.endswith(".") for t in tresc)                                  # całe zdania
    assert all(len(t) <= W.MAX_ZN for t in tresc[:-1]) and " ".join(tresc) == " ".join(tekst.split())


def test_potnij_krotka_koncowka_dolaczona():
    kaw = W.potnij("T", "A" * 450 + ". Krótkie.")
    assert len(kaw) == 1


def test_zapytania_per_aspekt_i_wybor():
    q = W.zapytania(TEMAT)
    assert len(q) == 4 and q[1].endswith("aspekt polityczny") and q[3].endswith("aspekt gospodarczy")
    wybor = ("Zimna wojna osiągnęła apogeum w latach 50. XX wieku. Zajmij stanowisko wobec powyższej tezy i je "
             "uzasadnij, charakteryzując trzy wybrane wydarzenia z tego okresu.")
    assert len(W.zapytania(wybor)) == 2


def test_rrf_laczy_rankingi():
    assert W.rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]


def test_spakuj_bez_przycinania_pod_budzetem():
    t = ["a" * 100, "b" * 400, "c" * 50]
    m = W.spakuj(t, 200)
    assert m == f"- {'a' * 100}\n- {'c' * 50}" and len(m) <= 200


class RerankerAtrapa:
    def oceny(self, zapytanie, teksty):   # wyżej teksty ze słowem z ostatniego członu zapytania
        klucz = zapytanie.split()[-1][:5]
        return [float(klucz in t.lower()) for t in teksty]


def test_indeks_ranking_i_material_w_budzecie():
    hasla = [{"id": f"h{i}", "tytul": f"Hasło {i}", "tekst": t, "dokumenty": [], "ikonografia": []}
             for i, t in enumerate(["Napoleon utworzył Księstwo Warszawskie w 1807 roku. Polityka Francji wobec Polski.",
                                    "Armia Księstwa walczyła militarnie pod Raszynem w 1809 roku. Napoleon dowodził.",
                                    "Blokada kontynentalna szkodziła gospodarczo handlowi zbożem. Napoleon ją wprowadził.",
                                    "Egipcjanie budowali piramidy w Gizie."])]
    ix = W.Indeks(hasla=hasla, karty=[], z_kartami=False, reranker=RerankerAtrapa())
    for wariant in ("bm25_kawalki", "bm25_rerank", "rerank_wszystko"):
        m = ix.material(TEMAT, budzet=300, wariant=wariant)
        assert 0 < len(m) <= 300 and "piramidy" not in m.split("\n")[0]
    with pytest.raises(ValueError):
        ix.material(TEMAT, wariant="losowy")


def test_material_v2_w_esej_sft_nie_zmienia_domyslnego():
    assert esej_sft.ZNAKI_HASLA == 1500 and esej_sft.material.__defaults__[-1] == "pelny"
    assert "wariant" in esej_sft.material_v2.__code__.co_varnames


def test_hasla_plus_rerank_hasla_najpierw_potem_kawalki(monkeypatch):
    hasla = [{"id": f"h{i}", "tytul": f"Hasło {i}", "tekst": t, "dokumenty": [], "ikonografia": []}
             for i, t in enumerate(["Napoleon utworzył Księstwo Warszawskie w 1807 roku. Polityka Francji wobec Polski.",
                                    "Blokada kontynentalna szkodziła gospodarczo handlowi zbożem. Napoleon ją wprowadził."])]
    ix = W.Indeks(hasla=hasla, karty=[], z_kartami=False, reranker=RerankerAtrapa())

    class Baza:
        def szukaj(self, q, n):
            return hasla[:1]

    ix._baza = Baza()
    m = ix.material(TEMAT, budzet=400, wariant="hasla_plus_rerank")
    linie = m.split("\n")
    assert linie[0].startswith("- [Hasło 0]") and len(m) <= 400
    assert all("[Hasło 0]" not in l for l in linie[1:])            # kawałki spoza wybranych haseł
