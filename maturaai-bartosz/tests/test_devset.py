"""Czyszczenie treści zadania z artefaktów arkusza (devset._oczysc)."""
from matura import devset


def test_usuwa_kratki_punktacji_i_szablon_pf():
    z = {"zrodla_tekst": "", "polecenie": "Zadanie 20.2. (0–2)\n1. Oba źródła dotyczą głodu. P F\n2. Opisano kolektywizację. P / F\n20.2. 0–1–2"}
    t = devset.tresc_dla_modelu(z)
    assert "P F" not in t and "P / F" not in t and "0–1–2" not in t
    assert "(0–2)" in t and "1. Oba źródła dotyczą głodu." in t


def test_usuwa_same_etykiety_ilustracji_ale_nie_opcje():
    z = {"zrodla_tekst": "Mapa\nA B C D", "polecenie": "Zaznacz właściwą odpowiedź.\nA. Sejm Niemy.\nB. Sejm Wielki."}
    t = devset.tresc_dla_modelu(z)
    assert "A B C D" not in t and "A. Sejm Niemy." in t and "B. Sejm Wielki." in t


def test_zostawia_punktacje_w_naglowku_i_usuwa_sam_numer_kratki():
    z = {"zrodla_tekst": "", "polecenie": "Zadanie 19.2. (0–1)\nPodaj nazwisko generała.\n19.1.\n0–1–2"}
    t = devset.tresc_dla_modelu(z)
    assert "Zadanie 19.2. (0–1)" in t and "19.1." not in t.replace("19.2.", "")


def _z(polecenie, rozwiazanie):
    return {"polecenie": polecenie, "rozwiazanie": rozwiazanie, "esej": False}


def test_typ_uzupelnij_tabele_nazwami_to_otwarte():
    """2025-maj-4: „Uzupełnij tabelę” z nazwami zakonów w kluczu to zadanie otwarte (reguła zamkniętych psuła odpowiedź)."""
    assert devset.typ_zadania(_z("Uzupełnij tabelę.", "A – franciszkanie\nB – benedyktyni\nC – jezuici")) == "otwarte"
    assert devset.typ_zadania(_z("Oceń prawdziwość zdań.", "FP")) == "zamkniete"
    assert devset.typ_zadania(_z("Oceń prawdziwość zdań.", "1 – F\n2 – P\n3 – P")) == "zamkniete"
    assert devset.typ_zadania(_z("Zaznacz właściwą odpowiedź.", "B")) == "zamkniete"
    assert devset.typ_zadania(_z("Przyporządkuj.", "A2, B3")) == "zamkniete"
