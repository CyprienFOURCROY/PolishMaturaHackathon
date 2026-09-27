"""Testy głosowania w zadaniach „rozstrzygnij” (matura/rozstrzygnij.py)."""
from matura import rozstrzygnij


def test_czy_rozstrzygnij():
    assert rozstrzygnij.czy_rozstrzygnij({"typ": "otwarte", "polecenie": "Rozstrzygnij, czy mapa przedstawia…"})
    assert not rozstrzygnij.czy_rozstrzygnij({"typ": "zamkniete", "polecenie": "Rozstrzygnij, czy…"})
    assert not rozstrzygnij.czy_rozstrzygnij({"typ": "otwarte", "polecenie": "Wyjaśnij przyczynę…"})


def test_decyzja():
    assert rozstrzygnij.decyzja("**Rozstrzygnięcie:** TAK, mapa przedstawia…\nUzasadnienie: …") == "tak"
    assert rozstrzygnij.decyzja("Rozstrzygnięcie:\nNie, rozkaz nie dotyczy…") == "nie"
    assert rozstrzygnij.decyzja("Rozstrzygnięcie: Wersja **B** ma wymowę zgodną…") == "B"
    assert rozstrzygnij.decyzja("Rozstrzygnięcie: fragment A jest późniejszy") == "A"
    assert rozstrzygnij.decyzja("Pomnik jest przykładem sztuki baroku.") == "pomnik jest przykładem sztuki baroku"
    assert rozstrzygnij.decyzja("") is None


def test_glosowanie_wybiera_wiekszosc():
    probki = iter(["Rozstrzygnięcie: Tak.\nUzasadnienie: A.", "Rozstrzygnięcie: Nie.\nUzasadnienie: B.",
                   "Rozstrzygnięcie: Nie.\nUzasadnienie: C."])
    temp = []

    def czat(url, system, user, obrazy, **kw):
        temp.append(kw["temperature"])
        return next(probki), 0.1

    t, _, meta = rozstrzygnij.odpowiedz("http://x", "Treść", czat_fn=czat)
    assert t.startswith("Rozstrzygnięcie: Nie.") and "B." in t and temp == [0.0, 0.7, 0.7]
    assert meta["decyzje"] == ["tak", "nie", "nie"]


def test_remis_zostaje_pierwsza_probka():
    probki = iter(["Rozstrzygnięcie: A", "Rozstrzygnięcie: B", "brak"])

    def czat(url, system, user, obrazy, **kw):
        return next(probki), 0.1

    t, _, _ = rozstrzygnij.odpowiedz("http://x", "Treść", czat_fn=czat)
    assert t == "Rozstrzygnięcie: A"
