"""Testy zadań zamkniętych (matura/zamkniete.py): forma odpowiedzi, odczyt ostatniej linii, głosowanie."""
from matura import zamkniete


def test_forma_z_answer_format():
    assert zamkniete.forma({"answer_format": "A", "typ": "zamkniete"}) == {"rodzaj": "jedna"}
    assert zamkniete.forma({"answer_format": "1: P\n2: F\n3: P"}) == {"rodzaj": "pary", "typ": "pf", "etykiety": ["1", "2", "3"]}
    assert zamkniete.forma({"answer_format": "A: 1\nB: 2"}) == {"rodzaj": "pary", "typ": "cyfra", "etykiety": ["A", "B"]}
    assert zamkniete.forma({"answer_format": "1: A\n2: B"})["typ"] == "litera"


def test_forma_z_klucza_bez_wartosci():
    z = {"typ": "zamkniete", "rozwiazanie": "1 – F\n2 – P\n3 – P", "polecenie": "Oceń prawdziwość", "pkt_max": 2}
    f = zamkniete.forma(z)
    assert f["rodzaj"] == "pary" and f["typ"] == "pf" and f["etykiety"] == ["1", "2", "3"]
    assert "F" not in str(f.get("etykiety"))  # struktura, nie odpowiedź


def test_odczyt_ostatniej_linii():
    f = {"rodzaj": "pary", "typ": "pf", "etykiety": ["1", "2", "3"]}
    assert zamkniete.odczytaj("Zdanie 1 jest zgodne ze źródłem.\nODPOWIEDŹ: P, F, prawda", f) == ["P", "F", "P"]
    assert zamkniete.odczytaj("ODPOWIEDZ: True, False, True", f) == ["P", "F", "P"]
    assert zamkniete.odczytaj("ODPOWIEDŹ: P, F", f) is None
    assert zamkniete.odczytaj("Uzasadnienie B jest błędne.\nODPOWIEDŹ: C", {"rodzaj": "jedna"}) == ["C"]
    fp = {"rodzaj": "pary", "typ": "cyfra", "etykiety": ["A", "B"]}
    assert zamkniete.odczytaj("ODPOWIEDŹ: A=3, B=2", fp) == ["3", "2"]


def test_glosowanie_i_format():
    f = {"rodzaj": "pary", "typ": "pf", "etykiety": ["1", "2"]}
    odp = iter(["ODPOWIEDŹ: P, F", "ODPOWIEDŹ: F, F", "ODPOWIEDŹ: P, P"])
    wywolania = []

    def czat(url, system, user, obrazy, **kw):
        wywolania.append(kw["temperature"])
        return next(odp), 0.1

    t, s, meta = zamkniete.odpowiedz(f, "http://x", "Treść zadania", czat_fn=czat)
    assert t == "1: P\n2: F" and wywolania == [0.0, 0.7, 0.7] and meta["glosy"] == 3


def test_brak_odczytu_zwraca_pierwsza_probe():
    def czat(url, system, user, obrazy, **kw):
        return "Nie wiem.", 0.1

    t, _, meta = zamkniete.odpowiedz({"rodzaj": "jedna"}, "http://x", "Treść", czat_fn=czat)
    assert t == "Nie wiem." and meta["glosy"] == 0
