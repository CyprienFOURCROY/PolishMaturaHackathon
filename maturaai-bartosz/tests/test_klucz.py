"""Testy automatycznej oceny zadań z kluczem (matura/klucz.py) na formatach z arkuszy CKE i odpowiedziach modeli z nocy."""
import pytest

from matura import klucz


def zad(rozw, polecenie="Zaznacz właściwą odpowiedź.", pkt=1, typ="zamkniete"):
    return {"rozwiazanie": rozw, "polecenie": polecenie, "pkt_max": pkt, "typ": typ, "esej": False}


@pytest.mark.parametrize("odp,pkt", [
    ("B", 1), ("C", 0), ("B.", 1), ("D. reformy Solona w Atenach.", 0), ("**B**", 1),
    ("Odpowiedź: B, ponieważ Aleksander zmarł w 323 r.", 1), ("Prawidłowa odpowiedź to C.", 0),
])
def test_jedna_litera(odp, pkt):
    assert klucz.ocen(zad("B"), odp)["pkt"] == pkt


def test_jedna_litera_gadatliwa_nierozstrzygnieta():
    o = klucz.ocen(zad("B"), "Zacznijmy od analizy fragmentu. Aleksander Wielki był królem Macedonii.")
    assert o["pewne"] is False and o["pkt"] is None


@pytest.mark.parametrize("odp,pkt", [
    ("1. F\n2. P\n3. P", 2), ("1. P – II wojna punicka\n2. F – edyle\n3. F – bitwa", 0),
    ("1. **F**  \nWyspa...\n2. **P**\n3. **F**", 1), ("1 – F, 2 – P, 3 – P", 2), ("FPP", 2),
])
def test_prawda_falsz_2pkt(odp, pkt):
    assert klucz.ocen(zad("1 – F\n2 – P\n3 – P", "Oceń prawdziwość zdań.", pkt=2), odp)["pkt"] == pkt


def test_pf_zwarty_klucz():
    z = zad("FP", "Oceń prawdziwość informacji.")
    assert klucz.ocen(z, "1. F\n2. P")["pkt"] == 1
    assert klucz.ocen(z, "FF")["pkt"] == 0


@pytest.mark.parametrize("odp,pkt", [
    ("1. C\n2. B\n3. A\n4. D", 1), ("C – 1, B – 2, A – 3, D – 4", 1), ("C, B, A, D", 1),
    ("1. D. Kapitulacja Warszawy\n2. A. Bitwa pod Ostrołęką\n3. C. Atak\n4. B. Bitwa pod Grochowem", 0),
    ("C. Atak podchorążych na Belweder.\nB. Bitwa pod Grochowem.\nA. Bitwa pod Ostrołęką.\nD. Kapitulacja.", 1),
])
def test_kolejnosc(odp, pkt):
    z = zad("C – 1, B – 2, A – 3, D – 4", "Uporządkuj chronologicznie wydarzenia.")
    assert klucz.ocen(z, odp)["pkt"] == pkt


def test_kolejnosc_ciag_liter_w_kluczu():
    z = zad("B, A, D, C", "Uporządkuj chronologicznie.")
    assert klucz.ocen(z, "B, A, D, C")["pkt"] == 1 and klucz.ocen(z, "A, B, C, D")["pkt"] == 0


def test_przyporzadkowanie_tekstowe():
    z = zad("Fragment A – Karol IX\nFragment B – Henryk IV", "Każdemu fragmentowi przyporządkuj władcę.")
    assert klucz.ocen(z, "Fragment A – Karol IX\nFragment B – Henryk IV")["pkt"] == 1
    assert klucz.ocen(z, "Fragment A – Franciszek II\nFragment B – Henryk IV")["pewne"] is False


def test_zakony_synonim_idzie_do_sedziego():
    z = zad("A – franciszkanie\nB – benedyktyni\nC – jezuici", "Uzupełnij tabelę.", pkt=2)
    o = klucz.ocen(z, "A. Zakon Braci Mniejszych\nB. Zakon Benedyktynów\nC. Towarzystwo Jezusowe (Jezuici)")
    assert o["pewne"] is False


@pytest.mark.parametrize("rozw,odp,pkt", [
    ("[Ignacy] Łukasiewicz", "Ignacy Łukasiewicz", 1), ("[Ignacy] Łukasiewicz", "Maria", None),
    ("[Ignacy] Łukasiewicz", "Łukasiewicza przedstawiono na monecie.", 1),
    ("Hohenzollernowie", "Z dynastii Hohenzollernów.", 1), ("Rumunia", "W Rumunii.", 1),
    ("Zygmunt III [Waza], Zygmunt Waza", "Zygmunt III Waza", 1), ("[Fryderyk] Engels, [Karol] Marks", "Karol Marks", 1),
    ("konsul\ntrybun ludowy", "konsul, trybun ludowy", 1), ("abdykacja / zrzeczenie się tronu", "abdykacja", 1),
    ("1 sierpnia [1944 r.]", "1 sierpnia 1944", 1), ("Wystawca – Jan\nZasadźca – Marcin", "Wystawca: Jan\nZasadźca: Marcin", 1),
])
def test_podaj_nazwe(rozw, odp, pkt):
    z = zad(rozw, "Podaj nazwisko postaci.", typ="otwarte")
    assert klucz.obslugiwane(z)
    assert klucz.ocen(z, odp)["pkt"] == pkt


def test_blad_serwera_to_zero():
    o = klucz.ocen(zad("B"), "(BŁĄD: 400 Client Error: Bad Request)")
    assert o["pkt"] == 0 and o["pewne"]


def test_otwarte_dlugie_nieobslugiwane():
    assert not klucz.obslugiwane(zad("Wyjaśnij przyczyny upadku..." * 3, "Wyjaśnij, dlaczego...", typ="otwarte"))


def test_przepisany_szablon_pf_to_bledne_wskazanie():
    z = zad("1 – F\n2 – P\n3 – F", "Oceń prawdziwość zdań.", pkt=2)
    assert klucz.ocen(z, "P F")["pkt"] == 0
    assert klucz.ocen(z, "1. P F\n2. P F\n3. P F")["pkt"] == 0
    assert klucz.ocen(z, "1. **P / F**  \nuzasadnienie\n2. F\n3. F")["pewne"] is False  # brak decyzji przy 1 → sędzia
    assert klucz.ocen(z, "1. **P / F**\n   Wyspa…\n   **F** – fałszywa.\n2. P F\n   Odpowiedź: P\n3. F")["pkt"] == 2


def test_dwie_rozne_litery_niejednoznaczne():
    o = klucz.ocen(zad("A"), "Plik zawiera fragmenty.\nDokładnie odpowiedź: **A**\nDokładnie odpowiedź: **D**")
    assert o["pewne"] is False


@pytest.mark.parametrize("odp,pkt", [
    ("Odpowiedź 11.2: **C. Sejmu Wielkiego.**", 1), ("Dokończ zdanie: **A. Sejmu Niemego.** Zasada…", 0),
    ("Odpowiedź do zadania 11.2: C", 1),
])
def test_litera_po_wstepie_z_numerem(odp, pkt):
    assert klucz.ocen(zad("C"), odp)["pkt"] == pkt
