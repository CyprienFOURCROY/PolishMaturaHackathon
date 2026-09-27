"""Testy lokalnego tłumacza (matura/tlumacz.py): podział na segmenty, backend „baza” z atrapą czatu."""
import pytest

from matura import tlumacz


def test_segmenty_zachowuja_uklad_linii():
    t = "Źródło 1. Fragment kroniki.\nKról zwołał sejm. Szlachta przybyła.\n\nZadanie 1. Wyjaśnij przyczynę."
    seg = tlumacz.segmenty(t)
    assert [len(l) for l in seg] == [2, 2, 0, 2]
    assert tlumacz.sklej(seg, [s.upper() for l in seg for s in l]) == (
        "ŹRÓDŁO 1. FRAGMENT KRONIKI.\nKRÓL ZWOŁAŁ SEJM. SZLACHTA PRZYBYŁA.\n\nZADANIE 1. WYJAŚNIJ PRZYCZYNĘ.")


def test_segmenty_dziela_dlugie_zdanie_po_przecinkach():
    zdanie = ", ".join(["słowo " * 30] * 4).strip() + "."
    seg = tlumacz.segmenty(zdanie, max_znakow=400)
    assert all(len(s) <= 400 for l in seg for s in l)
    assert tlumacz.sklej(seg, [s for l in seg for s in l]).replace(" ", "") == zdanie.replace(" ", "")


def test_baza_wysyla_kontekst_i_zwraca_tlumaczenia():
    wyslane = []

    def czat(url, system, user, obrazy, **kw):
        wyslane.append((system, user, kw))
        return "PRZEKŁAD: " + user.split("<<<\n", 1)[1].split("\n>>>", 1)[0], 0.1

    b = tlumacz.Baza("http://x", bez_myslenia=True, czat_fn=czat, rownolegle=2)
    wyn = b.tlumacz(["Jagiełło signed the union."], "en-pl", konteksty=["Wyjaśnij, dlaczego Jagiełło zawarł unię w Krewie."])
    assert wyn == ["Jagiełło signed the union."]  # etykieta „PRZEKŁAD:” usunięta, tekst = ostatnia linia promptu
    system, user, kw = wyslane[0]
    assert "unię w Krewie" in user and kw["temperature"] == 0.0 and kw["bez_myslenia"] is True
    assert "polski" in system.lower() or "polish" in system.lower()


def test_baza_czysci_prefiks_i_pusty_tekst():
    b = tlumacz.Baza("http://x", czat_fn=lambda *a, **k: ("Translation: Hello.", 0.0), rownolegle=1)
    assert b.tlumacz(["", "Cześć."], "pl-en") == ["", "Hello."]


def test_nieznany_kierunek():
    b = tlumacz.Baza("http://x", czat_fn=lambda *a, **k: ("x", 0.0))
    try:
        b.tlumacz(["a"], "de-pl")
    except ValueError:
        return
    raise AssertionError("oczekiwano ValueError")


class _Atrapa:
    """Tłumacz testowy: zapisuje wywołania, zwraca tekst z prefiksem kierunku."""

    def __init__(self):
        self.wywolania = []

    def tlumacz(self, teksty, kierunek, konteksty=None):
        self.wywolania.append((list(teksty), kierunek, konteksty))
        return [f"[{kierunek}] {t}" for t in teksty]


def test_pliki_cache_wg_tlumacza(monkeypatch):
    from matura import jezyk
    monkeypatch.setattr(jezyk, "TL_ZADAN", "claude"); monkeypatch.setattr(jezyk, "TL_ODP", "claude")
    assert jezyk.plik_zadan_en().name == "zadania_en.jsonl" and jezyk.plik_tl_odp().name == "odpowiedzi_tl.jsonl"
    monkeypatch.setattr(jezyk, "TL_ZADAN", "baza"); monkeypatch.setattr(jezyk, "TL_ODP", "marian")
    assert jezyk.plik_zadan_en("m1").name == "zadania_en__baza__m1.jsonl"
    assert jezyk.plik_tl_odp().name == "odpowiedzi_tl__marian.jsonl"


def test_odpowiedzi_zamkniete_bez_tlumaczenia(tmp_path):
    from matura import jezyk
    c = jezyk.Cache(tmp_path / "tl.jsonl")
    poz = [{"id": "z1", "tekst": "A", "tresc_pl": "Zadanie zamknięte"},
           {"id": "z2", "tekst": "Because of the union.", "tresc_pl": "Wyjaśnij unię w Krewie."},
           {"id": "z2", "tekst": "Because of the union.", "tresc_pl": "Wyjaśnij unię w Krewie."},
           {"id": "z3", "tekst": "(BŁĄD: timeout)", "tresc_pl": "x"}]
    tl = _Atrapa()
    jezyk.tlumacz_odpowiedzi_lokalnie(poz, tl, c, "atrapa", zamkniete={"z1"})
    assert tl.wywolania == [(["Because of the union."], "en-pl", ["Wyjaśnij unię w Krewie."])]
    assert jezyk.tlumaczenie_z_cache(c, "en-pl", "z1", "A") == "A"
    assert jezyk.tlumaczenie_z_cache(c, "en-pl", "z2", "Because of the union.") == "[en-pl] Because of the union."
    assert jezyk.tlumaczenie_z_cache(c, "en-pl", "z3", "(BŁĄD: timeout)") == "(BŁĄD: timeout)"
    jezyk.tlumacz_odpowiedzi_lokalnie(poz, tl, c, "atrapa", zamkniete={"z1"})
    assert len(tl.wywolania) == 1  # drugi przebieg: wszystko z cache


def test_zadania_lokalnie_do_cache(tmp_path, monkeypatch):
    from matura import jezyk
    monkeypatch.setattr(jezyk, "tekst_pl", lambda z: z["t"])
    zz = [{"id": "a", "t": "Treść A."}, {"id": "b", "t": "Treść B."}]
    tl = _Atrapa()
    out = jezyk.tlumacz_zadania_lokalnie(zz, tl, tmp_path / "zad.jsonl", "atrapa", porcja=1)
    assert out == {"a": "[pl-en] Treść A.", "b": "[pl-en] Treść B."} and len(tl.wywolania) == 2
    assert jezyk.tlumacz_zadania_lokalnie(zz, tl, tmp_path / "zad.jsonl", "atrapa") == out and len(tl.wywolania) == 2


def test_blok_kb_en_tekst_angielski_i_zapas_polski(monkeypatch):
    from matura import jezyk, harness

    class _Baza:
        h = [{"id": "h1"}, {"id": "h2"}]

        def szukaj(self, q, n):
            return [{"id": "h1", "tytul": "Unia w Krewie", "tekst": "Unia 1385."},
                    {"id": "h2", "tytul": "Grunwald", "tekst": "Bitwa 1410."}][:n]

    monkeypatch.setattr(harness, "_baza_hasel", lambda: _Baza())
    monkeypatch.setattr(jezyk, "_HASLA_EN", {"h1": "[Union of Krewo] The union of 1385."})
    b = jezyk.blok_kb_en({"polecenie": "Wyjaśnij unię.", "zrodla_tekst": ""})
    assert b.startswith(jezyk.NAGLOWEK_KB_EN) and "- [Union of Krewo] The union of 1385." in b
    assert "- [Grunwald] Bitwa 1410." in b and b.endswith("\n\n")


def test_zrodla_wg_opcji(monkeypatch):
    from matura import jezyk
    monkeypatch.setattr(jezyk, "KB", False)
    assert jezyk._zrodla(False) == ("goly_en",)
    monkeypatch.setattr(jezyk, "KB", True)
    assert jezyk._zrodla(True) == ("goly_en", "goly_vlm_en", "goly_en_kb")


def test_baza_zbyt_krotkie_tlumaczenie_po_akapitach():
    """Model „odpowiada” zamiast tłumaczyć (próba 26.09: 2025-maj-24.2 → „Poland”): ponowienie po akapitach,
    akapit nadal zbyt krótki zostaje po polsku."""
    zrodlo = "Źródło 1. " + "Ekipa Jaruzelskiego zachowała kontrolę nad państwem. " * 8
    polecenie = "Zadanie 24. Podaj nazwę państwa, o którym mowa w źródle."
    tekst = zrodlo.strip() + "\n" + polecenie

    def czat(url, system, user, obrazy, **kw):
        t = user.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
        if "\n" in t:
            return "Poland", 0.0                                  # cały tekst: model odpowiada na zadanie
        return ("Poland" if t.startswith("Zadanie") else "EN: " + t), 0.0   # akapit polecenia nadal „odpowiedziany”

    b = tlumacz.Baza("http://x", czat_fn=czat, rownolegle=1)
    wyn = b.tlumacz([tekst], "pl-en")[0]
    linie = wyn.split("\n")
    assert linie[0].startswith("EN: Źródło 1.") and linie[1] == polecenie


def test_marian_no_na_koncu_zdania(monkeypatch):
    m = tlumacz.Marian()
    wyslane = []
    monkeypatch.setattr(m, "_zdania", lambda zd, k: (wyslane.extend(zd), list(zd))[1])
    m.tlumacz(["Decision: No.\nNo, it was not. Answer: **No**"], "en-pl")
    assert wyslane == ["Decision: Nie.", "No, it was not.", "Answer: **Nie**"]


def test_marian_wariant_allegro_en_pl(monkeypatch):
    m = tlumacz.Marian(en_pl="allegro")
    assert m.sciezki["en-pl"][0].name == "bidi-eng-pol" and m.sciezki["en-pl"][1] == ">>pol<< "
    assert m.sciezki["pl-en"][0].name == "opus-mt-pl-en"
    assert tlumacz.Marian().sciezki["en-pl"][0].name == "opus-mt-en-zlw"
    wyslane = []
    monkeypatch.setattr(m, "_zdania", lambda zd, k: (wyslane.extend(zd), list(zd))[1])
    m.tlumacz(["Decision: No."], "en-pl")
    assert wyslane == ["Decision: Nie."]                     # poprawka „No” także w wariancie Allegro
    with pytest.raises(ValueError):
        tlumacz.Marian(en_pl="google")


def test_plik_zadan_osobnego_modelu(monkeypatch):
    from matura import jezyk
    monkeypatch.setattr(jezyk, "TL_ZADAN", "qwen3.5-2b-q4-bezwizji")
    assert jezyk.plik_zadan_en("inna-baza").name == "zadania_en__model__qwen3.5-2b-q4-bezwizji.jsonl"
