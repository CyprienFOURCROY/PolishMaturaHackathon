"""Konfiguracje zadań krótkich h2 i h2_ocr (matura/harness.py): reguły odpowiedzi w prompcie systemowym wg typu zadania.

Co to jest: testy bez GPU i sieci (czat i Wikipedia to atrapy, ilustracje podmieniane przez monkeypatch).
Po co: h2 = h0 + reguły; pilnują, że każdy typ zadania dostaje swoje reguły, że prompt nie ma dosłownych przykładów
formatu (D16) i że h2_ocr bez ilustracji wysyła dokładnie prompt h2. Co zrobić: uv run pytest -q tests/test_h2.py
"""
import re

import pytest

from matura import harness, obrazy

REGULY = ("R_ZAMKNIETE", "R_ROZSTRZYGNIJ", "R_UZASADNIENIE", "R_KROTKO", "R_LICZBA", "R_ETYKIETY", "R_ILUSTRACJA")


class Wiki:
    def szukaj(self, zap, k=4):
        return [{"tytul": "Grunwald", "tekst": "Bitwa pod Grunwaldem 1410."}][:k]


def _zad(polecenie, typ="otwarte", obrazy_=(), **kw):
    return {"id": "2099-maj-9", "nr_zadania": "9", "nr_glowny": "9", "zrodla_tekst": "Źródło 1. Fragment kroniki\nTreść.",
            "polecenie": polecenie, "obrazy": list(obrazy_), "esej": False, "typ": typ, "pkt_max": 1,
            "rozwiazanie": "", **kw}


def _przechwyc(monkeypatch):
    wolania = []

    def czat(url, system, user, obr=None, **kw):
        wolania.append((system, user, obr))
        return "odp", 0.1

    monkeypatch.setattr(harness, "czat", czat)
    return wolania


ZAMKNIETE = _zad("Zadanie 9. (0–1)\nOceń prawdziwość poniższych stwierdzeń. Zaznacz P, jeśli stwierdzenie jest prawdziwe, "
                 "albo F – jeśli jest fałszywe.\n1. Zdanie pierwsze.\n2. Zdanie drugie.", typ="zamkniete")
ROZSTRZYGNIJ = _zad("Zadanie 9. (0–1)\nRozstrzygnij, czy źródło dotyczy bitwy pod Grunwaldem. Odpowiedź uzasadnij, "
                    "odwołując się do jednego elementu źródła.\nRozstrzygnięcie:\nUzasadnienie:")
PODAJ = _zad("Zadanie 9. (0–1)\nPodaj nazwy dwóch urzędów republikańskich, o których mowa w tekście.")
ETYKIETY = _zad("Zadanie 9. (0–1)\nPodaj nazwę wydarzenia, do którego nawiązuje rycina. Odpowiedź uzasadnij, odwołując "
                "się do jej treści.\nWydarzenie:\nUzasadnienie:")


def test_zamkniete_zawsze_wybierz_bez_regul_otwartych():
    s = harness.system_h2(ZAMKNIETE)
    assert s.startswith(harness.SYS_ZAMKNIETE) and harness.R_ZAMKNIETE in s
    assert "Zawsze wybierz odpowiedź" in s and "nie odejmuje się punktów" in s
    for r in (harness.R_UZASADNIENIE, harness.R_ROZSTRZYGNIJ, harness.R_KROTKO):
        assert r not in s


def test_rozstrzygnij_format_zrodlo_i_fakt():
    s = harness.system_h2(ROZSTRZYGNIJ)
    assert s.startswith(harness.SYS_ROZSTRZYGNIJ)  # format „Rozstrzygnięcie:” / „Uzasadnienie:” z h0
    assert harness.R_ROZSTRZYGNIJ in s and harness.R_UZASADNIENIE in s and harness.R_ZAMKNIETE not in s
    assert "informację ze źródła" in s and "jeden konkretny fakt z wiedzy" in s
    assert "(„jednego”)" in s  # liczba z polecenia
    assert "Zapisz odpowiedź w osobnych liniach" not in s  # etykiety rozstrzygnięcia już są w SYS_ROZSTRZYGNIJ


def test_podaj_krotko_i_dokladna_liczba():
    s = harness.system_h2(PODAJ)
    assert s.startswith(harness.SYS_OTWARTE) and harness.R_KROTKO in s and "(„dwóch”)" in s
    assert "dokładnie tyle elementów" in s and harness.R_UZASADNIENIE not in s


def test_etykiety_linii_z_polecenia():
    s = harness.system_h2(ETYKIETY)
    assert "„Wydarzenie:”, „Uzasadnienie:”" in s and harness.R_UZASADNIENIE in s and harness.R_ROZSTRZYGNIJ not in s


def test_esej_bez_regul():
    z = {"id": "e", "esej": True, "typ": "esej", "polecenie": "Temat", "obrazy": []}
    assert harness.system_h2(z) == harness.SYS_ESEJ


def test_ilustracja_niewidoczna_dostaje_regule(monkeypatch):
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: [{"rel": "x.png"}])
    z = dict(PODAJ, obrazy=["x.png"])
    assert harness.R_ILUSTRACJA in harness.system_h2(z, widzi_obraz=False)
    assert harness.R_ILUSTRACJA not in harness.system_h2(z, widzi_obraz=True)
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: [])  # obraz devsetu nie należy do zadania (poprawione przypisanie)
    assert harness.R_ILUSTRACJA not in harness.system_h2(z, widzi_obraz=False)
    assert harness.R_ILUSTRACJA not in harness.system_h2(PODAJ)  # brak obrazów


def test_regul_bez_doslownych_przykladow_formatu():
    for n in REGULY:
        r = getattr(harness, n)
        assert "1. P, 2. F" not in r and not re.search(r"\b\d\.\s*[PF]\b", r), n
    for z in (ZAMKNIETE, ROZSTRZYGNIJ, PODAJ, ETYKIETY):
        s = harness.system_h2(z)
        assert "1. P, 2. F" not in s and not re.search(r"\b\d\.\s*[PF]\b", s)


def test_h2_przez_harness_to_h0_plus_reguly(monkeypatch):
    w = _przechwyc(monkeypatch)
    for k in ("h0", "h2"):
        harness.odpowiedz(k, ZAMKNIETE, url="u", wiki=Wiki(), obrazy=False, bez_myslenia=True)
    (sys_h0, user_h0, _), (sys_h2, user_h2, _) = w
    assert user_h2 == user_h0 and "Fragmenty encyklopedii" in user_h2  # ten sam RAG (k=4) i treść zadania
    assert sys_h2 == sys_h0 + "\n" + harness.R_ZAMKNIETE


def test_h2_ocr_bez_ilustracji_to_dokladnie_h2(monkeypatch):
    w = _przechwyc(monkeypatch)
    for z in (ZAMKNIETE, ROZSTRZYGNIJ, PODAJ):
        _, _, m_h2 = harness.odpowiedz("h2", z, url="u", wiki=Wiki(), obrazy=False, bez_myslenia=True)
        _, _, m_ocr = harness.odpowiedz("h2_ocr", z, url="u", wiki=Wiki(), obrazy=False, bez_myslenia=True)
        assert w[-1] == w[-2] and m_ocr["ilustracje_opisane"] == 0


def test_h2_ocr_z_ilustracja_blok_przed_zadaniem_i_bez_obrazu(tmp_path, monkeypatch):
    monkeypatch.setattr(obrazy, "CACHE", tmp_path / "opisy.jsonl")
    monkeypatch.setattr(obrazy, "_CACHE", None)
    img = str(tmp_path / "m.png")
    from PIL import Image
    Image.new("RGB", (20, 20), "white").save(img)
    il = [{"sciezka": img, "rel": img, "zrodlo": 1, "podpis": "Źródło 1. Mapa", "i": 1, "n": 1}]
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: il)
    obrazy._zapisz(img, "ocr", [{"t": "Toruń", "p": 0.97, "b": [0, 0, 10, 5]}], 0.1)
    w = _przechwyc(monkeypatch)
    z = dict(PODAJ, obrazy=[img])
    harness.odpowiedz("h2_ocr", z, url="u", wiki=Wiki(), obrazy=True, bez_myslenia=True)
    harness.odpowiedz("h2", z, url="u", wiki=Wiki(), obrazy=True, bez_myslenia=True)
    (sys_ocr, user_ocr, obr_ocr), (sys_h2, user_h2, obr_h2) = w
    assert obr_ocr is None and obr_h2 == [img]
    assert harness.R_ILUSTRACJA in sys_ocr and harness.R_ILUSTRACJA not in sys_h2  # h2 z obrazem: model go widzi
    assert "OPIS ILUSTRACJI (" in user_ocr and user_ocr.index("OPIS ILUSTRACJI") < user_ocr.index("ZADANIE:")
    assert user_ocr.replace(user_ocr[user_ocr.index(obrazy.NAGLOWEK_BLOKU):user_ocr.index("ZADANIE:")], "") == user_h2


def test_h2_bez_wikipedii_to_blad():
    with pytest.raises(ValueError, match="KONFIGI_Z_WIKI"):
        harness.odpowiedz("h2", PODAJ, url="u", wiki=None, obrazy=False, bez_myslenia=True)


def test_zaden_prompt_systemowy_nie_zawiera_dosłownego_pf():
    """Bielik-1.5B przepisywał dosłowne „P/F” z promptu w pętli (etap A, 26.09): prompt opisuje format słowami."""
    for z in (ZAMKNIETE, ROZSTRZYGNIJ, PODAJ, ETYKIETY):
        assert "P/F" not in harness.system_dla(z) and "P/F" not in harness.system_h2(z)


def test_h2_zamkniete_krotki_limit_tokenow(monkeypatch):
    limity = []
    monkeypatch.setattr(harness, "czat", lambda url, system, user, obr=None, **kw: (limity.append(kw["max_tokens"]), ("A", 0.1))[1])
    harness.odpowiedz("h2", ZAMKNIETE, url="x", wiki=Wiki(), obrazy=False, bez_myslenia=True)
    harness.odpowiedz("h2", PODAJ, url="x", wiki=Wiki(), obrazy=False, bez_myslenia=True)
    assert limity == [harness.MAX_TOKENS_ZAMKNIETE, 400]


def test_goly_vlm_dokleja_opis_ilustracji_do_golego_promptu(monkeypatch):
    wolania = _przechwyc(monkeypatch)
    monkeypatch.setattr(obrazy, "blok_opisu", lambda z, w: f"OPIS ILUSTRACJI (źródło 1.): mapa [{w}]")
    harness.odpowiedz("goly_vlm", PODAJ, url="x", wiki=None, obrazy=True, bez_myslenia=True)
    system, user, obr = wolania[0]
    assert system == harness.SYS_GOLY and obr is None
    assert user.startswith("OPIS ILUSTRACJI (źródło 1.): mapa [vlm_q2b_en]")


def test_goly_wiedza_dokleja_fakty_wzorcowe_i_bez_wpisu_zglasza_blad(monkeypatch):
    wolania = _przechwyc(monkeypatch)
    monkeypatch.setattr(harness, "_WIEDZA_WZORCOWA", {PODAJ["id"]: ["Konsulowie byli najwyższymi urzędnikami Rzymu."]})
    harness.odpowiedz("goly_wiedza", PODAJ, url="x", wiki=None, obrazy=False, bez_myslenia=True)
    assert wolania[0][1].startswith("Wiedza pomocnicza:\n- Konsulowie")
    with pytest.raises(KeyError):
        harness.odpowiedz("goly_wiedza", {**PODAJ, "id": "2099-maj-1"}, url="x", wiki=None, obrazy=False, bez_myslenia=True)
