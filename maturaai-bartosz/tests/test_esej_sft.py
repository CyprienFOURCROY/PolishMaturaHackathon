import pytest

from matura import esej_sft, jezyk


def test_polecenie_jak_zadanie_z_arkusza():
    temat = "Teza X. Zajmij stanowisko wobec powyższej tezy."
    assert esej_sft.polecenie(temat) == f"{esej_sft.DLUGOSC}\n\n{temat}"
    z_arkusza = f"{esej_sft.DLUGOSC}\n\n{temat}"
    assert esej_sft.polecenie(z_arkusza) == z_arkusza     # bez podwójnego wymogu długości


def test_stanowisko_na_koncu_i_walidacja():
    p = esej_sft.polecenie("Teza.", "zgadzam")
    assert p.endswith("Stanowisko: zgadzam się z tezą.")
    assert esej_sft.polecenie("Teza.", "nie zgadzam").endswith("nie zgadzam się z tezą.")
    with pytest.raises(ValueError):
        esej_sft.polecenie("Teza.", "raczej tak")


def test_przyklad_ten_sam_prompt_co_egzamin():
    ex = esej_sft.przyklad("Teza.", "  Esej.  ", "zgadzam")
    assert ex["prompt"] == esej_sft.wiadomosci("Teza.", "zgadzam")
    assert ex["prompt"][0] == {"role": "system", "content": esej_sft.SYSTEM}
    assert ex["completion"] == [{"role": "assistant", "content": "Esej."}]


def test_slowa():
    assert esej_sft.slowa("W 1410 r. pod Grunwaldem, polsko-litewskie wojska wygrały.") == 8


def test_modele_eseju_w_rejestrze():
    assert jezyk.MODELE["slayer-bielik-1.5b-sft2"][0].endswith("sft2-IQ4_XS.gguf")
    assert jezyk.MODELE["bielik-1.5b-esej-v1"][0].endswith("bielik-1.5b-esej-v1-Q8_0.gguf")


def test_zapytanie_bez_formulek_i_material_przed_tematem():
    temat = (f"{esej_sft.DLUGOSC}\n\nTeza o Jagielle. Zajmij stanowisko wobec powyższej tezy i je uzasadnij, "
             "uwzględniając w swojej argumentacji aspekty: polityczny i militarny.")
    assert esej_sft.zapytanie(temat) == "Teza o Jagielle. aspekty: polityczny i militarny."

    class Baza:
        def szukaj(self, q, n):
            assert q == esej_sft.zapytanie(temat) and n == 2
            return [{"tytul": "Unia w Krewie", "tekst": "x" * 3000}, {"tytul": "Grunwald", "tekst": "1410."}]

    m = esej_sft.material(temat, 2, baza=Baza())
    linie = m.splitlines()
    assert linie[0].startswith("- [Unia w Krewie]") and len(linie[0]) == 2 + esej_sft.ZNAKI_HASLA
    assert linie[1] == "- [Grunwald] 1410."
    p = esej_sft.polecenie(temat, "zgadzam", m)
    assert p.startswith(esej_sft.NAGLOWEK_MATERIALU) and p.endswith("zgadzam się z tezą.")
    assert esej_sft.polecenie(temat, "zgadzam", None) == esej_sft.polecenie(temat, "zgadzam")


def test_zapytanie_teza_bez_dopiskow_polecenia():
    temat = (f"{esej_sft.DLUGOSC}\n\nDecyzje przywódców w latach 1792-1794 miały większy wpływ. Zajmij stanowisko "
             "wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji trzy wybrane decyzje.")
    assert esej_sft.zapytanie_teza(temat) == "Decyzje przywódców w latach 1792-1794 miały większy wpływ."
    assert set(esej_sft.ZAPYTANIA) == {"pelny", "teza"}
