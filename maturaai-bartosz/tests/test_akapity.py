"""Dane eksperta akapitu: rozbiór tematów w obu formach CKE i zgodność promptu z harnessem."""
from matura import akapity, esej


def test_rozbior_temat_z_aspektami():
    t = ("Reformy Kazimierza Wielkiego przesądziły o rozwoju Polski w XIV wieku. Zajmij stanowisko wobec powyższej tezy "
         "i je uzasadnij, uwzględniając w swojej argumentacji aspekty: polityczny, gospodarczy i kulturowy.")
    teza, poz, wybor = akapity.rozbior(t, [])
    assert not wybor and poz == ["polityczny", "gospodarczy", "kulturowy"] and teza.startswith("Reformy")


def test_rozbior_temat_z_wyborem_wymaga_trzech_elementow():
    t = ("W XI–XII wieku dominowała decentralizacja. Zajmij stanowisko wobec powyższej tezy i je uzasadnij, "
         "uwzględniając w swojej argumentacji trzech wybranych władców z tego okresu.")
    assert akapity.rozbior(t, ["Bolesław Śmiały", "Władysław Herman", "Bolesław Krzywousty"])[2] is True
    assert akapity.rozbior(t, ["Bolesław Śmiały"]) is None


def test_rozbior_odrzuca_temat_bez_formy_cke():
    assert akapity.rozbior("Opisz dzieje Rzymu.", []) is None


def test_prompt_akapitu_taki_jak_w_harnessie():
    sys_, user = esej.prompt_akapitu("Teza X.", "polityczny", "Fakty:\n- f", "teza jest słuszna", "e3")
    assert "Nie zmieniaj stanowiska: teza jest słuszna" in sys_
    assert user == "Teza: Teza X.\nAspekt: polityczny\n\nMATERIAŁ:\nFakty:\n- f\n\nNapisz akapit."


def test_epoka_pewna_bez_sygnalu_to_none():
    from matura import router
    assert router.epoka_pewna("Warunki naturalne wpłynęły na rozwój cywilizacji starożytnego Egiptu.") == "starozytnosc-sredniowiecze"
    assert router.epoka_pewna("W XVI wieku reformacja zmieniła Europę.") == "nowozytnosc"
    assert router.epoka_pewna("Postęp techniczny zmienia życie ludzi.") is None
