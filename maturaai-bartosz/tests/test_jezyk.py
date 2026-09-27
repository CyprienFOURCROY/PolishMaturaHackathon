"""Testy eksperymentu językowego (matura/jezyk.py) bez GPU i bez sieci: T/F → P/F, parsowanie odpowiedzi tłumacza,
partie, wznawianie generacji i tłumaczeń (czat i tłumacz podmienione), wybór sondy, bootstrap, format pm_zestawy."""
import json

import numpy as np
import pytest

from matura import devset, jezyk, pelna_matura

jest_cke = pytest.mark.skipif(not (devset.JSON / "historia-2025-maj.json").exists(), reason="brak danych CKE")


# ---------------------------------------------------------------- T/F → P/F

@pytest.mark.parametrize("wej, wyj", [
    ("1. T\n2. F\n3. T", "1. P\n2. F\n3. P"),
    ("1 – T; 2 – F", "1 – P; 2 – F"),
    ("TFT", "PFP"),
    ("**T**", "**P**"),
    ("T/F", "P/F"),
    ("1. True\n2. False", "1. Prawda\n2. Fałsz"),
    ("1. P\n2. F", "1. P\n2. F"),                       # już po polsku: bez zmian
])
def test_tf_na_pf(wej, wyj):
    assert jezyk.tf_na_pf(wej) == wyj


def test_tf_na_pf_nie_rusza_wyrazow_ani_inicjalow():
    t = "Traktat w Toruniu, T. Kościuszko, Tatarzy; odpowiedź B, rok 1410"
    assert jezyk.tf_na_pf(t) == t


@jest_cke
def test_czy_pf_na_arkuszu():
    zz = {z["id"]: z for z in jezyk.zadania(["2025-maj"])}
    pf = [i for i, z in zz.items() if jezyk.czy_pf(z)]
    assert pf, "arkusz 2025-maj ma zadania prawda/fałsz"
    assert all(zz[i]["typ"] == "zamkniete" for i in pf)
    assert not any(jezyk.czy_pf(z) for z in zz.values() if z["esej"])


# ---------------------------------------------------------------- zadania eksperymentu 1

@jest_cke
def test_zadania_to_te_same_co_pelna_matura():
    zz = jezyk.zadania(["2024-maj", "2025-maj"])
    oczek = []
    for s in ("2024-maj", "2025-maj"):
        a = pelna_matura.zadania_sesji(s)
        oczek += [z["id"] for z in a["krotkie"]] + [t["id"] for t in a["tematy"]]
    assert [z["id"] for z in zz] == oczek
    assert {"2024-maj-26-t1", "2025-maj-25-t3"} <= {z["id"] for z in zz}
    assert not any(z["esej"] and "-t" not in z["id"] for z in zz)   # samo zadanie esejowe (bez tematu) nie wchodzi


# ---------------------------------------------------------------- parsowanie i partie

def test_rozpakuj_tylko_oczekiwane_id():
    w = {"tlumaczenia": [{"id": "a", "tekst": "  jeden \n"}, {"id": "x", "tekst": "obce"}, {"id": "b", "tekst": "dwa"},
                         {"id": "a", "tekst": "duplikat"}, {"id": "c"}]}
    assert jezyk.rozpakuj(w, ["a", "b", "c"]) == {"a": "jeden", "b": "dwa"}
    assert jezyk.rozpakuj({}, ["a"]) == {}
    assert jezyk.rozpakuj(None, ["a"]) == {}


def test_partie_wg_znakow():
    assert jezyk.partie_wg_znakow([("a", 5), ("b", 5), ("c", 5)], 10) == [["a", "b"], ["c"]]
    assert jezyk.partie_wg_znakow([("duzy", 50), ("a", 1)], 10) == [["duzy"], ["a"]]
    assert jezyk.partie_wg_znakow([], 10) == []


def test_partie_odpowiedzi_kontekst_raz_na_zadanie():
    poz = [{"id": "z1", "tekst": "x" * 10, "tresc_pl": "p" * 100, "tresc_en": "e" * 100} for _ in range(3)]
    poz += [{"id": "z2", "tekst": "x" * 10, "tresc_pl": "p" * 100, "tresc_en": "e" * 100}]
    # z1: 200 kontekstu + 3×10 = 230; z2 dokłada 210 → 440
    assert [len(p) for p in jezyk.partie_odpowiedzi(poz, 440)] == [4]
    assert [len(p) for p in jezyk.partie_odpowiedzi(poz, 300)] == [3, 1]


# ---------------------------------------------------------------- wznawianie

def _zad(i, esej=False):
    return {"id": i, "esej": esej}


def test_generacja_wznawia_i_ponawia_bledy(tmp_path):
    wywolania = []

    def czat(url, system, user, obrazy, *, max_tokens, bez_myslenia, temperature):
        wywolania.append((user, max_tokens, bez_myslenia, obrazy, temperature))
        return f"odp:{user}", 0.1

    poz = [(_zad("a"), "A"), (_zad("b"), "B"), (_zad("e", esej=True), "E")]
    p = jezyk.plik_odp("m", "goly_pl", tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"id": "a", "odpowiedz": "gotowe"}) + "\n"
                 + json.dumps({"id": "b", "odpowiedz": "(BŁĄD: timeout)"}) + "\n", encoding="utf-8")
    jezyk.generuj_warunek("m", "goly_pl", poz, "SYS", "http://x", True, 2, wyn=tmp_path, czat_fn=czat)
    assert sorted(w[0] for w in wywolania) == ["B", "E"]          # a gotowe, b ponowione po błędzie
    assert dict((w[0], w[1]) for w in wywolania) == {"B": 400, "E": 1400}
    assert dict((w[0], w[4]) for w in wywolania) == {"B": jezyk.TEMPERATURA["krotkie"], "E": jezyk.TEMPERATURA["esej"]}
    assert all(w[3] is None and w[2] is True for w in wywolania)  # bez obrazów, bez_myslenia przekazane
    d = jezyk.wczytaj_odp(p)
    assert d["b"]["odpowiedz"] == "odp:B" and d["a"]["odpowiedz"] == "gotowe"
    assert set(d["e"]) == {"id", "model", "konfig", "odpowiedz", "sekundy"} and d["e"]["konfig"] == "goly_pl"
    wywolania.clear()
    jezyk.generuj_warunek("m", "goly_pl", poz, "SYS", "http://x", True, 2, wyn=tmp_path, czat_fn=czat)
    assert wywolania == []


def test_tlumaczenie_odpowiedzi_cache_dedup_i_wznowienie(tmp_path):
    wywolania = []

    def llm(prompt, schema, system):
        lad = json.loads(prompt.split("\n", 1)[1])
        wywolania.append(lad)
        ids = [o["id"] for z in lad for o in z["odpowiedzi"]]
        return {"tlumaczenia": [{"id": i, "tekst": f"PL-{i}"} for i in ids[:-1]]}  # ostatnia zgubiona

    c = jezyk.Cache(tmp_path / "tl.jsonl")
    kont = {"tresc_pl": "zadanie", "tresc_en": "task"}
    poz = [{"id": "z1", "tekst": "one", **kont}, {"id": "z1", "tekst": "one", **kont},   # duplikat (dwa modele)
           {"id": "z2", "tekst": "two", **kont}, {"id": "z2", "tekst": "", **kont},      # pusta: bez tłumacza
           {"id": "z3", "tekst": "(BŁĄD: x)", **kont}]
    jezyk.tlumacz_odpowiedzi(poz, "en-pl", 1, llm_fn=llm, cache=c)
    assert sum(len(z["odpowiedzi"]) for z in wywolania[0]) == 2          # one, two (bez duplikatu i pustych)
    assert "model" not in json.dumps(wywolania[0])
    # zgubiona pozycja wraca w kolejnej próbie jako jedyna
    assert [sum(len(z["odpowiedzi"]) for z in w) for w in wywolania[1:]] == [1, 1]
    wywolania.clear()
    c2 = jezyk.Cache(tmp_path / "tl.jsonl")                              # wznowienie z pliku
    assert jezyk.tlumaczenie_z_cache(c2, "en-pl", "z2", "") == ""
    assert jezyk.tlumaczenie_z_cache(c2, "en-pl", "z3", "(BŁĄD: x)") == "(BŁĄD: x)"
    assert jezyk.tlumaczenie_z_cache(c2, "en-pl", "z9", "brak") is None
    got = [jezyk.tlumaczenie_z_cache(c2, "en-pl", z, t) for z, t in (("z1", "one"), ("z2", "two"))]
    assert sum(g is not None for g in got) == 1                          # jedna ciągle gubiona po 3 próbach


def test_zbuduj_wstecz_pf_i_kontrola(tmp_path, monkeypatch):
    zz = [{"id": "pf", "esej": False, "typ": "zamkniete", "polecenie": "Oceń prawdziwość zdań.", "rozwiazanie": "PF"},
          {"id": "o", "esej": False, "typ": "otwarte", "polecenie": "Podaj nazwę.", "rozwiazanie": "x"}]
    for konfig, odp in (("goly_en", {"pf": "1. T\n2. F", "o": "Jogaila"}), ("goly_pl", {"pf": "1. P\n2. F", "o": "Jagiełło"})):
        p = jezyk.plik_odp("m", konfig, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps({"id": i, "odpowiedz": t, "sekundy": 1}) + "\n" for i, t in odp.items()),
                     encoding="utf-8")
    c = jezyk.Cache(tmp_path / "tl.jsonl")
    w = lambda k, i, zr, wy: {"klucz": jezyk.klucz_tl(k, i, zr), "kierunek": k, "id": i, "zrodlo": zr, "wynik": wy}  # noqa: E731
    c.dodaj([w("en-pl", "pf", "1. T\n2. F", "1. T\n2. F"),        # tłumacz zostawił T: zabezpieczenie zamienia
             w("en-pl", "o", "Jogaila", "Jagiełło"),
             w("pl-en", "pf", "1. P\n2. F", "1. T\n2. F"), w("pl-en", "o", "Jagiełło", "Jogaila")])
    s = jezyk.zbuduj_wstecz("m", zz, {}, c, wyn=tmp_path)
    assert s == {"en_pl": 2, "pl_rt": 2, "n": 2, "goly_en_pl": 2, "goly_pl_rt": 2}
    en = jezyk.wczytaj_odp(jezyk.plik_odp("m", "goly_en_pl", tmp_path))
    rt = jezyk.wczytaj_odp(jezyk.plik_odp("m", "goly_pl_rt", tmp_path))
    assert en["pf"]["odpowiedz"] == "1. P\n2. F" and en["pf"]["odpowiedz_en"] == "1. T\n2. F"
    assert rt["o"] == {"id": "o", "model": "m", "konfig": "goly_pl_rt", "odpowiedz": "Jagiełło", "odpowiedz_en": "Jogaila",
                       "odpowiedz_pl": "Jagiełło", "sekundy": 1}


def test_zbuduj_wstecz_z_opisami_bez_kontroli(tmp_path):
    zz = [{"id": "pf", "esej": False, "typ": "zamkniete", "polecenie": "Oceń prawdziwość zdań.", "rozwiazanie": "PF"},
          {"id": "o", "esej": False, "typ": "otwarte", "polecenie": "Podaj nazwę.", "rozwiazanie": "x"}]
    for konfig, odp in (("goly_en", {"pf": "1. T\n2. F", "o": "Jogaila"}), ("goly_vlm_en", {"pf": "1. F\n2. F", "o": "Vasa"})):
        p = jezyk.plik_odp("m", konfig, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps({"id": i, "odpowiedz": t, "sekundy": 1}) + "\n" for i, t in odp.items()),
                     encoding="utf-8")
    c = jezyk.Cache(tmp_path / "tl.jsonl")
    w = lambda i, zr, wy: {"klucz": jezyk.klucz_tl("en-pl", i, zr), "kierunek": "en-pl", "id": i, "zrodlo": zr, "wynik": wy}  # noqa: E731
    c.dodaj([w("pf", "1. T\n2. F", "1. T\n2. F"), w("o", "Jogaila", "Jagiełło"),
             w("pf", "1. F\n2. F", "1. F\n2. F")])                  # „Vasa” bez tłumaczenia w cache
    s = jezyk.zbuduj_wstecz("m", zz, {}, c, wyn=tmp_path, zrodla=("goly_en", "goly_vlm_en"), kontrola=False)
    assert s["goly_en_pl"] == 2 and s["goly_vlm_en_pl"] == 1 and s["en_pl"] == 2 and "goly_pl_rt" not in s
    assert not jezyk.plik_odp("m", "goly_pl_rt", tmp_path).exists()
    v = jezyk.wczytaj_odp(jezyk.plik_odp("m", "goly_vlm_en_pl", tmp_path))
    assert v == {"pf": {"id": "pf", "model": "m", "konfig": "goly_vlm_en_pl", "odpowiedz": "1. F\n2. F",
                        "odpowiedz_en": "1. F\n2. F", "sekundy": 1}}


def test_blok_opisu_en(monkeypatch):
    from matura import obrazy
    il = [{"rel": "a.png", "zrodlo": 2, "podpis": "Źródło 2. Mapa", "i": 1, "n": 2},
          {"rel": "b.png", "zrodlo": 2, "podpis": "Źródło 2. Mapa", "i": 2, "n": 2},
          {"rel": "c.png", "zrodlo": None, "podpis": "Fotografia", "i": 1, "n": 1}]
    monkeypatch.setattr(obrazy, "ilustracje", lambda z: il if z["id"] == "x" else [])
    monkeypatch.setattr(obrazy, "z_cache", lambda rel, wariant: {"wynik": f"desc {rel} {wariant}"})
    b = jezyk.blok_opisu_en({"id": "x"})
    assert b.split("\n") == [jezyk.NAGLOWEK_BLOKU_EN,
                             "IMAGE DESCRIPTION (source 2, illustration 1 of 2): desc a.png vlm_q2b_en",
                             "IMAGE DESCRIPTION (source 2, illustration 2 of 2): desc b.png vlm_q2b_en",
                             "IMAGE DESCRIPTION (source): desc c.png vlm_q2b_en"]
    assert jezyk.blok_opisu_en({"id": "y"}) == ""
    assert jezyk.z_blokiem("", "T") == "T" and jezyk.z_blokiem("B", "T") == "B\n\nT"


def test_linie_zestawow_zgodne_z_pelna_matura():
    linie = jezyk.linie_zestawow(["qwen3-0.6b-q4"])
    assert len(linie) == 3
    zs = [pelna_matura.parsuj_zestaw(l) for l in linie]
    assert [z["etykieta"] for z in zs] == ["qwen3-0.6b-q4 | pl", "qwen3-0.6b-q4 | en→pl", "qwen3-0.6b-q4 | pl→en→pl"]
    assert all(z["krotkie"] == z["eseje"] for z in zs)
    assert zs[1]["krotkie"] == "review/jezyk-2026-09-26/odpowiedzi/qwen3-0.6b-q4__goly_en_pl.jsonl"
    linie = jezyk.linie_zestawow(["m"], "review/x", jezyk.ZESTAWY_OPIS)
    assert [pelna_matura.parsuj_zestaw(l)["krotkie"] for l in linie] == [
        "review/x/odpowiedzi/m__goly_vlm_pl.jsonl", "review/x/odpowiedzi/m__goly_vlm_en_pl.jsonl"]


# ---------------------------------------------------------------- sonda

def test_epoki_i_zakres_dzialow_kart():
    assert jezyk.epoka("Świat starożytnych Greków") == "starozytnosc"
    assert jezyk.epoka("Polska w XIV i XV w") == "sredniowiecze"
    assert jezyk.epoka("Rewolucje XVIII w") == "nowozytnosc"
    assert jezyk.epoka("Europa i świat w II połowie XIX i na początku XX w") == "xix"
    assert jezyk.epoka("II wojna światowa i jej etapy") == "xx_1945"
    assert jezyk.epoka("Proces przejmowania władzy przez komunistów w Polsce (1944–1948)") == "po_1945"
    assert jezyk.epoka("Historia jako nauka") is None
    assert jezyk.zakres_dzialu("Renesans w Polsce") == "polska"
    assert jezyk.zakres_dzialu("Upadek Rzeczypospolitej (wojna z Rosją i powstanie kościuszkowskie)") == "polska"
    assert jezyk.zakres_dzialu("Europa w dobie oświecenia") == "powszechna"


def test_wszystkie_dzialy_kanonu_maja_epoke():
    p = jezyk.ROOT / "data" / "karty" / "kanon.jsonl"
    if not p.exists():
        pytest.skip("brak kart kanonicznych")
    dzialy = {json.loads(l)["dzial"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert [d for d in dzialy if jezyk.epoka(d) is None] == ["Historia jako nauka"]


def _pytania(n_na_komorke=3):
    out = []
    for zakres, ep in jezyk.KWOTY:
        for i in range(jezyk.KWOTY[zakres, ep] + n_na_komorke):
            out.append({"klucz": f"{zakres}-{ep}-{i}", "ok": i % 7 != 3, "zakres": zakres, "epoka": ep,
                        "odpowiedz_pl": f"{ep} {zakres} {i // 2 if i < 2 else i}"})   # i=0,1 ta sama odpowiedź
    return out


def test_wybor_sondy_kwoty_kontrola_deterministyczny():
    s1, s2 = jezyk.wybierz_sonde(_pytania()), jezyk.wybierz_sonde(_pytania())
    assert s1 == s2 and len(s1) == 100
    assert sum(p["zakres"] == "polska" for p in s1) == 50
    assert all(p["ok"] for p in s1)
    assert len({jezyk._norm(p["odpowiedz_pl"]) for p in s1}) == 100          # bez powtórzonych odpowiedzi
    assert [p["id"] for p in s1] == [f"s{i:03d}" for i in range(1, 101)]
    k = [p for p in s1 if p["kontrola"]]
    assert len(k) == 30 and sum(p["zakres"] == "polska" for p in k) == 15


def test_wybor_sondy_uzupelnia_niedobor_z_tego_samego_zakresu():
    p = [x for x in _pytania(6) if not (x["zakres"] == "polska" and x["epoka"] == "xix")]  # zapas w innych epokach
    s = jezyk.wybierz_sonde(p)
    assert sum(x["zakres"] == "polska" for x in s) == 50
    assert not any(x["zakres"] == "polska" and x["epoka"] == "xix" for x in s)


def test_prompty_sondy():
    p = {"pytanie_pl": "Kiedy?", "pytanie_en": "When?", "zdanie_pl": "W 1410.", "zdanie_en": "In 1410.", "kontrola": True}
    pr = jezyk.prompty_sondy(p)
    assert pr["pl"] == ("Odpowiedz krótko.", "Kiedy?") and pr["en"] == ("Answer briefly.", "When?")
    assert pr["pl_ctx"] == ("Odpowiedz krótko.", "W 1410.\n\nKiedy?") and pr["en_ctx"][1] == "In 1410.\n\nWhen?"
    assert set(jezyk.prompty_sondy({**p, "kontrola": False})) == {"pl", "en"}


def test_sonda_odpowiedzi_i_oceny_wznawialne_na_slepo(tmp_path):
    sonda = [{"id": "s001", "pytanie_pl": "Kiedy Grunwald?", "pytanie_en": "When was Grunwald?", "zdanie_pl": "Z",
              "zdanie_en": "S", "kontrola": True, "odpowiedz_pl": "1410", "odpowiedz_en": "1410", "warianty": ["1410"],
              "zakres": "polska"},
             {"id": "s002", "pytanie_pl": "Kto?", "pytanie_en": "Who?", "kontrola": False, "odpowiedz_pl": "Mieszko I",
              "odpowiedz_en": "Mieszko I", "warianty": ["Mieszko"], "zakres": "polska"}]
    odp = {"Kiedy Grunwald?": "1410", "When was Grunwald?": "1410", "Z\n\nKiedy Grunwald?": "1410",
           "S\n\nWhen was Grunwald?": "1410", "Kto?": "Bolesław", "Who?": "Mieszko I"}
    wyw = []

    def czat(url, s, u, o, *, max_tokens, bez_myslenia):
        wyw.append(max_tokens)
        return odp[u], 0.0

    for m in ("a", "b"):
        jezyk.sonda_odpowiedz_model(m, sonda, "http://x", False, 2, wyn=tmp_path, czat_fn=czat)
    assert len(wyw) == 12 and set(wyw) == {40}
    jezyk.sonda_odpowiedz_model("a", sonda, "http://x", False, 2, wyn=tmp_path, czat_fn=czat)
    assert len(wyw) == 12                                                 # wznowienie: nic do zrobienia
    oceniane = []

    def llm(prompt, schema, system):
        lad = json.loads(prompt.split("\n", 1)[1])
        oceniane.extend(lad)
        return {"oceny": [{"id": x["id"], "poprawna": x["odpowiedz_zdajacego"] in x["akceptowane_warianty"]
                           or x["odpowiedz_zdajacego"] == x["poprawna_odpowiedz"].split(" / ")[0]} for x in lad]}

    c = jezyk.Cache(tmp_path / "oceny.jsonl")
    jezyk.sonda_oceny(["a", "b"], sonda, 1, llm_fn=llm, cache=c, wyn=tmp_path)
    # unikalne (id, język, odpowiedź): s001 pl/en „1410” (ctx dzieli ocenę), s002 pl „Bolesław”, en „Mieszko I”
    assert len(oceniane) == 4
    assert not any("model" in x or "warunek" in x for x in oceniane)
    ws = jezyk.wyniki_sondy(["a"], sonda, c, wyn=tmp_path)["a"]
    assert ws["sonda"]["pl"] == 0.5 and ws["sonda"]["en"] == 1.0 and ws["sonda"]["tylko_en"] == 1
    assert ws["kontrola_ctx"]["pl"] == ws["kontrola_ctx"]["en"] == 1.0
    n = len(oceniane)
    jezyk.sonda_oceny(["a", "b"], sonda, 1, llm_fn=llm, cache=jezyk.Cache(tmp_path / "oceny.jsonl"), wyn=tmp_path)
    assert len(oceniane) == n                                             # oceny z cache


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_deterministyczny_i_poprawny():
    rng = np.random.default_rng(1)
    a = (rng.random(100) < 0.3).astype(float)
    b = np.maximum(a, (rng.random(100) < 0.3).astype(float))            # b ≥ a na każdej pozycji
    r1, r2 = jezyk.bootstrap_roznica(a, b), jezyk.bootstrap_roznica(a, b)
    assert r1 == r2 and r1["n"] == 100
    assert r1["roznica"] == pytest.approx(b.mean() - a.mean())
    assert 0 <= r1["lo"] <= r1["roznica"] <= r1["hi"]


def test_bootstrap_brak_roznicy_i_wagi():
    a = [1, 0, 1, 0] * 10
    r = jezyk.bootstrap_roznica(a, a)
    assert r["roznica"] == r["lo"] == r["hi"] == 0
    r = jezyk.bootstrap_roznica([0, 0], [2, 1], wagi=[2, 2])
    assert r["roznica"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        jezyk.bootstrap_roznica([1, 0], [1])


def test_bootstrap_sparowany_wezszy_niz_niesparowany():
    rng = np.random.default_rng(2)
    a = (rng.random(200) < 0.5).astype(float)
    b = a.copy(); b[:20] = 1                                            # silna korelacja par
    r = jezyk.bootstrap_roznica(a, b)
    assert r["lo"] > 0                                                   # sparowany wykrywa małą, spójną poprawę


# ---------------------------------------------------------------- język odpowiedzi (opis danych)

@pytest.mark.parametrize("t, j", [
    ("Bitwa pod Grunwaldem odbyła się w 1410 roku i była wielkim zwycięstwem.", "pl"),
    ("The Battle of Grunwald was fought in 1410 by the Polish army.", "en"),
    ("1. P\n2. F", "krotka"),
    ("Kazimierz Wielki", "krotka"),
    ("这是一个关于波兰历史的回答，我们需要考虑很多因素。", "inny"),
])
def test_jezyk_tekstu(t, j):
    assert jezyk.jezyk_tekstu(t) == j


def test_jezyki_odpowiedzi_liczniki():
    assert jezyk.jezyki_odpowiedzi(["A", "The king was in the city of Rome.", "Król był w mieście, które się poddało."]) \
        == {"pl": 1, "en": 1, "inny": 0, "krotka": 1}


# ---------------------------------------------------------------- --sesje (sito modeli na jednym arkuszu)

@jest_cke
def test_sesje_ogranicza_zadania_wszystkich_krokow(tmp_path, monkeypatch):
    monkeypatch.setattr(jezyk, "SESJE", jezyk.SESJE)   # main zmienia globalne SESJE i WYN: przywrócenie po teście
    monkeypatch.setattr(jezyk, "WYN", jezyk.WYN)
    zlapane = {}
    monkeypatch.setattr(jezyk, "krok_generuj",
                        lambda modele, limit=None, opisy=False: zlapane.update(ids=[z["id"] for z in jezyk.zadania()]))
    assert jezyk.main(["--generuj", "--sesje", "2025-maj", "--wyniki", str(tmp_path), "--modele", "gemma3-1b-q4"]) == 0
    assert jezyk.SESJE == ("2025-maj",)
    assert zlapane["ids"] and all(i.startswith("2025-maj-") for i in zlapane["ids"])
    assert {"2025-maj-25-t1", "2025-maj-25-t3"} <= set(zlapane["ids"])
    assert jezyk.zadania(["2024-maj"])[0]["id"].startswith("2024-maj-")   # jawny argument wygrywa z globalnym


@jest_cke
def test_tylko_eseje_ogranicza_zadania_do_tematow(tmp_path, monkeypatch):
    monkeypatch.setattr(jezyk, "SESJE", jezyk.SESJE)
    monkeypatch.setattr(jezyk, "WYN", jezyk.WYN)
    monkeypatch.setattr(jezyk, "TYLKO_ESEJE", jezyk.TYLKO_ESEJE)
    zlapane = {}
    monkeypatch.setattr(jezyk, "krok_generuj",
                        lambda modele, limit=None, opisy=False: zlapane.update(ids=[z["id"] for z in jezyk.zadania()]))
    assert jezyk.main(["--generuj", "--sesje", "2025-maj", "--tylko-eseje", "--wyniki", str(tmp_path),
                       "--modele", "gemma3-1b-q4"]) == 0
    assert zlapane["ids"] == ["2025-maj-25-t1", "2025-maj-25-t2", "2025-maj-25-t3"]
    assert jezyk.main(["--generuj", "--sesje", "2025-maj", "--wyniki", str(tmp_path), "--modele", "gemma3-1b-q4"]) == 0
    assert len(zlapane["ids"]) > 3                                       # bez flagi: cały arkusz


def test_sesje_nieznana_testowa_i_domyslne(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jezyk, "SESJE", jezyk.SESJE)
    monkeypatch.setattr(jezyk, "WYN", jezyk.WYN)
    monkeypatch.setattr(jezyk, "krok_generuj", lambda *a, **k: None)
    assert jezyk.main(["--generuj", "--sesje", "2099-maj", "--wyniki", str(tmp_path)]) == 2
    assert "2099-maj" in capsys.readouterr().err
    arkusze = tmp_path / "json"
    arkusze.mkdir()
    (arkusze / "historia-2026-maj.json").write_text("[]", encoding="utf-8")   # arkusz testowy „jest”
    monkeypatch.setattr(devset, "JSON", arkusze)
    assert jezyk.main(["--generuj", "--sesje", "2026-maj", "--wyniki", str(tmp_path)]) == 2
    assert "testowe" in capsys.readouterr().err
    assert jezyk.main(["--generuj", "--wyniki", str(tmp_path)]) == 0
    assert jezyk.SESJE == ("2024-maj", "2025-maj")                        # bez --sesje: oba arkusze
