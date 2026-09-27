"""Adapter egzaminu organizatorów (matura/egzamin.py) na paczce egzaminu próbnego (data/egzamin-probny, poza gitem)."""
import json
from pathlib import Path

import pytest

from matura import egzamin, obrazy

PACZKA = Path(__file__).resolve().parent.parent / "data" / "egzamin-probny"
wymaga_paczki = pytest.mark.skipif(not (PACZKA / "exam.json").exists(), reason="brak paczki egzaminu próbnego")


@wymaga_paczki
def test_wczytaj_probny():
    exam, zad = egzamin.wczytaj(PACZKA, "2023-maj")
    krotkie = [z for z in zad if not z["esej"]]
    eseje = [z for z in zad if z["esej"]]
    assert exam["exam_id"] == "history-2023-mock-v1"
    assert len(krotkie) == 36 and len(eseje) == 3
    z1 = next(z for z in krotkie if z["pozycja"] == "1")
    assert z1["id"] == "2023-maj-1" and z1["typ"] == "otwarte"
    assert "[Obraz:" not in z1["zrodla_tekst"]
    assert obrazy.ilustracje(z1)[0]["sciezka"].endswith("Z01.png")
    assert next(z for z in krotkie if z["pozycja"] == "3")["typ"] == "zamkniete"
    z13 = next(z for z in krotkie if z["pozycja"] == "13.1")
    assert [il["podpis"] for il in obrazy.ilustracje(z13)] == ["Źródło 1. Plany bitew z okresu powstania listopadowego"] * 2
    assert [e["id"] for e in eseje] == ["2023-maj-26-t1", "2023-maj-26-t2", "2023-maj-26-t3"]
    assert eseje[1]["nr_tematu"] == 2 and eseje[1]["pozycja"] == "26"


@pytest.mark.parametrize("odp,fmt,oczek", [
    ("1. P\n2. F\n3. P", "1: P\n2: F\n3: P", "1: P\n2: F\n3: P"),
    ("P, F, P", "1: P\n2: F\n3: P", "1: P\n2: F\n3: P"),
    ("**B**", "A", "B"),
    ("1 – C, 2 – B", "1: A\n2: A", "1: C\n2: B"),
    ("A: 3\nB: 1", "A: 1\nB: 1", "A: 3\nB: 1"),
    ("nie wiem", "1: A\n2: A", "nie wiem"),
    ("**Rozstrzygnięcie:** neolit", "Tekst po polsku. Podaj wszystkie wymagane elementy odpowiedzi.", "Rozstrzygnięcie: neolit"),
])
def test_normalizuj(odp, fmt, oczek):
    assert egzamin.normalizuj(odp, fmt) == oczek


@wymaga_paczki
def test_sprawdz_szablon_i_bledy():
    szablon = json.loads((PACZKA / "answers-template.json").read_text(encoding="utf-8"))
    bledy, ostrz = egzamin.sprawdz(szablon, szablon)
    assert bledy == [] and len(ostrz) == 37                      # szablon: same puste odpowiedzi
    zly = {"exam_id": "x", "answers": szablon["answers"][:-1] + [{"id": "1", "answer": 5}]}
    bledy, _ = egzamin.sprawdz(zly, szablon)
    assert any("exam_id" in b for b in bledy) and any("powtórzone" in b for b in bledy)
    assert any("typ pola" in b for b in bledy)


@wymaga_paczki
def test_zloz_esej_i_formaty():
    exam, zad = egzamin.wczytaj(PACZKA, "2023-maj")
    wybrany = next(z for z in zad if z["esej"] and z["nr_tematu"] == 2)
    krotkie = {"2023-maj-3": {"odpowiedz": "1. F 2. P 3. P"}, "2023-maj-1": {"odpowiedz": "(BŁĄD: timeout)"}}
    eseje = {wybrany["id"]: {"odpowiedz": "słowo " * 320}}
    odp = egzamin.zloz(exam, zad, krotkie, eseje, wybrany)
    a = {x["id"]: x["answer"] for x in odp["answers"]}
    assert a["3"] == "1: F\n2: P\n3: P" and a["1"] == "" and a["26"].startswith("Temat 2.")
    szablon = json.loads((PACZKA / "answers-template.json").read_text(encoding="utf-8"))
    assert egzamin.sprawdz(odp, szablon)[0] == []


def test_domyslny_esej_najlepszy_w_testach():
    a = egzamin.parser().parse_args(["--paczka", "x", "--model", "m", "--gguf", "g.gguf", "--wyniki", "w"])
    assert a.esej == "e5_zgadzam" and a.krotkie == "goly"


def test_domyslna_regula_tematu_powszechna():
    # walidacja 2023-maj + 2023-czerwiec (review/esej-e8-2026-09-26): „powszechna” nie gorsza od „pokrycie”
    wymagane = ["--paczka", "p", "--model", "m", "--gguf", "g", "--wyniki", "w"]
    assert egzamin.parser().parse_args(wymagane).regula_tematu == "powszechna"
    assert egzamin.parser().parse_args(wymagane + ["--regula-tematu", "pokrycie"]).regula_tematu == "pokrycie"


def test_pf_litery_z_angielskiego():
    from matura.egzamin import normalizuj, pf_litery
    assert normalizuj(pf_litery("1. True\n2. False\n3. T"), "1: P\n2: F\n3: P") == "1: P\n2: F\n3: P"


def test_potok_en_tlumaczy_tylko_otwarte(tmp_path):
    import json
    from matura import egzamin

    class Tl:
        def __init__(self):
            self.wywolania = []

        def tlumacz(self, teksty, kierunek, konteksty=None):
            self.wywolania.append((kierunek, list(teksty)))
            return [f"<{kierunek}>{t}" for t in teksty]

    zz = [{"id": "e-1", "typ": "zamkniete", "esej": False, "polecenie": "Oceń prawdziwość.", "zrodla_tekst": "",
           "answer_format": "1: P\n2: F"},
          {"id": "e-2", "typ": "otwarte", "esej": False, "polecenie": "Wyjaśnij.", "zrodla_tekst": "Źródło."}]
    odp = {"e-1": "1. True 2. False", "e-2": "Because of the war."}

    def czat(url, system, user, obrazy, **kw):
        return (odp["e-1"] if "prawdziwo" in user else odp["e-2"]), 0.5

    tz, to = Tl(), Tl()
    p = egzamin.potok_en(zz, "m", tmp_path, "http://x", bez_myslenia=True, rownolegle=1, tl_zadan=tz, tl_odp=to,
                         czat_fn=czat)
    w = {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()}
    assert tz.wywolania[0][0] == "pl-en" and len(tz.wywolania[0][1]) == 2
    assert to.wywolania == [("en-pl", ["Because of the war."])]
    assert w["e-2"]["odpowiedz"] == "<en-pl>Because of the war." and w["e-1"]["odpowiedz"] == "1. P 2. F"
    assert p.name == "m__goly_en_pl.jsonl"


def test_opisz_brakujace_tylko_nowe(tmp_path, monkeypatch):
    from matura import obrazy
    monkeypatch.setattr(obrazy, "CACHE", tmp_path / "opisy.jsonl")
    monkeypatch.setattr(obrazy, "_CACHE", None)
    monkeypatch.setattr(obrazy, "ROOT", tmp_path)
    for n in ("a.png", "b.png"):
        (tmp_path / n).write_bytes(n.encode())
    obrazy._zapisz("a.png", "vlm_test", "stary opis", 0.1)
    wywolane = []

    def opisz(url, il, jezyk):
        wywolane.append(il["rel"])
        return f"opis {il['rel']}", 0.2

    ilu = [{"rel": "a.png"}, {"rel": "b.png"}, {"rel": "b.png"}]
    assert obrazy.opisz_brakujace(ilu, "http://x", "vlm_test", opisz_fn=opisz) == 1
    assert wywolane == ["b.png"] and obrazy.z_cache("b.png", "vlm_test")["wynik"] == "opis b.png"


def test_esej_en_osobny_model_ponawia_za_krotki(tmp_path):
    class TlZadan:
        def tlumacz(self, teksty, kierunek, konteksty=None):
            return ["Write an essay on the thesis. At least 300 words."] * len(teksty)

    class TlOdp:  # „tłumaczenie” bez zmian: liczba słów PL = liczba słów EN
        def tlumacz(self, teksty, kierunek, konteksty=None):
            return list(teksty)

    wywolania = []

    def czat(url, system, user, obr=None, **kw):
        wywolania.append((user, kw["temperature"]))
        return ("słowo " * (120 if len(wywolania) == 1 else 420)).strip(), 1.5

    z = {"id": "2023-maj-26-t2", "esej": True, "polecenie": "Temat.", "zrodla_tekst": "", "temat": "Temat."}
    p = egzamin.esej_en(z, "lfm2-2.6b-q4km", tmp_path, "http://x", bez_myslenia=False, tl_zadan=TlZadan(),
                        tl_odp=TlOdp(), czat_fn=czat)
    d = json.loads(p.read_text(encoding="utf-8").strip())
    assert d["id"] == "2023-maj-26-t2" and d["proby"] == 2 and d["slow"] == 420 and d["sekundy"] == 3.0
    assert len(wywolania) == 2 and "at least 450 words" in wywolania[1][0] and wywolania[1][1] > wywolania[0][1]


def test_parser_osobny_model_eseju():
    wymagane = ["--paczka", "p", "--model", "m", "--gguf", "g", "--wyniki", "w"]
    a = egzamin.parser().parse_args(wymagane)
    assert a.esej_model is None and a.esej_port == 8097 and a.esej_min_slow == 300
    assert a.esej_prompt == "goly"
    assert egzamin.parser().parse_args(wymagane + ["--esej-model", "lfm2-2.6b-q4km"]).esej_model == "lfm2-2.6b-q4km"


def test_esej_en_prompt_e9_regula_porownania_i_aspekty(tmp_path):
    class Tl:
        def tlumacz(self, teksty, kierunek, konteksty=None):
            return ["Write an essay on the thesis."] * len(teksty) if kierunek == "pl-en" else list(teksty)

    systemy, prompty = [], []

    def czat(url, system, user, obr=None, **kw):
        systemy.append(system); prompty.append(user)
        return ("słowo " * 400).strip(), 1.0

    # przykłady spoza oceny (arkusze 2015-2022)
    naj = {"id": "x-t1", "esej": True, "polecenie": "T", "zrodla_tekst": "",
           "temat": ("Polityka Napoleona Bonaparte wobec sprawy polskiej przede wszystkim służyła interesom Francji. "
                     "Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej argumentacji "
                     "aspekty: polityczny, militarny i gospodarczy.")}
    zwykly = {**naj, "id": "x-t2", "temat": ("Reformacja zarówno osłabiła, jak i wzmocniła Kościół katolicki. Zajmij "
                                            "stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając w swojej "
                                            "argumentacji aspekty: polityczny, społeczny i kulturowy.")}
    for z in (naj, zwykly):
        p = egzamin.esej_en(z, "lfm2", tmp_path, "http://x", bez_myslenia=False, tl_zadan=Tl(), tl_odp=Tl(), czat_fn=czat,
                            prompt="e9")
    assert p.name == "lfm2__e9_en_pl.jsonl"
    assert all("at most 20 words" in s and "leave it out" in s and "450-550 words" in s for s in systemy)
    assert "partly" in systemy[0] and "compares" in systemy[0] and "compares" not in systemy[1]
    assert "political, military, economic" in prompty[0] and "political, social, cultural" in prompty[1]
    rek = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [r["konfig"] for r in rek] == ["e9_en_pl", "e9_en_pl"]
    # domyślny prompt bez zmian
    egzamin.esej_en(zwykly, "lfm2", tmp_path, "http://x", bez_myslenia=False, tl_zadan=Tl(), tl_odp=Tl(), czat_fn=czat)
    from matura import jezyk
    assert systemy[-1] == jezyk.SYS_GOLY_EN and prompty[-1] == "Write an essay on the thesis."


def test_esej_pl_uczen_prompt_z_treningu_i_ponowienie(tmp_path):
    from matura import esej_sft
    wywolania = []

    def czat(url, system, user, obr=None, **kw):
        wywolania.append((system, user, kw["temperature"], kw["probkowanie"]))
        return ("słowo " * (150 if len(wywolania) == 1 else 380)).strip(), 2.0

    z = {"id": "x-t1", "esej": True, "polecenie": f"{esej_sft.DLUGOSC}\n\nTeza. Zajmij stanowisko.", "zrodla_tekst": ""}
    p = egzamin.esej_pl(z, "bielik-1.5b-esej-v1", tmp_path, "http://x", bez_myslenia=False, czat_fn=czat)
    d = json.loads(p.read_text(encoding="utf-8").strip())
    assert p.name == "bielik-1.5b-esej-v1__sft_pl.jsonl"
    assert d["proby"] == 2 and d["slow"] == 380 and d["sekundy"] == 4.0 and d["stanowisko"] == "zgadzam"
    sys_, user, t0, prob = wywolania[0]
    assert (sys_, user) == tuple(m["content"] for m in esej_sft.wiadomosci(z["polecenie"], "zgadzam"))
    assert wywolania[1][2] > t0 and prob == egzamin.PROBKOWANIE_PL


def test_parser_esej_po_polsku_i_lora():
    wymagane = ["--paczka", "p", "--model", "m", "--gguf", "g", "--wyniki", "w"]
    a = egzamin.parser().parse_args(wymagane)
    assert a.esej_jezyk == "en" and a.esej_lora is None      # zestaw v2 bez zmian
    a = egzamin.parser().parse_args(wymagane + ["--esej-model", "qwen3.5-4b-iq3xxs", "--esej-jezyk", "pl",
                                                "--esej-lora", "adapter.gguf"])
    assert a.esej_jezyk == "pl" and str(a.esej_lora) == "adapter.gguf"


def test_usun_powtorzenia_zdan():
    t = "W 800 r. Karol został cesarzem. To ważne.\n\nW 800 r. Karol został cesarzem! Nowe zdanie. to ważne"
    assert egzamin.usun_powtorzenia(t) == "W 800 r. Karol został cesarzem. To ważne.\n\nNowe zdanie."
    assert egzamin.PROBKOWANIE_PL["dry_penalty_last_n"] == 256 and egzamin.PROBKOWANIE_PL["dry_allowed_length"] == 5


def test_llama_server_zmienna_srodowiskowa(monkeypatch):
    from matura import noc
    monkeypatch.setenv("LLAMA_SERVER", "/opt/llama/llama-server")
    assert noc.llama_server() == "/opt/llama/llama-server"
    monkeypatch.delenv("LLAMA_SERVER")
    assert noc.llama_server().endswith("llama-server")
