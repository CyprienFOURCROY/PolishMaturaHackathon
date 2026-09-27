"""Adapter egzaminu organizatorów: paczka (exam.json + images/*.png + answers-template.json) → answers.json.

Co to jest: czyta paczkę w formacie `separate-text-and-images-v1` (https://matura-json-guide.ania-olchowik.chatgpt.site/),
zamienia pozycje na nasze zadania, uruchamia harness na lokalnym llama-server, sprowadza odpowiedzi do `answer_format`,
sprawdza plik i zapisuje answers.json. Wzorów `answer_format` model nie widzi (małe modele je kopiują); składnię
robi `normalizuj`.
Po co: jedyna droga, którą nasz zestaw zdaje egzamin próbny i finał.
Co zrobić (egzamin próbny; prefiks 2023-maj pozwala ocenić te same odpowiedzi naszym sędzią):
  uv run python -m matura.egzamin --paczka data/egzamin-probny --model qwen3.5-4b-q4-tekst \\
      --gguf data/modele/qwen3.5-4b-q4/Qwen3.5-4B-Q4_K_M.gguf --krotkie goly --esej e5_zgadzam \\
      --prefiks 2023-maj --wyniki review/egzamin-probny-2026-09-26
  potem: uv run python -m matura.pelna_matura --wyniki <wyniki> --sesje 2023-maj --zestawy <wyniki>/pm_zestawy.txt
Potok angielski (`--potok en`, krok E planu docs/plans/2026-09-26-plan-po-1a-1b.md): zadanie tłumaczy na angielski ta
sama baza (`--tl-zadan baza`) albo Marian, model odpowiada po angielsku (gołym promptem, z `--kb` dwa hasła bazy po
angielsku), odpowiedzi otwarte i esej tłumaczy na polski Marian (matura/tlumacz.py), zamknięte zostają (T/True → P).
Osobny model eseju (`--esej-model lfm2-2.6b-q4km`, krok E5, review/esej-e8-2026-09-26): po zadaniach krótkich własny
llama-server (port `--esej-port`, 8097), temat wybrany regułą tłumaczy na angielski sam model eseju, goły esej po
angielsku, Marian EN→PL; poniżej `--esej-min-slow` słów ponowienie z prośbą o dłuższy tekst (`esej_en`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import devset, esej
from .noc import ROOT, Serwer, generuj, llama_server, log, wczytaj_odp

LIMIT_BAJTOW = 1024 * 1024
RE_OBRAZ = re.compile(r"\[Obraz:\s*(images/[^\]]+)\]")
RE_ZRODLO_PLIKU = re.compile(r"-S(\d+)(?:[-.]|$)")      # images/Z05-S2.png → źródło 2


def typ_pozycji(it: dict) -> str:
    f = it["answer_format"]
    if f.startswith("Jeden tekst") or it["max_points"] >= 10:
        return "esej"
    if f.startswith("Tekst po polsku"):
        return "otwarte"
    return "zamkniete"


def _ilustracje(it: dict, katalog: Path) -> list[dict]:
    """Obrazy pozycji w formacie obrazy.ilustracje(); podpis = najbliższa wcześniejsza linia tekstu, która nie jest znacznikiem obrazu."""
    linie = it["source_text"].splitlines()
    out = []
    for n, im in enumerate(it["images"], 1):
        sciezka = (katalog / im["path"]).resolve()
        m = RE_ZRODLO_PLIKU.search(Path(im["path"]).stem + ".")
        podpis = ""
        for i, l in enumerate(linie):
            if im["path"] in l:
                podpis = next((w.strip() for w in reversed(linie[:i]) if w.strip() and not RE_OBRAZ.search(w)), "")
                break
        rel = str(sciezka.relative_to(ROOT)) if sciezka.is_relative_to(ROOT) else str(sciezka)
        out.append({"sciezka": str(sciezka), "rel": rel, "zrodlo": int(m.group(1)) if m else None,
                    "podpis": podpis[:90], "i": n, "n": len(it["images"])})
    return out


def zadanie(it: dict, katalog: Path, prefiks: str) -> dict:
    ilu = _ilustracje(it, katalog)
    return {"id": f"{prefiks}-{it['id']}", "pozycja": it["id"], "polecenie": it["question"],
            "zrodla_tekst": RE_OBRAZ.sub("[ilustracja]", it["source_text"]).strip(), "pkt_max": it["max_points"],
            "esej": False, "typ": typ_pozycji(it), "split": "egzamin", "obrazy": [il["sciezka"] for il in ilu],
            "ilustracje_egz": ilu, "answer_format": it["answer_format"], "zasady_oceniania": "", "rozwiazanie": ""}


def wczytaj(katalog: Path, prefiks: str) -> tuple[dict, list[dict]]:
    exam = json.loads((katalog / "exam.json").read_text(encoding="utf-8"))
    if exam.get("input_format") != "separate-text-and-images-v1":
        log(f"UWAGA: nieznany input_format {exam.get('input_format')!r}")
    zadania = []
    for it in exam["items"]:
        if typ_pozycji(it) != "esej":
            zadania.append(zadanie(it, katalog, prefiks))
            continue
        tem = esej.tematy(it["question"])
        if len(tem) != 3:
            raise ValueError(f"pozycja {it['id']}: oczekiwano 3 tematów, jest {len(tem)}")
        for k, t in enumerate(tem, 1):
            zadania.append(devset._esej(f"{prefiks}-{it['id']}-t{k}", t, "egzamin", pozycja=it["id"], nr_tematu=k))
    return exam, zadania


def normalizuj(odp: str, fmt: str) -> str:
    """Odpowiedź modelu → składnia z answer_format (np. „1: P\\n2: F”). Gdy nie da się jednoznacznie odczytać,
    zostaje oczyszczony tekst modelu (organizator ocenia modelem językowym, więc tekst i tak zostanie przeczytany)."""
    odp = odp.replace("**", "").strip()
    if fmt.startswith(("Tekst", "Jeden tekst")):
        return odp
    wzor = [l.split(":", 1) for l in fmt.splitlines() if ":" in l]
    if not wzor:                                             # jedna litera, np. „A”
        m = re.search(r"\b([A-F])\b", odp)
        return m.group(1) if m else odp
    etykiety, probka = [e.strip() for e, _ in wzor], wzor[0][1].strip()
    klasa = "[PF]" if probka in ("P", "F") else (r"\d+" if probka.isdigit() else "[A-F]")
    wart = {}
    for e in etykiety:
        m = re.search(rf"(?m)(?:^|\s){re.escape(e)}\s*[.:)\-–]?\s*({klasa})\b", odp)
        if m:
            wart[e] = m.group(1)
    if len(wart) < len(etykiety):                            # np. „P, F, P”: kolejne wartości bez etykiet
        ciag = re.findall(rf"\b({klasa})\b", odp)
        if len(ciag) == len(etykiety):
            wart = dict(zip(etykiety, ciag))
    if len(wart) < len(etykiety):
        return odp
    return "\n".join(f"{e}: {wart[e]}" for e in etykiety)


def pf_litery(t: str) -> str:
    """Odpowiedź P/F po angielsku → litery P/F (normalizuj szuka samodzielnych liter P i F)."""
    from .jezyk import tf_na_pf
    t = re.sub(r"\b(?:TRUE|True|true|Prawda|prawda)\b", "P", t)
    t = re.sub(r"\b(?:FALSE|False|false|Fałsz|fałsz)\b", "F", t)
    return tf_na_pf(t)


def potok_en(zadania: list[dict], model: str, wyn: Path, url: str, *, bez_myslenia: bool, rownolegle: int,
             kb: bool = False, tl_zadan=None, tl_odp=None, czat_fn=None) -> Path:
    """Potok angielski dla listy zadań → plik odpowiedzi po polsku (<model>__goly_en[_kb]_pl.jsonl, pola odpowiedz,
    odpowiedz_en, tresc_en). tl_zadan / tl_odp: obiekty z matura/tlumacz.py (domyślnie Baza na url i Marian)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import jezyk, tlumacz
    if czat_fn is None:
        from .llm import czat as czat_fn
    tl_zadan = tl_zadan or tlumacz.Baza(url, bez_myslenia, rownolegle=rownolegle)
    tl_odp = tl_odp or tlumacz.Marian()
    konfig = "goly_en_kb_pl" if kb else "goly_en_pl"
    p = wyn / "odpowiedzi" / f"{model}__{konfig}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    en = tl_zadan.tlumacz([devset.tresc_dla_modelu(z) for z in zadania], "pl-en")
    bloki = [jezyk.blok_kb_en(z) if kb else "" for z in zadania]

    def jedno(i: int) -> tuple[str, float]:
        z = zadania[i]
        try:
            return czat_fn(url, jezyk.SYS_GOLY_EN, bloki[i] + en[i], None, max_tokens=1400 if z["esej"] else 400,
                           temperature=jezyk.TEMPERATURA["esej" if z["esej"] else "krotkie"], bez_myslenia=bez_myslenia)
        except Exception as e:  # noqa: BLE001 (pusta odpowiedź zamiast przerwania całego egzaminu)
            return f"(BŁĄD: {e})", 0.0

    with ThreadPoolExecutor(max(1, rownolegle)) as ex:
        odp = list(ex.map(jedno, range(len(zadania))))
    otwarte = [i for i, z in enumerate(zadania) if z.get("typ") != "zamkniete" and not odp[i][0].startswith("(BŁĄD")]
    pl = [o[0] for o in odp]
    for i, t_pl in zip(otwarte, tl_odp.tlumacz([odp[i][0] for i in otwarte], "en-pl")):
        pl[i] = t_pl
    for i, z in enumerate(zadania):
        if z.get("typ") == "zamkniete" and re.search(r":\s*[PF]\b", z.get("answer_format", "")):
            pl[i] = pf_litery(pl[i])
    with open(p, "w", encoding="utf-8") as f:
        for i, z in enumerate(zadania):
            f.write(json.dumps({"id": z["id"], "model": model, "konfig": konfig, "odpowiedz": pl[i],
                                "odpowiedz_en": odp[i][0], "tresc_en": en[i], "sekundy": round(odp[i][1], 2)},
                               ensure_ascii=False) + "\n")
    return p


DOPISEK_DLUGOSCI = "\n\nYour essay must have at least 450 words."
PROMPTY_ESEJU = ("goly", "e9")
# e9 (review/esej-e8-2026-09-26): eseje LFM2 na dev tracą przez błędy merytoryczne (24/24), spójność po Marianie
# (21/24) i ogólniki (19/24); bez przykładowego eseju (małe modele go kopiują)
SYS_E9 = """You are a student taking the Polish history matura exam (extended level). Write the essay in English.
Rules:
- Write short, simple sentences of at most 20 words each.
- Do not write a date or a name unless you are sure it is correct. If you are not sure, leave it out.
- State your position clearly in the first paragraph: you agree or you disagree with the thesis. Repeat the same position in the last paragraph. Never write that you partly agree.
- Write one paragraph for each aspect or example named in the topic. In each paragraph give 3-4 specific facts (who, what, when, and the result) and end it with one sentence that links these facts to the thesis.{porownanie}
- Length: 450-550 words. Plain paragraphs only: no headings, no lists, no bullet points."""
E9_POROWNANIE = ("\n- The thesis says that something was the most important or mattered more than anything else. Add one "
                 "paragraph that compares it with at least one alternative of the same kind and explains the difference.")


def prompt_e9(z: dict, tresc_en: str) -> tuple[str, str]:
    """(system, user) wariantu e9. Akapit porównawczy z reguły językowej na polskiej tezie (`esej_en.wymaga_porownania`:
    „naj…”, „przede wszystkim”, „w największym stopniu”, „bardziej niż”); aspekty z tematu wymienione po angielsku."""
    from .esej_en import ASPEKTY_EN, wymaga_porownania
    temat = z.get("temat") or (esej.tematy(z["polecenie"]) or [z["polecenie"]])[0]
    teza, aspekty, wybor = esej.rozbierz(temat)
    sys_ = SYS_E9.format(porownanie=E9_POROWNANIE if wymaga_porownania(teza) else "")
    if aspekty and not wybor and all(a in ASPEKTY_EN for a in aspekty):
        tresc_en += "\n\nAspects (one paragraph each): " + ", ".join(ASPEKTY_EN[a] for a in aspekty) + "."
    return sys_, tresc_en


def blok_kb_en_eseju(z: dict, n: int = 3) -> str:
    """n haseł bazy wiedzy dla tematu eseju (zapytanie z samej tezy, `esej_sft.zapytanie_teza`) po angielsku
    (`jezyk.hasla_en`, brak tłumaczenia → tekst polski), każde przycięte do `esej_sft.ZNAKI_HASLA` znaków (e10)."""
    from . import esej_sft, jezyk
    from .harness import _baza_hasel
    from .wiedza import _tekst_hasla
    en = jezyk.hasla_en()
    hasla = _baza_hasel().szukaj(esej_sft.zapytanie_teza(devset.tresc_dla_modelu(z)), n)
    linie = [f"- {(en.get(h['id']) or _tekst_hasla(h))[:esej_sft.ZNAKI_HASLA]}" for h in hasla]
    return (f"{jezyk.NAGLOWEK_KB_EN} Use its facts (dates, names) only when they fit the essay topic; ignore unrelated "
            "entries.\n" + "\n".join(linie) + "\n\n") if linie else ""


def esej_en(z: dict, model: str, wyn: Path, url: str, *, bez_myslenia: bool, tl_zadan, tl_odp, czat_fn=None,
            min_slow: int = 300, proby: int = 3, prompt: str = "goly", sufiks: str = "", material: int = 0) -> Path:
    """Goły esej po angielsku osobnym modelem (jak walidacja E1: temat tłumaczy sam model, odpowiedź EN → Marian EN→PL)
    → rekord dopisany do <model>__goly_en_pl.jsonl (czytelnik bierze ostatni rekord id, więc kilka arkuszy w jednym pliku
    i powtórzony przebieg są bezpieczne). Esej po polsku krótszy niż `min_slow` słów (warunek CKE, B = 0)
    → ponowienie z dopiskiem o długości i temperaturą 0,7 (do `proby` prób); zostaje najdłuższa próba.
    `sekundy` = suma czasu modelu ze wszystkich prób (bez tłumaczeń). prompt: „goly” (SYS_GOLY_EN, jak walidacja E1)
    albo „e9” (`prompt_e9`); plik <model>__<goly|e9>_en_pl<sufiks>.jsonl (sufiks np. „_g2” dla drugiej generacji)."""
    from . import jezyk
    if prompt not in PROMPTY_ESEJU:
        raise ValueError(f"prompt eseju: {PROMPTY_ESEJU}, jest {prompt!r}")
    if czat_fn is None:
        from .llm import czat as czat_fn
    konfig = f"{prompt}_kb{material}_en_pl" if material else f"{prompt}_en_pl"   # material: e10 (hasła EN przed tematem)
    p = wyn / "odpowiedzi" / f"{model}__{konfig}{sufiks}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tresc_en = tl_zadan.tlumacz([devset.tresc_dla_modelu(z)], "pl-en")[0]
    system, user = prompt_e9(z, tresc_en) if prompt == "e9" else (jezyk.SYS_GOLY_EN, tresc_en)
    if material:
        user = blok_kb_en_eseju(z, material) + user
    najl, sek, n = None, 0.0, 0
    for k in range(max(1, proby)):
        n += 1
        try:
            t, s = czat_fn(url, system, user + (DOPISEK_DLUGOSCI if k else ""), None,
                           max_tokens=1400 if k == 0 else 1800,
                           temperature=jezyk.TEMPERATURA["esej"] if k == 0 else 0.7, bez_myslenia=bez_myslenia)
        except Exception as e:  # noqa: BLE001 (następna próba albo pusty esej zamiast przerwania egzaminu)
            log(f"esej: błąd próby {n}: {e}")
            continue
        sek += s
        pl = tl_odp.tlumacz([t], "en-pl")[0]
        slow = len(pl.split())
        if najl is None or slow > najl["slow"]:
            najl = {"odpowiedz": pl, "odpowiedz_en": t, "slow": slow}
        if slow >= min_slow:
            break
    najl = najl or {"odpowiedz": "(BŁĄD: brak eseju)", "odpowiedz_en": "", "slow": 0}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": z["id"], "model": model, "konfig": konfig, **najl, "tresc_en": tresc_en,
                            "sekundy": round(sek, 2), "proby": n}, ensure_ascii=False) + "\n")
    return p


# przeciw pętlom małych uczniów (pomiar 27.09, S2 v0: 26 zdań, 16 unikalnych przy domyślnym oknie DRY 64 tokenów).
# Okno 256 i dozwolone powtórzenie 5 tokenów: pętle zdań nadal karane, a daty i nazwy (≤ 5 tokenów) wolno powtórzyć;
# okno 4096 obejmowało prompt z materiałem i tłumiło przepisywanie faktów (S2 kb, dev: 25 lat z promptu w esejach
# wobec 42 przy 256/5; bez DRY 39, ale 147/174 unikalnych zdań). Bez repeat_penalty (karze też „w”, „i”).
PROBKOWANIE_PL = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 5, "dry_penalty_last_n": 256}
RE_ZDANIA_PL = re.compile(r"(?<=[.!?])\s+")


def usun_powtorzenia(tekst: str) -> str:
    """Usuwa zdania identyczne (bez wielkości liter i interpunkcji) z którymś wcześniejszym; akapity bez zmian poza tym.
    Siatka bezpieczeństwa po DRY: pętla małego modelu nie dostaje punktów, a zawyża liczbę słów."""
    widziane, akapity = set(), []
    for akapit in tekst.split("\n"):
        zostaja = []
        for zd in RE_ZDANIA_PL.split(akapit.strip()):
            klucz = " ".join(re.findall(r"\w+", zd.lower()))
            if klucz and klucz in widziane:
                continue
            widziane.add(klucz)
            zostaja.append(zd)
        akapity.append(" ".join(zostaja))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(akapity)).strip()


def esej_pl(z: dict, model: str, wyn: Path, url: str, *, bez_myslenia: bool, czat_fn=None, min_slow: int = 300,
            proby: int = 3, stanowisko: str | None = "zgadzam", sufiks: str = "", material: int = 0,
            material_tryb: str = "pelny") -> Path:
    """Esej po polsku z ucznia po SFT na esejach nauczyciela (sesja 27.09, `matura/esej_sft.py`): ten sam prompt co
    w treningu, bez tłumacza → rekord dopisany do <model>__sft_pl<sufiks>.jsonl (czytelnik bierze ostatni rekord id).
    Esej krótszy niż `min_slow` słów → ponowienie z temperaturą 0,7 (do `proby` prób); zostaje najdłuższa próba.
    material > 0: tyle haseł bazy wiedzy w prompcie (`esej_sft.material`, uczeń uczony z materiałem, runda kb)."""
    from . import esej_sft
    if czat_fn is None:
        from .llm import czat as czat_fn
    p = wyn / "odpowiedzi" / f"{model}__sft_pl{sufiks}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tresc = devset.tresc_dla_modelu(z)
    mat = esej_sft.material(tresc, material, tryb=material_tryb) if material else None
    system, user = (m["content"] for m in esej_sft.wiadomosci(tresc, stanowisko, mat))
    najl, sek, n = None, 0.0, 0
    for k in range(max(1, proby)):
        n += 1
        try:
            t, s = czat_fn(url, system, user, None, max_tokens=1600, temperature=0.2 if k == 0 else 0.7,
                           bez_myslenia=bez_myslenia, probkowanie=PROBKOWANIE_PL)
        except Exception as e:  # noqa: BLE001 (następna próba albo pusty esej zamiast przerwania egzaminu)
            log(f"esej: błąd próby {n}: {e}")
            continue
        sek += s
        t = usun_powtorzenia(t)
        slow = esej_sft.slowa(t)
        if najl is None or slow > najl["slow"]:
            najl = {"odpowiedz": t, "slow": slow}
        if slow >= min_slow:
            break
    najl = najl or {"odpowiedz": "(BŁĄD: brak eseju)", "slow": 0}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": z["id"], "model": model, "konfig": "sft_pl", **najl, "stanowisko": stanowisko,
                            "sekundy": round(sek, 2), "proby": n}, ensure_ascii=False) + "\n")
    return p


def zloz(exam: dict, zadania: list[dict], odp_krotkie: dict, odp_esej: dict, wybrany: dict) -> dict:
    ids = {z["pozycja"]: z["id"] for z in zadania if not z["esej"]}
    answers = []
    for it in exam["items"]:
        if typ_pozycji(it) == "esej":
            t = odp_esej.get(wybrany["id"], {}).get("odpowiedz", "").strip()
            a = f"Temat {wybrany['nr_tematu']}.\n\n{t}" if t and not t.startswith("(BŁĄD") else ""
        else:
            t = odp_krotkie.get(ids[it["id"]], {}).get("odpowiedz", "")
            a = "" if t.startswith("(BŁĄD") else normalizuj(t, it["answer_format"])
        answers.append({"id": it["id"], "answer": a})
    return {"exam_id": exam["exam_id"], "answers": answers}


def sprawdz(odp: dict, szablon: dict) -> tuple[list[str], list[str]]:
    """Te same reguły co strona zgłoszeń (id, duplikaty, typy, rozmiar) + ostrzeżenia (puste, esej)."""
    bledy, ostrz = [], []
    if set(odp) != {"exam_id", "answers"}:
        bledy.append(f"pola najwyższego poziomu: {sorted(odp)}")
    if odp.get("exam_id") != szablon["exam_id"]:
        bledy.append("exam_id różny od szablonu")
    lista = odp.get("answers", [])
    ids, wzor = [a.get("id") for a in lista], [a["id"] for a in szablon["answers"]]
    if len(ids) != len(set(ids)):
        bledy.append("powtórzone id")
    if set(ids) != set(wzor):
        bledy.append(f"id: brak {sorted(set(wzor) - set(ids))}, nadmiar {sorted(set(ids) - set(wzor), key=str)}")
    for a in lista:
        if set(a) != {"id", "answer"}:
            bledy.append(f"pola pozycji {a.get('id')!r}: {sorted(a)}")
        if not isinstance(a.get("id"), str) or not isinstance(a.get("answer"), str):
            bledy.append(f"typ pola w pozycji {a.get('id')!r}")
        elif not a["answer"].strip():
            ostrz.append(f"pusta odpowiedź: {a['id']}")
        elif a["answer"].startswith("Temat") and len(a["answer"].split()) < 302:
            ostrz.append(f"esej {a['id']}: mniej niż 300 słów")
    if len(json.dumps(odp, ensure_ascii=False).encode("utf-8")) > LIMIT_BAJTOW:
        bledy.append("plik większy niż 1 MiB")
    return bledy, ostrz


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Egzamin organizatorów: exam.json → answers.json")
    ap.add_argument("--paczka", required=True, type=Path)
    ap.add_argument("--model", required=True, help="nazwa w plikach wyników, np. qwen3.5-4b-q4-tekst")
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--mmproj", type=Path, help="projektor wizji: model dostaje obrazy PNG (liczy się do rozmiaru bazy)")
    ap.add_argument("--krotkie", default="goly")
    ap.add_argument("--esej", default="e5_zgadzam")  # najlepszy w testach (e5_zgadzam ≥ e5 w 9/13 porównań)
    ap.add_argument("--prefiks", default="egzamin", help="2023-maj dla egzaminu próbnego (ocena naszym sędzią)")
    ap.add_argument("--wyniki", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--rownolegle", type=int, default=8)
    ap.add_argument("--z-mysleniem", action="store_true", help="bez_myslenia=False (tak jak Bielik w konfiguracjach nocy)")
    ap.add_argument("--opisywacz-gguf", type=Path, help="opisywacz na żywo: GGUF modelu wizyjnego (np. Qwen3.5-2B-Q4_K_M)")
    ap.add_argument("--opisywacz-mmproj", type=Path, help="projektor wizji opisywacza (mmproj-F16.gguf)")
    ap.add_argument("--opis", default="vlm_q2b_en", help="wariant opisu w cache (jak w harnessie goly_vlm / goly_vlm_kb)")
    ap.add_argument("--opis-jezyk", choices=("en", "pl"), default="en")
    ap.add_argument("--potok", choices=("pl", "en"), default="pl", help="en: tłumaczenie PL→EN, odpowiedź EN, Marian EN→PL")
    ap.add_argument("--tl-zadan", choices=("baza", "marian", "model"), default="baza",
                    help="--potok en: tłumacz zadań; model = osobny GGUF z --tl-gguf (mniejszy od bazy)")
    ap.add_argument("--tl-gguf", type=Path, help="--tl-zadan model: GGUF tłumacza (drugi llama-server na porcie +1)")
    ap.add_argument("--kb", action="store_true", help="--potok en: 2 hasła bazy po angielsku przed zadaniem")
    ap.add_argument("--esej-model", help="osobny model eseju z jezyk.MODELE (np. lfm2-2.6b-q4km): goły esej EN, "
                                         "temat tłumaczy ten model, Marian EN→PL")
    ap.add_argument("--esej-port", type=int, default=8097)
    ap.add_argument("--esej-jezyk", choices=("en", "pl"), default="en", help="--esej-model: en = goły EN + Marian "
                    "(zestaw v2); pl = uczeń po SFT pisze po polsku bez tłumacza (matura/esej_sft.py, sesja 27.09)")
    ap.add_argument("--esej-lora", type=Path, help="--esej-model: adapter LoRA GGUF do modelu eseju (uczeń S2)")
    ap.add_argument("--esej-material", type=int, default=0, help="haseł bazy wiedzy w prompcie eseju (pl: uczeń kb; "
                    "en: e10, hasła po angielsku przed tematem)")
    ap.add_argument("--esej-material-tryb", choices=("pelny", "teza"), default="pelny",
                    help="zapytanie do bazy: pełny tekst tematu (runda kb) albo sama teza (runda kb2)")
    ap.add_argument("--esej-min-slow", type=int, default=300, help="krótszy esej po polsku → ponowienie")
    ap.add_argument("--esej-prompt", choices=PROMPTY_ESEJU, default="goly", help="--esej-model: prompt eseju (e9: krótkie "
                    "zdania, bez niepewnych dat i nazwisk, akapit na aspekt, porównanie przy tezie w stopniu najwyższym)")
    ap.add_argument("--regula-tematu", choices=esej.REGULY_TEMATU, default="powszechna",
                    help="wybór 1 z 3 tematów eseju (esej.wybierz_temat); powszechna: E1 i walidacja 2023, "
                         "review/esej-e8-2026-09-26")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    from .karty import Karty
    from .retrieval import Wikipedia
    exam, zadania = wczytaj(a.paczka, a.prefiks)
    szablon = json.loads((a.paczka / "answers-template.json").read_text(encoding="utf-8"))
    try:
        wiki = Wikipedia()
    except FileNotFoundError:   # świeży klon bez indeksu (2,5 GB): pokrycie tematu z samych kart; na 6 arkuszach
        wiki = None             # 2023-2025 reguła „powszechna” wybrała z Wikipedią i bez niej te same tematy (27.09)
        log("brak indeksu Wikipedii (data/wiki/bm25): wybór tematu eseju z pokrycia kartami")
    karty = Karty()
    eseje = [z for z in zadania if z["esej"]]
    wybrany = eseje[esej.wybierz_temat(eseje, karty, wiki, regula=a.regula_tematu)]
    log(f"wybrany temat eseju: {wybrany['nr_tematu']} (reguła {a.regula_tematu})")
    (a.wyniki / "logi").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    czasy: dict[str, float] = {}
    srv_tl = None
    if a.opisywacz_gguf:  # obrazy paczki → opisy w cache (wariant --opis), zanim baza zacznie odpowiadać
        from . import obrazy
        srv_op = Serwer(llama_server(), a.opisywacz_gguf, a.opisywacz_mmproj,
                        a.port + 2, 4, 8192, a.wyniki / "logi" / "serwer_opisywacz.log", obrazy.SERWER_VLM_ARGS)
        try:
            ilu = [il for z in zadania for il in obrazy.ilustracje(z)]
            n = obrazy.opisz_brakujace(ilu, srv_op.url, a.opis, a.opis_jezyk, rownolegle=4, model=a.opisywacz_gguf.name)
        finally:
            srv_op.stop()
        czasy["opisywacz"] = round(time.time() - t0, 1)
        log(f"opisywacz: {len(ilu)} ilustracji, {n} nowych opisów w {time.time() - t0:.0f} s")
    srv = Serwer(llama_server(), a.gguf, a.mmproj, a.port, a.rownolegle,
                 8192, a.wyniki / "logi" / f"serwer_{a.model}.log")
    try:
        if a.potok == "en":
            from .tlumacz import Baza, Marian
            tl_z = Marian() if a.tl_zadan == "marian" else None
            if a.tl_zadan == "model":
                if not a.tl_gguf:
                    raise SystemExit("--tl-zadan model wymaga --tl-gguf")
                srv_tl = Serwer(llama_server(), a.tl_gguf, None, a.port + 1,
                                a.rownolegle, 8192, a.wyniki / "logi" / "serwer_tlumacz.log")
                tl_z = Baza(srv_tl.url, True, rownolegle=a.rownolegle)
            pk = pe = potok_en([z for z in zadania if not z["esej"]] + ([] if a.esej_model else [wybrany]), a.model,
                               a.wyniki, srv.url, bez_myslenia=not a.z_mysleniem, rownolegle=a.rownolegle, kb=a.kb,
                               tl_zadan=tl_z)
            a.krotkie = a.esej = f"en-{a.tl_zadan}{'-kb' if a.kb else ''}"
        else:
            kw = dict(url=srv.url, wiki=wiki, obrazy=a.mmproj is not None, bez_myslenia=not a.z_mysleniem,
                      rownolegle=a.rownolegle, karty=karty)
            pk = generuj([z for z in zadania if not z["esej"]], a.model, a.krotkie, a.wyniki, **kw)
            if not a.esej_model:
                pe = generuj([wybrany], a.model, a.esej, a.wyniki, **kw)
    finally:
        srv.stop()
        if srv_tl is not None:
            srv_tl.stop()
    czasy["krotkie"] = round(time.time() - t0 - czasy.get("opisywacz", 0), 1)
    if a.esej_model:  # osobny model eseju (krok E5): własny serwer po zadaniach krótkich
        from . import jezyk
        from .tlumacz import Baza, Marian
        t1 = time.time()
        _, bez_myslenia_e, _ = jezyk.MODELE[a.esej_model]
        srv_e = Serwer(llama_server(), jezyk._gguf(a.esej_model), None,
                       a.esej_port, 1, 8192, a.wyniki / "logi" / f"serwer_{a.esej_model}.log",
                       extra=["--lora", str(a.esej_lora)] if a.esej_lora else None)
        try:
            if a.esej_jezyk == "pl":
                pe = esej_pl(wybrany, a.esej_model, a.wyniki, srv_e.url, bez_myslenia=bez_myslenia_e,
                             min_slow=a.esej_min_slow, material=a.esej_material, material_tryb=a.esej_material_tryb)
            else:
                pe = esej_en(wybrany, a.esej_model, a.wyniki, srv_e.url, bez_myslenia=bez_myslenia_e,
                             tl_zadan=Baza(srv_e.url, bez_myslenia_e, rownolegle=1), tl_odp=Marian(),
                             min_slow=a.esej_min_slow, prompt=a.esej_prompt, material=a.esej_material)
        finally:
            srv_e.stop()
        a.esej = (f"{a.esej_model}{'+lora' if a.esej_lora else ''}-sft-pl" if a.esej_jezyk == "pl"
                  else f"{a.esej_model}-{a.esej_prompt}-en")
        czasy["esej"] = round(time.time() - t1, 1)
        log(f"esej ({a.esej_model}): {time.time() - t1:.0f} s")
    odp = zloz(exam, zadania, wczytaj_odp(pk), wczytaj_odp(pe), wybrany)
    bledy, ostrz = sprawdz(odp, szablon)
    wyj = a.wyniki / "answers.json"
    wyj.write_text(json.dumps(odp, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.wyniki / "pm_zestawy.txt").write_text(f"{a.model} | {a.krotkie}+{a.esej}={pk},{pe}\n", encoding="utf-8")
    for o in ostrz:
        log(f"ostrzeżenie: {o}")
    for b in bledy:
        log(f"BŁĄD: {b}")
    czasy["calosc"] = round(time.time() - t0, 1)
    (a.wyniki / "przebieg.json").write_text(json.dumps(
        {"czasy_s": czasy, "wybrany_temat": wybrany["nr_tematu"], "regula_tematu": a.regula_tematu, "bledy": bledy,
         "ostrzezenia": ostrz, "krotkie": a.krotkie, "esej": a.esej, "esej_model": a.esej_model},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"zapisano {wyj} ({'BŁĘDY' if bledy else 'format OK'}); czas całości {time.time() - t0:.0f} s")
    return 1 if bledy else 0


if __name__ == "__main__":
    sys.exit(main())
