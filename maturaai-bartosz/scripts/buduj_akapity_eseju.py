"""Baza wiedzy do eseju e6 i zadań otwartych: akapity argumentacyjne (i opcjonalnie hasła) z Claude, wg działów podstawy.

Co to jest: ten sam generator co w dokumencie dla zespołu („MaturaAI: knowledge base cards (team guide)”), z wymuszonym
formatem (--json-schema), równoległy i wznawialny. Wyjście: data/wiedza/akapity_esej_claude.jsonl,
data/wiedza/hasla_claude.jsonl (id z sufiksem „-c”, żeby nie zderzały się z plikami zespołu; `wiedza.BazaAkapitow`
scala wszystkie pliki akapity_esej*.jsonl).
Po co: esej e6 wkleja akapity z bazy; fakty z bazy są poprawne, a małe modele zmyślają (sędzia: >5 błędów = −3 pkt).
Zasada: prompty dostają tylko nazwę działu podstawy programowej, nigdy zadań ani tematów z arkuszy CKE 2023-2026.
Co zrobić: uv run python scripts/buduj_akapity_eseju.py --dzialy 5,12,21 [--hasla] [--rownolegle 4] [--effort low]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAT = ROOT / "data" / "wiedza"
S = ["starozytnosc-sredniowiecze", "nowozytnosc", "xix", "xx-xxi"]
DZIALY = {
    1: ("Pradzieje i historia starożytnego Wschodu", S[0]), 2: ("Świat starożytnych Greków", S[0]),
    3: ("Społeczeństwo, życie polityczne i kultura starożytnego Rzymu", S[0]), 4: ("Bizancjum i świat islamu", S[0]),
    5: ("Europa wczesnego średniowiecza", S[0]), 6: ("Polska w okresie wczesnopiastowskim", S[0]),
    7: ("Europa w okresie krucjat", S[0]), 8: ("Gospodarcze i społeczne realia średniowiecznej Europy", S[0]),
    9: ("Kultura średniowiecza", S[0]), 10: ("Polska w okresie rozbicia dzielnicowego", S[0]),
    11: ("Europa późnego średniowiecza", S[0]), 12: ("Polska w XIV i XV wieku", S[0]),
    13: ("Odkrycia geograficzne i europejski kolonializm doby nowożytnej", S[1]), 14: ("Czasy renesansu", S[1]),
    15: ("Reformacja i jej skutki", S[1]), 16: ("Europa w XVI i XVII wieku", S[1]), 17: ("Renesans w Polsce", S[1]),
    18: ("Państwo polsko-litewskie w czasach ostatnich Jagiellonów", S[1]), 19: ("Powstanie Rzeczypospolitej Obojga Narodów", S[1]),
    20: ("Pierwsze wolne elekcje i ich następstwa", S[1]),
    21: ("Polityka wewnętrzna i zagraniczna Rzeczypospolitej Obojga Narodów w XVII wieku", S[1]),
    22: ("Europa w dobie oświecenia", S[1]), 23: ("Rewolucje XVIII wieku (amerykańska i francuska)", S[1]),
    24: ("Rzeczpospolita w XVIII wieku: czasy saskie, reformy, Konstytucja 3 maja, rozbiory, powstanie kościuszkowskie", S[1]),
    25: ("Epoka napoleońska", S[2]), 26: ("Europa i świat po kongresie wiedeńskim", S[2]),
    27: ("Ziemie polskie i ich mieszkańcy w latach 1815-1848", S[2]), 28: ("Powstanie styczniowe i jego następstwa", S[2]),
    29: ("Europa i świat w II połowie XIX i na początku XX wieku", S[2]),
    30: ("Ziemie polskie pod zaborami w II połowie XIX i na początku XX wieku", S[2]),
    31: ("Kultura i nauka polska w II połowie XIX i na początku XX wieku", S[2]), 32: ("I wojna światowa", S[3]),
    33: ("Sprawa polska w przededniu i podczas I wojny światowej", S[3]), 34: ("Europa i świat po I wojnie światowej", S[3]),
    35: ("Walka o odrodzenie państwa polskiego po I wojnie światowej", S[3]), 36: ("Dzieje polityczne II Rzeczypospolitej", S[3]),
    37: ("Społeczeństwo i gospodarka II Rzeczypospolitej", S[3]),
    38: ("Narodziny i rozwój totalitaryzmów w okresie międzywojennym", S[3]),
    39: ("Świat na drodze do II wojny światowej", S[3]), 40: ("II wojna światowa i jej etapy", S[3]),
    41: ("Polska pod okupacją niemiecką i sowiecką", S[3]),
    42: ("Proces przejmowania władzy przez komunistów w Polsce (1944-1948)", S[3]),
    43: ("Stalinizm w Polsce i jego erozja", S[3]), 44: ("Zimna wojna (1945-1991)", S[3]),
    45: ("Dekolonizacja, integracja europejska i nowe konflikty", S[3]), 46: ("Przemiany cywilizacyjne na świecie po 1945 roku", S[3]),
    47: ("Polska w latach 1957-1981", S[3]), 48: ("Dekada 1981-1989", S[3]),
    49: ("Świat na przełomie tysiącleci", S[3]), 50: ("Polska po 1989 roku", S[3]),
}
PROMPT_1 = """Przygotowujesz bazę wiedzy do matury z historii (poziom rozszerzony, formuła 2023) dla małego modelu,
który na egzaminie wyszukuje hasła słowami kluczowymi. Dział podstawy programowej: {DZIAL}.
Napisz 6-10 haseł, które razem pokrywają ten dział (wydarzenia, procesy, postaci, dokumenty, pojęcia).
Każde hasło: tytul; tekst 150-400 słów zwartych faktów (daty, nazwy, przyczyny, skutki); daty; postacie;
pojecia z definicją w jednym zdaniu; 1-5 kluczowych dokumentów lub źródeł z datą i jednozdaniową treścią;
0-5 znanych ilustracji związanych z tematem (obraz, plakat, karykatura, fotografia, mapa, moneta) z autorem,
datą i tym, co przedstawia. Pisz prostymi zdaniami, zawsze z nazwą podmiotu (bez „on”, „ten władca”).
Tylko fakty pewne; datę podaj tylko, jeśli jesteś jej pewien. Nie korzystaj z arkuszy maturalnych CKE
z lat 2023-2026. Zwróć wyłącznie JSON: {"hasla": [...]}."""
PROMPT_A = """Dział podstawy programowej historii (poziom rozszerzony): {DZIAL}.
Podaj 5 tez w stylu tematów wypracowań maturalnych CKE (oceniające: przełom, sukces albo klęska,
najważniejszy czynnik, dominowały, w największym stopniu, podobne przyczyny, apogeum). Dla każdej tezy:
podmiot, okres (lata), tryb („aspekty” albo „elementy”) i pozycje: trzy aspekty albo 3-5 przykładów
(władców, państw, wydarzeń, postaci). Nie kopiuj tematów z arkuszy CKE 2023-2026.
Zwróć wyłącznie JSON: {"tezy": [{"teza", "podmiot", "okres", "tryb", "pozycje": [...]}]}."""
PROMPT_B = """Teza: {TEZA}. Podmiot: {PODMIOT}, okres {OKRES}. Pozycje: {POZYCJE}.
Dla każdej pozycji napisz dwa akapity wypracowania maturalnego z historii: jeden potwierdzający tezę
(kierunek „potwierdza”), jeden przeczący jej (kierunek „przeczy”). Akapit: 90-140 słów po polsku; pierwsze
zdanie mówi, co ta pozycja pokazuje wobec tezy; potem 3-4 konkretne fakty z datami i nazwami; ostatnie
zdanie wiąże fakty z tezą. Zawsze pełna nazwa podmiotu, bez zaimków odsyłających poza akapit. Tylko fakty
pewne; styl nie jest oceniany, liczy się poprawność. Zwróć wyłącznie JSON:
{"akapity": [{"pozycja", "kierunek", "tekst", "fakty": [{"fakt", "data"}]}]}."""
_str = {"type": "string"}
_lista = lambda it: {"type": "array", "items": it}
_obj = lambda **p: {"type": "object", "additionalProperties": False, "required": list(p), "properties": p}
SCHEMAT_1 = _obj(hasla=_lista(_obj(tytul=_str, tekst=_str, daty=_lista(_str), postacie=_lista(_str),
                                   pojecia=_lista(_obj(termin=_str, definicja=_str)),
                                   dokumenty=_lista(_obj(nazwa=_str, data=_str, tresc=_str)),
                                   ikonografia=_lista(_obj(nazwa=_str, autor=_str, data=_str, co_przedstawia=_str)))))
SCHEMAT_A = _obj(tezy=_lista(_obj(teza=_str, podmiot=_str, okres=_str, tryb={"type": "string", "enum": ["aspekty", "elementy"]},
                                  pozycje=_lista(_str))))
SCHEMAT_B = _obj(akapity=_lista(_obj(pozycja=_str, kierunek={"type": "string", "enum": ["potwierdza", "przeczy"]}, tekst=_str,
                                     fakty=_lista(_obj(fakt=_str, data=_str)))))
_lock = threading.Lock()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def claude(prompt: str, schemat: dict, effort: str | None, model: str = "opus", timeout: int = 1800) -> dict:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--json-schema", json.dumps(schemat),
           "--no-session-persistence", "--strict-mcp-config"] + (["--effort", effort] if effort else [])
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=td)
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    out = json.loads(r.stdout)
    if out.get("structured_output") is not None:
        return out["structured_output"]
    t = out.get("result", "")
    return t if isinstance(t, dict) else json.loads(t[t.find("{"): t.rfind("}") + 1])


def daty_ok(tekst: str, daty) -> bool:
    return all(y in tekst for d in daty for y in re.findall(r"\d{3,4}", str(d)))


def gotowe(p: Path) -> set[str]:
    return {json.loads(l)["id"].split("-")[0] for l in p.open(encoding="utf-8") if l.strip()} if p.exists() else set()


def dopisz(p: Path, wiersze: list[dict]) -> None:
    with _lock, p.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in wiersze)


def dzial(n: int, hasla: bool, effort: str | None, ph: Path, pa: Path, akapity: bool = True) -> None:
    nazwa, epoka = DZIALY[n]
    baza = {"dzial": nazwa, "epoka": epoka, "zrodlo": "claude-opus, 2026-09-26, scripts/buduj_akapity_eseju.py"}
    if hasla and f"S{n}" not in gotowe(ph):
        w = []
        for i, h in enumerate(claude(PROMPT_1.replace("{DZIAL}", nazwa), SCHEMAT_1, effort)["hasla"], 1):
            if 150 <= len(h["tekst"].split()) <= 400 and daty_ok(h["tekst"], h["daty"]):
                w.append({"id": f"S{n}-{i:02d}-c", **baza, **h})
            else:
                log(f"dział {n}: odrzucone hasło {h['tytul']!r} ({len(h['tekst'].split())} słów)")
        dopisz(ph, w)
        log(f"dział {n}: hasła {len(w)}")
    if not akapity or f"E{n}" in gotowe(pa):
        return
    w = []  # dział zapisujemy w całości po ostatniej tezie: przerwany dział przy wznowieniu powstaje od nowa
    for j, t in enumerate(claude(PROMPT_A.replace("{DZIAL}", nazwa), SCHEMAT_A, effort)["tezy"], 1):
        p = (PROMPT_B.replace("{TEZA}", t["teza"]).replace("{PODMIOT}", t["podmiot"]).replace("{OKRES}", t["okres"])
             .replace("{POZYCJE}", ", ".join(t["pozycje"])))
        for k, a in enumerate(claude(p, SCHEMAT_B, effort)["akapity"], 1):
            if 90 <= len(a["tekst"].split()) <= 140 and daty_ok(a["tekst"], [f["data"] for f in a["fakty"]]):
                w.append({"id": f"E{n}-{j}{k:02d}-c", **baza, "teza": t["teza"], "podmiot": t["podmiot"],
                          "okres": t["okres"], "tryb": t["tryb"], **a})
            else:
                log(f"dział {n}: odrzucony akapit {a['pozycja']!r}/{a['kierunek']} ({len(a['tekst'].split())} słów)")
    dopisz(pa, w)
    log(f"dział {n}: akapity {len(w)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dzialy", default="", help="numery po przecinku (domyślnie wszystkie 1-50)")
    ap.add_argument("--hasla", action="store_true", help="także hasła do zadań otwartych (Task 1)")
    ap.add_argument("--bez-akapitow", action="store_true", help="tylko hasła (Task 1), bez akapitów eseju")
    ap.add_argument("--rownolegle", type=int, default=4)
    ap.add_argument("--effort", default=None, help="poziom wysiłku claude -p (np. low, medium); domyślnie ustawienie konta")
    a = ap.parse_args(argv)
    KAT.mkdir(parents=True, exist_ok=True)
    ph, pa = KAT / "hasla_claude.jsonl", KAT / "akapity_esej_claude.jsonl"
    nr = [int(x) for x in a.dzialy.split(",") if x.strip()] or sorted(DZIALY)
    bledy = []

    def jeden(n):
        try:
            dzial(n, a.hasla, a.effort, ph, pa, akapity=not a.bez_akapitow)
        except Exception as e:  # noqa: BLE001 (limit, timeout, zły JSON): dział do ponowienia
            bledy.append(n); log(f"dział {n}: BŁĄD {str(e)[:300]}")

    with ThreadPoolExecutor(max(1, a.rownolegle)) as ex:
        list(ex.map(jeden, nr))
    log(f"koniec; działy z błędem: {sorted(bledy)}")
    return 1 if bledy else 0


if __name__ == "__main__":
    sys.exit(main())
