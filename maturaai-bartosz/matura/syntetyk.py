"""Generator zadań w stylu matury z historii z fragmentów Wikipedii PL (nauczyciel: Claude albo Astra, subskrypcja).

Dlaczego z Wikipedii: organizatorzy opisują egzamin jako „matura-style questions from Polish Wikipedia".
Dwa rozłączne zbiory artykułów (podział po haszu tytułu → brak wycieku):
- pula EGZAMINACYJNA (hash % 10 == 0) → `--arkusze N`: sztuczne arkusze po 16 pkt → data/synt/arkusze.json
- pula TRENINGOWA (reszta)             → `--sft N`: zadania + wzorowe odpowiedzi → data/sft/zadania.jsonl
Wznawialne: każda partia dopisywana od razu; ponowne uruchomienie dobiera tylko brakujące.
Uruchomienie: uv run python -m matura.syntetyk --arkusze 6 --sft 2400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .sedzia import llm_json

ROOT = Path(__file__).resolve().parent.parent
FRAG = ROOT / "data" / "wiki" / "fragmenty.jsonl"
SYNT_DIR = ROOT / "data" / "synt"
SFT_DIR = ROOT / "data" / "sft"
_lock = threading.Lock()

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["zadania"], "properties": {"zadania": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["pkt_max", "rodzaj", "zrodla_tekst", "polecenie", "zasady_oceniania",
                                            "rozwiazanie", "odpowiedz_wzorowa", "zrodlo_tytul"],
                               "properties": {"pkt_max": {"type": "integer"}, "rodzaj": {"type": "string"},
                                              "zrodla_tekst": {"type": "string"}, "polecenie": {"type": "string"},
                                              "zasady_oceniania": {"type": "string"}, "rozwiazanie": {"type": "string"},
                                              "odpowiedz_wzorowa": {"type": "string"}, "zrodlo_tytul": {"type": "string"}}}}}}

INSTR = """Jesteś autorem arkuszy CKE: matura z historii, poziom rozszerzony, formuła 2023.
Na podstawie podanych fragmentów Wikipedii ułóż {n} zadań. Wymagania:
- Styl i polecenia jak w prawdziwych arkuszach CKE: „Wyjaśnij…", „Podaj dwa…", „Rozstrzygnij, czy… Odpowiedź uzasadnij, odwołując się do źródła", „Oceń prawdziwość informacji… Zaznacz P albo F", „Zaznacz właściwą odpowiedź spośród A–D", „Uporządkuj chronologicznie…".
- Mieszanka rodzajów: {mix}.
- Większość zadań ze źródłem: w polu zrodla_tekst podaj „Źródło. <typ>" i tekst źródła (3–8 zdań) — może to być parafraza fragmentu encyklopedii albo krótki, wierny cytat dokumentu epoki, jeśli jest znany. Zadanie bez źródła: pusty string.
- W zadaniach zamkniętych opcje/zdania umieść w poleceniu.
- zasady_oceniania: jak w kluczu CKE („1 pkt – za …", „0 pkt – za odpowiedź niepełną lub błędną albo za brak odpowiedzi.").
- rozwiazanie: poprawna odpowiedź jak w kluczu CKE (dla otwartych: przykładowe odpowiedzi).
- odpowiedz_wzorowa: to, co powinien napisać bardzo dobry maturzysta — zwięźle, dokładnie tyle elementów, ile wymaga polecenie; dla „rozstrzygnij" dwie linie „Rozstrzygnięcie: …" i „Uzasadnienie: …"; dla zamkniętych same oznaczenia.
- Fakty wyłącznie zgodne z fragmentami i z rzetelną wiedzą historyczną. Nie zdradzaj odpowiedzi w źródle wprost, jeśli zadanie sprawdza wiedzę.
- zrodlo_tytul: tytuł fragmentu, na którym opiera się zadanie. rodzaj: zamkniete / otwarte / rozstrzygnij.
- {pkt}
Nie uruchamiaj poleceń. Zwróć wyłącznie JSON zgodny ze schematem.

FRAGMENTY WIKIPEDII:
"""


def pula(egzaminacyjna: bool, limit: int = 400000, seed: int = 7) -> list[dict]:
    """Fragmenty historyczne z jednej z dwóch rozłącznych pul (podział po haszu tytułu)."""
    out = []
    with open(FRAG, encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if not d["hist"] or len(d["tekst"].split()) < 80:
                continue
            h = int(hashlib.md5(d["tytul"].encode()).hexdigest(), 16) % 10
            if (h == 0) == egzaminacyjna:
                out.append(d)
    random.Random(seed).shuffle(out)
    return out[:limit]


def _dopisz(p: Path, rekordy: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(p, "a", encoding="utf-8") as f:
        for r in rekordy:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _partia(frs: list[dict], n: int, mix: str, pkt: str, model: str) -> list[dict]:
    tekst = "\n\n".join(f"[{f['tytul']}] {f['tekst']}" for f in frs)
    for proba in range(3):
        try:
            return llm_json(INSTR.format(n=n, mix=mix, pkt=pkt) + tekst, SCHEMA, model)["zadania"]
        except Exception as e:
            print(f"błąd generacji (próba {proba+1}): {str(e)[:200]}", flush=True)
            time.sleep(30 * (proba + 1))
    return []


def arkusze(n: int, model: str, rown: int) -> None:
    surowe = SYNT_DIR / "arkusze_surowe.jsonl"
    gotowe = {r["arkusz"] for r in _jsonl(surowe)}
    frs = pula(True)
    todo = [i for i in range(n) if i not in gotowe]

    def jeden(i):
        zad = _partia(frs[i * 8:(i + 1) * 8], 12, "3 zamknięte, 6 otwartych krótkich, 3 „rozstrzygnij i uzasadnij” ze źródłem",
                      "Suma pkt_max w całym arkuszu = dokładnie 16 (zadania po 1 albo 2 pkt).", model)
        for k, z in enumerate(zad):
            z.update(arkusz=i, id=f"synt-{i}-{k+1}", nr_zadania=str(k + 1), rok=0, sesja=f"synt{i}", esej=False, obrazy=[])
        _dopisz(surowe, zad)
        print(f"arkusz {i}: {len(zad)} zadań, {sum(z['pkt_max'] for z in zad)} pkt", flush=True)

    with ThreadPoolExecutor(rown) as ex:
        list(ex.map(jeden, todo))
    wszystkie = _jsonl(surowe)
    (SYNT_DIR / "arkusze.json").write_text(json.dumps(wszystkie, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"arkusze sztuczne: {len({z['arkusz'] for z in wszystkie})}, zadań {len(wszystkie)} → data/synt/arkusze.json")


def sft(n_zadan: int, model: str, rown: int, na_partie: int = 12) -> None:
    wyj = SFT_DIR / "zadania.jsonl"
    juz = _jsonl(wyj)
    partie_gotowe = {r["partia"] for r in juz}
    n_partii = (n_zadan + na_partie - 1) // na_partie
    frs = pula(False)
    todo = [i for i in range(n_partii) if i not in partie_gotowe]
    print(f"SFT: mam {len(juz)} zadań, partii do zrobienia {len(todo)}", flush=True)

    def jedna(i):
        zad = _partia(frs[i * 4:(i + 1) * 4], na_partie, "4 zamknięte, 5 otwartych, 3 „rozstrzygnij i uzasadnij”",
                      "Punktacja jak w CKE: 1–2 pkt za zadanie.", model)
        for k, z in enumerate(zad):
            z.update(partia=i, id=f"sft-{i}-{k}", esej=False, obrazy=[])
        _dopisz(wyj, zad)
        if i % 10 == 0:
            print(f"SFT partia {i}: +{len(zad)}", flush=True)

    with ThreadPoolExecutor(rown) as ex:
        list(ex.map(jedna, todo))
    print(f"SFT gotowe: {len(_jsonl(wyj))} zadań → {wyj}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arkusze", type=int, default=0)
    ap.add_argument("--sft", type=int, default=0)
    ap.add_argument("--model", default="claude:fable", help="backend:model, np. claude:fable albo astra:gpt-6-astra")
    ap.add_argument("--rownolegle", type=int, default=3)
    a = ap.parse_args()
    if a.arkusze:
        arkusze(a.arkusze, a.model, a.rownolegle)
    if a.sft:
        sft(a.sft, a.model, a.rownolegle)
    sys.exit(0)
