"""Opisy wzorcowe ilustracji (Claude Opus z wizją) do pomiaru SUFITU narzędzia do obrazów.

Co to jest: dla każdej ilustracji zadań z podanych sesji dev pisze rzeczowy opis (typ źródła, napisy, co
przedstawione) i zapisuje go w cache matura/obrazy (wariant "wzorzec"). Opisujący NIE widzi polecenia zadania.
Po co: zmierzyć, ile punktów pełnej matury dałby idealny opis obrazu (konfiguracja h0_wzorzec). Jeśli mało,
obrazy nie są wąskim gardłem małych modeli; jeśli dużo, warto inwestować w lepszy opisywacz w limicie rozmiaru.
Co zrobić: `uv run python scripts/opisy_wzorcowe.py --sesje 2024-maj,2025-maj`. TYLKO pomiar na dev w fazie
budowy; na egzaminie zamknięte API są zakazane, a harness czyta wyłącznie cache.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matura import devset, obrazy  # noqa: E402

WARIANT = "wzorzec"
PROMPT = """Przeczytaj plik obrazu {sciezka} narzędziem Read. To ilustracja ze źródła historycznego z arkusza
maturalnego z historii. Podpis źródła w arkuszu: „{podpis}”.
Opisz ją rzeczowo po polsku dla osoby, która jej nie widzi (80-180 słów, zwykły tekst bez nagłówków):
1) typ źródła (fotografia, mapa, karykatura, plakat, moneta, medal, obraz, rysunek, schemat, tabela…);
2) wszystkie widoczne napisy dosłownie (daty, nazwy, hasła, legenda i oznaczenia mapy);
3) co i kto jest przedstawiony: postacie, symbole, atrybuty; na mapie obszary, granice, strzałki, miasta;
4) jeśli na podstawie widocznych cech rozpoznajesz powszechnie znany obiekt, postać albo wydarzenie, nazwij je.
Nie odpowiadaj na żadne pytanie i nie interpretuj ponad to, co widać. Zwróć tylko opis."""
# Wariant "wzorzec_wzrok": sam wzrok, bez wiedzy opisującego (czy zysk sufitu to widzenie, czy rozpoznanie).
PROMPT_WZROK = PROMPT.replace(
    "4) jeśli na podstawie widocznych cech rozpoznajesz powszechnie znany obiekt, postać albo wydarzenie, nazwij je.",
    "4) NIE nazywaj osób, miejsc, obiektów ani wydarzeń, których nazw nie ma w napisach na obrazie ani w podpisie; "
    "opisz tylko wygląd.")


def opisz(il: dict, timeout: int = 300, prompt: str = PROMPT) -> tuple[str, float]:
    t0 = time.time()
    cmd = ["claude", "-p", "--model", "opus", "--no-session-persistence", "--strict-mcp-config",
           "--disable-slash-commands", "--allowedTools", "Read", "--add-dir", str(ROOT / "data" / "cke" / "obrazy")]
    r = subprocess.run(cmd, input=prompt.format(sciezka=il["sciezka"], podpis=il["podpis"] or "(brak)"),
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT / "data" / "cke" / "obrazy")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"claude exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
    return r.stdout.strip(), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesje", default="2024-maj,2025-maj")
    ap.add_argument("--rownolegle", type=int, default=6)
    ap.add_argument("--wzrok", action="store_true", help="wariant wzorzec_wzrok: bez rozpoznawania (sam wygląd)")
    a = ap.parse_args()
    wariant, prompt = ("wzorzec_wzrok", PROMPT_WZROK) if a.wzrok else (WARIANT, PROMPT)
    ilu = {}
    for s in a.sesje.split(","):
        for z in devset._cke(s):
            for il in obrazy.ilustracje(z):
                ilu.setdefault(il["rel"], il)
    cache = obrazy.wczytaj_cache()
    todo = [il for rel, il in ilu.items() if obrazy.klucz(rel, wariant) not in cache]
    print(f"ilustracji: {len(ilu)}, do opisania: {len(todo)}", flush=True)
    bledy = 0
    with ThreadPoolExecutor(a.rownolegle) as ex:
        futs = {ex.submit(opisz, il, 300, prompt): il for il in todo}
        for i, f in enumerate(as_completed(futs), 1):
            il = futs[f]
            try:
                tekst, sek = f.result()
            except Exception as e:  # noqa: BLE001 (limit, timeout: ponowne uruchomienie dokończy)
                bledy += 1
                print(f"[{i}/{len(todo)}] BŁĄD {il['rel']}: {str(e)[:200]}", flush=True)
                continue
            obrazy._zapisz(il["rel"], wariant, tekst, sek, opisujacy="claude:opus")
            print(f"[{i}/{len(todo)}] {il['rel']} ({sek:.0f} s): {tekst[:90]}…", flush=True)
    print(f"KONIEC, błędów: {bledy}", flush=True)
    return 1 if bledy else 0


if __name__ == "__main__":
    sys.exit(main())
