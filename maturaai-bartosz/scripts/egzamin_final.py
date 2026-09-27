"""Egzamin finałowy jednym poleceniem: ZIP organizatorów → dwa answers.json (zestaw v2 i goła baza).

Co to jest: rozpakowuje paczkę (exam.json, images/, answers-template.json), uruchamia dokładnie polecenia z
`review/NAJLEPSZE.md` (zestaw v2, potem goła baza do przyrostu), sprawdza format i wypisuje ścieżki plików do formularza.
Po co: na scenie jest kilka minut na egzamin i prezentację; nic do przepisywania ręcznie.
Co zrobić: `uv run python scripts/egzamin_final.py --zip ~/Downloads/final.zip` (albo `--paczka <rozpakowany katalog>`).
Wynik: wyniki-final/zestaw/answers.json (Exam answers) i wyniki-final/baza/answers.json (Base model answers JSON).
Nie ustawiaj MATURA_TEMPERATURA (wyłącza głosowanie w zadaniach zamkniętych).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "data" / "modele"
ZESTAW = ["--model", "qwen3.5-4b-iq3xxs", "--gguf", str(M / "qwen3.5-4b-iq3xxs" / "Qwen3.5-4B-UD-IQ3_XXS.gguf"),
          "--krotkie", "goly_vlm_kb_z",
          "--opisywacz-gguf", str(M / "qwen3.5-2b-q4" / "Qwen3.5-2B-Q4_K_M.gguf"),
          "--opisywacz-mmproj", str(M / "qwen3.5-2b-q4" / "mmproj-F16.gguf"),
          "--esej-model", "lfm2-2.6b-q4km"]
BAZA = ["--model", "qwen3.5-4b-iq3xxs-goly", "--gguf", str(M / "qwen3.5-4b-iq3xxs" / "Qwen3.5-4B-UD-IQ3_XXS.gguf"),
        "--krotkie", "goly", "--esej", "goly"]


def paczka_z_zip(zip_: Path, cel: Path) -> Path:
    cel.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_) as z:
        z.extractall(cel)
    for p in [cel, *sorted(cel.rglob("*"))]:     # exam.json bywa w podkatalogu archiwum
        if p.is_dir() and (p / "exam.json").exists():
            return p
    raise SystemExit(f"no exam.json inside {zip_}")


def uruchom(nazwa: str, argumenty: list[str], paczka: Path, wyniki: Path) -> Path:
    t0 = time.time()
    cmd = [sys.executable, "-m", "matura.egzamin", "--paczka", str(paczka), *argumenty, "--wyniki", str(wyniki)]
    print(f"\n=== {nazwa}: {' '.join(cmd)}", flush=True)
    kod = subprocess.call(cmd, cwd=ROOT)
    odp = wyniki / "answers.json"
    if kod != 0 or not odp.exists():
        raise SystemExit(f"{nazwa}: FAILED (exit {kod}); log above")
    n = len(json.loads(odp.read_text(encoding="utf-8"))["answers"])
    print(f"=== {nazwa}: OK, {n} answers, {time.time() - t0:.0f} s → {odp}", flush=True)
    return odp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", type=Path, help="final exam ZIP downloaded from the submissions page")
    g.add_argument("--paczka", type=Path, help="already unpacked exam folder (exam.json, images/, answers-template.json)")
    ap.add_argument("--wyniki", type=Path, default=ROOT / "wyniki-final")
    ap.add_argument("--bez-bazy", action="store_true", help="only our setup (skip the bare base model run)")
    a = ap.parse_args(argv)
    if os.environ.get("MATURA_TEMPERATURA"):
        raise SystemExit("unset MATURA_TEMPERATURA first (it disables voting for closed tasks)")
    paczka = paczka_z_zip(a.zip, a.wyniki / "paczka") if a.zip else a.paczka
    zestaw = uruchom("our setup (v2)", ZESTAW, paczka, a.wyniki / "zestaw")
    baza = None if a.bez_bazy else uruchom("bare base model", BAZA, paczka, a.wyniki / "baza")
    print("\nSUBMIT:\n  Exam answers:            ", zestaw, "\n  Base model answers JSON: ", baza or "(skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
