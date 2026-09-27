"""Przygotowanie maszyny do egzaminu finałowego (laptop albo chmura; Linux, macOS, Windows).

Co to jest: pobiera 4 modele zestawu v2 (dokładne pliki i commity z SOURCE.md, `scripts/pobierz_modele.py`, około 5,8 GB)
i sprawdza, czy jest llama-server (zmienna LLAMA_SERVER, data/bin/llama-cuda albo PATH).
Po co: po `uv sync` jedno polecenie i maszyna jest gotowa; `scripts/egzamin_final.py` uruchamia potem cały egzamin.
Co zrobić: `uv run python scripts/przygotuj.py` (wznawialne: pobrane pliki o dobrym rozmiarze są pomijane).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ROLE_V2 = "baza,opisywacz,esej_lfm2,marian_en_pl"
PORADA = """llama-server not found. Install llama.cpp and either put llama-server on PATH or set LLAMA_SERVER:
  macOS:   brew install llama.cpp
  Windows: download llama-*-bin-win-cuda-12.4-x64.zip AND cudart-llama-bin-win-cuda-12.4-x64.zip (NVIDIA; unzip both
           into ONE folder) or -vulkan-/-cpu- from https://github.com/ggml-org/llama.cpp/releases, then
           set LLAMA_SERVER=C:\\path\\to\\llama-server.exe   (PowerShell: $env:LLAMA_SERVER="...")
  Linux:   the same page: NVIDIA = llama-*-bin-ubuntu-cuda-12.8-x64.tar.gz AND cudart-llama-*-bin-ubuntu-cuda-12.8-x64.tar.gz
           extracted into ONE folder; otherwise ubuntu-vulkan / ubuntu-x64; then export LLAMA_SERVER=/path/to/llama-server"""


def main() -> int:
    kod = subprocess.call([sys.executable, str(ROOT / "scripts" / "pobierz_modele.py"), "--tylko", ROLE_V2])
    from matura.noc import llama_server
    ls = llama_server()
    try:
        urz = subprocess.run([ls, "--list-devices"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        print(PORADA)
        return 1
    tekst = (urz.stdout + urz.stderr).split("Available devices:")[-1].strip()
    print(f"llama-server OK: {ls}\ndevices: {tekst or '(none)'}")
    if not tekst or "(none)" in tekst:
        print("WARNING: no GPU backend found, llama.cpp will run on CPU (our 16-core test: ~15 min per exam instead "
              "of ~1.5 min). NVIDIA: extract BOTH release archives (llama-...-cuda-... and cudart-llama-...-cuda-...) "
              "into the SAME folder; macOS: brew install llama.cpp (Metal).")
    print("models:", "OK" if kod == 0 else "ERROR (see above)")
    return kod


if __name__ == "__main__":
    sys.exit(main())
