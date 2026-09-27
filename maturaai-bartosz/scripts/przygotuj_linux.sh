#!/usr/bin/env bash
# Przygotowanie maszyny Linux x64 (laptop albo chmura, np. Forgehand) jednym poleceniem.
# Co to jest: instaluje uv (jeśli brak), pobiera oficjalny llama.cpp do data/bin/llama-cuda (tam szuka go harness,
# więc LLAMA_SERVER nie jest potrzebny), wybiera wariant, w którym llama-server widzi GPU (CUDA 12.8, CUDA 13.4,
# Vulkan, na końcu CPU), robi uv sync i scripts/przygotuj.py (4 modele, 5,8 GB); z --proba także egzamin próbny.
# Co zrobić: bash scripts/przygotuj_linux.sh [--proba]; potem uv run python scripts/egzamin_final.py --zip final.zip
set -euo pipefail
cd "$(dirname "$0")/.."
PROBA="${1:-}"
WERSJA=b11205
URL=https://github.com/ggml-org/llama.cpp/releases/download/$WERSJA
CEL=data/bin/llama-cuda
krok() { echo "=== $(date +%H:%M:%S) $*"; }

krok "1/4 uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

krok "2/4 llama.cpp $WERSJA"
urzadzenia() { "$CEL/llama-server" --list-devices 2>&1 | sed -n '/Available devices:/,$p' | tail -n +2; }
pobierz() {  # $1 = wariant archiwum (np. ubuntu-cuda-12.8-x64), $2 = archiwum cudart albo pusto
  rm -rf "$CEL" data/bin/tmp && mkdir -p data/bin/tmp
  curl -fsSL -o data/bin/tmp/llama.tar.gz "$URL/llama-$WERSJA-bin-$1.tar.gz" || return 1
  tar xzf data/bin/tmp/llama.tar.gz -C data/bin/tmp || return 1
  if [ -n "$2" ]; then  # biblioteki CUDA muszą leżeć obok llama-server, inaczej llama.cpp po cichu liczy na CPU
    curl -fsSL -o data/bin/tmp/cudart.tar.gz "$URL/$2" || return 1
    tar xzf data/bin/tmp/cudart.tar.gz -C data/bin/tmp || return 1
    cp data/bin/tmp/cudart-*/lib* data/bin/tmp/llama-$WERSJA/ || return 1
  fi
  mv data/bin/tmp/llama-$WERSJA "$CEL" && rm -rf data/bin/tmp
}
if [ -n "${LLAMA_SERVER:-}" ]; then
  echo "LLAMA_SERVER=$LLAMA_SERVER (set by you, not downloading llama.cpp)"
elif [ -x "$CEL/llama-server" ] && urzadzenia | grep -qE "CUDA|Vulkan"; then
  echo "already in $CEL:"; urzadzenia
else
  ok=""
  for w in "ubuntu-cuda-12.8-x64 cudart-llama-$WERSJA-bin-ubuntu-cuda-12.8-x64.tar.gz" \
           "ubuntu-cuda-13.4-x64 cudart-llama-$WERSJA-bin-ubuntu-cuda-13.4-x64.tar.gz" \
           "ubuntu-vulkan-x64 "; do
    set -- $w
    echo "trying $1"
    pobierz "$1" "${2:-}" || continue
    if urzadzenia | grep -qE "CUDA|Vulkan"; then ok=1; break; fi
  done
  if [ -z "$ok" ]; then
    echo "WARNING: no GPU visible to llama.cpp, using the CPU build (about 15 min per exam instead of 1.5 min)"
    pobierz ubuntu-x64 ""
  fi
  echo "devices:"; urzadzenia
fi

krok "3/4 Python dependencies and models"
uv sync
uv run python scripts/przygotuj.py

if [ "$PROBA" = "--proba" ]; then
  krok "4/4 rehearsal on the organisers' mock exam"
  curl -fsSL -o wyniki/mock.zip https://warsawmodeltrainers.dev/exams/history-2023-mock-v1.zip --create-dirs
  uv run python scripts/egzamin_final.py --zip wyniki/mock.zip --wyniki wyniki/proba
else
  krok "4/4 rehearsal skipped (add --proba to run the mock exam, about 2 min on a GPU)"
fi
echo
echo "READY. Final exam:  uv run python scripts/egzamin_final.py --zip final.zip"
if [ -x "$HOME/.local/bin/uv" ]; then echo "(in a new shell first: source \$HOME/.local/bin/env, so that uv is on PATH)"; fi
