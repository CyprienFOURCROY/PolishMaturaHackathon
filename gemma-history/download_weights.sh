#!/usr/bin/env bash
set -euo pipefail
model_root="${GEMMA_MODEL_ROOT:-/workspace/gemma-matura}"
base_url='https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF/resolve/main'
mkdir -p "$model_root/model/gemma3-12b-q4ks" "$model_root/model/gemma3-12b-iq3"
fetch_weight() {
  local remote_name="$1" target_file="$2" expected_bytes="$3"
  if [ -f "$target_file" ]; then
    python3 - "$target_file" "$expected_bytes" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1]);expected=int(sys.argv[2])
if p.stat().st_size!=expected:raise SystemExit('Existing file has unexpected size; inspect manually: '+str(p))
print('Keep existing:',p)
PY
    return
  fi
  curl --fail --location --retry 5 --continue-at - --output "$target_file.part" "$base_url/$remote_name"
  python3 - "$target_file.part" "$expected_bytes" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1])
if p.stat().st_size!=int(sys.argv[2]):raise SystemExit('Download size mismatch: '+str(p))
PY
  mv "$target_file.part" "$target_file"
}
fetch_weight google_gemma-3-12b-it-Q4_K_S.gguf "$model_root/model/gemma3-12b-q4ks/google_gemma-3-12b-it-Q4_K_S.gguf" 6935130144
fetch_weight mmproj-google_gemma-3-12b-it-f16.gguf "$model_root/model/gemma3-12b-iq3/mmproj-gemma-3-12b-it-f16.gguf" 854200224
