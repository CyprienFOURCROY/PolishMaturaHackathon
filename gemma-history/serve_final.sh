#!/usr/bin/env bash
set -euo pipefail
model_root="${GEMMA_MODEL_ROOT:-/workspace/gemma-matura}"
server_bin="$model_root/runtime/llama-b11200-local/llama-server"
model_file="$model_root/model/gemma3-12b-q4ks/google_gemma-3-12b-it-Q4_K_S.gguf"
projector_file="$model_root/model/gemma3-12b-iq3/mmproj-gemma-3-12b-it-f16.gguf"
test -x "$server_bin"
test -s "$model_file"
test -s "$projector_file"
export LD_LIBRARY_PATH="$model_root/runtime/llama-b11200-local${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$server_bin" -m "$model_file" --mmproj "$projector_file" --alias gemma-matura --host 127.0.0.1 --port 8080 -ngl 99 -c 16384 -np 1 -t 4 --jinja
