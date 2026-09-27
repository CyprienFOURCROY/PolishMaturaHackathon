#!/usr/bin/env bash
# Build without sudo, changing system packages, or replacing a working runtime.
set -euo pipefail
model_root="${GEMMA_MODEL_ROOT:-/workspace/gemma-matura}"
source_ref="${LLAMA_BUILD_REF:-b11200}"
build_jobs="${GEMMA_BUILD_JOBS:-4}"
cuda_arch="${GEMMA_CUDA_ARCH:-89}"
runtime_dir="$model_root/runtime/llama-b11200-local"
source_dir="$model_root/runtime/llama-source"
case "$model_root" in /workspace/*) ;; *) echo 'GEMMA_MODEL_ROOT must be under /workspace'; exit 1;; esac
for required_cmd in git cmake c++ nvcc python3; do
  command -v "$required_cmd" >/dev/null || { echo "Missing dependency: $required_cmd. Provision it before building; no system packages were changed."; exit 1; }
done
case "$build_jobs" in ''|*[!0-9]*|0) echo 'GEMMA_BUILD_JOBS must be a positive integer'; exit 1;; esac
mkdir -p "$model_root/runtime"
if [ -e "$runtime_dir" ]; then
  echo "Runtime already exists: $runtime_dir"
  echo 'Keeping it unchanged. To build separately, set GEMMA_MODEL_ROOT to a new /workspace directory.'
  exit 1
fi
if [ -e "$source_dir" ]; then
  echo "Source directory already exists: $source_dir"
  echo 'Keeping it unchanged. Use a new GEMMA_MODEL_ROOT for a clean build.'
  exit 1
fi
git clone --depth 1 --branch "$source_ref" https://github.com/ggml-org/llama.cpp.git "$source_dir"
source_commit="$(git -C "$source_dir" rev-parse HEAD)"
cmake -S "$source_dir" -B "$source_dir/build" -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$source_dir/build" --config Release -j "$build_jobs" --target llama-server
stage_dir="$(mktemp -d "$model_root/runtime/.llama-stage.XXXXXX")"
cp -a "$source_dir/build/bin/." "$stage_dir/"
export LD_LIBRARY_PATH="$stage_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$stage_dir/llama-server" --version 2>&1 | tee "$stage_dir/build-version.txt"
"$stage_dir/llama-server" --list-devices 2>&1 | tee "$stage_dir/build-devices.txt"
if ! grep -q 'CUDA[0-9]' "$stage_dir/build-devices.txt"; then
  echo "No CUDA device detected. Build retained for diagnosis at $stage_dir; runtime not installed."
  exit 1
fi
printf '%s\n' "$source_commit" > "$stage_dir/source-commit.txt"
printf '%s\n' "$source_ref" > "$stage_dir/source-ref.txt"
printf '%s\n' "$cuda_arch" > "$stage_dir/cuda-architecture.txt"
nvcc --version > "$stage_dir/nvcc-version.txt"
cmake --version > "$stage_dir/cmake-version.txt"
mv "$stage_dir" "$runtime_dir"
echo "READY: $runtime_dir/llama-server"
echo "Source commit: $source_commit"
echo 'Next: bash download_weights.sh, then bash serve_final.sh'
