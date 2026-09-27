# Installation and model files

## Preferred: existing L40S session

Use the existing installation under /workspace/gemma-matura. It already contains the working CUDA runtime and both model files. The repository client requires Python 3.10+ with the standard library only. Do not replace the server or Python environment merely to use this client

From this gemma-history directory:

    curl --fail --max-time 10 http://127.0.0.1:8080/health
    python3 final_run.py --exam /absolute/path/to/exam.json --out runs/preflight --check-only

If the server is absent, run bash serve_final.sh in a separate terminal. Do not launch it while port 8080 is already occupied. Server startup logs must show the selected Q4_K_S weights and F16 projector

## New Linux GPU machine: prerequisites

This is an installation recipe, not a newly validated clean-machine benchmark. Budget time for downloading about 7.8 GB and compiling. All project files below stay under /workspace

Requirements: NVIDIA driver with CUDA support, CUDA toolkit with nvcc, C/C++ compiler, CMake, Git, curl and Python 3.10+. NVIDIA-SMI displaying a CUDA version does not prove nvcc is installed. Provision missing build tools before continuing. Ubuntu 22.04 has glibc 2.35; prebuilt binaries requiring glibc 2.38 failed in our session

Check:

    nvidia-smi
    command -v nvcc
    cmake --version
    c++ --version
    python3 --version

The original session used a build referred to as b11200 and reporting short commit 81bc6b8. Resolve and record the actual full commit on the new machine; do not treat the short identifier as a complete reproducibility lock

Example source build for L40S (CUDA architecture 89):

    mkdir -p /workspace/gemma-matura/runtime
    git clone --branch b11200 --depth 1 https://github.com/ggml-org/llama.cpp.git /workspace/gemma-matura/runtime/llama-source
    cmake -S /workspace/gemma-matura/runtime/llama-source -B /workspace/gemma-matura/runtime/llama-source/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build /workspace/gemma-matura/runtime/llama-source/build -j 4 --target llama-server
    mkdir -p /workspace/gemma-matura/runtime/llama-b11200-local
    cp -a /workspace/gemma-matura/runtime/llama-source/build/bin/. /workspace/gemma-matura/runtime/llama-b11200-local/

Use a CUDA architecture appropriate to a different GPU. These commands require the source directory not to exist already; do not delete an existing working installation

Download weights with the included script:

    bash download_weights.sh

The remote URLs use main because the exact revision of the originally downloaded files was not captured. Expected byte sizes are checked, but this is not cryptographic reproduction of the original weights. Keep working files already present. Record local SHA-256 for the actual submission installation:

    sha256sum /workspace/gemma-matura/model/gemma3-12b-q4ks/google_gemma-3-12b-it-Q4_K_S.gguf /workspace/gemma-matura/model/gemma3-12b-iq3/mmproj-gemma-3-12b-it-f16.gguf

Then bash serve_final.sh, followed by the preflight command. No external model APIs are used at inference time. Acquisition of model files requires internet during setup. The model weights have their own upstream terms; repository documentation does not replace those terms
