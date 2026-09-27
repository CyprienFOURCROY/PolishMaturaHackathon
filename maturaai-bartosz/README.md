# MaturaAI: a small open model takes the Polish history matura

Warsaw Model Trainers hackathon, 25-27.09.2026. Our final setup (v2) scored **40.0%** on the organisers' mock exam
(bare base model 16.67%, **+23.3 p.p.**). Everything needed to run the final exam is in this folder; the models are
downloaded by a script (exact files and commits in `SOURCE.md`).

## Run the final exam (setup ~10 min, run ~2 min on a GPU)

**You need:** [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12 for you),
**llama.cpp `llama-server`**, ~7 GB of free disk, internet only for the setup.
- macOS (Apple Silicon): `brew install llama.cpp`
- Windows / Linux with NVIDIA: from https://github.com/ggml-org/llama.cpp/releases download **both** archives and
  extract them into **the same folder** (otherwise the GPU is silently not used):
  Linux `llama-<ver>-bin-ubuntu-cuda-12.8-x64.tar.gz` + `cudart-llama-<ver>-bin-ubuntu-cuda-12.8-x64.tar.gz`,
  Windows `llama-<ver>-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip`.
  Without NVIDIA: `*-vulkan-*` or `*-cpu-*`. Then point to it:
  `export LLAMA_SERVER=/path/to/llama-server` (PowerShell: `$env:LLAMA_SERVER="C:\path\llama-server.exe"`).
  If `llama-server` is on your PATH, nothing to set.

**Linux x64 (laptop or a cloud GPU machine, e.g. Forgehand): one command.** After cloning (the repository is private,
so log in to GitHub on that machine first: `gh auth login`, an SSH key, or HTTPS with a personal access token):
```bash
cd PolishMaturaHackathon/maturaai-bartosz
bash scripts/przygotuj_linux.sh --proba   # uv, official llama.cpp in data/bin (GPU variant picked by what it sees), models, mock exam
```
It must end with `SUBMIT:` (mock) and `READY.`; then go to step 3. On macOS or Windows follow steps 1-2 below.

**1. Setup** (once):
```bash
git clone -b bartek git@github.com:CyprienFOURCROY/PolishMaturaHackathon.git
cd PolishMaturaHackathon/maturaai-bartosz
uv sync                                   # Python deps (torch CPU is enough: it only runs the small translator)
uv run python scripts/przygotuj.py        # downloads the 4 models (5.8 GB) and checks llama-server
```
It prints the devices llama.cpp sees: your GPU must be listed (e.g. `CUDA0: NVIDIA ...`, or Metal on a Mac).
`(none)` means CPU only.

**2. Rehearsal** (recommended, same command as the final): download the organisers' mock exam
https://warsawmodeltrainers.dev/exams/history-2023-mock-v1.zip and run
```bash
uv run python scripts/egzamin_final.py --zip history-2023-mock-v1.zip
```
It must end with `SUBMIT:` and two paths. On our machine (RTX 5090): 51 s our setup + 29 s bare base.

**3. Final (before coding stops at 11:00):** open https://warsawmodeltrainers.dev/submissions.html?exam=final.
In "Get final exam questions" enter the team code and the repository link
(`https://github.com/CyprienFOURCROY/PolishMaturaHackathon/tree/bartek/maturaai-bartosz`), tick the confirmation
(from then on no work on any of the team's projects) and download the ZIP. The link is valid for 5 minutes (the button
renews it); on a remote machine copy the link and run `curl -L -o final.zip '<link>'`. Then:
```bash
uv run python scripts/egzamin_final.py --zip final.zip
```
It must end with `SUBMIT:` and two paths. Do not set `MATURA_TEMPERATURA`.

**4. Submit** (same page, "Submit your run"): team code, project name, and in "Models used" one row per model:

| model name or link | quantization |
|---|---|
| `unsloth/Qwen3.5-4B-GGUF` | `UD-IQ3_XXS` |
| `unsloth/Qwen3.5-2B-GGUF` | `Q4_K_M + mmproj F16` |
| `LiquidAI/LFM2-2.6B-GGUF` | `Q4_K_M` |
| `Helsinki-NLP/opus-mt-en-zlw` | `None` |

Category for this project: **Biggest improvement** (a category can be used by only one project of the team).
Answers JSON: `wyniki-final/zestaw/answers.json`; Most capable base model: the Qwen3.5-4B row; Base model answers JSON:
`wyniki-final/baza/answers.json` (improvement is measured against it). After "Upload answers" the page must show
"Submission received ... Base model answers also received. Receipt: ..."; keep the receipt.

**Tested** from a fresh clone with empty caches, the official llama.cpp release (b11205, CUDA 12.8), models
downloaded by `przygotuj.py`, the mock ZIP from the organisers, and **no internet** during the exam (Linux, RTX 5090):
68 s our setup + 27 s bare base, both `answers.json` valid. The same run on CPU only (16 threads) took about 13.5 min,
so without a GPU use a cloud GPU machine (Nebius, Labqoat/app.forgehand.app); the organisers allow it, the ban is on
external AI APIs, not on where the model runs. A laptop with an NVIDIA GPU or Apple Silicon (16 GB+) should take a
few minutes (not measured by us).

## Form fields
| role | model file | HF repo (commit) | bytes |
|---|---|---|---|
| answers short tasks (base model) | Qwen3.5-4B-UD-IQ3_XXS.gguf | unsloth/Qwen3.5-4B-GGUF (e87f176479) | 1,949,047,968 |
| describes illustrations | Qwen3.5-2B-Q4_K_M.gguf + mmproj-F16.gguf | unsloth/Qwen3.5-2B-GGUF (f6d5376be1) | 1,949,063,104 |
| writes the essay (English) | LFM2-2.6B-Q4_K_M.gguf | LiquidAI/LFM2-2.6B-GGUF (a759abdc59) | 1,563,668,704 |
| translates the essay EN→PL | opus-mt-en-zlw (MarianMT) | Helsinki-NLP/opus-mt-en-zlw (92ffd5aa93) | 300,812,519 |
| **total** | | | **5,762,592,295** (limit 8.8 GB) |

Improvement: yes; strongest bare base = Qwen3.5-4B UD-IQ3_XXS (the other models alone score lower).
"Small but crazy" track: largest model 1,949,063,104 B (the describer with its projector).

## What the setup does
- **Short tasks** (`--krotkie goly_vlm_kb_z`): Qwen3.5-4B UD-IQ3_XXS with (1) a text description of every
  illustration made live by Qwen3.5-2B with vision, (2) the top-2 entries of our knowledge base (BM25 over
  `data/wiedza/hasla_claude.jsonl`, 354 entries written with Claude at build time, never used during the exam),
  (3) closed tasks in a strict "ODPOWIEDŹ: ..." format with a majority vote over 3 samples.
- **Essay**: picks the topic by a rule (non-Polish-history topics first, then knowledge coverage), LFM2-2.6B writes
  it in English (≥ 300 words, retried if shorter), MarianMT translates it to Polish.
- Everything runs offline through `llama-server`; the harness code is `matura/egzamin.py`.

## Results (our judge `claude:opus` following the CKE rules; organisers' grading where stated)
| measurement | our setup v2 | bare base | improvement |
|---|---|---|---|
| mock exam 2023 (organisers' grading) | **40.00%** | 16.67% | **+23.3 p.p.** |
| forecast: CKE June 2024 + June 2025 sheets, untouched until the end, one run | 42.5% | 13.3% | +29.2 p.p. |

What did not work (measured, rejected): our own quantization and LoRA on the 3-bit base, structured essay variants,
another EN→PL translator, fine-tuning small students on teacher essays without knowledge (they learn the form but
invent facts; with knowledge-base entries in the prompt they improve but not yet above our current essay).

## Layout
`matura/` harness and experiments · `scripts/egzamin_final.py` final exam in one command · `scripts/przygotuj.py`
setup · `scripts/pobierz_modele.py` exact model downloads · `data/wiedza/`, `data/karty/` knowledge base ·
`tests/` (`uv run --group dev pytest -q`). Models go to `data/modele/` and `data/hf/` (not in git).
