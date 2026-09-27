Made during the Warsaw Model Trainers hackathon, Kolektyw3, 25–27.09.2026

# SOURCE: MaturaAI (Warsaw Model Trainers hackathon, 25-27.09.2026)

All sources of this folder: models, tools and data. No CKE exam sheets and no model weights are stored here; scripts
download them.

## Models (downloaded by `scripts/pobierz_modele.py`, exact commits)
| role | HF repo (commit) | file | licence | bytes |
|---|---|---|---|---|
| base model, short tasks | unsloth/Qwen3.5-4B-GGUF (e87f176479), from Qwen/Qwen3.5-4B | Qwen3.5-4B-UD-IQ3_XXS.gguf | Apache-2.0 | 1,949,047,968 |
| illustration describer | unsloth/Qwen3.5-2B-GGUF (f6d5376be1), from Qwen/Qwen3.5-2B | Qwen3.5-2B-Q4_K_M.gguf + mmproj-F16.gguf | Apache-2.0 | 1,280,835,840 + 668,227,264 |
| essay writer (English) | LiquidAI/LFM2-2.6B-GGUF (a759abdc59) | LFM2-2.6B-Q4_K_M.gguf | LFM Open License v1.0 | 1,563,668,704 |
| essay translator EN→PL | Helsinki-NLP/opus-mt-en-zlw (92ffd5aa93), prefix `>>pol<<` | whole folder (MarianMT) | Apache-2.0 | 300,812,519 |

The weights are used as published (no fine-tuning in the final setup).

## Tools
- llama.cpp `llama-server` (MIT), any recent release; we used b11190 (commit fcc891545) with CUDA.
- Python via `uv` (`pyproject.toml`, `uv.lock`): transformers, torch, bm25s, huggingface-hub and others (licences in the packages).

## Data
- **Knowledge base (RAG, in this folder):** `data/wiedza/hasla_claude.jsonl` (354 entries over the 50 sections of the
  history curriculum; written with Claude Opus at build time on 26.09.2026, not used during the exam),
  `data/karty/kanon.jsonl` (3,098 fact cards), `data/karty/slownik_kanon.jsonl` (glossary),
  `data/wiedza/akapity_esej_claude.jsonl` (essay paragraph bank). A 15% sample of the entries was checked by an
  independent model (GPT-6 "Astra"): 91.2% of 3,227 claims correct, 1.8% wrong, 7.0% uncertain.
- **CKE exam sheets** (history, extended level, 2015-2025) and the CKE marking rules were used for development and
  measurement only (not in this folder): cke.gov.pl. The 2026 exam was never used.
- **Organisers' mock exam** (CKE May 2023): https://warsawmodeltrainers.dev/exams/history-2023-mock-v1.zip.
- Claude was used only while building (writing knowledge, judging our measurements), never during the exam.
