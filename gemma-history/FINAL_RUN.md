# Final Gemma run

Owner: Olga Ivanova

Selected: Gemma 3 12B IT Q4_K_S, direct images, original timeline retrieval for essays, Polish answers. Practice score: 42/60 (70%). Not a final-exam score. Experimental fast, tiles, observation and hasla variants are not selected.

## Run on the existing GPU session

Python 3.10+, standard library only. The existing server listens on localhost:8080. Do not start a second copy. From gemma-history:

    python3 final_run.py --exam /absolute/path/to/exam.json --out runs/final-exam --check-only
    python3 -u final_run.py --exam /absolute/path/to/exam.json --out runs/final-exam

Repeat the exact command to resume. Use a new output directory for a different exam. Automatic essay detection prints the IDs; override with --essay-ids ID or --no-essays after inspecting the task if necessary.

After PASS upload runs/final-exam/submission.json. Audit archive: runs/final-exam.tar.gz. PASS checks structure and parsed word limits, not historical correctness. Do not submit practice answers for a different exam.

## Server startup when needed

    bash serve_final.sh

Uses the existing compiled runtime and weights under /workspace/gemma-matura. Override with GEMMA_MODEL_ROOT. This repository does not include weights or CUDA libraries and is not a self-contained fresh-machine installation.

Expected weights from bartowski/google_gemma-3-12b-it-GGUF:
- google_gemma-3-12b-it-Q4_K_S.gguf: 6,935,130,144 bytes
- mmproj-google_gemma-3-12b-it-f16.gguf: 854,200,224 bytes, locally renamed mmproj-gemma-3-12b-it-f16.gguf

Total: 7,789,330,368 bytes (7.789 GB decimal). Local llama.cpp reports commit 81bc6b8. GPU: L40S. Sizes do not establish cryptographic identity. Health confirms readiness, not loaded weights; inspect server startup/logs.

Requires checked-in timeline/events.jsonl and historical_context.jsonl. See README.md for data provenance and limitations.

## Checks before freeze

Original user payload content for all 37 practice items equals the 70% baseline. Essay ID changes, word bounds, mocked resume and packaging were checked. Existing 70% results passed packaging with byte-identical submission; empty answers were rejected. No new GPU score claimed for the wrapper.
