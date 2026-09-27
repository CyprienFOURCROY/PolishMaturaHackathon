# Gemma history exam pipeline

Olga Ivanova · Warsaw Model Trainers · September 2026

**Selected configuration: Gemma 3 12B IT Q4_K_S, direct image input, Polish answers, local essay retrieval. Practice platform score: 42/60 (70%).** No fine-tuning. Only one supplied calibration exam was used; this is not an unseen-test result

## Start here

- FINAL_RUN.md: existing GPU session, final command, resume and output validation
- SETUP.md: model files, runtime prerequisites and installation commands
- EXPERIMENTS.md: scores, rejected variants, evidence and remaining uncertainties
- experiments/evidence/MANIFEST.json: hashes of preserved experimental result files

## Selected inference path

Short tasks receive their original text and images directly. Essays receive locally retrieved timeline entries and seven supplementary context notes. The model chooses its own essay topic. No short-task RAG, separate image describer, translation model, voting or selection of answers from multiple runs is used

The original scored client is run_exam_gemma_auto.py. The final client run_exam_gemma_final.py preserves its practice-exam message content while adding automatic essay detection and adaptation to explicitly different word limits. final_run.py adds preflight checks, logging, validation and packaging. The final wrapper itself has not received a separate GPU score

Generation: temperature 0, seed 42, top_p 0.95, top_k 64, min_p 0, repeat penalty 1, 768 output tokens for short questions and 3072 for essays. llama.cpp context 16384, one slot, Jinja template, local endpoint on port 8080

## Run from this directory

    python3 final_run.py --exam /absolute/path/to/exam.json --out runs/final-exam --check-only
    python3 -u final_run.py --exam /absolute/path/to/exam.json --out runs/final-exam

Requires Python 3.10+ and the already running local server. Preserve the exam's relative image layout. Repeat the exact command to resume. Use a new output directory for different inputs or configuration. Ambiguous essay detection requires explicit --essay-ids ID or --no-essays

After PASS upload runs/final-exam/submission.json. The audit archive is runs/final-exam.tar.gz. PASS checks completeness, IDs, response termination and parsed word bounds, not historical accuracy. No automatic submission occurs

## Weights and hardware

HF repository: https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF

| Component | File | Bytes |
|---|---|---:|
| Language model | google_gemma-3-12b-it-Q4_K_S.gguf | 6,935,130,144 |
| Vision projector | mmproj-google_gemma-3-12b-it-f16.gguf | 854,200,224 |
| Total | | 7,789,330,368 |

The server stores the projector under mmproj-gemma-3-12b-it-f16.gguf. Runtime: locally compiled llama.cpp reporting commit 81bc6b8, NVIDIA L40S, Ubuntu 22.04. Exact full runtime commit and weight SHA-256 values were not captured in the original scored artifacts. Inherited run-config metadata is not an authoritative weight manifest. Disk size is distinct from GPU memory consumption

## Data provenance

The 1,244 timeline records originate from https://pl.wikipedia.org/wiki/Kalendarium_historii_Polski, revision 80582058. Fields include id, date_text, event_text, section_path, source_text, source_url, revision_id and citation_refs. One malformed empty-date record is preserved and excluded by the loader. Extraction fidelity was checked; historical truth was not independently verified. The source itself carries a quality warning. Retain source/revision attribution and applicable Wikipedia attribution/share-alike terms when redistributing

historical_context.jsonl contains seven source-based notes with source URLs, not comprehensive world-history coverage. timeline/manifest.json also records hashes of original acquisition files not bundled in this repository. Synthetic hasla_claude.jsonl was tested separately and is not used in the selected pipeline

## Results and limitations

results/practice-q4ks-70 contains the scored run's answers, records, config and retrieval audit. The 42/60 score was observed in the organisers' UI; no per-question breakdown was available. Sum of recorded answer times: about 105 seconds. These per-answer timings are not a complete deployment benchmark

Retrieval uses five-character lexical prefixes and date ranges. It can omit relevant events, confuse related words and introduce irrelevant context. Generated answers can hallucinate dates, people, visual details and causal links. Word counts are whitespace-based approximations. Model/runtime identity is not attested by the health endpoint. Resume assumes the operator retains the same loaded model

This branch includes experimental evidence for transparency, but the runner does not read it or past answers. Repeated tuning on one calibration exam creates overfitting risk. No historical answer corrections have been manually inserted into submission files
