# Gemma history pipeline

Practice platform result: 42/60 (70%), 27 September 2026. This is calibration on the supplied history-2023-mock-v1 exam, not an unseen-test result

Model: bartowski/google_gemma-3-12b-it-GGUF, google_gemma-3-12b-it-Q4_K_S.gguf plus mmproj-google_gemma-3-12b-it-f16.gguf, approximately 7.8 GB combined. No fine-tuning. Images are passed directly to Gemma. Local lexical/period retrieval is used only for essay IDs (default 26). Short questions receive no retrieved context

## Data

`timeline/events.jsonl`: 1,244 entries from https://pl.wikipedia.org/wiki/Kalendarium_historii_Polski, revision 80582058. Fields include id, date_text, event_text, section_path, source_text, source_url, revision_id, citation_refs. One malformed empty-date entry is retained for provenance and excluded by the loader. Parsing was validated; historical claims were not independently verified. Wikipedia text remains subject to its applicable attribution/share-alike terms; retain source URLs and revision attribution when redistributing

`historical_context.jsonl`: seven source-based summaries with source URLs. This is a small supplement, not comprehensive world-history coverage

## Run

Use the existing CUDA llama.cpp server with alias gemma-matura on port 8080, Q4_K_S weights, F16 vision projector, context 16384, one slot and --jinja. Model files and compiled runtime are not included

```bash
python3 run_exam_gemma_auto.py --exam /workspace/matura-qwen/exam/exam.json --out runs/q4ks-auto --timeline timeline/events.jsonl --context-notes historical_context.jsonl --temperature 0 --max-tokens 768 --essay-tokens 3072
```

Python 3.10+ standard library client. The exam JSON and images must keep their relative layout. Change --essay-ids if essay IDs differ. Use a new output directory for changed configurations

## Evidence and limitations

Saved practice outputs are in results/practice-q4ks-70. Score 42/60 was observed in the platform UI; no per-item scores were available. The historical run config uses the server alias and contains inherited metadata: it is not an authoritative model-weight hash manifest. Exact weight hashes were not recorded in these supplied artifacts

Retrieval uses truncated lexical terms and date ranges and can return irrelevant facts. Added retrieval for short questions reduced an earlier IQ3 run from 36/60 to 35/60 and is excluded here. Two-stage essay planning and image-crop experiments are also excluded. No automatic answer selection from multiple runs
