# Experiments and evidence

Olga Ivanova · 27 September 2026

Scores below are organiser practice scores reported during the session, not independent regrading. No per-item score breakdown was available. Manual inspection is described separately and is not a score. Experiments share one calibration exam; they do not establish generalisation

| Variant | Available outcome | Decision |
|---|---|---|
| Gemma 3 12B NF4, direct images | 37 answers, roughly 8m40s; 7.8385 GB checkpoint; no validated organiser score retained here | Superseded by GGUF runtime |
| Gemma 3 12B IQ3_XXS | Text+projector 5,638,930,368 bytes; full-run outputs preserved | Used for subsequent context experiments |
| IQ3 fixed essay context/rules | 33/60, 55% reported | Superseded |
| IQ3 automatic essay retrieval | 36/60, 60% reported | Better calibration score |
| IQ3 retrieval also on short tasks | 35/60, 58.33% reported; irrelevant contexts found | Rejected |
| IQ3 two-stage essay planning | Free-form plan copied into answer, unsupported causalities, short essay in inspected run; no official score | Rejected |
| Q4_K_S + original automatic essay retrieval | 42/60, 70%; 37 answers, no truncation, about 105s sum of generation times | Selected |
| Q4_K_S extended source instructions | 36 short answers, about 81s; mixed manual improvements/regressions; no official score | Rejected |
| Q4_K_S overlapping image crops | IDs 5.3, 8, 13.1; key mistakes persisted; no official score | Rejected |
| Q4_K_S observe then answer | Same three IDs; about 32s total; observations omitted key visual details and final map choice remained wrong | Rejected |
| Q4_K_S synthetic hasla KB | 354-entry KB inspected; client prepared; no usable completed run/result supplied | Inconclusive, not selected |
| Q4_K_S combined fast variant | 37 answers, structural PASS, 143.83s; all answers changed; 3529 words vs 2626 baseline | Not selected; visible regressions, no official score |
| Final packaging wrapper | Message-content equality for 37 practice items; mocked resume; saved baseline passes validation and packaging | Operational hardening only; no new score claimed |

## What the manual diagnostics established

Image crops did not make the model identify the relevant combined heraldic symbols. Two-stage observation read some map labels but still produced an incorrect source match, so resolution alone did not explain the failures. Extended short-answer instructions improved some individual answers and worsened others

The combined fast variant changed input order, short-answer instructions and essay retrieval simultaneously. Attribution to one change is therefore impossible. It recovered a missing helot reference in 2.2 but introduced regressions in 4.2, 5.2, 25.1 and 25.2. It also generated more text despite the concision instruction. It was not promoted to the selected pipeline

An English-translation experiment was interrupted by workspace access problems. No valid comparison was established. Earlier Gemma/Qwen small-model outputs were discussed in chat, but complete comparable run artifacts are not included here and no comparative scores are claimed from them

## Preserved evidence

experiments/evidence contains the supplied NF4 and IQ3 full runs, IQ3 comparison archives, the Q4 70% run archive, source-rules results, crop answers (answers-4.json), observation results and the combined fast full run. MANIFEST.json records byte sizes and SHA-256. These are result artifacts, not necessarily complete reproducible environments. Some contain nested run directories or historical metadata that may be stale

Experimental clients are included under experiments for inspection. They are not selected by final_run.py. Exact script-to-run hashes should be checked in each run config when reproducing an experiment; not every historical variant or failed attempt was preserved

Not included: model weights, build binaries, full raw Wikipedia acquisition transcript, platform receipts/per-item scores, completed translation/hasla results or a final-exam result. No completeness claim is made for every command attempted during the hackathon
