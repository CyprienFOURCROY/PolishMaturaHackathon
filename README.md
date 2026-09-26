# PolishMaturaHackathon

Answer the Polish history matura (May 2023, extended level, 60 pts, target ≥ 35%) with the smallest models possible.
The LLM is Gemma 3 270M, which only reads English, so Polish is translated in and out.

```
question + source + image
  → OCR (EasyOCR) → image description (SmolVLM2) → glossary + translate PL→EN (Marian)
  → knowledge-base keyword search → Gemma 270M answer (EN) → translate EN→PL (Marian) → final answer
```

## Setup

Python 3.10+, from the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

hf auth login                              # Gemma is gated: first accept the license at
                                           # https://huggingface.co/google/gemma-3-270m-it
python3 scripts/download_models.py         # ~3 GB into models/  (EasyOCR downloads itself on first OCR run)

cp .example.env .env                       # then put your OpenAI key in .env; only needed to grade open questions
```

## Use

```bash
python3 serve.py                           # open http://127.0.0.1:8765
```

In the viewer, pick a question and see its pipeline as a flow of arrows (Polish input, OCR, image description,
glossary terms, English translation, knowledge-base cards, answers, grade).

| Want to… | Do |
|---|---|
| Run one stage | **▶ Run** on its card |
| Run one question | **▶ Run this question** (skips stages already done) |
| Redo work | **↻ Re-run answer** (fast, keeps OCR/VLM/translation) or **↻ Re-run everything**; **🗑 Clear** deletes a saved result |
| Run many | Tick questions in the sidebar (**Select all**, **Not done**), then **▶ Run N selected**; **■ Stop** ends it |
| See the score | **📊 Final score**: points out of 60 vs the 35% target, per question with the judge's reason |
| Grade | **Grade** block on the last card, or **▶ Grade N…** in the score view (open questions use the paid API, closed ones are free) |
| Hand in results at any time | **⬇ Export answers** → `outputs/full_answers.json` (all 37 ids, `""` if unanswered) |
| Hide the sidebar | **«** or key **b** |

Without the viewer:

```bash
python3 run_pipeline.py --only 1,2.1,3     # run selected questions (omit --only for all)
python3 export_answers.py                  # write outputs/full_answers.json from what is cached
python3 evaluate.py --dry-run              # grading preview against the official scheme, no API call
python3 evaluate.py --only 24              # grade selected ids; without --only, grades everything answered
```

Results are cached per stage in `cache/<stage>/`, so nothing is computed twice. Restart `serve.py` after changing Python code.

## Models

| Role | Model | Params | On disk | In memory |
|---|---|---|---|---|
| LLM | `google/gemma-3-270m-it` | 268 M | 549 MB (bf16) | ~1.07 GB (fp32) |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 507 M | 1.9 GB (fp32) | ~1.01 GB (bf16) |
| Translate PL→EN | `Helsinki-NLP/opus-mt-pl-en` | 77 M | 298 MB | ~0.31 GB |
| Translate EN→PL | `Helsinki-NLP/opus-mt-en-zlw` | 74 M | 287 MB | ~0.30 GB |
| OCR | EasyOCR `craft_mlt_25k` + `latin_g2` | ~25 M | 94 MB | ~0.1 GB |

Grading only (not in the pipeline): `gpt-6-astra` for open questions; closed questions use exact match with the official key (no model).
Which model is "heaviest" depends on the rule: by parameters it is the VLM, by memory it is Gemma, by disk it is the VLM.

## Layout

```
serve.py run_pipeline.py evaluate.py export_answers.py    entry points
full_exam_question.json   ImagesMatura/                   the exam
data/rules/               official 2023 marking scheme (PDF)
data/glossary/glossary.json   data/kb/*.jsonl             glossary terms, knowledge-base fact cards
prompts/                  prompt templates
models/<role>/            downloaded weights (git-ignored)   cache/  outputs/   (git-ignored)
src/matura/               config.py (which model per role), stages/, models/, glossary/, kb/, eval/, viewer/
```

To swap a model, edit `ROLES` in `src/matura/config.py`. To improve answers, add entries to the glossary and knowledge base.

## Status

Works: full pipeline, viewer, batch runs, grading, export. Not done: the essay (question 26, 15 pts), real glossary and
knowledge-base content (only 2 example entries each), and closed-question accuracy is poor so far (prompt work needed).
