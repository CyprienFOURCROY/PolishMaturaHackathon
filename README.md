# PolishMaturaHackathon

Answer the Polish history matura (May 2023, extended level) with small models.

```
question + source + image
  → OCR → image description → glossary + translate PL→EN
  → knowledge-base keyword search → LLM answer (EN) → translate EN→PL → final answer
```

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p models/ocr models/vlm models/translator models/translator_back models/llm cache outputs

hf auth login                              # Gemma is gated: first accept the license at
                                           # https://huggingface.co/google/gemma-3-270m-it
python3 scripts/download_models.py         # ~3 GB into models/  (EasyOCR downloads itself on first OCR run)

cp .example.env .env                       # put your OpenAI key in .env (only needed to grade open questions)
```

## Use

```bash
python3 serve.py                           # open http://127.0.0.1:8765
```

Pick a question in the viewer to see its pipeline as a flow of arrows.

| Want to… | Do |
|---|---|
| Run one stage | **▶ Run** on its card |
| Run one question | **▶ Run this question** (skips stages already done) |
| Redo work | **↻ Re-run answer** (keeps OCR/VLM/translation) or **↻ Re-run everything**; **🗑 Clear** deletes a saved result |
| Run many | Tick questions in the sidebar (**Select all**, **Not done**), then **▶ Run N selected**; **■ Stop** ends it |
| See the score | **📊 Final score** |
| Grade | **Grade** block on the last card, or **▶ Grade N…** in the score view |
| Hand in results at any time | **⬇ Export answers** → `outputs/full_answers.json` |
| Hide the sidebar | **«** or key **b** |

Without the viewer:

```bash
python3 run_pipeline.py --only 1,2.1,3     # run selected questions (omit --only for all)
python3 export_answers.py                  # write outputs/full_answers.json from what is cached
python3 evaluate.py --dry-run              # grading preview, no API call
python3 evaluate.py --only 24              # grade selected ids (omit --only for everything answered)
```

Restart `serve.py` after changing Python code.
