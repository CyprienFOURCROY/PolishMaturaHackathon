# PolishMaturaHackathon — branch `krzysztof-public`

Essay module for the Polish history Matura (poziom rozszerzony), aimed at a small model. Team repo; this branch
holds Krzysztof's essay work. Bartek's full-exam pipeline and knowledge base are on `origin/bartek`.

## Goal

Given the exam's item 26 (three topics; we pick one), produce one CKE-compliant ~380-word Polish essay with a model
≤ 1.95 GB on disk (smaller than the team's base), consistently across topics we have never seen. Stance is fixed by
code ("Zgadzam się z tezą"). Facts come only from a local knowledge base; the model may not add any.

## Approach (current)

1. **Training data:** GPT-6 Sol writes structured essay records (topic, stance, per-aspect facts → paragraph, intro,
   conclusion) from `kanon.jsonl` facts, using only supplied facts (auto-checked). Records are structured so any
   code/model split (full essay, single paragraph, intro/conclusion) can be rendered from the same data.
2. **Fine-tune** Bielik-1.5B-v3 (Qwen2.5-1.5B with Polish pretraining) on those records. Not started yet.
3. **Harness:** code owns topic choice, structure, fact retrieval and checks; the model writes the prose.

## Established so far (see `essay_v2/calib/CALIBRATION.md`)

- Zero-shot, Bielik-1.5B uses injected facts (7/9 samples used all dates) but breaks format (markdown, lists,
  scaffold echo); Qwen2.5-1.5B's Polish is not exam-grade. Bielik is the model to fine-tune.
- Sol-written essays in the plain "argument-first" style (`gen_sol.py` variant v5) score 11–12/15 with Opus/Fable
  judges and **14/15 with the organisers' mock grader** (n=1). The target style is near ceiling; the open problem is
  the small model reproducing it.
- Sol as a judge is cheap (~$0.02/essay) and catches factual errors, but is harsher and noisier than Fable-class
  judges; use it for screening, not style decisions.
- Two confirmed wrong rows in `kanon.jsonl` and an aspect-vocabulary mismatch with CKE: `essay_v2/KB_TODO.md`.

## Layout

```
SOURCE.md                     required by the hackathon rules
essay_v2/
  rubryka.md                  CKE marking criteria (transcribed from the Informator) — the referee for everything
  topics_real.json            12 real CKE topics 2023–2026 — HELD OUT, never used for training data
  KNOWN_TRAPS.md              Ollama/text-handling traps that cost time; read before writing generation code
  KB_TODO.md                  what the knowledge base needs
  tools/gen_sol.py            generate structured essay records with Sol (prompt variants v3–v9)
  tools/judge_sol.py          blind CKE-rubric judge with Sol, cached
  tools/gen_fixed_topic.py    one essay for a fixed topic from hand-picked facts (calibration only)
  tools/build_viewer.py       single-file HTML viewer of all records + judge scores → out/viewer.html
  tools/probe_ollama.py       zero-shot paragraph probe for local Ollama models
  data/sol_batches/           the 51 judged Sol essays behind the calibration numbers
  calib/                      judge calibration inputs and outputs
out/                          gitignored working data (kanon copy, batches, viewer, mock package)
```

## Running

```zsh
python3 -m pip install openai
source ~/.forgehand.env          # FORGEHAND_TOKEN, FORGEHAND_TEAM_ID (never committed)
git show origin/bartek:maturaai-bartosz/knowledge_base/kanon.jsonl > out/kanon.jsonl
python3 essay_v2/tools/gen_sol.py 41 5 --variant v5        # 5 essays, ~$0.02 each
python3 essay_v2/tools/judge_sol.py out/sol/batch_41_v5.jsonl
python3 essay_v2/tools/build_viewer.py && open out/viewer.html
```

## Rules that shape the design

- Bare model = benchmark; the essay's bare score is ~0, so every essay point is "progress".
- "Mały, ale wariat": size = the largest model in the setup; a LoRA adapter does not count.
- No external APIs at exam time; local RAG allowed. Synthetic training data from closed LLMs is allowed.
- Tracks and FAQ: https://warsawmodeltrainers.dev/rules
