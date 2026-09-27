# Essay module — results as of Sunday 27.09.2026, 09:40

All scores are from the blind GPT-6 Sol judge with the CKE rubric (`essay_v2/tools/judge_sol.py`). Calibration
against the organisers' mock grader on one essay: Sol 12, Opus 11, Fable 12, **official 14** — Sol runs ~2–3 points
harsher than the official grader (n=1). Held-out topics = the 12 real `aspekty` CKE topics 2022–2026, never used to
build data or prompts.

## What the ceiling is (writer = GPT-6 Sol, facts from kanon)

| condition | n | mean /15 |
|---|---|---|
| thesis written after seeing the facts (unrealistic) | 20 | 11.05 |
| thesis first, random facts from the section | 30 | 6.53 |
| thesis first, relevance retrieval (harness prototype) | 30 | 8.17 |

The score is set by whether the KB contains facts that address the thesis. See `KB_TODO.md`.

## What the small models do inside the harness (held-out topics, n=4 candidates per paragraph)

| model | topics | mean /15 | fallback paragraphs | fact-error essays |
|---|---|---|---|---|
| bare Bielik-1.5B, no harness (bare prompt) | — | ~0–1 (Bartek's measurement; our mock file `answers_bare_bielik.json` pending upload) | — | — |
| Bielik-1.5B base + harness (first smoke, older retrieval) | 5 | 2.6 | 6/15 | 5/5 |
| Bielik-1.5B base + harness (current) | 4 | 2.75 | 0/12 | 4/4 |
| **Bielik-1.5B SFT r1** + harness (current) | 4 | 2.75 | 0/12 | 4/4 |
| Bielik-1.5B SFT r1 + harness, all 12 topics | 12 | 2.17 (min 1, max 7) | 0/36 | 12/12 |
| **no model**: retrieval + deterministic template only, all 12 topics | 12 | 2.92 (min 1, max 5) | 36/36 | 2/12 |
| Qwen3.5-4B IQ3 + harness (1 topic; every candidate failed the gates → template) | 1 | 4 | 3/3 | 0/1 |

Reading: template-only, base Bielik, fine-tuned Bielik and Bartek's LFM2 essay all land in the same ~3–5/15 band (official scale ≈ +2). The KB is the ceiling. The fine-tune fixed format (no markdown, no prompt echo, no empty outputs, shorter paragraphs, every
paragraph passes the gates) but did not add points. Judge complaints are the same for both: facts are listed, not
connected to the thesis; several topics get facts from the right period but the wrong sub-topic (aftermath instead of
causes for "Klęska Polski w 1939"). Fact-error deductions come from the model re-attaching dates/names inside a
paragraph (e.g. moving 1940 to 1939 because 1939 is in the thesis) — the gates block foreign years/names but not
re-combinations of allowed ones.

Under the official grader's leniency these ≈ 4–6/15, i.e. the same band as Bartek's e5 pipeline on the 4B (3–5),
with the difference that this pipeline never invents a date, never emits markdown, always produces a complete
5-paragraph essay in 1–3 minutes, and degrades gracefully (deterministic fallback).

## SFT details
- Data: 123 Sol essays (judge ≥ 10/15, no fact error) → 585 chat examples (full essay / one paragraph / intro+concl.)
- r1: full FT, lr 1e-5, 2 epochs, 74 steps, eval loss 1.45. r2: lr 5e-5, 3 epochs, eval loss 1.56 (worse; not used).
- Artifacts: `out/models/bielik-essay-sft.Q8_0.gguf` (1.70 GB, sha256 0381cce0…), Modelfile with ChatML template.

## What would move the number (in order)
1. KB depth per (topic, aspect) and aspect tagging — `KB_TODO.md`. Nothing else raises the ceiling.
2. Retrieval: distinguish causes/aftermath; the section lock still picks adjacent sections for 3/12 real theses.
3. A stronger writer under the same gates (Qwen3.5-4B test pending), or more/better SFT data with harder
   "connect fact to thesis" targets — but only after 1.
