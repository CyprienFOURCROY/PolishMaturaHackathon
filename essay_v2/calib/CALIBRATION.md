# Judge calibration (2026-09-27, ~01:30–02:40)

All essays written by GPT-6 Sol from `kanon.jsonl` facts. Judges see only topic + essay (blind), grade with
`rubryka.md` (CKE criteria A 0–12, B 0–3). "Astra" = the organisers' grader on the mock exam (essay-only
`answers.json`, item 26, everything else blank; total /60 = essay /15).

## Same six essays, three judges (`key.json`, `opus.json`, `fable.json`)

| essay | prompt variant | Sol (gpt-6-sol, low reasoning) | Opus | Fable |
|---|---|---|---|---|
| E1 Komuniści 1944–48 | v3 baseline | 7 | 11 | 11 |
| E6 Komuniści 1944–48 | v4 pogłębiona | 12 | 11 | 12 |
| E2 Kultura polska XIX | v3 baseline | 9 (−1 fact) | 10 | 10 |
| E3 Kultura polska XIX | v5 argument-first | 11 | 10 | 11 |
| E5 Renesans | v2 bland thesis | 8 | 10 | 11 |
| E4 Okupacja sowiecka | v1 complex, stance "zbyt kategoryczna" | 12 | 13 (one bogata) | 13 (one bogata) |

## One essay, four judges (`essay_2023-2_good.md`, mock topic 2023-2, hand-picked facts)

| Sol | Opus | Fable | **Astra (official mock grader)** |
|---|---|---|---|
| 12 | 11 (B=2: paragraphs swappable, choppy sentences) | 12 | **14** |

## What we take from it (n is tiny; directions only)

- Opus and Fable agree within 1 point; Sol is harsher and more spread. The 7-vs-12 gaps Sol showed between prompt
  variants mostly vanish under Fable/Opus (11 vs 11–12). Prompt variants matter less than Sol suggested.
- The official grader (n=1) was **2–3 points more lenient** than all three of ours. Our 11–12 ≈ 14 official.
  The writing target is therefore already near the ceiling; the work is in making the 1.5B reproduce it.
- Sol was the only judge that caught a real factual error (a wrong `kanon` row). Fable-class judges deducted
  nothing for facts on any essay.
- Recurring Fable/Opus complaint: the essay never weighs the thesis's evaluative word ("przede wszystkim",
  "przełom"); agreement is "declared, not earned". The only 13s went to the nuanced-stance essay.
- Opus docked B for parallel, swappable paragraphs and short choppy sentences — the plain register has a cost with
  at least one Fable-class judge.

Do not overfit to the 2023-2 mock topic: it is in the held-out set for model evaluation.
