# PolishMaturaHackathon

The objective is to perform well with a small model on the Polish history Matura.

## Tasks

- Harness: pipeline for the essay — Krzysztof (explored a deterministic approach, see below)
- Harness (general) — Olga
- Selection of the smallest model — Bartek
- Knowledge base (names, dates, to factcheck) — Krzysztof / Cyprian
- RAG layer connecting a harness to the KB — Cyprian
- Synthetic data for the Matura exam — Bartek (small set), Krzysztof (bigger set)

## What was tried on the essay harness (Krzysztof)

An abandoned first attempt, summarised so nobody repeats it. The full write-up is
kept locally and is not needed elsewhere.

**1. Deterministic (template-only) harness — does not work.**
Python assembled the whole essay from KB rows, with no model in the content path.
It reliably produced a compliant five-paragraph, ~430-word essay: correct length,
all required elements, no invented facts, and a clean run of our own hard checks.
It still scored around 5/15. A template cannot tell whether its evidence supports
the thesis it is asserting, so every paragraph reached the same conclusion
regardless of its facts. Polish-language reliability was never the bottleneck.

**2. Small models are not reliable generators of Polish at this size, yet.**
`bielik-1.5b-v3` did not use the facts it was given: across ~90 generations the
dominant failure was dropping injected dates, and asked to do abstract analysis
it returned an empty response (confirmed 4/4). A single A/B against
`qwen2.5:1.5b` on element extraction (n=4) was similarly unreliable for both.
This is the expected motivation for fine-tuning rather than a reason to drop the
small-model target.

**3. Coding agents are not usable as graders or as a fact source.**
Handing a coding agent the job of producing the KB produced 209 rows that were
well-formed, on-topic and plausible — and wrong. Every row self-declared
`status: "do-weryfikacji"` and none was checked. One row cited the Paris Panthéon
(begun c. 1757) as an achievement of *medieval* European engineering. Dates,
names and attributions were invented fluently and consistently, which is exactly
what makes them dangerous downstream: automated checks pass, because the checks
compare the essay to the same unverified source.

**4. Optimising a proxy instead of the goal.**
Most of the lost time went into making measurable things pass — word counts,
paragraph counts, diacritics, "all checks green" — while the actual
argument quality did not move. Worth stating plainly for the next person: a green
check is not a good essay, and the rubric is the only scoreboard that counts.

## Takeaway

The essay harness needs a **fine-tuned small model plus a real KB**. Templates can
supply structure and guarantee no invented facts, but they cannot supply
judgement. Treat generated data and self-graded output as unverified until a
human has checked the facts against sources.
