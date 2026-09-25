# PolishMaturaHackathon
The objective is to perform well with a small model on the Polish history Matura


Tasks :

- Harness : Pipeline for the essay -> in progress by krzystof
- Harness (general) - Olga
- Selection of the smallest model -> in progress by Bartek
- Knowledge base (names, dates, to factcheck) - Krzysztof


- Synthetic data for Matura exam : -> in progress by Bartek (small set) and Krzysztof (bigger set)
- Finding a lightweight model for translating Polish to English (Krzysztof)
- Harness : Having a way to select complicated polish words and look for their definition in the dictionary 

- Finding a model for describing data -> in progres by cyprien


## Krzysztof

Essay harness + knowledge base + synthetic Matura data.

- Base model for now: Bielik-1.5B-v3.0-Instruct (Polish, Apache 2.0), 9B as fallback
- Dev set of 12 real CKE essay topics (2023-2026) committed on branch `krzysztof`, in `essay_harness/topics_real.json`
- Next: design the essay pipeline (topic -> plan -> draft -> fact-check) on top of the knowledge base
