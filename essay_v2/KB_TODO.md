# Knowledge base: what the essay needs from it, and what is missing

For the agent/person taking over `kanon.jsonl`. Self-contained; read this file only. Written 2026-09-27 03:40 on
branch `krzysztof-public`, based on ~100 generated-and-judged essays.

## 0. The headline finding (why this file matters more than the model)

The essay module works like the exam: the CKE thesis exists first, then facts are retrieved from the KB, then an
essay is written **using only those facts**. We measured the ceiling of that pipeline with GPT-6 Sol as the writer
(a far stronger writer than anything we can run on exam day):

| condition | n | mean score /15 (Sol judge, CKE rubric) |
|---|---|---|
| thesis written *after* seeing the facts (unrealistic) | 20 | 11.05 |
| thesis first, facts retrieved from kanon by relevance (realistic) | 30 | **6.53** |
| — retrieval found a good fit (4 of 30) | 4 | 11.75 |
| — weak fit (26 of 30) | 26 | 5.73 |

**Same writer, same style. The score is set by whether the KB contains facts that address the thesis.** No model
work moves this ceiling; only KB depth and retrieval do. This is the single highest-leverage task on the essay.

Reproduce: `python3 essay_v2/tools/gen_essays.py <seed> 30 --tag X && python3 essay_v2/tools/judge_sol.py out/sol/essays_X.jsonl`
(needs `~/.forgehand.env`, `out/kanon_clean.jsonl`, `out/theses.jsonl`; ~$1 per 30 essays). Look at the `fit=` column.

## 1. Where the KB lives and how the essay reads it

- Source: `origin/bartek:maturaai-bartosz/knowledge_base/kanon.jsonl` — 3,098 rows, Claude-written per curriculum
  section from the CKE Informator, **unverified**. Row fields: `fakt` (one sentence), `data`, `postac`, `termin`,
  `aspekt`, `tytul`, `dzial` (section).
- Retrieval (prototype, `essay_v2/tools/gen_essays.py::sample_facts`): for each of the topic's three CKE aspects,
  take all facts whose `aspekt` maps to that CKE aspect (map below), score by word-stem overlap with the thesis,
  bonus for the thesis's own section, top 6, near-duplicates removed. Exam-day retrieval will be the same idea.
- So a fact is only usable if (a) its `aspekt` maps to the aspect asked, and (b) its text shares vocabulary with the
  thesis (names, events, terms). Facts phrased generically ("władcy dbali o rozwój") are never retrieved.

## 2. Scope: what topics look like (so you know what to cover)

- Only the `aspekty` formula matters: "<TEZA>. Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając
  w swojej argumentacji aspekty: X, Y i Z." Every real formula-2023 sheet (6 so far) has at least one such topic;
  the harness always picks it. Ignore "trzech wybranych władców/wydarzeń".
- Real aspect names: **polityczny, społeczno-gospodarczy, kulturowy, militarny, ustrojowy, ekonomiczny, społeczny,
  gospodarczy, dyplomatyczny, polityczno-ustrojowy.** Map used in code (CKE name → kanon `aspekt` tags):
  `społeczno-gospodarczy`→społeczny+gospodarczy · `ekonomiczny`→gospodarczy · `kulturowy`→kulturowy+religijny ·
  `dyplomatyczny`→międzynarodowy · `polityczno-ustrojowy`→polityczny+ustrojowy · others identity.
- Real theses are one blunt, contestable sentence about a *specific* topic: "Klęska Polski w 1939 roku była
  nieunikniona", "Unia w Krewie okazała się bardziej korzystna dla Litwy niż dla Polski", "O upadku Rzeczypospolitej
  zadecydowały w głównej mierze czynniki wewnętrzne". 18 real ones: `essay_v2/topics_real.json` (held out — do not
  write KB rows *from* them, but they show the granularity: a ruler, a treaty, a war, a period of ~30 years).
- 372 synthetic theses in the same style, per section: `out/theses.jsonl` (regenerate with `gen_theses.py`).
  A good KB lets ~80% of them be argued in all three aspects; today it is ~15%.

## 3. Work items, in order of value

### 3.1 Depth per (topic, aspect) — the big one
For each curriculum section, for each *specific topic* a thesis could be about (each ruler, war, treaty, reform,
movement in the Informator requirements), the KB needs **≥ 4 usable facts per CKE aspect**, each:
- one sentence, concrete (who / what / when / consequence), with `data` where applicable;
- containing the topic's proper names and key terms in the text (so relevance retrieval finds it);
- tagged with the right `aspekt` for what the sentence is *about* (a battle's economic consequence is gospodarczy).
Example of a gap found: thesis "Rządy Kazimierza Wielkiego bardziej wzmocniły państwo niż społeczeństwo" —
kanon has ~6 Kazimierz facts, 1 of them społeczny. Thesis "Powstanie Chmielnickiego było najpoważniejszym…" — 2 facts.
Suggested method: per section, list the Informator's requirement bullets → for each, write facts per aspect.

### 3.2 Aspect tagging errors
- "Kultura i nauka polska XIX w.": społeczny/gospodarczy rows are Skłodowska at the Sorbonne, Domeyko in Chile.
  Judges: "says nothing about Polish society or economy". Needs industrialisation, urbanisation, peasantry, class.
- "Przemiany cywilizacyjne": Sputnik/Gagarin/Apollo tagged `międzynarodowy`; judged as not supporting an
  international-relations argument.
- Check per (section, aspect): "could a paragraph about this aspect be written from these facts?"

### 3.3 Hygiene (mechanical)
- Section names: 55 distinct `dzial` for 50 sections ("...XX wieku" vs "...XX w", "XVI i XVII" vs "XVI–XVII").
  Code normalises; fix at source.
- Duplicates: 4 exact, many paraphrase pairs across sections (Skłodowska/rad ×3, Statuty Kazimierza ×2,
  unia horodelska ×2, Pomorze Gdańskie/zboże ×2). Code dedupes at retrieval; fix at source.
- Confirmed wrong (removed in `out/kanon_clean.jsonl`, fix at source): Skłodowska "w 1906 roku … jako profesor Sorbony"
  (lectured 1906, chair 1908); Kierbedź bridge "stalowy" (iron).

### 3.4 Verification (exam-day risk only)
- ~2 factual errors caught in ~700 fact uses; only the Sol judge noticed, Opus/Fable/the official mock grader
  deducted nothing. Expected cost ≈ 0 to −1 per essay. Do 3.1 first; verify dated Polish-history facts as you go.
- Training the small model on kanon is fine regardless: it learns to use supplied facts, not to know history.

## 4. Definition of done
Run the reproduction command on 30 theses: `fit=dobre` in ≥ 70% and mean judge score ≥ 10. That is the whole target.
