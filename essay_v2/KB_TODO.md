# Knowledge base TODO — for whoever owns `kanon.jsonl`

State as of 2026-09-27 03:10. Source: `maturaai-bartosz/knowledge_base/kanon.jsonl` (3,098 facts, Claude-written per
curriculum section, unverified). The essay module retrieves facts from it by (section, aspect). Everything below
was found by generating and judging ~70 essays from it; nothing here is speculative.

## Scope decision (saves work)
- **Only the `aspekty` topic formula matters.** Every formula-2023 sheet (pokazowy 2022, próbna 12/2022, May 2023–2026)
  contains at least one topic of the form "...uwzględniając w swojej argumentacji aspekty: X, Y i Z". The harness always
  picks that topic. No need to model rulers/events/states ("trzy wybrane") in the KB.
- Aspect names that actually occur in real topics: **polityczny, społeczno-gospodarczy, kulturowy, militarny,
  ustrojowy, ekonomiczny, społeczny, gospodarczy, dyplomatyczny, polityczno-ustrojowy.**

## 1. Aspect vocabulary (highest value, mechanical)
- kanon uses: polityczny, międzynarodowy, kulturowy, społeczny, ustrojowy, militarny, gospodarczy, religijny.
- Needed: a mapping or extra tags so retrieval can serve the CKE names above. Concretely:
  `społeczno-gospodarczy` = społeczny ∪ gospodarczy; `ekonomiczny` = gospodarczy; `dyplomatyczny` ≈ międzynarodowy;
  `polityczno-ustrojowy` = polityczny ∪ ustrojowy. `religijny` maps to `kulturowy` when a topic asks for it.

## 2. Aspect tagging errors (content)
- "Kultura i nauka polska XIX w.": facts tagged społeczny/gospodarczy are Skłodowska at the Sorbonne and Domeyko in
  Chile. Two judges independently said the resulting socio-economic paragraph "says nothing about Polish society or
  economy". The section needs real socio-economic facts (industrialisation, urbanisation, peasantry, class).
- "Przemiany cywilizacyjne": space race (Sputnik, Gagarin, Apollo) tagged `międzynarodowy`; judged as not supporting
  an international-relations argument. Needs a second tag or a better one.
- General check: for each (section, aspect) ask "could a paragraph about this aspect be written from these facts?"

## 3. Coverage
- Some (section, aspect) pairs have < 4 facts (e.g. starożytny Wschód, barok, wczesnopiastowska/ustrojowy: 2 facts).
  A paragraph needs 4; target ≥ 6 per pair so sampling has slack.
- Section names are inconsistent: 55 distinct `dzial` values for 50 sections (e.g. "...XX wieku" vs "...XX w").
  Normalise before grouping.

## 4. Confirmed wrong rows (already removed in our cleaned copy; fix at source)
- "Maria Skłodowska-Curie została w 1906 roku pierwszą kobietą prowadzącą wykłady jako profesor paryskiej Sorbony" —
  lectured from 1906, professorial chair 1908.
- "Stanisław Kierbedź zaprojektował otwarty w 1864 roku stalowy most na Wiśle" — iron, not steel.
- 4 exact-duplicate rows.

## 5. Verification (exam-day risk, not training risk)
- Observed error rate ≈ 2 caught in ~600 fact uses (only the Sol judge caught them; Opus/Fable/the official mock grader
  deducted nothing). Expected cost per essay ≈ 0 to −1 point. Not fatal, but the only fix is a source check.
- Priority order: dated facts about Polish history (most likely in a real topic) → European early modern → the rest.
- Training the small model on kanon is fine regardless: it learns to use supplied facts, not to know history.
