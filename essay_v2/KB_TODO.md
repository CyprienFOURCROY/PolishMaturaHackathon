# KB / kanon TODO (for Krzysztof, Bartek, or whoever owns the knowledge base)

Findings from building the essay pipeline on `maturaai-bartosz/knowledge_base/kanon.jsonl` (3,098 facts,
Claude-written per curriculum section, unverified). Written 2026-09-27 ~01:00.

## Aspect vocabulary does not match CKE
- `kanon.aspekt` values: polityczny, międzynarodowy, kulturowy, społeczny, ustrojowy, militarny, gospodarczy, religijny.
- Real CKE topics (2023-2026) use: polityczny, **społeczno-gospodarczy** (5 of 12 topics), kulturowy, militarny,
  ustrojowy, **dyplomatyczny**, **polityczno-ustrojowy**, społeczny, gospodarczy.
- Workaround in the generator: merge `społeczny`+`gospodarczy` into `społeczno-gospodarczy` half the time.
  Proper fix: tag facts with CKE's compound aspects too, or add a mapping table (międzynarodowy→dyplomatyczny etc.).

## Coverage is thin in places
- Sections with < 4 facts for an aspect produce 2-3-fact paragraphs (e.g. "Pradzieje i historia starożytnego
  Wschodu", "Europa w XVI–XVII wieku" / barok). The judge marks these "powierzchowna".
- Target: ≥ 6 facts per (section, aspect), each with a date where applicable, so a paragraph can pick 4.

## Aspect tagging is sometimes forced
- "Kultura i nauka polska XIX w.": facts tagged `społeczny`/`gospodarczy` are Skłodowska at the Sorbonne and
  Domeyko in Chile. Two independent judges (Opus, Fable) said the resulting "społeczno-gospodarczy" paragraph
  "says nothing about Polish society or economy". Facts about emigrant scientists are `kulturowy`, not
  `społeczno-gospodarczy`; the section needs real socio-economic facts (industrialisation, urbanisation, class).
- Space race / Sputnik / Apollo tagged `międzynarodowy`; the judge noted it does not support a thesis about
  international cooperation. Some facts need a second aspect tag or a better one.

## Confirmed wrong rows (found by the essay judge, verified against kanon text)
- "Maria Skłodowska-Curie została w 1906 roku pierwszą kobietą prowadzącą wykłady jako profesor paryskiej Sorbony" —
  she began lecturing in 1906; the professorial chair came in 1908. Cost −1 in 4 of 5 essays that used it.
- "Stanisław Kierbedź zaprojektował otwarty w 1864 roku stalowy most na Wiśle" — the bridge was iron, not steel.
- Duplicates: the Skłodowska/polon/rad 1898 fact appears 3 times with slightly different wording; Nobel 1903 and
  1911 appear twice each. Dedupe before retrieval, or a paragraph gets the same fact twice.

## Nothing is verified
- All 3,098 rows are unverified Claude output. 15 generated essays judged so far: 0 factual errors flagged by
  the (Sol) judge, but that is the judge's knowledge checking Claude's knowledge, not a source check.
- Exam-day risk: a wrong kanon fact becomes a confidently wrong essay (−1 to −3 pts). Highest-value rows to
  verify first: dated facts about Polish history (most likely to appear in a real topic).

## `trzy_wybrane` formula
- 3 of 12 real topics ask for "trzech wybranych władców / wydarzeń / państw" instead of aspects. kanon has
  `postac` and `tytul` fields; a `podmiot` (ruler/event/state) grouping would let the harness pick three
  subjects with ≥ 4 facts each. Not yet handled by the generator.
