# KNOWN TRAPS

Every item here cost real time on the first night. Read before writing any
generation code. All of these are things I got wrong, not things to discover.

> These are the *mechanical* traps — call paths, whitespace, casing. The
> *judgement* traps (proxying a green check for a real result, contaminating your
> own experiment, grading shape instead of content) are in the field guide at the
> top of `AGENTS.md`. Read that one first; it matters more than anything below.

## Ollama / transport

- **`ollama run` has no `--system` flag in this build.** It fails silently and
  returns empty. Use `POST /api/chat` with a real `system` role.
- **Never put `"stop"` in ollama `options`.** With a stop list this build drops
  the chat template, the model falls back to raw completion, and the output is
  `<|start_header_id|>...` instead of prose. This cost 36 wasted generations.
- **A system message is mandatory.** Without one the model emits raw tokenizer
  control tokens instead of answering.
- **The model reliably returns a valid response; the *content* is often junk.**
  A 10-call probe gave 10/10 HTTP-valid responses and ~3/10 unusable ones:
  verbatim prompt echo, a bare `*✅*`, a 7-word non-answer, a hallucinated year.
  Do not conclude "the call path is flaky" from short output — read the raw text.

## Text handling

- **Collapse whitespace before counting words.** The model emits newline spam
  like `W \n\n\n\n W 1386 roku \n\n\n` which counts as ~4 words.
- **Trim to the last complete sentence.** It trails off mid-sentence at the
  token cap.
- **Cut where the model starts echoing your prompt scaffolding** ("Teraz
  napisz…", "Fakty:"). It does this often after few-shot examples.
- **Never swallow an exception in the call path and return `""`.** I did that;
  a harness bug then showed up as a *model* failure labelled PUSTA, and I
  nearly drew a conclusion from it. Return the error.
- **Keep Polish diacritics in prompts and in example essays.** I stripped
  `ą ę ł ó ś ź ż ć ń` while rewriting and would have taught the model to answer
  without them, which is unacceptable in a Matura essay. Polish belongs in the
  prompts and examples; the harness code and comments are English.

## Model behaviour, measured

- Bielik returns `content: ""` with no error when asked to *analyse* ("explain
  what this fact means for the thesis"). 0 chars in 4/4 checked calls. It
  answers "write a paragraph"-shaped tasks. Abstract analytical instructions get
  an immediate EOS.
- In completion mode it emits **citation markers** (`[3][10][11]`, `[40][41]`)
  and meta-instructions (`Pamietaj o uzyciu odpowiedniej formy...`). Both must
  be stripped or the sample rejected.
- Even a clean continuation drifts: one sample left the 19th-century industrial
  topic entirely for a Komorowski-era administrative reform.

## Method

- **Do not name a module `try.py`.** `try` is a keyword, so the file cannot be
  imported at all, which makes it impossible to reuse its call function from a
  probe. It is `essay_harness/probe.py`.

- **One call, one job, ~60-150 words.** A 1.5B has no budget for a rubric.
- **n≥3, always.** I rebuilt the architecture off single samples four times.
- **A gate pass is not a quality pass.** One sample cleared every gate and was
  still nonsense — it passed by echoing the fragment back, then claimed cotton
  was processed into silk. Always read the raw output.
- **Test the call path with 1 cheap call before spending 30.** I spent 36
  generations on a broken path before adding the check.
- **A/B a prompt change before rebuilding the pipeline around it.** Changing the
  prompt can invalidate last week's conclusion.
