from ..config import PROMPTS_DIR
from ..models import registry
from ..router import route
from ..schema import Prepared


def _context(p: Prepared) -> str:
    parts = [f"Source: {p.source_en}"]
    if p.ocr_en:
        parts.append(f"Text read from the image: {p.ocr_en}")
    if p.vlm:
        parts.append(f"Image description (may be inaccurate): {p.vlm}")
    if p.glossary_block:
        parts.append(p.glossary_block)
    if p.cards:
        parts.append("Background:\n" + "\n".join(f"- {c['title']}: {c['text']}" for c in p.cards))
    return "\n\n".join(parts)


def _tpl(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def run(p: Prepared) -> str:
    r, llm, ctx = route(p.item), registry.get("llm"), _context(p)
    if r.kind == "essay":
        return ""  # essay handled later (stages/essay.py)
    if r.kind == "closed":
        answers = []
        for label, options in r.slots:
            prompt = _tpl("closed.txt").format(
                options=", ".join(options), context=ctx, question=p.question_en,
                slot_line=f"Statement/part: {label}" if label else "",
            )
            scores = llm.score_options(prompt, options)
            answers.append(f"{label}: {max(scores, key=scores.get)}" if label else max(scores, key=scores.get))
        return " ".join(answers)
    return llm.generate(_tpl("open.txt").format(context=ctx, question=p.question_en))
