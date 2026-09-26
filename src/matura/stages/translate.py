from ..cache import cached
from ..glossary.annotate import glossary_block, translate_with_glossary
from ..models import registry
from ..schema import Prepared


def run(p: Prepared, refresh: bool = False) -> None:
    """Fills question_en, source_en, glossary_block. Translates question, source text and OCR text separately
    (VLM output is already English and is left alone)."""
    def go():
        tr = registry.get("translator").translate
        q, used_q = translate_with_glossary(p.item.question, tr)
        s, used_s = translate_with_glossary(p.item.source_text, tr)
        o, used_o = translate_with_glossary(p.ocr, tr) if p.ocr else ("", [])
        return {"q": q, "s": s, "o": o, "gloss": glossary_block(used_q + used_s + used_o)}

    r = cached("translate", p.item.id, go, refresh)
    p.question_en, p.source_en, p.ocr_en, p.glossary_block = r["q"], r["s"], r["o"], r["gloss"]
