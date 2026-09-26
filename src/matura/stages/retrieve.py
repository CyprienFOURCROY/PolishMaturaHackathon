from ..kb import search
from ..schema import Prepared


def run(p: Prepared) -> None:
    # TODO: multi-call variant: first ask the LLM for keywords, then search on those
    p.cards = search.search(" ".join([p.question_en, p.source_en, p.ocr_en]))
