from .schema import Item, Prepared
from .stages import answer, final, ocr, retrieve, translate, vlm


def prepare(item: Item) -> Prepared:
    p = Prepared(item=item)
    p.ocr = ocr.run(item)
    p.vlm = vlm.run(item, p.ocr)
    translate.run(p)
    retrieve.run(p)
    return p


def answer_item(item: Item) -> tuple[str, str]:
    """Returns (english_answer, final_polish_answer)."""
    en = answer.run(prepare(item))
    return en, final.run(item, en)
