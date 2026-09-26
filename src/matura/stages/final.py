from ..models import registry
from ..router import route
from ..schema import Item


def run(item: Item, answer_en: str) -> str:
    """Stage 8: English answer -> Polish. Closed answers (P/F, letters) are language-neutral, so pass through."""
    if not answer_en or route(item).kind != "open":
        return answer_en
    return registry.get("translator_back").translate(answer_en)
