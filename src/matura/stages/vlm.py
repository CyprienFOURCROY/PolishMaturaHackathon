from PIL import Image

from ..cache import cached
from ..config import IMAGES_DIR, PROMPTS_DIR
from ..models import registry
from ..schema import Item


def run(item: Item, ocr_text: str, refresh: bool = False) -> str:
    """Describe images with the question + source text + OCR in context (cached per item)."""
    if not item.images:
        return ""
    prompt = (PROMPTS_DIR / "vlm.txt").read_text(encoding="utf-8").format(
        question=item.question, source=item.source_text, ocr=ocr_text or "(none)"
    )

    def describe():
        vlm = registry.get("vlm")
        return "\n".join(vlm.describe(Image.open(IMAGES_DIR / n).convert("RGB"), prompt) for n in item.images)

    return cached("vlm", item.id, describe, refresh)
