from ..cache import cached
from ..config import IMAGES_DIR
from ..models import registry
from ..schema import Item


def run(item: Item, refresh: bool = False) -> str:
    """OCR every image of the item (cached per image), joined in order."""
    texts = []
    for name in item.images:
        texts.append(cached("ocr", name, lambda n=name: registry.get("ocr").read(str(IMAGES_DIR / n)), refresh))
    return "\n".join(t for t in texts if t)
