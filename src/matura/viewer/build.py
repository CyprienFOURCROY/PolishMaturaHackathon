"""Collect everything the viewer shows, and build a static outputs/viewer.html from it.

Reads only what already exists (exam JSON, cache/<stage>/), so it works before any model has run:
missing stages are shown as "not run". Glossary hits and routing are computed live (no model needed).

Static (view only):   PYTHONPATH=src python -m matura.viewer.build
Interactive (Run buttons):   PYTHONPATH=src python -m matura.viewer.server
"""
import json
import os
from pathlib import Path
from typing import Callable

from .. import cache, config
from ..eval.judge import CLOSED_MODEL, DEFAULT_MODEL
from ..glossary import match
from ..kb import search
from ..router import route
from ..schema import load_exam

TEMPLATE = Path(__file__).with_name("template.html")
OUT = config.ROOT / "outputs" / "viewer.html"


def _hits(text: str, entries: list[dict]) -> list[list]:
    ids = {e["id"]: i for i, e in enumerate(entries)}
    return [[s, e, ids[en["id"]]] for s, e, en in match.find(text, entries)]


def collect(image_url: Callable[[str], str]) -> dict:
    exam_id, items = load_exam(config.EXAM_PATH)
    entries = match.load(config.GLOSSARY_PATH)
    rows = []
    for it in items:
        r = route(it)
        ocr = "\n".join(t for t in (cache.get("ocr", n) for n in it.images) if t)
        tr = cache.get("translate", it.id)
        answer = cache.get("answer", it.id)
        final = cache.get("answer_pl", it.id)
        rows.append({
            "id": it.id, "points": it.max_points, "kind": r.kind, "format": it.answer_format,
            "question": it.question, "source": it.source_text,
            "images": [image_url(n) for n in it.images],
            "ocr": ocr, "vlm": cache.get("vlm", it.id) or "",
            # "ran" is not the same as "has text": an OCR run on a photo can legitimately return ""
            "ocr_done": bool(it.images) and all(cache.exists("ocr", n) for n in it.images),
            "vlm_done": cache.exists("vlm", it.id),
            "en": tr and {"question": tr["q"], "source": tr["s"], "ocr": tr["o"]},
            "hits": {"question": _hits(it.question, entries), "source": _hits(it.source_text, entries),
                     "ocr": _hits(ocr, entries)},
            "cards": search.search(" ".join([tr["q"], tr["s"], tr["o"]])) if tr else [],
            "answer": answer,
            "final": final,
            "grade": cache.get("judge", it.id),
        })
    return {"exam_id": exam_id, "glossary": entries, "roles": config.ROLES, "judge_model": DEFAULT_MODEL, "judge_model_closed": CLOSED_MODEL, "target": 0.35, "exam_max": 60, "items": rows}


def render(data: dict, live: bool) -> str:
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__LIVE__", "true" if live else "false").replace("__DATA__", blob)


def build() -> Path:
    data = collect(lambda n: os.path.relpath(config.IMAGES_DIR / n, OUT.parent))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(data, live=False), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
