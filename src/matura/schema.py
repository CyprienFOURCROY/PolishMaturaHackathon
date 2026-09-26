import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    id: str
    max_points: int
    question: str
    source_text: str
    images: list[str]  # basenames, resolved against config.IMAGES_DIR
    answer_format: str
    group: int = 0


@dataclass
class Prepared:
    """Everything the answer stage needs, produced by the upstream stages."""
    item: Item
    ocr: str = ""
    vlm: str = ""
    question_en: str = ""
    source_en: str = ""
    ocr_en: str = ""
    glossary_block: str = ""
    cards: list[dict] = field(default_factory=list)


def load_exam(path: Path) -> tuple[str, list[Item]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [
        Item(
            id=q["id"], group=q.get("group", 0), max_points=q["max_points"], question=q["question"],
            source_text=q.get("source_text", ""), answer_format=q.get("answer_format", ""),
            images=[Path(i["path"]).name for i in q.get("images", [])],
        )
        for q in d["items"]
    ]
    return d["exam_id"], items


def write_answers(path: Path, exam_id: str, answers: dict[str, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    body = {"exam_id": exam_id, "answers": [{"id": k, "answer": v} for k, v in answers.items()]}
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
