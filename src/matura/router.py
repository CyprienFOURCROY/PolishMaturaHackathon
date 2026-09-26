"""Decide how an item is answered, from answer_format (essay/closed) and the question's own wording."""
import re
from dataclasses import dataclass

from .schema import Item


@dataclass
class Route:
    kind: str                         # essay | closed | open
    slots: list[tuple[str, list[str]]] = None  # closed only: [(label, options)]


def route(item: Item) -> Route:
    fmt = item.answer_format.strip()
    if "wypracowanie" in fmt.lower():
        return Route("essay")
    pairs = re.findall(r"(\w+):\s*(\S+)", fmt)
    if pairs and all(v in ("P", "F") for _, v in pairs):
        return Route("closed", [(k, ["P", "F"]) for k, _ in pairs])
    if pairs and all(re.fullmatch(r"[A-D]", v) for _, v in pairs):
        return Route("closed", [(k, ["A", "B", "C", "D"]) for k, _ in pairs])  # TODO: read real options from question
    if re.fullmatch(r"[A-D]", fmt):
        return Route("closed", [("", ["A", "B", "C", "D"])])
    return Route("open")  # includes matching formats like "A: 1 B: 1" for now
