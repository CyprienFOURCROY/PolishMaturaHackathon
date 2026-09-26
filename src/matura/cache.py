"""Per-stage JSON cache so any stage can be rerun alone: cache/<stage>/<key>.json"""
import json
from typing import Callable

from .config import CACHE_DIR


def _path(stage: str, key: str):
    return CACHE_DIR / stage / f"{key}.json"


def exists(stage: str, key: str) -> bool:
    return _path(stage, key).exists()


def get(stage: str, key: str):
    p = _path(stage, key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def put(stage: str, key: str, value) -> None:
    p = _path(stage, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def delete(stage: str, key: str) -> bool:
    p = _path(stage, key)
    if p.exists():
        p.unlink()
        return True
    return False


def cached(stage: str, key: str, fn: Callable[[], object], refresh: bool = False):
    if exists(stage, key) and not refresh:
        return get(stage, key)
    value = fn()
    put(stage, key, value)
    return value
