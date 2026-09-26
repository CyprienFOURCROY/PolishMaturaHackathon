"""python -m matura.run [--only 1,2.1] [--out outputs/full_answers.json]   (run from src/ or with PYTHONPATH=src)"""
import argparse

from . import cache, config, export
from .pipeline import answer_item
from .schema import load_exam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=str(config.EXAM_PATH))
    ap.add_argument("--out", default=str(config.OUT_PATH))
    ap.add_argument("--only", help="comma-separated item ids")
    args = ap.parse_args()

    exam_id, items = load_exam(args.exam)
    if args.only:
        wanted = set(args.only.split(","))
        items = [i for i in items if i.id in wanted]

    for item in items:
        en, pl = answer_item(item)
        cache.put("answer", item.id, en)
        cache.put("answer_pl", item.id, pl)
        print(f"[{item.id}] {pl[:100]!r}")
        r = export.write(args.out)  # after every item: the file is always complete-so-far, never shrinks
    print(f"Saved {r['answered']} of {r['total']} answers to {r['path']}")


if __name__ == "__main__":
    main()
