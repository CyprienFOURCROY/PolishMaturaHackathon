"""Write outputs/full_answers.json from whatever is cached, at any moment.

Always contains every question id in exam order (format of full_answers_template.json); questions
without a final answer yet get "". Also usable from the command line:  python3 export_answers.py
"""
from . import cache, config
from .schema import load_exam, write_answers


def collect_answers() -> tuple[str, dict[str, str]]:
    exam_id, items = load_exam(config.EXAM_PATH)
    return exam_id, {it.id: (cache.get("answer_pl", it.id) or "") for it in items}


def write(path=config.OUT_PATH) -> dict:
    exam_id, answers = collect_answers()
    write_answers(path, exam_id, answers)
    return {"path": str(path), "answered": sum(1 for a in answers.values() if a.strip()), "total": len(answers)}


def main():
    r = write()
    print(f"Wrote {r['path']}: {r['answered']} of {r['total']} questions answered (others are empty strings)")


if __name__ == "__main__":
    main()
