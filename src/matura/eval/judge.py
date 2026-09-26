"""Grade our final answers with an LLM judge against the official 2023 marking scheme.

    python3 evaluate.py --dry-run             # show what would be sent, no API call
    python3 evaluate.py --only 24 --model gpt-6-astra
    python3 evaluate.py                       # all answered items except the essay

Results are cached in cache/judge/<id>.json; cached items are skipped unless --refresh.
The API key is read from .env (API_KEY) and never printed.
"""
import argparse
import json
import re

from .. import cache, config
from ..router import route
from ..schema import Item, load_exam
from . import closed, scheme

DEFAULT_MODEL = "gpt-6-astra"        # open questions: needs judgement about content
CLOSED_MODEL = "exact"               # closed questions: exact match with the official key, no API call (free, 100% accurate).
                                     # Set to an OpenAI model name instead to use an LLM (cheap ones miscount partial credit).


def model_for(item: Item) -> str:
    return CLOSED_MODEL if route(item).kind == "closed" else DEFAULT_MODEL


def _extra(model: str) -> dict:
    return {"reasoning_effort": "minimal"} if model.startswith("gpt-5") else {}


def api_key() -> str:
    m = re.search(r"^API_KEY\s*=\s*(.+)$", (config.ROOT / ".env").read_text(), re.M)
    if not m or not m.group(1).strip().strip("\"'"):
        raise RuntimeError("No API_KEY in .env")
    return m.group(1).strip().strip("\"'")


def final_answer(item_id: str) -> str | None:
    return cache.get("answer_pl", item_id)


def build_prompt(item: Item, answer: str) -> str:
    r = scheme.load()[item.id]
    return (config.PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8").format(
        id=item.id, max=r["max"], question=item.question, source=item.source_text[:3000], rules=r["rules"], answer=answer)


def grade(item: Item, answer: str, model: str | None = None, client=None, save: bool = True) -> dict:
    """Grade one answer, cache and return {points, max, reason, model, tokens}. Empty answers cost no API call."""
    model = model or model_for(item)
    mx = scheme.load()[item.id]["max"]
    if not answer.strip():
        res = {"points": 0, "max": mx, "reason": "empty answer", "model": None, "tokens": {"in": 0, "out": 0, "reasoning": 0}}
        if save:
            cache.put("judge", item.id, res)
        return res
    if model == "exact":
        pts, mx2, reason = closed.grade(item.id, answer)
        res = {"points": pts, "max": mx, "reason": reason, "model": "exact match (no API)",
               "tokens": {"in": 0, "out": 0, "reasoning": 0}}
        if save:
            cache.put("judge", item.id, res)
        return res
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=api_key())
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": build_prompt(item, answer)}],
        response_format={"type": "json_object"}, max_completion_tokens=2000, **_extra(model))
    u = resp.usage
    try:
        g = json.loads(resp.choices[0].message.content)
        pts = max(0, min(int(g["points"]), mx))
    except Exception as e:
        raise ValueError(f"Unparseable judge reply: {resp.choices[0].message.content!r}") from e
    res = {"points": pts, "max": mx, "reason": g.get("reason", ""), "model": model,
           "tokens": {"in": u.prompt_tokens, "out": u.completion_tokens,
                      "reasoning": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0}}
    if save:
        cache.put("judge", item.id, res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override the judge for ALL questions (default: %s for open, %s for closed)" % (DEFAULT_MODEL, CLOSED_MODEL))
    ap.add_argument("--only", help="comma-separated item ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-grade items that already have a cached grade")
    ap.add_argument("--max-calls", type=int, default=50, help="hard cap on API calls")
    ap.add_argument("--include-essay", action="store_true")
    args = ap.parse_args()

    _, items = load_exam(config.EXAM_PATH)
    if args.only:
        items = [i for i in items if i.id in set(args.only.split(","))]

    client, calls, used = None, 0, {"in": 0, "out": 0, "reasoning": 0}
    for it in items:
        if it.id == "26" and not args.include_essay:
            continue
        ans = final_answer(it.id)
        if ans is None:
            print(f"[{it.id}] no answer yet, skipped")
            continue
        if cache.exists("judge", it.id) and not args.refresh:
            print(f"[{it.id}] already graded (cached), skipped. Use --refresh to re-grade")
            continue
        if args.dry_run:
            print(f"[{it.id}] would send ~{len(build_prompt(it, ans)) // 3} tokens")
            continue
        if calls >= args.max_calls:
            print(f"Reached --max-calls={args.max_calls}, stopping.")
            break
        if client is None and ans.strip() and (args.model or model_for(it)) != "exact":
            from openai import OpenAI
            client = OpenAI(api_key=api_key())
        try:
            g = grade(it, ans, args.model, client)
        except ValueError as e:
            print(f"[{it.id}] {e}")
            continue
        calls += g["tokens"]["in"] > 0
        for k in used:
            used[k] += g["tokens"][k]
        print(f"[{it.id}] {g['points']}/{g['max']}  ({g['model']})  {g['reason']}")

    graded = [cache.get("judge", i.id) for i in items if i.id != "26" and cache.exists("judge", i.id)]
    if graded:
        got, mx = sum(g["points"] for g in graded), sum(g["max"] for g in graded)
        print(f"\nGraded {len(graded)} items: {got}/{mx} points ({100 * got / mx:.0f}%)  [exam total 60, essay 15]")
    if calls:
        print(f"API calls: {calls}, tokens in/out: {used['in']}/{used['out']} (of which reasoning: {used['reasoning']})")


if __name__ == "__main__":
    main()
