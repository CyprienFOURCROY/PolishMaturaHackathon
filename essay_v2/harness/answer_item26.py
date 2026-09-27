"""Exam-day entry point: fill the essay item of an exam package.

    python3 essay_v2/harness/answer_item26.py --exam path/to/exam.json --out answers.json \
        [--merge existing_answers.json] [--model bielik-essay-sft] [--n 6]

- Finds the essay item (max_points == 15, or question mentions 'tematy' + 'wypowiedź'), builds the essay with
  essay.py (topic choice, retrieval, gated best-of-n, fallback), and writes '<topic number>.\n\n<essay>' as its answer.
- With --merge, all other answers are copied from that file (e.g. Bartek's full-sheet output) and only the essay
  item is replaced; without it, every other item is ''.
- Prints the chosen topic, word count, tiers, and the full essay so a human can read it before uploading.
Offline: needs only local Ollama and out/kanon_clean.jsonl (or KANON_PATH)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import essay as E

ap = argparse.ArgumentParser()
ap.add_argument("--exam", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--merge", default=None); ap.add_argument("--model", default="bielik-essay-sft")
ap.add_argument("--n", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

exam = json.load(open(a.exam, encoding="utf-8"))
items = exam["items"]
essay_items = [it for it in items if it.get("max_points") == 15 or
               ("temat" in it.get("question", "").lower() and "wypowied" in it.get("question", "").lower())]
if not essay_items:
    sys.exit("no essay item found in exam.json")
it = essay_items[0]
print(f"essay item id={it['id']} max_points={it.get('max_points')}")

kb = E.KB()
t0 = time.time()
rec = E.write_essay(it["question"], kb, a.model, n=a.n, seed=a.seed)
answer = f"{rec['topic_num']}.\n\n{rec['full_text']}"
print(f"topic {rec['topic_num']} of {rec['chosen_from']} | {rec['words']} words | tiers {rec['tiers']} | {time.time()-t0:.0f}s\n")
print(rec["full_text"])

if a.merge:
    out = json.load(open(a.merge, encoding="utf-8"))
else:
    out = {"exam_id": exam["exam_id"], "answers": [{"id": x["id"], "answer": ""} for x in items]}
ids = {x["id"] for x in out["answers"]}
assert ids == {x["id"] for x in items}, "answer ids do not match exam items"
for x in out["answers"]:
    if x["id"] == it["id"]:
        x["answer"] = answer
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(rec, open(a.out + ".essay_record.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n-> {a.out}  (record: {a.out}.essay_record.json)")
